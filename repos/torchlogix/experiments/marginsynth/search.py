#!/usr/bin/env python3
"""Deterministic greedy MarginSynth search with exact fail-closed checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.incremental import evaluate_rewrite
from experiments.marginsynth.rewrites import GateRewrite, generate_gate_rewrites
from experiments.marginsynth.trace import PackedCalibrationTrace, build_trace
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    write_artifact_manifest,
)
from torchlogix import Circuit


def circuit_sha256(circuit: Circuit) -> str:
    payload = json.dumps(
        circuit.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def circuit_cost(circuit: Circuit, config: dict) -> dict:
    gate_by_id = {gate.gate_id: gate for gate in circuit.gates}
    depths = {}
    connections = 0
    fanouts = Counter()
    for gate in circuit.gates:
        input_depths = []
        for node_id in (gate.in0, gate.in1):
            if node_id < 0:
                continue
            connections += 1
            fanouts[node_id] += 1
            input_depths.append(depths.get(node_id, 0))
        depths[gate.gate_id] = 1 + max(input_depths, default=-1)
    for reduction in circuit.sum_nodes:
        for node_id in reduction.input_ids:
            connections += 1
            fanouts[node_id] += 1
    live_depths = [
        depths.get(node_id, 0)
        for output_id in circuit.outputs
        for node_id in (
            circuit._sum_by_id[output_id].input_ids
            if output_id in circuit._sum_by_id
            else [output_id]
        )
    ]
    gates = len(circuit.gates)
    depth = max(live_depths, default=0)
    proxy = (
        gates
        + config["connection_weight"] * connections
        + config["depth_weight"] * depth
    )
    return {
        "live_gates": gates,
        "connections": connections,
        "logic_depth": depth,
        "maximum_fanout": max(fanouts.values(), default=0),
        "proxy": float(proxy),
    }


def deterministic_pool(
    candidates: list[GateRewrite],
    limit: int,
    seed: int,
    iteration: int,
) -> list[GateRewrite]:
    if limit <= 0 or limit >= len(candidates):
        return candidates

    def key(rewrite):
        serialized = json.dumps(
            rewrite.to_dict(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(
            f"{seed}:{iteration}:{serialized}".encode()
        ).digest()
        return digest, serialized

    return sorted(candidates, key=key)[:limit]


def reference_metrics(
    scores: np.ndarray,
    original_trace: PackedCalibrationTrace,
    mode: str,
    margin_reserve: float,
) -> dict:
    predictions = scores.argmax(axis=1).astype(np.int64)
    flips = predictions != original_trace.predictions
    original_winners = original_trace.predictions
    challenger = scores.copy()
    challenger[np.arange(len(scores)), original_winners] = -np.inf
    remaining_margins = (
        scores[np.arange(len(scores)), original_winners]
        - challenger.max(axis=1)
    )
    margin_loss = float(
        np.maximum(0.0, margin_reserve - remaining_margins).mean()
    )

    if mode == "decision-preserving":
        loss = float(flips.mean())
        group_ids = original_trace.predictions
        group_loss_events = flips
    elif mode == "accuracy-budgeted":
        current_correct = predictions == original_trace.labels
        loss = max(
            0.0,
            float(original_trace.correct.mean() - current_correct.mean()),
        )
        group_ids = original_trace.labels
        group_loss_events = original_trace.correct.astype(np.int8) - (
            current_correct.astype(np.int8)
        )
    else:
        raise ValueError(f"unsupported search mode: {mode}")

    group_counts = np.bincount(
        group_ids,
        minlength=original_trace.num_outputs,
    )
    if mode == "decision-preserving":
        group_losses = np.bincount(
            group_ids,
            weights=group_loss_events,
            minlength=original_trace.num_outputs,
        )
    else:
        group_losses = np.bincount(
            group_ids,
            weights=group_loss_events,
            minlength=original_trace.num_outputs,
        )
        group_losses = np.maximum(group_losses, 0)
    per_class_loss = np.divide(
        group_losses,
        group_counts,
        out=np.zeros(original_trace.num_outputs, dtype=np.float64),
        where=group_counts != 0,
    )
    return {
        "predictions": predictions,
        "global_loss": loss,
        "per_class_loss": per_class_loss,
        "maximum_per_class_loss": float(per_class_loss.max()),
        "decision_flip_count": int(flips.sum()),
        "decision_flip_rate": float(flips.mean()),
        "accuracy": float((predictions == original_trace.labels).mean()),
        "margin_loss_mean": margin_loss,
        "remaining_margin_minimum": float(remaining_margins.min()),
        "remaining_margin_mean": float(remaining_margins.mean()),
    }


def within_budget(metrics: dict, config: dict) -> bool:
    tolerance = 1e-12
    return (
        metrics["global_loss"] <= config["maximum_global_loss"] + tolerance
        and metrics["maximum_per_class_loss"]
        <= config["maximum_per_class_loss"] + tolerance
    )


def snapshot(
    circuit: Circuit,
    directory: Path,
    metadata: dict,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    circuit_path = directory / "circuit.json"
    c_path = directory / "circuit.c"
    hardware_path = directory / "hardware_circuit.json"
    verilog_path = directory / "circuit.v"
    circuit.write_json(str(circuit_path))
    circuit.write_c_code(str(c_path))
    hardware = circuit.normalized_for_hardware_argmax()
    hardware.write_json(str(hardware_path))
    hardware.write_verilog_code(str(verilog_path))
    record = dict(metadata)
    record["files"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (circuit_path, c_path, hardware_path, verilog_path)
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def run_search(
    circuit: Circuit,
    original_trace: PackedCalibrationTrace,
    config: dict,
    output_directory: Path,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    snapshots_directory = output_directory / "snapshots"
    encoded_inputs = np.stack(
        [
            original_trace.unpack_node(input_id)
            for input_id in range(original_trace.n_inputs)
        ],
        axis=1,
    )
    current_circuit = copy.deepcopy(circuit)
    current_trace = build_trace(
        current_circuit,
        encoded_inputs,
        original_trace.labels,
    )

    start_time = time.perf_counter()
    current_cost = circuit_cost(current_circuit, config)
    initial_cost = dict(current_cost)
    initial_metrics = reference_metrics(
        current_trace.scores,
        original_trace,
        config["mode"],
        config["margin_reserve"],
    )
    starting_scores_exact = bool(
        np.array_equal(current_trace.scores, original_trace.scores)
    )
    starting_predictions_exact = bool(
        np.array_equal(
            current_trace.predictions,
            original_trace.predictions,
        )
    )
    if not within_budget(initial_metrics, config):
        raise ValueError(
            "the configured starting circuit already exceeds the search budgets"
        )
    points = [
        {
            "step": 0,
            "circuit_sha256": circuit_sha256(current_circuit),
            "cost": current_cost,
            "behavior": {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in initial_metrics.items()
                if key != "predictions"
            },
            "snapshot": "snapshots/step_0000",
        }
    ]
    snapshot(
        current_circuit,
        snapshots_directory / "step_0000",
        points[0],
    )
    rewrite_log = []
    stop_reason = "maximum accepted rewrites reached"

    for iteration in range(1, config["max_accepted_rewrites"] + 1):
        current_reference_metrics = reference_metrics(
            current_trace.scores,
            original_trace,
            config["mode"],
            config["margin_reserve"],
        )
        all_candidates = generate_gate_rewrites(current_circuit)
        pool = deterministic_pool(
            all_candidates,
            config["candidate_limit_per_iteration"],
            config["search_seed"],
            iteration,
        )
        feasible = []
        evaluated_count = 0
        for candidate in pool:
            evaluation = evaluate_rewrite(
                current_circuit,
                current_trace,
                candidate,
                margin_reserve=config["margin_reserve"],
            )
            evaluated_count += 1
            metrics = reference_metrics(
                evaluation.scores,
                original_trace,
                config["mode"],
                config["margin_reserve"],
            )
            if within_budget(metrics, config):
                feasible.append((candidate, evaluation, metrics))

        def estimated_priority(item):
            candidate, evaluation, metrics = item
            incremental_margin_damage = max(
                0.0,
                metrics["margin_loss_mean"]
                - current_reference_metrics["margin_loss_mean"],
            )
            incremental_flip_damage = max(
                0.0,
                metrics["decision_flip_rate"]
                - current_reference_metrics["decision_flip_rate"],
            )
            behavior_penalty = (
                config["cost_epsilon"]
                + config.get("margin_penalty", 1.0)
                * incremental_margin_damage
                + config["flip_penalty"] * incremental_flip_damage
            )
            cleanup_upper_bound = 1 + len(evaluation.affected_gate_ids)
            return cleanup_upper_bound / behavior_penalty

        feasible.sort(
            key=lambda item: (
                -estimated_priority(item),
                item[2]["decision_flip_count"],
                item[2]["margin_loss_mean"],
                json.dumps(item[0].to_dict(), sort_keys=True),
            )
        )
        shortlist = feasible[: config["exact_cost_shortlist"]]
        best = None
        for candidate, evaluation, metrics in shortlist:
            rewritten = candidate.apply_to_copy(current_circuit)
            gates_before_cleanup = len(rewritten.gates)
            rewritten.simplify()
            candidate_cost = circuit_cost(rewritten, config)
            cost_reduction = current_cost["proxy"] - candidate_cost["proxy"]
            if cost_reduction <= 0.0:
                continue
            incremental_margin_damage = max(
                0.0,
                metrics["margin_loss_mean"]
                - current_reference_metrics["margin_loss_mean"],
            )
            incremental_flip_damage = max(
                0.0,
                metrics["decision_flip_rate"]
                - current_reference_metrics["decision_flip_rate"],
            )
            quality = cost_reduction / (
                config["cost_epsilon"]
                + config.get("margin_penalty", 1.0)
                * incremental_margin_damage
                + config["flip_penalty"] * incremental_flip_damage
            )
            selection_key = (
                quality,
                cost_reduction,
                -metrics["decision_flip_count"],
                -metrics["margin_loss_mean"],
            )
            if best is None or selection_key > best["selection_key"]:
                best = {
                    "candidate": candidate,
                    "evaluation": evaluation,
                    "metrics": metrics,
                    "circuit": rewritten,
                    "cost": candidate_cost,
                    "cost_reduction": cost_reduction,
                    "quality": quality,
                    "incremental_margin_damage": incremental_margin_damage,
                    "incremental_flip_damage": incremental_flip_damage,
                    "selection_key": selection_key,
                    "gates_before_cleanup": gates_before_cleanup,
                }

        if best is None:
            stop_reason = "no feasible shortlisted candidate reduced exact cost"
            break

        full_trace = build_trace(
            best["circuit"],
            encoded_inputs,
            original_trace.labels,
        )
        if not np.array_equal(
            best["evaluation"].scores,
            full_trace.scores,
        ) or not np.array_equal(
            best["evaluation"].predictions,
            full_trace.predictions,
        ):
            diagnostic = {
                "iteration": iteration,
                "rewrite": best["candidate"].to_dict(),
                "incremental_full_max_score_difference": float(
                    np.max(
                        np.abs(
                            best["evaluation"].scores - full_trace.scores
                        )
                    )
                ),
            }
            (output_directory / "FAIL_CLOSED.json").write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeError("accepted candidate failed full resimulation")

        measured_metrics = reference_metrics(
            full_trace.scores,
            original_trace,
            config["mode"],
            config["margin_reserve"],
        )
        if not within_budget(measured_metrics, config):
            raise RuntimeError("accepted candidate exceeded a hard budget")

        before_sha = circuit_sha256(current_circuit)
        after_sha = circuit_sha256(best["circuit"])
        incremental_summary = best["evaluation"].summary()
        # Wall-clock measurements belong in aggregate runtime records, not the
        # replay log: excluding them makes accepted rewrite logs byte-stable.
        incremental_summary.pop("evaluation_seconds", None)
        log_entry = {
            "step": iteration,
            "rewrite": best["candidate"].to_dict(),
            "circuit_before_sha256": before_sha,
            "circuit_after_sha256": after_sha,
            "candidate_pool_total": len(all_candidates),
            "candidate_pool_evaluated": evaluated_count,
            "feasible_candidates": len(feasible),
            "exact_cost_shortlist": len(shortlist),
            "affected_gate_ids": best["evaluation"].affected_gate_ids,
            "affected_output_indices": (
                best["evaluation"].affected_output_indices
            ),
            "incremental": incremental_summary,
            "measured_behavior": {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in measured_metrics.items()
                if key != "predictions"
            },
            "cost_before": current_cost,
            "cost_after": best["cost"],
            "proxy_cost_reduction": best["cost_reduction"],
            "selection_quality": best["quality"],
            "selection_incremental_margin_damage": best[
                "incremental_margin_damage"
            ],
            "selection_incremental_flip_damage": best[
                "incremental_flip_damage"
            ],
            "gates_before_exact_cleanup": best["gates_before_cleanup"],
            "gates_after_exact_cleanup": len(best["circuit"].gates),
            "incremental_full_scores_exact": True,
            "incremental_full_predictions_exact": True,
        }
        rewrite_log.append(log_entry)
        current_circuit = best["circuit"]
        current_trace = full_trace
        current_cost = best["cost"]
        point = {
            "step": iteration,
            "circuit_sha256": after_sha,
            "cost": current_cost,
            "behavior": log_entry["measured_behavior"],
            "snapshot": f"snapshots/step_{iteration:04d}",
        }
        points.append(point)
        snapshot(
            current_circuit,
            snapshots_directory / f"step_{iteration:04d}",
            point,
        )
        (output_directory / "rewrite_log.json").write_text(
            json.dumps(rewrite_log, indent=2, sort_keys=True) + "\n"
        )

    pareto = []
    for budget in config["pareto_budgets"]:
        eligible = [
            point
            for point in points
            if point["behavior"]["global_loss"] <= budget + 1e-12
            and point["behavior"]["maximum_per_class_loss"]
            <= config["maximum_per_class_loss"] + 1e-12
        ]
        if not eligible:
            continue
        selected = min(
            eligible,
            key=lambda point: (
                point["cost"]["proxy"],
                point["behavior"]["global_loss"],
                point["step"],
            ),
        )
        pareto.append(
            {
                "budget": budget,
                "selected_step": selected["step"],
                "snapshot": selected["snapshot"],
                "global_loss": selected["behavior"]["global_loss"],
                "maximum_per_class_loss": selected["behavior"][
                    "maximum_per_class_loss"
                ],
                "live_gates": selected["cost"]["live_gates"],
                "proxy_cost": selected["cost"]["proxy"],
            }
        )

    summary = {
        "format_version": 1,
        "status": "completed",
        "development_run": True,
        "stop_reason": stop_reason,
        "accepted_rewrites": len(rewrite_log),
        "accepted_by_kind": dict(
            sorted(
                Counter(
                    entry["rewrite"]["kind"] for entry in rewrite_log
                ).items()
            )
        ),
        "initial_cost": initial_cost,
        "starting_scores_exact_vs_reference": starting_scores_exact,
        "starting_predictions_exact_vs_reference": starting_predictions_exact,
        "final_cost": current_cost,
        "live_gate_reduction": (
            initial_cost["live_gates"] - current_cost["live_gates"]
        ),
        "live_gate_reduction_rate": (
            (initial_cost["live_gates"] - current_cost["live_gates"])
            / initial_cost["live_gates"]
        ),
        "final_behavior": points[-1]["behavior"],
        "wall_seconds": time.perf_counter() - start_time,
        "peak_process_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "pareto": pareto,
        "points": points,
    }
    (output_directory / "search_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not (output_directory / "rewrite_log.json").exists():
        (output_directory / "rewrite_log.json").write_text("[]\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    config = json.loads(cli.config.read_text())
    output_directory = run_dir / config["output"]
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "search_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "source_revision": git_revision(),
        "search_script_sha256": sha256_file(Path(__file__).resolve()),
        "config_source": str(cli.config),
        "config_source_sha256": sha256_file(cli.config),
        "input_circuit_sha256": sha256_file(
            run_dir / config["input_circuit"]
        ),
        "input_trace_metadata_sha256": sha256_file(
            run_dir / config["input_trace"] / "metadata.json"
        ),
        "validation_used": False,
        "test_used": False,
    }
    (output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    circuit = Circuit.from_json_file(str(run_dir / config["input_circuit"]))
    trace = PackedCalibrationTrace.load(
        run_dir / config["input_trace"],
        mmap_mode="r",
    )
    summary = run_search(circuit, trace, config, output_directory)
    write_artifact_manifest(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
