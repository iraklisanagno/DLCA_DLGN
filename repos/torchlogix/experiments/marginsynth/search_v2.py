#!/usr/bin/env python3
"""MarginSynth v2: dual-budget, synthesis-aware coordinated circuit search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.cost_model import (
    DEFAULT_OPERATION_COST,
    SynthCostEstimator,
    circuit_features,
)
from experiments.marginsynth.incremental import _validate_trace, evaluate_proposal
from experiments.marginsynth.rewrites import (
    GateRewrite,
    RewriteGroup,
    RewriteKind,
    RewriteProposal,
    generate_coordinated_cone_rewrites,
    generate_gate_rewrites,
    generate_targeted_rewrites,
    proposal_from_dict,
    proposal_target_ids,
)
from experiments.marginsynth.search import circuit_sha256
from experiments.marginsynth.trace import PackedCalibrationTrace, build_trace
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    write_artifact_manifest,
)
from torchlogix import Circuit


def proposal_key(proposal: RewriteProposal) -> str:
    return json.dumps(proposal.to_dict(), sort_keys=True, separators=(",", ":"))


def proposal_category(proposal: RewriteProposal) -> str:
    if isinstance(proposal, RewriteGroup):
        return "coordinated"
    if proposal.kind in (RewriteKind.CONSTANT_0, RewriteKind.CONSTANT_1):
        return "constant"
    if proposal.kind == RewriteKind.ALTERNATIVE_GATE:
        return "alternative"
    return "routing"


def stratified_ranked_pool(
    items: list,
    limit: int,
    proposal_of,
    key,
) -> list:
    """Round-robin ranked categories so one rewrite family cannot starve others."""
    categories = ("coordinated", "constant", "routing", "alternative")
    buckets = {
        category: sorted(
            (
                item
                for item in items
                if proposal_category(proposal_of(item)) == category
            ),
            key=key,
        )
        for category in categories
    }
    target = len(items) if limit <= 0 else min(limit, len(items))
    selected = []
    index = 0
    while len(selected) < target:
        added = False
        for category in categories:
            bucket = buckets[category]
            if index < len(bucket):
                selected.append(bucket[index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        index += 1
    return selected


def category_counts(items: list, proposal_of=lambda item: item) -> dict[str, int]:
    counts = {
        "coordinated": 0,
        "constant": 0,
        "routing": 0,
        "alternative": 0,
    }
    for item in items:
        counts[proposal_category(proposal_of(item))] += 1
    return counts


def gate_signatures(circuit: Circuit) -> dict[int, tuple]:
    return {
        gate.gate_id: (gate.op.name, gate.in0, gate.in1)
        for gate in circuit.gates
    }


def behavior_metrics(
    scores: np.ndarray,
    reference: PackedCalibrationTrace,
    margin_reserve: float,
) -> dict:
    predictions = scores.argmax(axis=1).astype(np.int64)
    flips = predictions != reference.predictions
    correct = predictions == reference.labels
    baseline_accuracy = float(reference.correct.mean())
    accuracy = float(correct.mean())
    accuracy_loss = max(0.0, baseline_accuracy - accuracy)

    class_counts = np.bincount(
        reference.labels,
        minlength=reference.num_outputs,
    )
    baseline_class_correct = np.bincount(
        reference.labels,
        weights=reference.correct,
        minlength=reference.num_outputs,
    )
    class_correct = np.bincount(
        reference.labels,
        weights=correct,
        minlength=reference.num_outputs,
    )
    baseline_class_accuracy = np.divide(
        baseline_class_correct,
        class_counts,
        out=np.zeros(reference.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    per_class_accuracy = np.divide(
        class_correct,
        class_counts,
        out=np.zeros(reference.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    per_class_accuracy_loss = np.maximum(
        0.0,
        baseline_class_accuracy - per_class_accuracy,
    )

    original_class_counts = np.bincount(
        reference.predictions,
        minlength=reference.num_outputs,
    )
    disagreement_counts = np.bincount(
        reference.predictions[flips],
        minlength=reference.num_outputs,
    )
    per_class_disagreement = np.divide(
        disagreement_counts,
        original_class_counts,
        out=np.zeros(reference.num_outputs, dtype=np.float64),
        where=original_class_counts != 0,
    )
    challenger = scores.copy()
    challenger[np.arange(len(scores)), reference.predictions] = -np.inf
    remaining_margins = (
        scores[np.arange(len(scores)), reference.predictions]
        - challenger.max(axis=1)
    )
    hinge = np.maximum(0.0, margin_reserve - remaining_margins)
    low_margin = remaining_margins <= margin_reserve
    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "accuracy_loss": accuracy_loss,
        "per_class_accuracy": per_class_accuracy,
        "per_class_accuracy_loss": per_class_accuracy_loss,
        "maximum_per_class_accuracy_loss": float(
            per_class_accuracy_loss.max(initial=0.0)
        ),
        "decision_flip_count": int(flips.sum()),
        "decision_flip_rate": float(flips.mean()),
        "per_class_disagreement": per_class_disagreement,
        "maximum_per_class_disagreement": float(
            per_class_disagreement.max(initial=0.0)
        ),
        "margin_loss_mean": float(hinge.mean()),
        "low_margin_rate": float(low_margin.mean()),
        "remaining_margin_minimum": float(remaining_margins.min()),
        "remaining_margin_mean": float(remaining_margins.mean()),
        "remaining_margin_p01": float(np.quantile(remaining_margins, 0.01)),
        "remaining_margin_p05": float(np.quantile(remaining_margins, 0.05)),
    }


def within_budgets(metrics: dict, config: dict) -> bool:
    tolerance = 1e-12
    return (
        metrics["accuracy_loss"]
        <= config["maximum_accuracy_loss"] + tolerance
        and metrics["maximum_per_class_accuracy_loss"]
        <= config["maximum_per_class_accuracy_loss"] + tolerance
        and metrics["decision_flip_rate"]
        <= config["maximum_disagreement"] + tolerance
        and metrics["maximum_per_class_disagreement"]
        <= config["maximum_per_class_disagreement"] + tolerance
    )


def serializable_metrics(metrics: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
        if key != "predictions"
    }


def cost_record(
    circuit: Circuit,
    estimator: SynthCostEstimator,
    cost_mode: str = "synthesis-aware",
) -> dict:
    features = circuit_features(circuit)
    if cost_mode == "synthesis-aware":
        estimated = estimator.estimate_from_features(features)
    elif cost_mode == "gate-count":
        estimated = float(features["live_gates"])
    else:
        raise ValueError(f"unsupported cost_mode: {cost_mode}")
    return {
        "live_gates": int(features["live_gates"]),
        "sum_inputs": int(features["sum_inputs"]),
        "connections": int(features["connections"]),
        "logic_depth": int(features["logic_depth"]),
        "operation_aig_units": float(features["operation_aig_units"]),
        "estimated_abc_nodes": estimated,
        "cost_mode": cost_mode,
        "gate_histogram": features["gate_histogram"],
    }


def structural_priority(
    proposal: RewriteProposal,
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    gate_by_id: dict | None = None,
    target_structure: dict | None = None,
    row_by_id: dict[int, int] | None = None,
) -> tuple:
    if gate_by_id is None:
        gate_by_id = {gate.gate_id: gate for gate in circuit.gates}
    if target_structure is None:
        target_structure = {}
    if row_by_id is None:
        row_by_id = trace.row_by_id
    combined_gate_bits = None
    combined_output_bits = None
    targets = proposal_target_ids(proposal)
    operation_saving = 0.0
    for target in targets:
        if target not in target_structure:
            row = row_by_id[target]
            gate_bits = trace.cone_gate_bits[row]
            output_bits = trace.cone_output_bits[row]
            target_structure[target] = (
                gate_bits,
                output_bits,
                sum(int(word).bit_count() for word in gate_bits),
                sum(int(word).bit_count() for word in output_bits),
            )
        gate_bits, output_bits, _, _ = target_structure[target]
        if len(targets) > 1:
            combined_gate_bits = (
                gate_bits.copy()
                if combined_gate_bits is None
                else np.bitwise_or(combined_gate_bits, gate_bits)
            )
            combined_output_bits = (
                output_bits.copy()
                if combined_output_bits is None
                else np.bitwise_or(combined_output_bits, output_bits)
            )
    rewrites = (
        proposal.rewrites
        if isinstance(proposal, RewriteGroup)
        else (proposal,)
    )
    for rewrite in rewrites:
        original = gate_by_id[rewrite.target_id]
        replacement = rewrite.replacement_gate(original)
        operation_saving += DEFAULT_OPERATION_COST.get(
            original.op,
            1.0,
        ) - DEFAULT_OPERATION_COST.get(replacement.op, 1.0)
    group_bonus = len(rewrites) - 1
    if len(targets) == 1:
        _, _, downstream_count, output_count = target_structure[targets[0]]
    else:
        downstream_count = sum(
            int(word).bit_count() for word in combined_gate_bits
        )
        output_count = sum(
            int(word).bit_count() for word in combined_output_bits
        )
    return (
        operation_saving,
        downstream_count + len(targets) + group_bonus,
        -output_count,
        -len(rewrites),
    )


def select_structural_pool(
    proposals: list[RewriteProposal],
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    limit: int,
) -> list[RewriteProposal]:
    gate_by_id = {gate.gate_id: gate for gate in circuit.gates}
    target_structure = {}
    row_by_id = trace.row_by_id
    def key(proposal):
        return (
            tuple(
                -value
                for value in structural_priority(
                    proposal,
                    circuit,
                    trace,
                    gate_by_id,
                    target_structure,
                    row_by_id,
                )
            ),
            proposal_key(proposal),
        )

    if limit <= 0 or limit >= len(proposals):
        return sorted(proposals, key=key)
    return heapq.nsmallest(limit, proposals, key=key)


def select_stratified_structural_pool(
    proposals: list[RewriteProposal],
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    limit: int,
) -> list[RewriteProposal]:
    gate_by_id = {gate.gate_id: gate for gate in circuit.gates}
    target_structure = {}
    row_by_id = trace.row_by_id

    def key(proposal):
        return (
            tuple(
                -value
                for value in structural_priority(
                    proposal,
                    circuit,
                    trace,
                    gate_by_id,
                    target_structure,
                    row_by_id,
                )
            ),
            proposal_key(proposal),
        )

    return stratified_ranked_pool(
        proposals,
        limit,
        proposal_of=lambda proposal: proposal,
        key=key,
    )


def select_deterministic_random_pool(
    proposals: list[RewriteProposal],
    limit: int,
    seed: int,
    iteration: int,
) -> list[RewriteProposal]:
    def key(proposal):
        serialized = proposal_key(proposal)
        digest = hashlib.sha256(
            f"{seed}:{iteration}:{serialized}".encode()
        ).digest()
        return digest, serialized

    ranked = sorted(proposals, key=key)
    return ranked if limit <= 0 else ranked[:limit]


def proposal_is_applicable(
    proposal: RewriteProposal,
    signatures: dict[int, tuple],
) -> bool:
    return all(target in signatures for target in proposal_target_ids(proposal))


def materialize_snapshots(
    input_circuit: Circuit,
    rewrite_log: list[dict],
    selected_steps: set[int],
    output_directory: Path,
    point_by_step: dict[int, dict],
) -> None:
    current = copy.deepcopy(input_circuit)
    selected_steps = set(selected_steps)
    selected_steps.add(0)
    if 0 in selected_steps:
        snapshot_dir = output_directory / "snapshots" / "step_0000"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        current.write_json(str(snapshot_dir / "circuit.json"))
        hardware = current.normalized_for_hardware_argmax()
        hardware.write_verilog_code(str(snapshot_dir / "circuit.v"))
    for entry in rewrite_log:
        proposal_from_dict(entry["proposal"]).apply(current)
        current.simplify()
        step = int(entry["step"])
        if step not in selected_steps:
            continue
        snapshot_dir = output_directory / "snapshots" / f"step_{step:04d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        current.write_json(str(snapshot_dir / "circuit.json"))
        hardware = current.normalized_for_hardware_argmax()
        hardware.write_verilog_code(str(snapshot_dir / "circuit.v"))
        point_by_step[step]["snapshot"] = f"snapshots/step_{step:04d}"


def run_search_v2(
    circuit: Circuit,
    reference: PackedCalibrationTrace,
    config: dict,
    output_directory: Path,
    estimator: SynthCostEstimator,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    encoded_inputs = np.stack(
        [reference.unpack_node(index) for index in range(reference.n_inputs)],
        axis=1,
    )
    initial_circuit = copy.deepcopy(circuit)
    current = copy.deepcopy(circuit)
    cost_mode = str(config.get("cost_mode", "synthesis-aware"))
    initial_cost = cost_record(current, estimator, cost_mode)
    existing_log_path = output_directory / "rewrite_log.json"
    rewrite_log = []
    points = []
    if config.get("resume", True) and existing_log_path.exists():
        rewrite_log = json.loads(existing_log_path.read_text())
        for entry in rewrite_log:
            if circuit_sha256(current) != entry["circuit_before_sha256"]:
                raise RuntimeError("resume log before-hash mismatch")
            proposal_from_dict(entry["proposal"]).apply(current)
            current.simplify()
            if circuit_sha256(current) != entry["circuit_after_sha256"]:
                raise RuntimeError("resume log after-hash mismatch")
    resumed_from_proposals = len(rewrite_log)
    current_trace = build_trace(current, encoded_inputs, reference.labels)
    current_cost = (
        rewrite_log[-1]["cost_after"]
        if rewrite_log
        else dict(initial_cost)
    )
    initial_metrics = behavior_metrics(
        reference.scores,
        reference,
        config["margin_reserve"],
    )
    if not within_budgets(initial_metrics, config):
        raise ValueError("starting circuit exceeds the configured v2 budgets")
    resumed_metrics = behavior_metrics(
        current_trace.scores,
        reference,
        config["margin_reserve"],
    )
    if not within_budgets(resumed_metrics, config):
        raise ValueError("resumed circuit exceeds the configured v2 budgets")

    start = time.perf_counter()
    points = [
        {
            "step": 0,
            "circuit_sha256": circuit_sha256(initial_circuit),
            "cost": initial_cost,
            "behavior": serializable_metrics(initial_metrics),
        }
    ]
    points.extend(
        {
            "step": int(entry["step"]),
            "circuit_sha256": entry["circuit_after_sha256"],
            "cost": entry["cost_after"],
            "behavior": entry["measured_behavior"],
        }
        for entry in rewrite_log
    )
    cache: dict[str, dict] = {}
    cache_hits = 0
    cache_misses = 0
    cache_invalidations = 0
    stop_reason = "maximum accepted rewrites reached"

    for iteration in range(
        len(rewrite_log) + 1,
        int(config["max_accepted_rewrites"]) + 1,
    ):
        _validate_trace(current, current_trace)
        signatures = gate_signatures(current)
        candidate_space = str(config.get("candidate_space", "targeted"))
        if candidate_space == "targeted":
            proposals: list[RewriteProposal] = list(
                generate_targeted_rewrites(current)
            )
        elif candidate_space == "full":
            proposals = list(generate_gate_rewrites(current))
        elif candidate_space == "constants-only":
            proposals = [
                proposal
                for proposal in generate_targeted_rewrites(current)
                if isinstance(proposal, GateRewrite)
                and proposal.kind.value.startswith("constant-")
            ]
        else:
            raise ValueError(f"unsupported candidate_space: {candidate_space}")
        if config.get("coordinated_rewrites", True):
            proposals.extend(
                generate_coordinated_cone_rewrites(
                    current,
                    int(config.get("maximum_coordinated_groups", 4096)),
                )
            )
        if config.get("structural_ranking", True):
            if config.get("stratified_shortlists", False):
                structural_pool = select_stratified_structural_pool(
                    proposals,
                    current,
                    current_trace,
                    int(config["structural_candidate_limit"]),
                )
            else:
                structural_pool = select_structural_pool(
                    proposals,
                    current,
                    current_trace,
                    int(config["structural_candidate_limit"]),
                )
        else:
            structural_pool = select_deterministic_random_pool(
                proposals,
                int(config["structural_candidate_limit"]),
                int(config.get("search_seed", 0)),
                iteration,
            )
        behavior_limit = int(config["behavior_candidate_limit"])
        if config.get("stratified_shortlists", False):
            structural_order = {
                proposal_key(proposal): index
                for index, proposal in enumerate(structural_pool)
            }
            behavior_pool = stratified_ranked_pool(
                structural_pool,
                behavior_limit,
                proposal_of=lambda proposal: proposal,
                key=lambda proposal: structural_order[proposal_key(proposal)],
            )
        else:
            behavior_pool = (
                structural_pool
                if behavior_limit <= 0
                else structural_pool[:behavior_limit]
            )
        current_metrics = behavior_metrics(
            current_trace.scores,
            reference,
            config["margin_reserve"],
        )
        feasible = []
        for proposal in behavior_pool:
            key = proposal_key(proposal)
            entry = cache.get(key)
            if entry is not None and proposal_is_applicable(proposal, signatures):
                cache_hits += 1
                scores = current_trace.scores + entry["score_delta"]
                metrics = behavior_metrics(
                    scores,
                    reference,
                    config["margin_reserve"],
                )
                affected_ids = entry["affected_ids"]
                affected_outputs = entry["affected_outputs"]
                source = "cache"
            else:
                cache_misses += 1
                evaluation = evaluate_proposal(
                    current,
                    current_trace,
                    proposal,
                    margin_reserve=config["margin_reserve"],
                    validate_trace=False,
                )
                scores = evaluation.scores
                metrics = behavior_metrics(
                    scores,
                    reference,
                    config["margin_reserve"],
                )
                affected_ids = set(evaluation.affected_gate_ids)
                affected_ids.update(proposal_target_ids(proposal))
                affected_outputs = set(evaluation.affected_output_indices)
                cache[key] = {
                    "score_delta": evaluation.scores - current_trace.scores,
                    "affected_ids": affected_ids,
                    "affected_outputs": affected_outputs,
                }
                source = "incremental"
            if within_budgets(metrics, config):
                incremental_risk = (
                    config["accuracy_penalty"]
                    * max(
                        0.0,
                        metrics["accuracy_loss"]
                        - current_metrics["accuracy_loss"],
                    )
                    + config["disagreement_penalty"]
                    * max(
                        0.0,
                        metrics["decision_flip_rate"]
                        - current_metrics["decision_flip_rate"],
                    )
                    + config["margin_penalty"]
                    * max(
                        0.0,
                        metrics["margin_loss_mean"]
                        - current_metrics["margin_loss_mean"],
                    )
                    + config.get("low_margin_penalty", 0.0)
                    * max(
                        0.0,
                        metrics["low_margin_rate"]
                        - current_metrics["low_margin_rate"],
                    )
                )
                feasible.append(
                    {
                        "proposal": proposal,
                        "scores": scores,
                        "metrics": metrics,
                        "risk": incremental_risk,
                        "source": source,
                        "affected_ids": affected_ids,
                        "affected_outputs": affected_outputs,
                    }
                )
        feasible.sort(
            key=lambda item: (
                item["risk"],
                item["metrics"]["accuracy_loss"],
                item["metrics"]["decision_flip_rate"],
                proposal_key(item["proposal"]),
            )
        )
        if config.get("stratified_shortlists", False):
            feasible_order = {
                proposal_key(item["proposal"]): index
                for index, item in enumerate(feasible)
            }
            shortlist = stratified_ranked_pool(
                feasible,
                int(config["exact_cost_shortlist"]),
                proposal_of=lambda item: item["proposal"],
                key=lambda item: feasible_order[proposal_key(item["proposal"])],
            )
        else:
            shortlist = feasible[: int(config["exact_cost_shortlist"])]
        best = None
        for item in shortlist:
            if item["source"] == "cache":
                revalidated = evaluate_proposal(
                    current,
                    current_trace,
                    item["proposal"],
                    margin_reserve=config["margin_reserve"],
                    validate_trace=False,
                )
                item = dict(item)
                item["scores"] = revalidated.scores
                item["metrics"] = behavior_metrics(
                    revalidated.scores,
                    reference,
                    config["margin_reserve"],
                )
                if not within_budgets(item["metrics"], config):
                    continue
                item["affected_ids"] = set(revalidated.affected_gate_ids)
                item["affected_ids"].update(
                    proposal_target_ids(item["proposal"])
                )
                item["affected_outputs"] = set(
                    revalidated.affected_output_indices
                )
                item["source"] = "cache-revalidated"
                item["risk"] = (
                    config["accuracy_penalty"]
                    * max(
                        0.0,
                        item["metrics"]["accuracy_loss"]
                        - current_metrics["accuracy_loss"],
                    )
                    + config["disagreement_penalty"]
                    * max(
                        0.0,
                        item["metrics"]["decision_flip_rate"]
                        - current_metrics["decision_flip_rate"],
                    )
                    + config["margin_penalty"]
                    * max(
                        0.0,
                        item["metrics"]["margin_loss_mean"]
                        - current_metrics["margin_loss_mean"],
                    )
                    + config.get("low_margin_penalty", 0.0)
                    * max(
                        0.0,
                        item["metrics"]["low_margin_rate"]
                        - current_metrics["low_margin_rate"],
                    )
                )
            rewritten = item["proposal"].apply_to_copy(current)
            rewritten.simplify()
            candidate_cost = cost_record(rewritten, estimator, cost_mode)
            reduction = (
                current_cost["estimated_abc_nodes"]
                - candidate_cost["estimated_abc_nodes"]
            )
            if reduction <= float(config.get("minimum_estimated_reduction", 0.0)):
                continue
            quality = reduction / (
                float(config["cost_epsilon"]) + item["risk"]
            )
            selection_key = (
                quality,
                reduction,
                -item["metrics"]["accuracy_loss"],
                -item["metrics"]["decision_flip_rate"],
            )
            if best is None or selection_key > best["selection_key"]:
                best = item | {
                    "circuit": rewritten,
                    "cost": candidate_cost,
                    "cost_reduction": reduction,
                    "quality": quality,
                    "selection_key": selection_key,
                }
        if best is None:
            stop_reason = "no feasible shortlisted proposal reduced estimated cost"
            break

        full_trace = build_trace(best["circuit"], encoded_inputs, reference.labels)
        if not np.array_equal(best["scores"], full_trace.scores):
            diagnostic = {
                "iteration": iteration,
                "proposal": best["proposal"].to_dict(),
                "source": best["source"],
                "maximum_score_difference": float(
                    np.max(np.abs(best["scores"] - full_trace.scores))
                ),
            }
            (output_directory / "FAIL_CLOSED.json").write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeError("v2 proposal failed exact full-trace verification")
        measured = behavior_metrics(
            full_trace.scores,
            reference,
            config["margin_reserve"],
        )
        if not within_budgets(measured, config):
            raise RuntimeError("accepted v2 proposal exceeded a hard budget")

        before_signatures = signatures
        after_signatures = gate_signatures(best["circuit"])
        changed_ids = {
            node_id
            for node_id in set(before_signatures) | set(after_signatures)
            if before_signatures.get(node_id) != after_signatures.get(node_id)
        }
        invalid_keys = []
        for key, entry in cache.items():
            if entry["affected_ids"] & changed_ids:
                invalid_keys.append(key)
        for key in invalid_keys:
            del cache[key]
        cache_invalidations += len(invalid_keys)

        before_sha = circuit_sha256(current)
        after_sha = circuit_sha256(best["circuit"])
        log_entry = {
            "step": iteration,
            "proposal": best["proposal"].to_dict(),
            "circuit_before_sha256": before_sha,
            "circuit_after_sha256": after_sha,
            "candidate_counts": {
                "generated": len(proposals),
                "structural_pool": len(structural_pool),
                "behavior_pool": len(behavior_pool),
                "feasible": len(feasible),
                "exact_cost_shortlist": len(shortlist),
            },
            "candidate_categories": {
                "generated": category_counts(proposals),
                "structural_pool": category_counts(structural_pool),
                "behavior_pool": category_counts(behavior_pool),
                "feasible": category_counts(
                    feasible,
                    proposal_of=lambda item: item["proposal"],
                ),
                "exact_cost_shortlist": category_counts(
                    shortlist,
                    proposal_of=lambda item: item["proposal"],
                ),
            },
            "evaluation_source": best["source"],
            "affected_gate_ids": sorted(best["affected_ids"]),
            "affected_output_indices": sorted(best["affected_outputs"]),
            "measured_behavior": serializable_metrics(measured),
            "cost_before": current_cost,
            "cost_after": best["cost"],
            "estimated_abc_reduction": best["cost_reduction"],
            "selection_quality": best["quality"],
        }
        rewrite_log.append(log_entry)
        current = best["circuit"]
        current_trace = full_trace
        current_cost = best["cost"]
        points.append(
            {
                "step": iteration,
                "circuit_sha256": after_sha,
                "cost": current_cost,
                "behavior": log_entry["measured_behavior"],
            }
        )
        print(
            json.dumps(
                {
                    "step": iteration,
                    "proposal_type": (
                        "group"
                        if isinstance(best["proposal"], RewriteGroup)
                        else "individual"
                    ),
                    "proposal_category": proposal_category(best["proposal"]),
                    "accuracy_loss": measured["accuracy_loss"],
                    "disagreement": measured["decision_flip_rate"],
                    "live_gates": current_cost["live_gates"],
                    "estimated_abc_nodes": current_cost[
                        "estimated_abc_nodes"
                    ],
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "elapsed_seconds": time.perf_counter() - start,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if iteration % int(config.get("log_flush_every", 10)) == 0:
            (output_directory / "rewrite_log.json").write_text(
                json.dumps(rewrite_log, indent=2, sort_keys=True) + "\n"
            )

    pareto = []
    for budget in config["pareto_accuracy_budgets"]:
        eligible = [
            point
            for point in points
            if point["behavior"]["accuracy_loss"] <= float(budget) + 1e-12
            and point["behavior"]["decision_flip_rate"]
            <= config["maximum_disagreement"] + 1e-12
            and point["behavior"]["maximum_per_class_accuracy_loss"]
            <= config["maximum_per_class_accuracy_loss"] + 1e-12
            and point["behavior"]["maximum_per_class_disagreement"]
            <= config["maximum_per_class_disagreement"] + 1e-12
        ]
        if not eligible:
            continue
        selected = min(
            eligible,
            key=lambda point: (
                point["cost"]["estimated_abc_nodes"],
                point["behavior"]["accuracy_loss"],
                point["step"],
            ),
        )
        pareto.append(
            {
                "accuracy_budget": float(budget),
                "selected_step": selected["step"],
                "accuracy_loss": selected["behavior"]["accuracy_loss"],
                "disagreement": selected["behavior"]["decision_flip_rate"],
                "maximum_per_class_accuracy_loss": selected["behavior"][
                    "maximum_per_class_accuracy_loss"
                ],
                "maximum_per_class_disagreement": selected["behavior"][
                    "maximum_per_class_disagreement"
                ],
                "live_gates": selected["cost"]["live_gates"],
                "estimated_abc_nodes": selected["cost"]["estimated_abc_nodes"],
            }
        )
    point_by_step = {point["step"]: point for point in points}
    selected_steps = {point["selected_step"] for point in pareto}
    selected_steps.add(points[-1]["step"])
    materialize_snapshots(
        initial_circuit,
        rewrite_log,
        selected_steps,
        output_directory,
        point_by_step,
    )
    for record in pareto:
        record["snapshot"] = point_by_step[record["selected_step"]].get(
            "snapshot",
            f"snapshots/step_{record['selected_step']:04d}",
        )

    summary = {
        "format_version": 2,
        "status": "completed",
        "development_run": True,
        "method": "MarginSynth-v2",
        "stop_reason": stop_reason,
        "accepted_proposals": len(rewrite_log),
        "resumed_from_proposals": resumed_from_proposals,
        "accepted_individual_rewrites": sum(
            (
                len(entry["proposal"]["rewrites"])
                if entry["proposal"].get("proposal_type") == "group"
                else 1
            )
            for entry in rewrite_log
        ),
        "accepted_coordinated_groups": sum(
            entry["proposal"].get("proposal_type") == "group"
            for entry in rewrite_log
        ),
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "final_behavior": points[-1]["behavior"],
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "invalidations": cache_invalidations,
            "remaining_entries": len(cache),
        },
        "wall_seconds": time.perf_counter() - start,
        "peak_process_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "pareto": pareto,
        "points": points,
        "cost_model": estimator.to_dict(),
    }
    (output_directory / "rewrite_log.json").write_text(
        json.dumps(rewrite_log, indent=2, sort_keys=True) + "\n"
    )
    (output_directory / "search_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
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
    model_path = run_dir / config["synth_cost_model"]
    estimator = SynthCostEstimator.from_json(model_path)
    provenance = {
        "source_revision": git_revision(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_source": str(cli.config),
        "config_sha256": sha256_file(cli.config),
        "input_circuit": config["input_circuit"],
        "input_circuit_sha256": sha256_file(run_dir / config["input_circuit"]),
        "input_trace": config["input_trace"],
        "input_trace_metadata_sha256": sha256_file(
            run_dir / config["input_trace"] / "metadata.json"
        ),
        "synth_cost_model": config["synth_cost_model"],
        "synth_cost_model_sha256": sha256_file(model_path),
        "selection_partition": "calibration",
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
    summary = run_search_v2(
        circuit,
        trace,
        config,
        output_directory,
        estimator,
    )
    write_artifact_manifest(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
