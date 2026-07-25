#!/usr/bin/env python3
"""Aggregate the locked three-seed, 20K Table 1 MNIST selection runs."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


RESULT_ROOT = Path("experiments/coverage_dlgn/results")
SUMMARY_ROOT = Path("experiments/coverage_dlgn/summary")
QUEUE_SUMMARY = Path(
    "experiments/coverage_dlgn/logs/table1_select_mnist/queue_summary.json"
)
EXPECTED_RUNS = 33
EXPECTED_SEEDS = [0, 1, 2]
TIE_TOLERANCE = 1e-8


def family(name: str) -> str:
    if "_v3_" in name:
        return "coverage_v3"
    if "_mommen_" in name:
        return "mommen"
    if "_lilogic_" in name:
        return "lilogic"
    if "_bitlogic_" in name:
        return "bitlogic"
    if "_random_" in name:
        return "random"
    raise ValueError(f"unrecognized selection family: {name}")


def candidate_and_seed(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"select_(.+)_seed(\d+)", name)
    if match is None:
        raise ValueError(f"unrecognized selection run name: {name}")
    return match.group(1), int(match.group(2))


def main() -> None:
    queue_summary = json.loads(QUEUE_SUMMARY.read_text())
    if queue_summary["failed"] or queue_summary["skipped"]:
        raise RuntimeError(
            "selection queue is incomplete: "
            f"failed={queue_summary['failed']}, "
            f"skipped={queue_summary['skipped']}"
        )
    finished = queue_summary["finished"]
    if len(finished) != EXPECTED_RUNS:
        raise RuntimeError(
            f"expected {EXPECTED_RUNS} successful queue entries, "
            f"got {len(finished)}"
        )
    if any(entry["return_code"] != 0 for entry in finished):
        raise RuntimeError("selection queue contains a nonzero return code")

    runs = []
    revisions = Counter()
    training_hashes = Counter()
    grouped = defaultdict(list)
    for summary_path in sorted(
        RESULT_ROOT.glob("select_table1_mnist_*/run_summary.json")
    ):
        run_dir = summary_path.parent
        candidate, seed = candidate_and_seed(run_dir.name)
        summary = json.loads(summary_path.read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        cost = summary["cost"]
        row = {
            "name": run_dir.name,
            "candidate": candidate,
            "family": family(run_dir.name),
            "seed": seed,
            "best_validation_hard_accuracy": (
                summary["best_validation_hard_accuracy"]
            ),
            "final_validation_hard_accuracy": (
                summary["final_metrics"]["val_acc_discrete"]
            ),
            "final_validation_relaxed_accuracy": (
                summary["final_metrics"]["val_acc_relaxed"]
            ),
            "wall_seconds": summary["wall_seconds"],
            "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
            "dense_gate_count": cost["dense_gate_count"],
            "trainable_parameters": cost["trainable_parameters"],
            "training_routing_parameters": (
                cost["training_routing_parameters"]
            ),
            "deployed_routing_bits": cost["deployed_routing_bits"],
            "architecture": config["architecture"],
            "parametrization": config["parametrization"],
        }
        runs.append(row)
        grouped[candidate].append(row)
        revisions[environment["source_revision"]] += 1
        training_hashes[
            environment["training_implementation_sha256"]
        ] += 1

    if len(runs) != EXPECTED_RUNS:
        raise RuntimeError(
            f"expected {EXPECTED_RUNS} complete result directories, "
            f"got {len(runs)}"
        )
    if len(revisions) != 1 or len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed implementations: revisions={revisions}, "
            f"training_hashes={training_hashes}"
        )

    candidates = []
    for candidate, candidate_runs in grouped.items():
        seeds = sorted(row["seed"] for row in candidate_runs)
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{candidate} has seeds {seeds}, expected {EXPECTED_SEEDS}"
            )
        stable_fields = [
            "family",
            "dense_gate_count",
            "trainable_parameters",
            "training_routing_parameters",
            "deployed_routing_bits",
            "architecture",
            "parametrization",
        ]
        for field in stable_fields:
            values = {row[field] for row in candidate_runs}
            if len(values) != 1:
                raise RuntimeError(
                    f"{candidate} has inconsistent {field}: {values}"
                )
        best = [
            row["best_validation_hard_accuracy"]
            for row in candidate_runs
        ]
        final_hard = [
            row["final_validation_hard_accuracy"]
            for row in candidate_runs
        ]
        final_relaxed = [
            row["final_validation_relaxed_accuracy"]
            for row in candidate_runs
        ]
        candidates.append({
            "candidate": candidate,
            **{
                field: candidate_runs[0][field]
                for field in stable_fields
            },
            "seed_count": len(candidate_runs),
            "mean_best_validation_hard_accuracy": statistics.mean(best),
            "std_best_validation_hard_accuracy": statistics.stdev(best),
            "mean_final_validation_hard_accuracy": statistics.mean(
                final_hard
            ),
            "std_final_validation_hard_accuracy": statistics.stdev(
                final_hard
            ),
            "mean_final_validation_relaxed_accuracy": statistics.mean(
                final_relaxed
            ),
            "mean_wall_seconds": statistics.mean(
                row["wall_seconds"] for row in candidate_runs
            ),
            "max_peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in candidate_runs
            ),
        })

    # Accuracy is primary. Values within 1e-8 are treated as a numerical tie;
    # the smaller deployed-routing representation and lexical candidate name
    # provide deterministic, predeclared-cost-preserving tie-breaks.
    selected = {}
    for method in ["random", "coverage_v3", "mommen", "lilogic", "bitlogic"]:
        method_rows = [row for row in candidates if row["family"] == method]
        best_accuracy = max(
            row["mean_best_validation_hard_accuracy"]
            for row in method_rows
        )
        tied = [
            row
            for row in method_rows
            if math.isclose(
                row["mean_best_validation_hard_accuracy"],
                best_accuracy,
                rel_tol=0,
                abs_tol=TIE_TOLERANCE,
            )
        ]
        winner = min(
            tied,
            key=lambda row: (
                row["deployed_routing_bits"],
                row["training_routing_parameters"],
                row["candidate"],
            ),
        )
        selected[method] = winner["candidate"]

    ranked = sorted(
        candidates,
        key=lambda row: (
            row["family"],
            -row["mean_best_validation_hard_accuracy"],
            row["candidate"],
        ),
    )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "table1_mnist_selection.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ranked[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ranked)

    payload = {
        "phase": "table1_select_mnist",
        "selection_metric": "mean best hardened validation accuracy",
        "test_set_used": False,
        "run_count": len(runs),
        "candidate_count": len(candidates),
        "seeds": EXPECTED_SEEDS,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": [
            "smaller deployed routing bit count",
            "smaller training routing parameter count",
            "lexical candidate name",
        ],
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "queue_wall_seconds": queue_summary["wall_seconds"],
        "selected_for_final": selected,
        "candidates": sorted(
            candidates,
            key=lambda row: (
                -row["mean_best_validation_hard_accuracy"],
                row["candidate"],
            ),
        ),
        "runs": sorted(runs, key=lambda row: row["name"]),
    }
    json_path = SUMMARY_ROOT / "table1_mnist_selection.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json_path)
    for method, candidate in selected.items():
        row = next(
            item for item in candidates if item["candidate"] == candidate
        )
        mean = 100 * row["mean_best_validation_hard_accuracy"]
        std = 100 * row["std_best_validation_hard_accuracy"]
        print(f"{method}: {mean:.3f}% +/- {std:.3f}% {candidate}")


if __name__ == "__main__":
    main()
