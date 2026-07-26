#!/usr/bin/env python3
"""Aggregate the paired three-seed, 20K CIFAR-10 compression selection."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_select_cifar10_compression.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table2_select_cifar10_compression" / "queue_summary.json"
)
SUMMARY_ROOT = ROOT / "summary"
EXPECTED_RUNS = 36
EXPECTED_SEEDS = [0, 1, 2]
TIE_TOLERANCE = 1e-8


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    queue_summary = json.loads(QUEUE_SUMMARY.read_text())
    if queue_summary["failed"] or queue_summary["skipped"]:
        raise RuntimeError(
            "selection queue is incomplete: "
            f"failed={queue_summary['failed']}, "
            f"skipped={queue_summary['skipped']}"
        )
    if len(queue_summary["finished"]) != EXPECTED_RUNS:
        raise RuntimeError(
            f"expected {EXPECTED_RUNS} successful queue entries, "
            f"got {len(queue_summary['finished'])}"
        )

    runs = []
    grouped = defaultdict(list)
    revisions = Counter()
    training_hashes = Counter()
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        summary = json.loads((run_dir / "run_summary.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        seed = config["seed"]
        cost = summary["cost"]
        row = {
            "name": entry["name"],
            "cell": entry["cell"],
            "family": entry["family"],
            "candidate": entry["candidate"],
            "seed": seed,
            "best_validation_hard_accuracy": (
                summary["best_validation_hard_accuracy"]
            ),
            "final_validation_hard_accuracy": (
                summary["final_metrics"]["val_acc_discrete"]
            ),
            "wall_seconds": summary["wall_seconds"],
            "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
            "dense_gate_count": cost["dense_gate_count"],
            "trainable_parameters": cost["trainable_parameters"],
            "training_routing_parameters": (
                cost["training_routing_parameters"]
            ),
            "architecture": config["architecture"],
            "parametrization": config["parametrization"],
        }
        runs.append(row)
        grouped[(entry["cell"], entry["family"], entry["candidate"])].append(row)
        revisions[environment["source_revision"]] += 1
        training_hashes[
            environment["training_implementation_sha256"]
        ] += 1

    if len(runs) != EXPECTED_RUNS:
        raise RuntimeError(f"expected {EXPECTED_RUNS} run artifacts, got {len(runs)}")
    if len(revisions) != 1 or len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed implementations: revisions={revisions}, "
            f"training_hashes={training_hashes}"
        )

    candidates = []
    for (cell, family, candidate), candidate_runs in grouped.items():
        seeds = sorted(row["seed"] for row in candidate_runs)
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{cell}/{candidate} has seeds {seeds}, expected {EXPECTED_SEEDS}"
            )
        best = [
            row["best_validation_hard_accuracy"] for row in candidate_runs
        ]
        candidates.append({
            "cell": cell,
            "family": family,
            "candidate": candidate,
            "seed_count": len(candidate_runs),
            "mean_best_validation_hard_accuracy": statistics.mean(best),
            "std_best_validation_hard_accuracy": statistics.stdev(best),
            "mean_final_validation_hard_accuracy": statistics.mean(
                row["final_validation_hard_accuracy"] for row in candidate_runs
            ),
            "mean_wall_seconds": statistics.mean(
                row["wall_seconds"] for row in candidate_runs
            ),
            "max_peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in candidate_runs
            ),
            "dense_gate_count": candidate_runs[0]["dense_gate_count"],
            "trainable_parameters": candidate_runs[0]["trainable_parameters"],
            "training_routing_parameters": (
                candidate_runs[0]["training_routing_parameters"]
            ),
            "architecture": candidate_runs[0]["architecture"],
            "parametrization": candidate_runs[0]["parametrization"],
        })

    selected = {}
    paired = {}
    for cell in ("128k", "256k", "384k"):
        coverage = [
            row for row in candidates
            if row["cell"] == cell and row["family"] == "coverage_v3"
        ]
        best_accuracy = max(
            row["mean_best_validation_hard_accuracy"] for row in coverage
        )
        tied = [
            row for row in coverage
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
                row["training_routing_parameters"],
                row["candidate"],
            ),
        )
        selected[cell] = winner["candidate"]

        random_runs = {
            row["seed"]: row["best_validation_hard_accuracy"]
            for row in runs
            if row["cell"] == cell and row["family"] == "random"
        }
        winner_runs = {
            row["seed"]: row["best_validation_hard_accuracy"]
            for row in runs
            if (
                row["cell"] == cell
                and row["family"] == "coverage_v3"
                and row["candidate"] == winner["candidate"]
            )
        }
        differences = [
            winner_runs[seed] - random_runs[seed] for seed in EXPECTED_SEEDS
        ]
        paired[cell] = {
            "winner": winner["candidate"],
            "random_mean": statistics.mean(random_runs.values()),
            "coverage_mean": statistics.mean(winner_runs.values()),
            "mean_difference": statistics.mean(differences),
            "mean_difference_percentage_points": (
                100 * statistics.mean(differences)
            ),
            "std_difference": statistics.stdev(differences),
            "per_seed_differences": {
                str(seed): winner_runs[seed] - random_runs[seed]
                for seed in EXPECTED_SEEDS
            },
        }

    ranked = sorted(
        candidates,
        key=lambda row: (
            row["cell"],
            row["family"],
            -row["mean_best_validation_hard_accuracy"],
            row["candidate"],
        ),
    )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "table2_cifar10_compression_selection.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ranked[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ranked)
    payload = {
        "phase": queue["phase"],
        "selection_metric": queue["selection_metric"],
        "test_set_used": False,
        "run_count": len(runs),
        "candidate_count": len(candidates),
        "seeds": EXPECTED_SEEDS,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": [
            "smaller training routing parameter count",
            "lexical candidate name",
        ],
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "queue_wall_seconds": queue_summary["wall_seconds"],
        "selected_for_full_training": selected,
        "paired_selection_results": paired,
        "candidates": ranked,
        "runs": sorted(runs, key=lambda row: row["name"]),
    }
    json_path = SUMMARY_ROOT / "table2_cifar10_compression_selection.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json_path)
    for cell in ("128k", "256k", "384k"):
        result = paired[cell]
        print(
            f"{cell}: {result['winner']}, "
            f"CoverageDLGN {100 * result['coverage_mean']:.3f}%, "
            f"random {100 * result['random_mean']:.3f}%, "
            f"paired gain {result['mean_difference_percentage_points']:+.3f} pp"
        )


if __name__ == "__main__":
    main()
