#!/usr/bin/env python3
"""Aggregate full-effort CIFAR-10 compression validation results."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_final_cifar10_compression.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table2_final_cifar10_compression" / "queue_summary.json"
)
SUMMARY_ROOT = ROOT / "summary"
EXPECTED_RUNS = 18
EXPECTED_SEEDS = [0, 1, 2]
MEDIUM_RANDOM_VALIDATION_MEAN = 0.5508399879932404
NONINFERIORITY_MARGIN = 0.003
STUDENT_T_975_DF2 = 4.302652729911275


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    queue_summary = json.loads(QUEUE_SUMMARY.read_text())
    if queue_summary["failed"] or queue_summary["skipped"]:
        raise RuntimeError(
            "full-effort queue is incomplete: "
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
        row = {
            "name": entry["name"],
            "cell": entry["cell"],
            "family": entry["family"],
            "candidate": entry["candidate"],
            "seed": config["seed"],
            "best_validation_hard_accuracy": (
                summary["best_validation_hard_accuracy"]
            ),
            "final_validation_hard_accuracy": (
                summary["final_metrics"]["val_acc_discrete"]
            ),
            "wall_seconds": summary["wall_seconds"],
            "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
            "dense_gate_count": summary["cost"]["dense_gate_count"],
            "trainable_parameters": summary["cost"]["trainable_parameters"],
            "training_routing_parameters": (
                summary["cost"]["training_routing_parameters"]
            ),
            "architecture": config["architecture"],
            "parametrization": config["parametrization"],
        }
        runs.append(row)
        grouped[(entry["cell"], entry["family"])].append(row)
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

    aggregates = []
    paired = {}
    for cell in ("128k", "256k", "384k"):
        by_family = {}
        for family in ("random", "coverage_v3"):
            family_runs = grouped[(cell, family)]
            seeds = sorted(row["seed"] for row in family_runs)
            if seeds != EXPECTED_SEEDS:
                raise RuntimeError(
                    f"{cell}/{family} has seeds {seeds}, expected {EXPECTED_SEEDS}"
                )
            values = [
                row["best_validation_hard_accuracy"] for row in family_runs
            ]
            aggregate = {
                "cell": cell,
                "family": family,
                "candidate": family_runs[0]["candidate"],
                "seed_count": len(family_runs),
                "mean_best_validation_hard_accuracy": statistics.mean(values),
                "std_best_validation_hard_accuracy": statistics.stdev(values),
                "mean_wall_seconds": statistics.mean(
                    row["wall_seconds"] for row in family_runs
                ),
                "max_peak_gpu_memory_bytes": max(
                    row["peak_gpu_memory_bytes"] for row in family_runs
                ),
                "dense_gate_count": family_runs[0]["dense_gate_count"],
                "trainable_parameters": family_runs[0]["trainable_parameters"],
                "training_routing_parameters": (
                    family_runs[0]["training_routing_parameters"]
                ),
                "architecture": family_runs[0]["architecture"],
                "parametrization": family_runs[0]["parametrization"],
            }
            aggregates.append(aggregate)
            by_family[family] = {
                row["seed"]: row["best_validation_hard_accuracy"]
                for row in family_runs
            }
        differences = [
            by_family["coverage_v3"][seed] - by_family["random"][seed]
            for seed in EXPECTED_SEEDS
        ]
        difference_mean = statistics.mean(differences)
        difference_std = statistics.stdev(differences)
        half_width = STUDENT_T_975_DF2 * difference_std / len(differences) ** 0.5
        paired[cell] = {
            "random_mean": statistics.mean(by_family["random"].values()),
            "coverage_mean": statistics.mean(by_family["coverage_v3"].values()),
            "mean_difference": difference_mean,
            "mean_difference_percentage_points": 100 * difference_mean,
            "std_difference": difference_std,
            "student_t_ci95": [
                difference_mean - half_width,
                difference_mean + half_width,
            ],
            "per_seed_differences": {
                str(seed): by_family["coverage_v3"][seed]
                - by_family["random"][seed]
                for seed in EXPECTED_SEEDS
            },
        }

    threshold = MEDIUM_RANDOM_VALIDATION_MEAN - NONINFERIORITY_MARGIN
    crossing = next(
        (
            cell for cell in ("128k", "256k", "384k")
            if paired[cell]["coverage_mean"] >= threshold
        ),
        None,
    )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "table2_cifar10_compression_final3.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(aggregates[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(aggregates)
    payload = {
        "phase": queue["phase"],
        "selection_metric": queue["selection_metric"],
        "test_set_used": False,
        "run_count": len(runs),
        "seeds": EXPECTED_SEEDS,
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "queue_wall_seconds": queue_summary["wall_seconds"],
        "medium_random_validation_reference": MEDIUM_RANDOM_VALIDATION_MEAN,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_threshold": threshold,
        "selected_crossing_for_five_seeds": crossing,
        "aggregates": aggregates,
        "paired_results": paired,
        "runs": sorted(runs, key=lambda row: row["name"]),
    }
    json_path = SUMMARY_ROOT / "table2_cifar10_compression_final3.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json_path)
    for cell in ("128k", "256k", "384k"):
        result = paired[cell]
        print(
            f"{cell}: CoverageDLGN {100 * result['coverage_mean']:.3f}%, "
            f"random {100 * result['random_mean']:.3f}%, "
            f"gain {result['mean_difference_percentage_points']:+.3f} pp"
        )
    print(f"selected crossing: {crossing}")


if __name__ == "__main__":
    main()
