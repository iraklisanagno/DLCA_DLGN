#!/usr/bin/env python3
"""Audit and summarize five-seed full CIFAR-10 L primary runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_l_final.json"
QUEUE_SUMMARY = ROOT / "logs" / "table2_l_final" / "queue_summary.json"
JSON_PATH = ROOT / "summary" / "table2_l_final.json"
CSV_PATH = ROOT / "summary" / "table2_l_final.csv"
EXPECTED_SEEDS = list(range(5))
T_CRIT_DF4 = 2.7764451051977987


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"L final queue failed: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != expected:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rows = []
    groups = defaultdict(list)
    revisions = Counter()
    hashes = Counter()
    test_count = 0
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        run = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        row = {
            "name": entry["name"],
            "family": entry["family"],
            "provenance": (
                "REPRODUCED" if entry["family"] == "random" else "OUR-FINAL"
            ),
            "seed": config["seed"],
            "best_validation_hard_accuracy": (
                run["best_validation_hard_accuracy"]
            ),
            "final_validation_hard_accuracy": (
                run["final_metrics"]["val_acc_discrete"]
            ),
            "wall_seconds": run["wall_seconds"],
            "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
            "dense_gate_count": run["cost"]["dense_gate_count"],
            "trainable_parameters": run["cost"]["trainable_parameters"],
            "training_routing_parameters": (
                run["cost"]["training_routing_parameters"]
            ),
            "source_revision": environment["source_revision"],
            "training_implementation_sha256": environment[
                "training_implementation_sha256"
            ],
        }
        test_path = run_dir / "test_metrics.json"
        if test_path.is_file():
            test = json.loads(test_path.read_text())
            row.update({
                "test_hard_accuracy": test["test_hard_accuracy"],
                "test_relaxed_accuracy": test["test_relaxed_accuracy"],
                "test_examples": test["test_examples"],
                "test_checkpoint": test["checkpoint"],
                "test_validation_selection_step": (
                    test["validation_selection_step"]
                ),
            })
            test_count += 1
        rows.append(row)
        groups[entry["family"]].append(row)
        revisions[row["source_revision"]] += 1
        hashes[row["training_implementation_sha256"]] += 1

    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    if test_count not in {0, len(rows)}:
        raise RuntimeError(f"partial test evaluation: {test_count}/{len(rows)}")
    test_set_used = test_count == len(rows)

    methods = []
    for family in ("random", "coverage_v3"):
        current = sorted(groups[family], key=lambda row: row["seed"])
        seeds = [row["seed"] for row in current]
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"{family} has seeds {seeds}")
        values = [
            row["best_validation_hard_accuracy"] for row in current
        ]
        method = {
            "family": family,
            "provenance": current[0]["provenance"],
            "seed_count": len(current),
            "seeds": seeds,
            "mean_best_validation_hard_accuracy": statistics.mean(values),
            "std_best_validation_hard_accuracy": statistics.stdev(values),
            "mean_wall_seconds": statistics.mean(
                row["wall_seconds"] for row in current
            ),
            "max_peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in current
            ),
            "dense_gate_count": current[0]["dense_gate_count"],
            "trainable_parameters": current[0]["trainable_parameters"],
            "training_routing_parameters": (
                current[0]["training_routing_parameters"]
            ),
        }
        if test_set_used:
            method.update({
                "mean_test_hard_accuracy": statistics.mean(
                    row["test_hard_accuracy"] for row in current
                ),
                "std_test_hard_accuracy": statistics.stdev(
                    row["test_hard_accuracy"] for row in current
                ),
            })
        methods.append(method)

    random_by_seed = {row["seed"]: row for row in groups["random"]}
    v3_by_seed = {row["seed"]: row for row in groups["coverage_v3"]}
    metric = (
        "test_hard_accuracy"
        if test_set_used
        else "best_validation_hard_accuracy"
    )
    diffs = [
        v3_by_seed[seed][metric] - random_by_seed[seed][metric]
        for seed in EXPECTED_SEEDS
    ]
    mean = statistics.mean(diffs)
    margin = T_CRIT_DF4 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    paired = {
        "metric": metric,
        "seed_differences_pp": [100 * value for value in diffs],
        "mean_gain_pp": 100 * mean,
        "ci95_pp": [100 * (mean - margin), 100 * (mean + margin)],
        "all_positive": all(value > 0 for value in diffs),
    }
    payload = {
        "phase": queue["phase"],
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": test_set_used,
        "run_count": len(rows),
        "source_revisions": dict(revisions),
        "training_implementation_sha256": next(iter(hashes)),
        "queue_audit": {
            "finished_count": len(audit["finished"]),
            "skipped_count": len(audit["skipped"]),
            "failed_count": len(audit["failed"]),
            "wall_seconds": audit["wall_seconds"],
        },
        "paired_comparison": paired,
        "methods": methods,
        "runs": sorted(rows, key=lambda row: row["name"]),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        csv_rows = []
        for method in methods:
            row = dict(method)
            row["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            csv_rows.append(row)
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(JSON_PATH)
    for method in methods:
        print(
            f"{method['family']}: "
            f"{100 * method['mean_best_validation_hard_accuracy']:.3f}% "
            f"+/- {100 * method['std_best_validation_hard_accuracy']:.3f}%"
        )
    print("paired", paired)


if __name__ == "__main__":
    main()
