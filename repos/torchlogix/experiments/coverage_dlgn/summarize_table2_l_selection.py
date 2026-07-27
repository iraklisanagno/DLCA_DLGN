#!/usr/bin/env python3
"""Audit and summarize paired three-seed CIFAR-10 L selection runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_l_selection.json"
QUEUE_SUMMARY = ROOT / "logs" / "table2_l_selection" / "queue_summary.json"
JSON_PATH = ROOT / "summary" / "table2_l_selection.json"
CSV_PATH = ROOT / "summary" / "table2_l_selection.csv"
EXPECTED_SEEDS = [0, 1, 2]
T_CRIT_DF2 = 4.302652729911275


def paired_stats(left: list[dict], right: list[dict]) -> dict:
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    diffs = [
        right_by_seed[seed]["best_validation_hard_accuracy"]
        - left_by_seed[seed]["best_validation_hard_accuracy"]
        for seed in EXPECTED_SEEDS
    ]
    mean = statistics.mean(diffs)
    margin = T_CRIT_DF2 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {
        "paired_seed_differences_pp": [100 * value for value in diffs],
        "mean_gain_pp": 100 * mean,
        "ci95_pp": [100 * (mean - margin), 100 * (mean + margin)],
        "all_positive": all(value > 0 for value in diffs),
    }


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"L selection queue failed: {audit['failed']}")
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
    hashes = set()
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        run = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"selection touched held-out test: {run_dir}")
        row = {
            "name": entry["name"],
            "family": entry["family"],
            "topology": entry["topology"],
            "parametrization": entry["parametrization"],
            "seed": config["seed"],
            "best_validation_hard_accuracy": (
                run["best_validation_hard_accuracy"]
            ),
            "wall_seconds": run["wall_seconds"],
            "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
            "dense_gate_count": run["cost"]["dense_gate_count"],
            "source_revision": environment["source_revision"],
            "training_implementation_sha256": environment[
                "training_implementation_sha256"
            ],
        }
        rows.append(row)
        groups[entry["family"]].append(row)
        hashes.add(row["training_implementation_sha256"])

    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    methods = []
    for family in ("raw_random", "raw_v3", "warp_random", "warp_v3"):
        current = sorted(groups[family], key=lambda row: row["seed"])
        seeds = [row["seed"] for row in current]
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"{family} has seeds {seeds}")
        values = [
            row["best_validation_hard_accuracy"] for row in current
        ]
        methods.append({
            "family": family,
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
        })

    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": False,
        "run_count": len(rows),
        "primary_raw_comparison": paired_stats(
            groups["raw_random"], groups["raw_v3"]
        ),
        "secondary_warp_comparison": paired_stats(
            groups["warp_random"], groups["warp_v3"]
        ),
        "frozen_primary_finalists": ["raw_random", "raw_v3"],
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
    print("raw", payload["primary_raw_comparison"])
    print("warp", payload["secondary_warp_comparison"])


if __name__ == "__main__":
    main()
