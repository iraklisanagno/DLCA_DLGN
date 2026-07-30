#!/usr/bin/env python3
"""Summarize the one-time CIFAR-10 256K/384K held-out evaluations."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_OUT = ROOT / "summary" / "table2_cifar10_compression_remaining_test.json"
CSV_OUT = ROOT / "summary" / "table2_cifar10_compression_remaining_test.csv"
T_DF2 = 4.302652729696142


def main() -> None:
    rows = []
    for budget in ("256k", "384k"):
        for family, candidate in (
            ("random", "random"),
            ("coverage_v3", "v3_incumbent"),
        ):
            for seed in (0, 1, 2):
                run_dir = (
                    ROOT / "results"
                    / f"final_table2_cifar10_{budget}_{candidate}_seed{seed}"
                )
                test = json.loads((run_dir / "test_metrics.json").read_text())
                run = json.loads((run_dir / "run_summary.json").read_text())
                rows.append({
                    "budget": budget,
                    "family": family,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "validation_hard_accuracy": test[
                        "validation_hard_accuracy"
                    ],
                    "test_hard_accuracy": test["test_hard_accuracy"],
                    "test_relaxed_accuracy": test["test_relaxed_accuracy"],
                    "dense_gate_count": run["cost"]["dense_gate_count"],
                    "trainable_parameters": run["cost"]["trainable_parameters"],
                    "deployed_routing_bits": run["cost"]["deployed_routing_bits"],
                })
    paired = {}
    family_stats = {}
    indexed = {
        (row["budget"], row["family"], row["seed"]): row for row in rows
    }
    for budget in ("256k", "384k"):
        family_stats[budget] = {}
        for family in ("random", "coverage_v3"):
            values = [
                indexed[(budget, family, seed)]["test_hard_accuracy"]
                for seed in (0, 1, 2)
            ]
            family_stats[budget][family] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "n": 3,
            }
        gains = [
            indexed[(budget, "coverage_v3", seed)]["test_hard_accuracy"]
            - indexed[(budget, "random", seed)]["test_hard_accuracy"]
            for seed in (0, 1, 2)
        ]
        mean = statistics.mean(gains)
        std = statistics.stdev(gains)
        half = T_DF2 * std / math.sqrt(3)
        paired[budget] = {
            "per_seed_gain_pp": [100 * gain for gain in gains],
            "paired_mean_gain_pp": 100 * mean,
            "paired_95ci_pp": [
                100 * (mean - half), 100 * (mean + half)
            ],
        }
    payload = {
        "phase": "table2_compression_remaining_test",
        "provenance": "OUR-FINAL",
        "test_set_used": True,
        "heldout_evaluation_count_per_checkpoint": 1,
        "family_stats": family_stats,
        "paired_results": paired,
        "rows": rows,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(JSON_OUT)
    for budget, result in paired.items():
        print(f"{budget}: {result['paired_mean_gain_pp']:+.3f} pp")


if __name__ == "__main__":
    main()
