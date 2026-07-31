#!/usr/bin/env python3
"""Summarize completed WARP Figure 4 / frozen Legacy V4 validation runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocols" / "warp_fig4_cifar10_medium.json"
OUTPUT_JSON = ROOT / "summary" / "warp_fig4_cifar10_medium.json"
OUTPUT_CSV = ROOT / "summary" / "warp_fig4_cifar10_medium.csv"
ARMS = [
    "warp_fixed_uniform",
    "warp_fixed_distributive",
    "warp_learnable",
    "paired_random_fixed_uniform",
    "legacy_v4_fixed_uniform",
    "paired_random_learnable",
    "legacy_v4_learnable",
]


def run_path(arm: str, seed: int) -> Path:
    return RESULTS / f"warp_fig4_medium_{arm}_seed{seed}"


def completed_accuracy(arm: str, seed: int) -> float | None:
    root = run_path(arm, seed)
    summary_path = root / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        return 100.0 * float(summary["best_validation_hard_accuracy"])
    early_stop_path = root / "early_stop.json"
    if early_stop_path.is_file():
        summary = json.loads(early_stop_path.read_text())
        if not summary["complete_for_frozen_30000_step_protocol"]:
            return None
        return float(summary["best_validation_hard_accuracy_percent"])
    return None


def paired_effect(left: list[float], right: list[float]) -> dict:
    differences = [r - l for l, r in zip(left, right)]
    payload = {
        "paired_differences_percentage_points": differences,
        "paired_mean_percentage_points": mean(differences),
        "positive_seed_count": sum(value > 0 for value in differences),
        "seed_count": len(differences),
    }
    if len(differences) >= 2:
        critical = {2: 12.7062047, 3: 4.3026527}.get(len(differences))
        if critical is not None:
            half = critical * stdev(differences) / math.sqrt(len(differences))
            payload["paired_t95_ci_percentage_points"] = [
                mean(differences) - half,
                mean(differences) + half,
            ]
    return payload


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol["common_training"]["seeds"]
    values = {
        arm: {
            str(seed): completed_accuracy(arm, seed)
            for seed in seeds
        }
        for arm in ARMS
    }
    means = {
        arm: mean(value for value in rows.values() if value is not None)
        for arm, rows in values.items()
        if any(value is not None for value in rows.values())
    }
    effects = {}
    for label, left_arm, right_arm in [
        (
            "legacy_v4_minus_paired_random_fixed_uniform",
            "paired_random_fixed_uniform",
            "legacy_v4_fixed_uniform",
        ),
        (
            "legacy_v4_minus_paired_random_learnable",
            "paired_random_learnable",
            "legacy_v4_learnable",
        ),
    ]:
        paired = [
            (values[left_arm][str(seed)], values[right_arm][str(seed)])
            for seed in seeds
        ]
        complete = [(left, right) for left, right in paired if left is not None and right is not None]
        if complete:
            effects[label] = paired_effect(
                [left for left, _ in complete],
                [right for _, right in complete],
            )

    reported = protocol["reported_validation_endpoints_percent_approximate"]
    reproduction_differences = {}
    for arm, key in [
        ("warp_fixed_uniform", "fixed_uniform"),
        ("warp_fixed_distributive", "fixed_distributive"),
        ("warp_learnable", "learnable"),
    ]:
        if arm in means:
            reproduction_differences[arm] = (
                means[arm] - float(reported[key])
            )

    payload = {
        "phase": protocol["phase"],
        "provenance": "TRIED-PARTIAL"
        if any(value is None for rows in values.values() for value in rows.values())
        else "TRIED-COMPLETE",
        "metric": protocol["metric"],
        "heldout_test_used": False,
        "per_seed_validation_percent": values,
        "completed_mean_validation_percent": means,
        "difference_from_approximate_reported_endpoint_pp": reproduction_differences,
        "paired_effects": effects,
        "frozen_method_policy": protocol["frozen_method_policy"],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arm", "seed", "best_hardened_validation_percent"])
        for arm, rows in values.items():
            for seed, value in rows.items():
                writer.writerow([arm, seed, "" if value is None else value])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
