#!/usr/bin/env python3
"""Summarize the predeclared five-seed unified-candidate promotion test."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocols" / "cifar10_conv_small_unified_five_seed.json"
OUTPUT_JSON = ROOT / "summary" / "cifar10_conv_small_unified_five_seed.json"
OUTPUT_CSV = ROOT / "summary" / "cifar10_conv_small_unified_five_seed.csv"
PATTERNS = {
    "random": "pilot_conv_cifar10_paper_small_random_seed{seed}",
    "frozen_v4": (
        "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}"
    ),
    "unified_candidate": {
        0: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed0",
        1: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed1",
        2: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed2",
        3: "pilot_conv_cifar10_paper_small_semantic_degree_balanced_seed3",
        4: "pilot_conv_cifar10_paper_small_semantic_degree_balanced_seed4",
    },
}


def _run_name(family: str, seed: int) -> str:
    pattern = PATTERNS[family]
    return pattern[seed] if isinstance(pattern, dict) else pattern.format(seed=seed)


def _run_summary(run_name: str) -> dict:
    path = RESULTS / run_name / "run_summary.json"
    return json.loads(path.read_text())


def _accuracy(run_name: str) -> float:
    return 100.0 * float(
        _run_summary(run_name)["best_validation_hard_accuracy"]
    )


def _paired_effect(left: list[float], right: list[float]) -> dict:
    differences = [r - l for l, r in zip(left, right)]
    center = mean(differences)
    standard_deviation = stdev(differences)
    half_width = 2.7764451051977987 * standard_deviation / math.sqrt(5)
    return {
        "paired_differences_percentage_points": differences,
        "paired_mean_percentage_points": center,
        "paired_std_percentage_points": standard_deviation,
        "paired_t95_ci_percentage_points": [
            center - half_width,
            center + half_width,
        ],
        "positive_seed_count": sum(value > 0.0 for value in differences),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol["training"]["paired_seeds"]
    values = {
        family: [_accuracy(_run_name(family, seed)) for seed in seeds]
        for family in PATTERNS
    }
    candidate_effect = _paired_effect(
        values["random"], values["unified_candidate"]
    )
    v4_effect = _paired_effect(values["random"], values["frozen_v4"])
    rule = protocol["promotion_rule"]
    mean_gate = (
        candidate_effect["paired_mean_percentage_points"]
        >= rule["minimum_paired_mean_gain_percentage_points"]
    )
    consistency_gate = (
        candidate_effect["positive_seed_count"]
        >= rule["minimum_positive_seed_count"]
    )
    promoted = mean_gate and consistency_gate
    resource_rows = {
        family: [
            _run_summary(_run_name(family, seed))
            for seed in protocol["training"]["new_seeds"]
        ]
        for family in PATTERNS
    }
    resource_check = {}
    for family, rows in resource_rows.items():
        resource_check[family] = {
            "mean_wall_seconds": mean(row["wall_seconds"] for row in rows),
            "mean_topology_construction_seconds": mean(
                sum(
                    layer["construction_seconds"]
                    for layer in row["topology"]
                    if layer.get("structure") == "conv"
                )
                for row in rows
            ),
            "mean_peak_gpu_memory_bytes": mean(
                row["peak_gpu_memory_bytes"] for row in rows
            ),
            "cost": rows[0]["cost"],
            "cost_identical_across_new_seeds": all(
                row["cost"] == rows[0]["cost"] for row in rows
            ),
        }
    common_cost = resource_check["random"]["cost"]
    cost_identical_across_methods = all(
        row["cost"] == common_cost
        for rows in resource_rows.values()
        for row in rows
    )
    payload = {
        "phase": protocol["phase"],
        "dataset": protocol["dataset"],
        "architecture": protocol["architecture"],
        "metric": protocol["metric"],
        "provenance": "TRIED",
        "heldout_test_used": False,
        "historical_candidate_seeds_reused_under_test_proven_equivalence": [
            0,
            1,
            2,
        ],
        "per_seed_percent": values,
        "mean_percent": {
            family: mean(family_values)
            for family, family_values in values.items()
        },
        "paired_effects": {
            "unified_candidate_minus_random": candidate_effect,
            "frozen_v4_minus_random": v4_effect,
            "unified_candidate_minus_frozen_v4": _paired_effect(
                values["frozen_v4"], values["unified_candidate"]
            ),
        },
        "promotion_decision": {
            "mean_gain_gate_passed": mean_gate,
            "positive_seed_gate_passed": consistency_gate,
            "promoted_to_convolutional_medium": promoted,
            "rule": rule,
        },
        "resource_check_new_seeds_3_4": {
            "families": resource_check,
            "cost_identical_across_methods": cost_identical_across_methods,
            "common_cost": common_cost,
            "scope_note": (
                "Resource instrumentation is summarized on the newly trained "
                "matched seeds 3 and 4; historical seeds predate some cost "
                "fields."
            ),
        },
        "frozen_method_policy": {
            "v3_changed": False,
            "v4_changed": False,
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "seed",
            "random_percent",
            "frozen_v4_percent",
            "unified_candidate_percent",
            "candidate_minus_random_pp",
        ])
        for index, seed in enumerate(seeds):
            writer.writerow([
                seed,
                values["random"][index],
                values["frozen_v4"][index],
                values["unified_candidate"][index],
                (
                    values["unified_candidate"][index]
                    - values["random"][index]
                ),
            ])
    print(json.dumps(payload["promotion_decision"], indent=2))


if __name__ == "__main__":
    main()
