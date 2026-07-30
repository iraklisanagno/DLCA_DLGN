#!/usr/bin/env python3
"""Summarize the frozen V4 ablation and channel-spatial validation pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary"
SEEDS = (0, 1, 2)
T95_DF2 = 4.302652729749


def best_validation(pattern: str) -> list[float]:
    values = []
    for seed in SEEDS:
        path = RESULTS / pattern.format(seed=seed) / "run_summary.json"
        payload = json.loads(path.read_text())
        values.append(100.0 * payload["best_validation_hard_accuracy"])
    return values


def paired(left: list[float], right: list[float]) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    center = mean(differences)
    spread = stdev(differences)
    half_width = T95_DF2 * spread / math.sqrt(len(differences))
    return {
        "paired_differences_percentage_points": differences,
        "paired_mean_percentage_points": center,
        "paired_std_percentage_points": spread,
        "paired_t95_ci_percentage_points": [
            center - half_width,
            center + half_width,
        ],
    }


def summarize_components() -> dict:
    random = best_validation(
        "pilot_conv_cifar10_paper_small_random_seed{seed}"
    )
    balanced = best_validation(
        "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed{seed}"
    )
    v4 = best_validation(
        "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}"
    )
    return {
        "phase": "cifar10_conv_small_v4_components",
        "dataset": "cifar-10",
        "architecture": "ClgnCifar10PaperSmall",
        "metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "controls_reused_not_rerun": True,
        "per_seed_percent": {
            "random": random,
            "balanced_channel_no_swaps": balanced,
            "frozen_v4": v4,
        },
        "mean_percent": {
            "random": mean(random),
            "balanced_channel_no_swaps": mean(balanced),
            "frozen_v4": mean(v4),
        },
        "paired_effects": {
            "balanced_channel_minus_random": paired(balanced, random),
            "frozen_v4_minus_balanced_channel": paired(v4, balanced),
            "frozen_v4_minus_random": paired(v4, random),
        },
        "frozen_method_policy": {
            "v3_changed": False,
            "v4_changed": False,
        },
    }


def summarize_revision() -> dict:
    random = best_validation(
        "pilot_conv_cifar10_paper_small_random_seed{seed}"
    )
    v4 = best_validation(
        "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}"
    )
    revised = best_validation(
        "pilot_conv_cifar10_small_channel_spatial_seed{seed}"
    )
    over_random = paired(revised, random)
    over_v4 = paired(revised, v4)
    passes_random = over_random["paired_mean_percentage_points"] >= 2.0
    passes_v4 = over_v4["paired_mean_percentage_points"] >= 1.0
    return {
        "phase": "cifar10_conv_small_channel_spatial",
        "dataset": "cifar-10",
        "architecture": "ClgnCifar10PaperSmall",
        "metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "controls_reused_not_rerun": True,
        "per_seed_percent": {
            "random": random,
            "frozen_v4": v4,
            "channel_spatial": revised,
        },
        "mean_percent": {
            "random": mean(random),
            "frozen_v4": mean(v4),
            "channel_spatial": mean(revised),
        },
        "paired_effects": {
            "channel_spatial_minus_random": over_random,
            "channel_spatial_minus_frozen_v4": over_v4,
        },
        "promotion_gate": {
            "required_gain_over_random_percentage_points": 2.0,
            "required_gain_over_frozen_v4_percentage_points": 1.0,
            "passes_random_gate": passes_random,
            "passes_v4_gate": passes_v4,
            "promote_to_medium": passes_random and passes_v4,
        },
        "cost": {
            "additional_trainable_parameters": 0,
            "additional_deployed_gates": 0,
            "additional_deployed_routing_entries": 0,
        },
        "frozen_method_policy": {
            "v3_changed": False,
            "v4_changed": False,
        },
    }


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    components = summarize_components()
    revision = summarize_revision()
    component_path = SUMMARY / "cifar10_conv_small_v4_components.json"
    revision_path = SUMMARY / "cifar10_conv_small_channel_spatial.json"
    component_path.write_text(json.dumps(components, indent=2) + "\n")
    revision_path.write_text(json.dumps(revision, indent=2) + "\n")
    print(component_path)
    print(revision_path)
    print(json.dumps(revision["promotion_gate"], indent=2))


if __name__ == "__main__":
    main()
