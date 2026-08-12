#!/usr/bin/env python3
"""Summarize the frozen three-seed U2 pilot against matched controls."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "summary" / "second_round_u2_pilot.json"
T_CRIT_DF2 = 4.302652729911275


def summary(name: str) -> dict:
    return json.loads((RESULTS / name / "run_summary.json").read_text())


def hard_accuracy(name: str) -> float:
    return 100 * summary(name)["best_validation_hard_accuracy"]


def prefix_hard_accuracy(name: str, maximum_step: int = 20_000) -> float:
    with (RESULTS / name / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return 100 * max(
        float(row["val_acc_discrete"])
        for row in rows
        if int(row["step"]) <= maximum_step
    )


def paired(candidate: list[float], baseline: list[float]) -> dict:
    differences = [new - old for new, old in zip(candidate, baseline)]
    mean = statistics.mean(differences)
    half_width = (
        T_CRIT_DF2 * statistics.stdev(differences) / math.sqrt(len(differences))
    )
    return {
        "baseline_mean_pct": statistics.mean(baseline),
        "candidate_mean_pct": statistics.mean(candidate),
        "per_seed_gain_pp": differences,
        "paired_mean_gain_pp": mean,
        "paired_95ci_pp": [mean - half_width, mean + half_width],
        "positive_pairs": sum(value > 0 for value in differences),
    }


def u2_resources(names: list[str]) -> dict:
    runs = [summary(name) for name in names]
    topology_seconds = [
        sum(layer["construction_seconds"] for layer in run["topology"])
        for run in runs
    ]
    costs = [run["cost"] for run in runs]
    if any(cost != costs[0] for cost in costs[1:]):
        raise RuntimeError("U2 cost changed across paired seeds")
    return {
        "mean_training_wall_minutes": statistics.mean(
            run["wall_seconds"] / 60 for run in runs
        ),
        "maximum_peak_gpu_gib": max(
            run["peak_gpu_memory_bytes"] / 2**30 for run in runs
        ),
        "mean_topology_construction_seconds": statistics.mean(topology_seconds),
        "cost": costs[0],
    }


def main() -> None:
    cells = {}

    for label, prefix, baseline_prefix, v3_prefix in [
        (
            "mnist_8k",
            "second_u2_mnist_8k_seed",
            "second_compression_mnist_8k_random_seed",
            "second_compression_mnist_8k_coverage_v3_seed",
        ),
        (
            "fashion_16k",
            "second_u2_fashion_16k_seed",
            "second_compression_fashion_16k_random_seed",
            "second_compression_fashion_16k_coverage_v3_seed",
        ),
    ]:
        names = [f"{prefix}{seed}" for seed in range(3)]
        candidate = [hard_accuracy(name) for name in names]
        random = [
            prefix_hard_accuracy(f"{baseline_prefix}{seed}")
            for seed in range(3)
        ]
        v3 = [
            prefix_hard_accuracy(f"{v3_prefix}{seed}")
            for seed in range(3)
        ]
        cells[label] = {
            "selection_effort": "best hardened validation through 20K",
            "u2_accuracy_pct": candidate,
            "comparisons": {
                "random": paired(candidate, random),
                "frozen_v3": paired(candidate, v3),
            },
            "resources": u2_resources(names),
        }

    names = [f"second_u2_cifar10_s_seed{seed}" for seed in range(3)]
    candidate = [hard_accuracy(name) for name in names]
    random = [
        hard_accuracy(f"pilot_cifar10_random_v3_seed{seed}")
        for seed in range(3)
    ]
    v3 = [
        hard_accuracy(f"pilot_cifar10_semantic_balanced_v3_seed{seed}")
        for seed in range(3)
    ]
    cells["dense_cifar10_s"] = {
        "selection_effort": "best hardened validation at 20K",
        "u2_accuracy_pct": candidate,
        "comparisons": {
            "random": paired(candidate, random),
            "frozen_v3": paired(candidate, v3),
        },
        "resources": u2_resources(names),
    }

    names = [f"second_u2_cifar100_3x128k_seed{seed}" for seed in range(3)]
    candidate = [hard_accuracy(name) for name in names]
    random_names = [
        "pilot_table4_cifar100_384k_depth3_random_seed0",
        "second_c100_3x128k_random_seed1",
        "second_c100_3x128k_random_seed2",
    ]
    v3_names = [
        "pilot_table4_cifar100_384k_depth3_coverage_v3_seed0",
        "second_c100_3x128k_coverage_v3_seed1",
        "second_c100_3x128k_coverage_v3_seed2",
    ]
    cells["dense_cifar100_3x128k"] = {
        "selection_effort": "best hardened validation at 20K",
        "u2_accuracy_pct": candidate,
        "comparisons": {
            "random": paired(
                candidate, [hard_accuracy(name) for name in random_names]
            ),
            "frozen_v3": paired(
                candidate, [hard_accuracy(name) for name in v3_names]
            ),
        },
        "resources": u2_resources(names),
    }

    names = [f"second_u2_conv_cifar10_s_seed{seed}" for seed in range(3)]
    candidate = [hard_accuracy(name) for name in names]
    comparison_prefixes = {
        "original_random": "pilot_conv_cifar10_paper_small_random_seed",
        "controlled_random": (
            "pilot_conv_cifar10_paper_small_random_controlled_seed"
        ),
        "frozen_v4": "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed",
        "unified_u1": "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed",
    }
    cells["conv_cifar10_s"] = {
        "selection_effort": "best hardened validation at 20K",
        "u2_accuracy_pct": candidate,
        "comparisons": {
            label: paired(
                candidate,
                [hard_accuracy(f"{prefix}{seed}") for seed in range(3)],
            )
            for label, prefix in comparison_prefixes.items()
        },
        "resources": u2_resources(names),
    }

    payload = {
        "method": "semantic_multiscale_balanced",
        "method_frozen_before_summary": True,
        "held_out_test_queried": False,
        "cells": cells,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
