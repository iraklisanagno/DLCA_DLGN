#!/usr/bin/env python3
"""Summarize frozen dense validation and one-time held-out test results."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FREEZE = ROOT / "summary" / "second_round_final_validation_freeze.json"
OUTPUT = ROOT / "summary" / "second_round_final_dense.json"
T_CRIT_DF2 = 4.302652729911275


CELLS = {
    "mnist_8k": {
        "random": "mnist_8k_random",
        "v3": "mnist_8k_v3",
        "u2": "mnist_8k_u2",
    },
    "fashion_16k": {
        "random": "fashion_16k_random",
        "v3": "fashion_16k_v3",
        "u2": "fashion_16k_u2",
    },
    "dense_cifar10_s": {
        "random": "dense_cifar10_s_random",
        "v3": "dense_cifar10_s_v3",
        "u2": "dense_cifar10_s_u2",
    },
    "dense_cifar100_3x128k": {
        "random": "dense_cifar100_3x128k_random",
        "v3": "dense_cifar100_3x128k_v3",
    },
}


def paired(candidate: list[float], baseline: list[float]) -> dict:
    differences = [new - old for new, old in zip(candidate, baseline)]
    mean = statistics.mean(differences)
    half_width = (
        T_CRIT_DF2 * statistics.stdev(differences) / math.sqrt(len(differences))
    )
    return {
        "per_seed_gain_pp": differences,
        "paired_mean_gain_pp": mean,
        "paired_95ci_pp": [mean - half_width, mean + half_width],
        "positive_pairs": sum(value > 0 for value in differences),
    }


def main() -> None:
    freeze = json.loads(FREEZE.read_text())
    output_cells = {}
    for cell, methods in CELLS.items():
        output_methods = {}
        for method, group in methods.items():
            rows = freeze["groups"][group]
            validation = [row["best_hard_validation_pct"] for row in rows]
            tests = []
            summaries = []
            for row in rows:
                run_dir = Path(row["run_dir"])
                test_path = run_dir / "test_metrics.json"
                if not test_path.is_file():
                    raise RuntimeError(f"missing held-out result: {test_path}")
                tests.append(
                    100 * json.loads(test_path.read_text())["test_hard_accuracy"]
                )
                summaries.append(
                    json.loads((run_dir / "run_summary.json").read_text())
                )
            # Early CIFAR-10 S result files predate cost accounting in
            # run_summary.json.  The architecture is identical across the
            # matched cell, so inherit the cost from a newer method below.
            costs = [summary.get("cost") for summary in summaries]
            measured_costs = [cost for cost in costs if cost is not None]
            if any(cost != measured_costs[0] for cost in measured_costs[1:]):
                raise RuntimeError(f"cost changed across {cell}/{method}")
            topology = [
                sum(
                    layer["construction_seconds"]
                    for layer in summary["topology"]
                )
                for summary in summaries
            ]
            seed0_benchmark = json.loads(
                (
                    Path(rows[0]["run_dir"])
                    / "synthetic_inference_benchmark_v2.json"
                ).read_text()
            )
            if seed0_benchmark["benchmark_implementation_version"] != 2:
                raise RuntimeError(f"obsolete GPU benchmark: {cell}/{method}")
            if seed0_benchmark["heldout_test_accessed"]:
                raise RuntimeError(
                    f"synthetic benchmark accessed held-out data: {cell}/{method}"
                )
            output_methods[method] = {
                "validation_hard_pct": validation,
                "validation_hard_mean_pct": statistics.mean(validation),
                "validation_hard_std_pct": statistics.stdev(validation),
                "test_hard_pct": tests,
                "test_hard_mean_pct": statistics.mean(tests),
                "test_hard_std_pct": statistics.stdev(tests),
                "mean_training_wall_minutes": statistics.mean(
                    summary["wall_seconds"] / 60 for summary in summaries
                ),
                "maximum_peak_gpu_gib": max(
                    summary["peak_gpu_memory_bytes"] / 2**30
                    for summary in summaries
                ),
                "mean_topology_construction_seconds": statistics.mean(topology),
                "representative_seed0_synthetic_gpu": {
                    "milliseconds_per_batch128": seed0_benchmark[
                        "milliseconds_per_batch"
                    ],
                    "examples_per_second": seed0_benchmark[
                        "examples_per_second"
                    ],
                    "peak_device_memory_gib": seed0_benchmark[
                        "peak_device_memory_bytes"
                    ] / 2**30,
                    "input_source": seed0_benchmark["input_source"],
                },
                "cost": measured_costs[0] if measured_costs else None,
            }
        cell_costs = [
            row["cost"] for row in output_methods.values() if row["cost"] is not None
        ]
        if not cell_costs:
            raise RuntimeError(f"no cost accounting available for {cell}")
        if any(cost != cell_costs[0] for cost in cell_costs[1:]):
            raise RuntimeError(f"cost changed across methods in {cell}")
        for row in output_methods.values():
            if row["cost"] is None:
                row["cost"] = cell_costs[0]
                row["cost_inherited_from_matched_architecture"] = True
        baseline = output_methods["random"]
        for method, row in output_methods.items():
            if method == "random":
                continue
            row["validation_vs_random"] = paired(
                row["validation_hard_pct"], baseline["validation_hard_pct"]
            )
            row["test_vs_random"] = paired(
                row["test_hard_pct"], baseline["test_hard_pct"]
            )
        if "u2" in output_methods and "v3" in output_methods:
            output_methods["u2"]["validation_vs_v3"] = paired(
                output_methods["u2"]["validation_hard_pct"],
                output_methods["v3"]["validation_hard_pct"],
            )
            output_methods["u2"]["test_vs_v3"] = paired(
                output_methods["u2"]["test_hard_pct"],
                output_methods["v3"]["test_hard_pct"],
            )
        output_cells[cell] = output_methods
    payload = {
        "validation_freeze": str(FREEZE),
        "held_out_test_evaluated_once": True,
        "cells": output_cells,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
