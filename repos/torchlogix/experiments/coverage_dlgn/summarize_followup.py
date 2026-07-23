#!/usr/bin/env python3
"""Summarize the medium, controlled-depth, and convolutional follow-up studies."""

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary"
REQUIRED = (
    "run_summary.json",
    "best_checkpoint.pt",
    "final_checkpoint.pt",
    "metrics.csv",
    "thresholds.csv",
    "topology.csv",
    "topology.json",
    "training_config.json",
    "environment.json",
)
ROUTING_FIELDS = {
    "config",
    "connections_init_method",
    "coverage_candidate_pool_size",
    "coverage_long_range_fraction",
    "coverage_swap_fraction",
    "coverage_alpha",
    "coverage_beta",
    "coverage_gamma",
    "coverage_delta",
    "coverage_local_radius",
    "coverage_hybrid_base",
    "coverage_novelty_weight",
    "output",
}
T_CRITICAL_95 = {3: 4.302652729911275, 5: 2.7764451051977987}


def load_json(path):
    return json.loads(path.read_text())


def load_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def depth_run(budget, depth, variant, seed):
    semantic = variant == "v3"
    if depth == 4 and budget == "48k":
        stem = "semantic_balanced" if semantic else "random"
        return f"pilot_cifar10_{stem}_v3_seed{seed}"
    if depth == 4:
        stem = "semantic_balanced" if semantic else "random"
        return f"pilot_cifar10_medium_{stem}_v3_seed{seed}"
    stem = "semantic_balanced" if semantic else "random"
    return f"depth_cifar10_budget{budget}_depth{depth}_{stem}_v3_seed{seed}"


def collect_run(study, run, variant, seed, budget=None, depth=None):
    run_dir = RESULTS / run
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{run}: missing {missing}")
    metrics = load_csv(run_dir / "metrics.csv")
    selected = max(metrics, key=lambda row: float(row["val_acc_discrete"]))
    config = load_json(run_dir / "training_config.json")
    environment = load_json(run_dir / "environment.json")
    summary = load_json(run_dir / "run_summary.json")
    test_path = run_dir / "test_metrics.json"
    benchmark_path = run_dir / "inference_benchmark.json"
    test = load_json(test_path) if test_path.exists() else {}
    benchmark = load_json(benchmark_path) if benchmark_path.exists() else {}
    topology = load_csv(run_dir / "topology.csv")
    return {
        "study": study,
        "run": run,
        "variant": variant,
        "seed": seed,
        "budget": budget,
        "depth": depth,
        "architecture": config["architecture"],
        "iterations": config["num_iterations"],
        "validation_hard_accuracy": float(selected["val_acc_discrete"]),
        "validation_selection_step": int(selected["step"]),
        "test_hard_accuracy": test.get("test_hard_accuracy"),
        "inference_milliseconds_per_batch": benchmark.get(
            "milliseconds_per_batch"
        ),
        "training_wall_seconds": summary["wall_seconds"],
        "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
        "dense_gate_count": sum(int(row["out_dim"]) for row in topology),
        "training_implementation_sha256": environment.get(
            "training_implementation_sha256"
        ),
        "_config": config,
        "_run_dir": run_dir,
    }


def paired_summary(rows, metric):
    variants = sorted({row["variant"] for row in rows})
    if len(variants) != 2:
        raise ValueError(f"expected two variants, found {variants}")
    baseline = "random"
    proposed = next(item for item in variants if item != baseline)
    by_variant = {
        variant: sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: row["seed"],
        )
        for variant in variants
    }
    n = len(by_variant[baseline])
    if n != len(by_variant[proposed]) or n not in T_CRITICAL_95:
        raise ValueError("paired cohort must contain three or five complete seeds")
    for left, right in zip(by_variant[baseline], by_variant[proposed]):
        if left["seed"] != right["seed"]:
            raise ValueError("seed pairing mismatch")
        left_config = {
            key: value
            for key, value in left["_config"].items()
            if key not in ROUTING_FIELDS
        }
        right_config = {
            key: value
            for key, value in right["_config"].items()
            if key not in ROUTING_FIELDS
        }
        if left_config != right_config:
            raise ValueError(f"protocol mismatch for seed {left['seed']}")
    baseline_values = [row[metric] for row in by_variant[baseline]]
    proposed_values = [row[metric] for row in by_variant[proposed]]
    if any(value is None for value in baseline_values + proposed_values):
        raise ValueError(f"missing paired metric {metric}")
    differences = [
        100 * (proposed_value - baseline_value)
        for baseline_value, proposed_value in zip(
            baseline_values, proposed_values
        )
    ]
    mean = statistics.mean(differences)
    std = statistics.stdev(differences)
    half_width = T_CRITICAL_95[n] * std / math.sqrt(n)
    return {
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "n": n,
        "baseline_mean_percent": 100 * statistics.mean(baseline_values),
        "proposed_mean_percent": 100 * statistics.mean(proposed_values),
        "paired_differences_percentage_points": differences,
        "paired_mean_percentage_points": mean,
        "paired_std_percentage_points": std,
        "paired_t95_ci_percentage_points": [
            mean - half_width,
            mean + half_width,
        ],
    }


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(5):
        rows.append(collect_run(
            "medium_full",
            f"paper_cifar10_medium_random_seed{seed}",
            "random",
            seed,
            "512k",
            4,
        ))
        rows.append(collect_run(
            "medium_full",
            f"paper_cifar10_medium_semantic_balanced_v3_seed{seed}",
            "v3",
            seed,
            "512k",
            4,
        ))
    for budget in ("48k", "512k"):
        for depth in (4, 8, 12):
            for seed in range(3):
                for variant in ("random", "v3"):
                    rows.append(collect_run(
                        "depth",
                        depth_run(budget, depth, variant, seed),
                        variant,
                        seed,
                        budget,
                        depth,
                    ))
    for seed in range(3):
        rows.append(collect_run(
            "conv_pilot",
            f"pilot_conv_cifar10_small_random_seed{seed}",
            "random",
            seed,
        ))
        rows.append(collect_run(
            "conv_pilot",
            f"pilot_conv_cifar10_small_semantic_channel_v4_seed{seed}",
            "v4",
            seed,
        ))

    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with (SUMMARY / "followup_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)

    medium = [row for row in rows if row["study"] == "medium_full"]
    depth_summaries = []
    for budget in ("48k", "512k"):
        for depth in (4, 8, 12):
            group = [
                row for row in rows
                if row["study"] == "depth"
                and row["budget"] == budget
                and row["depth"] == depth
            ]
            depth_summaries.append({
                "budget": budget,
                "depth": depth,
                **paired_summary(group, "validation_hard_accuracy"),
            })
    conv = [row for row in rows if row["study"] == "conv_pilot"]
    for seed in range(3):
        random_conv = next(
            row for row in conv
            if row["seed"] == seed and row["variant"] == "random"
        )
        semantic_conv = next(
            row for row in conv
            if row["seed"] == seed and row["variant"] == "v4"
        )
        random_layers = load_json(
            random_conv["_run_dir"] / "conv_topology.json"
        )["layers"]
        semantic_layers = load_json(
            semantic_conv["_run_dir"] / "conv_topology.json"
        )["layers"]
        if any(
            left["spatial_coordinates_sha256"]
            != right["spatial_coordinates_sha256"]
            for left, right in zip(random_layers, semantic_layers)
        ):
            raise ValueError(f"conv spatial coordinates changed for seed {seed}")

    payload = {
        "audit": {
            "status": "pass",
            "complete_runs": len(rows),
            "paired_protocols_equal_except_routing": True,
            "conv_spatial_coordinates_identical": True,
        },
        "medium_full_validation": paired_summary(
            medium, "validation_hard_accuracy"
        ),
        "medium_full_test": paired_summary(medium, "test_hard_accuracy"),
        "depth_validation": depth_summaries,
        "conv_pilot_validation": paired_summary(
            conv, "validation_hard_accuracy"
        ),
        "conv_pilot_test": paired_summary(conv, "test_hard_accuracy"),
        "conv_inference_milliseconds_per_batch": {
            variant: statistics.mean(
                row["inference_milliseconds_per_batch"]
                for row in conv if row["variant"] == variant
            )
            for variant in ("random", "v4")
        },
    }
    (SUMMARY / "followup_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
