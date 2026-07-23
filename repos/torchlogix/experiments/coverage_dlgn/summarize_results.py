#!/usr/bin/env python3
"""Regenerate CoverageDLGN tables, paired reports, audits, and figures."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


REQUIRED_RUN_FILES = (
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

PAPER_FASHION_RUNS = (
    *(f"paper_fashion_mnist_small_random_seed{seed}" for seed in range(5)),
    *(f"paper_fashion_mnist_small_hybrid_v2_seed{seed}" for seed in range(3)),
    "paper_fashion_mnist_small_hybrid_f000_v2_seed0",
    "paper_fashion_mnist_small_hybrid_f050_v2_seed0",
    "paper_fashion_mnist_small_hybrid_f075_v2_seed0",
    "paper_fashion_mnist_small_hybrid_f100_v2_seed0",
)
PAPER_CIFAR_RUNS = (
    *(f"paper_cifar10_small_random_seed{seed}" for seed in range(5)),
    *(f"paper_cifar10_small_hybrid_v2_seed{seed}" for seed in range(3)),
)
PAPER_V3_RUNS = (
    *(f"paper_fashion_mnist_small_semantic_balanced_v3_seed{seed}"
      for seed in range(5)),
    *(f"paper_cifar10_small_semantic_balanced_v3_seed{seed}"
      for seed in range(5)),
)
EXPECTED_PAPER_RUNS = (*PAPER_FASHION_RUNS, *PAPER_CIFAR_RUNS, *PAPER_V3_RUNS)
CENTRAL_PAPER_RUNS = (
    *(f"paper_fashion_mnist_small_random_seed{seed}" for seed in range(5)),
    *(f"paper_fashion_mnist_small_hybrid_v2_seed{seed}" for seed in range(3)),
    *PAPER_CIFAR_RUNS,
    *PAPER_V3_RUNS,
)

# Computed before reporting-code edits and preserved by the adjacent source archive.
PRE_ANALYSIS_STABLE_SOURCE_SHA256 = (
    "ddb1cfb4da65b0d41573a1a8796a7d8c772efe4ec385e575eb93b00b9e19b65d"
)
PRE_ANALYSIS_ARCHIVE_SHA256 = (
    "84dd8a2ac53ee14fba59c582606eb25a29ecb801b37d89dc27f8f58f07053c47"
)


def _read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _load_json(path):
    return json.loads(path.read_text())


def collect(results_dir):
    """Collect only complete runs; partial and failed runs are never summarized."""
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        required = [run_dir / name for name in REQUIRED_RUN_FILES]
        if not all(path.exists() for path in required):
            continue
        config = _load_json(run_dir / "training_config.json")
        environment = _load_json(run_dir / "environment.json")
        run_summary = _load_json(run_dir / "run_summary.json")
        test_path = run_dir / "test_metrics.json"
        test_metrics = _load_json(test_path) if test_path.exists() else None
        benchmark_path = run_dir / "inference_benchmark.json"
        benchmark = _load_json(benchmark_path) if benchmark_path.exists() else None
        metric_rows = _read_csv(run_dir / "metrics.csv")
        if not metric_rows:
            continue
        # Model selection is explicitly based on hardened validation accuracy.
        metrics = max(metric_rows, key=lambda row: float(row["val_acc_discrete"]))
        hard = float(metrics["val_acc_discrete"])
        reported_best = float(run_summary["best_validation_hard_accuracy"])
        if not math.isclose(hard, reported_best, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"{run_dir.name}: metric best {hard} != summary best {reported_best}"
            )
        topology_layers = _read_csv(run_dir / "topology.csv")
        semantic_path = run_dir / "semantic_topology.csv"
        semantic_layers = (
            _read_csv(semantic_path) if semantic_path.exists() else topology_layers
        )
        topology = topology_layers[-1]
        semantic_topology = semantic_layers[-1]
        semantic_first = semantic_layers[0]
        soft = float(metrics["val_acc_relaxed"])
        gate_count = sum(int(item["out_dim"]) for item in topology_layers)
        gate_function_bits = 1 << config["lut_rank"]
        lut_function_count = 1 << gate_function_bits
        deployed_index_bits = sum(
            int(item["deployed_index_bits"]) for item in topology_layers
        )
        rows.append({
            "run": run_dir.name,
            "variant": (
                "v3" if "_v3_" in run_dir.name
                else "v2" if "_v2_" in run_dir.name
                else "v1"
            ),
            "architecture": config["architecture"],
            "dataset": config["dataset"],
            "strategy": config["connections_init_method"],
            "seed": int(config["seed"]),
            "data_split_seed": config.get("data_split_seed"),
            "topology_seed": config.get("topology_seed"),
            "candidate_pool_size": config.get("coverage_candidate_pool_size"),
            "long_range_fraction": config.get("coverage_long_range_fraction"),
            "gamma": config.get("coverage_gamma"),
            "hard_accuracy": hard,
            "soft_accuracy": soft,
            "soft_hard_gap": soft - hard,
            "selection_step": int(metrics["step"]),
            "evaluation_count": len(metric_rows),
            "test_hard_accuracy": (
                test_metrics["test_hard_accuracy"] if test_metrics else None
            ),
            "test_soft_accuracy": (
                test_metrics["test_relaxed_accuracy"] if test_metrics else None
            ),
            "test_soft_hard_gap": (
                test_metrics["test_relaxed_accuracy"]
                - test_metrics["test_hard_accuracy"]
                if test_metrics else None
            ),
            "gate_count": gate_count,
            "trainable_gate_logits": gate_count * lut_function_count,
            "deployed_gate_function_bits": gate_count * gate_function_bits,
            "final_input_coverage": float(topology["input_coverage"]),
            "final_mean_gate_ancestry": float(topology["mean_gate_ancestry"]),
            "final_overlap_mean": float(topology["overlap_mean"]),
            "final_fanout_cv": float(topology["fanout_cv"]),
            "final_fanout_max": int(float(topology["fanout_max"])),
            "final_unused_outputs": int(float(topology["unused_outputs"])),
            "first_same_source_pair_fraction": (
                float(semantic_first["same_source_pair_fraction"])
                if "same_source_pair_fraction" in semantic_first else None
            ),
            "first_same_threshold_pair_fraction": (
                float(semantic_first["same_threshold_pair_fraction"])
                if "same_threshold_pair_fraction" in semantic_first else None
            ),
            "first_same_channel_pair_fraction": (
                float(semantic_first["same_channel_pair_fraction"])
                if "same_channel_pair_fraction" in semantic_first else None
            ),
            "first_spatial_manhattan_p50": (
                float(semantic_first["spatial_manhattan_p50"])
                if "spatial_manhattan_p50" in semantic_first else None
            ),
            "final_mean_source_ancestry": (
                float(semantic_topology["mean_source_ancestry"])
                if "mean_source_ancestry" in semantic_topology else None
            ),
            "final_source_cross_gate_jaccard": (
                float(semantic_topology["source_cross_gate_jaccard_mean"])
                if "source_cross_gate_jaccard_mean" in semantic_topology else None
            ),
            "final_source_group_coverage_min": (
                float(semantic_topology["source_group_coverage_min"])
                if "source_group_coverage_min" in semantic_topology else None
            ),
            "final_source_group_usage_cv_mean": (
                float(semantic_topology["source_group_usage_cv_mean"])
                if "source_group_usage_cv_mean" in semantic_topology else None
            ),
            "distinct_pairs": int(topology["distinct_predecessor_pairs"]),
            "deployed_index_bits": deployed_index_bits,
            "deployed_total_circuit_bits": (
                gate_count * gate_function_bits + deployed_index_bits
            ),
            "indices_tensor_bytes": sum(
                int(item["indices_tensor_bytes"]) for item in topology_layers
            ),
            "topology_construction_seconds": sum(
                float(item["construction_seconds"]) for item in topology_layers
            ),
            "topology_peak_temporary_bytes": max(
                int(item["generator_temporary_bytes"]) for item in topology_layers
            ),
            "training_wall_seconds": float(run_summary["wall_seconds"]),
            "peak_gpu_memory_bytes": int(run_summary["peak_gpu_memory_bytes"]),
            "inference_milliseconds_per_batch": (
                benchmark["milliseconds_per_batch"] if benchmark else None
            ),
            "inference_examples_per_second": (
                benchmark["examples_per_second"] if benchmark else None
            ),
            "source_revision": environment["source_revision"],
            "recorded_source_tree_sha256": environment["source_tree_sha256"],
            "python": environment["python"],
            "torch": environment["torch"],
            "cuda_build": environment["cuda_build"],
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    fields = (
        "dataset",
        "architecture",
        "strategy",
        "variant",
        "candidate_pool_size",
        "long_range_fraction",
        "gamma",
    )
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        values = [row["hard_accuracy"] for row in group]
        test_values = [
            row["test_hard_accuracy"]
            for row in group
            if row["test_hard_accuracy"] is not None
        ]
        inference_values = [
            row["inference_milliseconds_per_batch"]
            for row in group
            if row["inference_milliseconds_per_batch"] is not None
        ]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append({
            **dict(zip(fields, key)),
            "n": len(values),
            "hard_accuracy_mean": statistics.mean(values),
            "hard_accuracy_std": std,
            "hard_accuracy_ci95_normal_half_width": (
                1.96 * std / math.sqrt(len(values))
            ),
            "soft_hard_gap_mean": statistics.mean(
                row["soft_hard_gap"] for row in group
            ),
            "test_n": len(test_values),
            "test_hard_accuracy_mean": (
                statistics.mean(test_values) if test_values else None
            ),
            "test_hard_accuracy_std": (
                statistics.stdev(test_values) if len(test_values) > 1 else 0.0
            ),
            "mean_gate_ancestry": statistics.mean(
                row["final_mean_gate_ancestry"] for row in group
            ),
            "overlap_mean": statistics.mean(
                row["final_overlap_mean"] for row in group
            ),
            "fanout_cv_mean": statistics.mean(
                row["final_fanout_cv"] for row in group
            ),
            "mean_source_ancestry": (
                statistics.mean(
                    row["final_mean_source_ancestry"]
                    for row in group
                    if row["final_mean_source_ancestry"] is not None
                )
                if any(
                    row["final_mean_source_ancestry"] is not None
                    for row in group
                ) else None
            ),
            "same_source_pair_fraction_mean": (
                statistics.mean(
                    row["first_same_source_pair_fraction"]
                    for row in group
                    if row["first_same_source_pair_fraction"] is not None
                )
                if any(
                    row["first_same_source_pair_fraction"] is not None
                    for row in group
                ) else None
            ),
            "training_wall_seconds_mean": statistics.mean(
                row["training_wall_seconds"] for row in group
            ),
            "peak_gpu_memory_bytes_mean": statistics.mean(
                row["peak_gpu_memory_bytes"] for row in group
            ),
            "topology_construction_seconds_mean": statistics.mean(
                row["topology_construction_seconds"] for row in group
            ),
            "topology_peak_temporary_bytes_mean": statistics.mean(
                row["topology_peak_temporary_bytes"] for row in group
            ),
            "inference_n": len(inference_values),
            "inference_milliseconds_per_batch_mean": (
                statistics.mean(inference_values) if inference_values else None
            ),
            "gate_count": group[0]["gate_count"],
            "trainable_gate_logits": group[0]["trainable_gate_logits"],
            "deployed_gate_function_bits": group[0][
                "deployed_gate_function_bits"
            ],
            "deployed_index_bits": group[0]["deployed_index_bits"],
            "deployed_total_circuit_bits": group[0][
                "deployed_total_circuit_bits"
            ],
        })
    return output


def paired_comparison(
    rows, random_pattern, hybrid_pattern, note, metric="hard_accuracy"
):
    random_rows = {
        int(row["seed"]): row
        for row in rows
        if re.fullmatch(random_pattern, row["run"])
    }
    hybrid_rows = {
        int(row["seed"]): row
        for row in rows
        if re.fullmatch(hybrid_pattern, row["run"])
    }
    seeds = sorted(
        seed for seed in set(random_rows) & set(hybrid_rows)
        if random_rows[seed][metric] is not None
        and hybrid_rows[seed][metric] is not None
    )
    if not seeds:
        return {"seeds": [], "metric": metric, "note": "No paired runs found"}
    pairs = [
        {
            "seed": seed,
            "random": random_rows[seed][metric],
            "hybrid": hybrid_rows[seed][metric],
            "difference": hybrid_rows[seed][metric] - random_rows[seed][metric],
        }
        for seed in seeds
    ]
    differences = [pair["difference"] for pair in pairs]
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    standard_error = std / math.sqrt(len(differences))
    # Exact two-sided 95% Student-t critical values for the small samples used.
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(differences), 1.96
    )
    half_width = t_critical * standard_error
    mean_difference = statistics.mean(differences)
    report = {
        "seeds": seeds,
        "metric": metric,
        "pairs": pairs,
        "random_mean": statistics.mean(pair["random"] for pair in pairs),
        "hybrid_mean": statistics.mean(pair["hybrid"] for pair in pairs),
        "mean_difference": mean_difference,
        "std_difference": std,
        "student_t_ci95_half_width": half_width,
        "student_t_ci95": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
        "note": note,
    }
    if metric in {
        "hard_accuracy",
        "soft_accuracy",
        "test_hard_accuracy",
        "test_soft_accuracy",
    }:
        report.update({
            "mean_difference_percentage_points": 100 * mean_difference,
            "student_t_ci95_percentage_points": [
                100 * (mean_difference - half_width),
                100 * (mean_difference + half_width),
            ],
            "meets_positive_0_3pp_mean_threshold": mean_difference >= 0.003,
        })
    return report


def central_comparison(rows, dataset):
    if dataset == "fashion-mnist":
        random = r"paper_fashion_mnist_small_random_seed\d+"
        hybrid = r"paper_fashion_mnist_small_hybrid_v2_seed\d+"
    elif dataset == "cifar-10":
        random = r"paper_cifar10_small_random_seed\d+"
        hybrid = r"paper_cifar10_small_hybrid_v2_seed\d+"
    else:
        raise ValueError(dataset)
    note = (
        "Three paired seeds with identical split, architecture, gate budget, "
        "optimizer, batch size, and training steps. Expand to five seeds before "
        "a final DATE statistical claim."
    )
    metrics = (
        "hard_accuracy",
        "soft_accuracy",
        "test_hard_accuracy",
        "test_soft_accuracy",
        "final_mean_gate_ancestry",
        "final_overlap_mean",
        "final_fanout_cv",
        "training_wall_seconds",
        "peak_gpu_memory_bytes",
        "topology_construction_seconds",
        "topology_peak_temporary_bytes",
        "inference_milliseconds_per_batch",
    )
    return {
        metric: paired_comparison(rows, random, hybrid, note, metric)
        for metric in metrics
    }


def semantic_v3_comparison(rows, dataset):
    if dataset == "fashion-mnist":
        random = r"paper_fashion_mnist_small_random_seed\d+"
        semantic = (
            r"paper_fashion_mnist_small_semantic_balanced_v3_seed\d+"
        )
    elif dataset == "cifar-10":
        random = r"paper_cifar10_small_random_seed\d+"
        semantic = r"paper_cifar10_small_semantic_balanced_v3_seed\d+"
    else:
        raise ValueError(dataset)
    note = (
        "Five paired full-budget seeds. The semantic topology and its "
        "candidate pool, swap fraction, and novelty weight were selected on "
        "CIFAR-10 validation and then applied to Fashion-MNIST without retuning."
    )
    metrics = (
        "hard_accuracy",
        "soft_accuracy",
        "test_hard_accuracy",
        "test_soft_accuracy",
        "final_mean_gate_ancestry",
        "final_mean_source_ancestry",
        "final_overlap_mean",
        "final_fanout_cv",
        "training_wall_seconds",
        "peak_gpu_memory_bytes",
        "topology_construction_seconds",
        "topology_peak_temporary_bytes",
        "inference_milliseconds_per_batch",
    )
    return {
        metric: paired_comparison(rows, random, semantic, note, metric)
        for metric in metrics
    }


def _central_curve_run(run):
    return bool(re.fullmatch(
        r"paper_(fashion_mnist|cifar10)_small_"
        r"(random|hybrid_v2|semantic_balanced_v3)_seed\d+",
        run,
    ))


def learning_curve_rows(results_dir):
    grouped = {}
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or not _central_curve_run(run_dir.name):
            continue
        if not (run_dir / "run_summary.json").exists():
            continue
        config = _load_json(run_dir / "training_config.json")
        strategy = (
            "hybrid-v2" if config["connections_init_method"] == "coverage_hybrid"
            else "semantic-balanced-v3"
            if config["connections_init_method"] == "semantic_balanced_hybrid"
            else "random"
        )
        for row in _read_csv(run_dir / "metrics.csv"):
            key = (config["dataset"], strategy, int(row["step"]))
            grouped.setdefault(key, []).append(row)
    output = []
    for (dataset, strategy, step), group in sorted(grouped.items()):
        hard = [float(row["val_acc_discrete"]) for row in group]
        soft = [float(row["val_acc_relaxed"]) for row in group]
        output.append({
            "dataset": dataset,
            "strategy": strategy,
            "step": step,
            "n": len(group),
            "hard_accuracy_mean": statistics.mean(hard),
            "hard_accuracy_std": statistics.stdev(hard) if len(hard) > 1 else 0.0,
            "soft_accuracy_mean": statistics.mean(soft),
            "soft_accuracy_std": statistics.stdev(soft) if len(soft) > 1 else 0.0,
        })
    return output


def write_learning_curve_svg(path, rows, dataset):
    rows = [row for row in rows if row["dataset"] == dataset]
    if not rows:
        return
    width, height = 760, 480
    left, right, top, bottom = 72, 24, 44, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max = max(row["step"] for row in rows)
    values = [100 * row["hard_accuracy_mean"] for row in rows]
    y_min = math.floor(min(values) - 1)
    y_max = math.ceil(max(values) + 1)
    y_span = max(1.0, y_max - y_min)

    def point(row):
        x = left + row["step"] / x_max * plot_width
        y = top + (y_max - 100 * row["hard_accuracy_mean"]) / y_span * plot_height
        return x, y

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16">'
        f'{dataset} hardened validation learning curves (mean by strategy)</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        f'stroke="black"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle">'
        "Training step</text>",
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 '
        f'{height/2})" text-anchor="middle">Hard validation accuracy (%)</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        step = int(round(x_max * fraction))
        x = left + fraction * plot_width
        elements.extend([
            f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" '
            f'y2="{height-bottom+5}" stroke="black"/>',
            f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" '
            f'font-size="10">{step}</text>',
        ])
    for index in range(5):
        value = y_min + index * y_span / 4
        y = top + (y_max - value) / y_span * plot_height
        elements.extend([
            f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            f'stroke="black"/>',
            f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="10">{value:.1f}</text>',
        ])
    colors = {
        "random": "#1f77b4",
        "hybrid-v2": "#d62728",
        "semantic-balanced-v3": "#2ca02c",
    }
    strategies = [
        strategy
        for strategy in ("random", "hybrid-v2", "semantic-balanced-v3")
        if any(row["strategy"] == strategy for row in rows)
    ]
    for strategy in strategies:
        series = sorted(
            (row for row in rows if row["strategy"] == strategy),
            key=lambda row: row["step"],
        )
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(point, series))
        elements.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{colors[strategy]}" stroke-width="2"/>'
        )
    for index, strategy in enumerate(strategies):
        y = top + 18 * index
        elements.extend([
            f'<line x1="{width-150}" y1="{y}" x2="{width-125}" y2="{y}" '
            f'stroke="{colors[strategy]}" stroke-width="3"/>',
            f'<text x="{width-120}" y="{y+4}" font-size="11">{strategy}</text>',
        ])
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def write_coverage_svg(path, rows):
    if not rows:
        return
    width, height, margin = 760, 480, 60
    xs = [row["final_mean_gate_ancestry"] for row in rows]
    ys = [row["hard_accuracy"] * 100 for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    colors = {
        "random": "#1f77b4",
        "random_unique": "#9467bd",
        "butterfly": "#ff7f0e",
        "local_cyclic": "#8c564b",
        "coverage_greedy": "#2ca02c",
        "coverage_hybrid": "#d62728",
        "semantic_balanced_hybrid": "#17becf",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle">'
        "Final mean original-input ancestry per gate</text>",
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" '
        f'text-anchor="middle">Hard validation accuracy (%)</text>',
    ]
    for row, x, y in zip(rows, xs, ys):
        px = margin + (x - x_min) / x_span * (width - 2 * margin)
        py = height - margin - (y - y_min) / y_span * (height - 2 * margin)
        color = colors.get(row["strategy"], "#333333")
        elements.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{px+7:.1f}" y="{py-7:.1f}" '
            f'font-size="10">{row["run"]}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def _protocol_subset(config):
    keys = (
        "dataset",
        "architecture",
        "data_split_seed",
        "batch_size",
        "num_iterations",
        "eval_freq",
        "valid_set_size",
        "augmentation",
        "learning_rate",
        "lr_schedule",
        "weight_decay",
        "lut_rank",
        "parametrization",
        "parametrization_temperature",
        "forward_sampling",
        "weight_init",
        "binarization_num_batches",
        "binarization",
        "binarization_init",
        "binarization_per",
    )
    return {key: config.get(key) for key in keys}


def reproducibility_audit(results_dir, rows):
    rows_by_name = {row["run"]: row for row in rows}
    run_audits = {}
    for run in EXPECTED_PAPER_RUNS:
        run_dir = results_dir / run
        missing = [
            name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()
        ]
        run_audits[run] = {
            "complete": not missing,
            "missing": missing,
            "metrics_rows": (
                len(_read_csv(run_dir / "metrics.csv"))
                if (run_dir / "metrics.csv").exists() else 0
            ),
            "has_test_metrics": (run_dir / "test_metrics.json").exists(),
            "has_inference_benchmark": (
                run_dir / "inference_benchmark.json"
            ).exists(),
        }

    pair_audits = []
    for prefix in ("paper_fashion_mnist_small", "paper_cifar10_small"):
        for candidate, seeds in (
            ("hybrid_v2", range(3)),
            ("semantic_balanced_v3", range(5)),
        ):
            for seed in seeds:
                random_dir = results_dir / f"{prefix}_random_seed{seed}"
                candidate_dir = results_dir / f"{prefix}_{candidate}_seed{seed}"
                random_config = _load_json(
                    random_dir / "training_config.json"
                )
                candidate_config = _load_json(
                    candidate_dir / "training_config.json"
                )
                random_row = rows_by_name[random_dir.name]
                candidate_row = rows_by_name[candidate_dir.name]
                pair_audits.append({
                    "dataset": random_config["dataset"],
                    "candidate": candidate.replace("_", "-"),
                    "seed": seed,
                    "protocol_fields_identical": (
                        _protocol_subset(random_config)
                        == _protocol_subset(candidate_config)
                    ),
                    "gate_count_identical": (
                        random_row["gate_count"]
                        == candidate_row["gate_count"]
                    ),
                    "routing_bits_identical": (
                        random_row["deployed_index_bits"]
                        == candidate_row["deployed_index_bits"]
                    ),
                    "training_seed_paired": (
                        random_config["seed"]
                        == candidate_config["seed"]
                        == seed
                    ),
                    "topology_seed_paired": (
                        random_config["topology_seed"]
                        == candidate_config["topology_seed"]
                        == seed
                    ),
                })

    paper_rows = [
        rows_by_name[name]
        for name in EXPECTED_PAPER_RUNS
        if name in rows_by_name
    ]
    environments = {
        field: sorted({str(row[field]) for row in paper_rows})
        for field in ("source_revision", "python", "torch", "cuda_build")
    }
    recorded_hashes = sorted({
        row["recorded_source_tree_sha256"]
        for row in paper_rows
        if row["run"] not in PAPER_V3_RUNS
    })
    failed_dir = results_dir / "failed"
    failed_attempts = (
        sorted(path.name for path in failed_dir.iterdir() if path.is_dir())
        if failed_dir.exists() else []
    )
    checks = {
        "all_expected_runs_complete": all(
            item["complete"] for item in run_audits.values()
        ),
        "all_learning_curves_have_54_rows": all(
            item["metrics_rows"] == 54 for item in run_audits.values()
        ),
        "all_central_runs_have_test_metrics": all(
            run_audits[name]["has_test_metrics"]
            for name in CENTRAL_PAPER_RUNS
        ),
        "all_central_runs_have_inference_benchmarks": all(
            run_audits[name]["has_inference_benchmark"]
            for name in CENTRAL_PAPER_RUNS
        ),
        "all_pair_protocol_checks_pass": all(
            all(value for key, value in pair.items()
                if key not in {"dataset", "candidate", "seed"})
            for pair in pair_audits
        ),
        "environment_versions_consistent": all(
            len(values) == 1 for values in environments.values()
        ),
        "v3_source_hash_cohorts_explained": all(
            len({
                rows_by_name[name]["recorded_source_tree_sha256"]
                for name in PAPER_V3_RUNS
                if rows_by_name[name]["dataset"] == dataset
            }) == 2
            for dataset in ("fashion-mnist", "cifar-10")
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "expected_runs": run_audits,
        "paired_protocol": pair_audits,
        "environment_values": environments,
        "legacy_recorded_source_tree_hashes": recorded_hashes,
        "legacy_recorded_source_tree_hash_count": len(recorded_hashes),
        "v3_recorded_source_tree_hashes_by_dataset": {
            dataset: sorted({
                rows_by_name[name]["recorded_source_tree_sha256"]
                for name in PAPER_V3_RUNS
                if rows_by_name[name]["dataset"] == dataset
            })
            for dataset in ("fashion-mnist", "cifar-10")
        },
        "v3_source_hash_cohort_note": (
            "Seeds 0-2 and seeds 3-4 form two recorded broad-source-hash "
            "cohorts because reporting scripts and queue/config files were "
            "added between escalations. Training implementation files and "
            "resolved protocol fields were unchanged. Future manifests record "
            "a separate training_implementation_sha256 that excludes reporting "
            "and queue/config additions."
        ),
        "legacy_hash_scope_defect": (
            "The run-time fingerprint included generated JSON under results/ "
            "and therefore changed between otherwise source-identical runs."
        ),
        "stable_training_source_sha256_before_reporting_edits": (
            PRE_ANALYSIS_STABLE_SOURCE_SHA256
        ),
        "training_source_archive": (
            "training_source_pre_analysis.tar.gz"
        ),
        "training_source_archive_sha256": PRE_ANALYSIS_ARCHIVE_SHA256,
        "failed_attempts_excluded_from_tables": failed_attempts,
    }


def fashion_fraction_sweep(rows):
    selected = []
    for row in rows:
        match = re.fullmatch(
            r"paper_fashion_mnist_small_hybrid_f(\d{3})_v2_seed0",
            row["run"],
        )
        if match:
            fraction = int(match.group(1)) / 100
        elif row["run"] == "paper_fashion_mnist_small_hybrid_v2_seed0":
            fraction = 0.25
        else:
            continue
        selected.append({
            "fraction": fraction,
            "hard_accuracy": row["hard_accuracy"],
            "soft_accuracy": row["soft_accuracy"],
            "final_mean_gate_ancestry": row["final_mean_gate_ancestry"],
            "final_overlap_mean": row["final_overlap_mean"],
            "final_fanout_cv": row["final_fanout_cv"],
            "training_wall_seconds": row["training_wall_seconds"],
            "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
            "topology_construction_seconds": row[
                "topology_construction_seconds"
            ],
        })
    return sorted(selected, key=lambda row: row["fraction"])


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/coverage_dlgn/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/coverage_dlgn/summary"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.results)
    rows_by_name = {row["run"]: row for row in rows}
    paper_rows = [
        rows_by_name[name]
        for name in EXPECTED_PAPER_RUNS
        if name in rows_by_name
    ]
    curves = learning_curve_rows(args.results)

    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "aggregate.csv", aggregate(rows))
    write_csv(args.output / "paper_architecture_runs.csv", paper_rows)
    write_csv(args.output / "learning_curves.csv", curves)
    write_csv(
        args.output / "fashion_fraction_sweep.csv",
        fashion_fraction_sweep(rows),
    )
    write_coverage_svg(args.output / "coverage_vs_hard_accuracy.svg", rows)
    write_learning_curve_svg(
        args.output / "learning_curves_fashion_mnist.svg",
        curves,
        "fashion-mnist",
    )
    write_learning_curve_svg(
        args.output / "learning_curves_cifar10.svg",
        curves,
        "cifar-10",
    )

    write_json(
        args.output / "paired_random_vs_hybrid_v2.json",
        paired_comparison(
            rows,
            r"smoke_random_seed\d+",
            r"smoke_coverage_hybrid_v2_seed\d+",
            "Smoke-only interval; not a paper claim.",
        ),
    )
    write_json(
        args.output / "paired_paper_mnist_random_vs_hybrid_v2.json",
        paired_comparison(
            rows,
            r"paper_mnist_small_random_seed\d+",
            r"paper_mnist_small_hybrid_v2_seed\d+",
            "Three-seed paper-scale MNIST validation pilot.",
        ),
    )
    write_json(
        args.output / "paired_paper_mnist_test_random_vs_hybrid_v2.json",
        paired_comparison(
            rows,
            r"paper_mnist_small_random_seed\d+",
            r"paper_mnist_small_hybrid_v2_seed\d+",
            "Frozen best-validation checkpoints evaluated once on test.",
            metric="test_hard_accuracy",
        ),
    )
    fashion = central_comparison(rows, "fashion-mnist")
    cifar = central_comparison(rows, "cifar-10")
    fashion_v3 = semantic_v3_comparison(rows, "fashion-mnist")
    cifar_v3 = semantic_v3_comparison(rows, "cifar-10")
    write_json(args.output / "paper_fashion_mnist_central.json", fashion)
    write_json(args.output / "paper_cifar10_central.json", cifar)
    write_json(
        args.output / "paired_paper_fashion_mnist_validation.json",
        fashion["hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_fashion_mnist_test.json",
        fashion["test_hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_cifar10_validation.json",
        cifar["hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_cifar10_test.json",
        cifar["test_hard_accuracy"],
    )
    write_json(
        args.output / "paper_fashion_mnist_semantic_v3.json",
        fashion_v3,
    )
    write_json(
        args.output / "paper_cifar10_semantic_v3.json",
        cifar_v3,
    )
    write_json(
        args.output / "paired_paper_fashion_mnist_semantic_v3_validation.json",
        fashion_v3["hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_fashion_mnist_semantic_v3_test.json",
        fashion_v3["test_hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_cifar10_semantic_v3_validation.json",
        cifar_v3["hard_accuracy"],
    )
    write_json(
        args.output / "paired_paper_cifar10_semantic_v3_test.json",
        cifar_v3["test_hard_accuracy"],
    )
    write_json(
        args.output / "reproducibility_audit.json",
        reproducibility_audit(args.results, rows),
    )
    print(
        f"summarized {len(rows)} complete runs "
        f"({len(paper_rows)} paper-architecture runs) in {args.output}"
    )


if __name__ == "__main__":
    main()
