#!/usr/bin/env python3
"""Diagnose frozen V4 swaps against its degree-balanced no-swap base.

The script loads the historical CIFAR-10 S checkpoints instead of regenerating
their topologies. This makes the comparison an audit of the exact trained
models and preserves the frozen result directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model
from torchlogix.topology import analyze_conv_channel_topology


RUN_PATTERNS = {
    "random": "pilot_conv_cifar10_paper_small_random_seed{seed}",
    "v4": "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}",
    "no_swap": (
        "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed{seed}"
    ),
}
TOPOLOGY_METRICS = (
    "duplicate_channel_groups",
    "channel_pair_span_mean",
    "raw_predecessor_jaccard_mean",
    "raw_channel_ancestry_mean",
    "raw_channel_coverage_fraction",
    "channel_fanout_cv",
)


def _load_model(run_dir: Path):
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"
    payload = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    state_dict = payload["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    return model


def _conv_layers(model):
    return [
        module
        for module in model.modules()
        if hasattr(module, "connections")
        and isinstance(getattr(module.connections, "indices", None), list)
        and getattr(module.connections, "channel_group_size", None) is not None
    ]


def _channel_groups(model) -> list[np.ndarray]:
    groups = []
    for layer in _conv_layers(model):
        first_level = layer.connections.indices[0].detach().cpu().numpy()
        leaf_channels = first_level[:, :, 0, :, -1]
        groups.append(np.asarray([
            np.sort(np.unique(leaf_channels[:, kernel].reshape(-1)))
            for kernel in range(layer.num_kernels)
        ], dtype=np.int64))
    return groups


def _degree_vector(groups: np.ndarray, channels: int) -> np.ndarray:
    return np.bincount(groups.reshape(-1), minlength=channels)


def _multiset_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_counts = Counter(map(tuple, left.tolist()))
    right_counts = Counter(map(tuple, right.tolist()))
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    return float(intersection / union)


def _load_learning_curve(run_dir: Path) -> dict[int, dict[str, float]]:
    with (run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["step"]): {
            "hard_accuracy": 100.0 * float(row["val_acc_discrete"]),
            "relaxed_accuracy": 100.0 * float(row["val_acc_relaxed"]),
            "train_loss": float(row["train_loss"]),
        }
        for row in rows
    }


def analyze(results_dir: Path) -> dict:
    topology_by_seed: list[dict] = []
    curves: dict[str, dict[int, list[dict[str, float]]]] = {
        method: {} for method in RUN_PATTERNS
    }

    for seed in range(3):
        models = {
            method: _load_model(results_dir / pattern.format(seed=seed))
            for method, pattern in RUN_PATTERNS.items()
            if method != "random"
        }
        rows = {
            method: analyze_conv_channel_topology(model)
            for method, model in models.items()
        }
        groups = {
            method: _channel_groups(model)
            for method, model in models.items()
        }
        seed_layers = []
        for depth, (no_swap, v4) in enumerate(
            zip(groups["no_swap"], groups["v4"])
        ):
            no_swap_row = rows["no_swap"][depth]
            v4_row = rows["v4"][depth]
            no_swap_degree = _degree_vector(
                no_swap, int(no_swap_row["in_channels"])
            )
            v4_degree = _degree_vector(v4, int(v4_row["in_channels"]))
            seed_layers.append({
                "depth": depth,
                "in_channels": int(no_swap_row["in_channels"]),
                "out_kernels": int(no_swap_row["out_kernels"]),
                "changed_output_pair_fraction": float(
                    np.mean(np.any(no_swap != v4, axis=1))
                ),
                "pair_multiset_jaccard": _multiset_jaccard(no_swap, v4),
                "exact_fanout_preserved": bool(
                    np.array_equal(no_swap_degree, v4_degree)
                ),
                "spatial_coordinates_preserved": (
                    no_swap_row["spatial_coordinates_sha256"]
                    == v4_row["spatial_coordinates_sha256"]
                ),
                "no_swap": {
                    key: no_swap_row[key] for key in TOPOLOGY_METRICS
                },
                "v4": {
                    key: v4_row[key] for key in TOPOLOGY_METRICS
                },
                "v4_minus_no_swap": {
                    key: float(v4_row[key] - no_swap_row[key])
                    for key in TOPOLOGY_METRICS
                },
            })
        topology_by_seed.append({"seed": seed, "layers": seed_layers})

        for method, pattern in RUN_PATTERNS.items():
            curve = _load_learning_curve(
                results_dir / pattern.format(seed=seed)
            )
            for step, values in curve.items():
                curves[method].setdefault(step, []).append(values)

    layer_count = len(topology_by_seed[0]["layers"])
    topology_aggregate = []
    for depth in range(layer_count):
        layer_rows = [row["layers"][depth] for row in topology_by_seed]
        aggregate = {
            "depth": depth,
            "in_channels": layer_rows[0]["in_channels"],
            "out_kernels": layer_rows[0]["out_kernels"],
            "changed_output_pair_fraction_mean": mean(
                row["changed_output_pair_fraction"] for row in layer_rows
            ),
            "pair_multiset_jaccard_mean": mean(
                row["pair_multiset_jaccard"] for row in layer_rows
            ),
            "exact_fanout_preserved_all_seeds": all(
                row["exact_fanout_preserved"] for row in layer_rows
            ),
            "spatial_coordinates_preserved_all_seeds": all(
                row["spatial_coordinates_preserved"] for row in layer_rows
            ),
        }
        for method in ("no_swap", "v4"):
            aggregate[method] = {
                key: mean(row[method][key] for row in layer_rows)
                for key in TOPOLOGY_METRICS
            }
        aggregate["v4_minus_no_swap"] = {
            key: mean(row["v4_minus_no_swap"][key] for row in layer_rows)
            for key in TOPOLOGY_METRICS
        }
        topology_aggregate.append(aggregate)

    learning_curve = []
    common_steps = sorted(
        set.intersection(*(set(method_curves) for method_curves in curves.values()))
    )
    for step in common_steps:
        row: dict[str, object] = {"step": step}
        for method in RUN_PATTERNS:
            row[method] = {
                metric: mean(values[metric] for values in curves[method][step])
                for metric in ("hard_accuracy", "relaxed_accuracy", "train_loss")
            }
        row["no_swap_minus_random_hard_pp"] = (
            row["no_swap"]["hard_accuracy"] - row["random"]["hard_accuracy"]
        )
        row["no_swap_minus_v4_hard_pp"] = (
            row["no_swap"]["hard_accuracy"] - row["v4"]["hard_accuracy"]
        )
        learning_curve.append(row)

    learning_summary = {}
    steps = np.asarray(common_steps, dtype=np.float64)
    for method in RUN_PATTERNS:
        hard = np.asarray([
            row[method]["hard_accuracy"] for row in learning_curve
        ])
        learning_summary[method] = {
            "hard_accuracy_auc_percent": float(
                np.trapezoid(hard, steps) / (steps[-1] - steps[0])
            ),
            "hard_accuracy_at_20000_percent": float(hard[-1]),
            "best_mean_hard_accuracy_percent": float(hard.max()),
        }

    ancestry_delta = [
        abs(layer["v4_minus_no_swap"]["raw_predecessor_jaccard_mean"])
        for layer in topology_aggregate
    ]
    return {
        "protocol": {
            "dataset": "CIFAR-10",
            "architecture": "ClgnCifar10PaperSmall",
            "seeds": [0, 1, 2],
            "training_steps": 20_000,
            "source": "exact historical best checkpoints and metrics.csv files",
            "base_structure": (
                "affine-ordered degree-balanced butterfly; not round-robin"
            ),
        },
        "topology_by_seed": topology_by_seed,
        "topology_aggregate": topology_aggregate,
        "learning_curve": learning_curve,
        "learning_summary": learning_summary,
        "diagnosis": {
            "fanout_is_not_the_difference": all(
                layer["exact_fanout_preserved_all_seeds"]
                for layer in topology_aggregate
            ),
            "spatial_coordinates_are_not_the_difference": all(
                layer["spatial_coordinates_preserved_all_seeds"]
                for layer in topology_aggregate
            ),
            "v4_reduces_pair_duplication_in_middle_layers": all(
                topology_aggregate[depth]["v4_minus_no_swap"][
                    "duplicate_channel_groups"
                ] < 0.0
                for depth in (1, 2)
            ),
            "v4_increases_pair_span_in_every_layer": all(
                layer["v4_minus_no_swap"]["channel_pair_span_mean"] >= 0.0
                for layer in topology_aggregate
            ),
            "max_absolute_predecessor_jaccard_change": max(ancestry_delta),
            "interpretation": (
                "V4 swaps disrupt a subset of the balanced butterfly pairs and "
                "increase pair diversity/span, but do not materially reduce "
                "raw-channel ancestry overlap. The no-swap curve overtakes V4 "
                "late in training, so the extra rewiring is not justified by "
                "the intended coverage diagnostic on this architecture."
            ),
        },
    }


def write_outputs(payload: dict, output_json: Path, output_csv: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with output_csv.open("w", newline="") as handle:
        fieldnames = [
            "depth",
            "in_channels",
            "out_kernels",
            "changed_output_pair_fraction_mean",
            "pair_multiset_jaccard_mean",
            "no_swap_duplicate_groups",
            "v4_duplicate_groups",
            "no_swap_pair_span_mean",
            "v4_pair_span_mean",
            "no_swap_predecessor_jaccard_mean",
            "v4_predecessor_jaccard_mean",
            "no_swap_raw_ancestry_mean",
            "v4_raw_ancestry_mean",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in payload["topology_aggregate"]:
            writer.writerow({
                "depth": row["depth"],
                "in_channels": row["in_channels"],
                "out_kernels": row["out_kernels"],
                "changed_output_pair_fraction_mean": (
                    row["changed_output_pair_fraction_mean"]
                ),
                "pair_multiset_jaccard_mean": (
                    row["pair_multiset_jaccard_mean"]
                ),
                "no_swap_duplicate_groups": (
                    row["no_swap"]["duplicate_channel_groups"]
                ),
                "v4_duplicate_groups": row["v4"]["duplicate_channel_groups"],
                "no_swap_pair_span_mean": (
                    row["no_swap"]["channel_pair_span_mean"]
                ),
                "v4_pair_span_mean": row["v4"]["channel_pair_span_mean"],
                "no_swap_predecessor_jaccard_mean": (
                    row["no_swap"]["raw_predecessor_jaccard_mean"]
                ),
                "v4_predecessor_jaccard_mean": (
                    row["v4"]["raw_predecessor_jaccard_mean"]
                ),
                "no_swap_raw_ancestry_mean": (
                    row["no_swap"]["raw_channel_ancestry_mean"]
                ),
                "v4_raw_ancestry_mean": (
                    row["v4"]["raw_channel_ancestry_mean"]
                ),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "summary" / "cifar10_conv_small_no_swap_diagnostics.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "summary" / "cifar10_conv_small_no_swap_diagnostics.csv",
    )
    args = parser.parse_args()
    payload = analyze(args.results_dir)
    write_outputs(payload, args.output_json, args.output_csv)
    print(json.dumps(payload["diagnosis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
