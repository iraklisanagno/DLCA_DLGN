#!/usr/bin/env python3
"""Summarize the frozen one-seed 200K LogicTreeNet-M paired experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "cifar10_paper_medium_200k_freeze.json"
TEST_SUMMARY = (
    ROOT
    / "logs"
    / "cifar10_paper_medium_200k"
    / "test"
    / "test_evaluation_summary.json"
)
OUTPUT = ROOT / "summary" / "cifar10_paper_medium_200k_paired.json"
CURVE_OUTPUT = ROOT / "summary" / "cifar10_paper_medium_200k_curve.csv"
PAPER_TEST_ACCURACY = 0.7101


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curve(run_dir: Path) -> dict[int, dict[str, str]]:
    with (run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    curve = {int(row["step"]): row for row in rows}
    if sorted(curve) != list(range(2000, 200001, 2000)):
        raise RuntimeError(f"incomplete learning curve: {run_dir}")
    return curve


def topology_summary(run_dir: Path) -> dict:
    dense = json.loads((run_dir / "topology.json").read_text())
    conv = json.loads((run_dir / "conv_topology.json").read_text())
    dense_seconds = sum(row["construction_seconds"] for row in dense["layers"])
    conv_seconds = sum(row["construction_seconds"] for row in conv["layers"])
    return {
        "construction_seconds": {
            "convolutional": conv_seconds,
            "dense_classifier": dense_seconds,
            "total": conv_seconds + dense_seconds,
        },
        "convolutional_layers": [
            {
                key: row.get(key)
                for key in (
                    "depth",
                    "strategy",
                    "in_channels",
                    "out_kernels",
                    "channel_fanout_cv",
                    "channel_fanout_min",
                    "channel_fanout_max",
                    "distinct_channel_groups",
                    "duplicate_channel_groups",
                    "channel_pair_span_mean",
                    "raw_channel_coverage_fraction",
                    "raw_channel_ancestry_mean",
                    "raw_predecessor_jaccard_mean",
                    "cross_channel_leaf_pair_fraction",
                    "unique_spatial_offsets_mean",
                    "construction_seconds",
                )
            }
            for row in conv["layers"]
        ],
    }


def first_step_at(curve: dict[int, dict[str, str]], threshold: float):
    return next(
        (
            step
            for step in sorted(curve)
            if float(curve[step]["val_acc_discrete"]) >= threshold
        ),
        None,
    )


def main() -> int:
    freeze = json.loads(FREEZE.read_text())
    test_summary = json.loads(TEST_SUMMARY.read_text())
    if not test_summary["test_set_used"] or test_summary["completed_count"] != 2:
        raise RuntimeError("both one-time held-out evaluations must complete")

    run_dirs = {
        method: ROOT / record["run_dir"]
        for method, record in freeze["runs"].items()
    }
    for method, run_dir in run_dirs.items():
        record = freeze["runs"][method]
        if sha256(run_dir / record["checkpoint"]) != record["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint changed: {method}")

    curves = {method: load_curve(path) for method, path in run_dirs.items()}
    steps = sorted(curves["coverage_v4"])
    curve_rows = []
    hard_deltas = []
    relaxed_deltas = []
    for step in steps:
        v4 = curves["coverage_v4"][step]
        random = curves["fixed_random"][step]
        hard_delta = 100 * (
            float(v4["val_acc_discrete"])
            - float(random["val_acc_discrete"])
        )
        relaxed_delta = 100 * (
            float(v4["val_acc_relaxed"])
            - float(random["val_acc_relaxed"])
        )
        hard_deltas.append(hard_delta)
        relaxed_deltas.append(relaxed_delta)
        curve_rows.append(
            {
                "step": step,
                "v4_hard_validation": float(v4["val_acc_discrete"]),
                "random_hard_validation": float(random["val_acc_discrete"]),
                "hard_delta_pp": hard_delta,
                "v4_relaxed_validation": float(v4["val_acc_relaxed"]),
                "random_relaxed_validation": float(random["val_acc_relaxed"]),
                "relaxed_delta_pp": relaxed_delta,
                "v4_train_loss": float(v4["train_loss"]),
                "random_train_loss": float(random["train_loss"]),
            }
        )

    with CURVE_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(curve_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    tests = {
        method: json.loads((run_dir / "test_metrics.json").read_text())
        for method, run_dir in run_dirs.items()
    }
    benchmarks = {
        method: json.loads((run_dir / "inference_benchmark.json").read_text())
        for method, run_dir in run_dirs.items()
    }
    topology = {
        method: topology_summary(run_dir)
        for method, run_dir in run_dirs.items()
    }
    random_run_summary = json.loads(
        (run_dirs["fixed_random"] / "run_summary.json").read_text()
    )

    v4_test = tests["coverage_v4"]
    random_test = tests["fixed_random"]
    v4_benchmark = benchmarks["coverage_v4"]
    random_benchmark = benchmarks["fixed_random"]
    v4_topology_seconds = topology["coverage_v4"]["construction_seconds"]["total"]
    random_topology_seconds = topology["fixed_random"]["construction_seconds"]["total"]
    payload = {
        "phase": "cifar10_paper_medium_200k_paired",
        "status": "ONE-SEED-COMPLETE",
        "architecture": "ClgnCifar10PaperMedium",
        "paper_identifier": "LogicTreeNet-M",
        "input_boolean_channels": 9,
        "seed": 0,
        "steps": 200000,
        "validation_evaluations": len(steps),
        "test_set_used": True,
        "heldout_queries_per_checkpoint": 1,
        "checkpoint_freeze_manifest": str(FREEZE.relative_to(ROOT)),
        "learning_curve_csv": str(CURVE_OUTPUT.relative_to(ROOT)),
        "validation_selection": {
            method: {
                "step": freeze["runs"][method]["selection_step"],
                "hard_accuracy": freeze["runs"][method]["validation_hard_accuracy"],
                "relaxed_accuracy_at_selected_checkpoint": freeze["runs"][method][
                    "validation_relaxed_accuracy_at_selection"
                ],
            }
            for method in ("coverage_v4", "fixed_random")
        },
        "learning_curve_comparison": {
            "v4_hard_wins": sum(delta > 0 for delta in hard_deltas),
            "ties": sum(delta == 0 for delta in hard_deltas),
            "v4_hard_losses": sum(delta < 0 for delta in hard_deltas),
            "mean_hard_delta_pp": statistics.mean(hard_deltas),
            "median_hard_delta_pp": statistics.median(hard_deltas),
            "mean_relaxed_delta_pp": statistics.mean(relaxed_deltas),
            "first_step_at_70_percent_hard": {
                method: first_step_at(curves[method], 0.70)
                for method in ("coverage_v4", "fixed_random")
            },
        },
        "heldout_test": {
            "coverage_v4": v4_test,
            "fixed_random": random_test,
            "hard_delta_pp": 100
            * (v4_test["test_hard_accuracy"] - random_test["test_hard_accuracy"]),
            "relaxed_delta_pp": 100
            * (
                v4_test["test_relaxed_accuracy"]
                - random_test["test_relaxed_accuracy"]
            ),
            "validation_to_test_hard_gap_pp": {
                method: 100
                * (
                    tests[method]["test_hard_accuracy"]
                    - freeze["runs"][method]["validation_hard_accuracy"]
                )
                for method in ("coverage_v4", "fixed_random")
            },
            "reported_logic_tree_net_m_test_accuracy": PAPER_TEST_ACCURACY,
            "gap_to_reported_pp": {
                method: 100
                * (tests[method]["test_hard_accuracy"] - PAPER_TEST_ACCURACY)
                for method in ("coverage_v4", "fixed_random")
            },
        },
        "topology": topology,
        "offline_topology_overhead_seconds": (
            v4_topology_seconds - random_topology_seconds
        ),
        "training_runtime": {
            "coverage_v4_artifact_span_seconds": (
                (run_dirs["coverage_v4"] / "metrics.csv").stat().st_mtime
                - (run_dirs["coverage_v4"] / "training_config.json").stat().st_mtime
            ),
            "coverage_v4_span_is_approximate": True,
            "fixed_random_wall_seconds": random_run_summary["wall_seconds"],
            "fixed_random_peak_gpu_memory_bytes": random_run_summary[
                "peak_gpu_memory_bytes"
            ],
            "coverage_v4_peak_training_memory": (
                "not recorded because the run ended by controlled SIGINT"
            ),
        },
        "hardened_inference": {
            "coverage_v4": v4_benchmark,
            "fixed_random": random_benchmark,
            "runtime_delta_percent": 100
            * (
                v4_benchmark["milliseconds_per_batch"]
                / random_benchmark["milliseconds_per_batch"]
                - 1
            ),
            "peak_memory_delta_bytes": (
                v4_benchmark["peak_device_memory_bytes"]
                - random_benchmark["peak_device_memory_bytes"]
            ),
            "interpretation": (
                "One timing pass; the sub-percent difference is measurement noise, "
                "not a speed claim."
            ),
        },
        "matched_deployment_cost": {
            "reported_logic_gate_operations": 3080000,
            "reported_logic_gate_operations_are_approximate": True,
            **random_run_summary["cost"],
            "dense_gate_count_note": (
                "dense_gate_count covers the three classifier layers; the "
                "paper's operation count includes spatially instantiated convolution gates."
            ),
        },
        "limitations": [
            "This is one seed and is not a statistical accuracy claim.",
            "The 71.01% paper value is reported test accuracy; local validation values are not compared directly to it.",
            "Training-only crop/flip augmentation is an explicit local adaptation because the paper does not document augmentation.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(CURVE_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
