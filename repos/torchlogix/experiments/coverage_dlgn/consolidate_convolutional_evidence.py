#!/usr/bin/env python3
"""Build one provenance-safe snapshot of completed convolutional evidence.

This script never loads a dataset or evaluates a checkpoint.  It reads only
existing configurations, metrics, summaries, and historical test records.
The paper-faithful nine-channel studies and the WARP-style six-channel study
are deliberately kept as separate protocol families.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary"
OUTPUT = SUMMARY / "convolutional_evidence_snapshot.json"
CURVES = SUMMARY / "convolutional_evidence_curves.csv"
FREEZE = SUMMARY / "convolutional_evidence_freeze.json"

S_FAMILIES = {
    "fixed_random": {
        seed: f"pilot_conv_cifar10_paper_small_random_seed{seed}"
        for seed in range(5)
    },
    "frozen_v4": {
        seed: f"pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}"
        for seed in range(5)
    },
    "unified_u1": {
        0: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed0",
        1: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed1",
        2: "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed2",
        3: "pilot_conv_cifar10_paper_small_semantic_degree_balanced_seed3",
        4: "pilot_conv_cifar10_paper_small_semantic_degree_balanced_seed4",
    },
}

M_RUNS = {
    "fixed_random": "full_conv_cifar10_paper_medium_random_seed0_200k",
    "frozen_v4": "full_conv_cifar10_paper_medium_legacy_v4_seed0",
}

WARP_RUNS = {
    name: f"warp_fig4_medium_{name}_seed0"
    for name in (
        "warp_fixed_uniform",
        "warp_fixed_distributive",
        "warp_learnable",
        "paired_random_fixed_uniform",
        "legacy_v4_fixed_uniform",
    )
}

T95 = {2: 12.706204736, 3: 4.30265273, 5: 2.776445105}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_count(run_dir: Path) -> int:
    with (run_dir / "thresholds.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    columns = [name for name in header if name.startswith("thresh_")]
    if not columns:
        raise RuntimeError(f"no threshold columns: {run_dir}")
    return len(columns)


def paired(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired comparison requires equal lists of length >= 2")
    differences = [r - l for l, r in zip(left, right)]
    center = mean(differences)
    half = T95[len(differences)] * stdev(differences) / math.sqrt(len(differences))
    return {
        "paired_differences_percentage_points": differences,
        "paired_mean_percentage_points": center,
        "paired_std_percentage_points": stdev(differences),
        "paired_t95_ci_percentage_points": [center - half, center + half],
        "positive_seed_count": sum(value > 0 for value in differences),
        "seed_count": len(differences),
    }


def metrics(run_name: str) -> list[dict[str, str]]:
    with (RESULTS / run_name / "metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def best_validation(run_name: str) -> float:
    payload = read_json(RESULTS / run_name / "run_summary.json")
    return 100.0 * float(payload["best_validation_hard_accuracy"])


def historical_test(run_name: str) -> float:
    payload = read_json(RESULTS / run_name / "test_metrics.json")
    return 100.0 * float(payload["test_hard_accuracy"])


def aggregate_small_curves() -> list[dict]:
    curves: dict[str, dict[int, list[float]]] = {}
    for family, runs in S_FAMILIES.items():
        per_step: dict[int, list[float]] = {}
        for run_name in runs.values():
            for row in metrics(run_name):
                step = int(row["step"])
                per_step.setdefault(step, []).append(
                    100.0 * float(row["val_acc_discrete"])
                )
        if any(len(values) != 5 for values in per_step.values()):
            raise RuntimeError(f"incomplete five-seed S curve: {family}")
        curves[family] = per_step

    rows = []
    steps = sorted(curves["fixed_random"])
    for step in steps:
        row = {"protocol": "paper_faithful_s", "step": step}
        for family, curve in curves.items():
            row[f"{family}_hard_validation_mean_percent"] = mean(curve[step])
            row[f"{family}_hard_validation_std_percent"] = stdev(curve[step])
        row["v4_minus_random_pp"] = (
            row["frozen_v4_hard_validation_mean_percent"]
            - row["fixed_random_hard_validation_mean_percent"]
        )
        row["u1_minus_random_pp"] = (
            row["unified_u1_hard_validation_mean_percent"]
            - row["fixed_random_hard_validation_mean_percent"]
        )
        rows.append(row)
    return rows


def first_mean_step_at(rows: list[dict], family: str, threshold: float):
    key = f"{family}_hard_validation_mean_percent"
    return next((row["step"] for row in rows if row[key] >= threshold), None)


def freeze_inputs() -> dict:
    run_names = sorted(
        {
            *(
                run_name
                for runs in S_FAMILIES.values()
                for run_name in runs.values()
            ),
            *M_RUNS.values(),
            *WARP_RUNS.values(),
        }
    )
    records = {}
    for run_name in run_names:
        run_dir = RESULTS / run_name
        files = {}
        for name in (
            "training_config.json",
            "metrics.csv",
            "run_summary.json",
            "early_stop.json",
            "best_checkpoint.pt",
            "test_metrics.json",
            "inference_benchmark.json",
            "topology.json",
            "conv_topology.json",
        ):
            path = run_dir / name
            if path.is_file():
                files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        records[run_name] = {"files": files}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "purpose": "immutable input manifest for convolutional evidence consolidation",
        "source_revision_at_freeze": revision,
        "test_policy": {
            "paper_faithful_s": (
                "Historical random/V4 seeds 0-2 already have one test record; "
                "U1 and seeds 3-4 remain validation-only. No test evaluation is "
                "performed by the consolidation or deployment scripts."
            ),
            "paper_faithful_m": (
                "Both frozen best-validation checkpoints have exactly one existing "
                "held-out test record and must not be queried again."
            ),
            "warp_style_medium": "Validation-only; no held-out test record.",
        },
        "runs": records,
    }


def verify_or_create_freeze() -> dict:
    current = freeze_inputs()
    if FREEZE.is_file():
        frozen = read_json(FREEZE)
        if frozen != current:
            raise RuntimeError(
                "convolutional evidence changed after freeze; inspect before proceeding"
            )
        return frozen
    FREEZE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def architecture_audit() -> dict:
    representatives = {
        "paper_faithful_s": RESULTS / S_FAMILIES["fixed_random"][0],
        "paper_faithful_m": RESULTS / M_RUNS["fixed_random"],
        "warp_style_medium": RESULTS / WARP_RUNS["paired_random_fixed_uniform"],
    }
    expected = {
        "paper_faithful_s": ("ClgnCifar10PaperSmall", 3, 9, 32, 20),
        "paper_faithful_m": ("ClgnCifar10PaperMedium", 3, 9, 256, 40),
        "warp_style_medium": ("ClgnCifar10Medium", 2, 6, 256, 40),
    }
    rows = {}
    for protocol, run_dir in representatives.items():
        config = read_json(run_dir / "training_config.json")
        architecture, thresholds, boolean_channels, k_num, tau = expected[protocol]
        actual_thresholds = threshold_count(run_dir)
        if config["architecture"] != architecture or actual_thresholds != thresholds:
            raise RuntimeError(f"architecture mismatch: {protocol}")
        rows[protocol] = {
            "architecture": architecture,
            "input_precision_bits": 2,
            "thresholds_per_rgb_channel": actual_thresholds,
            "boolean_input_channels": boolean_channels,
            "k_num": k_num,
            "group_sum_tau": tau,
            "logic_tree_depth": 3,
            "convolutional_stages": 4,
            "receptive_field": "3x3",
            "parameterization": config["parametrization"],
            "binarization": config["binarization"],
            "binarization_initialization": config["binarization_init"],
            "augmentation": config["augmentation"],
            "data_split_seed": config["data_split_seed"],
        }
    rows["paper_faithful_s_m_same_principle"] = {
        "same_threshold_encoding": True,
        "same_boolean_channels": True,
        "same_convolutional_stage_pattern": True,
        "same_gate_parameterization": True,
        "differences": "width scale (k_num 32 vs 256) and GroupSum tau (20 vs 40)",
    }
    rows["warp_is_separate"] = (
        "The WARP-style Medium uses two thresholds/six Boolean channels and "
        "must not be pooled with the paper-faithful nine-channel S/M results."
    )
    return rows


def main() -> int:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    freeze = verify_or_create_freeze()
    unified = read_json(SUMMARY / "cifar10_conv_small_unified_five_seed.json")
    medium = read_json(SUMMARY / "cifar10_paper_medium_200k_paired.json")
    warp = read_json(SUMMARY / "warp_fig4_cifar10_medium.json")
    components = read_json(SUMMARY / "cifar10_conv_small_v4_components.json")
    channel_spatial = read_json(SUMMARY / "cifar10_conv_small_channel_spatial.json")

    random_test = [
        historical_test(S_FAMILIES["fixed_random"][seed]) for seed in range(3)
    ]
    v4_test = [
        historical_test(S_FAMILIES["frozen_v4"][seed]) for seed in range(3)
    ]
    curve_rows = aggregate_small_curves()
    with CURVES.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(curve_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    small_cost = unified["resource_check_new_seeds_3_4"]
    payload = {
        "status": "COMPLETED-EVIDENCE-SNAPSHOT",
        "scope": "existing convolutional results only; no new accuracy training or test query",
        "freeze_manifest": str(FREEZE.relative_to(ROOT)),
        "freeze_run_count": len(freeze["runs"]),
        "architecture_audit": architecture_audit(),
        "paper_faithful_nine_channel": {
            "small_20k": {
                "provenance": "TRIED-SELECTION",
                "selection_metric": "best hardened validation accuracy",
                "five_seed_validation": {
                    "mean_percent": unified["mean_percent"],
                    "paired_effects": unified["paired_effects"],
                    "promotion_decision": unified["promotion_decision"],
                },
                "historical_three_seed_test": {
                    "provenance": "TRIED-HISTORICAL-TEST",
                    "scope": "random versus V4 seeds 0-2 only; U1 was never tested",
                    "random_percent": random_test,
                    "frozen_v4_percent": v4_test,
                    "mean_percent": {
                        "random": mean(random_test),
                        "frozen_v4": mean(v4_test),
                    },
                    "frozen_v4_minus_random": paired(random_test, v4_test),
                },
                "learning_efficiency": {
                    "curve_csv": str(CURVES.relative_to(ROOT)),
                    "mean_first_step_at_threshold_percent": {
                        str(threshold): {
                            family: first_mean_step_at(curve_rows, family, threshold)
                            for family in S_FAMILIES
                        }
                        for threshold in (50.0, 55.0, 57.0)
                    },
                },
                "matched_resource_accounting": small_cost,
                "component_ablation": components,
                "negative_channel_spatial_adapter": channel_spatial,
                "heldout_policy": (
                    "Do not evaluate U1 or seeds 3-4 on test during this evidence pass."
                ),
            },
            "medium_200k": medium,
        },
        "warp_style_six_channel_medium_30k": {
            **warp,
            "comparison_scope": (
                "Separate validation-only compatibility study; approximate reported "
                "Figure 4 endpoints and one local seed, not a paper-faithful "
                "LogicTreeNet-M reproduction."
            ),
        },
        "deployment": {
            "summary": "summary/convolutional_deployment.json",
            "csv": "summary/convolutional_deployment.csv",
            "input_policy": "synthetic Boolean inputs; no dataset access",
        },
        "interpretation_constraints": [
            "Do not pool nine-channel and six-channel accuracies.",
            "Do not present U1 as having held-out test accuracy.",
            "Do not attach a confidence interval to the one-seed Medium result.",
            "Matched random/V4/U1 architectures have equal gate and parameter budgets.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(CURVES)
    print(FREEZE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
