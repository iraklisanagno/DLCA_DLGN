#!/usr/bin/env python3
"""Aggregate frozen third-round accuracy and deployment trade-offs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "third_round_validation_freeze.json"
JSON_OUTPUT = ROOT / "summary" / "third_round_results.json"
RUN_CSV = ROOT / "summary" / "third_round_runs.csv"
GROUP_CSV = ROOT / "summary" / "third_round_groups.csv"
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}

REPORTED_REFERENCES = {
    ("third_lilogic_cifar10", "m", "random"): {
        "accuracy_pct": 49.17,
        "label": "LILogicNet fixed M",
        "provenance": "reported",
    },
    ("third_lilogic_cifar10", "m", "top32"): {
        "accuracy_pct": 57.28,
        "spread_pct": 0.30,
        "label": "LILogicNet 1Top32-64K",
        "provenance": "reported",
    },
    ("third_lilogic_cifar10", "l", "random"): {
        "accuracy_pct": 54.76,
        "label": "LILogicNet fixed L",
        "provenance": "reported",
    },
    ("third_lilogic_cifar10", "l", "top32"): {
        "accuracy_pct": 60.98,
        "spread_pct": 0.19,
        "label": "LILogicNet 2Top32-128K",
        "provenance": "reported",
    },
    ("third_bitlogic_cifar10", "s", "best"): {
        "accuracy_pct": 38.93,
        "spread_pct": 0.19,
        "label": "BitLogic rank-4 S",
        "provenance": "reported",
    },
    ("third_bitlogic_cifar10", "m", "best"): {
        "accuracy_pct": 49.22,
        "spread_pct": 0.26,
        "label": "BitLogic rank-4 M",
        "provenance": "reported",
    },
    ("third_bitlogic_cifar10", "l", "best"): {
        "accuracy_pct": 58.06,
        "spread_pct": 0.14,
        "label": "BitLogic rank-4 L",
        "provenance": "reported",
    },
}

CURRENT_DENSE_REFERENCES = {
    "m": {
        "random": "paper_cifar10_medium_random_seed{seed}",
        "v3": "paper_cifar10_medium_semantic_balanced_v3_seed{seed}",
        "u2": "third_u2_cifar10_m_seed{seed}",
    },
    "l": {
        "random": "final_table2_cifar10_l_random_seed{seed}",
        "v3": "final_table2_cifar10_l_v3_swap0500_seed{seed}",
        "u2": "third_u2_cifar10_l_seed{seed}",
    },
}


def mean_sd_ci(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = (
        T_CRITICAL_95[n] * sd / math.sqrt(n)
        if n in T_CRITICAL_95 and sd is not None
        else None
    )
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half if half is not None else None,
        "ci95_high": mean + half if half is not None else None,
    }


def checkpoint_metrics(test: dict, name: str) -> dict:
    matches = [row for row in test["checkpoints"] if row["checkpoint"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one test row for {name}, found {len(matches)}")
    return matches[0]


def collect_run(row: dict) -> dict:
    run_dir = Path(row["run_dir"])
    test_path = run_dir / "third_round_test_metrics.json"
    benchmark_path = run_dir / "synthetic_inference_benchmark_v2.json"
    if not test_path.is_file():
        raise RuntimeError(f"missing held-out result: {test_path}")
    if not benchmark_path.is_file():
        raise RuntimeError(f"missing synthetic benchmark: {benchmark_path}")
    config = json.loads((run_dir / "training_config.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    test = json.loads(test_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    best = checkpoint_metrics(test, "best_checkpoint.pt")
    final = checkpoint_metrics(test, "final_checkpoint.pt")
    cost = summary["cost"]
    rank = config["lut_rank"]
    gates = cost["dense_gate_count"]
    topology_seconds = sum(
        layer.get("construction_seconds", 0.0)
        for layer in summary.get("topology", [])
    )
    test_hashes_match = all(
        checkpoint_metrics(test, name)["checkpoint_sha256"]
        == row["artifacts"][name]["sha256"]
        for name in ("best_checkpoint.pt", "final_checkpoint.pt")
    )
    return {
        "phase": row["phase"],
        "coordinate": row["coordinate"],
        "family": row["family"],
        "name": row["name"],
        "seed": row["seed"],
        "provenance": (
            "our-transfer" if row["family"] == "u2" else "reproduced-local"
        ),
        "architecture": row["architecture"],
        "rank": rank,
        "fan_in": rank,
        "gate_count": gates,
        "truth_table_bits": gates * (2 ** rank),
        "trainable_parameters": cost["trainable_parameters"],
        "training_routing_parameters": cost["training_routing_parameters"],
        "deployed_routing_bits": cost["deployed_routing_bits"],
        "best_validation_hard_pct": row["best_hard_validation_pct"],
        "final_validation_hard_pct": row["final_hard_validation_pct"],
        "best_test_hard_pct": 100 * best["test_hard_accuracy"],
        "final_test_hard_pct": 100 * final["test_hard_accuracy"],
        "best_test_relaxed_pct": 100 * best["test_relaxed_accuracy"],
        "final_test_relaxed_pct": 100 * final["test_relaxed_accuracy"],
        "training_wall_minutes": summary["wall_seconds"] / 60,
        "topology_construction_seconds": topology_seconds,
        "training_peak_gpu_gib": summary["peak_gpu_memory_bytes"] / 2**30,
        "inference_ms_per_batch128": benchmark["milliseconds_per_batch"],
        "inference_examples_per_second": benchmark["examples_per_second"],
        "inference_peak_gpu_gib": benchmark["peak_device_memory_bytes"] / 2**30,
        "benchmark_checkpoint_sha256_matches_freeze": (
            benchmark["checkpoint_sha256"]
            == row["artifacts"]["best_checkpoint.pt"]["sha256"]
        ),
        "test_checkpoint_sha256_matches_freeze": test_hashes_match,
        "heldout_checkpoint_queries": test["heldout_checkpoint_queries"],
    }


def aggregate_runs(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["phase"], row["coordinate"], row["family"])].append(row)
    result = []
    metrics = (
        "best_validation_hard_pct",
        "final_validation_hard_pct",
        "best_test_hard_pct",
        "final_test_hard_pct",
        "training_wall_minutes",
        "topology_construction_seconds",
        "training_peak_gpu_gib",
        "inference_ms_per_batch128",
        "inference_peak_gpu_gib",
    )
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        group = {
            "phase": key[0],
            "coordinate": key[1],
            "family": key[2],
            "seeds": sorted(row["seed"] for row in rows),
            "provenance": first["provenance"],
            "rank": first["rank"],
            "gate_count": first["gate_count"],
            "truth_table_bits": first["truth_table_bits"],
            "trainable_parameters": first["trainable_parameters"],
            "training_routing_parameters": first[
                "training_routing_parameters"
            ],
            "deployed_routing_bits": first["deployed_routing_bits"],
            "reported_reference": REPORTED_REFERENCES.get(key),
        }
        for metric in metrics:
            group[metric] = mean_sd_ci([row[metric] for row in rows])
        result.append(group)
    return result


def paired_effects(runs: list[dict]) -> list[dict]:
    indexed = {
        (row["phase"], row["coordinate"], row["family"], row["seed"]): row
        for row in runs
    }
    effects = []
    for phase in ("third_lilogic_cifar10", "third_bitlogic_cifar10"):
        coordinates = sorted({
            row["coordinate"] for row in runs if row["phase"] == phase
        })
        for coordinate in coordinates:
            random_seeds = {
                row["seed"] for row in runs
                if row["phase"] == phase
                and row["coordinate"] == coordinate
                and row["family"] == "random"
            }
            u2_seeds = {
                row["seed"] for row in runs
                if row["phase"] == phase
                and row["coordinate"] == coordinate
                and row["family"] == "u2"
            }
            seeds = sorted(random_seeds & u2_seeds)
            metrics = {}
            for metric in (
                "best_validation_hard_pct",
                "best_test_hard_pct",
                "final_test_hard_pct",
            ):
                deltas = [
                    indexed[(phase, coordinate, "u2", seed)][metric]
                    - indexed[(phase, coordinate, "random", seed)][metric]
                    for seed in seeds
                ]
                metrics[metric] = {
                    **mean_sd_ci(deltas),
                    "per_seed_u2_minus_random_pp": deltas,
                    "positive_seed_count": sum(delta > 0 for delta in deltas),
                }
            effects.append({
                "phase": phase,
                "coordinate": coordinate,
                "comparison": "u2_minus_fixed_random",
                "paired_seeds": seeds,
                "metrics": metrics,
            })
    return effects


def existing_best_test_pct(run_dir: Path, third_round: bool) -> float:
    if third_round:
        payload = json.loads(
            (run_dir / "third_round_test_metrics.json").read_text()
        )
        row = checkpoint_metrics(payload, "best_checkpoint.pt")
        return 100 * row["test_hard_accuracy"]
    payload = json.loads((run_dir / "test_metrics.json").read_text())
    return 100 * payload["test_hard_accuracy"]


def current_dense_cross_comparisons() -> list[dict]:
    results_root = ROOT / "results"
    rows = []
    for coordinate, families in CURRENT_DENSE_REFERENCES.items():
        values = {}
        for family, pattern in families.items():
            values[family] = [
                existing_best_test_pct(
                    results_root / pattern.format(seed=seed),
                    third_round=(family == "u2"),
                )
                for seed in range(3)
            ]
        u2_random = [
            u2 - random for u2, random in zip(values["u2"], values["random"])
        ]
        u2_v3 = [
            u2 - v3 for u2, v3 in zip(values["u2"], values["v3"])
        ]
        rows.append({
            "coordinate": coordinate,
            "paired_seeds": [0, 1, 2],
            "best_test_hard_pct": {
                family: {**mean_sd_ci(cell), "per_seed": cell}
                for family, cell in values.items()
            },
            "u2_minus_random_pp": {
                **mean_sd_ci(u2_random),
                "per_seed": u2_random,
                "positive_seed_count": sum(x > 0 for x in u2_random),
            },
            "u2_minus_v3_pp": {
                **mean_sd_ci(u2_v3),
                "per_seed": u2_v3,
                "positive_seed_count": sum(x > 0 for x in u2_v3),
            },
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    flattened = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        flattened.append(flat)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    freeze = json.loads(FREEZE.read_text())
    runs = [
        collect_run(row)
        for rows in freeze["groups"].values()
        for row in rows
    ]
    if not all(row["benchmark_checkpoint_sha256_matches_freeze"] for row in runs):
        raise RuntimeError("synthetic benchmark checkpoint hash mismatch")
    if not all(row["test_checkpoint_sha256_matches_freeze"] for row in runs):
        raise RuntimeError("held-out checkpoint hash mismatch")
    if not all(row["heldout_checkpoint_queries"] == 2 for row in runs):
        raise RuntimeError("unexpected held-out checkpoint query count")
    groups = aggregate_runs(runs)
    payload = {
        "protocol": "THIRD_ROUND_PROTOCOL.md",
        "validation_freeze": str(FREEZE),
        "run_count": len(runs),
        "provenance_labels": {
            "our-transfer": "U2 applied without method changes",
            "reproduced-local": "locally trained comparator/protocol control",
            "reported": "value transcribed from the named paper",
        },
        "accuracy_units": "percentage points",
        "spread": "sample standard deviation; CI uses Student t",
        "runs": runs,
        "groups": groups,
        "paired_effects": paired_effects(runs),
        "current_dense_cross_comparisons": current_dense_cross_comparisons(),
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(RUN_CSV, runs)
    write_csv(GROUP_CSV, groups)
    print(JSON_OUTPUT)
    print(RUN_CSV)
    print(GROUP_CSV)


if __name__ == "__main__":
    main()
