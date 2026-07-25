#!/usr/bin/env python3
"""Audit and aggregate a locked Table 1 final validation phase."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from experiments.coverage_dlgn.prepare_table1_final import (
        final_seeds_for,
    )
except ModuleNotFoundError:
    from prepare_table1_final import final_seeds_for


RESULT_ROOT = Path("experiments/coverage_dlgn/results")
SUMMARY_ROOT = Path("experiments/coverage_dlgn/summary")
LOG_ROOT = Path("experiments/coverage_dlgn/logs")
FAMILY_ORDER = [
    "random",
    "coverage_v3",
    "mommen",
    "lilogic",
    "bitlogic",
]
PROVENANCE = {
    "random": "REPRODUCED",
    "coverage_v3": "OUR-FINAL",
    "mommen": "ADAPTED",
    "lilogic": "ADAPTED",
    "bitlogic": "ADAPTED",
}
T_CRITICAL_95 = {
    2: 12.706204736,
    3: 4.30265273,
    4: 3.182446305,
    5: 2.776445105,
}


def mean_ci_95(values: list[float]) -> tuple[float, float, float]:
    """Return the paired mean and two-sided Student-t 95% interval."""
    count = len(values)
    if count not in T_CRITICAL_95:
        raise ValueError(f"unsupported sample count for CI: {count}")
    mean = statistics.mean(values)
    margin = (
        T_CRITICAL_95[count]
        * statistics.stdev(values)
        / math.sqrt(count)
    )
    return mean, mean - margin, mean + margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=["mnist", "fashion"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell = args.cell
    selection_path = (
        SUMMARY_ROOT / f"table1_{cell}_selection.json"
    )
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_for_final"]
    queue_path = (
        Path("experiments/coverage_dlgn/queues")
        / f"table1_final_{cell}.json"
    )
    queue = json.loads(queue_path.read_text())
    queue_summary_path = (
        LOG_ROOT / f"table1_final_{cell}" / "queue_summary.json"
    )
    queue_summary = json.loads(queue_summary_path.read_text())
    if queue_summary["failed"]:
        raise RuntimeError(
            f"final queue contains failures: {queue_summary['failed']}"
        )

    expected_names = {
        entry["name"] for entry in queue["entries"]
    }
    observed_queue_names = set(queue_summary["skipped"])
    observed_queue_names.update(
        entry["name"] for entry in queue_summary["finished"]
    )
    if observed_queue_names != expected_names:
        raise RuntimeError(
            "queue coverage mismatch: "
            f"missing={sorted(expected_names - observed_queue_names)}, "
            f"unexpected={sorted(observed_queue_names - expected_names)}"
        )

    revisions: Counter[str] = Counter()
    training_hashes: Counter[str] = Counter()
    runs = []
    family_runs: dict[str, list[dict]] = {}
    test_result_count = 0
    for family in FAMILY_ORDER:
        candidate = selected[family]
        expected_seeds = final_seeds_for(cell, family)
        rows = []
        for seed in expected_seeds:
            name = f"final_{candidate}_seed{seed}"
            run_dir = RESULT_ROOT / name
            required = [
                "run_summary.json",
                "training_config.json",
                "environment.json",
                "metrics.csv",
            ]
            missing = [
                filename
                for filename in required
                if not (run_dir / filename).is_file()
            ]
            if missing:
                raise RuntimeError(f"{name} is missing {missing}")
            summary = json.loads(
                (run_dir / "run_summary.json").read_text()
            )
            config = json.loads(
                (run_dir / "training_config.json").read_text()
            )
            environment = json.loads(
                (run_dir / "environment.json").read_text()
            )
            if config["seed"] != seed or config["topology_seed"] != seed:
                raise RuntimeError(f"{name} violates paired seed policy")
            cost = summary["cost"]
            row = {
                "name": name,
                "family": family,
                "candidate": candidate,
                "provenance": PROVENANCE[family],
                "seed": seed,
                "best_validation_hard_accuracy": (
                    summary["best_validation_hard_accuracy"]
                ),
                "final_validation_hard_accuracy": (
                    summary["final_metrics"]["val_acc_discrete"]
                ),
                "final_validation_relaxed_accuracy": (
                    summary["final_metrics"]["val_acc_relaxed"]
                ),
                "wall_seconds": summary["wall_seconds"],
                "peak_gpu_memory_bytes": (
                    summary["peak_gpu_memory_bytes"]
                ),
                "dense_gate_count": cost["dense_gate_count"],
                "trainable_parameters": cost["trainable_parameters"],
                "training_routing_parameters": (
                    cost["training_routing_parameters"]
                ),
                "deployed_routing_bits": (
                    cost["deployed_routing_bits"]
                ),
                "architecture": config["architecture"],
                "parametrization": config["parametrization"],
                "source_revision": environment["source_revision"],
                "training_implementation_sha256": environment[
                    "training_implementation_sha256"
                ],
            }
            test_path = run_dir / "test_metrics.json"
            if test_path.is_file():
                test = json.loads(test_path.read_text())
                row.update({
                    "test_hard_accuracy": test["test_hard_accuracy"],
                    "test_relaxed_accuracy": (
                        test["test_relaxed_accuracy"]
                    ),
                    "test_examples": test["test_examples"],
                    "test_checkpoint": test["checkpoint"],
                    "test_validation_selection_step": (
                        test["validation_selection_step"]
                    ),
                })
                test_result_count += 1
            rows.append(row)
            runs.append(row)
            revisions[row["source_revision"]] += 1
            training_hashes[
                row["training_implementation_sha256"]
            ] += 1
        family_runs[family] = rows

    if len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed training implementations: {training_hashes}"
        )
    if test_result_count not in {0, len(runs)}:
        raise RuntimeError(
            "partial held-out evaluation is forbidden: "
            f"{test_result_count}/{len(runs)} test results exist"
        )
    test_set_used = test_result_count == len(runs)

    methods = []
    stable_fields = [
        "candidate",
        "provenance",
        "dense_gate_count",
        "trainable_parameters",
        "training_routing_parameters",
        "deployed_routing_bits",
        "architecture",
        "parametrization",
    ]
    for family in FAMILY_ORDER:
        rows = family_runs[family]
        for field in stable_fields:
            values = {row[field] for row in rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"{family} has inconsistent {field}: {values}"
                )
        best = [
            row["best_validation_hard_accuracy"] for row in rows
        ]
        final_hard = [
            row["final_validation_hard_accuracy"] for row in rows
        ]
        final_relaxed = [
            row["final_validation_relaxed_accuracy"] for row in rows
        ]
        methods.append({
            "family": family,
            **{field: rows[0][field] for field in stable_fields},
            "seed_count": len(rows),
            "seeds": [row["seed"] for row in rows],
            "mean_best_validation_hard_accuracy": statistics.mean(best),
            "std_best_validation_hard_accuracy": statistics.stdev(best),
            "mean_final_validation_hard_accuracy": statistics.mean(
                final_hard
            ),
            "std_final_validation_hard_accuracy": statistics.stdev(
                final_hard
            ),
            "mean_final_validation_relaxed_accuracy": statistics.mean(
                final_relaxed
            ),
            "mean_wall_seconds": statistics.mean(
                row["wall_seconds"] for row in rows
            ),
            "max_peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in rows
            ),
        })
        if test_set_used:
            methods[-1].update({
                "mean_test_hard_accuracy": statistics.mean(
                    row["test_hard_accuracy"] for row in rows
                ),
                "std_test_hard_accuracy": statistics.stdev(
                    row["test_hard_accuracy"] for row in rows
                ),
                "mean_test_relaxed_accuracy": statistics.mean(
                    row["test_relaxed_accuracy"] for row in rows
                ),
            })

    random_by_seed = {
        row["seed"]: row for row in family_runs["random"]
    }
    v3_by_seed = {
        row["seed"]: row for row in family_runs["coverage_v3"]
    }
    paired_seeds = sorted(set(random_by_seed) & set(v3_by_seed))
    best_differences = [
        v3_by_seed[seed]["best_validation_hard_accuracy"]
        - random_by_seed[seed]["best_validation_hard_accuracy"]
        for seed in paired_seeds
    ]
    final_differences = [
        v3_by_seed[seed]["final_validation_hard_accuracy"]
        - random_by_seed[seed]["final_validation_hard_accuracy"]
        for seed in paired_seeds
    ]
    best_mean, best_low, best_high = mean_ci_95(best_differences)
    final_mean, final_low, final_high = mean_ci_95(
        final_differences
    )
    paired_payload = {
        "seeds": paired_seeds,
        "best_validation_hard_accuracy_difference": {
            "per_seed": best_differences,
            "mean": best_mean,
            "ci_95": [best_low, best_high],
        },
        "final_validation_hard_accuracy_difference": {
            "per_seed": final_differences,
            "mean": final_mean,
            "ci_95": [final_low, final_high],
        },
    }
    if test_set_used:
        test_differences = [
            v3_by_seed[seed]["test_hard_accuracy"]
            - random_by_seed[seed]["test_hard_accuracy"]
            for seed in paired_seeds
        ]
        test_mean, test_low, test_high = mean_ci_95(test_differences)
        paired_payload["test_hard_accuracy_difference"] = {
            "per_seed": test_differences,
            "mean": test_mean,
            "ci_95": [test_low, test_high],
        }

    payload = {
        "phase": f"table1_final_{cell}",
        "validation_metric": "hardened validation accuracy",
        "test_set_used": test_set_used,
        "run_count": len(runs),
        "source_revisions": dict(revisions),
        "training_implementation_sha256": next(iter(training_hashes)),
        "queue_audit": {
            "finished_count": len(queue_summary["finished"]),
            "skipped_count": len(queue_summary["skipped"]),
            "failed_count": len(queue_summary["failed"]),
            "wall_seconds": queue_summary["wall_seconds"],
        },
        "coverage_v3_vs_random_paired": paired_payload,
        "methods": methods,
        "runs": sorted(runs, key=lambda row: row["name"]),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_ROOT / f"table1_{cell}_final.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = SUMMARY_ROOT / f"table1_{cell}_final.csv"
    csv_rows = []
    for row in methods:
        csv_row = dict(row)
        csv_row["seeds"] = ",".join(
            str(seed) for seed in row["seeds"]
        )
        csv_rows.append(csv_row)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json_path)
    for row in methods:
        mean = 100 * row["mean_best_validation_hard_accuracy"]
        std = 100 * row["std_best_validation_hard_accuracy"]
        print(
            f"{row['family']}: {mean:.3f}% +/- {std:.3f}% "
            f"n={row['seed_count']} [{row['provenance']}]"
        )
    print(
        "coverage_v3_vs_random: "
        f"{100 * best_mean:+.3f} pp, "
        f"95% CI [{100 * best_low:+.3f}, {100 * best_high:+.3f}]"
    )
    if test_set_used:
        test = paired_payload["test_hard_accuracy_difference"]
        print(
            "coverage_v3_vs_random_test: "
            f"{100 * test['mean']:+.3f} pp, "
            f"95% CI [{100 * test['ci_95'][0]:+.3f}, "
            f"{100 * test['ci_95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
