#!/usr/bin/env python3
"""Audit and aggregate the five-seed CIFAR-10 compression crossing."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

try:
    from experiments.coverage_dlgn.summarize_table1_final import mean_ci_95
except ModuleNotFoundError:
    from summarize_table1_final import mean_ci_95


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results"
SUMMARY_ROOT = ROOT / "summary"
BASE_QUEUE = ROOT / "queues" / "table2_final_cifar10_compression.json"
EXTENSION_QUEUE = ROOT / "queues" / "table2_crossing_extension_cifar10.json"
BASE_QUEUE_SUMMARY = (
    ROOT / "logs" / "table2_final_cifar10_compression" / "queue_summary.json"
)
EXTENSION_QUEUE_SUMMARY = (
    ROOT / "logs" / "table2_crossing_extension_cifar10" / "queue_summary.json"
)
CELL = "128k"
FAMILIES = {
    "random": "random",
    "coverage_v3": "v3_swap0500",
}
SEEDS = [0, 1, 2, 3, 4]


def audit_queue(queue_path: Path, summary_path: Path, expected: set[str]) -> dict:
    queue = json.loads(queue_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["failed"]:
        raise RuntimeError(f"{summary_path} contains failures")
    declared = {entry["name"] for entry in queue["entries"]}
    observed = set(summary["skipped"])
    observed.update(row["name"] for row in summary["finished"])
    if declared != expected or observed != expected:
        raise RuntimeError(
            f"queue audit mismatch for {queue_path}: "
            f"declared={sorted(declared)}, observed={sorted(observed)}, "
            f"expected={sorted(expected)}"
        )
    return summary


def main() -> None:
    base_expected = {
        f"final_table2_cifar10_{CELL}_{candidate}_seed{seed}"
        for candidate in FAMILIES.values()
        for seed in (0, 1, 2)
    }
    extension_expected = {
        f"final_table2_cifar10_{CELL}_{candidate}_seed{seed}"
        for candidate in FAMILIES.values()
        for seed in (3, 4)
    }
    base_audit = audit_queue(BASE_QUEUE, BASE_QUEUE_SUMMARY, base_expected | {
        entry["name"]
        for entry in json.loads(BASE_QUEUE.read_text())["entries"]
        if entry["cell"] != CELL
    })
    extension_audit = audit_queue(
        EXTENSION_QUEUE, EXTENSION_QUEUE_SUMMARY, extension_expected
    )

    rows = []
    family_rows = {}
    revisions = Counter()
    training_hashes = Counter()
    test_count = 0
    for family, candidate in FAMILIES.items():
        current = []
        for seed in SEEDS:
            name = f"final_table2_cifar10_{CELL}_{candidate}_seed{seed}"
            run_dir = RESULT_ROOT / name
            required = [
                "run_summary.json",
                "training_config.json",
                "environment.json",
                "metrics.csv",
                "best_checkpoint.pt",
            ]
            missing = [item for item in required if not (run_dir / item).is_file()]
            if missing:
                raise RuntimeError(f"{name} is missing {missing}")
            summary = json.loads((run_dir / "run_summary.json").read_text())
            config = json.loads((run_dir / "training_config.json").read_text())
            environment = json.loads((run_dir / "environment.json").read_text())
            if config["seed"] != seed or config["topology_seed"] != seed:
                raise RuntimeError(f"{name} violates paired seed policy")
            cost = summary["cost"]
            row = {
                "name": name,
                "family": family,
                "candidate": candidate,
                "provenance": (
                    "REPRODUCED" if family == "random" else "OUR-FINAL"
                ),
                "seed": seed,
                "best_validation_hard_accuracy": (
                    summary["best_validation_hard_accuracy"]
                ),
                "final_validation_hard_accuracy": (
                    summary["final_metrics"]["val_acc_discrete"]
                ),
                "wall_seconds": summary["wall_seconds"],
                "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
                "dense_gate_count": cost["dense_gate_count"],
                "trainable_parameters": cost["trainable_parameters"],
                "training_routing_parameters": (
                    cost["training_routing_parameters"]
                ),
                "deployed_routing_bits": cost["deployed_routing_bits"],
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
                    "test_relaxed_accuracy": test["test_relaxed_accuracy"],
                    "test_examples": test["test_examples"],
                    "test_checkpoint": test["checkpoint"],
                    "test_validation_selection_step": (
                        test["validation_selection_step"]
                    ),
                })
                test_count += 1
            rows.append(row)
            current.append(row)
            revisions[row["source_revision"]] += 1
            training_hashes[row["training_implementation_sha256"]] += 1
        family_rows[family] = current

    if len(training_hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {training_hashes}")
    if test_count not in {0, len(rows)}:
        raise RuntimeError(f"partial test evaluation: {test_count}/{len(rows)}")
    test_set_used = test_count == len(rows)

    methods = []
    for family in FAMILIES:
        current = family_rows[family]
        values = [row["best_validation_hard_accuracy"] for row in current]
        method = {
            "family": family,
            "candidate": current[0]["candidate"],
            "provenance": current[0]["provenance"],
            "seed_count": len(current),
            "seeds": SEEDS,
            "mean_best_validation_hard_accuracy": statistics.mean(values),
            "std_best_validation_hard_accuracy": statistics.stdev(values),
            "mean_wall_seconds": statistics.mean(
                row["wall_seconds"] for row in current
            ),
            "max_peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in current
            ),
            "dense_gate_count": current[0]["dense_gate_count"],
            "trainable_parameters": current[0]["trainable_parameters"],
            "training_routing_parameters": (
                current[0]["training_routing_parameters"]
            ),
            "deployed_routing_bits": current[0]["deployed_routing_bits"],
            "architecture": current[0]["architecture"],
            "parametrization": current[0]["parametrization"],
        }
        if test_set_used:
            method.update({
                "mean_test_hard_accuracy": statistics.mean(
                    row["test_hard_accuracy"] for row in current
                ),
                "std_test_hard_accuracy": statistics.stdev(
                    row["test_hard_accuracy"] for row in current
                ),
                "mean_test_relaxed_accuracy": statistics.mean(
                    row["test_relaxed_accuracy"] for row in current
                ),
            })
        methods.append(method)

    random_by_seed = {row["seed"]: row for row in family_rows["random"]}
    v3_by_seed = {row["seed"]: row for row in family_rows["coverage_v3"]}
    validation_differences = [
        v3_by_seed[seed]["best_validation_hard_accuracy"]
        - random_by_seed[seed]["best_validation_hard_accuracy"]
        for seed in SEEDS
    ]
    mean, low, high = mean_ci_95(validation_differences)
    paired = {
        "seeds": SEEDS,
        "best_validation_hard_accuracy_difference": {
            "per_seed": validation_differences,
            "mean": mean,
            "ci_95": [low, high],
        },
    }
    if test_set_used:
        differences = [
            v3_by_seed[seed]["test_hard_accuracy"]
            - random_by_seed[seed]["test_hard_accuracy"]
            for seed in SEEDS
        ]
        test_mean, test_low, test_high = mean_ci_95(differences)
        paired["test_hard_accuracy_difference"] = {
            "per_seed": differences,
            "mean": test_mean,
            "ci_95": [test_low, test_high],
        }

    payload = {
        "phase": "table2_cifar10_compression_crossing_final",
        "cell": CELL,
        "validation_metric": "hardened validation accuracy",
        "test_set_used": test_set_used,
        "run_count": len(rows),
        "source_revisions": dict(revisions),
        "training_implementation_sha256": next(iter(training_hashes)),
        "queue_audit": {
            "base_finished_count": len(base_audit["finished"]),
            "base_skipped_count": len(base_audit["skipped"]),
            "extension_finished_count": len(extension_audit["finished"]),
            "extension_skipped_count": len(extension_audit["skipped"]),
            "failed_count": 0,
        },
        "coverage_v3_vs_random_paired": paired,
        "methods": methods,
        "runs": sorted(rows, key=lambda row: row["name"]),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_ROOT / "table2_cifar10_compression_crossing_final.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = SUMMARY_ROOT / "table2_cifar10_compression_crossing_final.csv"
    csv_rows = []
    for method in methods:
        row = dict(method)
        row["seeds"] = ",".join(str(seed) for seed in row["seeds"])
        csv_rows.append(row)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json_path)
    for method in methods:
        print(
            f"{method['family']}: "
            f"{100 * method['mean_best_validation_hard_accuracy']:.3f}% +/- "
            f"{100 * method['std_best_validation_hard_accuracy']:.3f}%"
        )
    print(
        f"paired: {100 * mean:+.3f} pp, "
        f"95% CI [{100 * low:+.3f}, {100 * high:+.3f}]"
    )
    if test_set_used:
        test = paired["test_hard_accuracy_difference"]
        print(
            f"paired test: {100 * test['mean']:+.3f} pp, "
            f"95% CI [{100 * test['ci_95'][0]:+.3f}, "
            f"{100 * test['ci_95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
