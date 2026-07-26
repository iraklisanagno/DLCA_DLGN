#!/usr/bin/env python3
"""Audit and aggregate full CIFAR-10 S comparator runs."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_s_comparator_final.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table2_s_comparator_final" / "queue_summary.json"
)
SUMMARY_ROOT = ROOT / "summary"
EXPECTED_SEEDS = [0, 1, 2]


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"final comparator queue failed: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if expected != observed:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    runs = []
    grouped = defaultdict(list)
    revisions = Counter()
    hashes = Counter()
    test_count = 0
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        summary = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        cost = summary["cost"]
        row = {
            "name": entry["name"],
            "family": entry["family"],
            "seed": config["seed"],
            "provenance": "ADAPTED",
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
            "connections_num_candidates": (
                config["connections_num_candidates"]
            ),
            "group_sum_temperature": config.get(
                "group_sum_temperature",
                1.0 / 0.03,
            ),
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
        runs.append(row)
        grouped[entry["family"]].append(row)
        revisions[row["source_revision"]] += 1
        hashes[row["training_implementation_sha256"]] += 1

    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    if test_count not in {0, len(runs)}:
        raise RuntimeError(f"partial test evaluation: {test_count}/{len(runs)}")
    test_set_used = test_count == len(runs)

    methods = []
    for family in ("mommen", "lilogic"):
        current = grouped[family]
        seeds = sorted(row["seed"] for row in current)
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{family} has seeds {seeds}, expected {EXPECTED_SEEDS}"
            )
        values = [row["best_validation_hard_accuracy"] for row in current]
        method = {
            "family": family,
            "provenance": "ADAPTED",
            "seed_count": len(current),
            "seeds": seeds,
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
            "connections_num_candidates": (
                current[0]["connections_num_candidates"]
            ),
            "group_sum_temperature": current[0]["group_sum_temperature"],
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

    payload = {
        "phase": queue["phase"],
        "validation_metric": "hardened validation accuracy",
        "test_set_used": test_set_used,
        "run_count": len(runs),
        "source_revisions": dict(revisions),
        "training_implementation_sha256": next(iter(hashes)),
        "queue_audit": {
            "finished_count": len(audit["finished"]),
            "skipped_count": len(audit["skipped"]),
            "failed_count": len(audit["failed"]),
            "wall_seconds": audit["wall_seconds"],
        },
        "methods": methods,
        "runs": sorted(runs, key=lambda row: row["name"]),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_ROOT / "table2_s_comparator_final.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = SUMMARY_ROOT / "table2_s_comparator_final.csv"
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
        if test_set_used:
            print(
                f"  test {100 * method['mean_test_hard_accuracy']:.3f}% +/- "
                f"{100 * method['std_test_hard_accuracy']:.3f}%"
            )


if __name__ == "__main__":
    main()
