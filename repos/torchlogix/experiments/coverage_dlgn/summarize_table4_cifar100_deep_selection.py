#!/usr/bin/env python3
"""Audit paired deep dense CIFAR-100 confirmations."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_selection.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table4_cifar100_deep_selection"
    / "queue_summary.json"
)
JSON_PATH = ROOT / "summary" / "table4_cifar100_deep_selection.json"
CSV_PATH = ROOT / "summary" / "table4_cifar100_deep_selection.csv"
T_CRITICAL_DF2 = 4.302652729696142


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(
            f"deep CIFAR-100 selection failed: {audit['failed']}"
        )
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != expected:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rows = []
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        run = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"selection touched held-out test: {run_dir}")
        rows.append({
            "name": entry["name"],
            "architecture_label": entry["architecture_label"],
            "family": entry["family"],
            "candidate": entry["candidate"],
            "seed": config["seed"],
            "best_validation_hard_accuracy": (
                run["best_validation_hard_accuracy"]
            ),
            "wall_seconds": run["wall_seconds"],
            "topology_construction_seconds": sum(
                layer["construction_seconds"] for layer in run["topology"]
            ),
            "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
            "dense_gate_count": run["cost"]["dense_gate_count"],
            "trainable_parameters": run["cost"]["trainable_parameters"],
            "source_revision": environment["source_revision"],
            "training_implementation_sha256": environment[
                "training_implementation_sha256"
            ],
        })

    hashes = {row["training_implementation_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    selections = {}
    for label in sorted({row["architecture_label"] for row in rows}):
        architecture_rows = [
            row for row in rows if row["architecture_label"] == label
        ]
        by_key = {
            (row["family"], row["seed"]): row
            for row in architecture_rows
        }
        gains = [
            by_key[("coverage_v3", seed)][
                "best_validation_hard_accuracy"
            ]
            - by_key[("random", seed)]["best_validation_hard_accuracy"]
            for seed in (0, 1, 2)
        ]
        gain_mean = statistics.mean(gains)
        gain_std = statistics.stdev(gains)
        half_width = T_CRITICAL_DF2 * gain_std / math.sqrt(len(gains))
        family_stats = {}
        for family in ("random", "coverage_v3"):
            values = [
                row["best_validation_hard_accuracy"]
                for row in architecture_rows
                if row["family"] == family
            ]
            family_stats[family] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "n": len(values),
            }
        selections[label] = {
            "selected_v3_candidate": next(
                row["candidate"] for row in architecture_rows
                if row["family"] == "coverage_v3"
            ),
            "family_stats": family_stats,
            "paired_gain_pp": 100 * gain_mean,
            "paired_gain_95ci_pp": [
                100 * (gain_mean - half_width),
                100 * (gain_mean + half_width),
            ],
            "per_seed_gain_pp": [100 * value for value in gains],
            "promote_to_paper_schedule": gain_mean > 0,
        }

    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": False,
        "training_implementation_sha256": next(iter(hashes)),
        "selections": selections,
        "rows": sorted(rows, key=lambda row: row["name"]),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["name"]))
    print(JSON_PATH)
    for label, selection in selections.items():
        print(
            f"{label}: paired gain={selection['paired_gain_pp']:+.3f} pp "
            f"CI={selection['paired_gain_95ci_pp']} "
            f"promote={selection['promote_to_paper_schedule']}"
        )


if __name__ == "__main__":
    main()
