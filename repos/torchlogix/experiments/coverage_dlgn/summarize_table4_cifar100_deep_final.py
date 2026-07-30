#!/usr/bin/env python3
"""Audit and summarize full-schedule 6x64K CIFAR-100 results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_final.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table4_cifar100_deep_final" / "queue_summary.json"
)
JSON_PATH = ROOT / "summary" / "table4_cifar100_deep_final.json"
CSV_PATH = ROOT / "summary" / "table4_cifar100_deep_final.csv"
T_CRITICAL_DF2 = 4.302652729696142


def paired_stats(rows: list[dict], metric: str) -> dict:
    by_key = {
        (row["family"], row["seed"]): row
        for row in rows
    }
    gains = [
        by_key[("coverage_v3", seed)][metric]
        - by_key[("random", seed)][metric]
        for seed in (0, 1, 2)
    ]
    mean_gain = statistics.mean(gains)
    half_width = (
        T_CRITICAL_DF2 * statistics.stdev(gains) / math.sqrt(len(gains))
    )
    families = {}
    for family in ("random", "coverage_v3"):
        values = [
            row[metric] for row in rows if row["family"] == family
        ]
        families[family] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "n": len(values),
        }
    return {
        "family_stats": families,
        "paired_gain_pp": 100 * mean_gain,
        "paired_gain_95ci_pp": [
            100 * (mean_gain - half_width),
            100 * (mean_gain + half_width),
        ],
        "per_seed_gain_pp": [100 * value for value in gains],
    }


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"deep CIFAR-100 final failed: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != expected:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rows = []
    test_presence = []
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        run = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        test_path = run_dir / "test_metrics.json"
        test_presence.append(test_path.exists())
        row = {
            "name": entry["name"],
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
            "test_hard_accuracy": None,
            "source_revision": environment["source_revision"],
            "training_implementation_sha256": environment[
                "training_implementation_sha256"
            ],
        }
        if test_path.exists():
            test = json.loads(test_path.read_text())
            row["test_hard_accuracy"] = test["test_hard_accuracy"]
        rows.append(row)

    if any(test_presence) and not all(test_presence):
        raise RuntimeError("partial held-out evaluation detected")
    hashes = {row["training_implementation_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    payload = {
        "phase": queue["phase"],
        "provenance": "OUR-FINAL" if all(test_presence) else "TRIED",
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": all(test_presence),
        "training_implementation_sha256": next(iter(hashes)),
        "validation": paired_stats(
            rows, "best_validation_hard_accuracy"
        ),
        "test": (
            paired_stats(rows, "test_hard_accuracy")
            if all(test_presence) else None
        ),
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
    validation = payload["validation"]
    print(
        f"validation gain={validation['paired_gain_pp']:+.3f} pp "
        f"CI={validation['paired_gain_95ci_pp']}"
    )
    if payload["test"] is not None:
        test = payload["test"]
        print(
            f"test gain={test['paired_gain_pp']:+.3f} pp "
            f"CI={test['paired_gain_95ci_pp']}"
        )


if __name__ == "__main__":
    main()
