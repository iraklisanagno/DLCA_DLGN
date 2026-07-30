#!/usr/bin/env python3
"""Audit and summarize the deeper dense CIFAR-100 one-seed screens."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_screen.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table4_cifar100_deep_screen" / "queue_summary.json"
)
JSON_PATH = ROOT / "summary" / "table4_cifar100_deep_screen.json"
CSV_PATH = ROOT / "summary" / "table4_cifar100_deep_screen.csv"


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"deep CIFAR-100 screen failed: {audit['failed']}")
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
            raise RuntimeError(f"screen touched held-out test: {run_dir}")
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
            "coverage_candidate_pool_size": config.get(
                "coverage_candidate_pool_size"
            ),
            "coverage_swap_fraction": config.get(
                "coverage_swap_fraction"
            ),
            "coverage_novelty_weight": config.get(
                "coverage_novelty_weight"
            ),
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
        random_row = next(
            row for row in architecture_rows if row["family"] == "random"
        )
        winner = max(
            (
                row for row in architecture_rows
                if row["family"] == "coverage_v3"
            ),
            key=lambda row: row["best_validation_hard_accuracy"],
        )
        gain_pp = 100 * (
            winner["best_validation_hard_accuracy"]
            - random_row["best_validation_hard_accuracy"]
        )
        selections[label] = {
            "random_name": random_row["name"],
            "selected_v3_name": winner["name"],
            "selected_v3_candidate": winner["candidate"],
            "screen_gain_pp": gain_pp,
            "advance_to_confirmation": gain_pp > 0,
        }

    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": False,
        "run_count": len(rows),
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
            f"{label}: winner={selection['selected_v3_candidate']} "
            f"gain={selection['screen_gain_pp']:+.3f} pp "
            f"advance={selection['advance_to_confirmation']}"
        )


if __name__ == "__main__":
    main()
