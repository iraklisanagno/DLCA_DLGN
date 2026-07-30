#!/usr/bin/env python3
"""Audit and summarize the fixed-budget CIFAR-100 depth pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    ROOT / "protocols" / "table4_dense_cifar100_depth384k.json"
)
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_depth384k_pilot.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table4_cifar100_depth384k_pilot"
    / "queue_summary.json"
)
JSON_PATH = (
    ROOT / "summary" / "table4_cifar100_depth384k_pilot.json"
)
CSV_PATH = ROOT / "summary" / "table4_cifar100_depth384k_pilot.csv"


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"depth pilot failed: {audit['failed']}")
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
            raise RuntimeError(f"pilot touched held-out test: {run_dir}")
        rows.append({
            "name": entry["name"],
            "architecture_label": entry["architecture_label"],
            "depth": entry["depth"],
            "width_per_layer": entry["width_per_layer"],
            "family": entry["family"],
            "seed": entry["seed"],
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
            "configured_temperature": 10.0,
            "coverage_swap_fraction": config.get(
                "coverage_swap_fraction"
            ),
        })
    hashes = {row["training_implementation_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")
    threshold = 1.0
    selections = {}
    for label in protocol["architectures"]:
        pair = {
            row["family"]: row
            for row in rows if row["architecture_label"] == label
        }
        gain = 100 * (
            pair["coverage_v3"]["best_validation_hard_accuracy"]
            - pair["random"]["best_validation_hard_accuracy"]
        )
        selections[label] = {
            "random_accuracy": pair["random"][
                "best_validation_hard_accuracy"
            ],
            "coverage_v3_accuracy": pair["coverage_v3"][
                "best_validation_hard_accuracy"
            ],
            "paired_gain_pp": gain,
            "promote_to_confirmation": gain >= threshold,
        }
    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": queue["selection_metric"],
        "test_set_used": False,
        "training_implementation_sha256": next(iter(hashes)),
        "promotion_threshold_pp": threshold,
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
            f"{label}: random={100 * selection['random_accuracy']:.3f}% "
            f"v3={100 * selection['coverage_v3_accuracy']:.3f}% "
            f"gain={selection['paired_gain_pp']:+.3f} pp "
            f"promote={selection['promote_to_confirmation']}"
        )


if __name__ == "__main__":
    main()
