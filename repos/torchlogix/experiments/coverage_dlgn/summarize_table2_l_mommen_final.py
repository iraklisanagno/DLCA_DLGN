#!/usr/bin/env python3
"""Audit and summarize the one-seed full CIFAR-10 L Mommen run."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_l_mommen_final.json"
QUEUE_SUMMARY = ROOT / "logs" / "table2_l_mommen_final" / "queue_summary.json"
JSON_PATH = ROOT / "summary" / "table2_l_mommen_final.json"
CSV_PATH = ROOT / "summary" / "table2_l_mommen_final.csv"


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"L Mommen queue failed: {audit['failed']}")
    if len(queue["entries"]) != 1:
        raise RuntimeError("L Mommen policy requires exactly one run")
    entry = queue["entries"][0]
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != {entry["name"]}:
        raise RuntimeError(f"queue mismatch: {sorted(observed)}")

    run_dir = Path(entry["output"])
    run = json.loads((run_dir / "run_summary.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    if (run_dir / "test_metrics.json").exists():
        raise RuntimeError("held-out test must remain locked at this stage")
    cost = run["cost"]
    row = {
        "name": entry["name"],
        "family": "mommen",
        "provenance": "TRIED",
        "seed": config["seed"],
        "seed_count": 1,
        "best_validation_hard_accuracy": (
            run["best_validation_hard_accuracy"]
        ),
        "final_validation_hard_accuracy": (
            run["final_metrics"]["val_acc_discrete"]
        ),
        "wall_seconds": run["wall_seconds"],
        "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
        "dense_gate_count": cost["dense_gate_count"],
        "trainable_parameters": cost["trainable_parameters"],
        "training_routing_parameters": (
            cost["training_routing_parameters"]
        ),
        "deployed_routing_bits": cost["deployed_routing_bits"],
        "architecture": config["architecture"],
        "connections_num_candidates": config["connections_num_candidates"],
        "source_revision": environment["source_revision"],
        "training_implementation_sha256": environment[
            "training_implementation_sha256"
        ],
    }
    payload = {
        "phase": queue["phase"],
        "policy": queue["policy"],
        "provenance": "TRIED",
        "test_set_used": False,
        "run_count": 1,
        "queue_audit": {
            "finished_count": len(audit["finished"]),
            "skipped_count": len(audit["skipped"]),
            "failed_count": len(audit["failed"]),
            "wall_seconds": audit["wall_seconds"],
        },
        "run": row,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    print(JSON_PATH)
    print(f"validation={100 * row['best_validation_hard_accuracy']:.3f}%")


if __name__ == "__main__":
    main()
