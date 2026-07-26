#!/usr/bin/env python3
"""Summarize the reduced CIFAR-10 S comparator screen."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_s_comparator_screen.json"
SUMMARY_PATH = ROOT / "summary" / "table2_s_comparator_screen.json"


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    rows = []
    revisions = Counter()
    training_hashes = Counter()
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"incomplete screen run: {entry['name']}")
        summary = json.loads(summary_path.read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        rows.append({
            "name": entry["name"],
            "family": entry["family"],
            "best_validation_hard_accuracy": (
                summary["best_validation_hard_accuracy"]
            ),
            "wall_seconds": summary["wall_seconds"],
            "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
            "dense_gate_count": summary["cost"]["dense_gate_count"],
            "trainable_parameters": summary["cost"]["trainable_parameters"],
            "training_routing_parameters": (
                summary["cost"]["training_routing_parameters"]
            ),
            "architecture": config["architecture"],
            "connections_num_candidates": (
                config["connections_num_candidates"]
            ),
            "config": entry["config"],
        })
        revisions[environment["source_revision"]] += 1
        training_hashes[
            environment["training_implementation_sha256"]
        ] += 1
    if len(rows) != 3:
        raise RuntimeError(f"expected three screen rows, got {len(rows)}")
    if len(revisions) != 1 or len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed implementations: revisions={revisions}, "
            f"training_hashes={training_hashes}"
        )

    mommen = max(
        (row for row in rows if row["family"] == "mommen"),
        key=lambda row: (
            row["best_validation_hard_accuracy"],
            -row["connections_num_candidates"],
        ),
    )
    lilogic = next(row for row in rows if row["family"] == "lilogic")
    payload = {
        "phase": queue["phase"],
        "selection_metric": queue["selection_metric"],
        "test_set_used": False,
        "run_count": len(rows),
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "selected_for_full_training": {
            "mommen": mommen["name"],
            "lilogic": lilogic["name"],
        },
        "rows": sorted(
            rows,
            key=lambda row: (
                row["family"],
                -row["best_validation_hard_accuracy"],
                row["name"],
            ),
        ),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(SUMMARY_PATH)
    for family, name in payload["selected_for_full_training"].items():
        row = next(row for row in rows if row["name"] == name)
        print(
            f"{family}: {100 * row['best_validation_hard_accuracy']:.3f}% "
            f"{name}"
        )


if __name__ == "__main__":
    main()
