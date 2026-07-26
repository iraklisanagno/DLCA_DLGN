#!/usr/bin/env python3
"""Aggregate the one-seed CIFAR-10 compression screen.

The advancement rule was frozen before the screen:

* retain the two highest validation-accuracy CoverageDLGN candidates;
* retain every candidate tied at the second-place boundary, up to three;
* retain the incumbent when it is not already retained, up to three total.

The held-out test set is never read by this script.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_screen_cifar10_compression.json"
SUMMARY_ROOT = ROOT / "summary"
EXPECTED_CELLS = ("128k", "256k", "384k")
EXPECTED_PER_CELL = 10


def select_coverage_candidates(rows: list[dict]) -> list[str]:
    """Apply the frozen top-two-plus-incumbent advancement rule."""
    coverage = sorted(
        (row for row in rows if row["family"] == "coverage_v3"),
        key=lambda row: (
            -row["best_validation_hard_accuracy"],
            row["candidate"],
        ),
    )
    if len(coverage) < 2:
        raise RuntimeError("fewer than two CoverageDLGN candidates")

    second_accuracy = coverage[1]["best_validation_hard_accuracy"]
    selected = [
        row for row in coverage
        if row["best_validation_hard_accuracy"] >= second_accuracy
    ][:3]
    incumbent = next(row for row in coverage if row["candidate"] == "incumbent")
    if incumbent not in selected:
        if len(selected) == 3:
            selected[-1] = incumbent
        else:
            selected.append(incumbent)
    return [row["name"] for row in selected]


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    rows = []
    revisions = Counter()
    training_hashes = Counter()
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"incomplete run: {entry['name']}")
        summary = json.loads(summary_path.read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        cost = summary["cost"]
        rows.append({
            "name": entry["name"],
            "cell": entry["cell"],
            "gate_count": entry["gate_count"],
            "family": entry["family"],
            "candidate": entry["candidate"],
            "best_validation_hard_accuracy": (
                summary["best_validation_hard_accuracy"]
            ),
            "wall_seconds": summary["wall_seconds"],
            "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
            "dense_gate_count": cost["dense_gate_count"],
            "trainable_parameters": cost["trainable_parameters"],
            "training_routing_parameters": (
                cost["training_routing_parameters"]
            ),
            "architecture": config["architecture"],
            "parametrization": config["parametrization"],
            "config": entry["config"],
        })
        revisions[environment["source_revision"]] += 1
        training_hashes[
            environment["training_implementation_sha256"]
        ] += 1

    if len(rows) != len(EXPECTED_CELLS) * EXPECTED_PER_CELL:
        raise RuntimeError(f"expected 30 complete runs, got {len(rows)}")
    if len(revisions) != 1 or len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed implementations: revisions={revisions}, "
            f"training_hashes={training_hashes}"
        )

    selected = {}
    for cell in EXPECTED_CELLS:
        cell_rows = [row for row in rows if row["cell"] == cell]
        if len(cell_rows) != EXPECTED_PER_CELL:
            raise RuntimeError(
                f"expected {EXPECTED_PER_CELL} {cell} runs, got {len(cell_rows)}"
            )
        selected[cell] = {
            "random": [
                next(row["name"] for row in cell_rows if row["family"] == "random")
            ],
            "coverage_v3": select_coverage_candidates(cell_rows),
        }

    ranked = sorted(
        rows,
        key=lambda row: (
            row["cell"],
            -row["best_validation_hard_accuracy"],
            row["name"],
        ),
    )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "table2_cifar10_compression_screen.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ranked[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ranked)
    payload = {
        "phase": queue["phase"],
        "selection_metric": queue["selection_metric"],
        "test_set_used": False,
        "run_count": len(rows),
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "advancement_rule": (
            "top two CoverageDLGN validation candidates including ties at the "
            "second-place boundary, plus incumbent if absent, maximum three"
        ),
        "selected_for_20k": selected,
        "rows": ranked,
    }
    json_path = SUMMARY_ROOT / "table2_cifar10_compression_screen.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json_path)
    for cell in EXPECTED_CELLS:
        print(cell)
        for name in selected[cell]["coverage_v3"]:
            row = next(row for row in rows if row["name"] == name)
            print(
                f"  {100 * row['best_validation_hard_accuracy']:.3f}% "
                f"{row['candidate']}"
            )


if __name__ == "__main__":
    main()
