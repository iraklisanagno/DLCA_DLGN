#!/usr/bin/env python3
"""Aggregate and select candidates from the locked Table 1 MNIST screen."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


RESULT_ROOT = Path("experiments/coverage_dlgn/results")
SUMMARY_ROOT = Path("experiments/coverage_dlgn/summary")
EXPECTED_RUNS = 26


def family(name: str) -> str:
    if "_v3_" in name:
        return "coverage_v3"
    if "_mommen_" in name:
        return "mommen"
    if "_lilogic_" in name:
        return "lilogic"
    if "_bitlogic_" in name:
        return "bitlogic"
    if "_random_" in name:
        return "random"
    raise ValueError(f"unrecognized screen family: {name}")


def main() -> None:
    rows = []
    revisions = Counter()
    training_hashes = Counter()
    for summary_path in sorted(
        RESULT_ROOT.glob("screen_table1_mnist_*/run_summary.json")
    ):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        cost = summary["cost"]
        row = {
            "name": run_dir.name,
            "family": family(run_dir.name),
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
            "deployed_routing_bits": cost["deployed_routing_bits"],
            "architecture": config["architecture"],
            "parametrization": config["parametrization"],
            "config": str(
                Path("experiments/coverage_dlgn/configs/table1_screen_mnist")
                / f"{run_dir.name}.json"
            ),
        }
        rows.append(row)
        revisions[environment["source_revision"]] += 1
        training_hashes[
            environment["training_implementation_sha256"]
        ] += 1
    if len(rows) != EXPECTED_RUNS:
        raise RuntimeError(f"expected {EXPECTED_RUNS} complete runs, got {len(rows)}")
    if len(revisions) != 1 or len(training_hashes) != 1:
        raise RuntimeError(
            f"mixed implementations: revisions={revisions}, "
            f"training_hashes={training_hashes}"
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["family"],
            -row["best_validation_hard_accuracy"],
            row["name"],
        ),
    )
    selected = {}
    selection_count = {
        "random": 1,
        "coverage_v3": 3,
        "mommen": 3,
        "lilogic": 3,
        "bitlogic": 1,
    }
    for method, count in selection_count.items():
        method_rows = sorted(
            (row for row in rows if row["family"] == method),
            key=lambda row: (
                -row["best_validation_hard_accuracy"],
                row["name"],
            ),
        )
        selected[method] = [row["name"] for row in method_rows[:count]]

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / "table1_mnist_screen.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ranked[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ranked)
    payload = {
        "phase": "table1_screen_mnist",
        "selection_metric": "best_validation_hard_accuracy",
        "test_set_used": False,
        "run_count": len(rows),
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(training_hashes)),
        "selected_for_20k": selected,
        "rows": sorted(
            rows,
            key=lambda row: (
                -row["best_validation_hard_accuracy"],
                row["name"],
            ),
        ),
    }
    json_path = SUMMARY_ROOT / "table1_mnist_screen.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json_path)
    for method, names in selected.items():
        print(method)
        for name in names:
            value = next(
                row["best_validation_hard_accuracy"]
                for row in rows
                if row["name"] == name
            )
            print(f"  {100 * value:.2f}% {name}")


if __name__ == "__main__":
    main()
