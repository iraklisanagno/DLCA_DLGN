#!/usr/bin/env python3
"""Summarize frozen full-schedule convolutional S validation and test."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "second_round_convolutional_validation_freeze.json"
OUTPUT = ROOT / "summary" / "second_round_convolutional_final.json"


def main() -> None:
    freeze = json.loads(FREEZE.read_text())
    rows = {}
    costs = []
    for method, frozen in freeze["runs"].items():
        run_dir = Path(frozen["run_dir"])
        summary = json.loads((run_dir / "run_summary.json").read_text())
        test_path = run_dir / "test_metrics.json"
        if not test_path.is_file():
            raise RuntimeError(f"missing one-time held-out result: {test_path}")
        test = json.loads(test_path.read_text())
        topology_seconds = sum(
            layer["construction_seconds"] for layer in summary["topology"]
        )
        row = {
            "best_hard_validation_pct": frozen["best_hard_validation_pct"],
            "final_hard_validation_pct": (
                100 * summary["final_metrics"]["val_acc_discrete"]
            ),
            "test_hard_pct": 100 * test["test_hard_accuracy"],
            "test_relaxed_pct": 100 * test["test_relaxed_accuracy"],
            "training_wall_hours": summary["wall_seconds"] / 3600,
            "peak_gpu_gib": summary["peak_gpu_memory_bytes"] / 2**30,
            "topology_construction_seconds": topology_seconds,
            "cost": summary["cost"],
            "cost_interpretation": {
                "dense_gate_count": "classifier gates only",
                "total_learned_gate_functions": 83552,
                "spatial_gate_applications": 874496,
            },
        }
        rows[method] = row
        costs.append(row["cost"])
    if any(cost != costs[0] for cost in costs[1:]):
        raise RuntimeError("full-schedule convolutional circuit costs do not match")
    random = rows["random"]
    for method, row in rows.items():
        row["validation_gain_vs_random_pp"] = (
            row["best_hard_validation_pct"]
            - random["best_hard_validation_pct"]
        )
        row["test_gain_vs_random_pp"] = (
            row["test_hard_pct"] - random["test_hard_pct"]
        )
    payload = {
        "validation_freeze": str(FREEZE),
        "held_out_test_evaluated_once": True,
        "single_seed_resource_cohort": True,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
