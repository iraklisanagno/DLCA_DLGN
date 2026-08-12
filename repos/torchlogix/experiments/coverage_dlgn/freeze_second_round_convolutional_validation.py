#!/usr/bin/env python3
"""Freeze full-schedule convolutional S selections before test evaluation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "summary" / "second_round_convolutional_validation_freeze.json"

RUNS = {
    "random": "second_full_conv_cifar10_s_random_seed0",
    "legacy_v4": "second_full_conv_cifar10_s_legacy_v4_seed0",
    "unified_u1": "second_full_conv_cifar10_s_unified_u1_seed0",
    "unified_u2": "second_final_u2_conv_cifar10_s_seed0",
}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite validation freeze: {OUTPUT}")
    rows = {}
    for method, name in RUNS.items():
        run_dir = RESULTS / name
        required = {
            "run_summary": run_dir / "run_summary.json",
            "checkpoint": run_dir / "best_checkpoint.pt",
            "config": run_dir / "training_config.json",
            "environment": run_dir / "environment.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"cannot freeze {name}; missing {missing}")
        run_summary = json.loads(required["run_summary"].read_text())
        config = json.loads(required["config"].read_text())
        environment = json.loads(required["environment"].read_text())
        rows[method] = {
            "name": name,
            "run_dir": str(run_dir),
            "seed": config["seed"],
            "best_hard_validation_pct": (
                100 * run_summary["best_validation_hard_accuracy"]
            ),
            "best_checkpoint": str(required["checkpoint"]),
            "test_metrics_existing_at_freeze": (
                run_dir / "test_metrics.json"
            ).is_file(),
            "source_tree_sha256": environment.get("source_tree_sha256"),
            "training_implementation_sha256": environment.get(
                "training_implementation_sha256"
            ),
        }
    payload = {
        "architecture": "DlgnConvCifar10PaperSmall",
        "boolean_input_channels": 9,
        "training_updates": 350000,
        "selection": "best hardened validation checkpoint",
        "validation_frozen": True,
        "held_out_test_access_added_by_this_script": False,
        "single_seed_resource_cohort": True,
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
