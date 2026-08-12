#!/usr/bin/env python3
"""Freeze promoted dense validation selections before held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "summary" / "second_round_final_validation_freeze.json"


def seeds(prefix: str) -> list[str]:
    return [f"{prefix}{seed}" for seed in range(3)]


GROUPS = {
    "mnist_8k_random": seeds("second_compression_mnist_8k_random_seed"),
    "mnist_8k_v3": seeds("second_compression_mnist_8k_coverage_v3_seed"),
    "mnist_8k_u2": seeds("second_final_u2_mnist_8k_seed"),
    "fashion_16k_random": seeds(
        "second_compression_fashion_16k_random_seed"
    ),
    "fashion_16k_v3": seeds(
        "second_compression_fashion_16k_coverage_v3_seed"
    ),
    "fashion_16k_u2": seeds("second_final_u2_fashion_16k_seed"),
    "dense_cifar10_s_random": seeds("paper_cifar10_small_random_seed"),
    "dense_cifar10_s_v3": seeds(
        "paper_cifar10_small_semantic_balanced_v3_seed"
    ),
    "dense_cifar10_s_u2": seeds("second_final_u2_cifar10_s_seed"),
    "dense_cifar100_3x128k_random": [
        "pilot_table4_cifar100_384k_depth3_random_seed0",
        "second_c100_3x128k_random_seed1",
        "second_c100_3x128k_random_seed2",
    ],
    "dense_cifar100_3x128k_v3": [
        "pilot_table4_cifar100_384k_depth3_coverage_v3_seed0",
        "second_c100_3x128k_coverage_v3_seed1",
        "second_c100_3x128k_coverage_v3_seed2",
    ],
}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite validation freeze: {OUTPUT}")
    groups = {}
    for label, names in GROUPS.items():
        rows = []
        for name in names:
            run_dir = RESULTS / name
            required = [
                run_dir / "run_summary.json",
                run_dir / "best_checkpoint.pt",
                run_dir / "training_config.json",
                run_dir / "environment.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"cannot freeze {name}; missing {missing}")
            run_summary = json.loads(required[0].read_text())
            training_config = json.loads(required[2].read_text())
            environment = json.loads(required[3].read_text())
            rows.append({
                "name": name,
                "run_dir": str(run_dir),
                "seed": training_config["seed"],
                "best_hard_validation_pct": (
                    100 * run_summary["best_validation_hard_accuracy"]
                ),
                "best_checkpoint": str(run_dir / "best_checkpoint.pt"),
                "test_metrics_existing_at_freeze": (
                    run_dir / "test_metrics.json"
                ).is_file(),
                "source_tree_sha256": environment.get("source_tree_sha256"),
                "training_implementation_sha256": environment.get(
                    "training_implementation_sha256"
                ),
            })
        groups[label] = rows
    payload = {
        "selection": "best hardened validation checkpoint",
        "validation_frozen": True,
        "held_out_test_access_added_by_this_script": False,
        "groups": groups,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
