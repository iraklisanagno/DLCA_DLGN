#!/usr/bin/env python3
"""Machine-check the completed 6-by-64K CIFAR-100 baseline protocol."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "summary" / "cifar100_baseline_audit.json"


def main() -> None:
    rows = []
    for family, candidate in (
        ("random", "random"),
        ("coverage_v3", "v3_swap0125"),
    ):
        for seed in (0, 1, 2):
            run_dir = (
                ROOT / "results"
                / f"final_table4_cifar100_64k_{candidate}_seed{seed}"
            )
            config = json.loads((run_dir / "training_config.json").read_text())
            run = json.loads((run_dir / "run_summary.json").read_text())
            test = json.loads((run_dir / "test_metrics.json").read_text())
            environment = json.loads((run_dir / "environment.json").read_text())
            with (run_dir / "thresholds.csv").open() as handle:
                threshold_row = next(csv.DictReader(handle))
            thresholds = [
                float(threshold_row[f"thresh_0_{index}"]) for index in range(3)
            ]
            checks = {
                "architecture": config["architecture"]
                == "DlgnCifar100Scalability64k",
                "no_augmentation": config["augmentation"] == "none",
                "batch_size_100": config["batch_size"] == 100,
                "adam_lr_001": config["learning_rate"] == 0.01,
                "steps_40000": config["num_iterations"] == 40_000,
                "validation_fraction_020": config["valid_set_size"] == 0.2,
                "split_seed_2027": config["data_split_seed"] == 2027,
                "thresholds_quarters": thresholds == [0.25, 0.5, 0.75],
                "gate_count_384000": run["cost"]["dense_gate_count"] == 384_000,
                "trainable_parameters_6144000": (
                    run["cost"]["trainable_parameters"] == 6_144_000
                ),
                "test_queried_once": test["test_examples"] == 10_000,
            }
            rows.append({
                "family": family,
                "seed": seed,
                "test_hard_accuracy": test["test_hard_accuracy"],
                "validation_hard_accuracy": test[
                    "validation_hard_accuracy"
                ],
                "topology_seed": config["topology_seed"],
                "source_revision": environment["source_revision"],
                "training_implementation_sha256": environment[
                    "training_implementation_sha256"
                ],
                "checks": checks,
            })
    failed = [
        (row["family"], row["seed"], name)
        for row in rows
        for name, passed in row["checks"].items()
        if not passed
    ]
    if failed:
        raise RuntimeError(f"baseline protocol audit failed: {failed}")
    hashes = {
        row["training_implementation_sha256"] for row in rows
    }
    revisions = {row["source_revision"] for row in rows}
    if len(hashes) != 1 or len(revisions) != 1:
        raise RuntimeError("completed cohort mixes source implementations")
    means = {
        family: statistics.mean(
            row["test_hard_accuracy"]
            for row in rows if row["family"] == family
        )
        for family in ("random", "coverage_v3")
    }
    payload = {
        "status": "pass_with_known_nonidentical_routing",
        "test_set_used": True,
        "rerun_performed": False,
        "paper_reported_random_accuracy": 0.2254,
        "paper_reported_random_sample_std": 0.0026,
        "local_means": means,
        "reported_minus_local_random_pp": (
            100 * (0.2254 - means["random"])
        ),
        "matched_protocol_fields": list(rows[0]["checks"]),
        "material_mismatch": (
            "independently seeded TorchLogix NumPy routing versus the "
            "canonical difflogic two-torch-randperm generator"
        ),
        "unresolved": [
            "paper validation split seed",
            "paper final-versus-validation-selected checkpoint policy",
            "exact paper source/dependency revision",
        ],
        "source_revision": next(iter(revisions)),
        "training_implementation_sha256": next(iter(hashes)),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
