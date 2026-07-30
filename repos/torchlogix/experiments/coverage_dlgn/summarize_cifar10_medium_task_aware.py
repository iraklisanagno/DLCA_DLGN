#!/usr/bin/env python3
"""Audit the task-aware CIFAR-10 M pilot and apply its frozen promotion gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols" / "cifar10_medium_task_aware.json"
QUEUE = ROOT / "queues" / "cifar10_medium_task_aware.json"
QUEUE_SUMMARY = ROOT / "logs" / "cifar10_medium_task_aware" / "queue_summary.json"
JSON_OUT = ROOT / "summary" / "cifar10_medium_task_aware.json"
CSV_OUT = ROOT / "summary" / "cifar10_medium_task_aware.csv"
T_DF2 = 4.302652729696142


def load_row(family: str, seed: int, run_dir: Path) -> dict:
    if (run_dir / "test_metrics.json").exists():
        raise RuntimeError(f"task-aware pilot touched held-out test: {run_dir}")
    run = json.loads((run_dir / "run_summary.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    report_path = run_dir / "task_aware_rewire.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    cost = run.get("cost")
    cost_recovered_from_architecture = cost is None
    if cost is None:
        cost = {
            "dense_gate_count": 512_000,
            "trainable_parameters": 8_192_000,
            "training_routing_parameters": 0,
            "deployed_routing_bits": 16_640_000,
        }
    return {
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_validation_hard_accuracy": run[
            "best_validation_hard_accuracy"
        ],
        "wall_seconds": run["wall_seconds"],
        "dense_gate_count": cost["dense_gate_count"],
        "trainable_parameters": cost["trainable_parameters"],
        "training_routing_parameters": cost[
            "training_routing_parameters"
        ],
        "deployed_routing_bits": cost["deployed_routing_bits"],
        "cost_recovered_from_architecture": cost_recovered_from_architecture,
        "connections_init_method": config["connections_init_method"],
        "task_aware_rewire_step": config.get("task_aware_rewire_step"),
        "task_aware_changed_gates": (
            sum(layer["changed_gates"] for layer in report["layers"])
            if report else 0
        ),
        "task_aware_accepted_swaps": (
            sum(layer["accepted_swaps"] for layer in report["layers"])
            if report else 0
        ),
        "task_aware_seconds": (
            sum(layer["construction_seconds"] for layer in report["layers"])
            if report else 0.0
        ),
    }


def paired(rows: list[dict], left: str, right: str) -> dict:
    indexed = {(row["family"], row["seed"]): row for row in rows}
    gains = [
        indexed[(right, seed)]["best_validation_hard_accuracy"]
        - indexed[(left, seed)]["best_validation_hard_accuracy"]
        for seed in (0, 1, 2)
    ]
    mean = statistics.mean(gains)
    std = statistics.stdev(gains)
    half = T_DF2 * std / math.sqrt(3)
    return {
        "left": left,
        "right": right,
        "per_seed_gain_pp": [100 * gain for gain in gains],
        "paired_mean_gain_pp": 100 * mean,
        "paired_95ci_pp": [100 * (mean - half), 100 * (mean + half)],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    queue = json.loads(QUEUE.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"task-aware queue failures: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(entry["name"] for entry in audit["finished"])
    if expected != observed:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )

    patterns = {
        "random": "pilot_cifar10_medium_random_v3_seed{seed}",
        "coverage_v3": (
            "pilot_cifar10_medium_semantic_balanced_v3_seed{seed}"
        ),
        "coverage_v3_task_aware": (
            "pilot_cifar10_medium_v3_task_aware_seed{seed}"
        ),
    }
    rows = []
    for family, pattern in patterns.items():
        for seed in protocol["pilot"]["paired_seeds"]:
            rows.append(load_row(
                family, seed, ROOT / "results" / pattern.format(seed=seed)
            ))
    for seed in protocol["pilot"]["paired_seeds"]:
        control_dir = (
            ROOT / "results"
            / f"pilot_cifar10_medium_semantic_balanced_v3_seed{seed}"
        )
        checkpoint = torch.load(
            control_dir / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state_dict"]
        index_keys = sorted(
            (key for key in state if key.endswith("connections.indices")),
            key=lambda key: int(key.split(".", 1)[0]),
        )
        expected_hashes = [
            hashlib.sha256(
                np.ascontiguousarray(state[key].numpy()).tobytes()
            ).hexdigest()
            for key in index_keys
        ]
        task_dir = (
            ROOT / "results"
            / f"pilot_cifar10_medium_v3_task_aware_seed{seed}"
        )
        report = json.loads(
            (task_dir / "task_aware_rewire.json").read_text()
        )
        observed_hashes = [
            layer["before_indices_sha256"] for layer in report["layers"]
        ]
        if observed_hashes != expected_hashes:
            raise RuntimeError(
                f"task-aware seed {seed} did not start from frozen V3"
            )
    costs = {
        (
            row["dense_gate_count"],
            row["trainable_parameters"],
            row["training_routing_parameters"],
            row["deployed_routing_bits"],
        )
        for row in rows
    }
    if len(costs) != 1:
        raise RuntimeError(f"task-aware cost mismatch: {costs}")
    stats = {}
    for family in patterns:
        values = [
            row["best_validation_hard_accuracy"]
            for row in rows if row["family"] == family
        ]
        stats[family] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "n": len(values),
        }
    versus_random = paired(rows, "random", "coverage_v3_task_aware")
    versus_v3 = paired(rows, "coverage_v3", "coverage_v3_task_aware")
    threshold = protocol["promotion"]
    promoted = (
        versus_random["paired_mean_gain_pp"]
        >= threshold["required_mean_gain_over_random_pp"]
        and versus_v3["paired_mean_gain_pp"]
        >= threshold["required_mean_gain_over_v3_pp"]
    )
    payload = {
        "phase": protocol["phase"],
        "provenance": "TRIED",
        "validation_metric": protocol["pilot"]["selection_metric"],
        "test_set_used": False,
        "controls_reused_not_rerun": True,
        "v3_modified": False,
        "frozen_v3_start_hashes_verified": True,
        "family_stats": stats,
        "task_aware_vs_random": versus_random,
        "task_aware_vs_v3": versus_v3,
        "promotion_thresholds_pp": {
            "over_random": threshold["required_mean_gain_over_random_pp"],
            "over_v3": threshold["required_mean_gain_over_v3_pp"],
        },
        "promoted": promoted,
        "decision": (
            "Promote to full schedule."
            if promoted
            else "Stop: no full schedule, held-out test, or transfer run."
        ),
        "common_cost": {
            "dense_gate_count": rows[0]["dense_gate_count"],
            "trainable_parameters": rows[0]["trainable_parameters"],
            "training_routing_parameters": rows[0][
                "training_routing_parameters"
            ],
            "deployed_routing_bits": rows[0]["deployed_routing_bits"],
        },
        "rows": sorted(rows, key=lambda row: (row["family"], row["seed"])),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(JSON_OUT)
    print(
        f"vs random {versus_random['paired_mean_gain_pp']:+.3f} pp; "
        f"vs V3 {versus_v3['paired_mean_gain_pp']:+.3f} pp; "
        f"promoted={promoted}"
    )


if __name__ == "__main__":
    main()
