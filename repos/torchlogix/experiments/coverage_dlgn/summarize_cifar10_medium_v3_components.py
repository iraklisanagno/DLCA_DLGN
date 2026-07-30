#!/usr/bin/env python3
"""Summarize the paired CIFAR-10 M frozen-V3 component ablation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols" / "cifar10_medium_v3_components.json"
QUEUE = ROOT / "queues" / "cifar10_medium_v3_components.json"
QUEUE_SUMMARY = ROOT / "logs" / "cifar10_medium_v3_components" / "queue_summary.json"
JSON_OUT = ROOT / "summary" / "cifar10_medium_v3_components.json"
CSV_OUT = ROOT / "summary" / "cifar10_medium_v3_components.csv"
T_DF2 = 4.302652729696142


def row(family: str, seed: int, run_dir: Path) -> dict:
    if (run_dir / "test_metrics.json").exists():
        raise RuntimeError(f"component ablation touched held-out test: {run_dir}")
    summary = json.loads((run_dir / "run_summary.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    if config["architecture"] != "DlgnCifar10Medium":
        raise RuntimeError(f"unexpected component architecture: {run_dir}")
    cost = summary.get("cost")
    cost_recovered_from_architecture = cost is None
    if cost is None:
        # The original 20K controls predate run-summary cost accounting.
        # These constants are audited against the newly trained arms of the
        # exact same architecture and against the analytic rank-2 count.
        cost = {
            "dense_gate_count": 512_000,
            "trainable_parameters": 8_192_000,
            "deployed_routing_bits": 16_640_000,
        }
    return {
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_validation_hard_accuracy": summary[
            "best_validation_hard_accuracy"
        ],
        "wall_seconds": summary["wall_seconds"],
        "topology_construction_seconds": sum(
            layer["construction_seconds"] for layer in summary["topology"]
        ),
        "dense_gate_count": cost["dense_gate_count"],
        "trainable_parameters": cost["trainable_parameters"],
        "deployed_routing_bits": cost["deployed_routing_bits"],
        "cost_recovered_from_architecture": cost_recovered_from_architecture,
        "connections_init_method": config["connections_init_method"],
        "coverage_swap_fraction": config.get("coverage_swap_fraction"),
    }


def paired(rows: list[dict], left: str, right: str) -> dict:
    indexed = {(item["family"], item["seed"]): item for item in rows}
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
        "per_seed_gain_pp": [100 * value for value in gains],
        "paired_mean_gain_pp": 100 * mean,
        "paired_95ci_pp": [100 * (mean - half), 100 * (mean + half)],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    queue = json.loads(QUEUE.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"component queue failures: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(entry["name"] for entry in audit["finished"])
    if expected != observed:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )

    rows = []
    patterns = {
        "random": "pilot_cifar10_medium_random_v3_seed{seed}",
        "coverage_v3": (
            "pilot_cifar10_medium_semantic_balanced_v3_seed{seed}"
        ),
        "balanced_butterfly": (
            "ablate_cifar10_medium_balanced_butterfly_seed{seed}"
        ),
        "semantic_first_no_swaps": (
            "ablate_cifar10_medium_semantic_first_no_swaps_seed{seed}"
        ),
    }
    for family, pattern in patterns.items():
        for seed in protocol["training"]["paired_seeds"]:
            rows.append(row(
                family, seed, ROOT / "results" / pattern.format(seed=seed)
            ))
    costs = {
        (
            item["dense_gate_count"],
            item["trainable_parameters"],
            item["deployed_routing_bits"],
        )
        for item in rows
    }
    if len(costs) != 1:
        raise RuntimeError(f"component cost mismatch: {costs}")

    families = list(patterns)
    stats = {}
    for family in families:
        values = [
            item["best_validation_hard_accuracy"]
            for item in rows if item["family"] == family
        ]
        stats[family] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "n": len(values),
        }
    effects = {
        "balanced_fanout": paired(rows, "random", "balanced_butterfly"),
        "semantic_first_layer": paired(
            rows, "balanced_butterfly", "semantic_first_no_swaps"
        ),
        "ancestry_swaps": paired(
            rows, "semantic_first_no_swaps", "coverage_v3"
        ),
        "complete_v3": paired(rows, "random", "coverage_v3"),
    }
    payload = {
        "phase": protocol["phase"],
        "provenance": "TRIED",
        "test_set_used": False,
        "controls_reused_not_rerun": True,
        "family_stats": stats,
        "paired_component_effects": effects,
        "common_cost": {
            "dense_gate_count": rows[0]["dense_gate_count"],
            "trainable_parameters": rows[0]["trainable_parameters"],
            "deployed_routing_bits": rows[0]["deployed_routing_bits"],
        },
        "rows": sorted(rows, key=lambda item: (item["family"], item["seed"])),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(JSON_OUT)
    for name, effect in effects.items():
        print(f"{name}: {effect['paired_mean_gain_pp']:+.3f} pp")


if __name__ == "__main__":
    main()
