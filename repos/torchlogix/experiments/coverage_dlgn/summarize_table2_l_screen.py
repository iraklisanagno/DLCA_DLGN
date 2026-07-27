#!/usr/bin/env python3
"""Audit and summarize the reduced one-seed CIFAR-10 L screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_PATH = ROOT / "queues" / "table2_l_screen.json"
QUEUE_SUMMARY = ROOT / "logs" / "table2_l_screen" / "queue_summary.json"
JSON_PATH = ROOT / "summary" / "table2_l_screen.json"
CSV_PATH = ROOT / "summary" / "table2_l_screen.csv"


def main() -> None:
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"CIFAR-10 L screen failed: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != expected:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rows = []
    for entry in queue["entries"]:
        run_dir = Path(entry["output"])
        run = json.loads((run_dir / "run_summary.json").read_text())
        config = json.loads((run_dir / "training_config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"screen touched held-out test: {run_dir}")
        row = {
            "name": entry["name"],
            "family": entry["family"],
            "candidate": entry["candidate"],
            "parametrization": config["parametrization"],
            "seed": config["seed"],
            "best_validation_hard_accuracy": (
                run["best_validation_hard_accuracy"]
            ),
            "wall_seconds": run["wall_seconds"],
            "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
            "dense_gate_count": run["cost"]["dense_gate_count"],
            "coverage_candidate_pool_size": config.get(
                "coverage_candidate_pool_size"
            ),
            "coverage_swap_fraction": config.get("coverage_swap_fraction"),
            "source_revision": environment["source_revision"],
            "training_implementation_sha256": environment[
                "training_implementation_sha256"
            ],
        }
        rows.append(row)

    random_raw = next(
        row for row in rows
        if row["family"] == "random" and row["parametrization"] == "raw"
    )
    coverage_raw = [
        row for row in rows
        if row["family"] == "coverage_v3"
        and row["parametrization"] == "raw"
    ]
    coverage_warp = next(
        row for row in rows
        if row["family"] == "coverage_v3"
        and row["parametrization"] == "warp"
    )
    raw_winner = max(
        coverage_raw, key=lambda row: row["best_validation_hard_accuracy"]
    )
    overall_winner = max(
        (row for row in rows if row["family"] == "coverage_v3"),
        key=lambda row: row["best_validation_hard_accuracy"],
    )
    hashes = {row["training_implementation_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"mixed training implementations: {hashes}")

    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": False,
        "run_count": len(rows),
        "architecture": queue["architecture"],
        "raw_random": random_raw["name"],
        "selected_raw_v3": raw_winner["name"],
        "selected_combined_v3": overall_winner["name"],
        "selected_warp_v3": coverage_warp["name"],
        "advance_to_20k": {
            "raw": [random_raw["name"], raw_winner["name"]],
            "warp": [
                "matched random-topology WARP control generated for 20K",
                coverage_warp["name"],
            ],
        },
        "raw_screen_gain_pp": 100 * (
            raw_winner["best_validation_hard_accuracy"]
            - random_raw["best_validation_hard_accuracy"]
        ),
        "rows": sorted(rows, key=lambda row: row["name"]),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["name"]))
    print(JSON_PATH)
    print(
        f"raw winner={raw_winner['candidate']} "
        f"{100 * raw_winner['best_validation_hard_accuracy']:.3f}%"
    )
    print(
        f"combined winner={overall_winner['candidate']} "
        f"{100 * overall_winner['best_validation_hard_accuracy']:.3f}%"
    )


if __name__ == "__main__":
    main()
