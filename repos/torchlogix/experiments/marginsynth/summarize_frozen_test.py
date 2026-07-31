#!/usr/bin/env python3
"""Aggregate an already-completed frozen test evaluation without reloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.summarize_paired_study import (
    exact_bootstrap_mean_ci,
    summarize,
)
from experiments.marginsynth.verify_checkpoint import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    raw = json.loads(cli.input.read_text())
    if raw.get("status") != "completed" or not raw.get("test_used"):
        raise RuntimeError("input is not a completed frozen test evaluation")
    records = raw["seeds"]
    methods = ["method"]
    if all("unit_tying" in record for record in records):
        methods.append("unit_tying")
    metrics = (
        "accuracy",
        "accuracy_loss",
        "decision_flip_rate",
        "maximum_per_class_disagreement",
    )
    aggregate = {
        "baseline_accuracy": summarize(
            [float(record["baseline_accuracy"]) for record in records]
        )
    }
    for method in methods:
        aggregate[method] = {
            metric: summarize(
                [float(record[method][metric]) for record in records]
            )
            for metric in metrics
        }
    paired = {}
    if "unit_tying" in methods:
        for metric in metrics:
            differences = [
                float(record["method"][metric])
                - float(record["unit_tying"][metric])
                for record in records
            ]
            paired[metric] = summarize(differences)
            paired[metric]["exact_bootstrap_mean_95_ci"] = (
                exact_bootstrap_mean_ci(differences)
            )
    payload = {
        "format_version": 1,
        "status": "completed",
        "data_partition": "test",
        "test_used": True,
        "test_opened_only_after_freeze": raw[
            "test_opened_only_after_freeze"
        ],
        "raw_result": str(cli.input),
        "raw_result_sha256": sha256_file(cli.input),
        "protocol_freeze": raw["protocol_freeze"],
        "protocol_freeze_sha256": raw["protocol_freeze_sha256"],
        "seeds": [int(record["seed"]) for record in records],
        "aggregate": aggregate,
        "paired_marginsynth_minus_unit_tying": paired,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
