#!/usr/bin/env python3
"""Consolidate matched 350K convolutional validation learning curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_OUTPUT = ROOT / "summary" / "second_round_convolutional_curves.csv"
JSON_OUTPUT = ROOT / "summary" / "second_round_convolutional_curves.json"

RUNS = {
    "random": "second_full_conv_cifar10_s_random_seed0",
    "legacy_v4": "second_full_conv_cifar10_s_legacy_v4_seed0",
    "unified_u1": "second_full_conv_cifar10_s_unified_u1_seed0",
    "unified_u2": "second_final_u2_conv_cifar10_s_seed0",
}
THRESHOLDS_PCT = (58.0, 59.0, 59.5)


def read_curve(name: str) -> list[dict]:
    path = RESULTS / name / "metrics.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = [
        {
            "step": int(row["step"]),
            "hard_validation_pct": 100 * float(row["val_acc_discrete"]),
            "relaxed_validation_pct": 100 * float(row["val_acc_relaxed"]),
            "train_loss": float(row["train_loss"]),
        }
        for row in rows
    ]
    if not parsed or parsed[-1]["step"] != 350000:
        raise RuntimeError(f"incomplete 350K curve: {path}")
    return parsed


def trapezoid_mean(rows: list[dict], key: str) -> float:
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        width = right["step"] - left["step"]
        area += width * (left[key] + right[key]) / 2
    return area / (rows[-1]["step"] - rows[0]["step"])


def main() -> None:
    curves = {method: read_curve(name) for method, name in RUNS.items()}
    reference_steps = [row["step"] for row in curves["random"]]
    for method, rows in curves.items():
        if [row["step"] for row in rows] != reference_steps:
            raise RuntimeError(f"evaluation steps do not match: {method}")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="") as handle:
        fieldnames = [
            "method",
            "step",
            "hard_validation_pct",
            "relaxed_validation_pct",
            "train_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for method, rows in curves.items():
            for row in rows:
                writer.writerow({"method": method, **row})

    summaries = {}
    for method, rows in curves.items():
        summaries[method] = {
            "best_hard_validation_pct": max(
                row["hard_validation_pct"] for row in rows
            ),
            "final_hard_validation_pct": rows[-1]["hard_validation_pct"],
            "hard_validation_curve_mean_pct": trapezoid_mean(
                rows, "hard_validation_pct"
            ),
            "relaxed_validation_curve_mean_pct": trapezoid_mean(
                rows, "relaxed_validation_pct"
            ),
            "first_step_at_hard_validation_pct": {
                str(threshold): next(
                    (
                        row["step"] for row in rows
                        if row["hard_validation_pct"] >= threshold
                    ),
                    None,
                )
                for threshold in THRESHOLDS_PCT
            },
        }
    payload = {
        "protocol": "matched seed-0, 350K, best hardened validation selection",
        "heldout_test_accessed": False,
        "summaries": summaries,
        "curve_csv": str(CSV_OUTPUT),
    }
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(JSON_OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
