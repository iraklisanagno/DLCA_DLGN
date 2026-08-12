#!/usr/bin/env python3
"""Aggregate frozen dense-CIFAR comparisons across topology/training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

from experiments.marginsynth.verify_checkpoint import sha256_file


METHODS = ("exact_simplification", "unit_tying_10pct", "current")
METRICS = (
    "validation_accuracy",
    "validation_accuracy_loss",
    "validation_disagreement",
    "live_gates",
    "live_gate_reduction_vs_exact",
    "abc_and_nodes",
    "abc_node_reduction_vs_exact",
    "abc_levels",
    "optimization_seconds",
    "method_total_seconds",
)


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def seed_from_protocol(protocol_name: str) -> int:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", protocol_name)
    if match is None:
        raise ValueError(f"cannot recover seed from protocol name: {protocol_name!r}")
    return int(match.group(1))


def distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "values": values,
    }


def finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def aggregate(comparison_paths: list[Path]) -> tuple[dict, list[dict]]:
    if len(comparison_paths) < 2:
        raise ValueError("at least two seed comparisons are required")
    per_seed = []
    architecture = None
    input_hashes = {}
    seen_seeds = set()
    for raw_path in comparison_paths:
        path = raw_path.resolve()
        comparison = load(path)
        if comparison.get("status") != "completed":
            raise ValueError(f"comparison is not completed: {path}")
        if comparison.get("test_used") is not False:
            raise ValueError(f"comparison must keep test data sealed: {path}")
        if comparison.get("dataset") != "cifar-10":
            raise ValueError(f"comparison is not CIFAR-10: {path}")
        seed = seed_from_protocol(comparison["protocol_name"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed}")
        seen_seeds.add(seed)
        if architecture is None:
            architecture = comparison["architecture"]
        elif comparison["architecture"] != architecture:
            raise ValueError("all comparisons must use the same architecture")
        by_method = {record["method"]: record for record in comparison["records"]}
        missing = set(METHODS) - set(by_method)
        if missing:
            raise ValueError(f"seed {seed} is missing methods: {sorted(missing)}")
        input_hashes[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
        for method in METHODS:
            record = by_method[method]
            if record.get("test_used") is not False:
                raise ValueError(f"seed {seed} method {method} used test data")
            row = {
                "seed": seed,
                "method": method,
                "guard_feasible": record.get("guard_feasible"),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "test_used": False,
            }
            for metric in METRICS:
                row[metric] = finite_number(
                    record[metric], field=f"seed {seed} {method} {metric}"
                )
            per_seed.append(row)
    per_seed.sort(key=lambda row: (row["seed"], METHODS.index(row["method"])))

    aggregates = {}
    for method in METHODS:
        rows = [row for row in per_seed if row["method"] == method]
        aggregates[method] = {
            metric: distribution([row[metric] for row in rows]) for metric in METRICS
        }
        applicable = [
            row["guard_feasible"] for row in rows if row["guard_feasible"] is not None
        ]
        aggregates[method]["guard_feasibility_applicable"] = bool(applicable)
        aggregates[method]["all_guard_feasible"] = (
            all(value is True for value in applicable) if applicable else None
        )

    rows_by_seed = {
        seed: {row["method"]: row for row in per_seed if row["seed"] == seed}
        for seed in sorted(seen_seeds)
    }
    paired_definitions = {
        "validation_accuracy_delta_current_minus_unit_tying": (
            "validation_accuracy",
            1.0,
        ),
        "validation_accuracy_loss_delta_current_minus_unit_tying": (
            "validation_accuracy_loss",
            1.0,
        ),
        "validation_disagreement_delta_current_minus_unit_tying": (
            "validation_disagreement",
            1.0,
        ),
        "live_gate_reduction_delta_current_minus_unit_tying": (
            "live_gate_reduction_vs_exact",
            1.0,
        ),
        "abc_node_reduction_delta_current_minus_unit_tying": (
            "abc_node_reduction_vs_exact",
            1.0,
        ),
        "abc_node_count_delta_current_minus_unit_tying": ("abc_and_nodes", 1.0),
    }
    paired = {}
    for name, (metric, scale) in paired_definitions.items():
        paired[name] = distribution(
            [
                scale
                * (
                    rows_by_seed[seed]["current"][metric]
                    - rows_by_seed[seed]["unit_tying_10pct"][metric]
                )
                for seed in sorted(seen_seeds)
            ]
        )
    paired["method_total_time_ratio_current_over_unit_tying"] = distribution(
        [
            rows_by_seed[seed]["current"]["method_total_seconds"]
            / rows_by_seed[seed]["unit_tying_10pct"]["method_total_seconds"]
            for seed in sorted(seen_seeds)
        ]
    )

    payload = {
        "format_version": 1,
        "status": "completed",
        "dataset": "cifar-10",
        "architecture": architecture,
        "seeds": sorted(seen_seeds),
        "methods": list(METHODS),
        "comparison_inputs": input_hashes,
        "aggregates": aggregates,
        "paired_current_vs_unit_tying": paired,
        "interpretation_units": {
            "accuracy_disagreement_and_reduction": "fraction; multiply by 100 for percentage points",
            "gate_and_node_counts": "count",
            "time": "seconds",
        },
        "test_used": False,
    }
    return payload, per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()
    payload, rows = aggregate(cli.comparison)
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dense_transfer_summary.json"
    csv_path = output_dir / "dense_transfer_per_seed.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
