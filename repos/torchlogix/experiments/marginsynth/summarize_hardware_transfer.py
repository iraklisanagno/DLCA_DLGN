#!/usr/bin/env python3
"""Aggregate frozen hardware-aware MarginSynth over dense CIFAR-10 seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.summarize_component_protocol import two_pass_record
from experiments.marginsynth.summarize_hardware_ablation import extended_hardware
from experiments.marginsynth.verify_checkpoint import sha256_file


METHODS = ("exact_simplification", "unit_tying_10pct", "hardware_marginsynth")
METRICS = (
    "validation_accuracy",
    "validation_accuracy_loss",
    "validation_disagreement",
    "validation_worst_class_accuracy_loss",
    "live_gates",
    "live_gate_reduction_vs_exact",
    "abc_and_nodes",
    "abc_node_reduction_vs_exact",
    "abc_levels",
    "yosys_generic_cells",
    "generic_cell_reduction_vs_exact",
    "sky130_operation_area_proxy_um2",
    "area_proxy_reduction_vs_exact",
    "optimization_seconds",
    "method_total_seconds",
)


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def protocol_seed(name: str) -> int:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", name)
    if match is None:
        raise ValueError(f"cannot recover seed from protocol {name!r}")
    return int(match.group(1))


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "values": values,
    }


def finite(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def add_reductions(records: list[dict]) -> None:
    exact = next(record for record in records if record["method"] == "exact_simplification")
    exact.setdefault("validation_worst_class_accuracy_loss", 0.0)
    for record in records:
        record["live_gate_reduction_vs_exact"] = 1.0 - record["live_gates"] / exact["live_gates"]
        record["abc_node_reduction_vs_exact"] = 1.0 - record["abc_and_nodes"] / exact["abc_and_nodes"]
        record["generic_cell_reduction_vs_exact"] = 1.0 - record["yosys_generic_cells"] / exact["yosys_generic_cells"]
        record["area_proxy_reduction_vs_exact"] = 1.0 - record["sky130_operation_area_proxy_um2"] / exact["sky130_operation_area_proxy_um2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-summary", required=True, type=Path)
    parser.add_argument("--transfer-protocol", action="append", default=[], type=Path)
    parser.add_argument("--reference-comparison", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()
    if len(cli.transfer_protocol) != len(cli.reference_comparison):
        raise ValueError("each transfer protocol needs one reference comparison")

    ablation_path = cli.ablation_summary.resolve()
    ablation = load(ablation_path)
    if ablation.get("test_used") is not False:
        raise ValueError("ablation used test data")
    selected_name = ablation["selected_component"]
    ablation_by_name = {record["method"]: dict(record) for record in ablation["records"]}
    seed0_records = [
        ablation_by_name["exact_simplification"],
        ablation_by_name["unit_tying_10pct"],
        ablation_by_name[selected_name],
    ]
    seed0_records[2]["method"] = "hardware_marginsynth"
    add_reductions(seed0_records)
    rows = []
    for record in seed0_records:
        rows.append({"seed": 0, **record})

    inputs = {
        "ablation": {"path": str(ablation_path), "sha256": sha256_file(ablation_path)},
        "transfer": [],
    }
    freeze_hash = None
    for protocol_raw, reference_raw in zip(cli.transfer_protocol, cli.reference_comparison):
        protocol_path = protocol_raw.resolve()
        reference_path = reference_raw.resolve()
        protocol = load(protocol_path)
        reference = load(reference_path)
        if protocol.get("profile") != "transfer":
            raise ValueError(f"not a transfer protocol: {protocol_path}")
        if protocol.get("data_policy", {}).get("test_used") is not False:
            raise ValueError("transfer protocol did not seal test data")
        if reference.get("test_used") is not False:
            raise ValueError("reference comparison did not seal test data")
        seed = protocol_seed(protocol["protocol_name"])
        if seed == 0:
            raise ValueError("seed 0 must come from the frozen ablation")
        current_freeze_hash = protocol["frozen_method"]["freeze_record_sha256"]
        if freeze_hash is None:
            freeze_hash = current_freeze_hash
        elif current_freeze_hash != freeze_hash:
            raise ValueError("transfer protocols use different freeze records")
        component_name = next(iter(protocol["two_pass_components"]))
        if component_name != selected_name:
            raise ValueError("transfer component differs from seed-0 selection")
        source_run = Path(protocol["source_run"])
        if not source_run.is_absolute():
            source_run = REPOSITORY_ROOT / source_run
        root = source_run / protocol["output_root"]
        component = root / component_name
        method = two_pass_record(
            component_name,
            source_run,
            component / "first_resynthesis",
            component / "second_resynthesis",
        )
        method.update(extended_hardware(component / "selected_snapshot"))
        method["method"] = "hardware_marginsynth"

        reference_by_name = {record["method"]: dict(record) for record in reference["records"]}
        exact = reference_by_name["exact_simplification"]
        unit = reference_by_name["unit_tying_10pct"]
        reference_root = reference_path.parent
        exact.update(extended_hardware(reference_root / "exact_baseline"))
        unit.update(extended_hardware(reference_root / "unit_tying" / "ratio_10"))
        records = [exact, unit, method]
        add_reductions(records)
        rows.extend({"seed": seed, **record} for record in records)
        inputs["transfer"].append(
            {
                "seed": seed,
                "protocol": str(protocol_path),
                "protocol_sha256": sha256_file(protocol_path),
                "reference": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
            }
        )

    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 3:
        raise ValueError("frozen transfer summary requires at least three seeds")
    rows.sort(key=lambda row: (row["seed"], METHODS.index(row["method"])))
    aggregates = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            metric: distribution(
                [finite(row[metric], f"{method} {metric}") for row in method_rows]
            )
            for metric in METRICS
        }
        if method == "hardware_marginsynth":
            aggregates[method]["all_guard_feasible"] = all(
                row["guard_feasible"] for row in method_rows
            )
            aggregates[method]["fallback_count"] = sum(
                row.get("snapshot_fallback_applied", False) for row in method_rows
            )

    by_seed = {
        seed: {row["method"]: row for row in rows if row["seed"] == seed}
        for seed in seeds
    }
    paired = {}
    for metric in METRICS:
        if metric in {"optimization_seconds"}:
            continue
        paired[f"{metric}_delta_marginsynth_minus_unit_tying"] = distribution(
            [
                finite(by_seed[seed]["hardware_marginsynth"][metric], metric)
                - finite(by_seed[seed]["unit_tying_10pct"][metric], metric)
                for seed in seeds
            ]
        )
    paired["method_total_time_ratio_marginsynth_over_unit_tying"] = distribution(
        [
            by_seed[seed]["hardware_marginsynth"]["method_total_seconds"]
            / by_seed[seed]["unit_tying_10pct"]["method_total_seconds"]
            for seed in seeds
        ]
    )
    payload = {
        "format_version": 1,
        "status": "completed",
        "dataset": "cifar-10",
        "architecture": "DlgnCifar10Medium",
        "seeds": seeds,
        "selected_component": selected_name,
        "freeze_record_sha256": freeze_hash,
        "methods": list(METHODS),
        "inputs": inputs,
        "aggregates": aggregates,
        "paired_marginsynth_vs_unit_tying": paired,
        "mapped_area_policy": ablation["mapped_area_policy"],
        "test_used": False,
    }
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hardware_transfer_summary.json"
    csv_path = output_dir / "hardware_transfer_per_seed.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
