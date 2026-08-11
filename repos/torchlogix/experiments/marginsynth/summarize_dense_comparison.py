#!/usr/bin/env python3
"""Create the paper-facing dense CIFAR method comparison and transfer gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.summarize_component_protocol import two_pass_record
from experiments.marginsynth.verify_checkpoint import sha256_file


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def hardware(export: dict) -> dict:
    return {
        "live_gates": export["live_gates"],
        "abc_and_nodes": export["abc_and_nodes"],
        "abc_levels": export["abc_levels"],
    }


def one_pass_record(name: str, method_dir: Path) -> dict:
    summary = load(method_dir / "summary.json")
    export = load(method_dir / "export_summary.json")
    guard = summary["guard_holdout"]
    validation = summary["validation"]
    return {
        "method": name,
        "family": "post-training-control",
        "guard_feasible": bool(summary["guard_holdout_feasible"]),
        "guard_accuracy": guard["accuracy"],
        "guard_accuracy_loss": guard["accuracy_loss"],
        "guard_disagreement": guard["decision_flip_rate"],
        "guard_worst_class_accuracy_loss": guard["maximum_per_class_accuracy_loss"],
        "guard_worst_class_disagreement": guard["maximum_per_class_disagreement"],
        "validation_accuracy": validation["accuracy"],
        "validation_accuracy_loss": validation["accuracy_loss"],
        "validation_disagreement": validation["decision_flip_rate"],
        "cumulative_retained_changes": summary["retained_changes"],
        "optimization_seconds": summary["timing"]["optimization_seconds"],
        "method_total_seconds": summary["timing"]["total_seconds"],
        "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
        **hardware(export),
        "checkpoint_sha256": export["checkpoint_sha256"],
        "test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load(protocol_path)
    root = source_run / protocol["output_root"]

    margin_records = []
    for name in protocol.get("two_pass_components", {}):
        component = root / name
        record = two_pass_record(
            name,
            source_run,
            component / "first_resynthesis",
            component / "second_resynthesis",
        )
        record["family"] = "marginsynth"
        margin_records.append(record)
    if not margin_records:
        raise ValueError("protocol contains no completed MarginSynth method")
    baseline_calibration = load(
        Path(margin_records[0]["first_dir"]) / "summary.json"
    )["baseline_calibration"]
    baseline_validation = load(
        Path(margin_records[0]["first_dir"]) / "summary.json"
    )["baseline_validation"]

    exact_export = load(root / "exact_baseline" / "export_summary.json")
    source_summary = load(source_run / "run_summary.json")
    exact = {
        "method": "exact_simplification",
        "family": "exact-baseline",
        "guard_feasible": True,
        "guard_accuracy": None,
        "guard_accuracy_loss": 0.0,
        "guard_disagreement": 0.0,
        "guard_worst_class_accuracy_loss": 0.0,
        "guard_worst_class_disagreement": 0.0,
        "validation_accuracy": source_summary["best_validation_hard_accuracy"],
        "validation_accuracy_loss": 0.0,
        "validation_disagreement": 0.0,
        "cumulative_retained_changes": 0,
        "optimization_seconds": 0.0,
        "method_total_seconds": 0.0,
        "peak_gpu_memory_bytes": 0,
        **hardware(exact_export),
        "checkpoint_sha256": exact_export["checkpoint_sha256"],
        "test_used": False,
    }
    records = [exact]

    if "unit_tying" in protocol:
        unit_config = protocol["unit_tying"]
        ratio = float(unit_config["selected_ratio"])
        ratio_name = f"ratio_{int(round(100 * ratio)):02d}"
        unit_dir = source_run / unit_config["output"] / ratio_name
        unit = load(unit_dir / "summary.json")
        unit_export = load(unit_dir / "export_summary.json")
        unit_validation = unit["validation"]
        unit_calibration = unit["calibration"]
        base_per_class = baseline_validation["per_class_accuracy"]
        unit_per_class = unit_validation["per_class_accuracy"]
        records.append(
            {
                "method": "unit_tying_10pct",
                "family": "published-baseline",
                "guard_feasible": None,
                "guard_accuracy": None,
                "guard_accuracy_loss": None,
                "guard_disagreement": None,
                "guard_worst_class_accuracy_loss": None,
                "guard_worst_class_disagreement": None,
                "calibration_accuracy": unit_calibration["accuracy"],
                "calibration_accuracy_loss": unit["baseline_calibration_accuracy"] - unit_calibration["accuracy"],
                "calibration_disagreement": unit_calibration["decision_flip_rate"],
                "validation_accuracy": unit_validation["accuracy"],
                "validation_accuracy_loss": unit["baseline_validation_accuracy"] - unit_validation["accuracy"],
                "validation_disagreement": unit_validation["decision_flip_rate"],
                "validation_worst_class_accuracy_loss": max(
                    base - candidate
                    for base, candidate in zip(base_per_class, unit_per_class)
                ),
                "cumulative_retained_changes": unit["newly_constant_tied_units"],
                "optimization_seconds": unit["wall_seconds"],
                "method_total_seconds": unit["wall_seconds"],
                "peak_gpu_memory_bytes": None,
                **hardware(unit_export),
                "checkpoint_sha256": unit_export["checkpoint_sha256"],
                "test_used": False,
            }
        )
    records.extend(margin_records)
    if "silicon_control" in protocol:
        records.append(one_pass_record("silicon_aware_post_training", root / "silicon_control"))

    for record in records:
        record["live_gate_reduction_vs_exact"] = 1.0 - record["live_gates"] / exact["live_gates"]
        record["abc_node_reduction_vs_exact"] = 1.0 - record["abc_and_nodes"] / exact["abc_and_nodes"]
        unit_record = next((item for item in records if item["method"] == "unit_tying_10pct"), None)
        record["abc_node_delta_vs_unit_tying"] = (
            None if unit_record is None
            else record["abc_and_nodes"] - unit_record["abc_and_nodes"]
        )

    by_name = {record["method"]: record for record in records}
    current = by_name.get("current")
    liveness = by_name.get("liveness")
    criterion = protocol.get("continuation_criterion", {})
    speedup_fraction = None
    if current is not None and liveness is not None and current["method_total_seconds"]:
        speedup_fraction = 1.0 - liveness["method_total_seconds"] / current["method_total_seconds"]
    selected = by_name.get(protocol.get("selected_component"))
    threshold = float(criterion.get("minimum_liveness_method_speedup_fraction", 0.10))
    transfer_recommended = bool(
        selected is not None
        and selected["guard_feasible"]
        and liveness is not None
        and liveness["guard_feasible"]
        and speedup_fraction is not None
        and speedup_fraction >= threshold
    )
    payload = {
        "format_version": 1,
        "status": "completed",
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": "cifar-10",
        "architecture": load(source_run / "training_config.json")["architecture"],
        "source_checkpoint_sha256": sha256_file(source_run / "best_checkpoint.pt"),
        "baseline_calibration": baseline_calibration,
        "baseline_validation": baseline_validation,
        "records": records,
        "continuation": {
            "criterion": criterion,
            "selected_component": protocol.get("selected_component"),
            "selected_method_guard_feasible": None if selected is None else selected["guard_feasible"],
            "liveness_guard_feasible": None if liveness is None else liveness["guard_feasible"],
            "liveness_method_speedup_fraction": speedup_fraction,
            "transfer_to_seeds_1_and_2_recommended": transfer_recommended,
        },
        "test_used": False,
    }
    json_path = root / "dense_comparison.json"
    csv_path = root / "dense_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    scalar_keys = sorted(
        {key for record in records for key, value in record.items() if not isinstance(value, (dict, list))}
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows(
            {key: record.get(key) for key in scalar_keys} for record in records
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
