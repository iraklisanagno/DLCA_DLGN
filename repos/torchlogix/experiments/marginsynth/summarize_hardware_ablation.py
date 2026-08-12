#!/usr/bin/env python3
"""Summarize seed-0 hardware-ranking ablations and freeze one method."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.cost_model import circuit_features
from experiments.marginsynth.summarize_component_protocol import two_pass_record
from experiments.marginsynth.verify_checkpoint import sha256_file
from torchlogix import Circuit


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def extended_hardware(method_dir: Path) -> dict:
    export_path = method_dir / "export_summary.json"
    export = load(export_path)
    synthesis_path = export.get("reused_synthesis_verification")
    if synthesis_path is None:
        synthesis_path = method_dir / "export_run" / "synthesis_verification.json"
    else:
        synthesis_path = Path(synthesis_path)
    synthesis = load(synthesis_path)
    area = export.get("sky130_operation_area_proxy_um2")
    if area is None:
        circuit_path = Path(export["export_dir"]) / "exact_simplified_circuit.json"
        circuit = Circuit.from_json_file(str(circuit_path))
        area = circuit_features(circuit)["sky130_operation_area_proxy_um2"]
    return {
        "live_gates": int(export["live_gates"]),
        "abc_and_nodes": int(export["abc_and_nodes"]),
        "abc_levels": int(export["abc_levels"]),
        "yosys_generic_cells": int(synthesis["yosys"]["cells"]["total"]),
        "sky130_operation_area_proxy_um2": float(area),
        "mapped_area": export.get("mapped_area"),
        "mapped_area_available": export.get("mapped_area") is not None,
        "export_summary_sha256": sha256_file(export_path),
    }


def correlations(records: list[dict]) -> dict:
    predictions = [0.0] + [
        float(record["estimated_hardware_gain"]) for record in records
    ]
    exact_nodes = records[0]["exact_abc_and_nodes"]
    observed = [0.0] + [
        float(exact_nodes - record["abc_and_nodes"]) for record in records
    ]
    if len(set(predictions)) < 2 or len(set(observed)) < 2:
        pearson = spearman = None
    else:
        pearson = float(pearsonr(predictions, observed).statistic)
        spearman = float(spearmanr(predictions, observed).statistic)
    return {
        "n": len(predictions),
        "predicted_hardware_gain": predictions,
        "observed_abc_node_reduction": observed,
        "pearson": pearson,
        "spearman": spearman,
        "minimum_spearman_for_transfer": 0.5,
        "validated_for_transfer": bool(
            spearman is not None and math.isfinite(spearman) and spearman >= 0.5
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--reference-comparison", required=True, type=Path)
    parser.add_argument("--freeze-output", required=True, type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load(protocol_path)
    reference_path = cli.reference_comparison.resolve()
    reference = load(reference_path)
    if reference.get("test_used") is not False:
        raise ValueError("reference comparison did not keep test data sealed")
    if protocol.get("profile") != "ablation":
        raise ValueError("protocol is not a hardware ablation")
    root = source_run / protocol["output_root"]

    reference_by_name = {record["method"]: record for record in reference["records"]}
    exact = dict(reference_by_name["exact_simplification"])
    unit = dict(reference_by_name["unit_tying_10pct"])
    current = dict(reference_by_name["current"])
    current["method"] = "current_reference"
    reference_root = reference_path.parent
    exact.update(extended_hardware(reference_root / "exact_baseline"))
    unit.update(extended_hardware(reference_root / "unit_tying" / "ratio_10"))
    current.update(extended_hardware(Path(reference_by_name["current"]["second_dir"])))

    new_records = []
    for name in protocol["two_pass_components"]:
        component = root / name
        record = two_pass_record(
            name,
            source_run,
            component / "first_resynthesis",
            component / "second_resynthesis",
        )
        record.update(extended_hardware(component / "selected_snapshot"))
        record["exact_abc_and_nodes"] = exact["abc_and_nodes"]
        new_records.append(record)

    margin_records = [current] + new_records
    feasible = [record for record in margin_records if record["guard_feasible"]]
    selected = min(
        feasible,
        key=lambda record: (
            record["abc_and_nodes"],
            record["guard_worst_class_accuracy_loss"],
            record["guard_accuracy_loss"],
            record["method_total_seconds"],
            record["method"],
        ),
    ) if feasible else None
    validation = correlations(new_records)
    for record in [exact, unit] + margin_records:
        record["abc_node_reduction_vs_exact"] = (
            1.0 - record["abc_and_nodes"] / exact["abc_and_nodes"]
        )
        record["live_gate_reduction_vs_exact"] = (
            1.0 - record["live_gates"] / exact["live_gates"]
        )
        record["generic_cell_reduction_vs_exact"] = (
            1.0 - record["yosys_generic_cells"] / exact["yosys_generic_cells"]
        )
        record["area_proxy_reduction_vs_exact"] = (
            1.0
            - record["sky130_operation_area_proxy_um2"]
            / exact["sky130_operation_area_proxy_um2"]
        )

    payload = {
        "format_version": 1,
        "status": "completed",
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": sha256_file(protocol_path),
        "reference_comparison": str(reference_path),
        "reference_comparison_sha256": sha256_file(reference_path),
        "selection_rule": protocol["selection_rule"],
        "selected_component": None if selected is None else selected["method"],
        "feasible_marginsynth_components": [
            record["method"] for record in feasible
        ],
        "hardware_estimator_validation": validation,
        "records": [exact, unit] + margin_records,
        "mapped_area_policy": (
            "No characterized common liberty library is installed. Generic Yosys "
            "cell count and the explicitly labeled SkyWater operation-area proxy "
            "are reported; mapped_area remains null."
        ),
        "test_used": False,
    }
    summary_path = root / "hardware_ablation_summary.json"
    csv_path = root / "hardware_ablation_summary.csv"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    scalar_keys = sorted(
        {
            key
            for record in payload["records"]
            for key, value in record.items()
            if not isinstance(value, (dict, list))
        }
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows(
            {key: record.get(key) for key in scalar_keys}
            for record in payload["records"]
        )

    if cli.freeze_output.exists():
        raise RuntimeError(f"refusing to overwrite freeze: {cli.freeze_output}")
    freeze_status = (
        "frozen"
        if selected is not None
        and selected["method"] != "current_reference"
        and validation["validated_for_transfer"]
        else "not-frozen"
    )
    freeze = {
        "format_version": 1,
        "status": freeze_status,
        "selected_component": None if selected is None else selected["method"],
        "selection_rule": protocol["selection_rule"],
        "ablation_summary": str(summary_path),
        "ablation_summary_sha256": sha256_file(summary_path),
        "ablation_protocol_sha256": sha256_file(protocol_path),
        "hardware_ranking_model_sha256": protocol["frozen_method"][
            "hardware_ranking_model_sha256"
        ],
        "hardware_estimator_validated": validation["validated_for_transfer"],
        "validation_used_for_selection": False,
        "test_used": False,
        "reason_if_not_frozen": (
            None
            if freeze_status == "frozen"
            else "Transfer requires a new component to beat the current reference under the frozen exact-ABC rule and structural-estimator Spearman >= 0.5."
        ),
    }
    cli.freeze_output.parent.mkdir(parents=True, exist_ok=True)
    cli.freeze_output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
