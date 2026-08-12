#!/usr/bin/env python3
"""Summarize exact component outcomes and select a feasible scaling candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def repository_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def change_categories(
    method_dir: Path,
    first: dict,
    second: dict,
    selected_stage: str = "second",
) -> dict:
    counts = {"constant": 0, "routing_or_inversion": 0, "binary": 0}
    if selected_stage == "source":
        return counts
    directories = [(method_dir.parent / "first_resynthesis", first)]
    if selected_stage == "second":
        directories.append((method_dir, second))
    for directory, summary in directories:
        records = load(directory / "learned_changes.json")
        for record in records[: int(summary["retained_changes"])]:
            lut_id = int(record["new_lut"])
            if lut_id in {0, 15}:
                counts["constant"] += 1
            elif lut_id in {3, 5, 10, 12}:
                counts["routing_or_inversion"] += 1
            else:
                counts["binary"] += 1
    return counts


def two_pass_record(name: str, source_run: Path, first_dir: Path, second_dir: Path) -> dict:
    first = load(first_dir / "summary.json")
    second = load(second_dir / "summary.json")
    selection_path = second_dir.parent / "selected_snapshot" / "selection.json"
    selection = load(selection_path) if selection_path.exists() else None
    selected_stage = "second" if selection is None else selection["selected_stage"]
    export_dir = second_dir if selection is None else selection_path.parent
    export = load(export_dir / "export_summary.json")
    if selected_stage == "source":
        guard = {
            "accuracy": first["baseline_guard_holdout"]["accuracy"],
            "accuracy_loss": 0.0,
            "decision_flip_rate": 0.0,
            "maximum_per_class_accuracy_loss": 0.0,
            "maximum_per_class_disagreement": 0.0,
        }
        baseline_validation = first.get("baseline_validation")
        validation = None if baseline_validation is None else {
            "accuracy": baseline_validation["accuracy"],
            "accuracy_loss": 0.0,
            "decision_flip_rate": 0.0,
        }
        selected_summary = None
    else:
        selected_summary = first if selected_stage == "first" else second
        guard = selected_summary["guard_holdout"]
        validation = selected_summary.get("validation")
    cumulative_retained = {
        "source": 0,
        "first": int(first["retained_changes"]),
        "second": int(second["locked_source_changes"])
        + int(second["retained_changes"]),
    }[selected_stage]
    return {
        "method": name,
        "first_dir": str(first_dir),
        "second_dir": str(second_dir),
        "checkpoint_sha256": export["checkpoint_sha256"],
        "source_checkpoint_sha256": sha256_file(source_run / "best_checkpoint.pt"),
        "selected_stage": selected_stage,
        "snapshot_fallback_applied": selected_stage != "second",
        "snapshot_selection_sha256": (
            None if selection is None else sha256_file(selection_path)
        ),
        "estimated_hardware_gain": (
            None
            if selection is None
            else selection["selected_candidate"]["estimated_hardware_gain"]
        ),
        "guard_feasible": (
            True
            if selected_summary is None
            else bool(selected_summary["guard_holdout_feasible"])
        ),
        "calibration_feasible": (
            True
            if selected_summary is None
            else bool(selected_summary["calibration_feasible"])
        ),
        "guard_accuracy": guard["accuracy"],
        "guard_accuracy_loss": guard["accuracy_loss"],
        "guard_disagreement": guard["decision_flip_rate"],
        "guard_worst_class_accuracy_loss": guard["maximum_per_class_accuracy_loss"],
        "guard_worst_class_disagreement": guard["maximum_per_class_disagreement"],
        "validation_accuracy": None if validation is None else validation["accuracy"],
        "validation_accuracy_loss": None if validation is None else validation["accuracy_loss"],
        "validation_disagreement": None if validation is None else validation["decision_flip_rate"],
        "validation_worst_class_accuracy_loss": (
            None
            if validation is None
            else validation.get("maximum_per_class_accuracy_loss")
        ),
        "validation_worst_class_disagreement": (
            None
            if validation is None
            else validation.get("maximum_per_class_disagreement")
        ),
        "first_learned_changes": first["learned_changes"],
        "first_retained_changes": first["retained_changes"],
        "second_learned_changes": second["learned_changes"],
        "second_retained_changes": second["retained_changes"],
        "cumulative_retained_changes": cumulative_retained,
        "retained_action_categories": change_categories(
            second_dir, first, second, selected_stage
        ),
        "live_gates": export["live_gates"],
        "abc_and_nodes": export["abc_and_nodes"],
        "abc_levels": export["abc_levels"],
        "first_optimization_seconds": first["timing"]["optimization_seconds"],
        "second_optimization_seconds": second["timing"]["optimization_seconds"],
        "optimization_seconds": first["timing"]["optimization_seconds"] + second["timing"]["optimization_seconds"],
        "method_total_seconds": first["timing"]["total_seconds"] + second["timing"]["total_seconds"],
        "peak_gpu_memory_bytes": max(first["peak_gpu_memory_bytes"], second["peak_gpu_memory_bytes"]),
        "liveness_mask": second.get("liveness_mask", "none"),
        "activity_ranking": second.get("activity_ranking", "none"),
        "hardware_ranking_model_sha256": second.get(
            "hardware_ranking_model_sha256"
        ),
        "alternative_binary_penalty": second.get(
            "alternative_binary_penalty", 0.0
        ),
        "action_space": second["action_space"],
        "optimizable_logic_gates_first": first.get("optimizable_logic_gates", first["eligible_logic_gates"]),
        "optimizable_logic_gates_second": second.get("optimizable_logic_gates", second["eligible_logic_gates"]),
        "test_used": False,
    }


def nondominated(records: list[dict]) -> list[str]:
    result = []
    for candidate in records:
        dominated = any(
            other is not candidate
            and other["abc_and_nodes"] <= candidate["abc_and_nodes"]
            and other["guard_accuracy_loss"] <= candidate["guard_accuracy_loss"]
            and (
                other["abc_and_nodes"] < candidate["abc_and_nodes"]
                or other["guard_accuracy_loss"] < candidate["guard_accuracy_loss"]
            )
            for other in records
        )
        if not dominated:
            result.append(candidate["method"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load(protocol_path)
    root = source_run / protocol["output_root"]
    if not root.exists():
        raise FileNotFoundError(root)

    records = []
    current = protocol.get("current_reference")
    if current:
        records.append(
            two_pass_record(
                "current",
                source_run,
                source_run / current["first_resynthesis"],
                source_run / current["second_resynthesis"],
            )
        )
    for name in protocol.get("two_pass_components", {}):
        component = root / name
        records.append(
            two_pass_record(
                name,
                source_run,
                component / "first_resynthesis",
                component / "second_resynthesis",
            )
        )
    feasible = [record for record in records if record["guard_feasible"]]
    selected = min(
        feasible,
        key=lambda record: (
            record["abc_and_nodes"],
            record["guard_accuracy_loss"],
            record["method_total_seconds"],
            record["method"],
        ),
    ) if feasible else None
    current_record = next((record for record in records if record["method"] == "current"), None)
    for record in records:
        if current_record is not None:
            record["abc_node_delta_vs_current"] = record["abc_and_nodes"] - current_record["abc_and_nodes"]
            record["optimization_speedup_vs_current"] = (
                current_record["optimization_seconds"] / record["optimization_seconds"]
                if record["optimization_seconds"]
                else None
            )
    payload = {
        "format_version": 1,
        "status": "completed",
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": sha256_file(protocol_path),
        "selection_rule": "Among guard-feasible MarginSynth components, minimize exact ABC AND nodes; break ties by guard accuracy loss, method time, and name.",
        "selected_component": None if selected is None else selected["method"],
        "feasible_components": [record["method"] for record in feasible],
        "accuracy_hardware_pareto": nondominated(feasible),
        "records": records,
        "test_used": False,
    }
    json_path = root / "component_comparison.json"
    csv_path = root / "component_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if records:
        scalar_keys = [
            key for key, value in records[0].items()
            if not isinstance(value, (dict, list))
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            writer.writerows(
                {key: record.get(key) for key in scalar_keys} for record in records
            )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
