#!/usr/bin/env python3
"""Evaluate seed-0 promotion gates for whole-circuit distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def method_record(run_dir: Path, relative: str, label: str) -> dict:
    method_dir = run_dir / relative
    summary = load(method_dir / "summary.json")
    synthesis = load(method_dir / "export_summary.json")
    changes = load(method_dir / "learned_changes.json")
    retained = int(summary["retained_changes"])
    selected = changes[:retained]
    return {
        "label": label,
        "relative_directory": relative,
        "validation_accuracy": summary["validation"]["accuracy"],
        "validation_disagreement": summary["validation"]["decision_flip_rate"],
        "calibration_feasible": summary["calibration_feasible"],
        "repair_holdout_feasible": summary.get("repair_holdout_feasible", True),
        "retained_changes": retained,
        "nonconstant_changes": sum(
            int(item["new_lut"] not in (0, 15)) for item in selected
        ),
        "abc_and_nodes": synthesis["abc_and_nodes"],
        "abc_levels": synthesis["abc_levels"],
        "live_gates": synthesis["live_gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    original_synth = load(run_dir / "synthesis_verification.json")
    original_summary = load(run_dir / "run_summary.json")
    unit_synth = load(
        run_dir / "baselines/two_stage_unit_tying/synthesis_aggregate.json"
    )[0]
    unit_summary = load(
        run_dir / "baselines/two_stage_unit_tying/ratio_10/summary.json"
    )
    original = {
        "label": "Original exact-simplified checkpoint",
        "validation_accuracy": original_summary["best_validation_hard_accuracy"],
        "abc_and_nodes": original_synth["abc"]["stats"]["and_nodes"],
        "abc_levels": original_synth["abc"]["stats"]["levels"],
        "live_gates": original_synth["hardware_circuit"]["logic_gates"],
    }
    unit_tying = {
        "label": "Two-Stage Unit Tying, 10%",
        "validation_accuracy": unit_summary["validation"]["accuracy"],
        "validation_disagreement": unit_summary["validation"]["decision_flip_rate"],
        "abc_and_nodes": unit_synth["abc_and_nodes"],
        "abc_levels": unit_synth["abc_levels"],
        "live_gates": unit_synth["exact_logic_gates"],
    }
    candidates = [
        method_record(
            run_dir,
            "distillation/margin_all_aig_holdout_v3_seed0",
            "Margin resynthesis, AIG proxy",
        ),
        method_record(
            run_dir,
            "distillation/ablation_matrix_v3_seed0/gate_count_proxy",
            "Margin resynthesis, gate-count proxy",
        ),
    ]
    eligible = [
        item
        for item in candidates
        if item["calibration_feasible"]
        and item["repair_holdout_feasible"]
        and item["nonconstant_changes"] / max(item["retained_changes"], 1) >= 0.25
    ]
    central = min(eligible, key=lambda item: item["abc_and_nodes"])
    material_target = int(unit_tying["abc_and_nodes"] * 0.99)
    criteria = {
        "calibration_and_repair_holdout_feasible": bool(
            central["calibration_feasible"] and central["repair_holdout_feasible"]
        ),
        "validation_accuracy_not_below_unit_tying": bool(
            central["validation_accuracy"] >= unit_tying["validation_accuracy"]
        ),
        "abc_nodes_strictly_below_unit_tying": bool(
            central["abc_and_nodes"] < unit_tying["abc_and_nodes"]
        ),
        "abc_nodes_at_least_one_percent_below_unit_tying": bool(
            central["abc_and_nodes"] <= material_target
        ),
        "at_least_25_percent_nonconstant_changes": bool(
            central["nonconstant_changes"] / max(central["retained_changes"], 1)
            >= 0.25
        ),
    }
    promotion = all(criteria.values())
    result = {
        "format_version": 1,
        "status": "completed",
        "original": original,
        "unit_tying_target": unit_tying,
        "full_action_candidates": candidates,
        "selected_central_candidate": central,
        "material_abc_target": material_target,
        "promotion_criteria": criteria,
        "five_seed_promotion_passed": promotion,
        "five_seed_action": "run" if promotion else "skip",
        "five_seed_reason": None if promotion else "seed-0 candidate did not beat the Unit-Tying ABC target",
        "second_dataset_promotion_passed": False,
        "second_dataset_action": "skip",
        "second_dataset_reason": (
            "requires a successful central five-seed study; seed-0 promotion failed"
            if not promotion
            else "requires completion and success of the central five-seed study"
        ),
        "validation_used_for_optimization_or_repair": False,
        "test_used": False,
    }
    output = run_dir / "distillation" / "seed0_advancement_decision.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
