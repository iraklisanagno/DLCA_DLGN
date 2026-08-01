#!/usr/bin/env python3
"""Build a compact cross-case table from a completed Bayesian exploration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASES = (
    "guarded_constrained",
    "guarded_unconstrained",
    "aggressive_constrained",
    "aggressive_unconstrained",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def event_timing(path: Path) -> dict[str, float | str]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    trial_starts = [event for event in events if event["event"] == "trial_started"]
    trial_ends = [event for event in events if event["event"] == "trial_completed"]
    if not trial_starts or not trial_ends:
        raise RuntimeError(f"no completed acquisition interval in {path}")
    started = min(parse_time(event["timestamp_utc"]) for event in trial_starts)
    finished = max(parse_time(event["timestamp_utc"]) for event in trial_ends)
    return {
        "acquisition_started_at_utc": started.isoformat(),
        "acquisition_finished_at_utc": finished.isoformat(),
        "acquisition_elapsed_seconds": (finished - started).total_seconds(),
        "completed_trial_wall_seconds": sum(
            float(event["wall_seconds"]) for event in trial_ends
        ),
    }


def reduction_percent(value: int, reference: int) -> float:
    return 100.0 * (reference - value) / reference


def best_exact_record(
    records: list[dict[str, Any]],
    references: dict[str, Any],
) -> dict[str, Any] | None:
    exact = [
        record
        for record in records
        if record.get("exact_promotion", {}).get("status") == "completed"
        and record.get("abc_and_nodes") is not None
    ]
    if not exact:
        return None
    record = min(exact, key=lambda item: (item["abc_and_nodes"], item["accuracy_loss"]))
    guard = record["metrics"]["selection_guard"]
    synthesis = record["metrics"]["synthesis"]
    original = references["original"]
    unit_tying = references["unit_tying"]
    return {
        "trial_number": record["trial_number"],
        "accuracy_loss": record["accuracy_loss"],
        "guard_accuracy": guard["accuracy"],
        "guard_macro_f1": guard["macro_f1"],
        "guard_disagreement": guard["decision_flip_rate"],
        "maximum_per_class_accuracy_loss": guard[
            "maximum_per_class_accuracy_loss"
        ],
        "maximum_per_class_disagreement": guard[
            "maximum_per_class_disagreement"
        ],
        "abc_and_nodes": record["abc_and_nodes"],
        "abc_levels": synthesis["abc_levels"],
        "live_gates": synthesis["live_gates"],
        "abc_reduction_vs_original_percent": reduction_percent(
            record["abc_and_nodes"], original["abc_and_nodes"]
        ),
        "abc_reduction_vs_unit_tying_percent": reduction_percent(
            record["abc_and_nodes"], unit_tying["abc_and_nodes"]
        ),
        "live_gate_reduction_vs_original_percent": reduction_percent(
            synthesis["live_gates"], original["live_gates"]
        ),
        "live_gate_reduction_vs_unit_tying_percent": reduction_percent(
            synthesis["live_gates"], unit_tying["live_gates"]
        ),
        "level_delta_vs_unit_tying": synthesis["abc_levels"]
        - unit_tying["abc_levels"],
        "method_wall_seconds": record["wall_seconds"],
        "exact_promotion_wall_seconds": record["exact_promotion"]["wall_seconds"],
        "selected_recovery_step": record.get("selected_recovery_step"),
        "checkpoint_sha256": record["artifacts"]["checkpoint_sha256"],
        "trial_directory": record["artifacts"]["trial_directory"],
        "parameters": record["parameters"],
    }


def summarize_case(
    case_dir: Path,
    references: dict[str, Any],
) -> dict[str, Any]:
    records = load_json(case_dir / "trials.json")
    summary = load_json(case_dir / "study_summary.json")
    promotion = load_json(case_dir / "exact_promotion_summary.json")
    timing = event_timing(case_dir / "events.jsonl")
    return {
        "case": case_dir.name,
        "completed_trials": sum(
            record.get("status") == "completed" for record in records
        ),
        "failed_trials": sum(record.get("status") == "failed" for record in records),
        "feasible_trials": sum(record.get("feasible") is True for record in records),
        "exact_promotions": sum(
            item.get("status") == "completed" for item in promotion["results"]
        ),
        "exact_promotion_wall_seconds": sum(
            float(item["wall_seconds"])
            for item in promotion["results"]
            if item.get("status") == "completed"
        ),
        "pareto_trial_numbers": summary["pareto_trial_numbers"],
        "test_used": summary["test_used"],
        "validation_used": summary["validation_used"],
        **timing,
        "best_exact_hardware": best_exact_record(records, references),
    }


def same_guard_unit_tying(run_dir: Path) -> dict[str, Any] | None:
    path = (
        run_dir
        / "baselines"
        / "two_stage_unit_tying"
        / "ratio_10"
        / "bayesian_guard_reference.json"
    )
    if not path.exists():
        return None
    result = load_json(path)
    guard = result["guard"]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "guard_indices_sha256": result["guard_indices_sha256"],
        "accuracy": guard["accuracy"],
        "accuracy_loss": guard["accuracy_loss"],
        "macro_f1": guard["macro_f1"],
        "disagreement": guard["decision_flip_rate"],
        "maximum_per_class_accuracy_loss": guard[
            "maximum_per_class_accuracy_loss"
        ],
        "maximum_per_class_disagreement": guard[
            "maximum_per_class_disagreement"
        ],
        "data_policy": result["data_policy"],
    }


def csv_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        best = case["best_exact_hardware"] or {}
        rows.append(
            {
                "case": case["case"],
                "completed_trials": case["completed_trials"],
                "failed_trials": case["failed_trials"],
                "feasible_trials": case["feasible_trials"],
                "exact_promotions": case["exact_promotions"],
                "acquisition_elapsed_seconds": case["acquisition_elapsed_seconds"],
                "completed_trial_wall_seconds": case["completed_trial_wall_seconds"],
                "exact_promotion_wall_seconds": case[
                    "exact_promotion_wall_seconds"
                ],
                "pareto_trial_numbers": json.dumps(case["pareto_trial_numbers"]),
                "best_trial_number": best.get("trial_number"),
                "accuracy_loss": best.get("accuracy_loss"),
                "guard_disagreement": best.get("guard_disagreement"),
                "maximum_per_class_accuracy_loss": best.get(
                    "maximum_per_class_accuracy_loss"
                ),
                "maximum_per_class_disagreement": best.get(
                    "maximum_per_class_disagreement"
                ),
                "abc_and_nodes": best.get("abc_and_nodes"),
                "live_gates": best.get("live_gates"),
                "abc_levels": best.get("abc_levels"),
                "abc_reduction_vs_unit_tying_percent": best.get(
                    "abc_reduction_vs_unit_tying_percent"
                ),
                "live_gate_reduction_vs_unit_tying_percent": best.get(
                    "live_gate_reduction_vs_unit_tying_percent"
                ),
                "method_wall_seconds": best.get("method_wall_seconds"),
                "selected_recovery_step": best.get("selected_recovery_step"),
                "test_used": case["test_used"],
                "validation_used": case["validation_used"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load_json(protocol_path)
    study_root = run_dir / "bayesian_search" / protocol["protocol_name"]
    if not study_root.is_dir():
        raise FileNotFoundError(study_root)

    cases = [
        summarize_case(study_root / name, protocol["references"]) for name in CASES
    ]
    starts = [parse_time(case["acquisition_started_at_utc"]) for case in cases]
    finishes = [parse_time(case["acquisition_finished_at_utc"]) for case in cases]
    result = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_name": protocol["protocol_name"],
        "protocol": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_object_sha256": object_sha256(protocol),
        "study_root": str(study_root),
        "references": protocol["references"],
        "unit_tying_same_guard_reference": same_guard_unit_tying(run_dir),
        "cases": cases,
        "totals": {
            "completed_trials": sum(case["completed_trials"] for case in cases),
            "failed_trials": sum(case["failed_trials"] for case in cases),
            "feasible_trials": sum(case["feasible_trials"] for case in cases),
            "exact_promotions": sum(case["exact_promotions"] for case in cases),
            "acquisition_elapsed_seconds": (
                max(finishes) - min(starts)
            ).total_seconds(),
            "completed_trial_wall_seconds": sum(
                case["completed_trial_wall_seconds"] for case in cases
            ),
            "exact_promotion_wall_seconds": sum(
                case["exact_promotion_wall_seconds"] for case in cases
            ),
        },
        "data_policy": {
            "test_used": any(case["test_used"] for case in cases),
            "validation_used": any(case["validation_used"] for case in cases),
        },
    }

    json_path = study_root / "cross_case_summary.json"
    csv_path = study_root / "cross_case_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = csv_rows(cases)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
