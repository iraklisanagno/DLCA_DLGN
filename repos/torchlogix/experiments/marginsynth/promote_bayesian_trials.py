#!/usr/bin/env python3
"""Promote prespecified feasible Bayesian trials to exact Yosys/ABC measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

TORCHLOGIX_ROOT = Path(__file__).resolve().parents[2]
if str(TORCHLOGIX_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLOGIX_ROOT))

from experiments.marginsynth.bayesian_protocol import (
    STUDY_CASES,
    load_json,
    object_sha256,
    select_promotion_records,
    validate_protocol,
)
from experiments.marginsynth.bayesian_search import (
    StudyExecutor,
    export_study_tables,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--study", choices=["all", *STUDY_CASES], default="all"
    )
    parser.add_argument("--max-per-study", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    run_dir = cli.run_dir.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path, run_dir)
    output_root = run_dir / "bayesian_search" / protocol["protocol_name"]
    snapshot_path = output_root / "protocol.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError("the Bayesian study has not been created")
    if object_sha256(load_json(snapshot_path)) != object_sha256(protocol):
        raise RuntimeError("study protocol snapshot does not match the supplied protocol")
    selected_cases = list(STUDY_CASES) if cli.study == "all" else [cli.study]
    limit = int(
        cli.max_per_study
        if cli.max_per_study is not None
        else protocol["execution"]["promotion_trials_per_study"]
    )
    if limit <= 0:
        raise ValueError("max-per-study must be positive")
    summaries = []
    for case_name in selected_cases:
        study_dir = output_root / case_name
        records_path = study_dir / "trials.json"
        if not records_path.exists():
            raise FileNotFoundError(f"missing completed study export: {records_path}")
        records = json.loads(records_path.read_text())
        completed = sum(record.get("status") == "completed" for record in records)
        expected = int(protocol["sampler"]["trials_per_study"])
        if completed < expected and not cli.allow_incomplete:
            raise RuntimeError(
                f"{case_name} has only {completed}/{expected} completed trials; "
                "use --allow-incomplete only for an explicitly diagnostic promotion"
            )
        selected = select_promotion_records(records, limit)
        executor = StudyExecutor(
            run_dir,
            TORCHLOGIX_ROOT,
            protocol_path,
            protocol,
            output_root,
            case_name,
            smoke=False,
            synthesize_infeasible=False,
        )
        selection_manifest = {
            "format_version": 1,
            "created_at_utc": utc_now(),
            "case": case_name,
            "algorithm": (
                "all feasible proxy-Pareto points, evenly thinned if needed, "
                "then deterministic joint-rank diverse fillers"
            ),
            "maximum_trials": limit,
            "available_completed_trials": completed,
            "available_feasible_trials": sum(
                record.get("status") == "completed"
                and record.get("feasible") is True
                for record in records
            ),
            "selected_trial_numbers": [
                record["trial_number"] for record in selected
            ],
            "selection_inputs": [
                {
                    "trial_number": record["trial_number"],
                    "accuracy_loss": record["accuracy_loss"],
                    "predicted_abc_and_nodes": record["hardware_proxy"],
                }
                for record in selected
            ],
            "validation_used": False,
            "test_used": False,
        }
        write_json(study_dir / "exact_promotion_manifest.json", selection_manifest)
        results = []
        for selection_rank, selected_record in enumerate(selected):
            number = int(selected_record["trial_number"])
            trial_dir = study_dir / "trials" / f"trial_{number:05d}"
            record_path = trial_dir / "trial_record.json"
            record = load_json(record_path)
            if record.get("abc_and_nodes") is not None:
                results.append(
                    {
                        "trial_number": number,
                        "status": "reused",
                        "abc_and_nodes": record["abc_and_nodes"],
                    }
                )
                continue
            started = time.perf_counter()
            record["exact_promotion"] = {
                "status": "running",
                "selection_rank": selection_rank,
                "started_at_utc": utc_now(),
            }
            write_json(record_path, record)
            try:
                synthesis = executor.synthesize(
                    number,
                    trial_dir,
                    Path(record["artifacts"]["method_directory"]),
                    record["artifacts"]["checkpoint"],
                )
                seconds = time.perf_counter() - started
                record["abc_and_nodes"] = int(synthesis["abc_and_nodes"])
                record["objective_fidelity"] = "exact_abc"
                record["metrics"]["synthesis"] = synthesis
                record["unit_tying_abc_node_delta"] = int(
                    synthesis["abc_and_nodes"]
                    - protocol["references"]["unit_tying"]["abc_and_nodes"]
                )
                record["exact_promotion"] = {
                    "status": "completed",
                    "selection_rank": selection_rank,
                    "finished_at_utc": utc_now(),
                    "wall_seconds": seconds,
                }
                results.append(
                    {
                        "trial_number": number,
                        "status": "completed",
                        "abc_and_nodes": record["abc_and_nodes"],
                        "wall_seconds": seconds,
                    }
                )
            except Exception as error:
                record["exact_promotion"] = {
                    "status": "failed",
                    "selection_rank": selection_rank,
                    "finished_at_utc": utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                results.append(
                    {
                        "trial_number": number,
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
            write_json(record_path, record)
            export_study_tables(study_dir)
        summary = {
            "format_version": 1,
            "case": case_name,
            "completed_at_utc": utc_now(),
            "selection_manifest": "exact_promotion_manifest.json",
            "results": results,
            "validation_used_for_selection": False,
            "test_used": False,
        }
        write_json(study_dir / "exact_promotion_summary.json", summary)
        summaries.append(summary)
    write_json(
        output_root / "exact_promotion_summary.json",
        {
            "format_version": 1,
            "completed_at_utc": utc_now(),
            "studies": summaries,
            "validation_used_for_selection": False,
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
