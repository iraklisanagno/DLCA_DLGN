import json
from datetime import datetime, timedelta, timezone

import pytest

from experiments.marginsynth.evaluate_calibration_guard import (
    evaluation_data_policy,
)
from experiments.marginsynth.summarize_bayesian_results import (
    best_exact_record,
    event_timing,
)


def _exact_record(trial_number, nodes, accuracy_loss):
    return {
        "trial_number": trial_number,
        "accuracy_loss": accuracy_loss,
        "abc_and_nodes": nodes,
        "wall_seconds": 4.0,
        "selected_recovery_step": None,
        "parameters": {"maximum_updates": 100},
        "artifacts": {
            "checkpoint_sha256": f"checkpoint-{trial_number}",
            "trial_directory": f"trial_{trial_number:05d}",
        },
        "exact_promotion": {"status": "completed", "wall_seconds": 2.0},
        "metrics": {
            "selection_guard": {
                "accuracy": 0.9,
                "macro_f1": 0.89,
                "decision_flip_rate": 0.01,
                "maximum_per_class_accuracy_loss": 0.02,
                "maximum_per_class_disagreement": 0.03,
            },
            "synthesis": {"abc_levels": 8, "live_gates": 800},
        },
    }


def test_best_exact_record_uses_exact_abc_cost_and_reports_reference_deltas():
    records = [
        _exact_record(1, 900, 0.0),
        _exact_record(2, 850, 0.01),
        {
            **_exact_record(3, 700, 0.0),
            "exact_promotion": {"status": "failed", "wall_seconds": 1.0},
        },
    ]
    references = {
        "original": {"abc_and_nodes": 1000, "live_gates": 1000},
        "unit_tying": {
            "abc_and_nodes": 900,
            "abc_levels": 7,
            "live_gates": 900,
        },
    }

    best = best_exact_record(records, references)

    assert best["trial_number"] == 2
    assert best["abc_reduction_vs_original_percent"] == pytest.approx(15.0)
    assert best["abc_reduction_vs_unit_tying_percent"] == pytest.approx(
        100 * 50 / 900
    )
    assert best["live_gate_reduction_vs_unit_tying_percent"] == pytest.approx(
        100 / 9
    )
    assert best["level_delta_vs_unit_tying"] == 1


def test_event_timing_counts_completed_acquisition_trials_only(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        {"event": "trial_started", "timestamp_utc": start.isoformat()},
        {
            "event": "trial_completed",
            "timestamp_utc": (start + timedelta(seconds=10)).isoformat(),
            "wall_seconds": 9.0,
        },
        {
            "event": "trial_started",
            "timestamp_utc": (start + timedelta(seconds=11)).isoformat(),
        },
        {
            "event": "trial_completed",
            "timestamp_utc": (start + timedelta(seconds=30)).isoformat(),
            "wall_seconds": 18.0,
        },
        {
            "event": "exact_promotion_completed",
            "timestamp_utc": (start + timedelta(seconds=90)).isoformat(),
            "wall_seconds": 60.0,
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    timing = event_timing(path)

    assert timing["acquisition_elapsed_seconds"] == 30.0
    assert timing["completed_trial_wall_seconds"] == 27.0


def test_report_only_data_policy_cannot_be_mistaken_for_search_selection():
    selection = evaluation_data_policy(report_only=False)
    report = evaluation_data_policy(report_only=True)

    assert selection["used_for_bayesian_selection"]
    assert not selection["report_only"]
    assert not report["used_for_bayesian_selection"]
    assert report["report_only"]
    assert not report["validation_loaded"]
    assert not report["test_loaded"]
