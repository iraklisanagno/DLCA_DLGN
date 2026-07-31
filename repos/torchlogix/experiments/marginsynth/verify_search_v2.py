#!/usr/bin/env python3
"""Replay and verify a completed MarginSynth v2 search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.rewrites import proposal_from_dict
from experiments.marginsynth.search import circuit_sha256
from experiments.marginsynth.search_v2 import (
    behavior_metrics,
    within_budgets,
)
from experiments.marginsynth.trace import PackedCalibrationTrace, build_trace
from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    write_artifact_manifest,
)
from torchlogix import Circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--search", required=True)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    search_dir = run_dir / cli.search
    config = json.loads((search_dir / "search_config.json").read_text())
    summary = json.loads((search_dir / "search_summary.json").read_text())
    log = json.loads((search_dir / "rewrite_log.json").read_text())
    reference = PackedCalibrationTrace.load(
        run_dir / config["input_trace"],
        mmap_mode="r",
    )
    current = Circuit.from_json_file(str(run_dir / config["input_circuit"]))
    checks = []
    selected_steps = {
        int(record["selected_step"]) for record in summary["pareto"]
    }
    selected_steps.add(int(summary["points"][-1]["step"]))
    snapshot_checks = []
    if 0 in selected_steps:
        snapshot_path = search_dir / "snapshots/step_0000/circuit.json"
        snapshot_checks.append(
            {
                "step": 0,
                "matches": circuit_sha256(current)
                == circuit_sha256(Circuit.from_json_file(str(snapshot_path))),
            }
        )
    for entry in log:
        before = circuit_sha256(current) == entry["circuit_before_sha256"]
        proposal_from_dict(entry["proposal"]).apply(current)
        current.simplify()
        after = circuit_sha256(current) == entry["circuit_after_sha256"]
        checks.append(
            {
                "step": entry["step"],
                "before_hash_matches": before,
                "after_hash_matches": after,
            }
        )
        if int(entry["step"]) in selected_steps:
            snapshot_path = (
                search_dir
                / "snapshots"
                / f"step_{int(entry['step']):04d}"
                / "circuit.json"
            )
            snapshot_checks.append(
                {
                    "step": entry["step"],
                    "matches": snapshot_path.exists()
                    and circuit_sha256(current)
                    == circuit_sha256(
                        Circuit.from_json_file(str(snapshot_path))
                    ),
                }
            )
    encoded = np.stack(
        [reference.unpack_node(index) for index in range(reference.n_inputs)],
        axis=1,
    )
    final_trace = build_trace(current, encoded, reference.labels)
    measured = behavior_metrics(
        final_trace.scores,
        reference,
        config["margin_reserve"],
    )
    expected = summary["final_behavior"]
    behavior_matches = all(
        np.isclose(measured[key], expected[key], rtol=0.0, atol=1e-12)
        for key in (
            "accuracy",
            "accuracy_loss",
            "decision_flip_rate",
            "maximum_per_class_accuracy_loss",
            "maximum_per_class_disagreement",
        )
    )
    passed = (
        all(
            record["before_hash_matches"] and record["after_hash_matches"]
            for record in checks
        )
        and all(record["matches"] for record in snapshot_checks)
        and behavior_matches
        and within_budgets(measured, config)
    )
    result = {
        "format_version": 2,
        "status": "passed" if passed else "failed",
        "search": cli.search,
        "rewrite_log_sha256": sha256_file(search_dir / "rewrite_log.json"),
        "replay_checks": checks,
        "snapshot_checks": snapshot_checks,
        "behavior_matches": behavior_matches,
        "measured_final_behavior": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in measured.items()
            if key != "predictions"
        },
        "data_policy": {
            "partition": "calibration",
            "validation_used": False,
            "test_used": False,
        },
    }
    (search_dir / "search_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("MarginSynth v2 replay verification failed")


if __name__ == "__main__":
    main()
