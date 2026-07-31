#!/usr/bin/env python3
"""Replay and fully verify a completed MarginSynth development search."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.rewrites import GateRewrite
from experiments.marginsynth.search import circuit_sha256, reference_metrics
from experiments.marginsynth.trace import PackedCalibrationTrace, build_trace
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
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
    rewrite_log = json.loads((search_dir / "rewrite_log.json").read_text())
    trace = PackedCalibrationTrace.load(
        run_dir / config["input_trace"],
        mmap_mode="r",
    )
    replayed = Circuit.from_json_file(str(run_dir / config["input_circuit"]))

    replay_checks = []
    for entry in rewrite_log:
        before_matches = (
            circuit_sha256(replayed) == entry["circuit_before_sha256"]
        )
        GateRewrite.from_dict(entry["rewrite"]).apply(replayed)
        replayed.simplify()
        after_matches = (
            circuit_sha256(replayed) == entry["circuit_after_sha256"]
        )
        replay_checks.append(
            {
                "step": entry["step"],
                "before_hash_matches": before_matches,
                "after_hash_matches": after_matches,
            }
        )

    final_snapshot = Circuit.from_json_file(
        str(
            search_dir
            / summary["points"][-1]["snapshot"]
            / "circuit.json"
        )
    )
    final_hash = summary["points"][-1]["circuit_sha256"]
    replay_matches_final = (
        circuit_sha256(replayed)
        == circuit_sha256(final_snapshot)
        == final_hash
    )

    encoded_inputs = np.stack(
        [trace.unpack_node(input_id) for input_id in range(trace.n_inputs)],
        axis=1,
    )
    final_trace = build_trace(final_snapshot, encoded_inputs, trace.labels)
    measured = reference_metrics(
        final_trace.scores,
        trace,
        config["mode"],
        config["margin_reserve"],
    )
    expected = summary["final_behavior"]
    behavior_matches = (
        measured["decision_flip_count"] == expected["decision_flip_count"]
        and measured["global_loss"] == expected["global_loss"]
        and measured["maximum_per_class_loss"]
        == expected["maximum_per_class_loss"]
        and measured["accuracy"] == expected["accuracy"]
    )

    snapshot_file_checks = []
    for point in summary["points"]:
        snapshot_dir = search_dir / point["snapshot"]
        metadata = json.loads((snapshot_dir / "metadata.json").read_text())
        files_match = all(
            (snapshot_dir / name).stat().st_size == record["bytes"]
            and sha256_file(snapshot_dir / name) == record["sha256"]
            for name, record in metadata["files"].items()
        )
        snapshot_file_checks.append(
            {"step": point["step"], "files_match": files_match}
        )

    passed = (
        all(
            check["before_hash_matches"] and check["after_hash_matches"]
            for check in replay_checks
        )
        and replay_matches_final
        and behavior_matches
        and all(check["files_match"] for check in snapshot_file_checks)
        and measured["global_loss"] <= config["maximum_global_loss"] + 1e-12
        and measured["maximum_per_class_loss"]
        <= config["maximum_per_class_loss"] + 1e-12
    )
    result = {
        "format_version": 1,
        "status": "passed" if passed else "failed",
        "source_revision": git_revision(),
        "verification_script_sha256": sha256_file(Path(__file__).resolve()),
        "search_script_sha256": sha256_file(Path(__file__).with_name("search.py")),
        "search": cli.search,
        "rewrite_log_sha256": sha256_file(search_dir / "rewrite_log.json"),
        "accepted_rewrites": len(rewrite_log),
        "replay_checks": replay_checks,
        "replay_matches_final_snapshot": replay_matches_final,
        "full_trace_behavior_matches_summary": behavior_matches,
        "measured_final_behavior": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in measured.items()
            if key != "predictions"
        },
        "snapshot_file_checks": snapshot_file_checks,
        "data_policy": {
            "partition": "calibration",
            "validation_used": False,
            "test_used": False,
        },
    }
    output_path = search_dir / "search_verification.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("search replay verification failed")


if __name__ == "__main__":
    main()
