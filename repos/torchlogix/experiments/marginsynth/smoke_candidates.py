#!/usr/bin/env python3
"""Run a non-mutating incremental-candidate smoke study on a saved trace."""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.incremental import evaluate_rewrite
from experiments.marginsynth.rewrites import generate_gate_rewrites
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
    parser.add_argument("--per-kind", type=int, default=32)
    parser.add_argument("--full-checks", type=int, default=14)
    cli = parser.parse_args()
    if cli.per_kind <= 0 or cli.full_checks < 0:
        raise ValueError("candidate counts must be non-negative, with --per-kind > 0")

    run_dir = cli.run_dir.resolve()
    circuit_path = run_dir / "exact_simplified_circuit.json"
    trace_path = run_dir / "calibration_trace"
    circuit = Circuit.from_json_file(str(circuit_path))
    trace = PackedCalibrationTrace.load(trace_path, mmap_mode="r")
    candidates = generate_gate_rewrites(circuit)

    by_kind = defaultdict(list)
    for candidate in candidates:
        by_kind[candidate.kind.value].append(candidate)
    selected = []
    for kind in sorted(by_kind):
        selected.extend(by_kind[kind][: cli.per_kind])

    start = time.perf_counter()
    evaluations = [
        evaluate_rewrite(circuit, trace, candidate)
        for candidate in selected
    ]
    evaluation_wall_seconds = time.perf_counter() - start

    exact_check_indices = (
        np.linspace(
            0,
            len(selected) - 1,
            min(cli.full_checks, len(selected)),
            dtype=int,
        ).tolist()
        if selected and cli.full_checks
        else []
    )
    encoded_inputs = np.stack(
        [trace.unpack_node(input_id) for input_id in range(trace.n_inputs)],
        axis=1,
    )
    full_checks = []
    for index in exact_check_indices:
        rewrite = selected[index]
        evaluation = evaluations[index]
        rewritten = rewrite.apply_to_copy(circuit)
        complete = build_trace(rewritten, encoded_inputs, trace.labels)
        scores_exact = bool(np.array_equal(evaluation.scores, complete.scores))
        predictions_exact = bool(
            np.array_equal(evaluation.predictions, complete.predictions)
        )
        full_checks.append(
            {
                "candidate_index": index,
                "rewrite": rewrite.to_dict(),
                "scores_exact": scores_exact,
                "predictions_exact": predictions_exact,
                "maximum_absolute_score_difference": float(
                    np.max(np.abs(evaluation.scores - complete.scores))
                ),
            }
        )
    if not all(
        check["scores_exact"] and check["predictions_exact"]
        for check in full_checks
    ):
        raise RuntimeError("incremental/full candidate smoke check failed")

    flip_counts = np.asarray(
        [evaluation.decision_flip_count for evaluation in evaluations]
    )
    accuracy_changes = np.asarray(
        [evaluation.accuracy_change for evaluation in evaluations]
    )
    cone_sizes = np.asarray(
        [len(evaluation.affected_gate_ids) for evaluation in evaluations]
    )
    evaluation_seconds = np.asarray(
        [evaluation.evaluation_seconds for evaluation in evaluations]
    )
    payload = {
        "format_version": 1,
        "status": "passed",
        "purpose": (
            "non-mutating mechanism smoke study; no candidate was accepted and "
            "this is not a Pareto search result"
        ),
        "source_revision": git_revision(),
        "circuit": circuit_path.name,
        "circuit_sha256": sha256_file(circuit_path),
        "trace": trace_path.name,
        "trace_metadata_sha256": sha256_file(trace_path / "metadata.json"),
        "rewrite_implementation_sha256": sha256_file(
            Path(__file__).with_name("rewrites.py")
        ),
        "incremental_implementation_sha256": sha256_file(
            Path(__file__).with_name("incremental.py")
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "data_policy": {
            "partition": "calibration",
            "examples": trace.num_samples,
            "labels_used_for_accuracy_metrics": True,
            "validation_used": False,
            "test_used": False,
        },
        "candidate_space": {
            "total": len(candidates),
            "targets": len({candidate.target_id for candidate in candidates}),
            "by_kind": dict(
                sorted(Counter(candidate.kind.value for candidate in candidates).items())
            ),
        },
        "smoke_selection": {
            "per_kind": cli.per_kind,
            "evaluated": len(selected),
            "by_kind": dict(
                sorted(Counter(candidate.kind.value for candidate in selected).items())
            ),
        },
        "aggregate": {
            "wall_seconds": evaluation_wall_seconds,
            "candidate_seconds_mean": float(evaluation_seconds.mean()),
            "candidate_seconds_p95": float(
                np.quantile(evaluation_seconds, 0.95)
            ),
            "decision_flip_count_min": int(flip_counts.min()),
            "decision_flip_count_median": float(np.median(flip_counts)),
            "decision_flip_count_max": int(flip_counts.max()),
            "zero_flip_candidates": int(np.count_nonzero(flip_counts == 0)),
            "accuracy_change_min": float(accuracy_changes.min()),
            "accuracy_change_median": float(np.median(accuracy_changes)),
            "accuracy_change_max": float(accuracy_changes.max()),
            "affected_cone_size_min": int(cone_sizes.min()),
            "affected_cone_size_median": float(np.median(cone_sizes)),
            "affected_cone_size_max": int(cone_sizes.max()),
        },
        "full_resimulation_checks": full_checks,
        "evaluations": [evaluation.summary() for evaluation in evaluations],
        "baseline_mutated": False,
        "accepted_candidates": 0,
    }
    output_path = run_dir / "candidate_smoke.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "candidate_space",
                    "smoke_selection",
                    "aggregate",
                    "baseline_mutated",
                    "accepted_candidates",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
