#!/usr/bin/env python3
"""Materialize predeclared replay steps from a completed MarginSynth-v2 log."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.rewrites import proposal_from_dict
from experiments.marginsynth.search import circuit_sha256
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
)
from torchlogix import Circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-search", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, nargs="+", type=int)
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    source_dir = run_dir / cli.source_search
    output_dir = run_dir / cli.output
    source_summary_path = source_dir / "search_summary.json"
    source_log_path = source_dir / "rewrite_log.json"
    source_config_path = source_dir / "search_config.json"
    source_summary = json.loads(source_summary_path.read_text())
    rewrite_log = json.loads(source_log_path.read_text())
    source_config = json.loads(source_config_path.read_text())
    requested = sorted(set(cli.steps))
    if not requested or requested[0] <= 0:
        raise ValueError("--steps must contain positive proposal indices")
    if requested[-1] > len(rewrite_log):
        raise ValueError(
            f"requested step {requested[-1]} exceeds log length "
            f"{len(rewrite_log)}"
        )

    points_by_step = {
        int(point["step"]): point for point in source_summary["points"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit = Circuit.from_json_file(
        str(run_dir / source_config["input_circuit"])
    )
    selected = []
    requested_set = set(requested)
    for entry in rewrite_log:
        step = int(entry["step"])
        if circuit_sha256(circuit) != entry["circuit_before_sha256"]:
            raise RuntimeError(f"replay before-hash mismatch at step {step}")
        proposal_from_dict(entry["proposal"]).apply(circuit)
        circuit.simplify()
        if circuit_sha256(circuit) != entry["circuit_after_sha256"]:
            raise RuntimeError(f"replay after-hash mismatch at step {step}")
        if step not in requested_set:
            continue
        snapshot = output_dir / "snapshots" / f"step_{step:04d}"
        snapshot.mkdir(parents=True, exist_ok=True)
        circuit.write_json(str(snapshot / "circuit.json"))
        circuit.normalized_for_hardware_argmax().write_verilog_code(
            str(snapshot / "circuit.v")
        )
        point = copy.deepcopy(points_by_step[step])
        point["snapshot"] = f"snapshots/step_{step:04d}"
        selected.append(point)

    pareto = [
        {
            "budget": f"replay-step-{point['step']}",
            "selected_step": point["step"],
            "accuracy_loss": point["behavior"]["accuracy_loss"],
            "disagreement": point["behavior"]["decision_flip_rate"],
            "maximum_per_class_accuracy_loss": point["behavior"][
                "maximum_per_class_accuracy_loss"
            ],
            "maximum_per_class_disagreement": point["behavior"][
                "maximum_per_class_disagreement"
            ],
            "live_gates": point["cost"]["live_gates"],
            "estimated_abc_nodes": point["cost"]["estimated_abc_nodes"],
            "snapshot": point["snapshot"],
        }
        for point in selected
    ]
    payload = {
        "format_version": 1,
        "status": "completed",
        "development_run": True,
        "method": "MarginSynth-v2-convergence-replay",
        "source_search": cli.source_search,
        "source_search_summary_sha256": sha256_file(source_summary_path),
        "source_rewrite_log_sha256": sha256_file(source_log_path),
        "source_search_config_sha256": sha256_file(source_config_path),
        "source_revision": git_revision(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "selected_steps": requested,
        "points": selected,
        "pareto": pareto,
        "data_policy": {
            "selection_partition": "calibration",
            "validation_used": False,
            "test_used": False,
        },
    }
    (output_dir / "search_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
