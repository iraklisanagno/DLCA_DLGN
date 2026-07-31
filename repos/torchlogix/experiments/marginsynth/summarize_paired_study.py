#!/usr/bin/env python3
"""Aggregate frozen validation and synthesis results without opening test data."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.freeze_protocol import (
    find_budget_point,
    resolve_record_path,
)
from experiments.marginsynth.verify_checkpoint import sha256_file


def one_match(records: list[dict], key: str, value: float) -> dict:
    matches = [
        record
        for record in records
        if abs(float(record[key]) - value) <= 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one record with {key}={value}, found "
            f"{len(matches)}"
        )
    return matches[0]


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "values": [float(value) for value in array],
    }


def exact_bootstrap_mean_ci(values: list[float]) -> list[float]:
    """Return an exact equal-tail bootstrap CI for small paired studies."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    if len(array) > 8:
        raise ValueError("exact bootstrap is intentionally limited to <=8 seeds")
    means = np.fromiter(
        (
            float(array[list(indices)].mean())
            for indices in itertools.product(range(len(array)), repeat=len(array))
        ),
        dtype=np.float64,
        count=len(array) ** len(array),
    )
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", required=True, type=Path)
    parser.add_argument("--accuracy-budget", required=True, type=float)
    parser.add_argument("--unit-tying-ratio", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    manifest = json.loads(cli.paired_manifest.read_text())
    records = []
    for manifest_record in manifest["records"]:
        seed = int(manifest_record["seed"])
        run_dir = resolve_record_path(
            manifest_record["run_dir"], cli.paired_manifest
        )
        search_config_path = resolve_record_path(
            manifest_record["search_config"], cli.paired_manifest
        )
        search_config = json.loads(search_config_path.read_text())
        search_dir = run_dir / search_config["output"]

        summary = json.loads((search_dir / "search_summary.json").read_text())
        margin_point = find_budget_point(summary, cli.accuracy_budget)
        step = int(margin_point["selected_step"])
        validation = json.loads(
            (search_dir / "validation_frontier.json").read_text()
        )
        validation_point = one_match(validation["records"], "step", float(step))
        synthesis = json.loads(
            (search_dir / "frontier_synthesis.json").read_text()
        )
        synthesis_point = one_match(synthesis["points"], "step", float(step))

        baseline_synthesis = json.loads(
            (run_dir / "synthesis_verification.json").read_text()
        )
        baseline_nodes = int(
            baseline_synthesis["abc"]["stats"]["and_nodes"]
        )
        baseline_live = int(
            baseline_synthesis["hardware_circuit"]["logic_gates"]
        )
        baseline_accuracy = float(validation["baseline_validation_accuracy"])

        unit_dir = run_dir / "baselines" / "two_stage_unit_tying"
        unit_point = one_match(
            json.loads((unit_dir / "aggregate.json").read_text()),
            "ratio",
            cli.unit_tying_ratio,
        )
        unit_synthesis = one_match(
            json.loads((unit_dir / "synthesis_aggregate.json").read_text()),
            "ratio",
            cli.unit_tying_ratio,
        )

        margin_nodes = int(synthesis_point["abc"]["stats"]["and_nodes"])
        margin_live = int(synthesis_point["live_gates"])
        unit_nodes = int(unit_synthesis["abc_and_nodes"])
        unit_live = int(unit_synthesis["exact_logic_gates"])
        margin_validation = validation_point["validation"]
        records.append(
            {
                "seed": seed,
                "baseline": {
                    "validation_accuracy": baseline_accuracy,
                    "abc_and_nodes": baseline_nodes,
                    "live_gates": baseline_live,
                },
                "marginsynth": {
                    "selected_step": step,
                    "validation_accuracy": margin_validation["accuracy"],
                    "validation_accuracy_loss": margin_validation["accuracy_loss"],
                    "validation_disagreement": margin_validation[
                        "decision_flip_rate"
                    ],
                    "maximum_per_class_disagreement": margin_validation[
                        "maximum_per_class_disagreement"
                    ],
                    "abc_and_nodes": margin_nodes,
                    "abc_reduction_fraction": 1 - margin_nodes / baseline_nodes,
                    "live_gates": margin_live,
                    "live_gate_reduction_fraction": 1
                    - margin_live / baseline_live,
                },
                "unit_tying": {
                    "ratio": cli.unit_tying_ratio,
                    "validation_accuracy": unit_point["validation_accuracy"],
                    "validation_accuracy_loss": baseline_accuracy
                    - float(unit_point["validation_accuracy"]),
                    "validation_disagreement": unit_point[
                        "validation_disagreement"
                    ],
                    "abc_and_nodes": unit_nodes,
                    "abc_reduction_fraction": 1 - unit_nodes / baseline_nodes,
                    "live_gates": unit_live,
                    "live_gate_reduction_fraction": 1 - unit_live / baseline_live,
                },
            }
        )

    metric_names = (
        "validation_accuracy",
        "validation_accuracy_loss",
        "validation_disagreement",
        "abc_and_nodes",
        "abc_reduction_fraction",
        "live_gates",
        "live_gate_reduction_fraction",
    )
    aggregate = {}
    for method in ("marginsynth", "unit_tying"):
        aggregate[method] = {
            metric: summarize(
                [float(record[method][metric]) for record in records]
            )
            for metric in metric_names
        }

    paired = {}
    for metric in (
        "validation_accuracy_loss",
        "validation_disagreement",
        "abc_reduction_fraction",
        "live_gate_reduction_fraction",
    ):
        differences = [
            float(record["marginsynth"][metric])
            - float(record["unit_tying"][metric])
            for record in records
        ]
        paired[metric] = summarize(differences)
        paired[metric]["exact_bootstrap_mean_95_ci"] = (
            exact_bootstrap_mean_ci(differences)
        )

    payload = {
        "format_version": 1,
        "status": "completed",
        "data_partition": "validation",
        "test_used": False,
        "paired_manifest": str(cli.paired_manifest),
        "paired_manifest_sha256": sha256_file(cli.paired_manifest),
        "accuracy_budget": cli.accuracy_budget,
        "unit_tying_ratio": cli.unit_tying_ratio,
        "seeds": [int(record["seed"]) for record in records],
        "records": records,
        "aggregate": aggregate,
        "paired_marginsynth_minus_unit_tying": paired,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
