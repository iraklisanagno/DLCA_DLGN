#!/usr/bin/env python3
"""Calibrate MarginSynth's operation-aware proxy from existing ABC results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.cost_model import (
    circuit_features,
    fit_nonnegative_estimator,
)
from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    write_artifact_manifest,
)
from torchlogix import Circuit


def collect_records(run_dir: Path) -> list[dict]:
    records = []
    baseline_result = run_dir / "synthesis_verification.json"
    baseline_circuit = run_dir / "exact_simplified_circuit.json"
    if baseline_result.exists() and baseline_circuit.exists():
        result = json.loads(baseline_result.read_text())
        records.append(
            {
                "name": "exact-baseline",
                "circuit": str(baseline_circuit.relative_to(run_dir)),
                "circuit_sha256": sha256_file(baseline_circuit),
                "abc_and_nodes": result["abc"]["stats"]["and_nodes"],
                "features": circuit_features(
                    Circuit.from_json_file(str(baseline_circuit))
                ),
            }
        )

    for synthesis_path in sorted(
        run_dir.glob("search_*/frontier_synthesis.json")
    ):
        payload = json.loads(synthesis_path.read_text())
        search_dir = synthesis_path.parent
        for record in payload.get("records", payload.get("points", [])):
            circuit_path = search_dir / record["snapshot"] / "circuit.json"
            records.append(
                {
                    "name": f"{search_dir.name}:step-{record['step']}",
                    "circuit": str(circuit_path.relative_to(run_dir)),
                    "circuit_sha256": sha256_file(circuit_path),
                    "abc_and_nodes": record["abc"]["stats"]["and_nodes"],
                    "features": circuit_features(
                        Circuit.from_json_file(str(circuit_path))
                    ),
                }
            )

    for result_path in sorted(
        run_dir.glob(
            "baselines/two_stage_unit_tying/ratio_*/synthesis/synthesis.json"
        )
    ):
        result = json.loads(result_path.read_text())
        circuit_path = result_path.parent / "exact_simplified_circuit.json"
        records.append(
            {
                "name": f"two-stage-ratio-{result['ratio']}",
                "circuit": str(circuit_path.relative_to(run_dir)),
                "circuit_sha256": sha256_file(circuit_path),
                "abc_and_nodes": result["abc"]["stats"]["and_nodes"],
                "features": circuit_features(
                    Circuit.from_json_file(str(circuit_path))
                ),
            }
        )
    deduplicated = {}
    for record in records:
        deduplicated[record["circuit_sha256"]] = record
    return list(deduplicated.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synth_cost_model.json"),
    )
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    records = collect_records(run_dir)
    estimator = fit_nonnegative_estimator(records)
    output = cli.output
    if not output.is_absolute():
        output = run_dir / output
    output.write_text(json.dumps(estimator.to_dict(), indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(estimator.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
