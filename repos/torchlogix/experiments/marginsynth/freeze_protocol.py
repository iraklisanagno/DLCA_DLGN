#!/usr/bin/env python3
"""Freeze a paper protocol before any held-out test evaluation is permitted."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def resolve_record_path(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repository_candidate = REPOSITORY_ROOT / candidate
    if repository_candidate.exists() or repository_candidate.parent.exists():
        return repository_candidate
    return (manifest_path.parent / candidate).resolve()


def find_budget_point(summary: dict, budget: float) -> dict:
    matches = [
        point
        for point in summary["pareto"]
        if abs(float(point["accuracy_budget"]) - budget) <= 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Pareto point for frozen budget {budget}, "
            f"found {len(matches)}"
        )
    return matches[0]


def unit_tying_circuit_path(run_dir: Path, selection: dict) -> Path:
    """Return the circuit emitted by the Unit-Tying synthesis stage."""
    return (
        run_dir
        / "baselines"
        / "two_stage_unit_tying"
        / selection["directory"]
        / "synthesis"
        / "exact_simplified_circuit.json"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", required=True, type=Path)
    parser.add_argument("--method-config", required=True, type=Path)
    parser.add_argument("--operating-point", required=True)
    parser.add_argument(
        "--unit-tying-ratio",
        type=float,
        help="Optional frozen Two-Stage Unit Tying comparison ratio.",
    )
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    paired = json.loads(cli.paired_manifest.read_text())
    if len(paired["seeds"]) < 5:
        raise ValueError("paper protocol requires at least five paired seeds")
    budget = float(cli.operating_point)
    missing = []
    seed_artifacts = []
    for record in paired["records"]:
        run_dir = resolve_record_path(record["run_dir"], cli.paired_manifest)
        search_config_path = resolve_record_path(
            record["search_config"], cli.paired_manifest
        )
        search_config = json.loads(search_config_path.read_text())
        search_dir = run_dir / search_config["output"]
        required = {
            "best_checkpoint": run_dir / "best_checkpoint.pt",
            "exact_baseline": run_dir / "exact_simplified_circuit.json",
            "training_config": run_dir / "training_config.json",
            "search_config": search_config_path,
            "search_summary": search_dir / "search_summary.json",
            "search_verification": search_dir / "search_verification.json",
            "validation_frontier": search_dir / "validation_frontier.json",
            "frontier_synthesis": search_dir / "frontier_synthesis.json",
        }
        unit_tying_selection = None
        if cli.unit_tying_ratio is not None:
            unit_tying_dir = run_dir / "baselines" / "two_stage_unit_tying"
            aggregate_path = unit_tying_dir / "aggregate.json"
            synthesis_path = unit_tying_dir / "synthesis_aggregate.json"
            required["unit_tying_aggregate"] = aggregate_path
            required["unit_tying_synthesis"] = synthesis_path
        record_missing = [
            str(path) for path in required.values() if not path.exists()
        ]
        missing.extend(record_missing)
        if record_missing:
            continue
        summary = json.loads(required["search_summary"].read_text())
        point = find_budget_point(summary, budget)
        snapshot_circuit = search_dir / point["snapshot"] / "circuit.json"
        if not snapshot_circuit.exists():
            missing.append(str(snapshot_circuit))
            continue
        validation = json.loads(required["validation_frontier"].read_text())
        validation_matches = [
            item
            for item in validation["records"]
            if int(item["step"]) == int(point["selected_step"])
        ]
        if len(validation_matches) != 1:
            raise RuntimeError(
                f"seed {record['seed']} lacks validation for selected step "
                f"{point['selected_step']}"
            )
        verification = json.loads(required["search_verification"].read_text())
        if verification.get("status") != "passed":
            raise RuntimeError(
                f"seed {record['seed']} search verification did not pass"
            )
        synthesis = json.loads(required["frontier_synthesis"].read_text())
        if not any(
            int(item["step"]) == int(point["selected_step"])
            and item["abc"]["returncode"] == 0
            and item["yosys"]["returncode"] == 0
            for item in synthesis["points"]
        ):
            raise RuntimeError(
                f"seed {record['seed']} selected point lacks successful synthesis"
            )
        if cli.unit_tying_ratio is not None:
            aggregate = json.loads(
                required["unit_tying_aggregate"].read_text()
            )
            unit_matches = [
                item
                for item in aggregate
                if abs(float(item["ratio"]) - cli.unit_tying_ratio) <= 1e-12
            ]
            if len(unit_matches) != 1:
                raise RuntimeError(
                    f"seed {record['seed']} lacks Two-Stage ratio "
                    f"{cli.unit_tying_ratio}"
                )
            unit_tying_selection = unit_matches[0]
            unit_circuit = unit_tying_circuit_path(
                run_dir, unit_tying_selection
            )
            if not unit_circuit.exists():
                missing.append(str(unit_circuit))
                continue
            unit_synthesis = json.loads(
                required["unit_tying_synthesis"].read_text()
            )
            if not any(
                abs(float(item["ratio"]) - cli.unit_tying_ratio) <= 1e-12
                and item["status"] == "passed"
                for item in unit_synthesis
            ):
                raise RuntimeError(
                    f"seed {record['seed']} Two-Stage ratio "
                    f"{cli.unit_tying_ratio} lacks successful synthesis"
                )
            required["unit_tying_circuit"] = unit_circuit
        artifacts = {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        }
        artifacts["selected_circuit"] = {
            "path": str(snapshot_circuit),
            "sha256": sha256_file(snapshot_circuit),
        }
        seed_artifacts.append(
            {
                "seed": int(record["seed"]),
                "run_dir": str(run_dir),
                "search_output": search_config["output"],
                "selected_step": int(point["selected_step"]),
                "calibration_selection": point,
                "validation": validation_matches[0]["validation"],
                "unit_tying_selection": unit_tying_selection,
                "artifacts": artifacts,
            }
        )
    if missing:
        raise RuntimeError(
            "cannot freeze before all paired validation and synthesis runs finish: "
            + ", ".join(missing)
        )
    payload = {
        "format_version": 2,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_manifest": str(cli.paired_manifest),
        "paired_manifest_sha256": sha256_file(cli.paired_manifest),
        "method_config": str(cli.method_config),
        "method_config_sha256": sha256_file(cli.method_config),
        "operating_point": budget,
        "unit_tying_ratio": cli.unit_tying_ratio,
        "seeds": paired["seeds"],
        "seed_artifacts": seed_artifacts,
        "test_access_authorized_after_this_freeze": True,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
