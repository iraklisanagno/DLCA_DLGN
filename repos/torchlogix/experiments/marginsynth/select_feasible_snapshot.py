#!/usr/bin/env python3
"""Select the highest-gain calibration-guard-feasible two-pass snapshot."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    write_artifact_manifest,
)


STAGE_ORDER = {"source": 0, "first": 1, "second": 2}


def select_candidate(candidates: list[dict]) -> dict:
    feasible = [
        candidate
        for candidate in candidates
        if candidate["repair_feasible"]
        and candidate["calibration_feasible"]
        and candidate["guard_feasible"]
    ]
    if not feasible:
        raise ValueError("snapshot set contains no feasible source fallback")
    return max(
        feasible,
        key=lambda candidate: (
            float(candidate["estimated_hardware_gain"]),
            int(candidate["cumulative_retained_changes"]),
            STAGE_ORDER[candidate["stage"]],
        ),
    )


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def estimate(summary: dict) -> float:
    payload = summary.get("retained_hardware_estimate")
    if not isinstance(payload, dict):
        raise ValueError("snapshot is missing retained_hardware_estimate")
    return float(payload["estimated_hardware_gain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("component_dir", type=Path)
    parser.add_argument("--output", default="selected_snapshot", type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    component_dir = cli.component_dir.resolve()
    output_dir = cli.output
    if not output_dir.is_absolute():
        output_dir = component_dir / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite snapshot selection: {output_dir}")
    output_dir.mkdir(parents=True)

    first_dir = component_dir / "first_resynthesis"
    second_dir = component_dir / "second_resynthesis"
    first_summary_path = first_dir / "summary.json"
    second_summary_path = second_dir / "summary.json"
    first = load(first_summary_path)
    second = load(second_summary_path)
    for name, summary in (("first", first), ("second", second)):
        if summary.get("status") != "completed":
            raise ValueError(f"{name} snapshot is incomplete")
        if summary.get("data_policy", {}).get("test_used") is not False:
            raise ValueError(f"{name} snapshot did not keep test data sealed")
        if summary.get("guard_holdout") is None:
            raise ValueError(f"{name} snapshot has no untouched guard holdout")

    first_gain = estimate(first)
    second_gain = first_gain + estimate(second)
    source_checkpoint = source_run / "best_checkpoint.pt"
    candidates = [
        {
            "stage": "source",
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": sha256_file(source_checkpoint),
            "summary": None,
            "summary_sha256": None,
            "repair_feasible": True,
            "calibration_feasible": True,
            "guard_feasible": True,
            "estimated_hardware_gain": 0.0,
            "cumulative_retained_changes": 0,
        },
        {
            "stage": "first",
            "checkpoint": str(first_dir / "distilled_checkpoint.pt"),
            "checkpoint_sha256": sha256_file(first_dir / "distilled_checkpoint.pt"),
            "summary": str(first_summary_path),
            "summary_sha256": sha256_file(first_summary_path),
            "repair_feasible": bool(first["repair_holdout_feasible"]),
            "calibration_feasible": bool(first["calibration_feasible"]),
            "guard_feasible": bool(first["guard_holdout_feasible"]),
            "estimated_hardware_gain": first_gain,
            "cumulative_retained_changes": int(first["retained_changes"]),
        },
        {
            "stage": "second",
            "checkpoint": str(second_dir / "distilled_checkpoint.pt"),
            "checkpoint_sha256": sha256_file(second_dir / "distilled_checkpoint.pt"),
            "summary": str(second_summary_path),
            "summary_sha256": sha256_file(second_summary_path),
            "repair_feasible": bool(second["repair_holdout_feasible"]),
            "calibration_feasible": bool(second["calibration_feasible"]),
            "guard_feasible": bool(second["guard_holdout_feasible"]),
            "estimated_hardware_gain": second_gain,
            "cumulative_retained_changes": int(second["locked_source_changes"])
            + int(second["retained_changes"]),
        },
    ]
    selected = select_candidate(candidates)
    selected_checkpoint = Path(selected["checkpoint"])
    output_checkpoint = output_dir / "selected_checkpoint.pt"
    shutil.copy2(selected_checkpoint, output_checkpoint)
    result = {
        "format_version": 1,
        "status": "completed",
        "method": "guard-feasible-structural-snapshot-selection",
        "selection_rule": (
            "Among source, first-pass, and second-pass snapshots satisfying "
            "repair, complete-calibration, and untouched-guard budgets, maximize "
            "the frozen structural hardware-gain estimate; break ties by retained "
            "changes and then later stage. Validation and test are not consulted."
        ),
        "selected_stage": selected["stage"],
        "selected_candidate": selected,
        "candidates": candidates,
        "selected_checkpoint": output_checkpoint.name,
        "selected_checkpoint_sha256": sha256_file(output_checkpoint),
        "fallback_applied": selected["stage"] != "second",
        "data_policy": {
            "selection_partition": "calibration repair and guard holdouts",
            "validation_used_for_selection": False,
            "test_used": False,
        },
        "software": {
            "source_revision": git_revision(),
            "python": platform.python_version(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    # The orchestrator uses summary.json as its uniform completion marker.
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
