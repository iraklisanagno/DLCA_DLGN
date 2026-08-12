#!/usr/bin/env python3
"""Reuse a verified exact-source export for a source snapshot fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("selected_dir", type=Path)
    parser.add_argument("--reference-export", required=True, type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    selected_dir = cli.selected_dir.resolve()
    reference_path = cli.reference_export.resolve()
    output_path = selected_dir / "export_summary.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite export summary: {output_path}")
    selection_path = selected_dir / "selection.json"
    selection = load(selection_path)
    if selection.get("selected_stage") != "source":
        raise ValueError("cached source export is valid only for a source fallback")
    source_checkpoint = source_run / "best_checkpoint.pt"
    selected_checkpoint = selected_dir / "selected_checkpoint.pt"
    source_hash = sha256_file(source_checkpoint)
    if sha256_file(selected_checkpoint) != source_hash:
        raise ValueError("selected fallback checkpoint differs from source")
    reference = load(reference_path)
    if reference.get("status") != "completed":
        raise ValueError("reference export is incomplete")
    if reference.get("checkpoint_sha256") != source_hash:
        raise ValueError("reference export checkpoint differs from source")
    if reference.get("verification_split") != "calibration":
        raise ValueError("source export cache must use calibration verification")
    if reference.get("test_used") is not False:
        raise ValueError("source export cache did not keep test data sealed")
    export_dir = Path(reference["export_dir"]).resolve()
    synthesis_path = export_dir / "synthesis_verification.json"
    synthesis = load(synthesis_path)
    if synthesis.get("status") != "passed":
        raise ValueError("cached synthesis verification did not pass")
    if synthesis.get("data_policy", {}).get("test_used") is not False:
        raise ValueError("cached synthesis verification used test data")
    result = dict(reference)
    result.update(
        {
            "checkpoint": selected_checkpoint.name,
            "checkpoint_sha256": source_hash,
            "method_dir": str(selected_dir),
            "status": "completed",
            "reused_source_export": True,
            "reused_export_summary": str(reference_path),
            "reused_export_summary_sha256": sha256_file(reference_path),
            "reused_synthesis_verification": str(synthesis_path),
            "reused_synthesis_verification_sha256": sha256_file(synthesis_path),
            "reuse_justification": (
                "The selected checkpoint is byte-identical to the source and the "
                "reference used the same calibration split and exact tool flow."
            ),
            "validation_used": False,
            "calibration_used": True,
            "test_used": False,
        }
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
