#!/usr/bin/env python3
"""Build a paper-ready validation/synthesis table from ablation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def selected_rewrite_counts(log: list[dict]) -> dict[str, int]:
    counts = {
        "coordinated_proposals": 0,
        "constant_edits": 0,
        "routing_edits": 0,
        "alternative_gate_edits": 0,
    }
    for entry in log:
        proposal = entry["proposal"]
        rewrites = (
            proposal["rewrites"]
            if proposal.get("proposal_type") == "group"
            else [proposal]
        )
        if proposal.get("proposal_type") == "group":
            counts["coordinated_proposals"] += 1
        for rewrite in rewrites:
            kind = rewrite["kind"]
            if kind.startswith("constant-"):
                counts["constant_edits"] += 1
            elif kind == "alternative-gate":
                counts["alternative_gate_edits"] += 1
            else:
                counts["routing_edits"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    manifest_path = run_dir / "ablation_configs" / "ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads((run_dir / "synthesis_verification.json").read_text())
    baseline_nodes = int(baseline["abc"]["stats"]["and_nodes"])
    baseline_live = int(baseline["hardware_circuit"]["logic_gates"])
    records = []
    for manifest_record in manifest["records"]:
        config_path = run_dir / manifest_record["config"]
        config = json.loads(config_path.read_text())
        search_dir = run_dir / config["output"]
        required = {
            "search_summary": search_dir / "search_summary.json",
            "rewrite_log": search_dir / "rewrite_log.json",
            "search_verification": search_dir / "search_verification.json",
            "validation_frontier": search_dir / "validation_frontier.json",
            "frontier_synthesis": search_dir / "frontier_synthesis.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise RuntimeError(
                f"ablation {manifest_record['variant']} is incomplete: {missing}"
            )
        verification = json.loads(required["search_verification"].read_text())
        if verification["status"] != "passed":
            raise RuntimeError(
                f"ablation {manifest_record['variant']} failed replay verification"
            )
        summary = json.loads(required["search_summary"].read_text())
        validation = json.loads(required["validation_frontier"].read_text())
        synthesis = json.loads(required["frontier_synthesis"].read_text())
        final_step = int(summary["accepted_proposals"])
        validation_matches = [
            item
            for item in validation["records"]
            if int(item["step"]) == final_step
        ]
        synthesis_matches = [
            item
            for item in synthesis["points"]
            if int(item["step"]) == final_step
        ]
        if len(validation_matches) != 1 or len(synthesis_matches) != 1:
            raise RuntimeError(
                f"ablation {manifest_record['variant']} lacks final-step results"
            )
        val = validation_matches[0]["validation"]
        synth = synthesis_matches[0]
        abc_nodes = int(synth["abc"]["stats"]["and_nodes"])
        live_gates = int(synth["live_gates"])
        record = {
            "variant": manifest_record["variant"],
            "accepted_proposals": final_step,
            **selected_rewrite_counts(
                json.loads(required["rewrite_log"].read_text())
            ),
            "calibration_accuracy_loss": summary["final_behavior"][
                "accuracy_loss"
            ],
            "calibration_disagreement": summary["final_behavior"][
                "decision_flip_rate"
            ],
            "validation_accuracy": val["accuracy"],
            "validation_accuracy_loss": val["accuracy_loss"],
            "validation_disagreement": val["decision_flip_rate"],
            "maximum_per_class_validation_disagreement": val[
                "maximum_per_class_disagreement"
            ],
            "live_gates": live_gates,
            "live_gate_reduction_fraction": 1 - live_gates / baseline_live,
            "abc_and_nodes": abc_nodes,
            "abc_reduction_fraction": 1 - abc_nodes / baseline_nodes,
            "abc_levels": int(synth["abc"]["stats"]["levels"]),
            "search_wall_seconds": summary["wall_seconds"],
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in required.items()
            },
        }
        records.append(record)

    payload = {
        "format_version": 1,
        "status": "completed",
        "partition": "validation",
        "test_used": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "baseline": {
            "abc_and_nodes": baseline_nodes,
            "live_gates": baseline_live,
        },
        "records": records,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = cli.output.with_suffix(".csv")
    scalar_keys = [
        key for key in records[0] if key != "artifacts"
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows(
            {key: record[key] for key in scalar_keys} for record in records
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
