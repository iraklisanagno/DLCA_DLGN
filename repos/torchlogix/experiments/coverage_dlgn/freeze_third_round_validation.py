#!/usr/bin/env python3
"""Freeze all third-round validation artifacts before held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_NAMES = (
    "third_u2_dense_cifar10_ml",
    "third_lilogic_cifar10",
    "third_bitlogic_cifar10",
)
OUTPUT = ROOT / "summary" / "third_round_validation_freeze.json"
REQUIRED_ARTIFACTS = (
    "training_config.json",
    "environment.json",
    "run_summary.json",
    "best_checkpoint.pt",
    "final_checkpoint.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_entry(phase: str, entry: dict) -> dict:
    run_dir = Path(entry["output"])
    missing = [
        str(run_dir / name)
        for name in REQUIRED_ARTIFACTS
        if not (run_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"cannot freeze {entry['name']}; missing {missing}")
    if (run_dir / "third_round_test_metrics.json").exists():
        raise RuntimeError(
            "held-out metrics existed before validation freeze: "
            f"{run_dir / 'third_round_test_metrics.json'}"
        )

    config = json.loads((run_dir / "training_config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    return {
        "phase": phase,
        "name": entry["name"],
        "family": entry["family"],
        "coordinate": entry["coordinate"],
        "seed": entry["seed"],
        "run_dir": str(run_dir),
        "architecture": config["architecture"],
        "best_hard_validation_pct": (
            100 * summary["best_validation_hard_accuracy"]
        ),
        "final_hard_validation_pct": (
            100 * summary["final_metrics"]["val_acc_discrete"]
        ),
        "source_tree_sha256": environment.get("source_tree_sha256"),
        "training_implementation_sha256": environment.get(
            "training_implementation_sha256"
        ),
        "artifacts": {
            name: {
                "sha256": sha256(run_dir / name),
                "bytes": (run_dir / name).stat().st_size,
            }
            for name in REQUIRED_ARTIFACTS
        },
        "test_metrics_existing_at_freeze": False,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite validation freeze: {OUTPUT}")
    groups = {}
    names = set()
    for phase in QUEUE_NAMES:
        queue_path = ROOT / "queues" / f"{phase}.json"
        queue = json.loads(queue_path.read_text())
        rows = []
        for entry in queue["entries"]:
            if entry["name"] in names:
                raise RuntimeError(f"duplicate run name: {entry['name']}")
            names.add(entry["name"])
            rows.append(freeze_entry(phase, entry))
        groups[phase] = rows

    payload = {
        "protocol": "THIRD_ROUND_PROTOCOL.md",
        "selection": (
            "predeclared best hardened-validation and final checkpoints"
        ),
        "validation_frozen": True,
        "held_out_test_access_added_by_this_script": False,
        "run_count": sum(len(rows) for rows in groups.values()),
        "checkpoint_count": 2 * sum(len(rows) for rows in groups.values()),
        "groups": groups,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
