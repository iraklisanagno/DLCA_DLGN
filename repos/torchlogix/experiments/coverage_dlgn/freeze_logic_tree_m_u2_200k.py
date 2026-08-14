#!/usr/bin/env python3
"""Freeze the 200K LogicTreeNet-M U2 validation selection before test."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "results" / "full_conv_cifar10_paper_medium_u2_seed0_200k"
OUTPUT = ROOT / "summary" / "cifar10_paper_medium_u2_200k_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {OUTPUT}")
    if (RUN_DIR / "test_metrics.json").exists():
        raise RuntimeError("held-out test was accessed before validation freeze")

    required = {
        "checkpoint": RUN_DIR / "best_checkpoint.pt",
        "config": RUN_DIR / "training_config.json",
        "environment": RUN_DIR / "environment.json",
        "metrics": RUN_DIR / "metrics.csv",
        "run_summary": RUN_DIR / "run_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze incomplete run; missing {missing}")

    with required["metrics"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_steps = list(range(2_000, 200_001, 2_000))
    observed_steps = [int(row["step"]) for row in rows]
    if observed_steps != expected_steps:
        raise RuntimeError("validation history is not complete through 200K")
    best = max(rows, key=lambda row: float(row["val_acc_discrete"]))

    config = json.loads(required["config"].read_text())
    expected_config = {
        "dataset": "cifar-10",
        "architecture": "ClgnCifar10PaperMedium",
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "num_iterations": 200_000,
        "eval_freq": 2_000,
        "connections": "fixed",
        "connections_init_method": "semantic_multiscale_balanced",
        "conv_connections_init_method": "semantic_multiscale_balanced",
        "classifier_connections_init_method": "semantic_multiscale_balanced",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"training configuration mismatch: {mismatches}")

    checkpoint = torch.load(
        required["checkpoint"], map_location="cpu", weights_only=False
    )
    metadata = checkpoint["metadata"]
    selected_step = int(best["step"])
    selected_accuracy = float(best["val_acc_discrete"])
    if int(metadata["step"]) != selected_step:
        raise RuntimeError("best checkpoint step does not match validation curve")
    if abs(
        float(metadata["metrics"]["val_acc_discrete"]) - selected_accuracy
    ) > 1e-12:
        raise RuntimeError("best checkpoint accuracy does not match validation curve")

    environment = json.loads(required["environment"].read_text())
    run_summary = json.loads(required["run_summary"].read_text())
    if abs(
        float(run_summary["best_validation_hard_accuracy"])
        - selected_accuracy
    ) > 1e-12:
        raise RuntimeError("run summary does not match validation selection")

    payload = {
        "phase": "cifar10_paper_medium_u2_200k",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "ClgnCifar10PaperMedium",
        "paper_identifier": "LogicTreeNet-M",
        "method": "CoverageDLGN U2 (semantic_multiscale_balanced)",
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "selection_metric": "maximum hardened validation accuracy",
        "validation_evaluations": len(rows),
        "last_validation_step": observed_steps[-1],
        "selected_checkpoint": "best_checkpoint.pt",
        "selected_step": selected_step,
        "selected_validation_hard_accuracy": selected_accuracy,
        "selected_validation_relaxed_accuracy": float(
            best["val_acc_relaxed"]
        ),
        "checkpoint_bytes": required["checkpoint"].stat().st_size,
        "checkpoint_sha256": sha256(required["checkpoint"]),
        "training_config_sha256": sha256(required["config"]),
        "metrics_sha256": sha256(required["metrics"]),
        "environment_sha256": sha256(required["environment"]),
        "source_revision": environment.get("source_revision"),
        "source_tree_sha256": environment.get("source_tree_sha256"),
        "training_implementation_sha256": environment.get(
            "training_implementation_sha256"
        ),
        "test_set_used": False,
        "heldout_checkpoint_queries": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(
        f"frozen step={selected_step} "
        f"validation_hard={selected_accuracy:.4%} "
        f"sha256={payload['checkpoint_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
