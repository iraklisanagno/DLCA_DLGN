#!/usr/bin/env python3
"""Freeze the matched 200K LogicTreeNet-M checkpoints before held-out test."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary" / "cifar10_paper_medium_200k_freeze.json"
RUNS = {
    "coverage_v4": RESULTS / "full_conv_cifar10_paper_medium_legacy_v4_seed0",
    "fixed_random": RESULTS / "full_conv_cifar10_paper_medium_random_seed0_200k",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = list(range(2000, 200001, 2000))
    observed = [int(row["step"]) for row in rows]
    if observed != expected:
        raise RuntimeError(f"incomplete 200K validation history: {path}")
    return rows


def normalized_configuration(config: dict) -> dict:
    normalized = dict(config)
    normalized["config"] = "<CONFIG>"
    normalized["num_iterations"] = 200000
    normalized["conv_connections_init_method"] = "<METHOD>"
    normalized["output"] = "<OUTPUT>"
    return normalized


def main() -> int:
    if SUMMARY.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {SUMMARY}")

    records = {}
    normalized = {}
    for method, run_dir in RUNS.items():
        checkpoint = run_dir / "best_checkpoint.pt"
        config_path = run_dir / "training_config.json"
        metrics_path = run_dir / "metrics.csv"
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"held-out test already exists: {run_dir}")
        for path in (checkpoint, config_path, metrics_path):
            if not path.is_file():
                raise RuntimeError(f"missing required artifact: {path}")

        config = json.loads(config_path.read_text())
        normalized[method] = normalized_configuration(config)
        rows = read_metrics(metrics_path)
        best = max(rows, key=lambda row: float(row["val_acc_discrete"]))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if int(metadata["step"]) != int(best["step"]):
            raise RuntimeError(f"checkpoint/metrics step mismatch: {run_dir}")
        if abs(
            float(metadata["metrics"]["val_acc_discrete"])
            - float(best["val_acc_discrete"])
        ) > 1e-12:
            raise RuntimeError(f"checkpoint/metrics accuracy mismatch: {run_dir}")
        records[method] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": "best_checkpoint.pt",
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "training_config_sha256": sha256(config_path),
            "metrics_sha256": sha256(metrics_path),
            "validation_evaluations": len(rows),
            "last_validation_step": int(rows[-1]["step"]),
            "selection_step": int(best["step"]),
            "validation_hard_accuracy": float(best["val_acc_discrete"]),
            "validation_relaxed_accuracy_at_selection": float(
                best["val_acc_relaxed"]
            ),
        }
        del payload

    if normalized["coverage_v4"] != normalized["fixed_random"]:
        differing = sorted(
            key
            for key in set(normalized["coverage_v4"]) | set(normalized["fixed_random"])
            if normalized["coverage_v4"].get(key)
            != normalized["fixed_random"].get(key)
        )
        raise RuntimeError(f"unexpected matched-config differences: {differing}")

    payload = {
        "phase": "cifar10_paper_medium_200k_paired",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "ClgnCifar10PaperMedium",
        "paper_identifier": "LogicTreeNet-M",
        "test_set_used": False,
        "test_queries_per_checkpoint": 0,
        "selection_metric": "maximum hardened validation accuracy",
        "intended_method_difference": {
            "coverage_v4": "semantic_channel_hybrid convolution routing",
            "fixed_random": "random convolution routing",
            "classifier_routing": "random for both",
        },
        "actual_budget_note": (
            "V4 was configured for 350K but stopped after its completed 200K "
            "evaluation; the control terminated normally at 200K."
        ),
        "runs": records,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
