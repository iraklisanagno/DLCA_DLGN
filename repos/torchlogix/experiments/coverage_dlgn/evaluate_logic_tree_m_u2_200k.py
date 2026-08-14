#!/usr/bin/env python3
"""Evaluate the frozen LogicTreeNet-M U2 checkpoint once on CIFAR-10 test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "cifar10_paper_medium_u2_200k_freeze.json"
LOG_DIR = ROOT / "logs" / "cifar10_paper_medium_u2_200k" / "test"
EVALUATOR = ROOT / "evaluate_checkpoint.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--data-path", type=Path, default=Path("/tmp/torchlogix-datasets")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FREEZE.is_file():
        raise RuntimeError(f"freeze validation selection first: {FREEZE}")
    freeze = json.loads(FREEZE.read_text())
    if freeze["test_set_used"] or freeze["heldout_checkpoint_queries"] != 0:
        raise RuntimeError("freeze manifest indicates prior held-out test access")

    run_dir = ROOT / freeze["run_dir"]
    checkpoint = run_dir / freeze["selected_checkpoint"]
    test_metrics = run_dir / "test_metrics.json"
    if sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash changed")
    if test_metrics.exists():
        raise RuntimeError(f"refusing a second test query: {test_metrics}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "evaluation.log"
    if log_path.exists():
        raise RuntimeError(f"refusing to overwrite evaluation log: {log_path}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["DATASET_PATH"] = str(args.data_path)
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                str(run_dir),
                "--device",
                "cuda",
            ],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not test_metrics.is_file():
        raise RuntimeError(
            f"test evaluation failed with return code {completed.returncode}; "
            f"inspect {log_path}"
        )

    result = json.loads(test_metrics.read_text())
    if int(result["validation_selection_step"]) != int(
        freeze["selected_step"]
    ):
        raise RuntimeError("evaluated checkpoint selection step changed")
    if abs(
        float(result["validation_hard_accuracy"])
        - float(freeze["selected_validation_hard_accuracy"])
    ) > 1e-12:
        raise RuntimeError("evaluated checkpoint validation metric changed")

    summary = {
        "phase": "cifar10_paper_medium_u2_200k",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": args.gpu,
        "run_dir": freeze["run_dir"],
        "checkpoint": freeze["selected_checkpoint"],
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "selection_step": freeze["selected_step"],
        "validation_hard_accuracy": freeze[
            "selected_validation_hard_accuracy"
        ],
        "heldout_checkpoint_queries": 1,
        "test_examples": result["test_examples"],
        "test_hard_accuracy": result["test_hard_accuracy"],
        "test_hard_loss": result["test_hard_loss"],
        "test_relaxed_accuracy": result["test_relaxed_accuracy"],
        "test_relaxed_loss": result["test_relaxed_loss"],
        "test_metrics_sha256": sha256(test_metrics),
        "log": str(log_path.relative_to(ROOT)),
    }
    output = LOG_DIR / "test_evaluation_summary.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation summary: {output}")
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"test hard={result['test_hard_accuracy']:.4%}, "
        f"relaxed={result['test_relaxed_accuracy']:.4%}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
