#!/usr/bin/env python3
"""Evaluate the two frozen LogicTreeNet-M checkpoints once on CIFAR-10 test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "cifar10_paper_medium_200k_freeze.json"
LOG_DIR = ROOT / "logs" / "cifar10_paper_medium_200k" / "test"
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
        raise RuntimeError(f"freeze validation checkpoints first: {FREEZE}")
    freeze = json.loads(FREEZE.read_text())
    if freeze["test_set_used"]:
        raise RuntimeError("freeze manifest unexpectedly marks test as used")

    run_dirs = []
    for method in ("coverage_v4", "fixed_random"):
        record = freeze["runs"][method]
        run_dir = ROOT / record["run_dir"]
        checkpoint = run_dir / record["checkpoint"]
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint hash changed: {checkpoint}")
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"refusing a second test query: {run_dir}")
        run_dirs.append((method, run_dir))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["DATASET_PATH"] = str(args.data_path)
    records = []
    for method, run_dir in run_dirs:
        log_path = LOG_DIR / f"{run_dir.name}.test.log"
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
        records.append(
            {
                "method": method,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "gpu": args.gpu,
                "return_code": completed.returncode,
                "log": str(log_path.relative_to(ROOT)),
            }
        )
        print(f"{method}: return_code={completed.returncode}", flush=True)
        if completed.returncode != 0:
            break

    completed_methods = {
        row["method"] for row in records if row["return_code"] == 0
    }
    summary = {
        "phase": "cifar10_paper_medium_200k_paired",
        "test_set_used": bool(completed_methods),
        "expected_count": 2,
        "completed_count": len(completed_methods),
        "missing": sorted({"coverage_v4", "fixed_random"} - completed_methods),
        "records": records,
    }
    output = LOG_DIR / "test_evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0 if len(completed_methods) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
