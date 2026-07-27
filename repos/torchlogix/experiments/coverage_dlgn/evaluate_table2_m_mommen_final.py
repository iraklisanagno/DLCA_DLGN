#!/usr/bin/env python3
"""Evaluate the frozen one-seed CIFAR-10 M Mommen checkpoint exactly once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table2_m_mommen_final.json"
RUN_DIR = Path(
    "experiments/coverage_dlgn/results/"
    "final_table2_cifar10_m_mommen_nc8_seed0"
)
EVALUATOR = Path("experiments/coverage_dlgn/evaluate_checkpoint.py")
LOG_PATH = (
    ROOT / "logs" / "table2_m_mommen_final" / "test"
    / "final_table2_cifar10_m_mommen_nc8_seed0.test.log"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    args = parser.parse_args()

    if not SUMMARY_PATH.is_file():
        raise RuntimeError("freeze validation summary before held-out test")
    summary = json.loads(SUMMARY_PATH.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("held-out test was already evaluated")
    if not (RUN_DIR / "best_checkpoint.pt").is_file():
        raise RuntimeError("frozen best-validation checkpoint is missing")
    if (RUN_DIR / "test_metrics.json").exists():
        raise RuntimeError("refusing overwrite: test_metrics.json exists")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["DATASET_PATH"] = str(args.data_path)
    with LOG_PATH.open("w") as handle:
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                str(RUN_DIR),
                "--device",
                "cuda",
            ],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    payload = {
        "cell": "table2_cifar10_m_mommen",
        "test_set_used": completed.returncode == 0,
        "expected_count": 1,
        "completed_count": int(completed.returncode == 0),
        "return_code": completed.returncode,
        "run": RUN_DIR.name,
        "log": str(LOG_PATH),
    }
    output = LOG_PATH.parent / "test_evaluation_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
