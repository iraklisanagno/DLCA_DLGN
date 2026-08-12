#!/usr/bin/env python3
"""Evaluate frozen full-schedule convolutional S checkpoints once on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.coverage_dlgn.evaluate_table1_final import evaluate_gpu
except ModuleNotFoundError:
    from evaluate_table1_final import evaluate_gpu


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "second_round_convolutional_validation_freeze.json"
LOG_DIR = ROOT / "logs" / "second_round_convolutional_final_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FREEZE.is_file():
        raise RuntimeError(f"freeze validation before test: {FREEZE}")
    freeze = json.loads(FREEZE.read_text())
    if not freeze.get("validation_frozen"):
        raise RuntimeError("validation manifest is not frozen")

    run_dirs = [Path(row["run_dir"]) for row in freeze["runs"].values()]
    pending = [
        run_dir for run_dir in run_dirs
        if not (run_dir / "test_metrics.json").is_file()
    ]
    reused = sorted(set(run_dirs) - set(pending))
    missing = [
        str(run_dir) for run_dir in pending
        if not (run_dir / "best_checkpoint.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing frozen checkpoints: {missing}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    records = evaluate_gpu(args.gpu, pending, args.data_path, LOG_DIR)
    failures = [row for row in records if row["return_code"] != 0]
    missing_after = [
        run_dir.name for run_dir in pending
        if not (run_dir / "test_metrics.json").is_file()
    ]
    payload = {
        "validation_freeze": str(FREEZE),
        "cuda_gpu": args.gpu,
        "pending_at_start": len(pending),
        "reused_existing": [run_dir.name for run_dir in reused],
        "records": sorted(records, key=lambda row: row["name"]),
        "failures": failures,
        "missing_after": sorted(missing_after),
    }
    output = LOG_DIR / "test_evaluation_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 1 if failures or missing_after else 0


if __name__ == "__main__":
    raise SystemExit(main())
