#!/usr/bin/env python3
"""Evaluate frozen promoted dense checkpoints once on held-out test data."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from experiments.coverage_dlgn.evaluate_table1_final import (
        evaluate_gpu,
        round_robin,
    )
except ModuleNotFoundError:
    from evaluate_table1_final import evaluate_gpu, round_robin


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "second_round_final_validation_freeze.json"
LOG_DIR = ROOT / "logs" / "second_round_final_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
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
    run_dirs = []
    reused = []
    for rows in freeze["groups"].values():
        for row in rows:
            run_dir = Path(row["run_dir"])
            test_path = run_dir / "test_metrics.json"
            if test_path.is_file():
                reused.append(run_dir.name)
            else:
                run_dirs.append(run_dir)
    if len(run_dirs) != len(set(run_dirs)):
        raise RuntimeError("duplicate pending run directory")
    missing = [
        str(run_dir) for run_dir in run_dirs
        if not (run_dir / "best_checkpoint.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing frozen checkpoints: {missing}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    assignments = round_robin(run_dirs, len(args.gpus))
    records = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                evaluate_gpu, gpu, assigned, args.data_path, LOG_DIR
            )
            for gpu, assigned in zip(args.gpus, assignments)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    failures = [row for row in records if row["return_code"] != 0]
    missing_after = [
        run_dir.name for run_dir in run_dirs
        if not (run_dir / "test_metrics.json").is_file()
    ]
    payload = {
        "validation_freeze": str(FREEZE),
        "cuda_gpus": args.gpus,
        "pending_at_start": len(run_dirs),
        "reused_existing": sorted(reused),
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
