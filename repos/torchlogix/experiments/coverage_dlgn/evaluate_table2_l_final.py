#!/usr/bin/env python3
"""Evaluate all frozen CIFAR-10 L checkpoints exactly once."""

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
PRIMARY_QUEUE = ROOT / "queues" / "table2_l_final.json"
MOMMEN_QUEUE = ROOT / "queues" / "table2_l_mommen_final.json"
PRIMARY_SUMMARY = ROOT / "summary" / "table2_l_final.json"
MOMMEN_SUMMARY = ROOT / "summary" / "table2_l_mommen_final.json"
LOG_DIR = ROOT / "logs" / "table2_l_final" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (PRIMARY_SUMMARY, MOMMEN_SUMMARY):
        if not path.is_file():
            raise RuntimeError(f"freeze validation before test: {path}")
        summary = json.loads(path.read_text())
        if summary["test_set_used"]:
            raise RuntimeError(f"held-out test was already evaluated: {path}")

    entries = []
    for path in (PRIMARY_QUEUE, MOMMEN_QUEUE):
        entries.extend(json.loads(path.read_text())["entries"])
    run_dirs = [Path(entry["output"]) for entry in entries]
    if len(run_dirs) != 11 or len(set(run_dirs)) != 11:
        raise RuntimeError(f"expected 11 unique frozen runs, got {len(run_dirs)}")
    missing = [
        str(run_dir) for run_dir in run_dirs
        if not (run_dir / "best_checkpoint.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing frozen checkpoints: {missing}")
    existing = [
        str(run_dir / "test_metrics.json") for run_dir in run_dirs
        if (run_dir / "test_metrics.json").exists()
    ]
    if existing:
        raise RuntimeError(
            "refusing overwrite; held-out results already exist: "
            + ", ".join(existing)
        )

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
    completed = {
        row["name"] for row in records if row["return_code"] == 0
    }
    expected = {run_dir.name for run_dir in run_dirs}
    payload = {
        "cell": "table2_cifar10_l",
        "test_set_used": bool(completed),
        "expected_count": len(expected),
        "completed_count": len(completed),
        "failures": failures,
        "missing": sorted(expected - completed),
        "records": sorted(records, key=lambda row: row["name"]),
    }
    output = LOG_DIR / "test_evaluation_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 1 if failures or completed != expected else 0


if __name__ == "__main__":
    raise SystemExit(main())
