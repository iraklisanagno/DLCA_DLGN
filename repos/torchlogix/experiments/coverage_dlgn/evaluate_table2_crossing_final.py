#!/usr/bin/env python3
"""Evaluate the frozen 128K CIFAR-10 compression crossing exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.coverage_dlgn.evaluate_table1_final import (
        evaluate_gpu,
        round_robin,
    )
except ModuleNotFoundError:
    from evaluate_table1_final import evaluate_gpu, round_robin

from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = (
    ROOT / "summary" / "table2_cifar10_compression_crossing_final.json"
)
RESULT_ROOT = ROOT / "results"
LOG_DIR = ROOT / "logs" / "table2_cifar10_compression_crossing_final" / "test"
RUN_NAMES = [
    f"final_table2_cifar10_128k_{candidate}_seed{seed}"
    for candidate in ("random", "v3_swap0500")
    for seed in range(5)
]


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
    if not SUMMARY_PATH.is_file():
        raise RuntimeError(f"freeze validation before test: {SUMMARY_PATH}")
    summary = json.loads(SUMMARY_PATH.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("held-out test was already evaluated")

    run_dirs = [RESULT_ROOT / name for name in RUN_NAMES]
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
    expected = set(RUN_NAMES)
    payload = {
        "cell": "table2_cifar10_128k_compression_crossing",
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
