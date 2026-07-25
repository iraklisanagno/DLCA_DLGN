#!/usr/bin/env python3
"""Evaluate a frozen Table 1 cell exactly once on held-out test data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


QUEUE_ROOT = Path("experiments/coverage_dlgn/queues")
SUMMARY_ROOT = Path("experiments/coverage_dlgn/summary")
LOG_ROOT = Path("experiments/coverage_dlgn/logs")
EVALUATOR = Path("experiments/coverage_dlgn/evaluate_checkpoint.py")


def round_robin(items: list[Path], buckets: int) -> list[list[Path]]:
    assignments = [[] for _ in range(buckets)]
    for index, item in enumerate(items):
        assignments[index % buckets].append(item)
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", choices=["mnist", "fashion"])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    return parser.parse_args()


def evaluate_gpu(
    gpu: int,
    run_dirs: list[Path],
    data_path: Path,
    log_dir: Path,
) -> list[dict]:
    records = []
    for run_dir in run_dirs:
        log_path = log_dir / f"{run_dir.name}.test.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["DATASET_PATH"] = str(data_path)
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
        record = {
            "name": run_dir.name,
            "gpu": gpu,
            "return_code": completed.returncode,
            "log": str(log_path),
        }
        records.append(record)
        print(
            f"test gpu={gpu} rc={completed.returncode} "
            f"name={run_dir.name}",
            flush=True,
        )
        if completed.returncode != 0:
            break
    return records


def main() -> int:
    args = parse_args()
    queue_path = QUEUE_ROOT / f"table1_final_{args.cell}.json"
    final_summary_path = (
        SUMMARY_ROOT / f"table1_{args.cell}_final.json"
    )
    if not final_summary_path.is_file():
        raise RuntimeError(
            f"freeze and summarize validation before test: "
            f"{final_summary_path}"
        )
    final_summary = json.loads(final_summary_path.read_text())
    if final_summary["test_set_used"]:
        raise RuntimeError("held-out test was already evaluated")

    queue = json.loads(queue_path.read_text())
    run_dirs = [
        Path(entry["output"]) for entry in queue["entries"]
    ]
    missing_checkpoints = [
        str(run_dir)
        for run_dir in run_dirs
        if not (run_dir / "best_checkpoint.pt").is_file()
    ]
    if missing_checkpoints:
        raise RuntimeError(
            f"missing frozen checkpoints: {missing_checkpoints}"
        )
    existing = [
        str(run_dir / "test_metrics.json")
        for run_dir in run_dirs
        if (run_dir / "test_metrics.json").exists()
    ]
    if existing:
        raise RuntimeError(
            "refusing any overwrite; held-out results already exist: "
            + ", ".join(existing)
        )

    log_dir = LOG_ROOT / f"table1_final_{args.cell}" / "test"
    log_dir.mkdir(parents=True, exist_ok=True)
    assignments = round_robin(run_dirs, len(args.gpus))
    records = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                evaluate_gpu,
                gpu,
                assigned,
                args.data_path,
                log_dir,
            )
            for gpu, assigned in zip(args.gpus, assignments)
        ]
        for future in as_completed(futures):
            records.extend(future.result())

    failures = [
        record for record in records if record["return_code"] != 0
    ]
    completed_names = {
        record["name"]
        for record in records
        if record["return_code"] == 0
    }
    expected_names = {run_dir.name for run_dir in run_dirs}
    summary = {
        "cell": args.cell,
        "test_set_used": bool(completed_names),
        "expected_count": len(run_dirs),
        "completed_count": len(completed_names),
        "failures": failures,
        "missing": sorted(expected_names - completed_names),
        "records": sorted(records, key=lambda row: row["name"]),
    }
    summary_path = log_dir / "test_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)
    return 1 if failures or completed_names != expected_names else 0


if __name__ == "__main__":
    raise SystemExit(main())
