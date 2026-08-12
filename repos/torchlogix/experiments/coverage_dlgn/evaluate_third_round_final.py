#!/usr/bin/env python3
"""Evaluate the frozen third-round cohort once across CUDA GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "summary" / "third_round_validation_freeze.json"
EVALUATOR = ROOT / "evaluate_third_round_run.py"
LOG_DIR = ROOT / "logs" / "third_round_final_test"


def round_robin(items: list[Path], buckets: int) -> list[list[Path]]:
    assignments = [[] for _ in range(buckets)]
    for index, item in enumerate(items):
        assignments[index % buckets].append(item)
    return assignments


def evaluate_gpu(
    gpu: int, run_dirs: list[Path], data_path: Path
) -> list[dict]:
    records = []
    for run_dir in run_dirs:
        output = run_dir / "third_round_test_metrics.json"
        if output.is_file():
            records.append({
                "name": run_dir.name,
                "gpu": gpu,
                "return_code": 0,
                "reused_existing": True,
            })
            continue
        log_path = LOG_DIR / f"{run_dir.name}.test.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["DATASET_PATH"] = str(data_path)
        with log_path.open("w") as log_handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    str(run_dir),
                    "--freeze",
                    str(FREEZE),
                    "--device",
                    "cuda",
                ],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        records.append({
            "name": run_dir.name,
            "gpu": gpu,
            "return_code": completed.returncode,
            "reused_existing": False,
            "log": str(log_path),
        })
        print(
            f"test gpu={gpu} rc={completed.returncode} "
            f"name={run_dir.name}",
            flush=True,
        )
        if completed.returncode != 0:
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text())
    if not freeze.get("validation_frozen"):
        raise RuntimeError("validation manifest is not frozen")
    run_dirs = [
        Path(row["run_dir"])
        for rows in freeze["groups"].values()
        for row in rows
    ]
    if len(run_dirs) != freeze["run_count"]:
        raise RuntimeError("freeze run count mismatch")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    assignments = round_robin(run_dirs, len(args.gpus))
    records = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(evaluate_gpu, gpu, assigned, args.data_path)
            for gpu, assigned in zip(args.gpus, assignments)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    failures = [row for row in records if row["return_code"] != 0]
    missing = [
        run_dir.name
        for run_dir in run_dirs
        if not (run_dir / "third_round_test_metrics.json").is_file()
    ]
    summary = {
        "validation_freeze": str(FREEZE),
        "cuda_gpus": args.gpus,
        "expected_runs": len(run_dirs),
        "completed_runs": len(run_dirs) - len(missing),
        "failures": failures,
        "missing": missing,
        "records": sorted(records, key=lambda row: row["name"]),
    }
    output = LOG_DIR / "test_evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
