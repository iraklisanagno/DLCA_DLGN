#!/usr/bin/env python3
"""Evaluate the frozen 256K/384K CIFAR-10 checkpoints exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "table2_compression_remaining_test"
SUMMARY = LOG_DIR / "evaluation_summary.json"


def targets() -> list[Path]:
    paths = []
    for budget in ("256k", "384k"):
        for family in ("random", "v3_incumbent"):
            for seed in (0, 1, 2):
                paths.append(
                    ROOT / "results"
                    / f"final_table2_cifar10_{budget}_{family}_seed{seed}"
                )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path", type=Path, default=Path("/tmp/torchlogix-datasets")
    )
    args = parser.parse_args()
    run_dirs = targets()
    missing = [
        str(path) for path in run_dirs
        if not (path / "best_checkpoint.pt").exists()
    ]
    if missing:
        raise RuntimeError(f"missing frozen checkpoints: {missing}")
    existing = [
        str(path) for path in run_dirs if (path / "test_metrics.json").exists()
    ]
    if existing:
        raise RuntimeError(
            "refusing to re-query held-out test; results already exist: "
            + ", ".join(existing)
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(run_dirs)
    available = list(args.gpus)
    running = {}
    finished = []
    failed = []
    started = time.perf_counter()
    while pending or running:
        while pending and available:
            gpu = available.pop(0)
            run_dir = pending.pop(0)
            log_path = LOG_DIR / f"{run_dir.name}.log"
            handle = log_path.open("w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["DATASET_PATH"] = str(args.data_path)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "evaluate_checkpoint.py"),
                    "--device",
                    "cuda",
                    str(run_dir),
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, handle, run_dir, log_path, time.perf_counter())
            print(f"launch gpu={gpu} run={run_dir.name}", flush=True)
        time.sleep(1.0)
        for gpu, state in list(running.items()):
            process, handle, run_dir, log_path, launched = state
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            record = {
                "name": run_dir.name,
                "gpu": gpu,
                "return_code": return_code,
                "elapsed_seconds": time.perf_counter() - launched,
                "log": str(log_path),
            }
            if return_code == 0 and (run_dir / "test_metrics.json").exists():
                finished.append(record)
            else:
                failed.append(record)
            del running[gpu]
            available.append(gpu)
            available.sort()
    payload = {
        "phase": "table2_compression_remaining_test",
        "heldout_test_used": True,
        "overwrite_protection": True,
        "wall_seconds": time.perf_counter() - started,
        "finished": finished,
        "failed": failed,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")
    print(SUMMARY)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
