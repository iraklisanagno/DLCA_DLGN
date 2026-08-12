#!/usr/bin/env python3
"""Run the auditable third-round benchmark, freeze, test, and summary chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_NAMES = (
    "third_u2_dense_cifar10_ml",
    "third_lilogic_cifar10",
    "third_bitlogic_cifar10",
)
FREEZE = ROOT / "summary" / "third_round_validation_freeze.json"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def all_run_dirs() -> list[Path]:
    run_dirs = []
    for name in QUEUE_NAMES:
        queue = json.loads((ROOT / "queues" / f"{name}.json").read_text())
        run_dirs.extend(Path(entry["output"]) for entry in queue["entries"])
    if len(run_dirs) != 38 or len(run_dirs) != len(set(run_dirs)):
        raise RuntimeError("expected 38 unique third-round run directories")
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/tmp/torchlogix-datasets"),
    )
    args = parser.parse_args()
    run_dirs = all_run_dirs()
    incomplete = [
        run_dir.name
        for run_dir in run_dirs
        if not (run_dir / "run_summary.json").is_file()
    ]
    if incomplete:
        raise RuntimeError(f"training is incomplete: {incomplete}")

    missing_benchmarks = [
        run_dir
        for run_dir in run_dirs
        if not (run_dir / "synthetic_inference_benchmark_v2.json").is_file()
    ]
    for index, run_dir in enumerate(missing_benchmarks):
        run([
            sys.executable,
            str(ROOT / "benchmark_checkpoint_synthetic.py"),
            str(run_dir),
            "--gpu",
            str(args.gpus[index % len(args.gpus)]),
        ])

    if not FREEZE.is_file():
        run([sys.executable, str(ROOT / "freeze_third_round_validation.py")])
    run([
        sys.executable,
        str(ROOT / "evaluate_third_round_final.py"),
        "--gpus",
        *(str(gpu) for gpu in args.gpus),
        "--data-path",
        str(args.data_path),
    ])
    run([sys.executable, str(ROOT / "summarize_third_round.py")])


if __name__ == "__main__":
    main()
