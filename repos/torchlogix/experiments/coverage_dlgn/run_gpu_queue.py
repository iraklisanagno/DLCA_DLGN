#!/usr/bin/env python3
"""Run independent TorchLogix configurations across a fixed GPU pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--data-path", type=Path, default=Path("/tmp/torchlogix-datasets")
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def is_complete(output: Path) -> bool:
    required = {
        "training_config.json",
        "environment.json",
        "metrics.csv",
        "run_summary.json",
    }
    return output.is_dir() and required.issubset(
        {path.name for path in output.iterdir()}
    )


def main() -> int:
    args = parse_args()
    with args.queue.open() as handle:
        queue_payload = json.load(handle)
    pending = []
    skipped = []
    for entry in queue_payload["entries"]:
        output = Path(entry["output"])
        if is_complete(output):
            skipped.append(entry["name"])
        elif output.exists():
            raise RuntimeError(
                f"incomplete output exists; classify it before retry: {output}"
            )
        else:
            pending.append(entry)

    log_dir = (
        Path("experiments/coverage_dlgn/logs")
        / queue_payload["phase"]
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, dict] = {}
    finished = []
    failed = []
    available = list(args.gpus)
    started = time.perf_counter()
    print(
        f"phase={queue_payload['phase']} pending={len(pending)} "
        f"skipped={len(skipped)} gpus={available}",
        flush=True,
    )

    while pending or running:
        while pending and available:
            gpu = available.pop(0)
            entry = pending.pop(0)
            log_path = log_dir / f"{entry['name']}.log"
            log_handle = log_path.open("w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["DATASET_PATH"] = str(args.data_path)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "experiments/train.py",
                    "--config",
                    entry["config"],
                ],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "entry": entry,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "started": time.perf_counter(),
            }
            print(
                f"launch gpu={gpu} pid={process.pid} name={entry['name']}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for gpu, state in list(running.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            state["log_handle"].close()
            elapsed = time.perf_counter() - state["started"]
            record = {
                "name": state["entry"]["name"],
                "gpu": gpu,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "log": str(state["log_path"]),
            }
            if return_code == 0 and is_complete(
                Path(state["entry"]["output"])
            ):
                finished.append(record)
                label = "complete"
            else:
                failed.append(record)
                label = "failed"
            print(
                f"{label} gpu={gpu} rc={return_code} "
                f"seconds={elapsed:.1f} name={state['entry']['name']}",
                flush=True,
            )
            del running[gpu]
            available.append(gpu)
            available.sort()

    summary = {
        "phase": queue_payload["phase"],
        "wall_seconds": time.perf_counter() - started,
        "skipped": skipped,
        "finished": finished,
        "failed": failed,
    }
    summary_path = log_dir / "queue_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
