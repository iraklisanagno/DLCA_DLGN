#!/usr/bin/env python3
"""Benchmark hardened forward passes from frozen best-validation checkpoints."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model, load_dataset


def benchmark_run(run_dir: Path, device: str, warmup: int, repeats: int):
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = device
    torch.manual_seed(args.seed)

    payload = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
    state_dict = payload["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    _, _, test_loader = load_dataset(args)
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(inputs)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            import time
            start = time.perf_counter()
            for _ in range(repeats):
                model(inputs)
            total_ms = 1000 * (time.perf_counter() - start)

    result = {
        "checkpoint": "best_checkpoint.pt",
        "mode": "hardened",
        "device": device,
        "batch_size": len(inputs),
        "warmup_batches": warmup,
        "timed_batches": repeats,
        "input_transfer_excluded": True,
        "total_milliseconds": total_ms,
        "milliseconds_per_batch": total_ms / repeats,
        "microseconds_per_example": 1000 * total_ms / (repeats * len(inputs)),
        "examples_per_second": 1000 * repeats * len(inputs) / total_ms,
    }
    (run_dir / "inference_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    for run_dir in args.run_dirs:
        result = benchmark_run(
            run_dir,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(
            f"{run_dir.name}: {result['milliseconds_per_batch']:.4f} ms/batch, "
            f"{result['examples_per_second']:.1f} examples/s"
        )


if __name__ == "__main__":
    main()
