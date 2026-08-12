#!/usr/bin/env python3
"""Benchmark hardened checkpoints on deterministic synthetic CUDA inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark(
    run_dir: Path,
    device: str,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    config_path = run_dir / "training_config.json"
    checkpoint = run_dir / "best_checkpoint.pt"
    config = json.loads(config_path.read_text())
    args = Namespace(**config)
    args.device = device
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    dataset = config["dataset"].lower()
    if dataset in {"mnist", "fashion-mnist"}:
        input_shape = (1, 28, 28)
    elif dataset in {"cifar-10", "cifar-100"}:
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"unsupported synthetic input shape for {dataset}")
    generator = torch.Generator().manual_seed(2027)
    inputs = torch.rand(
        batch_size, *input_shape, generator=generator
    ).to(device)
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(inputs)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(inputs)
            end.record()
            torch.cuda.synchronize(device)
            total_ms = start.elapsed_time(end)
    result = {
        "benchmark_implementation_version": 2,
        "checkpoint": "best_checkpoint.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "training_config_sha256": sha256(config_path),
        "mode": "hardened",
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "input_source": "torch.rand seed 2027; no dataset access",
        "input_shape": list(input_shape),
        "batch_size": batch_size,
        "warmup_batches": warmup,
        "timed_batches": repeats,
        "input_transfer_excluded": True,
        "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device),
        "total_milliseconds": total_ms,
        "milliseconds_per_batch": total_ms / repeats,
        "microseconds_per_example": (
            1000 * total_ms / (repeats * batch_size)
        ),
        "examples_per_second": (
            1000 * repeats * batch_size / total_ms
        ),
        "heldout_test_accessed": False,
    }
    output = run_dir / "synthetic_inference_benchmark_v2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.batch_size < 1:
        parser.error("warmup, repeats, and batch size must be positive")
    device = f"cuda:{args.gpu}"
    for run_dir in args.run_dirs:
        result = benchmark(
            run_dir,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(
            f"{run_dir.name}: {result['milliseconds_per_batch']:.4f} "
            f"ms/batch, {result['examples_per_second']:.1f} examples/s"
        )


if __name__ == "__main__":
    main()
