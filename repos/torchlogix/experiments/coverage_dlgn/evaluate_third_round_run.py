#!/usr/bin/env python3
"""Evaluate both frozen third-round checkpoints once on held-out CIFAR-10."""

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

from utils import evaluate_model, get_model, load_dataset  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_frozen_row(freeze: dict, run_dir: Path) -> dict:
    matches = [
        row
        for rows in freeze["groups"].values()
        for row in rows
        if Path(row["run_dir"]).resolve() == run_dir.resolve()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one frozen row for {run_dir}, found {len(matches)}"
        )
    return matches[0]


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: Namespace,
    test_loader,
) -> dict:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = payload["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)

    functions = {
        "loss": torch.nn.CrossEntropyLoss(),
        "acc": lambda predictions, labels: (
            predictions.argmax(-1) == labels
        ).to(torch.float32),
    }
    hard = evaluate_model(
        model, test_loader, functions, mode="eval", device=args.device
    )
    relaxed = evaluate_model(
        model, test_loader, functions, mode="train", device=args.device
    )
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation_step": payload["metadata"]["step"],
        "validation_hard_accuracy": payload["metadata"]["metrics"][
            "val_acc_discrete"
        ],
        "test_hard_accuracy": hard["acc"],
        "test_hard_loss": hard["loss"],
        "test_relaxed_accuracy": relaxed["acc"],
        "test_relaxed_loss": relaxed["loss"],
        "test_examples": len(test_loader.dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parsed = parser.parse_args()

    output = parsed.run_dir / "third_round_test_metrics.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite held-out metrics: {output}")
    freeze = json.loads(parsed.freeze.read_text())
    if not freeze.get("validation_frozen"):
        raise RuntimeError("validation manifest is not frozen")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for third-round held-out evaluation")
    row = find_frozen_row(freeze, parsed.run_dir)
    for name in row["artifacts"]:
        actual = sha256(parsed.run_dir / name)
        expected = row["artifacts"][name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {name}")

    config = json.loads(
        (parsed.run_dir / "training_config.json").read_text()
    )
    args = Namespace(**config)
    args.device = parsed.device
    _, _, test_loader = load_dataset(args)
    checkpoints = [
        evaluate_checkpoint(parsed.run_dir / name, args, test_loader)
        for name in ("best_checkpoint.pt", "final_checkpoint.pt")
    ]
    result = {
        "protocol": "THIRD_ROUND_PROTOCOL.md",
        "validation_freeze": str(parsed.freeze),
        "run_name": row["name"],
        "device": parsed.device,
        "heldout_checkpoint_queries": 2,
        "checkpoints": checkpoints,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"{row['name']}: best={checkpoints[0]['test_hard_accuracy']:.4%}, "
        f"final={checkpoints[1]['test_hard_accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
