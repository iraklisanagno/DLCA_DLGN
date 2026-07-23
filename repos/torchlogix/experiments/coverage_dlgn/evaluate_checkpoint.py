#!/usr/bin/env python3
"""Evaluate frozen best-validation checkpoints once on the held-out test set."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import evaluate_model, get_model, load_dataset


def evaluate_run(run_dir: Path, device: str):
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = device

    payload = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
    state_dict = payload["model_state_dict"]
    thresholds = state_dict["0.thresholds"]
    model = get_model(thresholds, args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    _, _, test_loader = load_dataset(args)
    loss = torch.nn.CrossEntropyLoss()
    functions = {
        "loss": loss,
        "acc": lambda predictions, labels: (
            predictions.argmax(-1) == labels
        ).to(torch.float32),
    }
    hard = evaluate_model(model, test_loader, functions, mode="eval", device=device)
    relaxed = evaluate_model(model, test_loader, functions, mode="train", device=device)
    result = {
        "checkpoint": "best_checkpoint.pt",
        "validation_selection_step": payload["metadata"]["step"],
        "validation_hard_accuracy": payload["metadata"]["metrics"]["val_acc_discrete"],
        "test_hard_accuracy": hard["acc"],
        "test_hard_loss": hard["loss"],
        "test_relaxed_accuracy": relaxed["acc"],
        "test_relaxed_loss": relaxed["loss"],
        "test_examples": len(test_loader.dataset),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        result = evaluate_run(run_dir, args.device)
        print(
            f"{run_dir.name}: test hard={result['test_hard_accuracy']:.4%}, "
            f"relaxed={result['test_relaxed_accuracy']:.4%}"
        )


if __name__ == "__main__":
    main()
