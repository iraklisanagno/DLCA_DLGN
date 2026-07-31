#!/usr/bin/env python3
"""Evaluate saved MarginSynth frontier circuits on validation only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)
from torchlogix import Circuit


def validation_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    reference_accuracy: float,
) -> dict:
    predictions = scores.argmax(axis=1)
    correct = predictions == labels
    flips = predictions != reference_predictions
    classes = scores.shape[1]
    label_counts = np.bincount(labels, minlength=classes)
    class_correct = np.bincount(labels, weights=correct, minlength=classes)
    per_class_accuracy = np.divide(
        class_correct,
        label_counts,
        out=np.zeros(classes, dtype=np.float64),
        where=label_counts != 0,
    )
    original_counts = np.bincount(reference_predictions, minlength=classes)
    class_flips = np.bincount(
        reference_predictions[flips],
        minlength=classes,
    )
    per_class_disagreement = np.divide(
        class_flips,
        original_counts,
        out=np.zeros(classes, dtype=np.float64),
        where=original_counts != 0,
    )
    accuracy = float(correct.mean())
    return {
        "accuracy": accuracy,
        "accuracy_loss": max(0.0, reference_accuracy - accuracy),
        "decision_flip_count": int(flips.sum()),
        "decision_flip_rate": float(flips.mean()),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "per_class_disagreement": per_class_disagreement.tolist(),
        "maximum_per_class_disagreement": float(
            per_class_disagreement.max(initial=0.0)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--search", required=True)
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--pack-bits", type=int, default=16)
    parser.add_argument("--compile-opt-level", type=int, default=0)
    cli = parser.parse_args()
    if cli.examples <= 0 or cli.examples % cli.pack_bits:
        raise ValueError("--examples must be positive and divisible by --pack-bits")

    run_dir = cli.run_dir.resolve()
    search_dir = run_dir / cli.search
    summary = json.loads((search_dir / "search_summary.json").read_text())
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint["model_state_dict"]
    model = get_model(state["0.thresholds"], args)
    model.load_state_dict(state, strict=True)
    model.eval()
    _, validation_loader, _, _ = load_dataset(args, include_calibration=True)
    images, labels_tensor = take_examples(validation_loader, cli.examples)
    with torch.no_grad():
        encoded = model[0](images).bool()
    labels = labels_tensor.numpy()

    baseline = Circuit.from_json_file(
        str(run_dir / "exact_simplified_circuit.json")
    )
    start = time.perf_counter()
    baseline.compile(
        opt_level=cli.compile_opt_level,
        pack_bits=cli.pack_bits,
    )
    baseline_scores = baseline(encoded.numpy(), use_compiled=True)
    baseline_predictions = baseline_scores.argmax(axis=1)
    baseline_accuracy = float((baseline_predictions == labels).mean())
    baseline_seconds = time.perf_counter() - start

    records = []
    seen = set()
    for point in summary["pareto"]:
        step = int(point["selected_step"])
        if step in seen:
            continue
        seen.add(step)
        snapshot = point["snapshot"]
        circuit_path = search_dir / snapshot / "circuit.json"
        circuit = Circuit.from_json_file(str(circuit_path))
        start = time.perf_counter()
        circuit.compile(
            opt_level=cli.compile_opt_level,
            pack_bits=cli.pack_bits,
        )
        scores = circuit(encoded.numpy(), use_compiled=True)
        seconds = time.perf_counter() - start
        records.append(
            {
                "step": step,
                "snapshot": snapshot,
                "circuit_sha256": sha256_file(circuit_path),
                "calibration_selection": point,
                "validation": validation_metrics(
                    scores,
                    labels,
                    baseline_predictions,
                    baseline_accuracy,
                ),
                "compiled_c_seconds": seconds,
            }
        )
    result = {
        "format_version": 1,
        "status": "passed",
        "search": cli.search,
        "baseline_validation_accuracy": baseline_accuracy,
        "baseline_compiled_c_seconds": baseline_seconds,
        "records": records,
        "data_policy": {
            "partition": "validation",
            "examples": cli.examples,
            "validation_indices_sha256": validation_loader.split_manifest[
                "partitions"
            ]["validation"]["indices_sha256"],
            "encoded_inputs_sha256": tensor_sha256(encoded),
            "labels_sha256": tensor_sha256(labels_tensor),
            "calibration_used_for_evaluation": False,
            "test_used": False,
        },
    }
    output = search_dir / "validation_frontier.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
