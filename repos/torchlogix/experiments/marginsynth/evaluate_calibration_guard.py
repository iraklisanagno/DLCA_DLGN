#!/usr/bin/env python3
"""Evaluate a checkpoint only on a prespecified calibration guard.

The guard is reconstructed from the partition seed and fractions used by the
source resynthesis pass.  Validation and test examples are never consumed.
This lets short recovery be scored on examples unseen by both its aggressive
resynthesis source and its recovery gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.circuit_distillation import (
    stratified_optimization_repair_guard_split,
)
from experiments.marginsynth.margin_aware_tying import (
    constraint_metrics,
    stratified_fold_ids,
)
from experiments.marginsynth.unit_tying import evaluate_encoded, metric_record
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
)


def metrics_for_indices(
    candidate,
    encoded,
    labels,
    teacher_scores,
    indices,
    batch_size,
    device,
):
    selected_labels = labels[indices]
    selected_teacher = teacher_scores[indices]
    reference_predictions = selected_teacher.argmax(1)
    baseline = metric_record(
        selected_teacher, selected_labels, reference_predictions
    )
    candidate_scores = evaluate_encoded(
        candidate, encoded[indices], batch_size, device
    )
    return constraint_metrics(
        candidate_scores,
        selected_labels,
        reference_predictions,
        baseline["accuracy"],
        baseline["per_class_accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--partition-config", required=True, type=Path)
    parser.add_argument("--output", default="bayesian_guard_evaluation.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--reporting-folds", type=int, default=4)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("calibration-guard evaluation requires CUDA")
    device = torch.device("cuda")
    run_dir = cli.run_dir.resolve()
    method_dir = cli.method_dir.resolve()
    output_path = method_dir / cli.output
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    partition_config = json.loads(cli.partition_config.read_text())
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"

    teacher_path = run_dir / partition_config.get(
        "teacher_checkpoint", "best_checkpoint.pt"
    )
    checkpoint_path = method_dir / cli.checkpoint
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    candidate_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    teacher_state = teacher_payload["model_state_dict"]
    candidate_state = candidate_payload["model_state_dict"]
    thresholds = teacher_state["0.thresholds"]
    if not torch.equal(thresholds, candidate_state["0.thresholds"]):
        raise ValueError("candidate and teacher thresholds differ")

    _, _, calibration_loader, _ = load_dataset(args, include_calibration=True)
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    encoder = get_model(thresholds, args)
    encoder.load_state_dict(teacher_state, strict=True)
    encoder.eval()
    with torch.no_grad():
        calibration_encoded = encoder[0](calibration_images).bool().cpu()
    del encoder, calibration_images

    teacher = get_model(thresholds, args)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.to(device).eval()
    teacher_scores = evaluate_encoded(
        teacher, calibration_encoded, cli.batch_size, device
    )
    del teacher
    candidate = get_model(thresholds, args)
    candidate.load_state_dict(candidate_state, strict=True)
    candidate.to(device).eval()

    partition_seed = int(
        partition_config.get("partition_seed", partition_config["seed"])
    )
    optimize, repair, guard = stratified_optimization_repair_guard_split(
        calibration_labels,
        float(partition_config["optimization_fraction"]),
        float(partition_config["repair_fraction"]),
        partition_seed + 104729,
    )
    aggregate = metrics_for_indices(
        candidate,
        calibration_encoded,
        calibration_labels,
        teacher_scores,
        guard,
        cli.batch_size,
        device,
    )
    guard_labels = calibration_labels[guard]
    fold_ids = stratified_fold_ids(
        guard_labels.numpy(), cli.reporting_folds, partition_seed + 65537
    )
    folds = []
    for fold in range(cli.reporting_folds):
        local = torch.from_numpy(np.flatnonzero(fold_ids == fold)).long()
        indices = guard[local]
        folds.append(
            {
                "fold": fold,
                "size": len(indices),
                "indices_sha256": tensor_sha256(indices),
                "metrics": metrics_for_indices(
                    candidate,
                    calibration_encoded,
                    calibration_labels,
                    teacher_scores,
                    indices,
                    cli.batch_size,
                    device,
                ),
            }
        )

    result = {
        "format_version": 1,
        "status": "completed",
        "dataset": training_config["dataset"],
        "architecture": training_config["architecture"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "teacher_checkpoint": str(teacher_path),
        "teacher_checkpoint_sha256": sha256_file(teacher_path),
        "partition_config": str(cli.partition_config.resolve()),
        "partition_config_sha256": sha256_file(cli.partition_config),
        "partition_seed": partition_seed,
        "partition_sizes": {
            "optimization": len(optimize),
            "repair": len(repair),
            "guard": len(guard),
        },
        "guard_indices_sha256": tensor_sha256(guard),
        "calibration_partition_indices_sha256": calibration_loader.split_manifest[
            "partitions"
        ]["calibration"]["indices_sha256"],
        "calibration_labels_sha256": tensor_sha256(calibration_labels),
        "guard": aggregate,
        "reporting_folds": folds,
        "data_policy": {
            "calibration_guard_only": True,
            "validation_loaded": False,
            "test_loaded": False,
            "used_for_bayesian_selection": True,
        },
        "software": {
            "source_revision": git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "cuda_device": torch.cuda.get_device_name(device),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
