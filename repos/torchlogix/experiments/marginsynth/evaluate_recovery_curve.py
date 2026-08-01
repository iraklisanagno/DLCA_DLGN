#!/usr/bin/env python3
"""Post-selection calibration/validation evaluation of every recovery snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.margin_aware_tying import constraint_metrics
from experiments.marginsynth.unit_tying import evaluate_encoded, metric_record
from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    take_examples,
    write_artifact_manifest,
)


def partition_metrics(model, encoded, labels, teacher_scores, batch_size, device):
    predictions = teacher_scores.argmax(1)
    baseline = metric_record(teacher_scores, labels, predictions)
    scores = evaluate_encoded(model, encoded, batch_size, device)
    return constraint_metrics(
        scores,
        labels,
        predictions,
        baseline["accuracy"],
        baseline["per_class_accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("recovery_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("recovery curve evaluation requires CUDA")
    device = torch.device("cuda")
    run_dir = cli.run_dir.resolve()
    recovery_dir = cli.recovery_dir.resolve()
    output = recovery_dir / "post_selection_curve.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    config = json.loads((recovery_dir / "config.json").read_text())
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    teacher_path = run_dir / config.get("teacher_checkpoint", "best_checkpoint.pt")
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_state = teacher_payload["model_state_dict"]
    thresholds = teacher_state["0.thresholds"]
    _, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    validation_images, validation_labels = take_examples(
        validation_loader, len(validation_loader.dataset)
    )
    encoder = get_model(thresholds, args)
    encoder.load_state_dict(teacher_state, strict=True)
    encoder.eval()
    with torch.no_grad():
        calibration_encoded = encoder[0](calibration_images).bool().cpu()
        validation_encoded = encoder[0](validation_images).bool().cpu()
    teacher = get_model(thresholds, args)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.to(device).eval()
    teacher_calibration = evaluate_encoded(
        teacher, calibration_encoded, cli.batch_size, device
    )
    teacher_validation = evaluate_encoded(
        teacher, validation_encoded, cli.batch_size, device
    )
    del teacher

    snapshot_records = json.loads((recovery_dir / "snapshot_metrics.json").read_text())
    selection = json.loads((recovery_dir / "sample_selection.json").read_text())
    rows = []
    for record in snapshot_records:
        checkpoint_path = recovery_dir / record["checkpoint"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = get_model(thresholds, args)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        rows.append(
            {
                "step": record["step"],
                "examples_processed": record["examples_processed"],
                "equivalent_recovery_epochs": (
                    record["examples_processed"]
                    / max(1, int(selection["recovery_examples"]))
                ),
                "elapsed_seconds": record["elapsed_seconds"],
                "hard_hardware_cost": record["hard_hardware_cost"],
                "changed_unlocked_rows_from_source": record[
                    "changed_unlocked_rows_from_source"
                ],
                "monitor": record["monitor"],
                "calibration": partition_metrics(
                    model,
                    calibration_encoded,
                    calibration_labels,
                    teacher_calibration,
                    cli.batch_size,
                    device,
                ),
                "validation": partition_metrics(
                    model,
                    validation_encoded,
                    validation_labels,
                    teacher_validation,
                    cli.batch_size,
                    device,
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
        del model
    payload = {
        "format_version": 1,
        "status": "completed",
        "purpose": "post-selection reporting only",
        "snapshot_selection_repeated": False,
        "validation_used_for_selection": False,
        "test_used": False,
        "teacher_checkpoint": str(teacher_path),
        "teacher_checkpoint_sha256": sha256_file(teacher_path),
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(recovery_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
