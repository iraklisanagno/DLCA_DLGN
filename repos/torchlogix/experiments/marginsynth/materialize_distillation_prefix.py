#!/usr/bin/env python3
"""Materialize a replayable intermediate prefix from a distillation run."""

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

from experiments.marginsynth.circuit_distillation import materialize_change_prefix
from experiments.marginsynth.margin_aware_tying import constraint_metrics
from experiments.marginsynth.unit_tying import evaluate_encoded, logic_layers, metric_record
from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    take_examples,
    write_artifact_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("distillation_dir", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    source_dir = cli.distillation_dir.resolve()
    output_dir = (run_dir / cli.output).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    config = json.loads((source_dir / "config.json").read_text())
    changes = json.loads((source_dir / "learned_changes.json").read_text())
    if cli.count < 0 or cli.count > len(changes):
        raise ValueError("count is outside learned change prefix")
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    teacher_path = run_dir / config.get("teacher_checkpoint", "best_checkpoint.pt")
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_state = teacher_payload["model_state_dict"]
    thresholds = teacher_state["0.thresholds"]
    model = get_model(thresholds, args)
    model.load_state_dict(teacher_state, strict=True)
    layers = logic_layers(model)
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    original_ids = {
        index: layers[index].weight.detach().argmax(1).cpu()
        for index in eligible_layers
    }
    materialize_change_prefix(
        model,
        original_ids,
        eligible_layers,
        changes,
        cli.count,
        float(config["hard_logit"]),
    )
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "_export_lut_ids" not in key
    }
    checkpoint_path = output_dir / "prefix_checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": state,
            "metadata": {
                "method": "replayed-distillation-prefix",
                "retained_changes": cli.count,
                "source_distillation": str(source_dir),
                "source_changes_sha256": sha256_file(source_dir / "learned_changes.json"),
                "validation_used_for_selection": False,
                "test_used": False,
            },
        },
        checkpoint_path,
    )
    (output_dir / "prefix_changes.json").write_text(
        json.dumps(changes[: cli.count], indent=2, sort_keys=True) + "\n"
    )

    # Evaluation is report-only; count is a predeclared hardware-interpolation
    # pilot and no snapshot is selected from these values.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    validation_images, validation_labels = take_examples(
        validation_loader, len(validation_loader.dataset)
    )
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    with torch.no_grad():
        validation_encoded = model[0](validation_images).bool().cpu()
        calibration_encoded = model[0](calibration_images).bool().cpu()
    teacher = get_model(thresholds, args)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.to(device).eval()
    model.to(device).eval()
    rows = {}
    for name, encoded, labels in (
        ("calibration", calibration_encoded, calibration_labels),
        ("validation", validation_encoded, validation_labels),
    ):
        teacher_scores = evaluate_encoded(teacher, encoded, 512, device)
        reference_predictions = teacher_scores.argmax(1)
        baseline = metric_record(teacher_scores, labels, reference_predictions)
        scores = evaluate_encoded(model, encoded, 512, device)
        rows[name] = constraint_metrics(
            scores,
            labels,
            reference_predictions,
            baseline["accuracy"],
            baseline["per_class_accuracy"],
        )
    summary = {
        "format_version": 1,
        "status": "completed",
        "method": "replayed-distillation-prefix",
        "retained_changes": cli.count,
        "available_changes": len(changes),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration": rows["calibration"],
        "validation": rows["validation"],
        "validation_used_for_selection": False,
        "test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
