#!/usr/bin/env python3
"""Calibrate structural MarginSynth ranking against same-flow ABC results."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr, spearmanr

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model

from experiments.marginsynth.hardware_ranking import (
    HARDWARE_FEATURE_NAMES,
    StructuralFeatureIndex,
    StructuralHardwareModel,
    is_alternative_binary,
)
from experiments.marginsynth.unit_tying import logic_layers
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    write_artifact_manifest,
)


def repository_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["model_state_dict"]


def feature_sums(
    feature_index: StructuralFeatureIndex,
    source_ids: dict[int, torch.Tensor],
    candidate_ids: dict[int, torch.Tensor],
    eligible_layers: list[int],
) -> tuple[dict[str, float], int, int]:
    sums = {name: 0.0 for name in HARDWARE_FEATURE_NAMES}
    changes = 0
    alternative_binary = 0
    for layer_index in eligible_layers:
        old = source_ids[layer_index]
        new = candidate_ids[layer_index]
        changed = torch.nonzero(old != new, as_tuple=False).flatten()
        for unit in changed.tolist():
            old_lut = int(old[unit])
            new_lut = int(new[unit])
            features = feature_index.features(
                layer_index, unit, old_lut, new_lut
            )
            for name in HARDWARE_FEATURE_NAMES:
                sums[name] += float(features[name])
            alternative_binary += int(is_alternative_binary(old_lut, new_lut))
        changes += len(changed)
    return sums, changes, alternative_binary


def fit_coefficients(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    scale = np.maximum(np.linalg.norm(x, axis=0), 1.0)
    scaled = x / scale
    augmented_x = np.vstack(
        (scaled, math.sqrt(ridge) * np.eye(scaled.shape[1]))
    )
    augmented_y = np.concatenate((y, np.zeros(scaled.shape[1])))
    fitted = lsq_linear(
        augmented_x,
        augmented_y,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
    )
    if not fitted.success:
        raise RuntimeError(f"hardware calibration failed: {fitted.message}")
    return fitted.x / scale


def correlation_record(targets: np.ndarray, predictions: np.ndarray) -> dict:
    if len(targets) < 2 or np.allclose(targets, targets[0]) or np.allclose(
        predictions, predictions[0]
    ):
        pearson = spearman = None
    else:
        pearson = float(pearsonr(targets, predictions).statistic)
        spearman = float(spearmanr(targets, predictions).statistic)
    errors = predictions - targets
    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse_abc_nodes": float(np.sqrt(np.mean(errors**2))),
        "mean_absolute_error_abc_nodes": float(np.mean(np.abs(errors))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()
    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text())
    output_dir = cli.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite calibration: {output_dir}")
    output_dir.mkdir(parents=True)

    source_run = repository_path(config["source_run"])
    training = json.loads((source_run / "training_config.json").read_text())
    source_checkpoint = repository_path(config["source_checkpoint"])
    source_state = checkpoint_state(source_checkpoint)
    args = Namespace(**training)
    args.device = "cpu"
    model = get_model(source_state["0.thresholds"], args)
    model.load_state_dict(source_state, strict=True)
    layers = logic_layers(model)
    source_ids = {
        index: layer.weight.detach().argmax(1).cpu()
        for index, layer in enumerate(layers)
    }
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    feature_index = StructuralFeatureIndex(layers, source_ids)

    records = []
    for sample in config["samples"]:
        checkpoint = repository_path(sample["checkpoint"])
        export_summary = repository_path(sample["export_summary"])
        export = json.loads(export_summary.read_text())
        candidate_state = checkpoint_state(checkpoint)
        candidate = get_model(source_state["0.thresholds"], args)
        candidate.load_state_dict(candidate_state, strict=True)
        candidate_ids = {
            index: layer.weight.detach().argmax(1).cpu()
            for index, layer in enumerate(logic_layers(candidate))
        }
        sums, changes, alternative_binary = feature_sums(
            feature_index, source_ids, candidate_ids, eligible_layers
        )
        records.append(
            {
                "name": sample["name"],
                "fit_sample": bool(sample.get("fit_sample", True)),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "export_summary": str(export_summary),
                "export_summary_sha256": sha256_file(export_summary),
                "abc_and_nodes": int(export["abc_and_nodes"]),
                "changed_luts": changes,
                "alternative_binary_actions": alternative_binary,
                "features": sums,
            }
        )
        del candidate

    baseline_name = config["baseline_sample"]
    baseline = next(record for record in records if record["name"] == baseline_name)
    baseline_nodes = baseline["abc_and_nodes"]
    for record in records:
        record["abc_node_reduction"] = baseline_nodes - record["abc_and_nodes"]

    x_all = np.asarray(
        [
            [record["features"][name] for name in HARDWARE_FEATURE_NAMES]
            for record in records
        ],
        dtype=np.float64,
    )
    y_all = np.asarray(
        [record["abc_node_reduction"] for record in records], dtype=np.float64
    )
    fit_mask = np.asarray([record["fit_sample"] for record in records], dtype=bool)
    if int(fit_mask.sum()) < 3:
        raise ValueError("hardware calibration requires at least three fit samples")
    x = x_all[fit_mask]
    y = y_all[fit_mask]
    ridge = float(config.get("ridge", 1e-3))
    parameters = fit_coefficients(x, y, ridge)
    predictions = x_all @ parameters
    for record, prediction in zip(records, predictions):
        record["predicted_abc_node_reduction"] = float(prediction)
        record["prediction_error"] = float(
            prediction - record["abc_node_reduction"]
        )

    loo_predictions = []
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        loo_parameters = fit_coefficients(x[keep], y[keep], ridge)
        loo_predictions.append(float(x[held_out] @ loo_parameters))
    loo_predictions_array = np.asarray(loo_predictions)

    coefficients = {
        name: float(parameters[index])
        for index, name in enumerate(HARDWARE_FEATURE_NAMES)
    }
    metadata = {
        "protocol_name": config["protocol_name"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "baseline_abc_and_nodes": baseline_nodes,
        "ridge": ridge,
        "samples": records,
        "in_sample": correlation_record(y, predictions[fit_mask]),
        "leave_one_out": correlation_record(y, loo_predictions_array),
        "leave_one_out_predictions": loo_predictions,
        "held_out": correlation_record(
            y_all[~fit_mask], predictions[~fit_mask]
        ) if bool((~fit_mask).any()) else None,
        "feature_policy": {
            "operation_gain": "source AIG units minus candidate AIG units",
            "constant_propagation_gain": "static downstream AIG savings after exact Boolean cofactors",
            "fanout_log": "log(1 + direct fixed-topology fanout) for structural actions",
            "downstream_influence_log": "log(1 + path multiplicity to class reductions) for structural actions",
            "reconvergence": "downstream propagation is additive and therefore approximate",
        },
        "data_policy": {
            "uses_prior_seed0_development_synthesis": True,
            "unit_tying_used_for_fit": any(
                record["fit_sample"] and "unit_tying" in record["name"]
                for record in records
            ),
            "unit_tying_role": (
                "held-out estimator validation only; never used to fit coefficients"
            ),
            "dataset_loaded": False,
            "validation_loaded": False,
            "test_loaded": False,
        },
    }
    model_payload = StructuralHardwareModel(
        coefficients=coefficients,
        alternative_binary_penalty=float(
            config.get("alternative_binary_penalty", 2.0)
        ),
        metadata=metadata,
    ).to_dict()
    output_path = output_dir / "hardware_ranking_model.json"
    output_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    report = {
        "format_version": 1,
        "status": "completed",
        "protocol_name": config["protocol_name"],
        "model": output_path.name,
        "model_sha256": sha256_file(output_path),
        "coefficients": coefficients,
        "in_sample": metadata["in_sample"],
        "leave_one_out": metadata["leave_one_out"],
        "held_out": metadata["held_out"],
        "records": records,
        "data_policy": metadata["data_policy"],
        "software": {
            "source_revision": git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
