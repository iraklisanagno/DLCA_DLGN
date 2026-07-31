#!/usr/bin/env python3
"""Reproduce Two-Stage Unit Tying on a trained dense rank-2 DLGN.

This is a local adaptation of Lee et al. (2026) to the Fashion-MNIST dense
architecture. It implements their gate-probability Gauss--Newton screening,
T+k overshoot, and finite-difference Binary Split refinement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import resource
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)
from torchlogix.layers import GroupSum, LogicDense


CONSTANT_FALSE_ID = 0
CONSTANT_TRUE_ID = 15


def logic_layers(model: torch.nn.Module) -> list[LogicDense]:
    layers = [module for module in model.children() if isinstance(module, LogicDense)]
    if not layers:
        raise ValueError("model has no dense logic layers")
    return layers


def group_sum(model: torch.nn.Module) -> GroupSum:
    reductions = [module for module in model.children() if isinstance(module, GroupSum)]
    if len(reductions) != 1:
        raise ValueError(f"expected exactly one GroupSum, found {len(reductions)}")
    return reductions[0]


def flatten_encoded(model: torch.nn.Module, encoded: torch.Tensor) -> torch.Tensor:
    flatteners = [
        module for module in model.children() if isinstance(module, torch.nn.Flatten)
    ]
    if len(flatteners) != 1:
        raise ValueError(f"expected exactly one Flatten, found {len(flatteners)}")
    return flatteners[0](encoded)


def forward_encoded(model: torch.nn.Module, encoded: torch.Tensor) -> torch.Tensor:
    x = flatten_encoded(model, encoded)
    for layer in logic_layers(model):
        x = layer(x)
    return group_sum(model)(x)


def forward_prefix(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    x = flatten_encoded(model, encoded)
    for layer in logic_layers(model)[:target_layer]:
        x = layer(x)
    return x


def forward_suffix(
    model: torch.nn.Module,
    target_output: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    x = target_output
    for layer in logic_layers(model)[target_layer + 1 :]:
        x = layer(x)
    return group_sum(model)(x)


def configured_logits_from_prefix(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    target_layer: int,
    masks: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Evaluate G tie configurations in one suffix pass.

    ``masks`` and ``values`` have shape ``(G, out_dim)``. Samples are tiled in
    contiguous configuration blocks to mirror the paper's batch-tiling scheme.
    """
    layers = logic_layers(model)
    configurations = masks.shape[0]
    batch = prefix.shape[0]
    tiled_prefix = prefix.repeat((configurations, 1))
    target = layers[target_layer](tiled_prefix).reshape(
        configurations,
        batch,
        layers[target_layer].out_dim,
    )
    target = torch.where(
        masks[:, None, :],
        values[:, None, :].to(dtype=target.dtype),
        target,
    )
    x = target.reshape(configurations * batch, -1)
    for layer in layers[target_layer + 1 :]:
        x = layer(x)
    logits = group_sum(model)(x)
    return logits.reshape(configurations, batch, -1)


def gauss_newton_constant_scores(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    """Return paper Eq. (2)/(3) scores with shape ``(out_dim, 2)``.

    For a rank-2 raw gate, the layer output is linear in its 16-way gate
    distribution. The directional derivative toward a constant is therefore
    ``d z / d y_u * (c - y_u)``, exactly the Jacobian-vector product in Eq. (3)
    without materializing the probability Jacobian.
    """
    model.train(True)
    with torch.no_grad():
        prefix = forward_prefix(model, encoded, target_layer)
        target = logic_layers(model)[target_layer](prefix)
    target = target.detach().requires_grad_(True)
    logits = forward_suffix(model, target, target_layer)
    score_zero = torch.zeros(target.shape[-1], device=target.device)
    score_one = torch.zeros_like(score_zero)
    for class_index in range(logits.shape[-1]):
        gradient = torch.autograd.grad(
            logits[:, class_index].sum(),
            target,
            retain_graph=class_index + 1 < logits.shape[-1],
        )[0]
        score_zero += (gradient * (-target)).square().sum(dim=0)
        score_one += (gradient * (1.0 - target)).square().sum(dim=0)
    return 0.5 * torch.stack((score_zero, score_one), dim=1) / target.shape[0]


def apply_permanent_ties(
    layer: LogicDense,
    units: list[int] | np.ndarray,
    directions: list[int] | np.ndarray,
    tie_logit: float,
) -> None:
    units_tensor = torch.as_tensor(units, dtype=torch.long, device=layer.weight.device)
    direction_tensor = torch.as_tensor(
        directions,
        dtype=torch.long,
        device=layer.weight.device,
    )
    if units_tensor.numel() != direction_tensor.numel():
        raise ValueError("units and directions must have equal length")
    if units_tensor.numel() == 0:
        return
    if not bool(torch.all((direction_tensor == 0) | (direction_tensor == 1))):
        raise ValueError("tie directions must be zero or one")
    with torch.no_grad():
        layer.weight[units_tensor] = -float(tie_logit)
        constant_ids = torch.where(
            direction_tensor == 0,
            torch.full_like(direction_tensor, CONSTANT_FALSE_ID),
            torch.full_like(direction_tensor, CONSTANT_TRUE_ID),
        )
        layer.weight[units_tensor, constant_ids] = float(tie_logit)


def binary_split_identify(
    candidates: list[int],
    evaluate_pair: Callable[[list[int], list[int]], tuple[float, float]],
) -> tuple[int, list[dict]]:
    """Algorithm 1 from Lee et al. (2026)."""
    if not candidates:
        raise ValueError("binary split requires a nonempty candidate set")
    full_set = list(candidates)
    current = list(candidates)
    path = []
    while len(current) > 1:
        midpoint = (len(current) + 1) // 2
        first = current[:midpoint]
        second = current[midpoint:]
        second_set = set(second)
        first_set = set(first)
        tie_first = [unit for unit in full_set if unit not in second_set]
        tie_second = [unit for unit in full_set if unit not in first_set]
        error_first, error_second = evaluate_pair(tie_first, tie_second)
        choose_first = error_first > error_second
        path.append(
            {
                "current_size": len(current),
                "first_size": len(first),
                "second_size": len(second),
                "error_first": float(error_first),
                "error_second": float(error_second),
                "chosen_half": "first" if choose_first else "second",
            }
        )
        current = first if choose_first else second
    return current[0], path


def refine_overshoot(
    candidates: list[int],
    target_size: int,
    evaluate_pair: Callable[[list[int], list[int]], tuple[float, float]],
) -> tuple[list[int], list[dict]]:
    """Repeatedly apply Algorithm 1 until exactly ``target_size`` remain."""
    if target_size < 0 or target_size > len(candidates):
        raise ValueError("target size must lie within the candidate set")
    selected = list(candidates)
    removals = []
    while len(selected) > target_size:
        harmful, path = binary_split_identify(selected, evaluate_pair)
        selected.remove(harmful)
        removals.append(
            {
                "removal_index": len(removals),
                "removed_unit": int(harmful),
                "set_size_before": len(selected) + 1,
                "binary_split_path": path,
            }
        )
    return selected, removals


def metric_record(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
) -> dict:
    scores = scores.cpu()
    labels = labels.cpu().to(torch.long)
    predictions = scores.argmax(dim=-1)
    classes = scores.shape[-1]
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    for true_class, predicted_class in zip(labels.tolist(), predictions.tolist()):
        confusion[true_class, predicted_class] += 1
    per_class_accuracy = []
    f1_values = []
    for class_index in range(classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        class_total = int(confusion[class_index, :].sum())
        per_class_accuracy.append(
            0.0 if class_total == 0 else true_positive / class_total
        )
    return {
        "examples": len(labels),
        "accuracy": float((predictions == labels).float().mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_accuracy": per_class_accuracy,
        "decision_flip_count": int(
            (predictions != reference_predictions.cpu()).sum()
        ),
        "decision_flip_rate": float(
            (predictions != reference_predictions.cpu()).float().mean()
        ),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_encoded(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size].to(device)
            outputs.append(forward_encoded(model, batch).cpu())
    return torch.cat(outputs)


def selection_indices(
    population: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0 or count > population:
        raise ValueError("sample count must be positive and no larger than population")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(population, size=count, replace=False))


def json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help="Optional subset of configured tied ratios",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    run_dir = cli.run_dir.resolve()
    method_config = json.loads(cli.config.read_text())
    ratios = cli.ratios if cli.ratios is not None else method_config["ratios"]
    configured_ratios = set(float(value) for value in method_config["ratios"])
    if any(float(ratio) not in configured_ratios for ratio in ratios):
        raise ValueError("--ratios must be a subset of the frozen configuration")

    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    checkpoint_path = run_dir / method_config["source_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    original_state = checkpoint["model_state_dict"]
    thresholds = original_state["0.thresholds"]

    cpu_model = get_model(thresholds, args)
    cpu_model.load_state_dict(original_state, strict=True)
    cpu_model.eval()
    _, validation_loader, calibration_loader, _ = load_dataset(
        args,
        include_calibration=True,
    )
    calibration_images, calibration_labels = take_examples(
        calibration_loader,
        len(calibration_loader.dataset),
    )
    validation_images, validation_labels = take_examples(
        validation_loader,
        len(validation_loader.dataset),
    )
    with torch.no_grad():
        calibration_encoded = cpu_model[0](calibration_images).bool().cpu()
        validation_encoded = cpu_model[0](validation_images).bool().cpu()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Two-Stage Unit Tying pilot requires CUDA")
    base_model = get_model(thresholds, args)
    base_model.load_state_dict(original_state, strict=True)
    base_model.to(device)
    evaluation_batch_size = int(method_config["evaluation_batch_size"])
    baseline_calibration_scores = evaluate_encoded(
        base_model,
        calibration_encoded,
        evaluation_batch_size,
        device,
    )
    baseline_validation_scores = evaluate_encoded(
        base_model,
        validation_encoded,
        evaluation_batch_size,
        device,
    )
    baseline_calibration_predictions = baseline_calibration_scores.argmax(-1)
    baseline_validation_predictions = baseline_validation_scores.argmax(-1)
    del base_model

    stage_a_indices = selection_indices(
        len(calibration_encoded),
        int(method_config["stage_a_samples"]),
        int(method_config["calibration_seed"]),
    )
    stage_b_indices = selection_indices(
        len(calibration_encoded),
        int(method_config["stage_b_samples"]),
        int(method_config["calibration_seed"]) + 1,
    )
    output_root = run_dir / method_config["output"]
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = copy.deepcopy(method_config)
    frozen_config["source_config"] = str(cli.config)
    frozen_config["source_config_sha256"] = sha256_file(cli.config)
    (output_root / "config.json").write_text(
        json.dumps(frozen_config, indent=2, sort_keys=True) + "\n"
    )
    sample_record = {
        "partition": "calibration",
        "partition_indices_sha256": calibration_loader.split_manifest[
            "partitions"
        ]["calibration"]["indices_sha256"],
        "stage_a_local_indices": stage_a_indices.tolist(),
        "stage_a_local_indices_sha256": tensor_sha256(
            torch.from_numpy(stage_a_indices)
        ),
        "stage_b_local_indices": stage_b_indices.tolist(),
        "stage_b_local_indices_sha256": tensor_sha256(
            torch.from_numpy(stage_b_indices)
        ),
        "stage_a_samples": len(stage_a_indices),
        "stage_b_samples": len(stage_b_indices),
        "validation_used_for_selection": False,
        "test_used": False,
    }
    (output_root / "sample_selection.json").write_text(
        json.dumps(sample_record, indent=2, sort_keys=True) + "\n"
    )

    aggregate_path = output_root / "aggregate.json"
    existing_aggregate = []
    if aggregate_path.exists():
        existing_aggregate = json.loads(aggregate_path.read_text())
    aggregate_by_ratio = {
        float(record["ratio"]): record for record in existing_aggregate
    }
    for ratio in ratios:
        ratio = float(ratio)
        ratio_name = f"ratio_{int(round(100 * ratio)):02d}"
        ratio_dir = output_root / ratio_name
        ratio_dir.mkdir(parents=True, exist_ok=True)
        model = get_model(thresholds, args)
        model.load_state_dict(original_state, strict=True)
        model.to(device)
        layers = logic_layers(model)
        eligible_layers = [int(value) for value in method_config["eligible_logic_layers"]]
        if any(index < 0 or index >= len(layers) for index in eligible_layers):
            raise ValueError("eligible layer index is outside the model")

        ratio_start = time.perf_counter()
        ties_by_layer = {}
        layer_records = []
        for layer_index in eligible_layers:
            layer = layers[layer_index]
            target = int(round(ratio * layer.out_dim))
            overshoot = min(
                layer.out_dim,
                target + int(method_config["overshoot"]),
            )
            stage_a_encoded = calibration_encoded[stage_a_indices].to(device)
            stage_a_start = time.perf_counter()
            scores = gauss_newton_constant_scores(
                model,
                stage_a_encoded,
                layer_index,
            ).detach().cpu().numpy()
            stage_a_seconds = time.perf_counter() - stage_a_start
            saliency = scores.min(axis=1)
            directions = scores.argmin(axis=1).astype(np.int64)
            ordering = np.lexsort((np.arange(layer.out_dim), saliency))
            candidates = ordering[:overshoot].astype(int).tolist()
            np.savez_compressed(
                ratio_dir / f"layer_{layer_index}_stage_a.npz",
                score_constant_0=scores[:, 0],
                score_constant_1=scores[:, 1],
                saliency=saliency,
                direction=directions,
                ordering=ordering,
                overshoot_candidates=np.asarray(candidates, dtype=np.int64),
            )

            model.eval()
            stage_b_encoded = calibration_encoded[stage_b_indices].to(device)
            teacher_scores = baseline_calibration_scores[stage_b_indices].to(device)
            with torch.no_grad():
                prefix = forward_prefix(model, stage_b_encoded, layer_index)
            assignment_tensor = torch.from_numpy(directions).to(device)
            evaluation_count = 0

            def evaluate_pair(
                first_units: list[int],
                second_units: list[int],
            ) -> tuple[float, float]:
                nonlocal evaluation_count
                masks = torch.zeros(
                    2,
                    layer.out_dim,
                    dtype=torch.bool,
                    device=device,
                )
                values = torch.zeros(
                    2,
                    layer.out_dim,
                    dtype=torch.float32,
                    device=device,
                )
                for config_index, units in enumerate((first_units, second_units)):
                    unit_tensor = torch.as_tensor(
                        units,
                        dtype=torch.long,
                        device=device,
                    )
                    masks[config_index, unit_tensor] = True
                    values[config_index, unit_tensor] = assignment_tensor[
                        unit_tensor
                    ].to(values.dtype)
                with torch.no_grad():
                    logits = configured_logits_from_prefix(
                        model,
                        prefix,
                        layer_index,
                        masks,
                        values,
                    )
                    distortion = 0.5 * (
                        logits - teacher_scores.unsqueeze(0)
                    ).square().sum(dim=-1).mean(dim=-1)
                evaluation_count += 2
                return float(distortion[0]), float(distortion[1])

            stage_b_start = time.perf_counter()
            selected, removals = refine_overshoot(
                candidates,
                target,
                evaluate_pair,
            )
            stage_b_seconds = time.perf_counter() - stage_b_start
            selected_directions = directions[selected].astype(int).tolist()
            apply_permanent_ties(
                layer,
                selected,
                selected_directions,
                float(method_config["tie_logit"]),
            )
            ties_by_layer[str(layer_index)] = [
                {"unit": int(unit), "constant": int(direction)}
                for unit, direction in zip(selected, selected_directions)
            ]
            refinement_path = ratio_dir / f"layer_{layer_index}_stage_b.json"
            refinement_payload = {
                "layer_index": layer_index,
                "target": target,
                "overshoot_size": overshoot,
                "removed": len(removals),
                "finite_difference_configurations": evaluation_count,
                "stage_a_seconds": stage_a_seconds,
                "stage_b_seconds": stage_b_seconds,
                "removals": removals,
                "selected_units": selected,
                "selected_directions": selected_directions,
            }
            refinement_path.write_text(
                json.dumps(refinement_payload, indent=2, sort_keys=True) + "\n"
            )
            current_calibration_scores = evaluate_encoded(
                model,
                calibration_encoded,
                evaluation_batch_size,
                device,
            )
            layer_records.append(
                {
                    key: value
                    for key, value in refinement_payload.items()
                    if key not in ("removals", "selected_units", "selected_directions")
                }
                | {
                    "calibration_after_layer": metric_record(
                        current_calibration_scores,
                        calibration_labels,
                        baseline_calibration_predictions,
                    )
                }
            )

        final_calibration_scores = evaluate_encoded(
            model,
            calibration_encoded,
            evaluation_batch_size,
            device,
        )
        final_validation_scores = evaluate_encoded(
            model,
            validation_encoded,
            evaluation_batch_size,
            device,
        )
        hard_ids_before = [
            original_state[f"{layer_index + 2}.weight"].argmax(dim=1)
            for layer_index in eligible_layers
        ]
        originally_constant = 0
        total_tied = 0
        for ids, layer_index in zip(hard_ids_before, eligible_layers):
            selected_units = [
                record["unit"] for record in ties_by_layer[str(layer_index)]
            ]
            selected_ids = ids[selected_units]
            originally_constant += int(
                ((selected_ids == CONSTANT_FALSE_ID) | (selected_ids == CONSTANT_TRUE_ID)).sum()
            )
            total_tied += len(selected_units)

        tied_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if "_export_lut_ids" not in key
        }
        tied_model_path = ratio_dir / "tied_model.pt"
        torch.save(tied_state, tied_model_path)
        tied_checkpoint = {
            "format_version": 1,
            "model_state_dict": tied_state,
            "metadata": {
                "method": "two-stage-unit-tying",
                "source_checkpoint": str(checkpoint_path.relative_to(run_dir)),
                "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "tied_ratio": ratio,
                "eligible_logic_layers": eligible_layers,
                "total_tied_units": total_tied,
                "ties_sha256": json_sha256(ties_by_layer),
                "validation_used_for_selection": False,
                "test_used": False,
            },
        }
        tied_checkpoint_path = ratio_dir / "tied_checkpoint.pt"
        torch.save(tied_checkpoint, tied_checkpoint_path)
        ties_path = ratio_dir / "ties.json"
        ties_path.write_text(json.dumps(ties_by_layer, indent=2, sort_keys=True) + "\n")
        summary = {
            "format_version": 1,
            "status": "completed",
            "development_run": True,
            "method": "two-stage-unit-tying",
            "adaptation": (
                "Fashion-MNIST dense-model adaptation of the paper's main "
                "accuracy protocol; first and final logic layers excluded"
            ),
            "ratio": ratio,
            "architecture": training_config["architecture"],
            "nominal_logic_units": sum(layer.out_dim for layer in layers),
            "eligible_logic_units": sum(layers[index].out_dim for index in eligible_layers),
            "total_tied_units": total_tied,
            "originally_constant_selected_units": originally_constant,
            "newly_constant_tied_units": total_tied - originally_constant,
            "layers": layer_records,
            "calibration": metric_record(
                final_calibration_scores,
                calibration_labels,
                baseline_calibration_predictions,
            ),
            "validation": metric_record(
                final_validation_scores,
                validation_labels,
                baseline_validation_predictions,
            ),
            "baseline_calibration_accuracy": float(
                (baseline_calibration_predictions == calibration_labels).float().mean()
            ),
            "baseline_validation_accuracy": float(
                (baseline_validation_predictions == validation_labels).float().mean()
            ),
            "wall_seconds": time.perf_counter() - ratio_start,
            "peak_process_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "checkpoint": tied_checkpoint_path.name,
            "checkpoint_sha256": sha256_file(tied_checkpoint_path),
            "ties": ties_path.name,
            "ties_sha256": sha256_file(ties_path),
            "data_policy": {
                "selection_partition": "calibration",
                "stage_a_samples": len(stage_a_indices),
                "stage_b_samples": len(stage_b_indices),
                "validation_used_only_for_final_evaluation": True,
                "test_used": False,
            },
            "software": {
                "source_revision": git_revision(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        summary_path = ratio_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        aggregate_by_ratio[ratio] = {
            "ratio": ratio,
            "directory": ratio_name,
            "total_tied_units": total_tied,
            "calibration_accuracy": summary["calibration"]["accuracy"],
            "validation_accuracy": summary["validation"]["accuracy"],
            "calibration_disagreement": summary["calibration"][
                "decision_flip_rate"
            ],
            "validation_disagreement": summary["validation"][
                "decision_flip_rate"
            ],
            "wall_seconds": summary["wall_seconds"],
        }
        print(
            json.dumps(aggregate_by_ratio[ratio], indent=2, sort_keys=True),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    aggregate = [
        aggregate_by_ratio[ratio] for ratio in sorted(aggregate_by_ratio)
    ]
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(output_root)
    print(
        json.dumps(
            {
                "status": "completed",
                "ratios": aggregate,
                "output": str(output_root),
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
