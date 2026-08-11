#!/usr/bin/env python3
"""GPU post-training, margin-constrained whole-circuit LUT resynthesis.

This is the primary MarginSynth research path.  It deliberately does not call
Unit Tying, use a Unit-Tying checkpoint, form a Gauss--Newton shortlist, or
impose a fixed gate-removal quota.  Connections and the teacher are frozen;
all eligible LUTs are optimized jointly over their allowed Boolean functions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

# Required by PyTorch for deterministic CUDA matrix products. This must be set
# before torch initializes cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.margin_aware_tying import (
    constraint_metrics,
    stratified_fold_ids,
    within_constraints,
)
from experiments.marginsynth.liveness_activity import (
    CONSTANTS_AND_ROUTING_IDS,
    collect_activity_risks,
    liveness_summary,
)
from experiments.marginsynth.unit_tying import (
    evaluate_encoded,
    forward_encoded,
    logic_layers,
    metric_record,
)
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)


# Cost of the 16 rank-2 functions in two-input AIG nodes. Inverters, wires,
# and constants are free in ABC's AND-node statistic; XOR/XNOR require three.
AIG_LUT_COSTS = (0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0)
GATE_COUNT_LUT_COSTS = (0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0)
# SkyWater 130 nm Cadence standard-cell areas (in square micrometres) reported
# by the Silicon-Aware Neural Networks paper for the 16 rank-2 functions.
SKY130_CELL_AREA_LUT_COSTS = (
    5.713, 9.522, 13.331, 7.618,
    13.331, 7.618, 15.235, 9.522,
    7.618, 15.235, 5.713, 13.331,
    5.713, 13.331, 7.618, 5.713,
)


def cost_vector(kind: str, device: torch.device | str = "cpu") -> torch.Tensor:
    if kind == "aig":
        values = AIG_LUT_COSTS
    elif kind == "gate-count":
        values = GATE_COUNT_LUT_COSTS
    elif kind == "sky130-cell-area":
        values = SKY130_CELL_AREA_LUT_COSTS
    else:
        raise ValueError(
            "cost kind must be 'aig', 'gate-count', or 'sky130-cell-area'"
        )
    return torch.tensor(values, dtype=torch.float32, device=device)


def allowed_lut_mask(original_ids: torch.Tensor, action_space: str) -> torch.Tensor:
    """Return a per-unit mask while always retaining the original LUT."""
    if action_space == "all":
        return torch.ones((len(original_ids), 16), dtype=torch.bool, device=original_ids.device)
    if action_space not in {"constants", "constants-routing"}:
        raise ValueError(
            "action_space must be 'all', 'constants', or 'constants-routing'"
        )
    mask = torch.zeros((len(original_ids), 16), dtype=torch.bool, device=original_ids.device)
    allowed = (
        (0, 15) if action_space == "constants" else CONSTANTS_AND_ROUTING_IDS
    )
    mask[:, list(allowed)] = True
    mask.scatter_(1, original_ids[:, None], True)
    return mask


def initialize_resynthesis_logits(
    layers,
    eligible_layers: list[int],
    original_ids: dict[int, torch.Tensor],
    action_space: str,
    initial_gap: float,
    forbidden_logit: float,
) -> dict[int, torch.Tensor]:
    masks = {}
    for layer_index in eligible_layers:
        layer = layers[layer_index]
        ids = original_ids[layer_index].to(layer.weight.device)
        mask = allowed_lut_mask(ids, action_space)
        with torch.no_grad():
            layer.weight.fill_(0.0)
            layer.weight.scatter_(1, ids[:, None], float(initial_gap))
            layer.weight.masked_fill_(~mask, float(forbidden_logit))
        masks[layer_index] = mask
    return masks


def expected_hardware_cost(
    layers,
    eligible_layers: list[int],
    costs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = []
    for layer_index in eligible_layers:
        probabilities = torch.softmax(layers[layer_index].weight / temperature, dim=1)
        values.append(probabilities @ costs)
    return torch.cat(values).mean()


def decision_margin_losses(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    retention: float,
    reserve: float,
    target_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample boundary loss and teacher winner/runner indices."""
    winners = teacher_scores.argmax(dim=1)
    excluded = teacher_scores.clone()
    excluded.scatter_(1, winners[:, None], -torch.inf)
    runners = excluded.argmax(dim=1)
    teacher_margin = (
        teacher_scores.gather(1, winners[:, None])
        - teacher_scores.gather(1, runners[:, None])
    ).squeeze(1)
    student_margin = (
        student_scores.gather(1, winners[:, None])
        - student_scores.gather(1, runners[:, None])
    ).squeeze(1)
    targets = torch.clamp(teacher_margin * retention, max=target_cap)
    targets = torch.maximum(targets, torch.full_like(targets, reserve))
    return F.relu(targets - student_margin), winners, runners


def robust_group_loss(
    losses: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_means = [losses[labels == value].mean() for value in labels.unique()]
    fold_means = [losses[folds == value].mean() for value in folds.unique()]
    return torch.stack(class_means).max(), torch.stack(fold_means).max()


def stratified_optimization_repair_split(
    labels: torch.Tensor,
    optimization_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split calibration data without letting optimization see repair cases."""
    if not 0.0 < optimization_fraction < 1.0:
        raise ValueError("optimization_fraction must be strictly between zero and one")
    values = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    optimize, repair = [], []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        indices = indices[rng.permutation(len(indices))]
        cut = min(len(indices) - 1, max(1, int(round(len(indices) * optimization_fraction))))
        optimize.extend(indices[:cut].tolist())
        repair.extend(indices[cut:].tolist())
    return (
        torch.tensor(sorted(optimize), dtype=torch.long),
        torch.tensor(sorted(repair), dtype=torch.long),
    )


def stratified_optimization_repair_guard_split(
    labels: torch.Tensor,
    optimization_fraction: float,
    repair_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three disjoint class-stratified sets for repeat resynthesis.

    Optimization supplies gradients, repair chooses the hardened prefix, and
    guard is untouched until the selected prefix has been fixed.
    """
    if not 0.0 < optimization_fraction < 1.0:
        raise ValueError("optimization_fraction must be strictly between zero and one")
    if not 0.0 < repair_fraction < 1.0 - optimization_fraction:
        raise ValueError("repair_fraction must leave a nonempty guard fraction")
    values = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    optimize, repair, guard = [], [], []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        indices = indices[rng.permutation(len(indices))]
        first = min(
            len(indices) - 2,
            max(1, int(round(len(indices) * optimization_fraction))),
        )
        second = min(
            len(indices) - 1,
            max(first + 1, first + int(round(len(indices) * repair_fraction))),
        )
        optimize.extend(indices[:first].tolist())
        repair.extend(indices[first:second].tolist())
        guard.extend(indices[second:].tolist())
    return (
        torch.tensor(sorted(optimize), dtype=torch.long),
        torch.tensor(sorted(repair), dtype=torch.long),
        torch.tensor(sorted(guard), dtype=torch.long),
    )


def changed_lut_records(
    layers,
    eligible_layers: list[int],
    original_ids: dict[int, torch.Tensor],
    cost_kind: str,
    activity_risks: dict[int, torch.Tensor] | None = None,
    activity_ranking: str = "none",
) -> list[dict]:
    costs = cost_vector(cost_kind)
    records = []
    for layer_index in eligible_layers:
        weights = layers[layer_index].weight.detach().cpu()
        new_ids = weights.argmax(dim=1)
        old_ids = original_ids[layer_index].detach().cpu()
        changed = torch.nonzero(new_ids != old_ids, as_tuple=False).flatten()
        for unit in changed.tolist():
            old_id = int(old_ids[unit])
            new_id = int(new_ids[unit])
            confidence = float(weights[unit, new_id] - weights[unit, old_id])
            benefit = float(costs[old_id] - costs[new_id])
            records.append(
                {
                    "layer": int(layer_index),
                    "unit": int(unit),
                    "original_lut": old_id,
                    "new_lut": new_id,
                    "proxy_benefit": benefit,
                    "preference_margin": confidence,
                    "activity_risk": (
                        None
                        if activity_risks is None
                        else float(activity_risks[layer_index][unit, new_id])
                    ),
                }
            )
    # Prefixes keep the transformations most likely to save synthesis cost and
    # most strongly preferred by joint optimization. This order is replayable.
    if activity_ranking == "none":
        records.sort(
            key=lambda item: (
                -item["proxy_benefit"],
                -item["preference_margin"],
                item["layer"],
                item["unit"],
            )
        )
    elif activity_ranking == "class-fold":
        if activity_risks is None:
            raise ValueError("class-fold activity ranking requires activity risks")
        records.sort(
            key=lambda item: (
                -item["proxy_benefit"],
                item["activity_risk"],
                -item["preference_margin"],
                item["layer"],
                item["unit"],
            )
        )
    else:
        raise ValueError("activity_ranking must be 'none' or 'class-fold'")
    return records


def materialize_change_prefix(
    model,
    original_ids: dict[int, torch.Tensor],
    eligible_layers: list[int],
    changes: list[dict],
    count: int,
    hard_logit: float,
) -> None:
    """Set eligible layers to the exact original/selected hard LUT prefix."""
    layers = logic_layers(model)
    chosen = {
        (int(item["layer"]), int(item["unit"])): int(item["new_lut"])
        for item in changes[:count]
    }
    with torch.no_grad():
        for layer_index in eligible_layers:
            ids = original_ids[layer_index].to(layers[layer_index].weight.device).clone()
            for (record_layer, unit), lut_id in chosen.items():
                if record_layer == layer_index:
                    ids[unit] = lut_id
            layers[layer_index].weight.fill_(-float(hard_logit))
            layers[layer_index].weight.scatter_(1, ids[:, None], float(hard_logit))


def tool_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()[:4000]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("circuit distillation requires CUDA")
    device = torch.device("cuda")
    run_dir = cli.run_dir.resolve()
    config = json.loads(cli.config.read_text())
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    output_dir = run_dir / config["output"]
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = run_dir / config.get("source_checkpoint", "best_checkpoint.pt")
    teacher_checkpoint_path = run_dir / config.get(
        "teacher_checkpoint", config.get("source_checkpoint", "best_checkpoint.pt")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    teacher_checkpoint = torch.load(
        teacher_checkpoint_path, map_location="cpu", weights_only=True
    )
    source_state = checkpoint["model_state_dict"]
    teacher_state = teacher_checkpoint["model_state_dict"]
    thresholds = source_state["0.thresholds"]
    if not torch.equal(thresholds, teacher_state["0.thresholds"]):
        raise ValueError("source and teacher binarization thresholds differ")

    _, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    report_validation = bool(config.get("report_validation", True))
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    validation_images = validation_labels = None
    if report_validation:
        validation_images, validation_labels = take_examples(
            validation_loader, len(validation_loader.dataset)
        )
    encoder = get_model(thresholds, args)
    encoder.load_state_dict(source_state, strict=True)
    encoder.eval()
    with torch.no_grad():
        calibration_encoded = encoder[0](calibration_images).bool().cpu()
        validation_encoded = (
            encoder[0](validation_images).bool().cpu()
            if report_validation
            else None
        )
    del encoder, calibration_images, validation_images

    teacher = get_model(thresholds, args)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.to(device).eval()
    evaluation_batch_size = int(config["evaluation_batch_size"])
    teacher_calibration_scores = evaluate_encoded(
        teacher, calibration_encoded, evaluation_batch_size, device
    )
    teacher_validation_scores = (
        evaluate_encoded(teacher, validation_encoded, evaluation_batch_size, device)
        if report_validation
        else None
    )
    teacher_calibration_predictions = teacher_calibration_scores.argmax(1)
    teacher_validation_predictions = (
        teacher_validation_scores.argmax(1) if report_validation else None
    )
    baseline_calibration = metric_record(
        teacher_calibration_scores,
        calibration_labels,
        teacher_calibration_predictions,
    )
    baseline_validation = (
        metric_record(
            teacher_validation_scores,
            validation_labels,
            teacher_validation_predictions,
        )
        if report_validation
        else None
    )
    guard_fraction = float(config.get("guard_fraction", 0.0))
    partition_seed = int(config.get("partition_seed", seed))
    if guard_fraction > 0.0:
        optimization_fraction = float(config.get("optimization_fraction", 0.6))
        repair_fraction = float(config.get("repair_fraction", 0.2))
        if abs(optimization_fraction + repair_fraction + guard_fraction - 1.0) > 1e-12:
            raise ValueError("optimization, repair, and guard fractions must sum to one")
        optimization_indices, repair_indices, guard_indices = (
            stratified_optimization_repair_guard_split(
                calibration_labels,
                optimization_fraction,
                repair_fraction,
                partition_seed + 104729,
            )
        )
    else:
        optimization_indices, repair_indices = stratified_optimization_repair_split(
            calibration_labels,
            float(config.get("optimization_fraction", 0.75)),
            partition_seed + 104729,
        )
        guard_indices = torch.empty(0, dtype=torch.long)
    repair_labels = calibration_labels[repair_indices]
    repair_teacher_scores = teacher_calibration_scores[repair_indices]
    repair_teacher_predictions = teacher_calibration_predictions[repair_indices]
    baseline_repair = metric_record(
        repair_teacher_scores, repair_labels, repair_teacher_predictions
    )
    baseline_guard = None
    if len(guard_indices):
        baseline_guard = metric_record(
            teacher_calibration_scores[guard_indices],
            calibration_labels[guard_indices],
            teacher_calibration_predictions[guard_indices],
        )

    student = get_model(thresholds, args)
    student.load_state_dict(source_state, strict=True)
    student.to(device).eval()
    layers = logic_layers(student)
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    if not eligible_layers or any(index < 0 or index >= len(layers) for index in eligible_layers):
        raise ValueError("invalid eligible_logic_layers")
    original_ids = {
        index: layers[index].weight.detach().argmax(dim=1).cpu()
        for index in eligible_layers
    }
    all_source_ids = {
        index: layer.weight.detach().argmax(dim=1).cpu()
        for index, layer in enumerate(layers)
    }
    graph_liveness, topological_masks = liveness_summary(
        layers, all_source_ids, eligible_layers
    )
    liveness_mode = config.get("liveness_mask", "none")
    if liveness_mode not in {"none", "topological"}:
        raise ValueError("liveness_mask must be 'none' or 'topological'")
    masks = initialize_resynthesis_logits(
        layers,
        eligible_layers,
        original_ids,
        config["action_space"],
        float(config["initial_logit_gap"]),
        float(config.get("forbidden_logit", -1000.0)),
    )
    locked_source_changes = 0
    if bool(config.get("lock_source_changes", False)):
        lock_reference_path = run_dir / config.get(
            "lock_reference_checkpoint", config.get("source_checkpoint", "best_checkpoint.pt")
        )
        lock_reference_payload = torch.load(
            lock_reference_path, map_location="cpu", weights_only=True
        )
        teacher_id_model = get_model(thresholds, args)
        teacher_id_model.load_state_dict(teacher_state, strict=True)
        teacher_id_layers = logic_layers(teacher_id_model)
        lock_reference_model = get_model(thresholds, args)
        lock_reference_model.load_state_dict(
            lock_reference_payload["model_state_dict"], strict=True
        )
        lock_reference_layers = logic_layers(lock_reference_model)
        for layer_index in eligible_layers:
            source_layer_ids = original_ids[layer_index].to(device)
            teacher_layer_ids = teacher_id_layers[layer_index].weight.detach().argmax(1).to(device)
            reference_layer_ids = (
                lock_reference_layers[layer_index].weight.detach().argmax(1).to(device)
            )
            locked_rows = reference_layer_ids != teacher_layer_ids
            if bool((source_layer_ids[locked_rows] != reference_layer_ids[locked_rows]).any()):
                raise ValueError("source altered a locked first-pass LUT function")
            locked_source_changes += int(locked_rows.sum())
            masks[layer_index][locked_rows] = False
            masks[layer_index][locked_rows, source_layer_ids[locked_rows]] = True
        del teacher_id_model, lock_reference_model
    else:
        lock_reference_path = None
    if liveness_mode == "topological":
        for layer_index in eligible_layers:
            live_rows = topological_masks[layer_index].to(device)
            source_layer_ids = original_ids[layer_index].to(device)
            masks[layer_index][~live_rows] = False
            masks[layer_index][~live_rows, source_layer_ids[~live_rows]] = True
    optimizable_gate_count = sum(
        int((masks[index].sum(1) > 1).sum()) for index in eligible_layers
    )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for layer_index in eligible_layers:
        layer = layers[layer_index]
        layer.weight.requires_grad_(True)
        layer.train(True)
        layer.parametrization.forward_sampling = config.get("forward_sampling", "hard")
        parameters.append(layer.weight)
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))

    fold_ids_np = stratified_fold_ids(
        calibration_labels.numpy(), int(config["stability_folds"]), seed + 7919
    )
    fold_ids = torch.from_numpy(fold_ids_np)
    activity_ranking = config.get("activity_ranking", "none")
    activity_risks = None
    activity_analysis = None
    if activity_ranking == "class-fold":
        activity_started = time.perf_counter()
        activity_risks, activity_analysis = collect_activity_risks(
            student,
            calibration_encoded,
            calibration_labels,
            fold_ids,
            optimization_indices,
            eligible_layers,
            int(config.get("activity_batch_size", min(evaluation_batch_size, 64))),
            device,
        )
        activity_analysis["elapsed_seconds"] = time.perf_counter() - activity_started
        (output_dir / "activity_analysis.json").write_text(
            json.dumps(activity_analysis, indent=2, sort_keys=True) + "\n"
        )
        # Activity collection deliberately evaluates the hard source model.
        # Restore the eligible layers to training mode for straight-through
        # resynthesis after the collector's model.eval() call.
        for layer_index in eligible_layers:
            layers[layer_index].train(True)
    elif activity_ranking != "none":
        raise ValueError("activity_ranking must be 'none' or 'class-fold'")
    (output_dir / "liveness_analysis.json").write_text(
        json.dumps(graph_liveness, indent=2, sort_keys=True) + "\n"
    )
    generator = torch.Generator().manual_seed(seed + 17)
    permutations = []
    trace = []
    steps = int(config["steps"])
    batch_size = int(config["batch_size"])
    cost_weights = config.get("loss_weights", {})
    cost_kind = config["cost_kind"]
    costs = cost_vector(cost_kind, device)
    train_started = time.perf_counter()
    epoch_order = optimization_indices[
        torch.randperm(len(optimization_indices), generator=generator)
    ]
    cursor = 0

    for step in range(1, steps + 1):
        if cursor + batch_size > len(epoch_order):
            permutations.append(tensor_sha256(epoch_order))
            epoch_order = optimization_indices[
                torch.randperm(len(optimization_indices), generator=generator)
            ]
            cursor = 0
        indices = epoch_order[cursor : cursor + batch_size]
        cursor += batch_size
        encoded = calibration_encoded[indices].to(device)
        labels = calibration_labels[indices].to(device)
        teacher_scores = teacher_calibration_scores[indices].to(device)
        batch_folds = fold_ids[indices].to(device)

        progress = (step - 1) / max(steps - 1, 1)
        temperature = float(config["temperature_start"]) * (
            float(config["temperature_end"]) / float(config["temperature_start"])
        ) ** progress
        for layer_index in eligible_layers:
            layers[layer_index].parametrization.update_temperature(temperature)
        student_scores = forward_encoded(student, encoded)
        if config["objective"] == "margin":
            sample_loss, _, _ = decision_margin_losses(
                student_scores,
                teacher_scores,
                float(config["margin_retention"]),
                float(config["margin_reserve"]),
                float(config["margin_target_cap"]),
            )
            decision_loss = sample_loss.mean()
            if bool(config.get("robust_groups", True)):
                class_loss, fold_loss = robust_group_loss(sample_loss, labels, batch_folds)
            else:
                class_loss = decision_loss.new_zeros(())
                fold_loss = decision_loss.new_zeros(())
            mse_loss = F.mse_loss(student_scores, teacher_scores)
        elif config["objective"] == "mse":
            mse_loss = F.mse_loss(student_scores, teacher_scores)
            decision_loss = mse_loss
            class_loss = mse_loss.new_zeros(())
            fold_loss = mse_loss.new_zeros(())
        elif config["objective"] == "cross-entropy":
            label_objective = F.cross_entropy(student_scores, labels)
            decision_loss = label_objective
            mse_loss = label_objective.new_zeros(())
            class_loss = label_objective.new_zeros(())
            fold_loss = label_objective.new_zeros(())
        else:
            raise ValueError("objective must be 'margin', 'mse', or 'cross-entropy'")
        label_loss = F.cross_entropy(student_scores, labels)
        hardware_cost = expected_hardware_cost(
            layers, eligible_layers, costs, max(temperature, 1e-4)
        )
        ramp = min(1.0, step / max(1, int(config.get("cost_warmup_steps", 1))))
        total_loss = (
            float(cost_weights.get("decision", 1.0)) * decision_loss
            + float(cost_weights.get("class_worst", 0.0)) * class_loss
            + float(cost_weights.get("fold_worst", 0.0)) * fold_loss
            + float(cost_weights.get("labels", 0.0)) * label_loss
            + ramp * float(cost_weights.get("hardware", 0.0)) * hardware_cost
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for layer_index, parameter in zip(eligible_layers, parameters):
            parameter.grad.masked_fill_(~masks[layer_index], 0.0)
        torch.nn.utils.clip_grad_norm_(parameters, float(config.get("gradient_clip", 5.0)))
        optimizer.step()
        with torch.no_grad():
            for layer_index in eligible_layers:
                layers[layer_index].weight.clamp_(-20.0, 20.0)
                layers[layer_index].weight.masked_fill_(
                    ~masks[layer_index], float(config.get("forbidden_logit", -1000.0))
                )

        if step == 1 or step % int(config["log_every"]) == 0 or step == steps:
            changed = sum(
                int((layers[index].weight.argmax(1).cpu() != original_ids[index]).sum())
                for index in eligible_layers
            )
            record = {
                "step": step,
                "temperature": temperature,
                "total_loss": float(total_loss.detach()),
                "decision_loss": float(decision_loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "class_worst_loss": float(class_loss.detach()),
                "fold_worst_loss": float(fold_loss.detach()),
                "label_loss": float(label_loss.detach()),
                "expected_hardware_cost": float(hardware_cost.detach()),
                "hard_changed_luts": changed,
                "elapsed_seconds": time.perf_counter() - train_started,
                "gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    permutations.append(tensor_sha256(epoch_order))

    soft_state = {
        key: value.detach().cpu()
        for key, value in student.state_dict().items()
        if "_export_lut_ids" not in key
    }
    soft_checkpoint = output_dir / "optimization_checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": soft_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "step": steps,
            "config_sha256": sha256_file(cli.config),
        },
        soft_checkpoint,
    )

    changes = changed_lut_records(
        layers,
        eligible_layers,
        original_ids,
        cost_kind,
        activity_risks=activity_risks,
        activity_ranking=activity_ranking,
    )
    (output_dir / "learned_changes.json").write_text(
        json.dumps(changes, indent=2, sort_keys=True) + "\n"
    )

    # Exact repair evaluates hardened models on all calibration examples. The
    # prefix starts with the strongest cost-reducing learned transformations.
    repair_log = []
    repair_model = get_model(thresholds, args)
    repair_model.load_state_dict(source_state, strict=True)
    repair_model.to(device).eval()

    def exact_prefix_metrics(count: int) -> tuple[dict, torch.Tensor]:
        materialize_change_prefix(
            repair_model,
            original_ids,
            eligible_layers,
            changes,
            count,
            float(config["hard_logit"]),
        )
        scores = evaluate_encoded(
            repair_model,
            calibration_encoded[repair_indices],
            evaluation_batch_size,
            device,
        )
        metrics = constraint_metrics(
            scores,
            repair_labels,
            repair_teacher_predictions,
            baseline_repair["accuracy"],
            baseline_repair["per_class_accuracy"],
        )
        metrics["within_budgets"] = within_constraints(
            metrics, config.get("selection_budgets", config["budgets"])
        )
        repair_log.append({"retained_changes": count, "metrics": metrics})
        return metrics, scores

    candidate_metrics, _ = exact_prefix_metrics(len(changes))
    retained = len(changes)
    if bool(config.get("repair", True)) and not candidate_metrics["within_budgets"]:
        zero_metrics, _ = exact_prefix_metrics(0)
        if zero_metrics["within_budgets"]:
            low, high = 0, len(changes)
            while low + 1 < high:
                middle = (low + high) // 2
                metrics, _ = exact_prefix_metrics(middle)
                if metrics["within_budgets"]:
                    low = middle
                else:
                    high = middle
            retained = low
            # Probe a deterministic window because feasibility need not be perfectly
            # monotone for interacting Boolean changes.
            for count in range(
                low + 1,
                min(len(changes), low + int(config.get("repair_scan", 32))) + 1,
            ):
                metrics, _ = exact_prefix_metrics(count)
                if metrics["within_budgets"]:
                    retained = count
        else:
            # A repeated pass may start outside the cumulative teacher budget.
            # Never silently treat its zero-change prefix as feasible.
            retained = min(
                ((0, zero_metrics), (len(changes), candidate_metrics)),
                key=lambda pair: (
                    pair[1]["accuracy_loss"],
                    pair[1]["maximum_per_class_accuracy_loss"],
                    pair[1]["decision_flip_rate"],
                    pair[1]["maximum_per_class_disagreement"],
                    -pair[0],
                ),
            )[0]
    final_repair_metrics, _ = exact_prefix_metrics(retained)
    materialize_change_prefix(
        repair_model,
        original_ids,
        eligible_layers,
        changes,
        retained,
        float(config["hard_logit"]),
    )
    final_calibration_scores = evaluate_encoded(
        repair_model, calibration_encoded, evaluation_batch_size, device
    )
    final_calibration_metrics = constraint_metrics(
        final_calibration_scores,
        calibration_labels,
        teacher_calibration_predictions,
        baseline_calibration["accuracy"],
        baseline_calibration["per_class_accuracy"],
    )
    final_calibration_metrics["within_budgets"] = within_constraints(
        final_calibration_metrics, config["budgets"]
    )
    final_guard_metrics = None
    if len(guard_indices):
        final_guard_scores = evaluate_encoded(
            repair_model,
            calibration_encoded[guard_indices],
            evaluation_batch_size,
            device,
        )
        final_guard_metrics = constraint_metrics(
            final_guard_scores,
            calibration_labels[guard_indices],
            teacher_calibration_predictions[guard_indices],
            baseline_guard["accuracy"],
            baseline_guard["per_class_accuracy"],
        )
        final_guard_metrics["within_budgets"] = within_constraints(
            final_guard_metrics, config["budgets"]
        )
    final_validation_metrics = None
    if report_validation:
        final_validation_scores = evaluate_encoded(
            repair_model, validation_encoded, evaluation_batch_size, device
        )
        final_validation_metrics = constraint_metrics(
            final_validation_scores,
            validation_labels,
            teacher_validation_predictions,
            baseline_validation["accuracy"],
            baseline_validation["per_class_accuracy"],
        )
        final_validation_metrics["within_budgets"] = within_constraints(
            final_validation_metrics, config["budgets"]
        )

    final_state = {
        key: value.detach().cpu()
        for key, value in repair_model.state_dict().items()
        if "_export_lut_ids" not in key
    }
    hard_checkpoint = output_dir / "distilled_checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": final_state,
            "metadata": {
                "method": "margin-constrained-whole-circuit-lut-resynthesis",
                "experiment_variant": config.get("method"),
                "step": int(checkpoint.get("metadata", {}).get("step", 0)),
                "optimization_steps": steps,
                "source_checkpoint": str(checkpoint_path.relative_to(run_dir)),
                "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "teacher_checkpoint": str(teacher_checkpoint_path.relative_to(run_dir)),
                "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint_path),
                "locked_source_changes": locked_source_changes,
                "lock_reference_checkpoint": (
                    None if lock_reference_path is None
                    else str(lock_reference_path.relative_to(run_dir))
                ),
                "learned_changes": len(changes),
                "retained_changes_after_exact_repair": retained,
                "unit_tying_warm_start": False,
                "gauss_newton_shortlist": False,
                "fixed_rewrite_quota": False,
                "validation_used_for_selection": False,
                "test_used": False,
            },
        },
        hard_checkpoint,
    )
    torch.save(final_state, output_dir / "distilled_model.pt")
    (output_dir / "repair_log.json").write_text(
        json.dumps(repair_log, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "optimization_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    (output_dir / "optimization_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )
    resolved_config = copy.deepcopy(config)
    resolved_config["source_config"] = str(cli.config.resolve())
    resolved_config["source_config_sha256"] = sha256_file(cli.config)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
    )
    samples = {
        "selection_partition": "calibration",
        "partition_seed": partition_seed,
        "partition_indices_sha256": calibration_loader.split_manifest["partitions"]["calibration"]["indices_sha256"],
        "partition_size": len(calibration_encoded),
        "optimization_size": len(optimization_indices),
        "repair_size": len(repair_indices),
        "guard_size": len(guard_indices),
        "optimization_indices_sha256": tensor_sha256(optimization_indices),
        "repair_indices_sha256": tensor_sha256(repair_indices),
        "guard_indices_sha256": tensor_sha256(guard_indices),
        "labels_sha256": tensor_sha256(calibration_labels),
        "stratified_fold_ids_sha256": tensor_sha256(fold_ids),
        "epoch_permutation_sha256": permutations,
        "validation_used_for_selection": False,
        "test_used": False,
    }
    (output_dir / "sample_selection.json").write_text(
        json.dumps(samples, indent=2, sort_keys=True) + "\n"
    )
    software = {
        "source_revision": git_revision(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(device),
        "yosys": tool_output(["yosys", "-V"]),
        "abc": tool_output(["berkeley-abc", "-h"]),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (output_dir / "software.json").write_text(
        json.dumps(software, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "format_version": 1,
        "status": "completed",
        "development_run": bool(config.get("development_run", True)),
        "method": "margin-constrained-whole-circuit-lut-resynthesis",
        "experiment_variant": config.get("method"),
        "architecture": training_config["architecture"],
        "dataset": training_config["dataset"],
        "nominal_logic_gates": sum(layer.out_dim for layer in layers),
        "eligible_logic_gates": sum(layers[index].out_dim for index in eligible_layers),
        "eligible_logic_layers": eligible_layers,
        "optimization_steps": steps,
        "action_space": config["action_space"],
        "objective": config["objective"],
        "cost_kind": cost_kind,
        "liveness_mask": liveness_mode,
        "activity_ranking": activity_ranking,
        "optimizable_logic_gates": optimizable_gate_count,
        "liveness": graph_liveness,
        "activity_analysis": activity_analysis,
        "learned_changes": len(changes),
        "retained_changes": retained,
        "locked_source_changes": locked_source_changes,
        "repair_applied": retained != len(changes),
        "repair_holdout_feasible": bool(final_repair_metrics["within_budgets"]),
        "guard_holdout_feasible": (
            None if final_guard_metrics is None
            else bool(final_guard_metrics["within_budgets"])
        ),
        "calibration_feasible": bool(final_calibration_metrics["within_budgets"]),
        "baseline_calibration": baseline_calibration,
        "baseline_repair_holdout": baseline_repair,
        "baseline_guard_holdout": baseline_guard,
        "baseline_validation": baseline_validation,
        "calibration": final_calibration_metrics,
        "repair_holdout": final_repair_metrics,
        "guard_holdout": final_guard_metrics,
        "validation": final_validation_metrics,
        "timing": {
            "optimization_seconds": time.perf_counter() - train_started,
            "total_seconds": time.perf_counter() - started,
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "data_policy": {
            "selection_partition": "calibration",
            "validation_loaded": report_validation,
            "validation_used_only_for_final_evaluation": report_validation,
            "test_used": False,
        },
        "independence_from_unit_tying": {
            "unit_tying_checkpoint_loaded": False,
            "unit_tying_warm_start": False,
            "gauss_newton_shortlist": False,
            "binary_split": False,
            "fixed_rewrite_quota": False,
        },
        "artifacts": {
            "optimization_checkpoint": soft_checkpoint.name,
            "optimization_checkpoint_sha256": sha256_file(soft_checkpoint),
            "distilled_checkpoint": hard_checkpoint.name,
            "distilled_checkpoint_sha256": sha256_file(hard_checkpoint),
            "learned_changes_sha256": sha256_file(output_dir / "learned_changes.json"),
            "repair_log_sha256": sha256_file(output_dir / "repair_log.json"),
            "liveness_analysis_sha256": sha256_file(
                output_dir / "liveness_analysis.json"
            ),
            "activity_analysis_sha256": (
                None
                if activity_analysis is None
                else sha256_file(output_dir / "activity_analysis.json")
            ),
        },
        "software": software,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
