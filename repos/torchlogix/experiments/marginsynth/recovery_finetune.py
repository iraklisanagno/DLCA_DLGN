#!/usr/bin/env python3
"""Short, locked-rewrite recovery for a hardened MarginSynth checkpoint.

The original trained model is a fixed teacher.  Rows changed by the input
checkpoint are inferred from hard LUT IDs and locked.  Only unchanged rows in
the requested layers may adapt, using straight-through hard forwards.  Model
selection uses a deterministic holdout from the original training partition;
calibration and validation are report-only and test is never loaded/evaluated.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.circuit_distillation import (
    cost_vector,
    decision_margin_losses,
    robust_group_loss,
    stratified_optimization_repair_split,
)
from experiments.marginsynth.margin_aware_tying import (
    constraint_metrics,
    stratified_fold_ids,
    within_constraints,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def tool_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()[:4000]


def state_without_export_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "_export_lut_ids" not in key
    }


def hard_ids_by_layer(model, eligible_layers: list[int]) -> dict[int, torch.Tensor]:
    layers = logic_layers(model)
    return {
        index: layers[index].weight.detach().argmax(1).cpu()
        for index in eligible_layers
    }


def initialize_recovery_logits(
    layers,
    eligible_layers: list[int],
    source_ids: dict[int, torch.Tensor],
    logit_gap: float,
) -> None:
    """Keep the exact source function while restoring useful softmax gradients."""
    with torch.no_grad():
        for index in eligible_layers:
            ids = source_ids[index].to(layers[index].weight.device)
            layers[index].weight.zero_()
            layers[index].weight.scatter_(1, ids[:, None], float(logit_gap))


def locked_row_masks(
    source_ids: dict[int, torch.Tensor],
    teacher_ids: dict[int, torch.Tensor],
    lock_source_changes: bool,
) -> dict[int, torch.Tensor]:
    if source_ids.keys() != teacher_ids.keys():
        raise ValueError("source and teacher layer IDs do not match")
    return {
        index: (source_ids[index] != teacher_ids[index])
        if lock_source_changes
        else torch.zeros_like(source_ids[index], dtype=torch.bool)
        for index in source_ids
    }


def straight_through_hardware_cost(
    layers,
    eligible_layers: list[int],
    costs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = []
    for index in eligible_layers:
        soft = torch.softmax(layers[index].weight / temperature, dim=1)
        hard = F.one_hot(soft.argmax(1), soft.shape[1]).to(soft.dtype)
        straight_through = (hard - soft).detach() + soft
        values.append(straight_through @ costs)
    return torch.cat(values).mean()


def hard_hardware_cost(
    layers,
    eligible_layers: list[int],
    costs: torch.Tensor,
) -> float:
    ids = torch.cat([layers[index].weight.detach().argmax(1) for index in eligible_layers])
    return float(costs[ids].mean().cpu())


def gate_distribution_stats(layers, eligible_layers: list[int]) -> dict[str, float]:
    probabilities = torch.cat(
        [torch.softmax(layers[index].weight.detach(), dim=1) for index in eligible_layers]
    )
    top2 = probabilities.topk(2, dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
    return {
        "mean_entropy": float(entropy.mean().cpu()),
        "mean_top1_probability": float(top2[:, 0].mean().cpu()),
        "mean_top1_top2_gap": float((top2[:, 0] - top2[:, 1]).mean().cpu()),
    }


def set_recovery_training_mode(model, eligible_layers: list[int], sampling: str) -> None:
    model.eval()
    for index in eligible_layers:
        layer = logic_layers(model)[index]
        layer.train(True)
        layer.parametrization.forward_sampling = sampling


def encode_partition(model, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model[0](images).bool().cpu()


def evaluate_metrics(
    model,
    encoded: torch.Tensor,
    labels: torch.Tensor,
    teacher_scores: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    teacher_predictions = teacher_scores.argmax(1)
    baseline = metric_record(teacher_scores, labels, teacher_predictions)
    scores = evaluate_encoded(model, encoded, batch_size, device)
    metrics = constraint_metrics(
        scores,
        labels,
        teacher_predictions,
        baseline["accuracy"],
        baseline["per_class_accuracy"],
    )
    return metrics, scores


def choose_snapshot(records: list[dict], budgets: dict, ceiling: float) -> dict:
    """Select without calibration/validation: first feasible, else best recovery."""
    feasible = [
        row
        for row in records
        if within_constraints(row["monitor"], budgets)
        and row["hard_hardware_cost"] <= ceiling + 1e-12
    ]
    if feasible:
        return min(feasible, key=lambda row: row["step"])
    return min(
        records,
        key=lambda row: (
            max(0.0, row["hard_hardware_cost"] - ceiling),
            -row["monitor"]["accuracy"],
            row["monitor"]["decision_flip_rate"],
            row["step"],
        ),
    )


def main() -> None:
    cli = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("recovery fine-tuning requires CUDA")
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
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir()

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    teacher_path = run_dir / config.get("teacher_checkpoint", "best_checkpoint.pt")
    source_path = run_dir / config["source_checkpoint"]
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
    teacher_state = teacher_payload["model_state_dict"]
    source_state = source_payload["model_state_dict"]
    thresholds = teacher_state["0.thresholds"]
    if not torch.equal(thresholds, source_state["0.thresholds"]):
        raise ValueError("source and teacher binarization thresholds differ")

    train_loader, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    # Deterministic reads of the exact original train partition; the shuffled
    # training loader is intentionally not consumed for selection bookkeeping.
    deterministic_train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=int(config.get("encoding_batch_size", 512)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    train_images, train_labels = take_examples(
        deterministic_train_loader, len(deterministic_train_loader.dataset)
    )
    report_calibration = bool(config.get("report_calibration", True))
    report_validation = bool(config.get("report_validation", True))
    calibration_images = calibration_labels = None
    validation_images = validation_labels = None
    if report_calibration:
        calibration_images, calibration_labels = take_examples(
            calibration_loader, len(calibration_loader.dataset)
        )
    if report_validation:
        validation_images, validation_labels = take_examples(
            validation_loader, len(validation_loader.dataset)
        )

    encoder = get_model(thresholds, args)
    encoder.load_state_dict(teacher_state, strict=True)
    train_encoded = encode_partition(encoder, train_images)
    calibration_encoded = (
        encode_partition(encoder, calibration_images)
        if report_calibration
        else None
    )
    validation_encoded = (
        encode_partition(encoder, validation_images)
        if report_validation
        else None
    )
    del encoder, train_images, calibration_images, validation_images

    evaluation_batch_size = int(config["evaluation_batch_size"])
    teacher = get_model(thresholds, args)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.to(device).eval()
    teacher_train_scores = evaluate_encoded(
        teacher, train_encoded, evaluation_batch_size, device
    )
    teacher_calibration_scores = (
        evaluate_encoded(teacher, calibration_encoded, evaluation_batch_size, device)
        if report_calibration
        else None
    )
    teacher_validation_scores = (
        evaluate_encoded(teacher, validation_encoded, evaluation_batch_size, device)
        if report_validation
        else None
    )
    del teacher

    recovery_indices, monitor_indices = stratified_optimization_repair_split(
        train_labels,
        float(config.get("recovery_fraction", 0.9)),
        seed + 104729,
    )
    fold_ids = torch.from_numpy(
        stratified_fold_ids(
            train_labels.numpy(), int(config.get("stability_folds", 4)), seed + 7919
        )
    )

    model = get_model(thresholds, args)
    model.load_state_dict(source_state, strict=True)
    model.to(device).eval()
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    layers = logic_layers(model)
    if not eligible_layers or any(index < 0 or index >= len(layers) for index in eligible_layers):
        raise ValueError("invalid eligible_logic_layers")
    source_ids = hard_ids_by_layer(model, eligible_layers)
    teacher_id_model = get_model(thresholds, args)
    teacher_id_model.load_state_dict(teacher_state, strict=True)
    teacher_ids = hard_ids_by_layer(teacher_id_model, eligible_layers)
    del teacher_id_model
    lock_masks = locked_row_masks(
        source_ids, teacher_ids, bool(config.get("lock_source_changes", True))
    )
    initialize_recovery_logits(
        layers, eligible_layers, source_ids, float(config.get("initial_logit_gap", 4.0))
    )
    locked_values = {
        index: layers[index].weight.detach()[lock_masks[index].to(device)].clone()
        for index in eligible_layers
    }

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for index in eligible_layers:
        layers[index].weight.requires_grad_(True)
        parameters.append(layers[index].weight)
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
    costs = cost_vector(config.get("cost_kind", "aig"), device)
    source_hardware_cost = hard_hardware_cost(layers, eligible_layers, costs)
    ceiling = float(
        config.get(
            "hardware_cost_ceiling",
            source_hardware_cost * (1.0 + float(config.get("hardware_ceiling_tolerance", 0.0))),
        )
    )

    snapshots = sorted(set(int(value) for value in config["snapshot_steps"]))
    steps = int(config["steps"])
    if not snapshots or snapshots[0] != 0 or snapshots[-1] > steps:
        raise ValueError("snapshot_steps must contain 0 and not exceed steps")
    trace = []
    snapshot_records = []
    train_started = time.perf_counter()

    def save_snapshot(step: int) -> None:
        monitor_metrics, hard_monitor_scores = evaluate_metrics(
            model,
            train_encoded[monitor_indices],
            train_labels[monitor_indices],
            teacher_train_scores[monitor_indices],
            evaluation_batch_size,
            device,
        )
        monitor_metrics["within_budgets"] = within_constraints(
            monitor_metrics, config["selection_budgets"]
        )
        # Quantify the training-to-inference gap on the same selection holdout.
        set_recovery_training_mode(model, eligible_layers, "soft")
        with torch.no_grad():
            relaxed_monitor_scores = forward_encoded(
                model, train_encoded[monitor_indices].to(device)
            ).cpu()
        relaxed_hard_disagreement = float(
            (
                relaxed_monitor_scores.argmax(1)
                != hard_monitor_scores.argmax(1)
            ).float().mean()
        )
        model.eval()
        hard_cost = hard_hardware_cost(layers, eligible_layers, costs)
        changed_from_source = sum(
            int((layers[index].weight.argmax(1).cpu() != source_ids[index]).sum())
            for index in eligible_layers
        )
        locked_violations = sum(
            int(
                (
                    layers[index].weight.argmax(1).cpu()[lock_masks[index]]
                    != source_ids[index][lock_masks[index]]
                ).sum()
            )
            for index in eligible_layers
        )
        if locked_violations:
            raise RuntimeError(f"{locked_violations} locked rows changed hard function")
        checkpoint_path = snapshot_dir / f"step_{step:06d}.pt"
        torch.save(
            {
                "format_version": 1,
                "model_state_dict": state_without_export_buffers(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "source_checkpoint_sha256": sha256_file(source_path),
                "teacher_checkpoint_sha256": sha256_file(teacher_path),
                "config_sha256": sha256_file(cli.config),
            },
            checkpoint_path,
        )
        record = {
            "step": step,
            "monitor": monitor_metrics,
            "hard_hardware_cost": hard_cost,
            "hardware_ceiling": ceiling,
            "within_hardware_ceiling": hard_cost <= ceiling + 1e-12,
            "changed_unlocked_rows_from_source": changed_from_source,
            "locked_row_violations": locked_violations,
            "relaxed_hard_monitor_disagreement": relaxed_hard_disagreement,
            "gate_distribution": gate_distribution_stats(layers, eligible_layers),
            "examples_processed": step * int(config["batch_size"]),
            "elapsed_seconds": time.perf_counter() - train_started,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        snapshot_records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    save_snapshot(0)
    generator = torch.Generator().manual_seed(seed + 17)
    epoch_order = recovery_indices[
        torch.randperm(len(recovery_indices), generator=generator)
    ]
    permutations = []
    cursor = 0
    batch_size = int(config["batch_size"])
    weights = config.get("loss_weights", {})
    for step in range(1, steps + 1):
        if cursor + batch_size > len(epoch_order):
            permutations.append(tensor_sha256(epoch_order))
            epoch_order = recovery_indices[
                torch.randperm(len(recovery_indices), generator=generator)
            ]
            cursor = 0
        indices = epoch_order[cursor : cursor + batch_size]
        cursor += batch_size
        encoded = train_encoded[indices].to(device)
        labels = train_labels[indices].to(device)
        teacher_scores = teacher_train_scores[indices].to(device)
        batch_folds = fold_ids[indices].to(device)
        progress = (step - 1) / max(1, steps - 1)
        temperature = float(config.get("temperature_start", 1.0)) * (
            float(config.get("temperature_end", 0.25))
            / float(config.get("temperature_start", 1.0))
        ) ** progress
        for index in eligible_layers:
            layers[index].parametrization.update_temperature(temperature)
        set_recovery_training_mode(
            model, eligible_layers, config.get("forward_sampling", "hard")
        )
        scores = forward_encoded(model, encoded)
        sample_loss, _, _ = decision_margin_losses(
            scores,
            teacher_scores,
            float(config.get("margin_retention", 0.5)),
            float(config.get("margin_reserve", 0.25)),
            float(config.get("margin_target_cap", 2.0)),
        )
        decision_loss = sample_loss.mean()
        class_loss, fold_loss = robust_group_loss(sample_loss, labels, batch_folds)
        label_loss = F.cross_entropy(scores, labels)
        hardware_cost = straight_through_hardware_cost(
            layers, eligible_layers, costs, max(temperature, 1e-4)
        )
        hardware_excess = F.relu(hardware_cost - ceiling)
        probabilities = torch.cat(
            [torch.softmax(layers[index].weight, dim=1) for index in eligible_layers]
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(1).mean()
        loss = (
            float(weights.get("labels", 1.0)) * label_loss
            + float(weights.get("decision", 1.0)) * decision_loss
            + float(weights.get("class_worst", 0.0)) * class_loss
            + float(weights.get("fold_worst", 0.0)) * fold_loss
            + float(weights.get("hardware_ceiling", 0.0)) * hardware_excess
            + float(weights.get("entropy", 0.0)) * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for index, parameter in zip(eligible_layers, parameters):
            parameter.grad[lock_masks[index].to(device)] = 0.0
        torch.nn.utils.clip_grad_norm_(parameters, float(config.get("gradient_clip", 5.0)))
        optimizer.step()
        with torch.no_grad():
            for index in eligible_layers:
                layers[index].weight.clamp_(-20.0, 20.0)
                mask = lock_masks[index].to(device)
                layers[index].weight[mask] = locked_values[index]
        if step == 1 or step % int(config.get("log_every", 100)) == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "label_loss": float(label_loss.detach()),
                "decision_loss": float(decision_loss.detach()),
                "class_worst_loss": float(class_loss.detach()),
                "fold_worst_loss": float(fold_loss.detach()),
                "straight_through_hardware_cost": float(hardware_cost.detach()),
                "hardware_excess": float(hardware_excess.detach()),
                "entropy": float(entropy.detach()),
                "temperature": temperature,
                "elapsed_seconds": time.perf_counter() - train_started,
            }
            trace.append(row)
        if step in snapshots:
            save_snapshot(step)
    permutations.append(tensor_sha256(epoch_order))

    selected = choose_snapshot(snapshot_records, config["selection_budgets"], ceiling)
    selected_payload = torch.load(
        output_dir / selected["checkpoint"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(selected_payload["model_state_dict"], strict=True)
    model.to(device).eval()
    calibration_metrics = validation_metrics = None
    if report_calibration:
        calibration_metrics, _ = evaluate_metrics(
            model,
            calibration_encoded,
            calibration_labels,
            teacher_calibration_scores,
            evaluation_batch_size,
            device,
        )
        calibration_metrics["within_budgets"] = within_constraints(
            calibration_metrics, config["report_budgets"]
        )
    if report_validation:
        validation_metrics, _ = evaluate_metrics(
            model,
            validation_encoded,
            validation_labels,
            teacher_validation_scores,
            evaluation_batch_size,
            device,
        )
        validation_metrics["within_budgets"] = within_constraints(
            validation_metrics, config["report_budgets"]
        )
    recovered_path = output_dir / "recovered_checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": selected_payload["model_state_dict"],
            "metadata": {
                "method": "locked-rewrite-short-recovery",
                "selected_step": selected["step"],
                "source_checkpoint": str(source_path.relative_to(run_dir)),
                "source_checkpoint_sha256": sha256_file(source_path),
                "teacher_checkpoint": str(teacher_path.relative_to(run_dir)),
                "teacher_checkpoint_sha256": sha256_file(teacher_path),
                "locked_rewrite_rows": int(sum(mask.sum() for mask in lock_masks.values())),
                "validation_used_for_selection": False,
                "calibration_used_for_selection": False,
                "test_used": False,
            },
        },
        recovered_path,
    )
    torch.save(selected_payload["model_state_dict"], output_dir / "recovered_model.pt")

    with (output_dir / "training_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    (output_dir / "training_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "snapshot_metrics.json").write_text(
        json.dumps(snapshot_records, indent=2, sort_keys=True) + "\n"
    )
    resolved_config = copy.deepcopy(config)
    resolved_config["source_config"] = str(cli.config.resolve())
    resolved_config["source_config_sha256"] = sha256_file(cli.config)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
    )
    sample_selection = {
        "optimization_partition": "original_train",
        "selection_partition": "held_out_original_train",
        "train_partition_indices_sha256": train_loader.split_manifest["partitions"]["train"]["indices_sha256"],
        "recovery_indices_sha256": tensor_sha256(recovery_indices),
        "monitor_indices_sha256": tensor_sha256(monitor_indices),
        "train_labels_sha256": tensor_sha256(train_labels),
        "fold_ids_sha256": tensor_sha256(fold_ids),
        "epoch_permutation_sha256": permutations,
        "recovery_examples": len(recovery_indices),
        "monitor_examples": len(monitor_indices),
        "calibration_loaded_for_reporting": report_calibration,
        "calibration_used_for_selection": False,
        "validation_loaded_for_reporting": report_validation,
        "validation_used_for_selection": False,
        "test_used": False,
    }
    (output_dir / "sample_selection.json").write_text(
        json.dumps(sample_selection, indent=2, sort_keys=True) + "\n"
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
        "development_run": True,
        "method": "locked-rewrite-short-recovery",
        "architecture": training_config["architecture"],
        "dataset": training_config["dataset"],
        "nominal_logic_gates": sum(layer.out_dim for layer in layers),
        "eligible_logic_gates": sum(layers[index].out_dim for index in eligible_layers),
        "eligible_logic_layers": eligible_layers,
        "source_changed_rows_locked": int(sum(mask.sum() for mask in lock_masks.values())),
        "source_hard_hardware_cost": source_hardware_cost,
        "hardware_cost_ceiling": ceiling,
        "configured_steps": steps,
        "selected_step": selected["step"],
        "selected_monitor": selected["monitor"],
        "selected_hard_hardware_cost": selected["hard_hardware_cost"],
        "calibration": calibration_metrics,
        "validation": validation_metrics,
        "timing": {
            "training_seconds": time.perf_counter() - train_started,
            "total_seconds": time.perf_counter() - started,
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "data_policy": {
            "gradient_partition": "original_train_recovery_subset",
            "selection_partition": "original_train_monitor_holdout",
            "calibration_report_only": report_calibration,
            "validation_report_only": report_validation,
            "calibration_loaded": report_calibration,
            "validation_loaded": report_validation,
            "test_used": False,
        },
        "artifacts": {
            "recovered_checkpoint": recovered_path.name,
            "recovered_checkpoint_sha256": sha256_file(recovered_path),
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
