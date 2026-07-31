#!/usr/bin/env python3
"""Margin-, class-, stability-, and synthesis-aware structured unit tying.

This method retains the scalable structured action from Two-Stage Unit Tying
(forcing complete DLGN units to constants), but selects those actions globally
using projected decision-margin risk and an operation-aware hardware-benefit
proxy.  A small number of exact GPU batch checks enforce calibration budgets.
Validation is used only for final reporting and test data is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import resource
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

from experiments.marginsynth.unit_tying import (
    CONSTANT_FALSE_ID,
    CONSTANT_TRUE_ID,
    apply_permanent_ties,
    evaluate_encoded,
    flatten_encoded,
    forward_prefix,
    forward_suffix,
    gauss_newton_constant_scores,
    group_sum,
    json_sha256,
    logic_layers,
    metric_record,
    selection_indices,
)
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)


def stratified_fold_ids(
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign deterministic, approximately class-balanced fold identifiers."""
    labels = np.asarray(labels, dtype=np.int64)
    if folds < 2 or folds > len(labels):
        raise ValueError("fold count must be between 2 and the sample count")
    rng = np.random.default_rng(seed)
    result = np.empty(len(labels), dtype=np.int64)
    for class_index in np.unique(labels):
        indices = np.flatnonzero(labels == class_index)
        indices = indices[rng.permutation(len(indices))]
        result[indices] = np.arange(len(indices), dtype=np.int64) % folds
    return result


def _hard_lut_ids(layer) -> np.ndarray:
    return layer.weight.detach().argmax(dim=1).cpu().numpy().astype(np.int64)


def _truth_table(lut_id: int) -> np.ndarray:
    return ((int(lut_id) >> np.arange(4, dtype=np.int64)) & 1).astype(np.int8)


def _cofactor_kind(lut_id: int, input_position: int, constant: int) -> str:
    """Classify the one-input cofactor of a rank-2 truth table."""
    table = _truth_table(lut_id).reshape(2, 2)
    cofactor = (
        table[constant, :]
        if input_position == 0
        else table[:, constant]
    )
    return "constant" if int(cofactor[0]) == int(cofactor[1]) else "unary"


def structural_tie_benefits(
    layers,
    layer_index: int,
    constant_bonus: float = 1.0,
    unary_bonus: float = 0.25,
) -> np.ndarray:
    """Estimate direction-aware simplification benefit for every unit.

    The score counts removal of the tied nonconstant unit itself and the
    cofactors induced in its immediate fan-out gates.  Constant cofactors have
    greater value than buffer/inverter cofactors.  This is deliberately a
    cheap structural proxy; final claims still use identical ABC synthesis.
    """
    layer = layers[layer_index]
    current_ids = _hard_lut_ids(layer)
    benefit = np.ones((layer.out_dim, 2), dtype=np.float64)
    already_constant = np.isin(
        current_ids, [CONSTANT_FALSE_ID, CONSTANT_TRUE_ID]
    )
    benefit[already_constant, :] = 0.0
    if layer_index + 1 >= len(layers):
        return benefit
    next_layer = layers[layer_index + 1]
    indices = next_layer.connections.indices.detach().cpu().numpy()
    next_ids = _hard_lut_ids(next_layer)
    # A unit may feed both inputs of one gate. Count each induced gate once and
    # retain the stronger cofactor consequence.
    consequences: dict[tuple[int, int, int], float] = {}
    for input_position in range(indices.shape[0]):
        for next_unit, source_unit in enumerate(indices[input_position]):
            source_unit = int(source_unit)
            for constant in (0, 1):
                kind = _cofactor_kind(
                    int(next_ids[next_unit]), input_position, constant
                )
                value = constant_bonus if kind == "constant" else unary_bonus
                key = (source_unit, constant, int(next_unit))
                consequences[key] = max(consequences.get(key, 0.0), value)
    for (source_unit, constant, _), value in consequences.items():
        benefit[source_unit, constant] += value
    return benefit


def normalized_rank(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Stable [0,1] ranks, with one denoting the preferred extreme."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("normalized_rank expects a vector")
    ordering = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[ordering] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks /= len(values) - 1
    else:
        ranks[:] = 0.0
    return ranks if higher_is_better else 1.0 - ranks


def projected_risk_statistics(
    projected_margins: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    margin_floor: float,
) -> dict[str, np.ndarray]:
    """Compute fold-stable and worst-class hinge risks for Ux2 candidates."""
    projected = np.asarray(projected_margins, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if projected.ndim != 3 or projected.shape[0] != len(labels):
        raise ValueError("projected margins must have shape (samples, units, 2)")
    hinge = np.maximum(float(margin_floor) - projected, 0.0)
    flip = projected <= 0.0
    fold_risks = np.stack(
        [hinge[fold_ids == fold].mean(axis=0) for fold in np.unique(fold_ids)]
    )
    class_risks = np.stack(
        [hinge[labels == label].mean(axis=0) for label in np.unique(labels)]
    )
    return {
        "mean_hinge": hinge.mean(axis=0),
        "maximum_fold_hinge": fold_risks.max(axis=0),
        "fold_standard_deviation": fold_risks.std(axis=0),
        "maximum_class_hinge": class_risks.max(axis=0),
        "projected_flip_rate": flip.mean(axis=0),
    }


def margin_risk_for_layer(
    model,
    encoded: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: np.ndarray,
    layer_index: int,
    margin_floor: float,
    unit_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Vectorize first-order decision-margin risk over all constant ties."""
    model.train(True)
    with torch.no_grad():
        prefix = forward_prefix(model, encoded, layer_index)
        target_value = logic_layers(model)[layer_index](prefix)
    target = target_value.detach().requires_grad_(True)
    logits = forward_suffix(model, target, layer_index)
    with torch.no_grad():
        winners = logits.argmax(dim=1)
        runner_logits = logits.clone()
        runner_logits.scatter_(1, winners[:, None], -torch.inf)
        runners = runner_logits.argmax(dim=1)
    margins = (
        logits.gather(1, winners[:, None])
        - logits.gather(1, runners[:, None])
    ).squeeze(1)
    gradient = torch.autograd.grad(margins.sum(), target)[0].detach()
    margins = margins.detach()
    labels_np = labels.detach().cpu().numpy()
    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, target.shape[1], unit_chunk_size):
        stop = min(target.shape[1], start + unit_chunk_size)
        value = target.detach()[:, start:stop]
        grad = gradient[:, start:stop]
        projected = torch.stack(
            (
                margins[:, None] + grad * (-value),
                margins[:, None] + grad * (1.0 - value),
            ),
            dim=-1,
        ).cpu().numpy()
        stats = projected_risk_statistics(
            projected,
            labels_np,
            fold_ids,
            margin_floor,
        )
        for name, array in stats.items():
            pieces.setdefault(name, []).append(array)
    model.eval()
    return {name: np.concatenate(chunks, axis=0) for name, chunks in pieces.items()}


def constraint_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    reference_accuracy: float,
    reference_per_class_accuracy: list[float],
) -> dict:
    record = metric_record(scores, labels, reference_predictions)
    predictions = scores.argmax(dim=1).cpu()
    labels = labels.cpu().to(torch.long)
    reference_predictions = reference_predictions.cpu()
    per_class_disagreement = []
    per_class_accuracy_loss = []
    for class_index, baseline_accuracy in enumerate(reference_per_class_accuracy):
        mask = labels == class_index
        per_class_disagreement.append(
            float((predictions[mask] != reference_predictions[mask]).float().mean())
            if bool(mask.any())
            else 0.0
        )
        per_class_accuracy_loss.append(
            float(baseline_accuracy - record["per_class_accuracy"][class_index])
        )
    record.update(
        {
            "accuracy_loss": float(reference_accuracy - record["accuracy"]),
            "per_class_accuracy_loss": per_class_accuracy_loss,
            "maximum_per_class_accuracy_loss": max(per_class_accuracy_loss),
            "per_class_disagreement": per_class_disagreement,
            "maximum_per_class_disagreement": max(per_class_disagreement),
        }
    )
    return record


def within_constraints(metrics: dict, budgets: dict) -> bool:
    return (
        metrics["accuracy_loss"] <= float(budgets["accuracy_loss"]) + 1e-12
        and metrics["decision_flip_rate"]
        <= float(budgets["disagreement"]) + 1e-12
        and metrics["maximum_per_class_accuracy_loss"]
        <= float(budgets["per_class_accuracy_loss"]) + 1e-12
        and metrics["maximum_per_class_disagreement"]
        <= float(budgets["per_class_disagreement"]) + 1e-12
    )


def compose_candidate_table(
    layer_index: int,
    gauss_scores: np.ndarray,
    margin_stats: dict[str, np.ndarray],
    benefits: np.ndarray,
    weights: dict,
) -> list[dict]:
    """Choose each unit's direction and create globally comparable records."""
    gauss_badness = np.stack(
        [normalized_rank(gauss_scores[:, c], True) for c in (0, 1)], axis=1
    )
    benefit_rank = np.stack(
        [normalized_rank(benefits[:, c], True) for c in (0, 1)], axis=1
    )
    benefit_multiplier = 0.5 + benefit_rank
    risk = (
        float(weights["mean_margin"]) * margin_stats["mean_hinge"]
        + float(weights["fold_worst"]) * margin_stats["maximum_fold_hinge"]
        + float(weights["fold_std"])
        * margin_stats["fold_standard_deviation"]
        + float(weights["class_worst"]) * margin_stats["maximum_class_hinge"]
        + float(weights["projected_flip"])
        * margin_stats["projected_flip_rate"]
        + float(weights["gauss_newton"]) * gauss_badness
    )
    # Benefits act multiplicatively so a risky tie cannot win merely through
    # high fan-out. Deterministic ranks avoid dependence on physical units.
    utility = benefit_multiplier / (float(weights["risk_epsilon"]) + risk)
    directions = utility.argmax(axis=1)
    records = []
    for unit, direction in enumerate(directions.tolist()):
        records.append(
            {
                "layer": int(layer_index),
                "unit": int(unit),
                "constant": int(direction),
                "utility": float(utility[unit, direction]),
                "risk": float(risk[unit, direction]),
                "structural_benefit": float(benefits[unit, direction]),
                "mean_margin_hinge": float(
                    margin_stats["mean_hinge"][unit, direction]
                ),
                "maximum_fold_hinge": float(
                    margin_stats["maximum_fold_hinge"][unit, direction]
                ),
                "fold_standard_deviation": float(
                    margin_stats["fold_standard_deviation"][unit, direction]
                ),
                "maximum_class_hinge": float(
                    margin_stats["maximum_class_hinge"][unit, direction]
                ),
                "projected_flip_rate": float(
                    margin_stats["projected_flip_rate"][unit, direction]
                ),
                "gauss_newton": float(gauss_scores[unit, direction]),
                "minimum_gauss_newton": float(gauss_scores[unit].min()),
            }
        )
    return records


def balanced_shortlist_order(
    records_by_layer: dict[int, list[dict]],
    targets_by_layer: dict[int, int],
    overshoot: int,
) -> tuple[list[dict], dict[int, int]]:
    """GN-screen, margin-rank, and interleave layer-balanced candidates."""
    ranked_by_layer = {}
    pool_sizes = {}
    for layer_index in sorted(records_by_layer):
        target = int(targets_by_layer[layer_index])
        records = sorted(
            records_by_layer[layer_index],
            key=lambda item: (
                item["minimum_gauss_newton"], item["unit"]
            ),
        )[: min(len(records_by_layer[layer_index]), target + int(overshoot))]
        records.sort(
            key=lambda item: (-item["utility"], item["unit"])
        )
        ranked_by_layer[layer_index] = records[:target]
        pool_sizes[layer_index] = len(records)
    ordering = []
    maximum = max((len(records) for records in ranked_by_layer.values()), default=0)
    for position in range(maximum):
        for layer_index in sorted(ranked_by_layer):
            if position < len(ranked_by_layer[layer_index]):
                ordering.append(ranked_by_layer[layer_index][position])
    return ordering, pool_sizes


def apply_candidate_prefix(model, candidates: list[dict], count: int, tie_logit: float):
    by_layer: dict[int, list[dict]] = {}
    for candidate in candidates[:count]:
        by_layer.setdefault(int(candidate["layer"]), []).append(candidate)
    layers = logic_layers(model)
    for layer_index, records in by_layer.items():
        apply_permanent_ties(
            layers[layer_index],
            [record["unit"] for record in records],
            [record["constant"] for record in records],
            tie_logit,
        )


def interleave_layers(candidates: list[dict]) -> list[dict]:
    by_layer: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_layer.setdefault(int(candidate["layer"]), []).append(candidate)
    result = []
    maximum = max((len(items) for items in by_layer.values()), default=0)
    for position in range(maximum):
        for layer_index in sorted(by_layer):
            if position < len(by_layer[layer_index]):
                result.append(by_layer[layer_index][position])
    return result


def quality_tuple(metrics: dict) -> tuple[float, float, float, float]:
    """Accuracy-first deterministic objective for exact swap refinement."""
    return (
        float(metrics["accuracy_loss"]),
        float(metrics["maximum_per_class_accuracy_loss"]),
        float(metrics["decision_flip_rate"]),
        float(metrics["maximum_per_class_disagreement"]),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main():
    cli = parse_args()
    start_time = time.perf_counter()
    run_dir = cli.run_dir.resolve()
    config = json.loads(cli.config.read_text())
    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    checkpoint_path = run_dir / config["source_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    original_state = checkpoint["model_state_dict"]
    thresholds = original_state["0.thresholds"]

    if not torch.cuda.is_available():
        raise RuntimeError("margin-aware structured tying requires CUDA")
    device = torch.device("cuda")
    cpu_model = get_model(thresholds, args)
    cpu_model.load_state_dict(original_state, strict=True)
    cpu_model.eval()
    _, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    validation_images, validation_labels = take_examples(
        validation_loader, len(validation_loader.dataset)
    )
    with torch.no_grad():
        calibration_encoded = cpu_model[0](calibration_images).bool().cpu()
        validation_encoded = cpu_model[0](validation_images).bool().cpu()

    model = get_model(thresholds, args)
    model.load_state_dict(original_state, strict=True)
    model.to(device)
    batch_size = int(config["evaluation_batch_size"])
    baseline_calibration_scores = evaluate_encoded(
        model, calibration_encoded, batch_size, device
    )
    baseline_validation_scores = evaluate_encoded(
        model, validation_encoded, batch_size, device
    )
    baseline_calibration_predictions = baseline_calibration_scores.argmax(1)
    baseline_validation_predictions = baseline_validation_scores.argmax(1)
    baseline_calibration_record = metric_record(
        baseline_calibration_scores,
        calibration_labels,
        baseline_calibration_predictions,
    )
    baseline_validation_record = metric_record(
        baseline_validation_scores,
        validation_labels,
        baseline_validation_predictions,
    )

    stage_a_indices = selection_indices(
        len(calibration_encoded), int(config["stage_a_samples"]), int(config["seed"])
    )
    margin_indices = selection_indices(
        len(calibration_encoded),
        int(config["margin_samples"]),
        int(config["seed"]) + 1,
    )
    margin_labels = calibration_labels[margin_indices]
    fold_ids = stratified_fold_ids(
        margin_labels.numpy(), int(config["stability_folds"]), int(config["seed"]) + 2
    )
    layers = logic_layers(model)
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    if any(index < 0 or index >= len(layers) for index in eligible_layers):
        raise ValueError("eligible layer index is outside the model")

    scoring_start = time.perf_counter()
    candidates_by_layer = {}
    layer_scoring = []
    for layer_index in eligible_layers:
        layer_start = time.perf_counter()
        stage_a_encoded = calibration_encoded[stage_a_indices].to(device)
        gauss_scores = gauss_newton_constant_scores(
            model, stage_a_encoded, layer_index
        ).detach().cpu().numpy()
        model.eval()
        margin_stats = margin_risk_for_layer(
            model,
            calibration_encoded[margin_indices].to(device),
            margin_labels.to(device),
            fold_ids,
            layer_index,
            float(config["margin_floor"]),
            int(config["unit_chunk_size"]),
        )
        benefits = structural_tie_benefits(
            layers,
            layer_index,
            float(config["constant_cofactor_bonus"]),
            float(config["unary_cofactor_bonus"]),
        )
        layer_candidates = compose_candidate_table(
            layer_index,
            gauss_scores,
            margin_stats,
            benefits,
            config["objective_weights"],
        )
        candidates_by_layer[layer_index] = layer_candidates
        layer_scoring.append(
            {
                "layer": layer_index,
                "units": len(layer_candidates),
                "seconds": time.perf_counter() - layer_start,
                "benefit_mean": float(benefits.mean()),
                "benefit_maximum": float(benefits.max()),
            }
        )
    targets_by_layer = {
        index: int(round(float(config["target_ratio"]) * layers[index].out_dim))
        for index in eligible_layers
    }
    if config.get("allocation", "per-layer-balanced") == "per-layer-balanced":
        candidates, pool_sizes = balanced_shortlist_order(
            candidates_by_layer,
            targets_by_layer,
            int(config["candidate_overshoot"]),
        )
    elif config["allocation"] == "global":
        candidates = [
            item for records in candidates_by_layer.values() for item in records
        ]
        candidates.sort(
            key=lambda item: (-item["utility"], item["layer"], item["unit"])
        )
        pool_sizes = {index: len(records) for index, records in candidates_by_layer.items()}
    else:
        raise ValueError("allocation must be per-layer-balanced or global")
    ranking_by_unit = {
        (int(item["layer"]), int(item["unit"])): item
        for records in candidates_by_layer.values()
        for item in records
    }
    warm_start_path = config.get("warm_start_ties")
    if warm_start_path:
        warm_payload = json.loads((run_dir / warm_start_path).read_text())
        warm_candidates = []
        for layer_text, records in warm_payload.items():
            layer_index = int(layer_text)
            if layer_index not in eligible_layers:
                continue
            for tie in records:
                ranked = copy.deepcopy(
                    ranking_by_unit[(layer_index, int(tie["unit"]))]
                )
                ranked["constant"] = int(tie["constant"])
                ranked["warm_start"] = True
                warm_candidates.append(ranked)
        if len(warm_candidates) != sum(targets_by_layer.values()):
            raise RuntimeError(
                "warm-start tie count does not match the configured target"
            )
        candidates = interleave_layers(warm_candidates)
    scoring_seconds = time.perf_counter() - scoring_start

    output_root = run_dir / config["output"]
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_config = copy.deepcopy(config)
    resolved_config["source_config"] = str(cli.config)
    resolved_config["source_config_sha256"] = sha256_file(cli.config)
    (output_root / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
    )
    sample_payload = {
        "partition": "calibration",
        "partition_indices_sha256": calibration_loader.split_manifest["partitions"][
            "calibration"
        ]["indices_sha256"],
        "stage_a_indices": stage_a_indices.tolist(),
        "stage_a_indices_sha256": tensor_sha256(torch.from_numpy(stage_a_indices)),
        "margin_indices": margin_indices.tolist(),
        "margin_indices_sha256": tensor_sha256(torch.from_numpy(margin_indices)),
        "fold_ids": fold_ids.tolist(),
        "fold_ids_sha256": tensor_sha256(torch.from_numpy(fold_ids)),
        "validation_used_for_selection": False,
        "test_used": False,
    }
    (output_root / "sample_selection.json").write_text(
        json.dumps(sample_payload, indent=2, sort_keys=True) + "\n"
    )
    candidate_path = output_root / "candidate_ranking.json"
    candidate_path.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")

    eligible_units = sum(layers[index].out_dim for index in eligible_layers)
    target = sum(targets_by_layer.values())
    rounds = int(config["selection_rounds"])
    selected_count = 0
    round_records = []
    selection_start = time.perf_counter()
    # Every trial starts from the same baseline and applies a deterministic
    # global prefix. This makes rollback exact and the final result replayable.
    for round_index in range(rounds):
        if selected_count >= target:
            break
        remaining_rounds = rounds - round_index
        desired = min(
            target,
            selected_count
            + int(math.ceil((target - selected_count) / remaining_rounds)),
        )
        lower = selected_count
        upper = desired
        trials = []
        while lower < upper:
            count = upper
            trial = get_model(thresholds, args)
            trial.load_state_dict(original_state, strict=True)
            trial.to(device)
            apply_candidate_prefix(
                trial, candidates, count, float(config["tie_logit"])
            )
            scores = evaluate_encoded(
                trial, calibration_encoded, batch_size, device
            )
            metrics = constraint_metrics(
                scores,
                calibration_labels,
                baseline_calibration_predictions,
                baseline_calibration_record["accuracy"],
                baseline_calibration_record["per_class_accuracy"],
            )
            feasible = within_constraints(metrics, config["budgets"])
            trials.append(
                {"selected_prefix": count, "feasible": feasible, "metrics": metrics}
            )
            del trial
            torch.cuda.empty_cache()
            if feasible:
                lower = count
                break
            upper = (lower + upper) // 2
        if lower == selected_count:
            round_records.append(
                {
                    "round": round_index,
                    "accepted_before": selected_count,
                    "accepted_after": selected_count,
                    "trials": trials,
                    "stopped": "no feasible nonempty prefix",
                }
            )
            break
        round_records.append(
            {
                "round": round_index,
                "accepted_before": selected_count,
                "accepted_after": lower,
                "trials": trials,
            }
        )
        selected_count = lower
    selection_seconds = time.perf_counter() - selection_start

    selected = candidates[:selected_count]
    swap_records = []
    if config.get("swap_refinement", {}).get("enabled") and selected_count == target:
        selected_keys = {
            (int(item["layer"]), int(item["unit"])) for item in selected
        }
        harmful = sorted(
            selected,
            key=lambda item: (
                -float(ranking_by_unit[(item["layer"], item["unit"])]["risk"]),
                item["layer"],
                item["unit"],
            ),
        )
        alternatives = []
        for layer_index in eligible_layers:
            pool = sorted(
                candidates_by_layer[layer_index],
                key=lambda item: (item["minimum_gauss_newton"], item["unit"]),
            )[: targets_by_layer[layer_index] + int(config["candidate_overshoot"])]
            alternatives.extend(
                item
                for item in pool
                if (item["layer"], item["unit"]) not in selected_keys
            )
        alternatives.sort(
            key=lambda item: (-item["utility"], item["layer"], item["unit"])
        )

        def exact_metrics(ties: list[dict]) -> dict:
            trial = get_model(thresholds, args)
            trial.load_state_dict(original_state, strict=True)
            trial.to(device)
            apply_candidate_prefix(trial, ties, len(ties), float(config["tie_logit"]))
            scores = evaluate_encoded(trial, calibration_encoded, batch_size, device)
            result = constraint_metrics(
                scores,
                calibration_labels,
                baseline_calibration_predictions,
                baseline_calibration_record["accuracy"],
                baseline_calibration_record["per_class_accuracy"],
            )
            del trial
            torch.cuda.empty_cache()
            return result

        best_metrics = exact_metrics(selected)
        best_quality = quality_tuple(best_metrics)
        base_selected = list(selected)
        for count in config["swap_refinement"]["counts"]:
            count = min(int(count), len(harmful), len(alternatives))
            removed = {(item["layer"], item["unit"]) for item in harmful[:count]}
            proposal = [
                item for item in base_selected
                if (item["layer"], item["unit"]) not in removed
            ] + alternatives[:count]
            metrics = exact_metrics(proposal)
            feasible = within_constraints(metrics, config["budgets"])
            improved = feasible and quality_tuple(metrics) < best_quality
            swap_records.append(
                {
                    "swaps": count,
                    "feasible": feasible,
                    "improved": improved,
                    "metrics": metrics,
                }
            )
            if improved:
                selected = proposal
                best_metrics = metrics
                best_quality = quality_tuple(metrics)

    final_model = get_model(thresholds, args)
    final_model.load_state_dict(original_state, strict=True)
    final_model.to(device)
    apply_candidate_prefix(final_model, selected, len(selected), float(config["tie_logit"]))
    final_calibration_scores = evaluate_encoded(
        final_model, calibration_encoded, batch_size, device
    )
    final_validation_scores = evaluate_encoded(
        final_model, validation_encoded, batch_size, device
    )
    calibration_record = constraint_metrics(
        final_calibration_scores,
        calibration_labels,
        baseline_calibration_predictions,
        baseline_calibration_record["accuracy"],
        baseline_calibration_record["per_class_accuracy"],
    )
    validation_record = constraint_metrics(
        final_validation_scores,
        validation_labels,
        baseline_validation_predictions,
        baseline_validation_record["accuracy"],
        baseline_validation_record["per_class_accuracy"],
    )
    ties_by_layer = {
        str(layer_index): [
            {"unit": item["unit"], "constant": item["constant"]}
            for item in selected
            if item["layer"] == layer_index
        ]
        for layer_index in eligible_layers
    }
    ties_path = output_root / "ties.json"
    ties_path.write_text(json.dumps(ties_by_layer, indent=2, sort_keys=True) + "\n")
    rounds_path = output_root / "selection_rounds.json"
    rounds_path.write_text(json.dumps(round_records, indent=2, sort_keys=True) + "\n")
    swaps_path = output_root / "swap_refinement.json"
    swaps_path.write_text(json.dumps(swap_records, indent=2, sort_keys=True) + "\n")
    final_state = {
        key: value.detach().cpu()
        for key, value in final_model.state_dict().items()
        if "_export_lut_ids" not in key
    }
    checkpoint_payload = {
        "format_version": 1,
        "model_state_dict": final_state,
        "metadata": {
            "method": "margin-aware-structured-unit-tying",
            "step": int(checkpoint.get("metadata", {}).get("step", 0)),
            "source_checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "target_ratio": float(config["target_ratio"]),
            "selected_ties": selected_count,
            "ties_sha256": json_sha256(ties_by_layer),
            "validation_used_for_selection": False,
            "test_used": False,
        },
    }
    checkpoint_output = output_root / "tied_checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint_output)
    torch.save(final_state, output_root / "tied_model.pt")
    summary = {
        "format_version": 1,
        "status": "completed",
        "development_run": True,
        "method": "margin-aware-structured-unit-tying",
        "architecture": training_config["architecture"],
        "eligible_logic_units": eligible_units,
        "target_ratio": float(config["target_ratio"]),
        "target_ties": target,
        "selected_ties": selected_count,
        "achieved_ratio": selected_count / eligible_units,
        "selected_by_layer": {
            layer: len(records) for layer, records in ties_by_layer.items()
        },
        "target_by_layer": {str(key): value for key, value in targets_by_layer.items()},
        "candidate_pool_by_layer": {str(key): value for key, value in pool_sizes.items()},
        "baseline_calibration": baseline_calibration_record,
        "baseline_validation": baseline_validation_record,
        "calibration": calibration_record,
        "validation": validation_record,
        "timing": {
            "scoring_seconds": scoring_seconds,
            "selection_seconds": selection_seconds,
            "total_seconds": time.perf_counter() - start_time,
            "layer_scoring": layer_scoring,
        },
        "candidate_evaluations": sum(len(record["trials"]) for record in round_records),
        "selection_rounds": len(round_records),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": {
            "checkpoint": checkpoint_output.name,
            "checkpoint_sha256": sha256_file(checkpoint_output),
            "ties": ties_path.name,
            "ties_sha256": sha256_file(ties_path),
            "candidate_ranking": candidate_path.name,
            "candidate_ranking_sha256": sha256_file(candidate_path),
            "selection_rounds": rounds_path.name,
            "selection_rounds_sha256": sha256_file(rounds_path),
            "swap_refinement": swaps_path.name,
            "swap_refinement_sha256": sha256_file(swaps_path),
        },
        "data_policy": {
            "selection_partition": "calibration",
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
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(output_root)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
