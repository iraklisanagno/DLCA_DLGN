"""Safe graph liveness and calibration-activity utilities for MarginSynth.

Topological liveness is independent of data and gate functions: an output is
live when a path of fixed connections reaches the final class sums.  It is
therefore safe to use as a hard optimization mask.  Algebraic liveness and
activity are descriptive only because changing a LUT can reactivate an input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


# Truth-table columns are ordered as AB = 00, 01, 10, 11.
LUT_TRUTH_TABLE = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=torch.bool,
)

# Whether the represented Boolean function depends on input A or input B.
LUT_INPUT_SUPPORT = torch.tensor(
    [
        [False, False],
        [True, True],
        [True, True],
        [True, False],
        [True, True],
        [False, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [False, True],
        [True, True],
        [True, False],
        [True, True],
        [True, True],
        [False, False],
    ],
    dtype=torch.bool,
)

CONSTANTS_AND_ROUTING_IDS = (0, 3, 5, 10, 12, 15)


def _connection_indices(layer) -> torch.Tensor:
    indices = getattr(getattr(layer, "connections", None), "indices", None)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise TypeError("liveness analysis requires fixed dense rank-2 connections")
    if indices.shape[0] != 2:
        raise ValueError("liveness analysis currently supports rank-2 LUTs only")
    return indices.detach().cpu().to(torch.long)


def topological_live_masks(layers: Sequence) -> dict[int, torch.Tensor]:
    """Return gates that can reach any final GroupSum input.

    Both inputs of each live gate are followed, regardless of its current LUT.
    Consequently these masks remain valid after arbitrary LUT replacements.
    """
    if not layers:
        raise ValueError("at least one logic layer is required")
    masks: dict[int, torch.Tensor] = {}
    current = torch.ones(int(layers[-1].out_dim), dtype=torch.bool)
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        if len(current) != int(layer.out_dim):
            raise ValueError("adjacent dense layer dimensions are inconsistent")
        masks[layer_index] = current.clone()
        if layer_index:
            predecessor_live = torch.zeros(int(layer.in_dim), dtype=torch.bool)
            selected = _connection_indices(layer)[:, current].reshape(-1)
            predecessor_live[selected] = True
            current = predecessor_live
    return masks


def algebraic_live_masks(
    layers: Sequence,
    lut_ids: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Return liveness under the checkpoint's current hard LUT functions.

    This is descriptive, not a safe rewrite mask: a changed constant/unary LUT
    can make a currently ignored predecessor relevant again.
    """
    if not layers:
        raise ValueError("at least one logic layer is required")
    masks: dict[int, torch.Tensor] = {}
    current = torch.ones(int(layers[-1].out_dim), dtype=torch.bool)
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        ids = lut_ids[layer_index].detach().cpu().to(torch.long)
        if len(ids) != int(layer.out_dim):
            raise ValueError("LUT IDs do not match layer width")
        masks[layer_index] = current.clone()
        if layer_index:
            predecessor_live = torch.zeros(int(layer.in_dim), dtype=torch.bool)
            indices = _connection_indices(layer)
            support = LUT_INPUT_SUPPORT[ids]
            for input_index in range(2):
                used = current & support[:, input_index]
                predecessor_live[indices[input_index, used]] = True
            current = predecessor_live
    return masks


def liveness_summary(
    layers: Sequence,
    lut_ids: Mapping[int, torch.Tensor],
    eligible_layers: Sequence[int],
) -> tuple[dict, dict[int, torch.Tensor]]:
    topological = topological_live_masks(layers)
    algebraic = algebraic_live_masks(layers, lut_ids)
    eligible = set(int(index) for index in eligible_layers)
    records = []
    for layer_index, layer in enumerate(layers):
        top_count = int(topological[layer_index].sum())
        algebraic_count = int(algebraic[layer_index].sum())
        total = int(layer.out_dim)
        records.append(
            {
                "layer": layer_index,
                "gates": total,
                "eligible": layer_index in eligible,
                "topologically_live": top_count,
                "topologically_dead": total - top_count,
                "algebraically_live": algebraic_count,
                "algebraically_dead": total - algebraic_count,
            }
        )
    eligible_total = sum(int(layers[index].out_dim) for index in eligible)
    eligible_top = sum(int(topological[index].sum()) for index in eligible)
    eligible_alg = sum(int(algebraic[index].sum()) for index in eligible)
    summary = {
        "definition": {
            "topological": "path through fixed connections to any class-sum input; safe after any LUT replacement",
            "algebraic": "path through inputs used by current hard LUTs; descriptive and not a safe replacement mask",
        },
        "layers": records,
        "eligible_gates": eligible_total,
        "eligible_topologically_live": eligible_top,
        "eligible_topologically_dead": eligible_total - eligible_top,
        "eligible_algebraically_live": eligible_alg,
        "eligible_algebraically_dead": eligible_total - eligible_alg,
    }
    return summary, topological


@torch.no_grad()
def collect_activity_risks(
    model,
    encoded: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
    sample_indices: torch.Tensor,
    eligible_layers: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Measure source-to-candidate output mismatch on optimization samples.

    Counts are accumulated on the GPU for global, per-class, and per-fold
    groups. Returned risk[layer][gate, LUT] is the worst mismatch rate across
    those groups. No repair, guard, validation, or test example is inspected.
    """
    from experiments.marginsynth.unit_tying import (
        flatten_encoded,
        logic_layers,
    )

    if batch_size <= 0:
        raise ValueError("activity batch size must be positive")
    layers = logic_layers(model)
    eligible = set(int(index) for index in eligible_layers)
    selected_labels = labels[sample_indices].to(torch.long)
    selected_folds = folds[sample_indices].to(torch.long)
    class_count = int(labels.max()) + 1
    fold_count = int(folds.max()) + 1
    counts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer_index in eligible:
        width = int(layers[layer_index].out_dim)
        counts[layer_index] = (
            torch.zeros(width * 4, device=device),
            torch.zeros(class_count * width * 4, device=device),
            torch.zeros(fold_count * width * 4, device=device),
        )

    model.eval()
    for start in range(0, len(sample_indices), batch_size):
        stop = min(len(sample_indices), start + batch_size)
        indices = sample_indices[start:stop]
        x = flatten_encoded(model, encoded[indices].to(device))
        batch_labels = selected_labels[start:stop].to(device)
        batch_folds = selected_folds[start:stop].to(device)
        for layer_index, layer in enumerate(layers):
            selected = layer.connections(x)
            if layer_index in eligible:
                pattern = (
                    selected[:, 0].to(torch.long) * 2
                    + selected[:, 1].to(torch.long)
                )
                width = int(layer.out_dim)
                offsets = torch.arange(width, device=device, dtype=torch.long) * 4
                flat = pattern + offsets[None, :]
                ones = torch.ones(flat.numel(), device=device)
                global_counts, class_counts, fold_counts = counts[layer_index]
                global_counts.scatter_add_(0, flat.reshape(-1), ones)
                class_flat = flat + batch_labels[:, None] * (width * 4)
                class_counts.scatter_add_(0, class_flat.reshape(-1), ones)
                fold_flat = flat + batch_folds[:, None] * (width * 4)
                fold_counts.scatter_add_(0, fold_flat.reshape(-1), ones)
            x = layer(x)

    truth = LUT_TRUTH_TABLE.to(device=device, dtype=torch.float32)
    risks: dict[int, torch.Tensor] = {}
    layer_summaries = []
    thresholds = (0.0, 0.01, 0.05, 0.10)
    for layer_index in sorted(eligible):
        layer = layers[layer_index]
        width = int(layer.out_dim)
        original = layer.weight.detach().argmax(1).to(device)
        source_truth = truth[original]
        mismatch = (source_truth[:, None, :] != truth[None, :, :]).float()
        global_counts, class_counts, fold_counts = counts[layer_index]
        global_counts = global_counts.view(width, 4)
        class_counts = class_counts.view(class_count, width, 4)
        fold_counts = fold_counts.view(fold_count, width, 4)
        global_rates = (
            mismatch * global_counts[:, None, :]
        ).sum(2) / global_counts.sum(1).clamp_min(1)[:, None]
        class_rates = (
            mismatch[None, :, :, :] * class_counts[:, :, None, :]
        ).sum(3) / class_counts.sum(2).clamp_min(1)[:, :, None]
        fold_rates = (
            mismatch[None, :, :, :] * fold_counts[:, :, None, :]
        ).sum(3) / fold_counts.sum(2).clamp_min(1)[:, :, None]
        robust = torch.maximum(
            global_rates,
            torch.maximum(class_rates.max(0).values, fold_rates.max(0).values),
        )
        risks[layer_index] = robust.cpu()

        source_one_rate = (
            source_truth * global_counts
        ).sum(1) / global_counts.sum(1).clamp_min(1)
        constant_error = torch.minimum(source_one_rate, 1.0 - source_one_rate)
        category_counts = {
            "constant": int(((original == 0) | (original == 15)).sum()),
            "routing_or_inversion": int(
                torch.isin(
                    original,
                    torch.tensor((3, 5, 10, 12), device=device),
                ).sum()
            ),
        }
        category_counts["binary"] = width - sum(category_counts.values())
        layer_summaries.append(
            {
                "layer": layer_index,
                "gates": width,
                "source_lut_categories": category_counts,
                "source_output_one_rate_mean": float(source_one_rate.mean()),
                "source_output_constant_error_mean": float(constant_error.mean()),
                "source_output_constant_error_threshold_counts": {
                    f"le_{value:.2f}": int((constant_error <= value).sum())
                    for value in thresholds
                },
                "best_constant_global_mismatch_mean": float(
                    global_rates[:, [0, 15]].min(1).values.mean()
                ),
                "best_constant_robust_mismatch_mean": float(
                    robust[:, [0, 15]].min(1).values.mean()
                ),
            }
        )
    return risks, {
        "definition": "candidate output disagreement with the source LUT on optimization samples",
        "policy": "robust risk is the maximum of global, per-class, and stratified-fold mismatch",
        "selection_partition": "calibration optimization subset only",
        "examples": int(len(sample_indices)),
        "classes": class_count,
        "folds": fold_count,
        "batch_size": int(batch_size),
        "layers": layer_summaries,
    }
