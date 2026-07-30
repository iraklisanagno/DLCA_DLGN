"""One-shot task-aware rewiring for fixed dense logic networks.

This module is deliberately separate from :mod:`torchlogix.topology`.  The
frozen CoverageDLGN V3 constructor remains an offline, data-free method.
Task-aware rewiring is an optional complementary training event: it measures
activation-gradient signatures on one ordinary training batch, performs
degree-preserving two-edge swaps, and then discards all calibration state.
Only the resulting fixed integer indices are deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch


@dataclass(frozen=True)
class TaskAwareRewireResult:
    """Result of one layer's fixed-cost task-aware refinement."""

    indices: np.ndarray
    changed_mask: np.ndarray
    accepted_swaps: int
    eligible_gates: int
    score_improvement: float
    construction_seconds: float


def _normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    normalized = np.divide(
        values,
        norms[:, None],
        out=np.zeros_like(values),
        where=norms[:, None] > 0,
    )
    return normalized, norms


def task_aware_degree_preserving_refine(
    base_indices: np.ndarray,
    input_signatures: np.ndarray,
    output_signatures: np.ndarray,
    *,
    topology_seed: int = 0,
    layer_index: int = 0,
    change_fraction: float = 0.125,
    candidate_pool_size: int = 8,
    diversity_weight: float = 0.25,
) -> TaskAwareRewireResult:
    """Improve task-signature affinity with degree-preserving edge swaps.

    ``input_signatures`` and ``output_signatures`` contain one column per
    class.  They are normally the class-conditional mean absolute
    activation-gradient products measured on a single training batch.
    Accepted swaps preserve every predecessor occurrence exactly, along with
    gate count, LUT rank, weights, and output-group membership.
    """
    started = time.perf_counter()
    indices = np.asarray(base_indices, dtype=np.int64)
    input_signatures = np.asarray(input_signatures, dtype=np.float64)
    output_signatures = np.asarray(output_signatures, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError("base_indices must have shape (2, out_dim)")
    if input_signatures.ndim != 2 or output_signatures.ndim != 2:
        raise ValueError("task signatures must be two-dimensional")
    if input_signatures.shape[1] != output_signatures.shape[1]:
        raise ValueError("input and output signatures need the same class axis")
    if output_signatures.shape[0] != indices.shape[1]:
        raise ValueError("one output signature is required per gate")
    if indices.size and (
        indices.min() < 0 or indices.max() >= input_signatures.shape[0]
    ):
        raise ValueError("base_indices contain an out-of-bounds predecessor")
    if not 0.0 <= change_fraction <= 1.0:
        raise ValueError("change_fraction must be in [0, 1]")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative")

    indices = np.array(indices, copy=True)
    input_unit, _ = _normalize_rows(input_signatures)
    output_unit, output_strength = _normalize_rows(output_signatures)
    out_dim = indices.shape[1]
    eligible = min(out_dim, int(round(change_fraction * out_dim)))
    eligible -= eligible % 2
    changed = np.zeros(out_dim, dtype=bool)
    if eligible == 0:
        return TaskAwareRewireResult(
            indices=indices,
            changed_mask=changed,
            accepted_swaps=0,
            eligible_gates=0,
            score_improvement=0.0,
            construction_seconds=time.perf_counter() - started,
        )

    # Focus the bounded intervention on gates with the weakest observed task
    # signal. A seeded tie breaker makes zero-utility ties reproducible.
    rng = np.random.default_rng(
        (
            int(topology_seed)
            + 0xD1B54A32 * (int(layer_index) + 1)
        )
        % (1 << 63)
    )
    tie_break = rng.random(out_dim)
    order = np.lexsort((tie_break, output_strength))
    available = order[:eligible].tolist()

    pair_counts: dict[tuple[int, int], int] = {}
    for left, right in indices.T:
        key = tuple(sorted((int(left), int(right))))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    def score(gate: int, left: int, right: int) -> float:
        target = output_unit[gate]
        left_signature = input_unit[left]
        right_signature = input_unit[right]
        affinity = float(left_signature @ target + right_signature @ target)
        diversity = 1.0 - float(left_signature @ right_signature)
        return affinity + diversity_weight * diversity

    accepted = 0
    total_improvement = 0.0
    tolerance = 1e-12
    while len(available) >= 2:
        gate_a = int(available.pop())
        a, b = map(int, indices[:, gate_a])
        old_a = score(gate_a, a, b)
        pool_count = min(candidate_pool_size, len(available))
        positions = np.asarray(
            rng.choice(len(available), size=pool_count, replace=False)
        ).tolist()
        best = None
        for position in positions:
            gate_b = int(available[position])
            c, d = map(int, indices[:, gate_b])
            old_b = score(gate_b, c, d)
            old_keys = [
                tuple(sorted((a, b))),
                tuple(sorted((c, d))),
            ]
            for new_a, new_b in (
                ((a, d), (c, b)),
                ((a, c), (b, d)),
            ):
                if new_a[0] == new_a[1] or new_b[0] == new_b[1]:
                    continue
                new_keys = [tuple(sorted(new_a)), tuple(sorted(new_b))]
                if new_keys[0] == new_keys[1]:
                    continue
                if any(
                    pair_counts.get(key, 0) - old_keys.count(key) > 0
                    for key in new_keys
                ):
                    continue
                improvement = (
                    score(gate_a, *new_a)
                    + score(gate_b, *new_b)
                    - old_a
                    - old_b
                )
                tie_key = (*new_a, *new_b, gate_b)
                if improvement > tolerance and (
                    best is None
                    or improvement > best[0] + tolerance
                    or (
                        abs(improvement - best[0]) <= tolerance
                        and tie_key < best[1]
                    )
                ):
                    best = (
                        improvement,
                        tie_key,
                        position,
                        gate_b,
                        new_a,
                        new_b,
                        old_keys,
                        new_keys,
                    )
        if best is None:
            continue
        (
            improvement,
            _,
            position,
            gate_b,
            new_a,
            new_b,
            old_keys,
            new_keys,
        ) = best
        available.pop(position)
        for key in old_keys:
            pair_counts[key] -= 1
        for key in new_keys:
            pair_counts[key] = pair_counts.get(key, 0) + 1
        indices[:, gate_a] = new_a
        indices[:, gate_b] = new_b
        changed[[gate_a, gate_b]] = True
        accepted += 1
        total_improvement += float(improvement)

    return TaskAwareRewireResult(
        indices=np.ascontiguousarray(indices),
        changed_mask=changed,
        accepted_swaps=accepted,
        eligible_gates=eligible,
        score_improvement=total_improvement,
        construction_seconds=time.perf_counter() - started,
    )


class TaskSignatureCollector:
    """Capture one normal minibatch's dense activation-gradient signatures."""

    def __init__(self, model: torch.nn.Module):
        from .layers import LogicDense

        self.layers = [
            module for module in model.modules()
            if isinstance(module, LogicDense)
        ]
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        def pre_hook(_module, inputs):
            value = inputs[0]
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)
                inputs = (value, *inputs[1:])
            value.retain_grad()
            self.inputs.append(value)
            return inputs

        def forward_hook(_module, _inputs, output):
            output.retain_grad()
            self.outputs.append(output)

        for layer in self.layers:
            self.handles.append(layer.register_forward_pre_hook(pre_hook))
            self.handles.append(layer.register_forward_hook(forward_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def _signature(
        value: torch.Tensor,
        labels: torch.Tensor,
        class_count: int,
    ) -> np.ndarray:
        if value.grad is None:
            raise RuntimeError("task-signature tensor has no retained gradient")
        saliency = (value.detach() * value.grad.detach()).abs().to(torch.float32)
        saliency = saliency.reshape(labels.shape[0], -1)
        sums = torch.zeros(
            class_count,
            saliency.shape[1],
            dtype=saliency.dtype,
            device=saliency.device,
        )
        sums.index_add_(0, labels, saliency)
        counts = torch.bincount(labels, minlength=class_count).clamp_min(1)
        sums /= counts.to(sums.dtype).unsqueeze(1)
        return sums.transpose(0, 1).cpu().numpy()

    def signatures(
        self,
        labels: torch.Tensor,
        class_count: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(self.inputs) != len(self.layers):
            raise RuntimeError("not every dense layer was captured")
        return [
            (
                self._signature(layer_input, labels, class_count),
                self._signature(layer_output, labels, class_count),
            )
            for layer_input, layer_output in zip(self.inputs, self.outputs)
        ]


def rewire_fixed_dense_model(
    model: torch.nn.Module,
    signatures: list[tuple[np.ndarray, np.ndarray]],
    *,
    topology_seed: int,
    change_fraction: float,
    candidate_pool_size: int,
    diversity_weight: float,
) -> list[dict[str, object]]:
    """Apply one task-aware refinement to every fixed dense layer in-place."""
    from .layers import LogicDense

    layers = [
        module for module in model.modules()
        if isinstance(module, LogicDense)
    ]
    if len(layers) != len(signatures):
        raise ValueError("signature count does not match dense layer count")
    reports = []
    for layer_index, (layer, (input_signature, output_signature)) in enumerate(
        zip(layers, signatures)
    ):
        indices = getattr(layer.connections, "indices", None)
        if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
            raise ValueError("task-aware rewiring requires fixed rank-2 indices")
        if getattr(layer.connections, "weights", None) is not None:
            raise ValueError("task-aware rewiring does not support learned routing")
        before_indices = np.ascontiguousarray(
            indices.detach().cpu().numpy()
        )
        before_hash = hashlib.sha256(before_indices.tobytes()).hexdigest()
        before_degree = np.bincount(
            before_indices.reshape(-1),
            minlength=layer.in_dim,
        )
        result = task_aware_degree_preserving_refine(
            indices.detach().cpu().numpy(),
            input_signature,
            output_signature,
            topology_seed=topology_seed,
            layer_index=layer_index,
            change_fraction=change_fraction,
            candidate_pool_size=candidate_pool_size,
            diversity_weight=diversity_weight,
        )
        after_degree = np.bincount(
            result.indices.reshape(-1), minlength=layer.in_dim
        )
        if not np.array_equal(before_degree, after_degree):
            raise RuntimeError("task-aware refinement changed predecessor degree")
        indices.copy_(
            torch.from_numpy(result.indices).to(
                device=indices.device, dtype=indices.dtype
            )
        )
        reports.append({
            "layer_index": layer_index,
            "changed_gates": int(result.changed_mask.sum()),
            "eligible_gates": result.eligible_gates,
            "accepted_swaps": result.accepted_swaps,
            "score_improvement": result.score_improvement,
            "construction_seconds": result.construction_seconds,
            "predecessor_degree_preserved": True,
            "before_indices_sha256": before_hash,
            "after_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(result.indices).tobytes()
            ).hexdigest(),
        })
    return reports
