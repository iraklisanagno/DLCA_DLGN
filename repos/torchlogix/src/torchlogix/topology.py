"""Fixed dense topology construction and ancestry analysis.

Coverage-aware construction is deliberately kept separate from the training
path.  The functions in this module run on CPU, return immutable integer
indices, and do not introduce trainable or deployed routing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
import torch


_POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)

STRATEGY_ALIASES = {
    "random": "random",
    "random-unique": "random_unique",
    "random_unique": "random_unique",
    "local-cyclic": "local_cyclic",
    "local_cyclic": "local_cyclic",
    "butterfly": "butterfly",
    "coverage-greedy": "coverage_greedy",
    "coverage_greedy": "coverage_greedy",
    "coverage-hybrid": "coverage_hybrid",
    "coverage_hybrid": "coverage_hybrid",
    "semantic-balanced-hybrid": "semantic_balanced_hybrid",
    "semantic_balanced_hybrid": "semantic_balanced_hybrid",
    "semantic-classifier-hybrid": "semantic_classifier_hybrid",
    "semantic_classifier_hybrid": "semantic_classifier_hybrid",
    "semantic-channel-hybrid": "semantic_channel_hybrid",
    "semantic_channel_hybrid": "semantic_channel_hybrid",
    "ancestry-channel-hybrid": "ancestry_channel_hybrid",
    "ancestry_channel_hybrid": "ancestry_channel_hybrid",
    "coverage-reuse-hybrid": "coverage_reuse_hybrid",
    "coverage_reuse_hybrid": "coverage_reuse_hybrid",
}


def canonical_strategy(strategy: str) -> str:
    """Return the canonical underscore spelling for a topology strategy."""
    try:
        return STRATEGY_ALIASES[strategy]
    except KeyError as exc:
        choices = ", ".join(sorted(STRATEGY_ALIASES))
        raise ValueError(f"Unknown fixed topology strategy {strategy!r}; choose from {choices}") from exc


def strategy_choices() -> list[str]:
    """CLI spellings accepted for fixed dense topology strategies."""
    return sorted(STRATEGY_ALIASES)


def packed_identity(n_inputs: int) -> np.ndarray:
    """Create packed singleton ancestry sets for ``n_inputs`` features."""
    if n_inputs <= 0:
        raise ValueError("n_inputs must be positive")
    words = (n_inputs + 63) // 64
    ancestry = np.zeros((n_inputs, words), dtype=np.uint64)
    rows = np.arange(n_inputs)
    ancestry[rows, rows // 64] = np.left_shift(
        np.uint64(1), (rows % 64).astype(np.uint64)
    )
    return ancestry


def packed_popcount(words: np.ndarray, axis: int = -1) -> np.ndarray:
    """Count set bits in packed uint64 arrays without Python set objects."""
    words = np.ascontiguousarray(words, dtype=np.uint64)
    byte_view = words.view(np.uint8).reshape(*words.shape, 8)
    normalized_axis = axis % words.ndim
    return _POPCOUNT[byte_view].sum(axis=(normalized_axis, byte_view.ndim - 1))


def _row_popcount(words: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(words, dtype=np.uint64)
    return _POPCOUNT[words.view(np.uint8).reshape(words.shape[0], -1)].sum(axis=1)


def propagate_packed_ancestry(
    input_ancestry: np.ndarray,
    indices: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Union predecessor ancestry sets for every output gate."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    input_ancestry = np.asarray(input_ancestry, dtype=np.uint64)
    if indices.ndim != 2:
        raise ValueError(f"indices must have shape (rank, out_dim), got {indices.shape}")
    if input_ancestry.ndim != 2:
        raise ValueError("input_ancestry must have shape (in_dim, packed_words)")
    if indices.size and (indices.min() < 0 or indices.max() >= input_ancestry.shape[0]):
        raise ValueError("connection index is outside input ancestry bounds")
    selected = input_ancestry[indices]
    return np.bitwise_or.reduce(selected, axis=0)


def packed_identity_in_universe(
    count: int,
    universe_size: int,
    *,
    offset: int = 0,
) -> np.ndarray:
    """Create singleton ancestry rows inside a shared packed-bit universe."""
    if count <= 0:
        raise ValueError("count must be positive")
    if universe_size < count + offset:
        raise ValueError("universe_size is too small for count and offset")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    words = (universe_size + 63) // 64
    ancestry = np.zeros((count, words), dtype=np.uint64)
    positions = offset + np.arange(count, dtype=np.int64)
    ancestry[np.arange(count), positions // 64] = np.left_shift(
        np.uint64(1), (positions % 64).astype(np.uint64)
    )
    return ancestry


def add_identity_to_ancestry(
    ancestry: np.ndarray,
    *,
    offset: int,
    universe_size: int,
) -> np.ndarray:
    """Add one stage-local identity bit to every ancestry row."""
    ancestry = np.asarray(ancestry, dtype=np.uint64)
    if ancestry.ndim != 2:
        raise ValueError("ancestry must have shape (count, packed_words)")
    identity = packed_identity_in_universe(
        ancestry.shape[0],
        universe_size,
        offset=offset,
    )
    if ancestry.shape[1] > identity.shape[1]:
        raise ValueError("ancestry uses a larger universe than requested")
    result = np.zeros_like(identity)
    result[:, :ancestry.shape[1]] = ancestry
    result |= identity
    return result


def combine_channel_spatial_ancestry(
    channel_ancestry: np.ndarray,
    *,
    spatial_positions: int,
) -> np.ndarray:
    """Expand channel ancestry to flattened channel-major spatial features.

    Each flattened activation retains its upstream channel ancestry and gains
    a unique source bit. This lets a dense classifier schedule jointly measure
    feature-channel diversity and spatial-source coverage.
    """
    channel_ancestry = np.asarray(channel_ancestry, dtype=np.uint64)
    if channel_ancestry.ndim != 2:
        raise ValueError(
            "channel_ancestry must have shape (channels, packed_words)"
        )
    if spatial_positions <= 0:
        raise ValueError("spatial_positions must be positive")
    channels = channel_ancestry.shape[0]
    upstream_universe = channel_ancestry.shape[1] * 64
    flattened = np.repeat(channel_ancestry, spatial_positions, axis=0)
    unique = packed_identity_in_universe(
        channels * spatial_positions,
        upstream_universe + channels * spatial_positions,
        offset=upstream_universe,
    )
    result = unique
    result[:, :channel_ancestry.shape[1]] |= flattened
    return result


@dataclass(frozen=True)
class DenseTopologyResult:
    """Construction result; only ``indices`` is deployed with the model."""

    indices: np.ndarray
    output_ancestry: np.ndarray
    construction_seconds: float
    temporary_bytes: int
    greedy_mask: np.ndarray


@dataclass(frozen=True)
class InputSemantics:
    """Semantic coordinates of flattened thermometer-encoded image inputs.

    ``source_ids`` collapse all threshold bits belonging to one raw scalar
    input. For grayscale images a source is one pixel; for color images it is
    one channel-pixel. The remaining arrays retain the tensor coordinates used
    by the first-layer semantic schedule and diagnostics.
    """

    source_ids: np.ndarray
    threshold_ids: np.ndarray
    channel_ids: np.ndarray
    y: np.ndarray
    x: np.ndarray
    channels: int
    height: int
    width: int
    threshold_bits: int
    layout: str

    @property
    def n_inputs(self) -> int:
        return int(self.source_ids.size)

    @property
    def n_sources(self) -> int:
        return int(self.channels * self.height * self.width)

    def source_ancestry(self) -> np.ndarray:
        ancestry = packed_identity(self.n_sources)
        return ancestry[self.source_ids]


def image_input_semantics(
    channels: int,
    height: int,
    width: int,
    threshold_bits: int,
    *,
    layout: str,
) -> InputSemantics:
    """Describe the exact flattening used by TorchLogix image binarizers."""
    if min(channels, height, width, threshold_bits) <= 0:
        raise ValueError("image semantic dimensions must all be positive")
    if layout not in {"pixel_interleaved", "channel_interleaved"}:
        raise ValueError(
            "layout must be 'pixel_interleaved' or 'channel_interleaved'"
        )
    coordinates = []
    if layout == "pixel_interleaved":
        # Binarization merges the threshold axis with image width:
        # (channel, y, x, threshold).
        for channel in range(channels):
            for row in range(height):
                for column in range(width):
                    for threshold in range(threshold_bits):
                        coordinates.append((channel, row, column, threshold))
    else:
        # CIFAR binarization merges thresholds with channels:
        # (channel, threshold, y, x).
        for channel in range(channels):
            for threshold in range(threshold_bits):
                for row in range(height):
                    for column in range(width):
                        coordinates.append((channel, row, column, threshold))
    array = np.asarray(coordinates, dtype=np.int64)
    channel_ids, y, x, threshold_ids = array.T
    source_ids = channel_ids * (height * width) + y * width + x
    return InputSemantics(
        source_ids=np.ascontiguousarray(source_ids),
        threshold_ids=np.ascontiguousarray(threshold_ids),
        channel_ids=np.ascontiguousarray(channel_ids),
        y=np.ascontiguousarray(y),
        x=np.ascontiguousarray(x),
        channels=channels,
        height=height,
        width=width,
        threshold_bits=threshold_bits,
        layout=layout,
    )


def _validate_dimensions(
    in_dim: int,
    out_dim: int,
    lut_rank: int,
    *,
    allow_partial_input_coverage: bool = False,
) -> None:
    if in_dim < lut_rank:
        raise ValueError(f"in_dim ({in_dim}) must be at least lut_rank ({lut_rank})")
    if out_dim <= 0:
        raise ValueError("out_dim must be positive")
    if not allow_partial_input_coverage and out_dim * lut_rank < in_dim:
        raise ValueError(
            f"out_dim * lut_rank must cover the inputs ({out_dim} * {lut_rank} < {in_dim})"
        )


def _random_indices(
    in_dim: int,
    out_dim: int,
    lut_rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_slots = lut_rank * out_dim
    if n_slots < in_dim:
        values = rng.permutation(in_dim)[:n_slots]
    else:
        values = rng.permutation(n_slots) % in_dim
    return values.reshape(lut_rank, out_dim).astype(np.int64, copy=False)


def _random_unique_pairs(
    in_dim: int,
    out_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    max_pairs = in_dim * (in_dim - 1) // 2
    if out_dim > max_pairs:
        raise ValueError(
            f"Cannot create {out_dim} unique rank-2 pairs from {in_dim} inputs; maximum is {max_pairs}"
        )
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    while len(pairs) < out_dim:
        remaining = out_dim - len(pairs)
        draw = max(32, remaining * 2)
        left = rng.integers(0, in_dim, size=draw, dtype=np.int64)
        right = rng.integers(0, in_dim - 1, size=draw, dtype=np.int64)
        right += right >= left
        for a, b in zip(left.tolist(), right.tolist()):
            key = (min(a, b), max(a, b))
            if key in used:
                continue
            used.add(key)
            pairs.append((a, b))
            if len(pairs) == out_dim:
                break
    return np.asarray(pairs, dtype=np.int64).T


def _local_cyclic_indices(
    in_dim: int,
    out_dim: int,
    layer_index: int,
    local_radius: int,
) -> np.ndarray:
    if local_radius < 1:
        raise ValueError("local_radius must be at least one")
    gate = np.arange(out_dim, dtype=np.int64)
    left = gate % in_dim
    offset = 1 + ((gate // in_dim + layer_index) % min(local_radius, in_dim - 1))
    right = (left + offset) % in_dim
    return np.stack((left, right))


def _coprime_step(length: int, key: int) -> int:
    if length <= 1:
        return 1
    step = 1 + (key % (length - 1))
    while math.gcd(step, length) != 1:
        step = 1 + (step % (length - 1))
    return step


def _butterfly_indices(
    in_dim: int,
    out_dim: int,
    layer_index: int,
    topology_seed: int = 0,
) -> np.ndarray:
    n_stages = max(1, math.ceil(math.log2(in_dim)))
    pairs: list[tuple[int, int]] = []
    stage_offset = 0
    while len(pairs) < out_dim:
        stride = 1 << ((layer_index + stage_offset) % n_stages)
        stage_pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for left in range(in_dim):
            if in_dim & (in_dim - 1) == 0:
                right = left ^ stride
            else:
                right = (left + stride) % in_dim
            key = (min(left, right), max(left, right))
            if left == right or key in seen:
                continue
            seen.add(key)
            stage_pairs.append((left, right))
        # Affine reordering keeps the exact same regular pair set and fan-out,
        # but prevents consecutive output/class groups from seeing consecutive
        # flattened image regions.  The permutation is formulaic and requires
        # no deployed routing state beyond the generated indices already used
        # by every TorchLogix fixed topology.
        permutation_key = (
            int(topology_seed)
            + 0x9E3779B1 * (layer_index + 1)
            + 0x85EBCA77 * (stage_offset + 1)
        )
        step = _coprime_step(len(stage_pairs), permutation_key)
        offset = (permutation_key // max(1, len(stage_pairs))) % len(stage_pairs)
        order = (offset + step * np.arange(len(stage_pairs))) % len(stage_pairs)
        for pair_index in order.tolist():
            pair = stage_pairs[pair_index]
            pairs.append(pair)
            if len(pairs) == out_dim:
                break
        stage_offset += 1
    return np.asarray(pairs, dtype=np.int64).T


def _axis_strides(length: int) -> list[int]:
    """Return deterministic local-to-global cyclic butterfly strides."""
    if length <= 1:
        return []
    strides = []
    stride = 1
    while stride < length:
        strides.append(min(stride, length // 2))
        if strides[-1] == length // 2:
            break
        stride *= 2
    return list(dict.fromkeys(strides))


def _semantic_butterfly_indices(
    semantics: InputSemantics,
    out_dim: int,
    layer_index: int,
    topology_seed: int,
) -> np.ndarray:
    """Build a balanced first-layer butterfly over image coordinates.

    Spatial stages retain channel and threshold identity. Channel stages mix
    aligned color samples, while threshold stages also move spatially so two
    correlated thermometer bits from the same raw scalar are never paired.
    """
    lookup = {
        (int(channel), int(row), int(column), int(threshold)): index
        for index, (channel, row, column, threshold) in enumerate(zip(
            semantics.channel_ids,
            semantics.y,
            semantics.x,
            semantics.threshold_ids,
        ))
    }
    stages: list[tuple[str, int]] = []
    x_strides = _axis_strides(semantics.width)
    y_strides = _axis_strides(semantics.height)
    for level in range(max(len(x_strides), len(y_strides))):
        if level < len(x_strides):
            stages.append(("x", x_strides[level]))
        if level < len(y_strides):
            stages.append(("y", y_strides[level]))
        # Mix the non-spatial semantic axes at the finest scale, before the
        # schedule expands to longer spatial strides.
        if level == 0 and semantics.channels > 1:
            stages.append(("channel", 1))
        if level == 0 and semantics.threshold_bits > 1:
            stages.append(("threshold_spatial", 1))
    if not stages:
        raise ValueError("semantic butterfly needs at least two raw sources")

    ordered_stages: list[list[tuple[int, int]]] = []
    for stage_offset, (axis, stride) in enumerate(stages):
        stage_pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for left in range(semantics.n_inputs):
            channel = int(semantics.channel_ids[left])
            row = int(semantics.y[left])
            column = int(semantics.x[left])
            threshold = int(semantics.threshold_ids[left])
            if axis == "x":
                column = (column + stride) % semantics.width
            elif axis == "y":
                row = (row + stride) % semantics.height
            elif axis == "channel":
                channel = (channel + stride) % semantics.channels
            elif axis == "threshold_spatial":
                threshold = (threshold + stride) % semantics.threshold_bits
                column = (column + 1) % semantics.width
            else:  # pragma: no cover - stages are constructed locally
                raise AssertionError(axis)
            right = lookup[(channel, row, column, threshold)]
            if semantics.source_ids[left] == semantics.source_ids[right]:
                continue
            key = (min(left, right), max(left, right))
            if key in seen:
                continue
            seen.add(key)
            stage_pairs.append(key)

        permutation_key = (
            int(topology_seed)
            + 0x9E3779B1 * (layer_index + 1)
            + 0x85EBCA77 * (stage_offset + 1)
        )
        step = _coprime_step(len(stage_pairs), permutation_key)
        offset = (permutation_key // max(1, len(stage_pairs))) % len(stage_pairs)
        order = (offset + step * np.arange(len(stage_pairs))) % len(stage_pairs)
        ordered_stages.append([stage_pairs[index] for index in order.tolist()])

    # Round-robin across coordinate axes. Within each stage, select from a
    # bounded look-ahead window by current fan-out. This retains equal semantic
    # axis representation without the highly uneven prefixes produced by
    # merely truncating independently shuffled stage lists.
    pairs: list[tuple[int, int]] = []
    cursors = np.zeros(len(ordered_stages), dtype=np.int64)
    fanout = np.zeros(semantics.n_inputs, dtype=np.int64)
    used: set[tuple[int, int]] = set()
    remaining = np.asarray(
        [len(stage) for stage in ordered_stages], dtype=np.int64
    )
    pair_stages: dict[tuple[int, int], list[int]] = {}
    for stage_index, stage in enumerate(ordered_stages):
        for pair in stage:
            pair_stages.setdefault(pair, []).append(stage_index)
    # Once every distinct pair in a stage has been used, the original rule
    # chooses the global minimum by (fan-out sum, maximum fan-out, left,
    # right). A lazy heap implements exactly that ordering without rescanning
    # the complete stage for every additional output in wide networks.
    exhausted_heaps: list[list[tuple[int, int, int, int]] | None] = [
        None for _ in ordered_stages
    ]
    look_ahead = 64
    for gate in range(out_dim):
        stage_index = (gate + layer_index) % len(ordered_stages)
        stage = ordered_stages[stage_index]
        if remaining[stage_index] == 0:
            heap = exhausted_heaps[stage_index]
            if heap is None:
                heap = [
                    (
                        int(fanout[left] + fanout[right]),
                        int(max(fanout[left], fanout[right])),
                        left,
                        right,
                    )
                    for left, right in stage
                ]
                heapq.heapify(heap)
                exhausted_heaps[stage_index] = heap
            while True:
                stored = heapq.heappop(heap)
                left, right = stored[2], stored[3]
                current = (
                    int(fanout[left] + fanout[right]),
                    int(max(fanout[left], fanout[right])),
                    left,
                    right,
                )
                if stored == current:
                    pair = (left, right)
                    break
                heapq.heappush(heap, current)
        else:
            candidates = []
            attempts = 0
            while len(candidates) < min(look_ahead, len(stage)):
                position = int(cursors[stage_index] % len(stage))
                cursors[stage_index] += 1
                attempts += 1
                candidate = stage[position]
                if candidate not in used:
                    candidates.append(candidate)
                if attempts > 2 * len(stage):
                    break
            if not candidates:  # pragma: no cover - tracked by remaining
                raise RuntimeError(
                    "semantic stage exhaustion tracking is inconsistent"
                )
            left = np.asarray(
                [candidate[0] for candidate in candidates], dtype=np.int64
            )
            right = np.asarray(
                [candidate[1] for candidate in candidates], dtype=np.int64
            )
            score = fanout[left] + fanout[right]
            maximum = np.maximum(fanout[left], fanout[right])
            order = np.lexsort((right, left, maximum, score))
            pair = candidates[int(order[0])]
        pairs.append(pair)
        if pair not in used:
            used.add(pair)
            for containing_stage in pair_stages[pair]:
                remaining[containing_stage] -= 1
        fanout[pair[0]] += 1
        fanout[pair[1]] += 1
        heap = exhausted_heaps[stage_index]
        if heap is not None:
            left, right = pair
            heapq.heappush(
                heap,
                (
                    int(fanout[left] + fanout[right]),
                    int(max(fanout[left], fanout[right])),
                    left,
                    right,
                ),
            )
    return np.asarray(pairs, dtype=np.int64).T


def _candidate_pairs(
    rng: np.random.Generator,
    in_dim: int,
    pool_size: int,
    *,
    long_range: bool,
    local_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    min_distance = min(max(local_radius + 1, in_dim // 4), max(1, in_dim // 2))
    attempts = 0
    max_unique_attempts = max(100, pool_size * 50)
    while len(pairs) < pool_size:
        attempts += 1
        a = int(rng.integers(in_dim))
        if long_range and in_dim > 3:
            max_distance = max(min_distance, in_dim - min_distance)
            offset = int(rng.integers(min_distance, max_distance + 1))
            b = (a + offset) % in_dim
        else:
            b = int(rng.integers(in_dim - 1))
            b += b >= a
        key = (min(a, b), max(a, b))
        if key in seen:
            # Tiny cyclic graphs may have fewer eligible long-range pairs than
            # the requested pool.  After a bounded search, retain duplicate
            # candidates rather than stalling topology construction.
            if attempts >= max_unique_attempts:
                pairs.append((a, b))
            continue
        seen.add(key)
        pairs.append((a, b))
    array = np.asarray(pairs, dtype=np.int64)
    return array[:, 0], array[:, 1]


def _cyclic_distance(left: np.ndarray, right: np.ndarray, size: int) -> np.ndarray:
    distance = np.abs(left - right)
    return np.minimum(distance, size - distance).astype(np.float64)


def _select_greedy_pair(
    input_ancestry: np.ndarray,
    fanout: np.ndarray,
    used_pairs: set[tuple[int, int]],
    rng: np.random.Generator,
    candidate_pool_size: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    local_radius: int,
    long_range: bool,
) -> tuple[int, int, int]:
    in_dim = input_ancestry.shape[0]
    left, right = _candidate_pairs(
        rng,
        in_dim,
        candidate_pool_size,
        long_range=long_range,
        local_radius=local_radius,
    )
    a_sets = input_ancestry[left]
    b_sets = input_ancestry[right]
    union_count = _row_popcount(np.bitwise_or(a_sets, b_sets)).astype(np.float64)
    overlap_count = _row_popcount(np.bitwise_and(a_sets, b_sets)).astype(np.float64)
    fanout_penalty = (fanout[left] + fanout[right]).astype(np.float64)
    distance_penalty = _cyclic_distance(left, right, in_dim) / max(1.0, in_dim / 2.0)
    score = (
        alpha * union_count
        - beta * overlap_count
        - gamma * fanout_penalty
        - delta * distance_penalty
    )
    duplicate = np.asarray(
        [(min(int(a), int(b)), max(int(a), int(b))) in used_pairs for a, b in zip(left, right)]
    )
    if np.any(~duplicate):
        score = np.where(duplicate, -np.inf, score)
    best_score = np.max(score)
    tied = np.flatnonzero(score == best_score)
    if tied.size > 1:
        tie_order = np.lexsort((right[tied], left[tied]))
        best = int(tied[tie_order[0]])
    else:
        best = int(tied[0])
    temporary_bytes = sum(
        value.nbytes
        for value in (
            left,
            right,
            a_sets,
            b_sets,
            union_count,
            overlap_count,
            fanout_penalty,
            distance_penalty,
            score,
            duplicate,
        )
    )
    return int(left[best]), int(right[best]), temporary_bytes


def _greedy_indices(
    input_ancestry: np.ndarray,
    out_dim: int,
    rng: np.random.Generator,
    candidate_pool_size: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    local_radius: int,
) -> tuple[np.ndarray, int]:
    in_dim = input_ancestry.shape[0]
    indices = np.empty((2, out_dim), dtype=np.int64)
    fanout = np.zeros(in_dim, dtype=np.int64)
    used_pairs: set[tuple[int, int]] = set()
    temporary_bytes = indices.nbytes + fanout.nbytes
    for gate in range(out_dim):
        left, right, candidate_bytes = _select_greedy_pair(
            input_ancestry,
            fanout,
            used_pairs,
            rng,
            candidate_pool_size,
            alpha,
            beta,
            gamma,
            delta,
            local_radius,
            False,
        )
        indices[:, gate] = (left, right)
        fanout[left] += 1
        fanout[right] += 1
        used_pairs.add((min(left, right), max(left, right)))
        temporary_bytes = max(temporary_bytes, candidate_bytes + indices.nbytes + fanout.nbytes)
    return indices, temporary_bytes


def _hybrid_indices(
    input_ancestry: np.ndarray,
    out_dim: int,
    layer_index: int,
    topology_seed: int,
    rng: np.random.Generator,
    long_range_fraction: float,
    candidate_pool_size: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    local_radius: int,
    hybrid_base: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 <= long_range_fraction <= 1.0:
        raise ValueError("long_range_fraction must be in [0, 1]")
    in_dim = input_ancestry.shape[0]
    if hybrid_base == "butterfly":
        indices = _butterfly_indices(in_dim, out_dim, layer_index, topology_seed)
    elif hybrid_base in {"local_cyclic", "local-cyclic"}:
        indices = _local_cyclic_indices(in_dim, out_dim, layer_index, local_radius)
    else:
        raise ValueError("hybrid_base must be 'butterfly' or 'local_cyclic'")

    n_greedy = int(round(long_range_fraction * out_dim))
    greedy_mask = np.zeros(out_dim, dtype=bool)
    if n_greedy == 0:
        return indices, greedy_mask, indices.nbytes
    positions = np.linspace(0, out_dim - 1, n_greedy, dtype=np.int64)
    greedy_mask[positions] = True
    fanout = np.bincount(indices[:, ~greedy_mask].reshape(-1), minlength=in_dim).astype(np.int64)
    used_pairs = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in indices[:, ~greedy_mask].T
    }
    temporary_bytes = indices.nbytes + greedy_mask.nbytes + fanout.nbytes
    for gate in positions.tolist():
        left, right, candidate_bytes = _select_greedy_pair(
            input_ancestry,
            fanout,
            used_pairs,
            rng,
            candidate_pool_size,
            alpha,
            beta,
            gamma,
            delta,
            local_radius,
            True,
        )
        indices[:, gate] = (left, right)
        fanout[left] += 1
        fanout[right] += 1
        used_pairs.add((min(left, right), max(left, right)))
        temporary_bytes = max(
            temporary_bytes,
            candidate_bytes + indices.nbytes + greedy_mask.nbytes + fanout.nbytes,
        )
    return indices, greedy_mask, temporary_bytes


def _single_popcount(words: np.ndarray) -> int:
    return int(_POPCOUNT[np.ascontiguousarray(words).view(np.uint8)].sum())


def _sample_group_references(
    gate: int,
    out_dim: int,
    output_groups: int,
    count: int = 8,
) -> np.ndarray:
    group = min(output_groups - 1, gate * output_groups // out_dim)
    start = group * out_dim // output_groups
    stop = (group + 1) * out_dim // output_groups
    size = stop - start
    if size <= 1:
        return np.empty(0, dtype=np.int64)
    step = _coprime_step(size, gate + 0x9E3779B1)
    refs = []
    offset = (gate - start + step) % size
    while len(refs) < min(count, size - 1):
        candidate = start + offset
        if candidate != gate and candidate not in refs:
            refs.append(candidate)
        offset = (offset + step) % size
    return np.asarray(refs, dtype=np.int64)


def _balanced_gate_score(
    left: int,
    right: int,
    *,
    gate: int,
    input_ancestry: np.ndarray,
    current_output_ancestry: np.ndarray,
    out_dim: int,
    output_groups: int,
    novelty_weight: float,
    base_pair_counts: dict[tuple[int, int], int] | None = None,
    max_base_pair_count: int = 1,
    reuse_weight: float = 0.0,
) -> tuple[float, np.ndarray]:
    ancestry = np.bitwise_or(input_ancestry[left], input_ancestry[right])
    intersection = np.bitwise_and(input_ancestry[left], input_ancestry[right])
    union_count = _single_popcount(ancestry)
    input_mass = (
        _single_popcount(input_ancestry[left])
        + _single_popcount(input_ancestry[right])
    )
    union_efficiency = union_count / max(1, input_mass)
    within_overlap = _single_popcount(intersection) / max(1, union_count)
    refs = _sample_group_references(gate, out_dim, output_groups)
    if refs.size:
        reference_sets = current_output_ancestry[refs]
        intersections = _row_popcount(
            np.bitwise_and(reference_sets, ancestry)
        ).astype(np.float64)
        unions = _row_popcount(
            np.bitwise_or(reference_sets, ancestry)
        ).astype(np.float64)
        mean_jaccard = float(np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions != 0,
        ).mean())
    else:
        mean_jaccard = 0.0
    pair_key = (min(left, right), max(left, right))
    base_pair_support = (
        base_pair_counts.get(pair_key, 0) / max(1, max_base_pair_count)
        if base_pair_counts is not None
        else 0.0
    )
    return (
        union_efficiency - within_overlap
        + novelty_weight * (1.0 - mean_jaccard)
        + reuse_weight * base_pair_support,
        ancestry,
    )


def _degree_preserving_coverage_swaps(
    indices: np.ndarray,
    input_ancestry: np.ndarray,
    *,
    rng: np.random.Generator,
    swap_fraction: float,
    candidate_pool_size: int,
    output_groups: int,
    novelty_weight: float,
    base_pair_counts: dict[tuple[int, int], int] | None = None,
    reuse_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Improve semantic ancestry using accepted 2-edge swaps.

    Each accepted move uses exactly the same four predecessor occurrences as
    before, so the complete predecessor degree sequence is an invariant.
    """
    if not 0.0 <= swap_fraction <= 1.0:
        raise ValueError("coverage_swap_fraction must be in [0, 1]")
    if output_groups < 1:
        raise ValueError("coverage_output_groups must be positive")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    if reuse_weight < 0.0:
        raise ValueError("coverage_reuse_weight must be non-negative")
    indices = np.array(indices, dtype=np.int64, copy=True)
    out_dim = indices.shape[1]
    changed = np.zeros(out_dim, dtype=bool)
    target_gates = min(out_dim, int(round(swap_fraction * out_dim)))
    target_gates -= target_gates % 2
    if target_gates == 0:
        ancestry = propagate_packed_ancestry(input_ancestry, indices)
        return indices, changed, indices.nbytes + ancestry.nbytes

    current_output = propagate_packed_ancestry(input_ancestry, indices)
    pair_counts: dict[tuple[int, int], int] = {}
    for left, right in indices.T:
        key = (min(int(left), int(right)), max(int(left), int(right)))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    max_base_pair_count = (
        max(base_pair_counts.values(), default=1)
        if base_pair_counts is not None
        else 1
    )

    permutation = rng.permutation(out_dim)
    available = list(permutation[:target_gates])
    temporary_bytes = indices.nbytes + current_output.nbytes + changed.nbytes
    tolerance = 1e-12
    while len(available) >= 2:
        gate_a = int(available.pop())
        a, b = map(int, indices[:, gate_a])
        old_score_a, _ = _balanced_gate_score(
            a,
            b,
            gate=gate_a,
            input_ancestry=input_ancestry,
            current_output_ancestry=current_output,
            out_dim=out_dim,
            output_groups=output_groups,
            novelty_weight=novelty_weight,
            base_pair_counts=base_pair_counts,
            max_base_pair_count=max_base_pair_count,
            reuse_weight=reuse_weight,
        )
        pool_count = min(candidate_pool_size, len(available))
        pool_positions = rng.choice(len(available), size=pool_count, replace=False)
        best = None
        for pool_position in np.asarray(pool_positions).tolist():
            gate_b = int(available[pool_position])
            c, d = map(int, indices[:, gate_b])
            old_keys = [
                (min(a, b), max(a, b)),
                (min(c, d), max(c, d)),
            ]
            old_score_b, _ = _balanced_gate_score(
                c,
                d,
                gate=gate_b,
                input_ancestry=input_ancestry,
                current_output_ancestry=current_output,
                out_dim=out_dim,
                output_groups=output_groups,
                novelty_weight=novelty_weight,
                base_pair_counts=base_pair_counts,
                max_base_pair_count=max_base_pair_count,
                reuse_weight=reuse_weight,
            )
            for proposed_a, proposed_b in (
                ((a, d), (c, b)),
                ((a, c), (b, d)),
            ):
                if proposed_a[0] == proposed_a[1] or proposed_b[0] == proposed_b[1]:
                    continue
                new_keys = [
                    tuple(sorted(proposed_a)),
                    tuple(sorted(proposed_b)),
                ]
                if new_keys[0] == new_keys[1]:
                    continue
                if base_pair_counts is None and any(
                    pair_counts.get(key, 0) - old_keys.count(key) > 0
                    for key in new_keys
                ):
                    continue
                if base_pair_counts is not None and any(
                    pair_counts.get(key, 0) - old_keys.count(key) > 0
                    and base_pair_counts.get(key, 0) == 0
                    for key in new_keys
                ):
                    continue
                new_score_a, ancestry_a = _balanced_gate_score(
                    *proposed_a,
                    gate=gate_a,
                    input_ancestry=input_ancestry,
                    current_output_ancestry=current_output,
                    out_dim=out_dim,
                    output_groups=output_groups,
                    novelty_weight=novelty_weight,
                    base_pair_counts=base_pair_counts,
                    max_base_pair_count=max_base_pair_count,
                    reuse_weight=reuse_weight,
                )
                new_score_b, ancestry_b = _balanced_gate_score(
                    *proposed_b,
                    gate=gate_b,
                    input_ancestry=input_ancestry,
                    current_output_ancestry=current_output,
                    out_dim=out_dim,
                    output_groups=output_groups,
                    novelty_weight=novelty_weight,
                    base_pair_counts=base_pair_counts,
                    max_base_pair_count=max_base_pair_count,
                    reuse_weight=reuse_weight,
                )
                improvement = (
                    new_score_a + new_score_b - old_score_a - old_score_b
                )
                tie_key = (*proposed_a, *proposed_b, gate_b)
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
                        pool_position,
                        gate_b,
                        proposed_a,
                        proposed_b,
                        ancestry_a,
                        ancestry_b,
                        old_keys,
                        new_keys,
                    )
        if best is None:
            continue
        (
            _,
            _,
            pool_position,
            gate_b,
            proposed_a,
            proposed_b,
            ancestry_a,
            ancestry_b,
            old_keys,
            new_keys,
        ) = best
        available.pop(pool_position)
        for key in old_keys:
            pair_counts[key] -= 1
        for key in new_keys:
            pair_counts[key] = pair_counts.get(key, 0) + 1
        indices[:, gate_a] = proposed_a
        indices[:, gate_b] = proposed_b
        current_output[gate_a] = ancestry_a
        current_output[gate_b] = ancestry_b
        changed[[gate_a, gate_b]] = True
    return indices, changed, int(temporary_bytes)


def coverage_reuse_refine(
    base_indices: np.ndarray,
    input_ancestry: np.ndarray,
    *,
    topology_seed: int = 0,
    layer_index: int = 0,
    change_fraction: float = 0.25,
    candidate_pool_size: int = 8,
    novelty_weight: float = 1.0,
    reuse_weight: float = 1.0,
    output_groups: int = 1,
) -> DenseTopologyResult:
    """Refine any rank-2 topology while balancing coverage and base reuse.

    The transformation is architecture-independent: it accepts fixed integer
    indices plus packed input ancestry. Accepted two-edge swaps preserve the
    exact predecessor degree sequence. ``change_fraction`` bounds the fraction
    of outputs that may differ, while ``reuse_weight`` rewards predecessor
    motifs already present in the supplied base topology.
    """
    started = time.perf_counter()
    base_indices = np.asarray(base_indices, dtype=np.int64)
    input_ancestry = np.asarray(input_ancestry, dtype=np.uint64)
    if base_indices.ndim != 2 or base_indices.shape[0] != 2:
        raise ValueError("base_indices must have shape (2, out_dim)")
    if input_ancestry.ndim != 2:
        raise ValueError("input_ancestry must have shape (in_dim, packed_words)")
    if base_indices.size and (
        base_indices.min() < 0
        or base_indices.max() >= input_ancestry.shape[0]
    ):
        raise ValueError("base_indices contain an out-of-bounds predecessor")
    base_pair_counts: dict[tuple[int, int], int] = {}
    for left, right in base_indices.T:
        key = (min(int(left), int(right)), max(int(left), int(right)))
        base_pair_counts[key] = base_pair_counts.get(key, 0) + 1
    seed = (
        int(topology_seed)
        + 0x94D049BB * (int(layer_index) + 1)
    ) % (1 << 63)
    indices, changed, swap_bytes = _degree_preserving_coverage_swaps(
        base_indices,
        input_ancestry,
        rng=np.random.default_rng(seed),
        swap_fraction=change_fraction,
        candidate_pool_size=candidate_pool_size,
        output_groups=output_groups,
        novelty_weight=novelty_weight,
        base_pair_counts=base_pair_counts,
        reuse_weight=reuse_weight,
    )
    output_ancestry = propagate_packed_ancestry(input_ancestry, indices)
    temporary_bytes = max(
        swap_bytes,
        base_indices.nbytes + input_ancestry.nbytes + output_ancestry.nbytes,
    )
    return DenseTopologyResult(
        indices=np.ascontiguousarray(indices, dtype=np.int64),
        output_ancestry=np.ascontiguousarray(output_ancestry, dtype=np.uint64),
        construction_seconds=time.perf_counter() - started,
        temporary_bytes=int(temporary_bytes),
        greedy_mask=changed,
    )


def _balanced_round_robin_pairs(
    in_dim: int,
    out_dim: int,
    *,
    layer_index: int,
    topology_seed: int,
    semantic_threshold_count: int | None = None,
) -> np.ndarray:
    """Return balanced, maximally distinct pairs via graph factorization.

    Complete-graph rounds use every input at most once per round. Prefixes
    therefore have tightly balanced fan-out, unlike truncating a shuffled list
    of all pairs. For encoded RGB inputs, threshold-major player ordering makes
    early rounds mix colors and thresholds rather than depend on flattened
    channel adjacency.
    """
    if in_dim < 2:
        raise ValueError("in_dim must be at least two")
    if out_dim <= 0:
        raise ValueError("out_dim must be positive")
    if (
        semantic_threshold_count is not None
        and semantic_threshold_count > 0
        and in_dim % semantic_threshold_count == 0
    ):
        colors = in_dim // semantic_threshold_count
        players = [
            color * semantic_threshold_count + threshold
            for threshold in range(semantic_threshold_count)
            for color in range(colors)
        ]
    else:
        players = list(range(in_dim))

    dummy = None
    if len(players) % 2:
        dummy = in_dim
        players.append(dummy)
    key = (
        int(topology_seed)
        + 0x9E3779B1 * (int(layer_index) + 1)
    )
    total_rounds = len(players) - 1
    round_step = _coprime_step(total_rounds, key)
    round_offset = key % total_rounds
    pairs_per_round = in_dim // 2
    rounds_needed = math.ceil(out_dim / max(1, pairs_per_round))
    ordered: list[tuple[int, int]] = []
    for sequence_index in range(rounds_needed):
        cycle, position = divmod(sequence_index, total_rounds)
        round_index = (
            round_offset + round_step * position
        ) % total_rounds
        position_pairs = [(round_index, len(players) - 1)]
        for pair_index in range(1, len(players) // 2):
            position_pairs.append((
                (round_index + pair_index) % total_rounds,
                (round_index - pair_index) % total_rounds,
            ))
        round_pairs = []
        for left_position, right_position in position_pairs:
            left = players[left_position]
            right = players[right_position]
            if left == dummy or right == dummy:
                continue
            round_pairs.append((min(left, right), max(left, right)))
        pair_key = key + 0x85EBCA77 * (round_index + 1 + cycle)
        pair_step = _coprime_step(len(round_pairs), pair_key)
        pair_offset = pair_key % len(round_pairs)
        pair_order = (
            pair_offset + pair_step * np.arange(len(round_pairs))
        ) % len(round_pairs)
        for pair_position in pair_order.tolist():
            ordered.append(round_pairs[pair_position])
            if len(ordered) == out_dim:
                break
    return np.asarray(ordered, dtype=np.int64).T


def generate_conv_channel_topology(
    in_dim: int,
    out_dim: int,
    *,
    topology_seed: int = 0,
    layer_index: int = 0,
    input_ancestry: np.ndarray | None = None,
    candidate_pool_size: int = 8,
    swap_fraction: float = 0.25,
    novelty_weight: float = 1.0,
    semantic_threshold_count: int | None = None,
) -> DenseTopologyResult:
    """Construct a fixed, ancestry-aware convolutional channel schedule."""
    started = time.perf_counter()
    _validate_dimensions(in_dim, out_dim, 2)
    if input_ancestry is None:
        input_ancestry = packed_identity(in_dim)
    input_ancestry = np.asarray(input_ancestry, dtype=np.uint64)
    if input_ancestry.shape[0] != in_dim:
        raise ValueError(
            f"input_ancestry has {input_ancestry.shape[0]} rows, "
            f"expected in_dim={in_dim}"
        )
    base = _balanced_round_robin_pairs(
        in_dim,
        out_dim,
        layer_index=layer_index,
        topology_seed=topology_seed,
        semantic_threshold_count=semantic_threshold_count,
    )
    seed = (int(topology_seed) + 0xD1B54A32 * (int(layer_index) + 1)) % (1 << 63)
    rng = np.random.default_rng(seed)
    indices, changed, swap_bytes = _degree_preserving_coverage_swaps(
        base,
        input_ancestry,
        rng=rng,
        swap_fraction=swap_fraction,
        candidate_pool_size=candidate_pool_size,
        output_groups=1,
        novelty_weight=novelty_weight,
    )
    output_ancestry = propagate_packed_ancestry(input_ancestry, indices)
    temporary_bytes = max(
        input_ancestry.nbytes + indices.nbytes + output_ancestry.nbytes,
        swap_bytes,
    )
    return DenseTopologyResult(
        indices=np.ascontiguousarray(indices, dtype=np.int64),
        output_ancestry=np.ascontiguousarray(output_ancestry, dtype=np.uint64),
        construction_seconds=time.perf_counter() - started,
        temporary_bytes=int(temporary_bytes),
        greedy_mask=changed,
    )


def generate_coverage_reuse_conv_topology(
    in_dim: int,
    out_dim: int,
    *,
    topology_seed: int = 0,
    layer_index: int = 0,
    input_ancestry: np.ndarray | None = None,
    candidate_pool_size: int = 8,
    base_swap_fraction: float = 0.25,
    change_fraction: float = 0.25,
    novelty_weight: float = 1.0,
    reuse_weight: float = 1.0,
) -> DenseTopologyResult:
    """Apply generic coverage--reuse refinement to the frozen v4 base."""
    started = time.perf_counter()
    _validate_dimensions(in_dim, out_dim, 2)
    if input_ancestry is None:
        input_ancestry = packed_identity(in_dim)
    input_ancestry = np.asarray(input_ancestry, dtype=np.uint64)
    if input_ancestry.shape[0] != in_dim:
        raise ValueError(
            f"input_ancestry has {input_ancestry.shape[0]} rows, "
            f"expected in_dim={in_dim}"
        )
    # This is exactly the frozen v4 channel topology. Its ancestry remains
    # local to the current channel set; the refinement below receives the
    # propagated cross-layer ancestry.
    base = generate_dense_topology(
        in_dim,
        out_dim,
        strategy="semantic_balanced_hybrid",
        topology_seed=topology_seed,
        layer_index=layer_index,
        candidate_pool_size=candidate_pool_size,
        swap_fraction=base_swap_fraction,
        novelty_weight=novelty_weight,
    )
    refined = coverage_reuse_refine(
        base.indices,
        input_ancestry,
        topology_seed=topology_seed,
        layer_index=layer_index,
        change_fraction=change_fraction,
        candidate_pool_size=candidate_pool_size,
        novelty_weight=novelty_weight,
        reuse_weight=reuse_weight,
    )
    return DenseTopologyResult(
        indices=refined.indices,
        output_ancestry=refined.output_ancestry,
        construction_seconds=time.perf_counter() - started,
        temporary_bytes=max(base.temporary_bytes, refined.temporary_bytes),
        greedy_mask=refined.greedy_mask,
    )


def generate_dense_topology(
    in_dim: int,
    out_dim: int,
    *,
    strategy: str,
    lut_rank: int = 2,
    topology_seed: int = 0,
    layer_index: int = 0,
    input_ancestry: np.ndarray | None = None,
    candidate_pool_size: int = 64,
    long_range_fraction: float = 0.25,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 0.0,
    local_radius: int = 4,
    hybrid_base: str = "butterfly",
    input_semantics: InputSemantics | None = None,
    swap_fraction: float = 0.25,
    output_groups: int = 1,
    novelty_weight: float = 1.0,
    reuse_change_fraction: float = 0.25,
    reuse_weight: float = 1.0,
    allow_partial_input_coverage: bool = False,
) -> DenseTopologyResult:
    """Construct a fixed dense topology and its packed output ancestry."""
    started = time.perf_counter()
    strategy = canonical_strategy(strategy)
    if strategy in {"semantic_channel_hybrid", "ancestry_channel_hybrid"}:
        raise ValueError(
            f"{strategy} is a convolutional channel schedule; "
            "use semantic_balanced_hybrid for dense layers"
        )
    _validate_dimensions(
        in_dim,
        out_dim,
        lut_rank,
        allow_partial_input_coverage=allow_partial_input_coverage,
    )
    if strategy in {
        "local_cyclic",
        "butterfly",
        "coverage_greedy",
        "coverage_hybrid",
        "semantic_balanced_hybrid",
        "semantic_classifier_hybrid",
        "coverage_reuse_hybrid",
    }:
        if lut_rank != 2:
            raise NotImplementedError(f"{strategy} currently supports rank-2 LUTs only")
    if input_ancestry is None:
        input_ancestry = (
            input_semantics.source_ancestry()
            if strategy in {
                "semantic_balanced_hybrid",
                "semantic_classifier_hybrid",
                "coverage_reuse_hybrid",
            }
            and input_semantics is not None
            else packed_identity(in_dim)
        )
    input_ancestry = np.asarray(input_ancestry, dtype=np.uint64)
    if input_ancestry.shape[0] != in_dim:
        raise ValueError(
            f"input_ancestry has {input_ancestry.shape[0]} rows, expected in_dim={in_dim}"
        )

    seed = (int(topology_seed) + 0x9E3779B1 * int(layer_index)) % (1 << 63)
    rng = np.random.default_rng(seed)
    greedy_mask = np.zeros(out_dim, dtype=bool)
    temporary_bytes = input_ancestry.nbytes

    if strategy == "random":
        indices = _random_indices(in_dim, out_dim, lut_rank, rng)
    elif strategy == "random_unique":
        if lut_rank != 2:
            raise NotImplementedError("random_unique currently supports rank-2 LUTs only")
        indices = _random_unique_pairs(in_dim, out_dim, rng)
    elif strategy == "local_cyclic":
        indices = _local_cyclic_indices(in_dim, out_dim, layer_index, local_radius)
    elif strategy == "butterfly":
        indices = _butterfly_indices(in_dim, out_dim, layer_index, topology_seed)
    elif strategy == "coverage_greedy":
        indices, greedy_bytes = _greedy_indices(
            input_ancestry,
            out_dim,
            rng,
            candidate_pool_size,
            alpha,
            beta,
            gamma,
            delta,
            local_radius,
        )
        greedy_mask[:] = True
        temporary_bytes = max(temporary_bytes, greedy_bytes)
    elif strategy == "coverage_hybrid":
        indices, greedy_mask, hybrid_bytes = _hybrid_indices(
            input_ancestry,
            out_dim,
            layer_index,
            topology_seed,
            rng,
            long_range_fraction,
            candidate_pool_size,
            alpha,
            beta,
            gamma,
            delta,
            local_radius,
            hybrid_base,
        )
        temporary_bytes = max(temporary_bytes, hybrid_bytes)
    elif strategy == "semantic_balanced_hybrid":
        if input_semantics is not None:
            if input_semantics.n_inputs != in_dim:
                raise ValueError(
                    "input semantics do not match the first-layer input dimension"
                )
            base = _semantic_butterfly_indices(
                input_semantics,
                out_dim,
                layer_index,
                topology_seed,
            )
        else:
            base = _butterfly_indices(
                in_dim,
                out_dim,
                layer_index,
                topology_seed,
            )
        indices, greedy_mask, swap_bytes = _degree_preserving_coverage_swaps(
            base,
            input_ancestry,
            rng=rng,
            swap_fraction=swap_fraction,
            candidate_pool_size=candidate_pool_size,
            output_groups=output_groups,
            novelty_weight=novelty_weight,
        )
        temporary_bytes = max(temporary_bytes, swap_bytes)
    elif strategy == "semantic_classifier_hybrid":
        if input_semantics is not None:
            if input_semantics.n_inputs != in_dim:
                raise ValueError(
                    "input semantics do not match the classifier input dimension"
                )
            base = _semantic_butterfly_indices(
                input_semantics,
                out_dim,
                layer_index,
                topology_seed,
            )
        else:
            base = _balanced_round_robin_pairs(
                in_dim,
                out_dim,
                layer_index=layer_index,
                topology_seed=topology_seed,
            )
        indices, greedy_mask, swap_bytes = _degree_preserving_coverage_swaps(
            base,
            input_ancestry,
            rng=rng,
            swap_fraction=swap_fraction,
            candidate_pool_size=candidate_pool_size,
            output_groups=output_groups,
            novelty_weight=novelty_weight,
        )
        temporary_bytes = max(temporary_bytes, swap_bytes)
    elif strategy == "coverage_reuse_hybrid":
        if input_semantics is not None:
            if input_semantics.n_inputs != in_dim:
                raise ValueError(
                    "input semantics do not match the first-layer input dimension"
                )
            base = _semantic_butterfly_indices(
                input_semantics,
                out_dim,
                layer_index,
                topology_seed,
            )
        else:
            base = _butterfly_indices(
                in_dim,
                out_dim,
                layer_index,
                topology_seed,
            )
        base, _, base_bytes = _degree_preserving_coverage_swaps(
            base,
            input_ancestry,
            rng=rng,
            swap_fraction=swap_fraction,
            candidate_pool_size=candidate_pool_size,
            output_groups=output_groups,
            novelty_weight=novelty_weight,
        )
        refined = coverage_reuse_refine(
            base,
            input_ancestry,
            topology_seed=topology_seed,
            layer_index=layer_index,
            change_fraction=reuse_change_fraction,
            candidate_pool_size=candidate_pool_size,
            novelty_weight=novelty_weight,
            reuse_weight=reuse_weight,
            output_groups=output_groups,
        )
        indices = refined.indices
        greedy_mask = refined.greedy_mask
        temporary_bytes = max(
            temporary_bytes,
            base_bytes,
            refined.temporary_bytes,
        )
    else:  # pragma: no cover - canonical_strategy has already validated this
        raise AssertionError(strategy)

    if indices.min() < 0 or indices.max() >= in_dim:
        raise RuntimeError("generated topology contains an out-of-bounds index")
    output_ancestry = propagate_packed_ancestry(input_ancestry, indices)
    temporary_bytes = max(
        temporary_bytes,
        input_ancestry.nbytes + indices.nbytes + output_ancestry.nbytes,
    )
    return DenseTopologyResult(
        indices=np.ascontiguousarray(indices, dtype=np.int64),
        output_ancestry=np.ascontiguousarray(output_ancestry, dtype=np.uint64),
        construction_seconds=time.perf_counter() - started,
        temporary_bytes=int(temporary_bytes),
        greedy_mask=greedy_mask,
    )


def topology_layer_metrics(
    input_ancestry: np.ndarray,
    indices: np.ndarray | torch.Tensor,
    *,
    n_original_inputs: int,
    depth: int,
    strategy: str | None = None,
    construction_seconds: float | None = None,
    temporary_bytes: int | None = None,
) -> tuple[dict[str, int | float | str], np.ndarray]:
    """Measure one dense layer and return its output ancestry for propagation."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    output_ancestry = propagate_packed_ancestry(input_ancestry, indices)
    sizes = _row_popcount(output_ancestry).astype(np.float64)
    combined = np.bitwise_or.reduce(output_ancestry, axis=0, keepdims=True)
    covered = int(_row_popcount(combined)[0])
    fanout = np.bincount(indices.reshape(-1), minlength=input_ancestry.shape[0]).astype(np.float64)
    mean_fanout = float(fanout.mean())
    pairs = np.sort(indices.T, axis=1) if indices.shape[0] == 2 else indices.T
    distinct_pairs = int(np.unique(pairs, axis=0).shape[0])

    if indices.shape[0] == 2:
        intersections = _row_popcount(
            np.bitwise_and(input_ancestry[indices[0]], input_ancestry[indices[1]])
        ).astype(np.float64)
        unions = _row_popcount(
            np.bitwise_or(input_ancestry[indices[0]], input_ancestry[indices[1]])
        ).astype(np.float64)
        jaccard = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions != 0,
        )
    else:
        intersections = np.zeros(indices.shape[1], dtype=np.float64)
        jaccard = intersections.copy()

    index_bits = max(1, math.ceil(math.log2(input_ancestry.shape[0])))
    metrics: dict[str, int | float | str] = {
        "depth": depth,
        "strategy": canonical_strategy(strategy) if strategy is not None else "unknown",
        "in_dim": int(input_ancestry.shape[0]),
        "out_dim": int(indices.shape[1]),
        "lut_rank": int(indices.shape[0]),
        "original_inputs": int(n_original_inputs),
        "covered_inputs": covered,
        "input_coverage": covered / n_original_inputs,
        "mean_gate_ancestry": float(sizes.mean()),
        "std_gate_ancestry": float(sizes.std()),
        "max_gate_ancestry": int(sizes.max()),
        "reachable_gate_fraction": float(np.mean(sizes > 0)),
        "overlap_mean": float(intersections.mean()),
        "overlap_std": float(intersections.std()),
        "overlap_p50": float(np.quantile(intersections, 0.50)),
        "overlap_p90": float(np.quantile(intersections, 0.90)),
        "overlap_max": int(intersections.max()),
        "jaccard_overlap_mean": float(jaccard.mean()),
        "fanout_mean": mean_fanout,
        "fanout_max": int(fanout.max()),
        "fanout_cv": float(fanout.std() / mean_fanout) if mean_fanout else 0.0,
        "unused_outputs": int(np.count_nonzero(fanout == 0)),
        "distinct_predecessor_pairs": distinct_pairs,
        "duplicate_predecessor_pairs": int(indices.shape[1] - distinct_pairs),
        "deployed_index_bits": int(indices.size * index_bits),
        "deployed_index_bytes_packed": int(math.ceil(indices.size * index_bits / 8)),
        "indices_tensor_bytes": int(indices.nbytes),
        "ancestry_bytes": int(output_ancestry.nbytes),
    }
    if construction_seconds is not None:
        metrics["construction_seconds"] = float(construction_seconds)
    if temporary_bytes is not None:
        metrics["generator_temporary_bytes"] = int(temporary_bytes)
    return metrics, output_ancestry


def analyze_dense_indices(
    n_original_inputs: int,
    layers: Iterable[tuple[np.ndarray | torch.Tensor, str]],
) -> list[dict[str, int | float | str]]:
    """Analyze already-generated fixed indices without training a model."""
    ancestry = packed_identity(n_original_inputs)
    rows: list[dict[str, int | float | str]] = []
    for depth, (indices, strategy) in enumerate(layers):
        row, ancestry = topology_layer_metrics(
            ancestry,
            indices,
            n_original_inputs=n_original_inputs,
            depth=depth,
            strategy=strategy,
        )
        rows.append(row)
    return rows


def semantic_first_layer_pair_metrics(
    indices: np.ndarray | torch.Tensor,
    semantics: InputSemantics,
) -> dict[str, float]:
    """Measure whether first-layer pairs mix independent image sources."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    left, right = indices
    spatial_distance = (
        np.abs(semantics.y[left] - semantics.y[right])
        + np.abs(semantics.x[left] - semantics.x[right])
    ).astype(np.float64)
    return {
        "same_source_pair_fraction": float(np.mean(
            semantics.source_ids[left] == semantics.source_ids[right]
        )),
        "same_spatial_location_pair_fraction": float(np.mean(
            (semantics.y[left] == semantics.y[right])
            & (semantics.x[left] == semantics.x[right])
        )),
        "same_threshold_pair_fraction": float(np.mean(
            semantics.threshold_ids[left] == semantics.threshold_ids[right]
        )),
        "same_channel_pair_fraction": float(np.mean(
            semantics.channel_ids[left] == semantics.channel_ids[right]
        )),
        "spatial_manhattan_mean": float(spatial_distance.mean()),
        "spatial_manhattan_p50": float(np.quantile(spatial_distance, 0.50)),
        "spatial_manhattan_p90": float(np.quantile(spatial_distance, 0.90)),
    }


def semantic_ancestry_metrics(
    input_ancestry: np.ndarray,
    indices: np.ndarray | torch.Tensor,
    *,
    n_sources: int,
    output_groups: int,
) -> tuple[dict[str, int | float], np.ndarray]:
    """Report source-level and cross-gate ancestry diagnostics."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=np.int64)
    output = propagate_packed_ancestry(input_ancestry, indices)
    sizes = _row_popcount(output).astype(np.float64)
    combined = np.bitwise_or.reduce(output, axis=0, keepdims=True)
    covered = int(_row_popcount(combined)[0])
    intersections = _row_popcount(
        np.bitwise_and(
            input_ancestry[indices[0]],
            input_ancestry[indices[1]],
        )
    ).astype(np.float64)
    unions = _row_popcount(
        np.bitwise_or(
            input_ancestry[indices[0]],
            input_ancestry[indices[1]],
        )
    ).astype(np.float64)
    predecessor_jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions != 0,
    )

    adjacent_left = []
    adjacent_right = []
    group_coverages = []
    usage_cvs = []
    unpacked = np.unpackbits(
        np.ascontiguousarray(output).view(np.uint8),
        axis=1,
        bitorder="little",
    )[:, :n_sources]
    for group in range(output_groups):
        start = group * output.shape[0] // output_groups
        stop = (group + 1) * output.shape[0] // output_groups
        if stop - start > 1:
            adjacent_left.extend(range(start, stop - 1))
            adjacent_right.extend(range(start + 1, stop))
        group_usage = unpacked[start:stop].sum(axis=0).astype(np.float64)
        group_coverages.append(float(np.mean(group_usage > 0)))
        usage_cvs.append(
            float(group_usage.std() / group_usage.mean())
            if group_usage.mean() else 0.0
        )
    if adjacent_left:
        adjacent_left = np.asarray(adjacent_left, dtype=np.int64)
        adjacent_right = np.asarray(adjacent_right, dtype=np.int64)
        cross_intersections = _row_popcount(np.bitwise_and(
            output[adjacent_left],
            output[adjacent_right],
        )).astype(np.float64)
        cross_unions = _row_popcount(np.bitwise_or(
            output[adjacent_left],
            output[adjacent_right],
        )).astype(np.float64)
        cross_jaccard = np.divide(
            cross_intersections,
            cross_unions,
            out=np.zeros_like(cross_intersections),
            where=cross_unions != 0,
        )
        cross_mean = float(cross_jaccard.mean())
    else:
        cross_mean = 0.0
    unique_rows = int(np.unique(output, axis=0).shape[0])
    return {
        "source_original_inputs": int(n_sources),
        "source_covered_inputs": covered,
        "source_input_coverage": covered / n_sources,
        "mean_source_ancestry": float(sizes.mean()),
        "std_source_ancestry": float(sizes.std()),
        "max_source_ancestry": int(sizes.max()),
        "source_predecessor_overlap_mean": float(intersections.mean()),
        "source_predecessor_jaccard_mean": float(predecessor_jaccard.mean()),
        "source_ancestry_unique_fraction": unique_rows / output.shape[0],
        "source_cross_gate_jaccard_mean": cross_mean,
        "source_group_coverage_mean": float(np.mean(group_coverages)),
        "source_group_coverage_min": float(np.min(group_coverages)),
        "source_group_usage_cv_mean": float(np.mean(usage_cvs)),
    }, output


def generate_dense_stack(
    n_original_inputs: int,
    widths: Sequence[int],
    *,
    strategy: str,
    topology_seed: int = 0,
    input_semantics: InputSemantics | None = None,
    **kwargs,
) -> tuple[list[DenseTopologyResult], list[dict[str, int | float | str]]]:
    """Generate and analyze a stack of fixed rank-2 dense layers."""
    ancestry = (
        input_semantics.source_ancestry()
        if canonical_strategy(strategy) == "semantic_balanced_hybrid"
        and input_semantics is not None
        else packed_identity(n_original_inputs)
    )
    in_dim = n_original_inputs
    results: list[DenseTopologyResult] = []
    rows: list[dict[str, int | float | str]] = []
    for depth, out_dim in enumerate(widths):
        layer_input_ancestry = ancestry
        layer_kwargs = dict(kwargs)
        configured_output_groups = int(layer_kwargs.pop("output_groups", 1))
        layer_output_groups = (
            configured_output_groups if depth == len(widths) - 1 else 1
        )
        if (
            depth == 0
            and input_semantics is not None
            and canonical_strategy(strategy) == "semantic_balanced_hybrid"
        ):
            layer_kwargs["swap_fraction"] = 0.0
        result = generate_dense_topology(
            in_dim,
            int(out_dim),
            strategy=strategy,
            topology_seed=topology_seed,
            layer_index=depth,
            input_ancestry=ancestry,
            input_semantics=input_semantics if depth == 0 else None,
            output_groups=layer_output_groups,
            **layer_kwargs,
        )
        row, ancestry = topology_layer_metrics(
            layer_input_ancestry,
            result.indices,
            n_original_inputs=(
                input_semantics.n_sources
                if canonical_strategy(strategy) == "semantic_balanced_hybrid"
                and input_semantics is not None
                else n_original_inputs
            ),
            depth=depth,
            strategy=strategy,
            construction_seconds=result.construction_seconds,
            temporary_bytes=result.temporary_bytes,
        )
        if input_semantics is not None:
            source_row, _ = semantic_ancestry_metrics(
                layer_input_ancestry,
                result.indices,
                n_sources=input_semantics.n_sources,
                output_groups=layer_output_groups,
            )
            row.update(source_row)
            if depth == 0:
                row.update(semantic_first_layer_pair_metrics(
                    result.indices,
                    input_semantics,
                ))
        results.append(result)
        rows.append(row)
        in_dim = int(out_dim)
    return results, rows


def analyze_model_topology(model: torch.nn.Module) -> list[dict[str, int | float | str]]:
    """Extract and analyze fixed dense layers from an instantiated model."""
    dense_layers = [
        module
        for module in model.modules()
        if hasattr(module, "in_dim")
        and hasattr(module, "out_dim")
        and hasattr(module, "connections")
        and hasattr(module.connections, "indices")
        and isinstance(module.connections.indices, torch.Tensor)
        and module.connections.indices.ndim == 2
    ]
    if not dense_layers:
        return []
    n_original_inputs = int(dense_layers[0].in_dim)
    ancestry = packed_identity(n_original_inputs)
    semantics = getattr(model, "input_semantics", None)
    source_ancestry = (
        semantics.source_ancestry() if semantics is not None else None
    )
    rows: list[dict[str, int | float | str]] = []
    for depth, layer in enumerate(dense_layers):
        connections = layer.connections
        row, ancestry = topology_layer_metrics(
            ancestry,
            connections.indices,
            n_original_inputs=n_original_inputs,
            depth=depth,
            strategy=getattr(connections, "strategy", connections.init_method),
            construction_seconds=getattr(connections, "construction_seconds", None),
            temporary_bytes=getattr(connections, "generator_temporary_bytes", None),
        )
        if source_ancestry is not None:
            source_row, source_ancestry = semantic_ancestry_metrics(
                source_ancestry,
                connections.indices,
                n_sources=semantics.n_sources,
                output_groups=(
                    int(getattr(model, "class_count", 1))
                    if depth == len(dense_layers) - 1
                    else 1
                ),
            )
            row.update(source_row)
            if depth == 0:
                row.update(semantic_first_layer_pair_metrics(
                    connections.indices,
                    semantics,
                ))
        rows.append(row)
    return rows


def analyze_conv_channel_topology(
    model: torch.nn.Module,
) -> list[dict[str, int | float | str]]:
    """Measure channel routing while treating spatial indexing as immutable.

    Each grouped TorchLogix convolution uses the same selected channels for all
    sliding-window positions. Metrics therefore count one channel-group
    occurrence per output kernel, while the spatial-coordinate hash verifies
    that paired schedules leave the receptive-field samples unchanged.
    """
    layers = [
        module
        for module in model.modules()
        if hasattr(module, "connections")
        and isinstance(getattr(module.connections, "indices", None), list)
        and getattr(module.connections, "channel_group_size", None) is not None
    ]
    rows: list[dict[str, int | float | str]] = []
    input_thresholds = (
        getattr(model, "n_input_thresholds", None)
        or getattr(model, "n_input_bits", None)
    )
    raw_ancestry = (
        packed_identity(layers[0].channels) if layers else None
    )
    n_raw_channels = layers[0].channels if layers else 0
    for depth, layer in enumerate(layers):
        connections = layer.connections
        first_level = connections.indices[0].detach().cpu().numpy()
        # (rank, kernels, positions, leaves, spatial+channel)
        leaf_channels = first_level[:, :, 0, :, -1]
        groups = []
        for kernel in range(layer.num_kernels):
            group = np.unique(leaf_channels[:, kernel].reshape(-1))
            if group.size != connections.channel_group_size:
                raise ValueError(
                    f"kernel {kernel} uses {group.size} channels, expected "
                    f"{connections.channel_group_size}"
                )
            groups.append(np.sort(group))
        groups = np.asarray(groups, dtype=np.int64)
        unique_groups = np.unique(groups, axis=0)
        fanout = np.bincount(
            groups.reshape(-1), minlength=layer.channels
        ).astype(np.float64)
        pair_span = (
            np.abs(groups[:, 0] - groups[:, 1]).astype(np.float64)
            if groups.shape[1] == 2 else np.zeros(len(groups), dtype=np.float64)
        )
        spatial = np.ascontiguousarray(first_level[..., :-1], dtype=np.int64)
        tree_gates_per_kernel = sum(
            layer.lut_rank ** level for level in range(layer.tree_depth)
        )
        row: dict[str, int | float | str] = {
            "structure": "conv_channel",
            "depth": depth,
            "strategy": getattr(connections, "strategy", connections.init_method),
            "in_channels": int(layer.channels),
            "out_kernels": int(layer.num_kernels),
            "tree_depth": int(layer.tree_depth),
            "spatial_output_positions": int(first_level.shape[2]),
            "learned_tree_gates": int(
                layer.num_kernels * tree_gates_per_kernel
            ),
            "spatial_gate_applications": int(
                layer.num_kernels
                * first_level.shape[2]
                * tree_gates_per_kernel
            ),
            "channel_group_size": int(connections.channel_group_size),
            "distinct_channel_groups": int(unique_groups.shape[0]),
            "duplicate_channel_groups": int(
                layer.num_kernels - unique_groups.shape[0]
            ),
            "channel_fanout_mean": float(fanout.mean()),
            "channel_fanout_max": int(fanout.max()),
            "channel_fanout_min": int(fanout.min()),
            "channel_fanout_cv": float(
                fanout.std() / fanout.mean() if fanout.mean() else 0.0
            ),
            "unused_channels": int(np.count_nonzero(fanout == 0)),
            "channel_pair_span_mean": float(pair_span.mean()),
            "channel_pair_span_p50": float(np.quantile(pair_span, 0.5)),
            "channel_pair_span_p90": float(np.quantile(pair_span, 0.9)),
            "spatial_coordinates_sha256": hashlib.sha256(
                spatial.tobytes()
            ).hexdigest(),
            "construction_seconds": float(
                getattr(connections, "construction_seconds", 0.0)
            ),
            "generator_temporary_bytes": int(
                getattr(connections, "generator_temporary_bytes", 0)
            ),
        }
        if raw_ancestry is not None:
            if raw_ancestry.shape[0] != layer.channels:
                raise ValueError(
                    "convolutional channel ancestry does not match the next "
                    f"layer: {raw_ancestry.shape[0]} != {layer.channels}"
                )
            intersections = _row_popcount(
                np.bitwise_and(
                    raw_ancestry[groups[:, 0]],
                    raw_ancestry[groups[:, 1]],
                )
            ).astype(np.float64)
            unions = _row_popcount(
                np.bitwise_or(
                    raw_ancestry[groups[:, 0]],
                    raw_ancestry[groups[:, 1]],
                )
            ).astype(np.float64)
            predecessor_jaccard = np.divide(
                intersections,
                unions,
                out=np.zeros_like(intersections),
                where=unions != 0,
            )
            raw_ancestry = propagate_packed_ancestry(
                raw_ancestry,
                groups.T,
            )
            raw_sizes = _row_popcount(raw_ancestry).astype(np.float64)
            covered = _row_popcount(
                np.bitwise_or.reduce(
                    raw_ancestry,
                    axis=0,
                    keepdims=True,
                )
            )[0]
            row.update({
                "raw_channel_ancestry_mean": float(raw_sizes.mean()),
                "raw_channel_ancestry_min": int(raw_sizes.min()),
                "raw_channel_ancestry_max": int(raw_sizes.max()),
                "raw_channel_coverage_fraction": float(
                    covered / n_raw_channels
                ),
                "raw_predecessor_jaccard_mean": float(
                    predecessor_jaccard.mean()
                ),
            })
        if depth == 0 and input_thresholds:
            row.update({
                "same_input_channel_fraction": float(np.mean(
                    groups[:, 0] // input_thresholds
                    == groups[:, 1] // input_thresholds
                )),
                "same_threshold_fraction": float(np.mean(
                    groups[:, 0] % input_thresholds
                    == groups[:, 1] % input_thresholds
                )),
            })
        rows.append(row)
    return rows


def model_topology_metadata(model: torch.nn.Module) -> list[dict[str, int | float | str | None]]:
    """Return checkpoint-safe metadata for each fixed dense connection layer."""
    metadata = []
    for module in model.modules():
        if hasattr(module, "topology_metadata"):
            metadata.append(module.topology_metadata())
    return metadata


def write_topology_report(
    rows: Sequence[dict[str, int | float | str]],
    output_dir: str | Path,
    *,
    stem: str = "topology",
    metadata: dict | None = None,
) -> None:
    """Write raw per-depth CSV plus a JSON report without third-party tools."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    payload = {"metadata": metadata or {}, "layers": list(rows)}
    with (output_dir / f"{stem}.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
