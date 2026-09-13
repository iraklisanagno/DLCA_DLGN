"""Shared rank-two construction primitives; frozen TorchLogix remains unchanged."""
from __future__ import annotations

import hashlib
import numpy as np

from torchlogix.topology import (
    _multiscale_saturation_balanced_indices,
    _semantic_butterfly_indices,
)


def fingerprint(indices):
    array = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def validate_indices(indices, in_dim):
    array = np.asarray(indices)
    if (array.ndim != 2 or array.shape[0] != 2 or array.shape[1] < 1
            or not np.issubdtype(array.dtype, np.integer)
            or array.min() < 0 or array.max() >= in_dim
            or np.any(array[0] == array[1])):
        raise ValueError("expected bounded integer rank-two indices with distinct parents")
    return np.ascontiguousarray(array, dtype=np.int64)


def nominal_indices(in_dim, out_dim, *, layer_index=0, topology_seed=0,
                    input_semantics=None):
    """UR1-N: ancestry-free, index-equivalent interface to frozen nominal wiring.

    Identical primitive for dense predecessor sets and convolutional channel
    sets. Geometry/threshold encoding is explicit metadata, never tuned per data
    set. Diagnostics requiring ancestry are intentionally not produced here.
    """
    if in_dim < 2 or out_dim < 1:
        raise ValueError("at least two predecessors and one gate required")
    if input_semantics is not None:
        if input_semantics.n_inputs != in_dim:
            raise ValueError("semantic dimensions differ")
        indices = _semantic_butterfly_indices(input_semantics, out_dim,
                                              layer_index, topology_seed)
    else:
        # With novelty disabled the frozen selector reads only shape[0]. This
        # zero-byte shape carrier is NOT a dummy source-support calculation.
        shape_only = np.empty((in_dim, 0), dtype=np.uint64)
        indices = _multiscale_saturation_balanced_indices(
            shape_only, out_dim, layer_index=layer_index,
            topology_seed=topology_seed, select_novelty=False)
    return validate_indices(indices, in_dim)


def degree_null(indices, in_dim, *, seed, rounds_per_slot=40):
    """Fixed within-slot disjoint swap sweeps, not a uniform graph sampler.

    Forty sweeps per slot attempt approximately 20 proposals per edge. Sweeps
    are disjoint gate pairings, so vectorized accepted swaps do not conflict.
    Reject swaps that introduce a repeated parent. Preserve EACH node's degree
    in EACH input slot exactly. No accuracy-dependent tuning or RNG side effects.
    """
    reference = validate_indices(indices, in_dim)
    if not isinstance(rounds_per_slot, int) or rounds_per_slot < 1:
        raise ValueError("positive integer sweeps required")
    rng = np.random.default_rng(seed)
    output = reference.copy()
    width = output.shape[1]
    proposed = accepted = changed = 0
    for _ in range(rounds_per_slot):
        for slot in (0, 1):
            order = rng.permutation(width)
            a, b = order[:width // 2], order[width // 2:2 * (width // 2)]
            left, right = output[slot, a].copy(), output[slot, b].copy()
            valid = ((right != output[1 - slot, a]) &
                     (left != output[1 - slot, b]))
            proposed += len(a)
            accepted += int(valid.sum())
            changed += int((valid & (left != right)).sum())
            output[slot, a[valid]] = right[valid]
            output[slot, b[valid]] = left[valid]
    validate_indices(output, in_dim)
    for slot in (0, 1):
        if not np.array_equal(np.bincount(output[slot], minlength=in_dim),
                              np.bincount(reference[slot], minlength=in_dim)):
            raise RuntimeError("null changed per-node slot degrees")
    # A changed unordered pair multiset rules out merely permuting gate order.
    def pairs(array):
        pair = np.sort(array, axis=0)
        return np.sort(pair[0] * np.int64(in_dim) + pair[1])
    return output, dict(seed=int(seed), rounds_per_slot=rounds_per_slot,
        proposals=proposed, accepted=accepted, nontrivial_swaps=changed,
        unchanged_edge_fraction=float(np.mean(output == reference)),
        changed_pair_multiset=not np.array_equal(pairs(output), pairs(reference)),
        exact_per_node_slot_degrees=True, reference_sha256=fingerprint(reference),
        output_sha256=fingerprint(output))


def factorial_indices(reference, dimensions, arm, topology_seed):
    if arm not in {"SS", "SR", "RS", "RR"} or len(reference) != len(dimensions):
        raise ValueError("invalid factorial specification")
    result, records = [], []
    for depth, (indices, in_dim) in enumerate(zip(reference, dimensions)):
        structured = arm[0 if depth == 0 else 1] == "S"
        if structured:
            output = validate_indices(indices, in_dim).copy()
            record = dict(structured=True, output_sha256=fingerprint(output))
        else:
            entropy = [20260906, int(topology_seed), depth]
            seed = int(np.random.SeedSequence(entropy).generate_state(1)[0])
            output, record = degree_null(indices, in_dim, seed=seed)
            record.update(structured=False, seed_entropy=entropy)
        result.append(output)
        records.append(dict(depth=depth, **record))
    return result, records
