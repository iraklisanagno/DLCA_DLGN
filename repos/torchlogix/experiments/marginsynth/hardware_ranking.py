"""Structural hardware-gain features for MarginSynth LUT replacements.

The estimator is deliberately static and inexpensive.  It never calls Yosys
or ABC while repairing thousands of learned edits.  Instead, it describes an
edit using its operation-cost reduction, constant-propagation opportunity,
direct fan-out, and fixed-topology downstream influence.  Coefficients are
calibrated separately against same-flow ABC measurements and frozen before a
method is transferred to another seed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from experiments.marginsynth.liveness_activity import (
    CONSTANTS_AND_ROUTING_IDS,
    LUT_TRUTH_TABLE,
)


AIG_LUT_COSTS = torch.tensor(
    (0, 1, 1, 0, 1, 0, 3, 1, 1, 3, 0, 1, 0, 1, 1, 0),
    dtype=torch.float64,
)
HARDWARE_FEATURE_NAMES = (
    "operation_gain",
    "constant_propagation_gain",
    "fanout_log",
    "downstream_influence_log",
)
BINARY_LUT_IDS = frozenset(set(range(16)) - set(CONSTANTS_AND_ROUTING_IDS))


def _connection_indices(layer) -> torch.Tensor:
    indices = getattr(getattr(layer, "connections", None), "indices", None)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise TypeError("hardware ranking requires fixed dense connections")
    if indices.shape[0] != 2:
        raise ValueError("hardware ranking currently supports rank-2 LUTs only")
    return indices.detach().cpu().to(torch.long)


def _truth_table_id(bits: torch.Tensor) -> int:
    weights = torch.tensor((8, 4, 2, 1), dtype=torch.long)
    return int((bits.to(torch.long) * weights).sum())


def cofactor_lut_id(lut_id: int, input_index: int, value: int) -> int:
    """Return the rank-2 ID after fixing input A/B to zero or one."""
    if lut_id not in range(16):
        raise ValueError("lut_id must be in [0, 15]")
    if input_index not in (0, 1) or value not in (0, 1):
        raise ValueError("input_index and value must be binary")
    truth = LUT_TRUTH_TABLE[lut_id]
    if input_index == 0:
        unary = truth[2 * value : 2 * value + 2]
        expanded = torch.stack((unary[0], unary[1], unary[0], unary[1]))
    else:
        unary = truth[[value, 2 + value]]
        expanded = torch.stack((unary[0], unary[0], unary[1], unary[1]))
    return _truth_table_id(expanded)


COFACTOR_LUT_IDS = torch.tensor(
    [
        [
            [cofactor_lut_id(lut_id, input_index, value) for value in (0, 1)]
            for input_index in (0, 1)
        ]
        for lut_id in range(16)
    ],
    dtype=torch.long,
)


def direct_fanout_counts(layers: Sequence) -> dict[int, torch.Tensor]:
    """Count direct uses of each layer output in the following logic layer."""
    result = {
        index: torch.zeros(int(layer.out_dim), dtype=torch.float64)
        for index, layer in enumerate(layers)
    }
    for layer_index in range(len(layers) - 1):
        indices = _connection_indices(layers[layer_index + 1]).reshape(-1)
        result[layer_index] = torch.bincount(
            indices, minlength=int(layers[layer_index].out_dim)
        ).to(torch.float64)
    # Every final-layer output contributes once to its class reduction in the
    # dense architectures used here.
    result[len(layers) - 1].fill_(1.0)
    return result


def downstream_influence(layers: Sequence) -> dict[int, torch.Tensor]:
    """Return fixed-topology path multiplicity from gates to class reductions."""
    if not layers:
        raise ValueError("at least one logic layer is required")
    result: dict[int, torch.Tensor] = {
        len(layers) - 1: torch.ones(
            int(layers[-1].out_dim), dtype=torch.float64
        )
    }
    for layer_index in range(len(layers) - 2, -1, -1):
        consumer_influence = result[layer_index + 1]
        indices = _connection_indices(layers[layer_index + 1])
        values = torch.cat((consumer_influence, consumer_influence))
        result[layer_index] = torch.zeros(
            int(layers[layer_index].out_dim), dtype=torch.float64
        ).scatter_add_(0, indices.reshape(-1), values)
    return result


def constant_propagation_gains(
    layers: Sequence,
    lut_ids: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Estimate downstream AIG savings when a gate is forced to 0 or 1.

    The calculation follows constants through the frozen hard LUT network.
    Unary cofactors stop propagation but still remove the consumer's local AIG
    operation.  Reconvergent paths are intentionally additive, so this remains
    an estimator rather than an exact synthesis result.
    """
    if not layers:
        raise ValueError("at least one logic layer is required")
    gains = {
        index: torch.zeros((int(layer.out_dim), 2), dtype=torch.float64)
        for index, layer in enumerate(layers)
    }
    for layer_index in range(len(layers) - 2, -1, -1):
        consumer_index = layer_index + 1
        consumer_ids = lut_ids[consumer_index].detach().cpu().to(torch.long)
        if len(consumer_ids) != int(layers[consumer_index].out_dim):
            raise ValueError("LUT IDs do not match layer width")
        indices = _connection_indices(layers[consumer_index])
        old_cost = AIG_LUT_COSTS[consumer_ids]
        duplicate_inputs = indices[0] == indices[1]

        for input_index in (0, 1):
            active = ~duplicate_inputs
            predecessor = indices[input_index, active]
            active_ids = consumer_ids[active]
            for value in (0, 1):
                cofactor = COFACTOR_LUT_IDS[active_ids, input_index, value]
                contribution = old_cost[active].clone()
                constant_zero = cofactor == 0
                constant_one = cofactor == 15
                if bool(constant_zero.any()):
                    contribution[constant_zero] += gains[consumer_index][
                        active.nonzero().flatten()[constant_zero], 0
                    ]
                if bool(constant_one.any()):
                    contribution[constant_one] += gains[consumer_index][
                        active.nonzero().flatten()[constant_one], 1
                    ]
                gains[layer_index][:, value].scatter_add_(
                    0, predecessor, contribution
                )

        # A consumer fed twice by the same predecessor must be counted once.
        duplicate_units = duplicate_inputs.nonzero().flatten()
        if len(duplicate_units):
            predecessor = indices[0, duplicate_units]
            duplicate_ids = consumer_ids[duplicate_units]
            truth = LUT_TRUTH_TABLE[duplicate_ids]
            for value in (0, 1):
                output_value = truth[:, 3 * value].to(torch.long)
                contribution = old_cost[duplicate_units] + gains[consumer_index][
                    duplicate_units, output_value
                ]
                gains[layer_index][:, value].scatter_add_(
                    0, predecessor, contribution
                )
    return gains


@dataclass(frozen=True)
class StructuralHardwareModel:
    coefficients: dict[str, float]
    alternative_binary_penalty: float = 2.0
    metadata: dict | None = None

    def __post_init__(self) -> None:
        missing = set(HARDWARE_FEATURE_NAMES) - set(self.coefficients)
        if missing:
            raise ValueError(f"missing hardware coefficients: {sorted(missing)}")
        if any(
            not math.isfinite(float(self.coefficients[name]))
            or float(self.coefficients[name]) < 0.0
            for name in HARDWARE_FEATURE_NAMES
        ):
            raise ValueError("hardware coefficients must be finite and nonnegative")
        if not math.isfinite(self.alternative_binary_penalty) or self.alternative_binary_penalty < 0:
            raise ValueError("alternative_binary_penalty must be nonnegative")

    def estimate(self, features: Mapping[str, float], *, alternative_binary: bool) -> float:
        adjusted_operation = float(features["operation_gain"])
        if alternative_binary:
            adjusted_operation -= self.alternative_binary_penalty
        total = self.coefficients["operation_gain"] * adjusted_operation
        total += sum(
            self.coefficients[name] * float(features[name])
            for name in HARDWARE_FEATURE_NAMES[1:]
        )
        return float(total)

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "kind": "structural-hardware-gain",
            "feature_names": list(HARDWARE_FEATURE_NAMES),
            "coefficients": {
                name: float(self.coefficients[name])
                for name in HARDWARE_FEATURE_NAMES
            },
            "alternative_binary_penalty": float(self.alternative_binary_penalty),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> StructuralHardwareModel:
        if payload.get("kind") != "structural-hardware-gain":
            raise ValueError("not a structural hardware-gain model")
        return cls(
            coefficients={
                name: float(payload["coefficients"][name])
                for name in HARDWARE_FEATURE_NAMES
            },
            alternative_binary_penalty=float(
                payload.get("alternative_binary_penalty", 2.0)
            ),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> StructuralHardwareModel:
        return cls.from_dict(json.loads(path.read_text()))


class StructuralFeatureIndex:
    """Precomputed graph features shared by every candidate LUT replacement."""

    def __init__(self, layers: Sequence, lut_ids: Mapping[int, torch.Tensor]):
        self.lut_ids = {
            int(index): values.detach().cpu().to(torch.long)
            for index, values in lut_ids.items()
        }
        self.fanout = direct_fanout_counts(layers)
        self.influence = downstream_influence(layers)
        self.constant_gain = constant_propagation_gains(layers, self.lut_ids)

    def features(
        self,
        layer_index: int,
        unit: int,
        old_lut: int,
        new_lut: int,
    ) -> dict[str, float]:
        structural = new_lut in CONSTANTS_AND_ROUTING_IDS
        propagation = (
            float(self.constant_gain[layer_index][unit, int(new_lut == 15)])
            if new_lut in (0, 15)
            else 0.0
        )
        return {
            "operation_gain": float(AIG_LUT_COSTS[old_lut] - AIG_LUT_COSTS[new_lut]),
            "constant_propagation_gain": propagation,
            "fanout_log": (
                math.log1p(float(self.fanout[layer_index][unit]))
                if structural
                else 0.0
            ),
            "downstream_influence_log": (
                math.log1p(float(self.influence[layer_index][unit]))
                if structural
                else 0.0
            ),
        }


def is_alternative_binary(old_lut: int, new_lut: int) -> bool:
    return new_lut in BINARY_LUT_IDS and new_lut != old_lut


def aggregate_hardware_records(records: Sequence[dict]) -> dict:
    feature_sums = {
        name: float(sum(float(record["hardware_features"][name]) for record in records))
        for name in HARDWARE_FEATURE_NAMES
    }
    return {
        "records": len(records),
        "feature_sums": feature_sums,
        "estimated_hardware_gain": float(
            sum(float(record.get("estimated_hardware_gain", 0.0)) for record in records)
        ),
        "alternative_binary_actions": sum(
            bool(record.get("alternative_binary", False)) for record in records
        ),
    }
