"""Operation-aware circuit cost features and calibrated ABC estimator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from torchlogix import Circuit
from torchlogix.circuit import GateOp


DEFAULT_OPERATION_COST = {
    GateOp.CONST_FALSE: 0.0,
    GateOp.CONST_TRUE: 0.0,
    GateOp.WIRE: 0.0,
    GateOp.NOT: 0.0,
    GateOp.NOT_A: 0.0,
    GateOp.NOT_B: 0.0,
    GateOp.AND: 1.0,
    GateOp.OR: 1.0,
    GateOp.NAND: 1.0,
    GateOp.NOR: 1.0,
    GateOp.AND_NOT_A: 1.0,
    GateOp.AND_NOT_B: 1.0,
    GateOp.OR_NOT_A: 1.0,
    GateOp.OR_NOT_B: 1.0,
    GateOp.XOR: 3.0,
    GateOp.XNOR: 3.0,
}

# Operation-area values (um^2) for the rank-2 SkyWater implementations used by
# Silicon-Aware Neural Networks.  This is an operation-weighted proxy over the
# exactly simplified circuit, not a placement-and-routing area measurement.
SKY130_OPERATION_AREA = {
    GateOp.CONST_FALSE: 5.713,
    GateOp.CONST_TRUE: 5.713,
    GateOp.WIRE: 7.618,
    GateOp.NOT: 5.713,
    GateOp.NOT_A: 5.713,
    GateOp.NOT_B: 5.713,
    GateOp.AND: 9.522,
    GateOp.OR: 9.522,
    GateOp.NAND: 7.618,
    GateOp.NOR: 7.618,
    GateOp.AND_NOT_A: 13.331,
    GateOp.AND_NOT_B: 13.331,
    GateOp.OR_NOT_A: 13.331,
    GateOp.OR_NOT_B: 13.331,
    GateOp.XOR: 15.235,
    GateOp.XNOR: 15.235,
}

FEATURE_NAMES = (
    "operation_aig_units",
    "sum_inputs",
    "connections",
    "logic_depth",
)


def operation_cost_by_name() -> dict[str, float]:
    return {
        operation.name: float(value)
        for operation, value in DEFAULT_OPERATION_COST.items()
    }


def circuit_features(circuit: Circuit) -> dict[str, float]:
    depths = {}
    connections = 0
    for gate in circuit.gates:
        input_depths = []
        for node_id in (gate.in0, gate.in1):
            if node_id < 0:
                continue
            connections += 1
            input_depths.append(depths.get(node_id, 0))
        depths[gate.gate_id] = 1 + max(input_depths, default=-1)
    sum_inputs = sum(len(node.input_ids) for node in circuit.sum_nodes)
    connections += sum_inputs
    live_depths = [
        depths.get(node_id, 0)
        for reduction in circuit.sum_nodes
        for node_id in reduction.input_ids
    ]
    histogram = Counter(gate.op.name for gate in circuit.gates)
    operation_units = sum(
        DEFAULT_OPERATION_COST.get(gate.op, 1.0) for gate in circuit.gates
    )
    return {
        "live_gates": float(len(circuit.gates)),
        "operation_aig_units": float(operation_units),
        "sum_inputs": float(sum_inputs),
        "connections": float(connections),
        "logic_depth": float(max(live_depths, default=0)),
        "gate_histogram": dict(sorted(histogram.items())),
        "sky130_operation_area_proxy_um2": float(
            sum(SKY130_OPERATION_AREA.get(gate.op, 0.0) for gate in circuit.gates)
        ),
    }


class SynthCostEstimator:
    """A non-negative linear estimator fitted to same-flow ABC measurements."""

    def __init__(
        self,
        coefficients: dict[str, float] | None = None,
        intercept: float = 0.0,
        metadata: dict | None = None,
    ):
        self.coefficients = coefficients or {
            "operation_aig_units": 1.0,
            "sum_inputs": 4.0,
            "connections": 0.05,
            "logic_depth": 0.25,
        }
        self.intercept = float(intercept)
        self.metadata = metadata or {"kind": "analytic-prior"}

    def estimate_from_features(self, features: dict) -> float:
        return float(
            max(
                0.0,
                self.intercept
                + sum(
                    self.coefficients.get(name, 0.0) * float(features[name])
                    for name in FEATURE_NAMES
                ),
            )
        )

    def estimate(self, circuit: Circuit) -> float:
        return self.estimate_from_features(circuit_features(circuit))

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "feature_names": list(FEATURE_NAMES),
            "operation_cost": operation_cost_by_name(),
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SynthCostEstimator:
        return cls(
            coefficients={
                str(key): float(value)
                for key, value in data["coefficients"].items()
            },
            intercept=float(data.get("intercept", 0.0)),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> SynthCostEstimator:
        return cls.from_dict(json.loads(path.read_text()))


def fit_nonnegative_estimator(records: list[dict]) -> SynthCostEstimator:
    """Fit a deterministic non-negative ridge model to ABC AND-node counts."""
    if len(records) < len(FEATURE_NAMES) + 1:
        raise ValueError("at least five synthesis records are required")
    from scipy.optimize import lsq_linear

    x = np.asarray(
        [
            [float(record["features"][name]) for name in FEATURE_NAMES]
            + [1.0]
            for record in records
        ],
        dtype=np.float64,
    )
    y = np.asarray([record["abc_and_nodes"] for record in records], dtype=np.float64)
    scale = np.maximum(np.linalg.norm(x, axis=0), 1.0)
    scaled = x / scale
    regularization = 1e-4
    augmented_x = np.vstack(
        (scaled, np.sqrt(regularization) * np.eye(scaled.shape[1]))
    )
    augmented_y = np.concatenate((y, np.zeros(scaled.shape[1])))
    fit = lsq_linear(
        augmented_x,
        augmented_y,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
    )
    parameters = fit.x / scale
    predictions = x @ parameters
    errors = predictions - y
    coefficients = {
        name: float(parameters[index])
        for index, name in enumerate(FEATURE_NAMES)
    }
    return SynthCostEstimator(
        coefficients=coefficients,
        intercept=float(parameters[-1]),
        metadata={
            "kind": "same-flow-nonnegative-ridge",
            "records": len(records),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_absolute_percentage_error": float(
                np.mean(np.abs(errors) / np.maximum(y, 1.0))
            ),
            "training_records": records,
        },
    )
