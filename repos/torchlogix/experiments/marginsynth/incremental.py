"""Exact affected-cone evaluation for MarginSynth gate rewrites."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from torchlogix import Circuit

from .rewrites import (
    GateRewrite,
    RewriteGroup,
    RewriteProposal,
    proposal_target_ids,
)
from .trace import PackedCalibrationTrace, _apply_gate, _unpack_words


@dataclass
class CandidateEvaluation:
    rewrite: RewriteProposal
    affected_gate_ids: list[int]
    affected_output_indices: list[int]
    scores: np.ndarray
    predictions: np.ndarray
    remaining_margins: np.ndarray
    decision_flip_count: int
    decision_flip_rate: float
    accuracy: float
    accuracy_change: float
    per_class_accuracy: np.ndarray
    per_class_accuracy_change: np.ndarray
    flips_by_original_class: np.ndarray
    margin_loss_mean: float
    evaluation_seconds: float
    updated_values: dict[int, np.ndarray]

    def summary(self) -> dict:
        return {
            "rewrite": self.rewrite.to_dict(),
            "affected_gate_ids": self.affected_gate_ids,
            "affected_output_indices": self.affected_output_indices,
            "decision_flip_count": self.decision_flip_count,
            "decision_flip_rate": self.decision_flip_rate,
            "accuracy": self.accuracy,
            "accuracy_change": self.accuracy_change,
            "per_class_accuracy": self.per_class_accuracy.tolist(),
            "per_class_accuracy_change": (
                self.per_class_accuracy_change.tolist()
            ),
            "flips_by_original_class": self.flips_by_original_class.tolist(),
            "remaining_margin_minimum": float(self.remaining_margins.min()),
            "remaining_margin_mean": float(self.remaining_margins.mean()),
            "margin_loss_mean": self.margin_loss_mean,
            "evaluation_seconds": self.evaluation_seconds,
        }


def _validate_trace(circuit: Circuit, trace: PackedCalibrationTrace) -> None:
    if trace.n_inputs != circuit.n_inputs:
        raise ValueError("trace input count does not match Circuit")
    if trace.gate_ids.tolist() != [gate.gate_id for gate in circuit.gates]:
        raise ValueError("trace gate IDs/order do not match Circuit")
    if trace.gate_ops.tolist() != [gate.op.name for gate in circuit.gates]:
        raise ValueError("trace gate operations do not match Circuit")
    if trace.gate_in0.tolist() != [gate.in0 for gate in circuit.gates]:
        raise ValueError("trace gate A inputs do not match Circuit")
    if trace.gate_in1.tolist() != [gate.in1 for gate in circuit.gates]:
        raise ValueError("trace gate B inputs do not match Circuit")


def evaluate_rewrite(
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    rewrite: GateRewrite,
    margin_reserve: float = 0.0,
    validate_trace: bool = True,
) -> CandidateEvaluation:
    """Evaluate one rewrite by resimulating only its transitive fan-out cone."""
    start = time.perf_counter()
    if validate_trace:
        _validate_trace(circuit, trace)
    row_by_id = trace.row_by_id
    gate_index_by_id = {
        int(gate_id): index
        for index, gate_id in enumerate(trace.gate_ids.tolist())
    }
    if rewrite.target_id not in gate_index_by_id:
        raise ValueError(f"rewrite target {rewrite.target_id} is not a live gate")

    target_index = gate_index_by_id[rewrite.target_id]
    original_target = circuit.gates[target_index]
    replacement = rewrite.replacement_gate(original_target)
    downstream_indices = trace.affected_gate_indices(rewrite.target_id)
    affected_output_indices = trace.affected_output_indices(rewrite.target_id)

    valid_bits = trace.valid_last_word_bits
    last_word_mask = np.uint64(
        (1 << valid_bits) - 1
        if valid_bits < 64
        else np.iinfo(np.uint64).max
    )
    zero = np.zeros(trace.values.shape[1], dtype=np.uint64)
    updated_values: dict[int, np.ndarray] = {}

    def value(node_id: int) -> np.ndarray:
        row = row_by_id.get(node_id)
        if row is None:
            return zero
        return updated_values.get(row, trace.values[row])

    target_row = row_by_id[rewrite.target_id]
    updated_values[target_row] = _apply_gate(
        replacement.op,
        value(replacement.in0),
        value(replacement.in1),
        last_word_mask,
    )
    for gate_index in downstream_indices:
        gate = circuit.gates[gate_index]
        row = row_by_id[gate.gate_id]
        updated_values[row] = _apply_gate(
            gate.op,
            value(gate.in0),
            value(gate.in1),
            last_word_mask,
        )

    scores = trace.scores.copy()
    sum_by_id = circuit._sum_by_id
    for output_index in affected_output_indices:
        output_id = circuit.outputs[output_index]
        reduction = sum_by_id.get(output_id)
        if reduction is None:
            row = row_by_id[output_id]
            output_values = _unpack_words(
                updated_values.get(row, trace.values[row]),
                trace.num_samples,
            )
            scores[:, output_index] = output_values.astype(np.float32)
            continue
        if reduction.input_ids:
            words = np.stack(
                [
                    updated_values.get(
                        row_by_id[node_id],
                        trace.values[row_by_id[node_id]],
                    )
                    for node_id in reduction.input_ids
                ]
            )
            summed = _unpack_words(words, trace.num_samples).sum(
                axis=0,
                dtype=np.int64,
            )
        else:
            summed = np.zeros(trace.num_samples, dtype=np.int64)
        scores[:, output_index] = (
            (summed + reduction.beta) / reduction.tau
        ).astype(np.float32)

    predictions = scores.argmax(axis=1).astype(np.int64)
    flips = predictions != trace.predictions
    correct = predictions == trace.labels
    accuracy = float(correct.mean())
    baseline_accuracy = float(trace.correct.mean())

    class_counts = np.bincount(trace.labels, minlength=trace.num_outputs)
    class_correct = np.bincount(
        trace.labels,
        weights=correct,
        minlength=trace.num_outputs,
    )
    baseline_class_correct = np.bincount(
        trace.labels,
        weights=trace.correct,
        minlength=trace.num_outputs,
    )
    per_class_accuracy = np.divide(
        class_correct,
        class_counts,
        out=np.zeros(trace.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    baseline_per_class_accuracy = np.divide(
        baseline_class_correct,
        class_counts,
        out=np.zeros(trace.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    flips_by_original_class = np.bincount(
        trace.predictions[flips],
        minlength=trace.num_outputs,
    )

    challenger_scores = scores.copy()
    challenger_scores[
        np.arange(trace.num_samples),
        trace.predictions,
    ] = -np.inf
    remaining_margins = (
        scores[np.arange(trace.num_samples), trace.predictions]
        - challenger_scores.max(axis=1)
    ).astype(np.float32)
    margin_loss = np.maximum(0.0, margin_reserve - remaining_margins)

    return CandidateEvaluation(
        rewrite=rewrite,
        affected_gate_ids=[
            int(trace.gate_ids[index]) for index in downstream_indices
        ],
        affected_output_indices=affected_output_indices,
        scores=scores,
        predictions=predictions,
        remaining_margins=remaining_margins,
        decision_flip_count=int(flips.sum()),
        decision_flip_rate=float(flips.mean()),
        accuracy=accuracy,
        accuracy_change=accuracy - baseline_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_accuracy_change=(
            per_class_accuracy - baseline_per_class_accuracy
        ),
        flips_by_original_class=flips_by_original_class,
        margin_loss_mean=float(margin_loss.mean()),
        evaluation_seconds=time.perf_counter() - start,
        updated_values=updated_values,
    )


def evaluate_proposal(
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    proposal: RewriteProposal,
    margin_reserve: float = 0.0,
    validate_trace: bool = True,
) -> CandidateEvaluation:
    """Evaluate an individual or coordinated proposal on its affected union."""
    if isinstance(proposal, GateRewrite):
        return evaluate_rewrite(
            circuit,
            trace,
            proposal,
            margin_reserve,
            validate_trace=validate_trace,
        )
    return evaluate_rewrite_group(
        circuit,
        trace,
        proposal,
        margin_reserve,
        validate_trace=validate_trace,
    )


def evaluate_rewrite_group(
    circuit: Circuit,
    trace: PackedCalibrationTrace,
    proposal: RewriteGroup,
    margin_reserve: float = 0.0,
    validate_trace: bool = True,
) -> CandidateEvaluation:
    """Exactly simulate the union of fan-out cones for an atomic group."""
    start = time.perf_counter()
    if validate_trace:
        _validate_trace(circuit, trace)
    row_by_id = trace.row_by_id
    gate_index_by_id = {
        int(gate_id): index
        for index, gate_id in enumerate(trace.gate_ids.tolist())
    }
    targets = proposal_target_ids(proposal)
    if any(target not in gate_index_by_id for target in targets):
        raise ValueError("coordinated proposal contains a non-live target")

    replacement_by_id = {}
    for rewrite in proposal.rewrites:
        original = circuit.gates[gate_index_by_id[rewrite.target_id]]
        replacement_by_id[rewrite.target_id] = rewrite.replacement_gate(original)
    affected_indices = set()
    affected_outputs = set()
    for target in targets:
        affected_indices.add(gate_index_by_id[target])
        affected_indices.update(trace.affected_gate_indices(target))
        affected_outputs.update(trace.affected_output_indices(target))

    valid_bits = trace.valid_last_word_bits
    last_word_mask = np.uint64(
        (1 << valid_bits) - 1
        if valid_bits < 64
        else np.iinfo(np.uint64).max
    )
    zero = np.zeros(trace.values.shape[1], dtype=np.uint64)
    updated_values: dict[int, np.ndarray] = {}

    def value(node_id: int) -> np.ndarray:
        row = row_by_id.get(node_id)
        if row is None:
            return zero
        return updated_values.get(row, trace.values[row])

    for gate_index in sorted(affected_indices):
        original = circuit.gates[gate_index]
        gate = replacement_by_id.get(original.gate_id, original)
        row = row_by_id[original.gate_id]
        updated_values[row] = _apply_gate(
            gate.op,
            value(gate.in0),
            value(gate.in1),
            last_word_mask,
        )

    scores = trace.scores.copy()
    sum_by_id = circuit._sum_by_id
    affected_output_indices = sorted(affected_outputs)
    for output_index in affected_output_indices:
        output_id = circuit.outputs[output_index]
        reduction = sum_by_id.get(output_id)
        if reduction is None:
            row = row_by_id[output_id]
            output_values = _unpack_words(
                updated_values.get(row, trace.values[row]),
                trace.num_samples,
            )
            scores[:, output_index] = output_values.astype(np.float32)
            continue
        if reduction.input_ids:
            words = np.stack(
                [
                    updated_values.get(
                        row_by_id[node_id],
                        trace.values[row_by_id[node_id]],
                    )
                    for node_id in reduction.input_ids
                ]
            )
            summed = _unpack_words(words, trace.num_samples).sum(
                axis=0,
                dtype=np.int64,
            )
        else:
            summed = np.zeros(trace.num_samples, dtype=np.int64)
        scores[:, output_index] = (
            (summed + reduction.beta) / reduction.tau
        ).astype(np.float32)

    predictions = scores.argmax(axis=1).astype(np.int64)
    flips = predictions != trace.predictions
    correct = predictions == trace.labels
    accuracy = float(correct.mean())
    baseline_accuracy = float(trace.correct.mean())
    class_counts = np.bincount(trace.labels, minlength=trace.num_outputs)
    class_correct = np.bincount(
        trace.labels,
        weights=correct,
        minlength=trace.num_outputs,
    )
    baseline_class_correct = np.bincount(
        trace.labels,
        weights=trace.correct,
        minlength=trace.num_outputs,
    )
    per_class_accuracy = np.divide(
        class_correct,
        class_counts,
        out=np.zeros(trace.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    baseline_per_class_accuracy = np.divide(
        baseline_class_correct,
        class_counts,
        out=np.zeros(trace.num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    flips_by_original_class = np.bincount(
        trace.predictions[flips],
        minlength=trace.num_outputs,
    )
    challenger_scores = scores.copy()
    challenger_scores[
        np.arange(trace.num_samples),
        trace.predictions,
    ] = -np.inf
    remaining_margins = (
        scores[np.arange(trace.num_samples), trace.predictions]
        - challenger_scores.max(axis=1)
    ).astype(np.float32)
    margin_loss = np.maximum(0.0, margin_reserve - remaining_margins)
    return CandidateEvaluation(
        rewrite=proposal,
        affected_gate_ids=[
            int(trace.gate_ids[index])
            for index in sorted(affected_indices)
            if int(trace.gate_ids[index]) not in targets
        ],
        affected_output_indices=affected_output_indices,
        scores=scores,
        predictions=predictions,
        remaining_margins=remaining_margins,
        decision_flip_count=int(flips.sum()),
        decision_flip_rate=float(flips.mean()),
        accuracy=accuracy,
        accuracy_change=accuracy - baseline_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_accuracy_change=(
            per_class_accuracy - baseline_per_class_accuracy
        ),
        flips_by_original_class=flips_by_original_class,
        margin_loss_mean=float(margin_loss.mean()),
        evaluation_seconds=time.perf_counter() - start,
        updated_values=updated_values,
    )
