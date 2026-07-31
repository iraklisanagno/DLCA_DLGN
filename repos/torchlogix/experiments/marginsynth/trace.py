"""Bit-packed calibration traces and graph indices for MarginSynth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchlogix import Circuit
from torchlogix.circuit import GateOp


WORD_BITS = 64


def _pack_examples(inputs: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack a ``(samples, nodes)`` Boolean matrix into uint64 node traces."""
    inputs = np.asarray(inputs, dtype=np.bool_)
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (samples, nodes)")
    num_samples, num_nodes = inputs.shape
    num_words = (num_samples + WORD_BITS - 1) // WORD_BITS
    padded = np.zeros((num_nodes, num_words * WORD_BITS), dtype=np.bool_)
    padded[:, :num_samples] = inputs.T
    packed_bytes = np.packbits(padded, axis=1, bitorder="little")
    packed = packed_bytes.view("<u8").astype(np.uint64, copy=False)
    valid_last_word_bits = num_samples - (num_words - 1) * WORD_BITS
    return packed, valid_last_word_bits


def _unpack_words(words: np.ndarray, num_samples: int) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint64)
    byte_view = np.ascontiguousarray(words).view(np.uint8)
    return np.unpackbits(byte_view, axis=-1, bitorder="little")[..., :num_samples]


def _apply_gate(
    op: GateOp,
    a: np.ndarray,
    b: np.ndarray,
    last_word_mask: np.uint64,
) -> np.ndarray:
    if op == GateOp.CONST_FALSE:
        result = np.zeros_like(a)
    elif op == GateOp.CONST_TRUE:
        result = np.full_like(a, np.iinfo(np.uint64).max)
    elif op == GateOp.WIRE:
        result = a.copy()
    elif op in (GateOp.NOT, GateOp.NOT_A):
        result = ~a
    elif op == GateOp.NOT_B:
        result = ~b
    elif op == GateOp.AND:
        result = a & b
    elif op == GateOp.OR:
        result = a | b
    elif op == GateOp.XOR:
        result = a ^ b
    elif op == GateOp.NAND:
        result = ~(a & b)
    elif op == GateOp.NOR:
        result = ~(a | b)
    elif op == GateOp.XNOR:
        result = ~(a ^ b)
    elif op == GateOp.AND_NOT_B:
        result = a & ~b
    elif op == GateOp.AND_NOT_A:
        result = ~a & b
    elif op == GateOp.OR_NOT_B:
        result = a | ~b
    elif op == GateOp.OR_NOT_A:
        result = ~a | b
    else:
        raise ValueError(f"unsupported gate operation: {op}")
    result[-1] &= last_word_mask
    return result


def _integers_to_words(values: list[int], num_words: int) -> np.ndarray:
    result = np.zeros((len(values), num_words), dtype=np.uint64)
    mask = (1 << WORD_BITS) - 1
    for row, value in enumerate(values):
        for word in range(num_words):
            result[row, word] = np.uint64((value >> (word * WORD_BITS)) & mask)
    return result


def _word_indices(words: np.ndarray, limit: int) -> list[int]:
    indices = []
    for word_index, word in enumerate(np.asarray(words, dtype=np.uint64)):
        value = int(word)
        while value:
            least_bit = value & -value
            index = word_index * WORD_BITS + least_bit.bit_length() - 1
            if index < limit:
                indices.append(index)
            value ^= least_bit
    return indices


@dataclass
class PackedCalibrationTrace:
    num_samples: int
    valid_last_word_bits: int
    n_inputs: int
    node_ids: np.ndarray
    values: np.ndarray
    gate_ids: np.ndarray
    gate_ops: np.ndarray
    gate_in0: np.ndarray
    gate_in1: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray
    runner_ups: np.ndarray
    winner_margins: np.ndarray
    pairwise_margins: np.ndarray
    correct: np.ndarray
    fanout_indptr: np.ndarray
    fanout_indices: np.ndarray
    direct_output_bits: np.ndarray
    cone_gate_bits: np.ndarray
    cone_output_bits: np.ndarray
    metadata: dict

    @property
    def num_gates(self) -> int:
        return len(self.gate_ids)

    @property
    def num_outputs(self) -> int:
        return self.scores.shape[1]

    @property
    def row_by_id(self) -> dict[int, int]:
        return {
            int(node_id): row for row, node_id in enumerate(self.node_ids.tolist())
        }

    def unpack_node(self, node_id: int) -> np.ndarray:
        row = self.row_by_id[node_id]
        return _unpack_words(self.values[row], self.num_samples).astype(np.bool_)

    def direct_fanout_rows(self, node_id: int) -> list[int]:
        row = self.row_by_id[node_id]
        start, end = self.fanout_indptr[row:row + 2]
        return self.fanout_indices[start:end].astype(int).tolist()

    def affected_gate_indices(self, node_id: int) -> list[int]:
        row = self.row_by_id[node_id]
        return _word_indices(self.cone_gate_bits[row], self.num_gates)

    def affected_gate_ids(self, node_id: int) -> list[int]:
        return self.gate_ids[self.affected_gate_indices(node_id)].astype(int).tolist()

    def affected_output_indices(self, node_id: int) -> list[int]:
        row = self.row_by_id[node_id]
        return _word_indices(self.cone_output_bits[row], self.num_outputs)

    def save(self, directory: Path, extra_metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "node_ids": self.node_ids,
            "values": self.values,
            "gate_ids": self.gate_ids,
            "gate_ops": self.gate_ops,
            "gate_in0": self.gate_in0,
            "gate_in1": self.gate_in1,
            "labels": self.labels,
            "scores": self.scores,
            "predictions": self.predictions,
            "runner_ups": self.runner_ups,
            "winner_margins": self.winner_margins,
            "pairwise_margins": self.pairwise_margins,
            "correct": self.correct,
            "fanout_indptr": self.fanout_indptr,
            "fanout_indices": self.fanout_indices,
            "direct_output_bits": self.direct_output_bits,
            "cone_gate_bits": self.cone_gate_bits,
            "cone_output_bits": self.cone_output_bits,
        }
        for name, array in arrays.items():
            np.save(directory / f"{name}.npy", array, allow_pickle=False)

        metadata = dict(self.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata.update(
            {
                "format_version": 1,
                "word_bits": WORD_BITS,
                "num_samples": self.num_samples,
                "valid_last_word_bits": self.valid_last_word_bits,
                "n_inputs": self.n_inputs,
                "num_nodes": len(self.node_ids),
                "num_gates": self.num_gates,
                "num_outputs": self.num_outputs,
                "arrays": {
                    name: {
                        "file": f"{name}.npy",
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "bytes": array.nbytes,
                    }
                    for name, array in arrays.items()
                },
            }
        )
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        mmap_mode: str | None = None,
    ) -> PackedCalibrationTrace:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        arrays = {
            name: np.load(
                directory / record["file"],
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )
            for name, record in metadata["arrays"].items()
        }
        return cls(
            num_samples=metadata["num_samples"],
            valid_last_word_bits=metadata["valid_last_word_bits"],
            n_inputs=metadata["n_inputs"],
            metadata=metadata,
            **arrays,
        )


def build_trace(
    circuit: Circuit,
    encoded_inputs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> PackedCalibrationTrace:
    """Build exact packed values, scores, margins, fan-out, and cone indices."""
    if isinstance(encoded_inputs, torch.Tensor):
        encoded_inputs = encoded_inputs.detach().cpu().numpy()
    encoded_inputs = np.asarray(encoded_inputs, dtype=np.bool_)
    encoded_inputs = encoded_inputs.reshape(len(encoded_inputs), -1)
    if encoded_inputs.shape[1] != circuit.n_inputs:
        raise ValueError(
            f"encoded inputs have {encoded_inputs.shape[1]} features; "
            f"circuit requires {circuit.n_inputs}"
        )
    if len(encoded_inputs) == 0:
        raise ValueError("calibration trace requires at least one example")

    labels = np.asarray(
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.int64,
    )
    if labels.shape != (len(encoded_inputs),):
        raise ValueError("labels must have shape (samples,)")

    packed_inputs, valid_last_word_bits = _pack_examples(encoded_inputs)
    num_words = packed_inputs.shape[1]
    last_word_mask = np.uint64(
        (1 << valid_last_word_bits) - 1
        if valid_last_word_bits < WORD_BITS
        else np.iinfo(np.uint64).max
    )

    gate_ids = np.asarray([gate.gate_id for gate in circuit.gates], dtype=np.int64)
    node_ids = np.concatenate(
        [np.arange(circuit.n_inputs, dtype=np.int64), gate_ids]
    )
    if len(set(node_ids.tolist())) != len(node_ids):
        raise ValueError("Circuit node IDs must be unique")
    row_by_id = {
        int(node_id): row for row, node_id in enumerate(node_ids.tolist())
    }

    values = np.zeros((len(node_ids), num_words), dtype=np.uint64)
    values[:circuit.n_inputs] = packed_inputs
    zero = np.zeros(num_words, dtype=np.uint64)
    gate_in0 = np.asarray([gate.in0 for gate in circuit.gates], dtype=np.int64)
    gate_in1 = np.asarray([gate.in1 for gate in circuit.gates], dtype=np.int64)
    gate_ops = np.asarray([gate.op.name for gate in circuit.gates], dtype="<U16")
    for gate_index, gate in enumerate(circuit.gates):
        row = circuit.n_inputs + gate_index
        in0_row = row_by_id.get(gate.in0)
        in1_row = row_by_id.get(gate.in1)
        if in0_row is not None and in0_row >= row:
            raise ValueError("Circuit gates must be in topological order")
        if in1_row is not None and in1_row >= row:
            raise ValueError("Circuit gates must be in topological order")
        a = values[in0_row] if in0_row is not None else zero
        b = values[in1_row] if in1_row is not None else zero
        values[row] = _apply_gate(gate.op, a, b, last_word_mask)

    num_outputs = len(circuit.outputs)
    scores = np.zeros((len(encoded_inputs), num_outputs), dtype=np.float32)
    sum_by_id = circuit._sum_by_id
    direct_output_ints = [0] * len(node_ids)
    for output_index, output_id in enumerate(circuit.outputs):
        reduction = sum_by_id.get(output_id)
        if reduction is None:
            output_values = _unpack_words(
                values[row_by_id[output_id]], len(encoded_inputs)
            )
            scores[:, output_index] = output_values.astype(np.float32)
            direct_output_ints[row_by_id[output_id]] |= 1 << output_index
            continue
        if reduction.input_ids:
            reduction_rows = [row_by_id[node_id] for node_id in reduction.input_ids]
            reduction_values = _unpack_words(
                values[reduction_rows], len(encoded_inputs)
            )
            summed = reduction_values.sum(axis=0, dtype=np.int64)
        else:
            summed = np.zeros(len(encoded_inputs), dtype=np.int64)
        scores[:, output_index] = (
            (summed + reduction.beta) / reduction.tau
        ).astype(np.float32)
        for node_id in reduction.input_ids:
            direct_output_ints[row_by_id[node_id]] |= 1 << output_index

    predictions = scores.argmax(axis=1).astype(np.int64)
    if num_outputs > 1:
        challenger_scores = scores.copy()
        challenger_scores[np.arange(len(scores)), predictions] = -np.inf
        runner_ups = challenger_scores.argmax(axis=1).astype(np.int64)
        winner_margins = (
            scores[np.arange(len(scores)), predictions]
            - scores[np.arange(len(scores)), runner_ups]
        ).astype(np.float32)
    else:
        runner_ups = np.full(len(scores), -1, dtype=np.int64)
        winner_margins = np.full(len(scores), np.inf, dtype=np.float32)
    pairwise_margins = (
        scores[np.arange(len(scores)), predictions, None] - scores
    ).astype(np.float32)
    pairwise_margins[np.arange(len(scores)), predictions] = np.inf
    correct = predictions == labels

    fanouts = [[] for _ in node_ids]
    for gate_index, gate in enumerate(circuit.gates):
        consumer_row = circuit.n_inputs + gate_index
        for input_id in sorted({gate.in0, gate.in1}):
            if input_id in row_by_id:
                fanouts[row_by_id[input_id]].append(consumer_row)
    fanout_indptr = np.zeros(len(node_ids) + 1, dtype=np.int64)
    fanout_indices_list = []
    for row, consumers in enumerate(fanouts):
        fanout_indices_list.extend(consumers)
        fanout_indptr[row + 1] = len(fanout_indices_list)
    fanout_indices = np.asarray(fanout_indices_list, dtype=np.int64)

    cone_gate_ints = [0] * len(node_ids)
    cone_output_ints = direct_output_ints.copy()
    for row in range(len(node_ids) - 1, -1, -1):
        for consumer_row in fanouts[row]:
            consumer_gate_index = consumer_row - circuit.n_inputs
            cone_gate_ints[row] |= 1 << consumer_gate_index
            cone_gate_ints[row] |= cone_gate_ints[consumer_row]
            cone_output_ints[row] |= cone_output_ints[consumer_row]

    direct_output_bits = _integers_to_words(
        direct_output_ints, (num_outputs + WORD_BITS - 1) // WORD_BITS
    )
    cone_gate_bits = _integers_to_words(
        cone_gate_ints, (len(circuit.gates) + WORD_BITS - 1) // WORD_BITS
    )
    cone_output_bits = _integers_to_words(
        cone_output_ints, (num_outputs + WORD_BITS - 1) // WORD_BITS
    )

    class_counts = np.bincount(labels, minlength=num_outputs)
    class_correct = np.bincount(labels, weights=correct, minlength=num_outputs)
    per_class_accuracy = np.divide(
        class_correct,
        class_counts,
        out=np.zeros(num_outputs, dtype=np.float64),
        where=class_counts != 0,
    )
    metadata = {
        "tie_breaking": "numpy/torch argmax: lowest class index wins ties",
        "class_counts": class_counts.astype(int).tolist(),
        "baseline_accuracy": float(correct.mean()),
        "per_class_accuracy": per_class_accuracy.tolist(),
    }
    return PackedCalibrationTrace(
        num_samples=len(encoded_inputs),
        valid_last_word_bits=valid_last_word_bits,
        n_inputs=circuit.n_inputs,
        node_ids=node_ids,
        values=values,
        gate_ids=gate_ids,
        gate_ops=gate_ops,
        gate_in0=gate_in0,
        gate_in1=gate_in1,
        labels=labels,
        scores=scores,
        predictions=predictions,
        runner_ups=runner_ups,
        winner_margins=winner_margins,
        pairwise_margins=pairwise_margins,
        correct=correct,
        fanout_indptr=fanout_indptr,
        fanout_indices=fanout_indices,
        direct_output_bits=direct_output_bits,
        cone_gate_bits=cone_gate_bits,
        cone_output_bits=cone_output_bits,
        metadata=metadata,
    )
