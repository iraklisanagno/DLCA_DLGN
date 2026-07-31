"""Safe, deterministic gate rewrites for MarginSynth."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

from torchlogix import Circuit
from torchlogix.circuit import Gate, GateOp


class RewriteKind(str, Enum):
    CONSTANT_0 = "constant-0"
    CONSTANT_1 = "constant-1"
    COPY_A = "copy-a"
    COPY_B = "copy-b"
    NOT_A = "not-a"
    NOT_B = "not-b"
    ALTERNATIVE_GATE = "alternative-gate"


BINARY_GATE_OPS = (
    GateOp.AND,
    GateOp.OR,
    GateOp.XOR,
    GateOp.NAND,
    GateOp.NOR,
    GateOp.XNOR,
    GateOp.AND_NOT_A,
    GateOp.AND_NOT_B,
    GateOp.OR_NOT_A,
    GateOp.OR_NOT_B,
)


def _gate_index(circuit: Circuit, gate_id: int) -> int:
    matches = [
        index for index, gate in enumerate(circuit.gates)
        if gate.gate_id == gate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"rewrite target {gate_id} must identify exactly one gate; "
            f"found {len(matches)}"
        )
    return matches[0]


def _gate_dict(gate: Gate) -> dict:
    return {
        "gate_id": gate.gate_id,
        "op": gate.op.name,
        "in0": gate.in0,
        "in1": gate.in1,
        "node_idx": gate.node_idx,
    }


def _gate_from_dict(data: dict) -> Gate:
    return Gate(
        gate_id=int(data["gate_id"]),
        op=GateOp[data["op"]],
        in0=int(data["in0"]),
        in1=int(data["in1"]),
        node_idx=int(data.get("node_idx", -1)),
    )


@dataclass(frozen=True)
class UndoRecord:
    target_id: int
    original_gate: Gate
    applied_gate: Gate

    def undo(self, circuit: Circuit) -> None:
        index = _gate_index(circuit, self.target_id)
        if circuit.gates[index] != self.applied_gate:
            raise RuntimeError(
                "cannot undo rewrite because the target gate changed after apply"
            )
        circuit.gates[index] = copy.deepcopy(self.original_gate)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "original_gate": _gate_dict(self.original_gate),
            "applied_gate": _gate_dict(self.applied_gate),
        }


@dataclass(frozen=True)
class GateRewrite:
    target_id: int
    kind: RewriteKind
    replacement_op: GateOp | None = None

    def replacement_gate(self, original: Gate) -> Gate:
        if original.gate_id != self.target_id:
            raise ValueError("rewrite target does not match the supplied gate")
        if self.kind == RewriteKind.CONSTANT_0:
            op, in0, in1 = GateOp.CONST_FALSE, -1, -1
        elif self.kind == RewriteKind.CONSTANT_1:
            op, in0, in1 = GateOp.CONST_TRUE, -1, -1
        elif self.kind == RewriteKind.COPY_A:
            if original.in0 < 0:
                raise ValueError("copy-a requires a valid A input")
            op, in0, in1 = GateOp.WIRE, original.in0, -1
        elif self.kind == RewriteKind.COPY_B:
            if original.in1 < 0:
                raise ValueError("copy-b requires a valid B input")
            op, in0, in1 = GateOp.WIRE, original.in1, -1
        elif self.kind == RewriteKind.NOT_A:
            if original.in0 < 0:
                raise ValueError("not-a requires a valid A input")
            op, in0, in1 = GateOp.NOT, original.in0, -1
        elif self.kind == RewriteKind.NOT_B:
            if original.in1 < 0:
                raise ValueError("not-b requires a valid B input")
            op, in0, in1 = GateOp.NOT, original.in1, -1
        elif self.kind == RewriteKind.ALTERNATIVE_GATE:
            if self.replacement_op not in BINARY_GATE_OPS:
                raise ValueError(
                    "alternative-gate requires a supported binary replacement_op"
                )
            if original.in0 < 0 or original.in1 < 0:
                raise ValueError(
                    "alternative-gate requires two valid original inputs"
                )
            op, in0, in1 = self.replacement_op, original.in0, original.in1
        else:
            raise ValueError(f"unsupported rewrite kind: {self.kind}")
        return Gate(
            gate_id=original.gate_id,
            op=op,
            in0=in0,
            in1=in1,
            node_idx=original.node_idx,
        )

    def apply(self, circuit: Circuit) -> UndoRecord:
        index = _gate_index(circuit, self.target_id)
        original = copy.deepcopy(circuit.gates[index])
        replacement = self.replacement_gate(original)
        circuit.gates[index] = replacement
        return UndoRecord(
            target_id=self.target_id,
            original_gate=original,
            applied_gate=copy.deepcopy(replacement),
        )

    def apply_to_copy(self, circuit: Circuit) -> Circuit:
        rewritten = copy.deepcopy(circuit)
        self.apply(rewritten)
        return rewritten

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "kind": self.kind.value,
            "replacement_op": (
                self.replacement_op.name
                if self.replacement_op is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GateRewrite:
        replacement = data.get("replacement_op")
        return cls(
            target_id=int(data["target_id"]),
            kind=RewriteKind(data["kind"]),
            replacement_op=GateOp[replacement] if replacement else None,
        )


@dataclass(frozen=True)
class RewriteGroup:
    """A small coordinated rewrite proposal applied atomically."""

    rewrites: tuple[GateRewrite, ...]
    group_kind: str = "coordinated-cone"

    def __post_init__(self):
        if len(self.rewrites) < 2:
            raise ValueError("a coordinated rewrite requires at least two members")
        targets = [rewrite.target_id for rewrite in self.rewrites]
        if len(targets) != len(set(targets)):
            raise ValueError("coordinated rewrite targets must be unique")

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(rewrite.target_id for rewrite in self.rewrites)

    def apply(self, circuit: Circuit) -> list[UndoRecord]:
        records = []
        try:
            for rewrite in self.rewrites:
                records.append(rewrite.apply(circuit))
        except Exception:
            for record in reversed(records):
                record.undo(circuit)
            raise
        return records

    def apply_to_copy(self, circuit: Circuit) -> Circuit:
        rewritten = copy.deepcopy(circuit)
        self.apply(rewritten)
        return rewritten

    def to_dict(self) -> dict:
        return {
            "proposal_type": "group",
            "group_kind": self.group_kind,
            "rewrites": [rewrite.to_dict() for rewrite in self.rewrites],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RewriteGroup:
        return cls(
            rewrites=tuple(
                GateRewrite.from_dict(record) for record in data["rewrites"]
            ),
            group_kind=str(data.get("group_kind", "coordinated-cone")),
        )


RewriteProposal = GateRewrite | RewriteGroup


def proposal_from_dict(data: dict) -> RewriteProposal:
    if data.get("proposal_type") == "group":
        return RewriteGroup.from_dict(data)
    return GateRewrite.from_dict(data)


def proposal_target_ids(proposal: RewriteProposal) -> tuple[int, ...]:
    if isinstance(proposal, RewriteGroup):
        return proposal.target_ids
    return (proposal.target_id,)


def replay_rewrites(circuit: Circuit, rewrites: list[GateRewrite]) -> Circuit:
    replayed = copy.deepcopy(circuit)
    for rewrite in rewrites:
        rewrite.apply(replayed)
    return replayed


def replay_proposals(
    circuit: Circuit,
    proposals: list[RewriteProposal],
) -> Circuit:
    replayed = copy.deepcopy(circuit)
    for proposal in proposals:
        proposal.apply(replayed)
        replayed.simplify()
    return replayed


def generate_gate_rewrites(circuit: Circuit) -> list[GateRewrite]:
    """Generate a deterministic, duplicate-free rewrite set."""
    candidates = []
    for gate in circuit.gates:
        proposed = [
            GateRewrite(gate.gate_id, RewriteKind.CONSTANT_0),
            GateRewrite(gate.gate_id, RewriteKind.CONSTANT_1),
        ]
        if gate.in0 >= 0:
            proposed.extend(
                [
                    GateRewrite(gate.gate_id, RewriteKind.COPY_A),
                    GateRewrite(gate.gate_id, RewriteKind.NOT_A),
                ]
            )
        if gate.in1 >= 0:
            proposed.extend(
                [
                    GateRewrite(gate.gate_id, RewriteKind.COPY_B),
                    GateRewrite(gate.gate_id, RewriteKind.NOT_B),
                ]
            )
        if gate.in0 >= 0 and gate.in1 >= 0:
            proposed.extend(
                GateRewrite(
                    gate.gate_id,
                    RewriteKind.ALTERNATIVE_GATE,
                    replacement_op=op,
                )
                for op in BINARY_GATE_OPS
            )

        seen_replacements = set()
        original_key = (gate.op, gate.in0, gate.in1)
        for rewrite in proposed:
            replacement = rewrite.replacement_gate(gate)
            replacement_key = (
                replacement.op,
                replacement.in0,
                replacement.in1,
            )
            if (
                replacement_key == original_key
                or replacement_key in seen_replacements
            ):
                continue
            seen_replacements.add(replacement_key)
            candidates.append(rewrite)
    return candidates


def generate_targeted_rewrites(
    circuit: Circuit,
    operation_cost: dict[GateOp, float] | None = None,
) -> list[GateRewrite]:
    """Generate candidates likely to reduce structure or mapped operation cost.

    Constants, wires, and inversions are retained. Alternative binary gates are
    retained only when their operation cost is lower than the original gate,
    avoiding a large pool of structurally neutral substitutions.
    """
    if operation_cost is None:
        operation_cost = {
            GateOp.CONST_FALSE: 0.0,
            GateOp.CONST_TRUE: 0.0,
            GateOp.WIRE: 0.0,
            GateOp.NOT: 0.0,
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
    candidates = []
    for gate in circuit.gates:
        proposed = [
            GateRewrite(gate.gate_id, RewriteKind.CONSTANT_0),
            GateRewrite(gate.gate_id, RewriteKind.CONSTANT_1),
        ]
        if gate.in0 >= 0:
            proposed.extend(
                (
                    GateRewrite(gate.gate_id, RewriteKind.COPY_A),
                    GateRewrite(gate.gate_id, RewriteKind.NOT_A),
                )
            )
        if gate.in1 >= 0:
            proposed.extend(
                (
                    GateRewrite(gate.gate_id, RewriteKind.COPY_B),
                    GateRewrite(gate.gate_id, RewriteKind.NOT_B),
                )
            )
        if gate.in0 >= 0 and gate.in1 >= 0:
            proposed.extend(
                GateRewrite(
                    gate.gate_id,
                    RewriteKind.ALTERNATIVE_GATE,
                    replacement_op=operation,
                )
                for operation in BINARY_GATE_OPS
                if operation_cost.get(operation, 1.0)
                < operation_cost.get(gate.op, 1.0)
            )
        original_key = (gate.op, gate.in0, gate.in1)
        seen = set()
        for rewrite in proposed:
            replacement = rewrite.replacement_gate(gate)
            key = (replacement.op, replacement.in0, replacement.in1)
            if key == original_key or key in seen:
                continue
            seen.add(key)
            candidates.append(rewrite)
    return candidates


def generate_coordinated_cone_rewrites(
    circuit: Circuit,
    maximum_groups: int = 4096,
) -> list[RewriteGroup]:
    """Generate deterministic two-target proposals around shared consumers.

    The two inputs of a binary consumer form a natural local cut. When both
    inputs are live gates, jointly tying them can expose simplification that
    neither individual change reveals.
    """
    gate_by_id = {gate.gate_id: gate for gate in circuit.gates}
    groups = []
    seen = set()
    for consumer in circuit.gates:
        if consumer.in0 not in gate_by_id or consumer.in1 not in gate_by_id:
            continue
        if consumer.in0 == consumer.in1:
            continue
        targets = tuple(sorted((consumer.in0, consumer.in1)))
        for first_value in (0, 1):
            for second_value in (0, 1):
                members = tuple(
                    GateRewrite(
                        target,
                        (
                            RewriteKind.CONSTANT_0
                            if value == 0
                            else RewriteKind.CONSTANT_1
                        ),
                    )
                    for target, value in zip(
                        targets,
                        (first_value, second_value),
                    )
                )
                group = RewriteGroup(members)
                key = tuple(
                    (
                        rewrite.target_id,
                        rewrite.kind.value,
                    )
                    for rewrite in group.rewrites
                )
                if key in seen:
                    continue
                seen.add(key)
                groups.append(group)
                if maximum_groups > 0 and len(groups) >= maximum_groups:
                    return groups
    return groups
