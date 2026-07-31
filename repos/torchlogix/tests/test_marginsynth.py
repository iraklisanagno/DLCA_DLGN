import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.marginsynth.trace import PackedCalibrationTrace, build_trace
from experiments.marginsynth.incremental import evaluate_proposal, evaluate_rewrite
from experiments.marginsynth.cost_model import (
    SynthCostEstimator,
    circuit_features,
)
from experiments.marginsynth.verify_checkpoint import encoded_sample_shape
from experiments.marginsynth.verify_synthesis import integer_score_predictions
from experiments.marginsynth.freeze_protocol import (
    find_budget_point,
    unit_tying_circuit_path,
)
from experiments.marginsynth.summarize_paired_study import (
    exact_bootstrap_mean_ci,
    summarize,
)
from experiments.marginsynth.margin_aware_tying import (
    balanced_shortlist_order,
    compose_candidate_table,
    interleave_layers,
    normalized_rank,
    projected_risk_statistics,
    stratified_fold_ids,
    structural_tie_benefits,
    within_constraints,
    quality_tuple,
)
from experiments.marginsynth.unit_tying import (
    apply_permanent_ties,
    binary_split_identify,
    refine_overshoot,
    selection_indices,
)
from experiments.marginsynth.search import (
    circuit_sha256,
    run_search,
)
from experiments.marginsynth.rewrites import (
    GateRewrite,
    RewriteGroup,
    RewriteKind,
    generate_coordinated_cone_rewrites,
    generate_gate_rewrites,
    generate_targeted_rewrites,
    proposal_from_dict,
    replay_rewrites,
)
from experiments.marginsynth.search_v2 import (
    behavior_metrics,
    category_counts,
    stratified_ranked_pool,
    within_budgets,
)
from torchlogix.circuit import Circuit, Gate, GateOp, SumReduction
from torchlogix.layers import LogicDense


def test_unit_tying_binary_split_isolates_largest_distortion():
    costs = {0: 0.2, 1: 1.0, 2: 8.0, 3: 0.5, 4: 0.1}

    def evaluate_pair(first, second):
        return sum(costs[unit] for unit in first), sum(
            costs[unit] for unit in second
        )

    harmful, path = binary_split_identify(list(costs), evaluate_pair)
    assert harmful == 2
    assert len(path) == 2


def test_unit_tying_refinement_removes_overshoot_harm():
    costs = {0: 0.2, 1: 7.0, 2: 0.3, 3: 5.0, 4: 0.1}

    def evaluate_pair(first, second):
        return sum(costs[unit] for unit in first), sum(
            costs[unit] for unit in second
        )

    selected, removals = refine_overshoot(list(costs), 3, evaluate_pair)
    assert set(selected) == {0, 2, 4}
    assert [record["removed_unit"] for record in removals] == [1, 3]


def test_apply_permanent_ties_selects_constant_luts():
    layer = LogicDense(
        in_dim=4,
        out_dim=3,
        lut_rank=2,
        parametrization="raw",
        connections="fixed",
        connections_kwargs={"init_method": "random"},
    )
    unchanged_middle = layer.weight[1].detach().clone()
    apply_permanent_ties(layer, [0, 2], [0, 1], tie_logit=1000.0)
    assert layer.weight.argmax(dim=1).tolist() == [0, int(unchanged_middle.argmax()), 15]
    assert torch.equal(layer.weight[1], unchanged_middle)


def test_unit_tying_sample_selection_is_deterministic_and_unique():
    first = selection_indices(100, 16, 4)
    second = selection_indices(100, 16, 4)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 16


def test_margin_aware_folds_are_deterministic_and_class_balanced():
    labels = np.repeat(np.arange(3), 11)
    first = stratified_fold_ids(labels, folds=4, seed=7)
    second = stratified_fold_ids(labels, folds=4, seed=7)
    assert np.array_equal(first, second)
    for label in np.unique(labels):
        counts = np.bincount(first[labels == label], minlength=4)
        assert counts.max() - counts.min() <= 1


def test_projected_margin_risk_exposes_fold_and_class_failures():
    projected = np.ones((8, 2, 2), dtype=np.float64)
    projected[:4, 0, 0] = -1.0
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    folds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = projected_risk_statistics(projected, labels, folds, 0.25)
    assert result["projected_flip_rate"][0, 0] == 0.5
    assert result["maximum_class_hinge"][0, 0] > result["mean_hinge"][0, 0]
    assert result["projected_flip_rate"][1, 1] == 0.0


def test_margin_aware_candidate_selection_prefers_safe_high_benefit_tie():
    gauss = np.array([[0.1, 0.2], [0.1, 0.1]])
    zeros = np.zeros((2, 2), dtype=np.float64)
    stats = {
        "mean_hinge": zeros.copy(),
        "maximum_fold_hinge": zeros.copy(),
        "fold_standard_deviation": zeros.copy(),
        "maximum_class_hinge": zeros.copy(),
        "projected_flip_rate": zeros.copy(),
    }
    stats["projected_flip_rate"][0, 1] = 0.5
    benefits = np.array([[1.0, 8.0], [2.0, 3.0]])
    weights = {
        "mean_margin": 1.0,
        "fold_worst": 1.0,
        "fold_std": 1.0,
        "class_worst": 1.0,
        "projected_flip": 10.0,
        "gauss_newton": 0.1,
        "risk_epsilon": 0.01,
    }
    records = compose_candidate_table(1, gauss, stats, benefits, weights)
    assert records[0]["constant"] == 0
    assert records[1]["constant"] == 1


def test_margin_aware_constraint_gate_checks_every_budget():
    metrics = {
        "accuracy_loss": 0.01,
        "decision_flip_rate": 0.02,
        "maximum_per_class_accuracy_loss": 0.03,
        "maximum_per_class_disagreement": 0.04,
    }
    budgets = {
        "accuracy_loss": 0.01,
        "disagreement": 0.02,
        "per_class_accuracy_loss": 0.03,
        "per_class_disagreement": 0.04,
    }
    assert within_constraints(metrics, budgets)
    for key in budgets:
        tightened = dict(budgets)
        tightened[key] -= 0.001
        assert not within_constraints(metrics, tightened)


def test_structural_tie_benefit_rewards_constant_fanout_cofactor():
    first = LogicDense(
        in_dim=2,
        out_dim=2,
        lut_rank=2,
        parametrization="raw",
        connections="fixed",
        connections_kwargs={"init_method": "random"},
    )
    second = LogicDense(
        in_dim=2,
        out_dim=2,
        lut_rank=2,
        parametrization="raw",
        connections="fixed",
        connections_kwargs={"init_method": "random"},
    )
    with torch.no_grad():
        first.weight.fill_(-100.0)
        first.weight[:, 6] = 100.0
        second.weight.fill_(-100.0)
        # AND has a constant cofactor for input=0 and a unary cofactor for 1.
        second.weight[:, 8] = 100.0
        second.connections.indices[:] = torch.tensor([[0, 1], [1, 0]])
    benefit = structural_tie_benefits([first, second], 0)
    assert benefit[0, 0] > benefit[0, 1]
    assert benefit[1, 0] > benefit[1, 1]


def test_normalized_rank_is_stable_and_directional():
    values = np.array([3.0, 1.0, 2.0])
    assert normalized_rank(values, True).tolist() == [1.0, 0.0, 0.5]
    assert normalized_rank(values, False).tolist() == [0.0, 1.0, 0.5]


def test_balanced_shortlist_preserves_layer_targets_and_gn_screen():
    records = {
        1: [
            {"layer": 1, "unit": unit, "minimum_gauss_newton": float(unit), "utility": float(9 - unit)}
            for unit in range(6)
        ],
        2: [
            {"layer": 2, "unit": unit, "minimum_gauss_newton": float(unit), "utility": float(unit)}
            for unit in range(6)
        ],
    }
    ordered, pools = balanced_shortlist_order(records, {1: 2, 2: 2}, overshoot=1)
    assert pools == {1: 3, 2: 3}
    assert [item["layer"] for item in ordered] == [1, 2, 1, 2]
    assert {item["unit"] for item in ordered if item["layer"] == 2} <= {0, 1, 2}


def test_interleave_and_quality_tuple_are_deterministic():
    candidates = [
        {"layer": 2, "unit": 0},
        {"layer": 1, "unit": 0},
        {"layer": 1, "unit": 1},
    ]
    assert [(x["layer"], x["unit"]) for x in interleave_layers(candidates)] == [
        (1, 0), (2, 0), (1, 1)
    ]
    metrics = {
        "accuracy_loss": 0.1,
        "maximum_per_class_accuracy_loss": 0.2,
        "decision_flip_rate": 0.3,
        "maximum_per_class_disagreement": 0.4,
    }
    assert quality_tuple(metrics) == (0.1, 0.2, 0.3, 0.4)


def test_encoded_sample_shape_supports_multi_threshold_image_inputs():
    encoded = torch.zeros(7, 1, 28, 84, dtype=torch.bool)
    assert encoded_sample_shape(encoded) == [1, 28, 84]

    with pytest.raises(ValueError, match="batch axis"):
        encoded_sample_shape(torch.tensor(True))


def test_freeze_budget_requires_one_unambiguous_pareto_point():
    summary = {
        "pareto": [
            {"accuracy_budget": 0.0, "selected_step": 2},
            {"accuracy_budget": 0.001, "selected_step": 5},
        ]
    }
    assert find_budget_point(summary, 0.001)["selected_step"] == 5
    with pytest.raises(RuntimeError, match="exactly one"):
        find_budget_point(summary, 0.005)


def test_unit_tying_freeze_uses_synthesized_circuit(tmp_path):
    assert unit_tying_circuit_path(
        tmp_path, {"directory": "ratio_10"}
    ) == (
        tmp_path
        / "baselines"
        / "two_stage_unit_tying"
        / "ratio_10"
        / "synthesis"
        / "exact_simplified_circuit.json"
    )


def test_paired_summary_and_exact_bootstrap_are_deterministic():
    metrics = summarize([1.0, 2.0, 3.0])
    assert metrics["mean"] == 2.0
    assert metrics["standard_deviation"] == 1.0
    first = exact_bootstrap_mean_ci([1.0, 2.0, 3.0])
    second = exact_bootstrap_mean_ci([1.0, 2.0, 3.0])
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_stratified_shortlist_prevents_rewrite_family_starvation():
    proposals = [
        RewriteGroup(
            (
                GateRewrite(index, RewriteKind.CONSTANT_0),
                GateRewrite(index + 100, RewriteKind.CONSTANT_1),
            )
        )
        for index in range(8)
    ]
    proposals += [
        GateRewrite(200 + index, RewriteKind.CONSTANT_0)
        for index in range(8)
    ]
    proposals += [
        GateRewrite(300 + index, RewriteKind.COPY_A)
        for index in range(8)
    ]
    proposals += [
        GateRewrite(
            400 + index,
            RewriteKind.ALTERNATIVE_GATE,
            replacement_op=GateOp.AND,
        )
        for index in range(8)
    ]
    selected = stratified_ranked_pool(
        proposals,
        8,
        proposal_of=lambda proposal: proposal,
        key=lambda proposal: proposal.target_ids
        if isinstance(proposal, RewriteGroup)
        else (proposal.target_id,),
    )
    assert category_counts(selected) == {
        "coordinated": 2,
        "constant": 2,
        "routing": 2,
        "alternative": 2,
    }


def test_integer_score_predictions_supports_uint16_class_sums():
    scores = torch.tensor([[256, 511, 7], [800, 799, 800]], dtype=torch.uint16)
    assert integer_score_predictions(scores).tolist() == [1, 0]

    with pytest.raises(TypeError, match="unsigned integer"):
        integer_score_predictions(scores.float())


def small_trace_circuit():
    return Circuit(
        n_inputs=3,
        input_shape=[3],
        gates=[
            Gate(3, GateOp.AND, 0, 1),
            Gate(4, GateOp.NOT, 2),
            Gate(5, GateOp.XOR, 3, 4),
            Gate(6, GateOp.OR, 3, 2),
        ],
        outputs=[7, 8],
        output_shape=[2],
        sum_nodes=[
            SumReduction(7, [5, 3], tau=2.0, beta=0.0),
            SumReduction(8, [6, 4], tau=2.0, beta=0.0),
        ],
    )


def exhaustive_inputs(n_inputs):
    return torch.tensor(
        [
            [(value >> bit) & 1 for bit in range(n_inputs)]
            for value in range(1 << n_inputs)
        ],
        dtype=torch.bool,
    )


def test_coordinated_rewrite_incremental_evaluation_matches_full_trace():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(circuit, inputs, labels)
    group = RewriteGroup(
        (
            GateRewrite(3, RewriteKind.CONSTANT_0),
            GateRewrite(4, RewriteKind.CONSTANT_1),
        )
    )
    evaluation = evaluate_proposal(circuit, trace, group, margin_reserve=0.1)
    rewritten = group.apply_to_copy(circuit)
    complete = build_trace(rewritten, inputs, labels)
    assert np.array_equal(evaluation.scores, complete.scores)
    assert np.array_equal(evaluation.predictions, complete.predictions)
    assert proposal_from_dict(group.to_dict()) == group


def test_coordinated_generator_uses_shared_consumer_inputs():
    groups = generate_coordinated_cone_rewrites(
        small_trace_circuit(),
        maximum_groups=32,
    )
    assert groups
    assert all(len(group.rewrites) == 2 for group in groups)
    assert any(set(group.target_ids) == {3, 4} for group in groups)


def test_targeted_rewrites_keep_only_cheaper_alternative_operations():
    circuit = Circuit(
        n_inputs=2,
        input_shape=[2],
        gates=[Gate(2, GateOp.XOR, 0, 1)],
        outputs=[2],
        output_shape=[1],
    )
    targeted = generate_targeted_rewrites(circuit)
    alternative_ops = {
        rewrite.replacement_op
        for rewrite in targeted
        if rewrite.kind == RewriteKind.ALTERNATIVE_GATE
    }
    assert GateOp.AND in alternative_ops
    assert GateOp.OR in alternative_ops
    assert GateOp.XNOR not in alternative_ops


def test_operation_aware_cost_distinguishes_xor_from_and():
    and_circuit = Circuit(
        n_inputs=2,
        input_shape=[2],
        gates=[Gate(2, GateOp.AND, 0, 1)],
        outputs=[2],
        output_shape=[1],
    )
    xor_circuit = copy.deepcopy(and_circuit)
    xor_circuit.gates[0].op = GateOp.XOR
    estimator = SynthCostEstimator()
    assert circuit_features(xor_circuit)["operation_aig_units"] > circuit_features(
        and_circuit
    )["operation_aig_units"]
    assert estimator.estimate(xor_circuit) > estimator.estimate(and_circuit)


def test_v2_dual_budgets_retain_accuracy_and_disagreement_constraints():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    reference = build_trace(circuit, inputs, labels)
    changed = reference.scores.copy()
    original_prediction = int(reference.predictions[0])
    changed[0, original_prediction] = -100.0
    changed[0, 1 - original_prediction] = 100.0
    metrics = behavior_metrics(changed, reference, margin_reserve=0.1)
    permissive = {
        "maximum_accuracy_loss": 1.0,
        "maximum_per_class_accuracy_loss": 1.0,
        "maximum_disagreement": 1.0,
        "maximum_per_class_disagreement": 1.0,
    }
    assert within_budgets(metrics, permissive)
    strict_disagreement = permissive | {"maximum_disagreement": 0.0}
    assert not within_budgets(metrics, strict_disagreement)


def exhaustive_affected(circuit, start_id):
    fanouts = {}
    output_consumers = {}
    for gate in circuit.gates:
        for node_id in {gate.in0, gate.in1}:
            fanouts.setdefault(node_id, set()).add(gate.gate_id)
    for output_index, reduction in enumerate(circuit.sum_nodes):
        for node_id in reduction.input_ids:
            output_consumers.setdefault(node_id, set()).add(output_index)

    gates = set()
    outputs = set(output_consumers.get(start_id, set()))
    queue = list(fanouts.get(start_id, set()))
    while queue:
        gate_id = queue.pop()
        if gate_id in gates:
            continue
        gates.add(gate_id)
        outputs.update(output_consumers.get(gate_id, set()))
        queue.extend(fanouts.get(gate_id, set()))
    return sorted(gates), sorted(outputs)


def test_packed_trace_matches_full_simulation_and_graph_reachability():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    trace = build_trace(circuit, inputs, labels)

    expected_gate_values = {
        3: inputs[:, 0] & inputs[:, 1],
        4: ~inputs[:, 2],
    }
    expected_gate_values[5] = expected_gate_values[3] ^ expected_gate_values[4]
    expected_gate_values[6] = expected_gate_values[3] | inputs[:, 2]
    for gate_id, expected in expected_gate_values.items():
        assert np.array_equal(trace.unpack_node(gate_id), expected.numpy())

    expected_scores = circuit(inputs)
    assert np.array_equal(trace.scores, expected_scores.numpy())
    assert np.array_equal(
        trace.predictions,
        expected_scores.argmax(dim=-1).numpy(),
    )
    assert trace.valid_last_word_bits == 8
    assert int(trace.values[:, -1].max()) <= 0xFF

    for node_id in trace.node_ids.tolist():
        expected_gates, expected_outputs = exhaustive_affected(circuit, node_id)
        assert trace.affected_gate_ids(node_id) == expected_gates
        assert trace.affected_output_indices(node_id) == expected_outputs


def test_trace_tie_breaking_margins_and_direct_fanout():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.zeros(len(inputs), dtype=torch.int64)
    trace = build_trace(circuit, inputs, labels)

    tie_rows = np.flatnonzero(trace.scores[:, 0] == trace.scores[:, 1])
    assert len(tie_rows) > 0
    assert np.all(trace.predictions[tie_rows] == 0)
    assert np.all(trace.winner_margins[tie_rows] == 0.0)
    assert np.all(np.isinf(trace.pairwise_margins[tie_rows, 0]))

    row_by_id = trace.row_by_id
    assert trace.direct_fanout_rows(0) == [row_by_id[3]]
    assert trace.direct_fanout_rows(3) == [row_by_id[5], row_by_id[6]]


def test_trace_save_load_roundtrip_with_memory_mapping():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(circuit, inputs, labels)

    with tempfile.TemporaryDirectory() as directory:
        trace.save(Path(directory), extra_metadata={"fixture": "tiny"})
        loaded = PackedCalibrationTrace.load(Path(directory), mmap_mode="r")

        assert loaded.metadata["fixture"] == "tiny"
        assert loaded.num_samples == trace.num_samples
        assert loaded.valid_last_word_bits == trace.valid_last_word_bits
        for field in (
            "node_ids",
            "values",
            "gate_ids",
            "gate_ops",
            "gate_in0",
            "gate_in1",
            "labels",
            "scores",
            "predictions",
            "runner_ups",
            "winner_margins",
            "pairwise_margins",
            "correct",
            "fanout_indptr",
            "fanout_indices",
            "direct_output_bits",
            "cone_gate_bits",
            "cone_output_bits",
        ):
            assert np.array_equal(getattr(loaded, field), getattr(trace, field))


@pytest.mark.parametrize(
    "rewrite,expected",
    [
        (
            GateRewrite(2, RewriteKind.CONSTANT_0),
            [False, False, False, False],
        ),
        (
            GateRewrite(2, RewriteKind.CONSTANT_1),
            [True, True, True, True],
        ),
        (
            GateRewrite(2, RewriteKind.COPY_A),
            [False, True, False, True],
        ),
        (
            GateRewrite(2, RewriteKind.COPY_B),
            [False, False, True, True],
        ),
        (
            GateRewrite(2, RewriteKind.NOT_A),
            [True, False, True, False],
        ),
        (
            GateRewrite(2, RewriteKind.NOT_B),
            [True, True, False, False],
        ),
        (
            GateRewrite(
                2,
                RewriteKind.ALTERNATIVE_GATE,
                replacement_op=GateOp.XOR,
            ),
            [False, True, True, False],
        ),
    ],
)
def test_rewrite_truth_tables_apply_and_undo(rewrite, expected):
    circuit = Circuit(
        n_inputs=2,
        input_shape=[2],
        gates=[Gate(2, GateOp.AND, 0, 1)],
        outputs=[2],
        output_shape=[1],
    )
    inputs = exhaustive_inputs(2)
    original_dict = circuit.to_dict()

    undo = rewrite.apply(circuit)
    assert circuit(inputs).flatten().tolist() == expected
    undo.undo(circuit)
    assert circuit.to_dict() == original_dict


def test_rewrite_copy_json_replay_and_candidate_uniqueness():
    circuit = Circuit(
        n_inputs=2,
        input_shape=[2],
        gates=[
            Gate(2, GateOp.AND, 0, 1),
            Gate(3, GateOp.OR, 2, 1),
        ],
        outputs=[3],
        output_shape=[1],
    )
    original_dict = circuit.to_dict()
    rewrites = [
        GateRewrite(
            2,
            RewriteKind.ALTERNATIVE_GATE,
            replacement_op=GateOp.XOR,
        ),
        GateRewrite(3, RewriteKind.COPY_A),
    ]
    deserialized = [
        GateRewrite.from_dict(rewrite.to_dict()) for rewrite in rewrites
    ]
    replayed = replay_rewrites(circuit, deserialized)
    independently_rewritten = rewrites[1].apply_to_copy(
        rewrites[0].apply_to_copy(circuit)
    )

    assert circuit.to_dict() == original_dict
    assert replayed.to_dict() == independently_rewritten.to_dict()
    assert torch.equal(
        replayed(exhaustive_inputs(2)),
        independently_rewritten(exhaustive_inputs(2)),
    )

    candidates = generate_gate_rewrites(circuit)
    replacement_keys = []
    for rewrite in candidates:
        gate = next(g for g in circuit.gates if g.gate_id == rewrite.target_id)
        replacement = rewrite.replacement_gate(gate)
        replacement_keys.append(
            (
                rewrite.target_id,
                replacement.op,
                replacement.in0,
                replacement.in1,
            )
        )
    assert len(replacement_keys) == len(set(replacement_keys))


def test_invalid_or_stale_rewrite_fails_closed():
    circuit = Circuit(
        n_inputs=1,
        input_shape=[1],
        gates=[Gate(1, GateOp.NOT, 0)],
        outputs=[1],
        output_shape=[1],
    )
    invalid = GateRewrite(
        1,
        RewriteKind.ALTERNATIVE_GATE,
        replacement_op=GateOp.XOR,
    )
    with pytest.raises(ValueError, match="two valid"):
        invalid.apply(circuit)

    rewrite = GateRewrite(1, RewriteKind.CONSTANT_0)
    undo = rewrite.apply(circuit)
    circuit.gates[0] = Gate(1, GateOp.CONST_TRUE)
    with pytest.raises(RuntimeError, match="changed after apply"):
        undo.undo(circuit)


@pytest.mark.parametrize(
    "rewrite",
    [
        GateRewrite(3, RewriteKind.CONSTANT_0),
        GateRewrite(3, RewriteKind.CONSTANT_1),
        GateRewrite(3, RewriteKind.COPY_A),
        GateRewrite(3, RewriteKind.COPY_B),
        GateRewrite(3, RewriteKind.NOT_A),
        GateRewrite(3, RewriteKind.NOT_B),
        GateRewrite(
            3,
            RewriteKind.ALTERNATIVE_GATE,
            replacement_op=GateOp.NOR,
        ),
        GateRewrite(4, RewriteKind.CONSTANT_0),
        GateRewrite(
            5,
            RewriteKind.ALTERNATIVE_GATE,
            replacement_op=GateOp.OR_NOT_A,
        ),
    ],
)
def test_incremental_rewrite_matches_complete_resimulation(rewrite):
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    trace = build_trace(circuit, inputs, labels)
    original_circuit = circuit.to_dict()
    original_values = trace.values.copy()

    evaluation = evaluate_rewrite(circuit, trace, rewrite)
    rewritten = rewrite.apply_to_copy(circuit)
    complete = build_trace(rewritten, inputs, labels)

    assert np.array_equal(evaluation.scores, complete.scores)
    assert np.array_equal(evaluation.predictions, complete.predictions)
    assert evaluation.decision_flip_count == int(
        np.count_nonzero(complete.predictions != trace.predictions)
    )
    for row, values in evaluation.updated_values.items():
        assert np.array_equal(values, complete.values[row])
    assert circuit.to_dict() == original_circuit
    assert np.array_equal(trace.values, original_values)


def test_incremental_overlapping_sequence_matches_complete_resimulation():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(circuit, inputs, labels)
    sequence = [
        GateRewrite(3, RewriteKind.CONSTANT_1),
        GateRewrite(5, RewriteKind.NOT_B),
        GateRewrite(6, RewriteKind.COPY_A),
    ]

    for rewrite in sequence:
        evaluation = evaluate_rewrite(circuit, trace, rewrite, margin_reserve=0.25)
        circuit = rewrite.apply_to_copy(circuit)
        complete = build_trace(circuit, inputs, labels)
        assert np.array_equal(evaluation.scores, complete.scores)
        assert np.array_equal(evaluation.predictions, complete.predictions)
        trace = complete


def test_greedy_search_saves_replayable_exact_snapshots():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(circuit, inputs, labels)
    config = {
        "mode": "decision-preserving",
        "search_seed": 0,
        "candidate_limit_per_iteration": 0,
        "exact_cost_shortlist": 64,
        "max_accepted_rewrites": 4,
        "maximum_global_loss": 1.0,
        "maximum_per_class_loss": 1.0,
        "margin_reserve": 0.0,
        "flip_penalty": 1.0,
        "cost_epsilon": 1e-9,
        "connection_weight": 0.05,
        "depth_weight": 0.25,
        "pareto_budgets": [0.0, 0.5, 1.0],
    }

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        summary = run_search(circuit, trace, config, output)
        log_bytes = (output / "rewrite_log.json").read_bytes()
        log = json.loads(log_bytes)

        assert summary["accepted_rewrites"] > 0
        assert summary["final_cost"]["live_gates"] < len(circuit.gates)
        assert len(log) == summary["accepted_rewrites"]
        replayed = copy.deepcopy(circuit)
        for entry in log:
            assert circuit_sha256(replayed) == entry["circuit_before_sha256"]
            GateRewrite.from_dict(entry["rewrite"]).apply(replayed)
            replayed.simplify()
            assert circuit_sha256(replayed) == entry["circuit_after_sha256"]
        assert circuit_sha256(replayed) == summary["points"][-1][
            "circuit_sha256"
        ]
        for point in summary["points"]:
            snapshot_path = output / point["snapshot"] / "circuit.json"
            assert snapshot_path.exists()

        second_output = output / "repeat"
        repeated_summary = run_search(circuit, trace, config, second_output)
        assert (second_output / "rewrite_log.json").read_bytes() == log_bytes
        assert [
            point["circuit_sha256"] for point in repeated_summary["points"]
        ] == [
            point["circuit_sha256"] for point in summary["points"]
        ]


def test_zero_budget_search_never_changes_a_reference_decision():
    circuit = small_trace_circuit()
    inputs = exhaustive_inputs(circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(circuit, inputs, labels)
    config = {
        "mode": "decision-preserving",
        "search_seed": 4,
        "candidate_limit_per_iteration": 0,
        "exact_cost_shortlist": 64,
        "max_accepted_rewrites": 4,
        "maximum_global_loss": 0.0,
        "maximum_per_class_loss": 0.0,
        "margin_reserve": 0.1,
        "margin_penalty": 1.0,
        "flip_penalty": 0.25,
        "cost_epsilon": 0.001,
        "connection_weight": 0.05,
        "depth_weight": 0.25,
        "pareto_budgets": [0.0],
    }
    with tempfile.TemporaryDirectory() as directory:
        summary = run_search(circuit, trace, config, Path(directory))
    assert summary["final_behavior"]["decision_flip_count"] == 0
    assert summary["final_behavior"]["global_loss"] == 0.0
    assert summary["final_behavior"]["maximum_per_class_loss"] == 0.0


def test_search_reports_starting_score_and_prediction_equivalence_separately():
    reference_circuit = small_trace_circuit()
    starting_circuit = copy.deepcopy(reference_circuit)
    for reduction in starting_circuit.sum_nodes:
        reduction.tau = 1.0
    inputs = exhaustive_inputs(reference_circuit.n_inputs)
    labels = torch.arange(len(inputs)) % 2
    trace = build_trace(reference_circuit, inputs, labels)
    config = {
        "mode": "decision-preserving",
        "search_seed": 0,
        "candidate_limit_per_iteration": 0,
        "exact_cost_shortlist": 1,
        "max_accepted_rewrites": 0,
        "maximum_global_loss": 0.0,
        "maximum_per_class_loss": 0.0,
        "margin_reserve": 0.0,
        "margin_penalty": 1.0,
        "flip_penalty": 1.0,
        "cost_epsilon": 0.001,
        "connection_weight": 0.05,
        "depth_weight": 0.25,
        "pareto_budgets": [0.0],
    }
    with tempfile.TemporaryDirectory() as directory:
        summary = run_search(
            starting_circuit,
            trace,
            config,
            Path(directory),
        )
    assert not summary["starting_scores_exact_vs_reference"]
    assert summary["starting_predictions_exact_vs_reference"]
