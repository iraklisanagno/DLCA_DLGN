import numpy as np
import pytest
import torch
from argparse import Namespace

from torchlogix.connections import FixedDenseConnections
from torchlogix.topology import (
    analyze_dense_indices,
    canonical_strategy,
    generate_dense_stack,
    generate_dense_topology,
    image_input_semantics,
    packed_identity,
    packed_popcount,
    propagate_packed_ancestry,
    semantic_first_layer_pair_metrics,
)
from experiments.train import checkpoint_payload


def _unpack_rows(packed, n_inputs):
    return [
        {bit for bit in range(n_inputs) if int(row[bit // 64]) & (1 << (bit % 64))}
        for row in packed
    ]


def _brute_force_stack(n_inputs, layers):
    ancestry = [{i} for i in range(n_inputs)]
    result = []
    for indices in layers:
        ancestry = [
            set().union(*(ancestry[int(indices[rank, gate])] for rank in range(indices.shape[0])))
            for gate in range(indices.shape[1])
        ]
        result.append(ancestry)
    return result


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("random-unique", "random_unique"),
        ("local-cyclic", "local_cyclic"),
        ("coverage-greedy", "coverage_greedy"),
        ("coverage-hybrid", "coverage_hybrid"),
        ("semantic-balanced-hybrid", "semantic_balanced_hybrid"),
        ("semantic-channel-hybrid", "semantic_channel_hybrid"),
    ],
)
def test_strategy_aliases(alias, canonical):
    assert canonical_strategy(alias) == canonical


def test_local_cyclic_expected_small_example():
    result = generate_dense_topology(
        8, 8, strategy="local_cyclic", layer_index=0, local_radius=1
    )
    assert result.indices.tolist() == [
        list(range(8)),
        [1, 2, 3, 4, 5, 6, 7, 0],
    ]


def test_butterfly_expected_small_examples():
    depth_zero = generate_dense_topology(8, 8, strategy="butterfly", layer_index=0)
    depth_one = generate_dense_topology(8, 8, strategy="butterfly", layer_index=1)
    assert depth_zero.indices.T.tolist() == [
        [4, 5], [6, 7], [0, 1], [2, 3],
        [5, 7], [4, 6], [1, 3], [0, 2],
    ]
    assert depth_one.indices.T.tolist() == [
        [4, 6], [1, 3], [0, 2], [5, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        "random",
        "random_unique",
        "local_cyclic",
        "butterfly",
        "coverage_greedy",
        "coverage_hybrid",
        "semantic_balanced_hybrid",
    ],
)
def test_generators_are_deterministic_and_in_bounds(strategy):
    semantics = (
        image_input_semantics(
            1, 1, 17, 1, layout="pixel_interleaved"
        )
        if strategy == "semantic_balanced_hybrid" else None
    )
    kwargs = dict(
        in_dim=17,
        out_dim=20,
        strategy=strategy,
        topology_seed=91,
        layer_index=2,
        candidate_pool_size=12,
        long_range_fraction=0.3,
        input_semantics=semantics,
        swap_fraction=0.3,
    )
    first = generate_dense_topology(**kwargs)
    second = generate_dense_topology(**kwargs)
    assert np.array_equal(first.indices, second.indices)
    assert first.indices.shape == (2, 20)
    assert first.indices.min() >= 0
    assert first.indices.max() < 17
    if strategy != "random":
        assert np.all(first.indices[0] != first.indices[1])


def test_topology_rng_does_not_change_torch_rng():
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    generate_dense_topology(
        16, 16, strategy="coverage_greedy", topology_seed=4, candidate_pool_size=8
    )
    actual = torch.rand(5)
    assert torch.equal(actual, expected)


def test_hybrid_uses_exact_requested_greedy_fraction():
    result = generate_dense_topology(
        32,
        40,
        strategy="coverage_hybrid",
        topology_seed=2,
        candidate_pool_size=8,
        long_range_fraction=0.25,
    )
    assert result.greedy_mask.sum() == 10


def test_image_input_semantics_matches_binarizer_flattening():
    fashion = image_input_semantics(
        1, 2, 2, 3, layout="pixel_interleaved"
    )
    assert fashion.source_ids.tolist() == [
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3
    ]
    assert fashion.threshold_ids.tolist() == [0, 1, 2] * 4

    cifar = image_input_semantics(
        2, 1, 2, 3, layout="channel_interleaved"
    )
    assert cifar.source_ids.tolist() == [
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
    ]
    assert cifar.threshold_ids.tolist() == [
        0, 0, 1, 1, 2, 2,
        0, 0, 1, 1, 2, 2,
    ]


def test_semantic_butterfly_never_pairs_thresholds_of_same_source():
    semantics = image_input_semantics(
        1, 4, 4, 3, layout="pixel_interleaved"
    )
    result = generate_dense_topology(
        semantics.n_inputs,
        64,
        strategy="semantic_balanced_hybrid",
        topology_seed=7,
        input_semantics=semantics,
        swap_fraction=0.0,
    )
    metrics = semantic_first_layer_pair_metrics(result.indices, semantics)
    assert metrics["same_source_pair_fraction"] == 0.0
    assert metrics["spatial_manhattan_p50"] > 0


def test_semantic_butterfly_exhausted_stage_matches_original_rule():
    semantics = image_input_semantics(
        1, 2, 2, 2, layout="pixel_interleaved"
    )
    result = generate_dense_topology(
        semantics.n_inputs,
        32,
        strategy="semantic_balanced_hybrid",
        topology_seed=3,
        input_semantics=semantics,
        swap_fraction=0.0,
    )
    # Frozen before replacing repeated exhaustive stage scans with an exact
    # lazy priority queue. This exercises outputs beyond all unique pairs.
    expected = np.asarray([
        [
            0, 1, 4, 1, 2, 0, 4, 3, 5, 5, 0, 1, 0, 1, 4, 1,
            2, 0, 4, 1, 4, 0, 3, 5, 0, 1, 4, 1, 2, 0, 4, 1,
        ],
        [
            2, 5, 7, 3, 6, 3, 6, 7, 6, 7, 4, 2, 2, 5, 7, 3,
            6, 3, 6, 5, 7, 2, 7, 6, 2, 5, 7, 3, 6, 3, 6, 5,
        ],
    ], dtype=np.int64)
    assert np.array_equal(result.indices, expected)


def test_semantic_swaps_preserve_exact_predecessor_degrees():
    semantics = image_input_semantics(
        1, 4, 4, 3, layout="pixel_interleaved"
    )
    kwargs = dict(
        in_dim=semantics.n_inputs,
        out_dim=96,
        strategy="semantic_balanced_hybrid",
        topology_seed=11,
        input_semantics=semantics,
        candidate_pool_size=12,
        novelty_weight=1.0,
        output_groups=4,
    )
    base = generate_dense_topology(**kwargs, swap_fraction=0.0)
    swapped = generate_dense_topology(**kwargs, swap_fraction=0.5)
    base_degree = np.bincount(
        base.indices.reshape(-1), minlength=semantics.n_inputs
    )
    swapped_degree = np.bincount(
        swapped.indices.reshape(-1), minlength=semantics.n_inputs
    )
    assert np.array_equal(swapped_degree, base_degree)
    assert np.all(swapped.indices[0] != swapped.indices[1])
    assert swapped.greedy_mask.sum() % 2 == 0


def test_semantic_stack_reports_source_and_group_diagnostics():
    semantics = image_input_semantics(
        1, 4, 4, 3, layout="pixel_interleaved"
    )
    _, rows = generate_dense_stack(
        semantics.n_inputs,
        [64, 64],
        strategy="semantic_balanced_hybrid",
        topology_seed=3,
        input_semantics=semantics,
        candidate_pool_size=8,
        swap_fraction=0.25,
        output_groups=4,
    )
    assert rows[0]["source_original_inputs"] == 16
    assert rows[0]["same_source_pair_fraction"] == 0.0
    assert 0.0 <= rows[-1]["source_cross_gate_jaccard_mean"] <= 1.0
    assert 0.0 <= rows[-1]["source_group_coverage_min"] <= 1.0


def test_tiny_long_range_pool_larger_than_available_pairs_terminates():
    result = generate_dense_topology(
        8,
        8,
        strategy="coverage_hybrid",
        topology_seed=3,
        candidate_pool_size=64,
        long_range_fraction=0.5,
    )
    assert result.indices.shape == (2, 8)


def test_packed_ancestry_matches_brute_force():
    first = np.asarray([[0, 2, 4, 6], [1, 3, 5, 7]])
    second = np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]])
    packed = packed_identity(8)
    packed_first = propagate_packed_ancestry(packed, first)
    packed_second = propagate_packed_ancestry(packed_first, second)
    brute = _brute_force_stack(8, [first, second])
    assert _unpack_rows(packed_first, 8) == brute[0]
    assert _unpack_rows(packed_second, 8) == brute[1]
    assert packed_popcount(packed_second).tolist() == [4, 4, 4, 4]


def test_metrics_report_known_coverage_overlap_and_fanout():
    indices = np.asarray([[0, 2, 4, 6], [1, 3, 5, 7]])
    rows = analyze_dense_indices(8, [(indices, "butterfly")])
    row = rows[0]
    assert row["input_coverage"] == 1.0
    assert row["mean_gate_ancestry"] == 2.0
    assert row["overlap_mean"] == 0.0
    assert row["fanout_cv"] == 0.0
    assert row["distinct_predecessor_pairs"] == 4
    assert row["unused_outputs"] == 0


def test_generated_stack_propagates_previous_layer_ancestry():
    results, rows = generate_dense_stack(
        16,
        [16, 16, 16],
        strategy="coverage_greedy",
        topology_seed=5,
        candidate_pool_size=16,
    )
    brute = _brute_force_stack(16, [result.indices for result in results])
    for result, expected in zip(results, brute):
        assert _unpack_rows(result.output_ancestry, 16) == expected
    assert rows[-1]["mean_gate_ancestry"] >= rows[0]["mean_gate_ancestry"]


def test_fixed_dense_connection_integration_and_checkpoint_indices():
    connection = FixedDenseConnections(
        in_dim=16,
        out_dim=16,
        init_method="coverage_hybrid",
        topology_seed=8,
        layer_index=0,
        input_ancestry=packed_identity(16),
        candidate_pool_size=8,
    )
    x = torch.rand(3, 16)
    assert connection(x).shape == (3, 2, 16)
    assert "indices" in connection.state_dict()
    assert connection.consume_output_ancestry().shape == (16, 1)


def test_coverage_strategies_reject_higher_rank():
    with pytest.raises(NotImplementedError):
        generate_dense_topology(8, 8, strategy="coverage_hybrid", lut_rank=4)


def test_rich_checkpoint_uses_safe_plain_metadata(tmp_path):
    model = torch.nn.Linear(2, 1)
    payload = checkpoint_payload(
        model,
        Namespace(output=tmp_path, seed=np.int64(7)),
        3,
        {"val_acc_discrete": np.float64(0.5)},
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    loaded = torch.load(path)
    assert loaded["metadata"]["configuration"]["seed"] == 7
    assert loaded["metadata"]["metrics"]["val_acc_discrete"] == 0.5
