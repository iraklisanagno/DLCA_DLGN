import numpy as np
import pytest
import torch
from argparse import Namespace

from torchlogix.connections import FixedDenseConnections
from torchlogix.layers import LogicDense
from torchlogix.models import Dlgn
from torchlogix.topology import (
    add_identity_to_ancestry,
    analyze_dense_indices,
    canonical_strategy,
    combine_channel_spatial_ancestry,
    classwise_ancestry_metrics,
    class_conditional_refine,
    coverage_reuse_refine,
    generate_conv_channel_topology,
    generate_dense_stack,
    generate_dense_topology,
    image_input_semantics,
    packed_identity,
    packed_identity_in_universe,
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
        ("semantic-degree-balanced", "semantic_degree_balanced"),
        (
            "semantic-multiscale-balanced",
            "semantic_multiscale_balanced",
        ),
        ("semantic-classifier-hybrid", "semantic_classifier_hybrid"),
        ("class-conditional-coverage", "class_conditional_coverage"),
        ("semantic-channel-hybrid", "semantic_channel_hybrid"),
        (
            "semantic-channel-spatial-hybrid",
            "semantic_channel_spatial_hybrid",
        ),
        ("ancestry-channel-hybrid", "ancestry_channel_hybrid"),
        ("coverage-reuse-hybrid", "coverage_reuse_hybrid"),
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
        "semantic_degree_balanced",
        "semantic_multiscale_balanced",
    ],
)
def test_generators_are_deterministic_and_in_bounds(strategy):
    semantics = (
        image_input_semantics(
            1, 1, 17, 1, layout="pixel_interleaved"
        )
        if strategy in {
            "semantic_balanced_hybrid",
            "semantic_degree_balanced",
        } else None
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


def test_conv_channel_ancestry_schedule_is_deterministic_unique_and_balanced():
    universe = 9 + 32 + 128
    inputs = packed_identity_in_universe(9, universe)
    first = generate_conv_channel_topology(
        9,
        32,
        topology_seed=7,
        layer_index=0,
        input_ancestry=inputs,
        semantic_threshold_count=3,
    )
    repeated = generate_conv_channel_topology(
        9,
        32,
        topology_seed=7,
        layer_index=0,
        input_ancestry=inputs,
        semantic_threshold_count=3,
    )
    assert np.array_equal(first.indices, repeated.indices)
    assert first.indices.min() == 0
    assert first.indices.max() == 8
    assert np.all(first.indices[0] != first.indices[1])
    assert np.unique(np.sort(first.indices.T, axis=1), axis=0).shape[0] == 32
    fanout = np.bincount(first.indices.reshape(-1), minlength=9)
    assert fanout.max() - fanout.min() <= 1

    first_ancestry = add_identity_to_ancestry(
        first.output_ancestry,
        offset=9,
        universe_size=universe,
    )
    second = generate_conv_channel_topology(
        32,
        128,
        topology_seed=7,
        layer_index=1,
        input_ancestry=first_ancestry,
    )
    assert np.unique(
        np.sort(second.indices.T, axis=1), axis=0
    ).shape[0] == 128
    fanout = np.bincount(second.indices.reshape(-1), minlength=32)
    assert fanout.min() == fanout.max() == 8
    assert packed_popcount(second.output_ancestry).min() >= 4


def test_channel_spatial_ancestry_retains_upstream_and_unique_source_bits():
    channels = packed_identity_in_universe(3, 5)
    expanded = combine_channel_spatial_ancestry(
        channels,
        spatial_positions=4,
    )
    assert expanded.shape[0] == 12
    rows = _unpack_rows(expanded, expanded.shape[1] * 64)
    assert all(len(row) == 2 for row in rows)
    assert [min(row) for row in rows[:4]] == [0, 0, 0, 0]
    unique_bits = [max(row) for row in rows]
    assert len(set(unique_bits)) == 12


def test_classifier_hybrid_preserves_exact_compression_fanout():
    result = generate_dense_topology(
        32,
        16,
        strategy="semantic_classifier_hybrid",
        topology_seed=4,
        layer_index=5,
        candidate_pool_size=8,
        swap_fraction=0.5,
    )
    fanout = np.bincount(result.indices.reshape(-1), minlength=32)
    assert fanout.min() == fanout.max() == 1
    assert np.unique(
        np.sort(result.indices.T, axis=1), axis=0
    ).shape[0] == 16


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


def test_semantic_degree_balanced_is_the_frozen_no_swap_base():
    semantics = image_input_semantics(
        1, 4, 4, 3, layout="pixel_interleaved"
    )
    common = {
        "topology_seed": 11,
        "candidate_pool_size": 12,
        "novelty_weight": 1.0,
    }
    semantic_base = generate_dense_topology(
        semantics.n_inputs,
        96,
        strategy="semantic_balanced_hybrid",
        input_semantics=semantics,
        swap_fraction=0.0,
        **common,
    )
    candidate_first = generate_dense_topology(
        semantics.n_inputs,
        96,
        strategy="semantic_degree_balanced",
        input_semantics=semantics,
        # U1 has no swap hyperparameter: even an adversarial caller value
        # cannot alter the declared method.
        swap_fraction=1.0,
        **common,
    )
    assert np.array_equal(candidate_first.indices, semantic_base.indices)
    assert not candidate_first.greedy_mask.any()

    butterfly_base = generate_dense_topology(
        96,
        128,
        strategy="semantic_balanced_hybrid",
        layer_index=1,
        swap_fraction=0.0,
        **common,
    )
    candidate_deeper = generate_dense_topology(
        96,
        128,
        strategy="semantic_degree_balanced",
        layer_index=1,
        swap_fraction=1.0,
        **common,
    )
    assert np.array_equal(candidate_deeper.indices, butterfly_base.indices)
    base_degree = np.bincount(butterfly_base.indices.reshape(-1), minlength=96)
    candidate_degree = np.bincount(
        candidate_deeper.indices.reshape(-1), minlength=96
    )
    assert np.array_equal(candidate_degree, base_degree)
    # A partial cyclic butterfly stage at a non-power-of-two width can touch
    # both endpoints twice. The schedule remains bounded and exactly preserves
    # the declared base degree sequence; power-of-two paper convolution widths
    # are exactly uniform.
    assert int(candidate_degree.max() - candidate_degree.min()) <= 2


def test_semantic_multiscale_balanced_is_deterministic_and_degree_balanced():
    # Four ancestry groups make the regular stages differ in normalized
    # novelty. The method may choose a scale, but may never swap individual
    # edges or disturb the fan-out of complete stages.
    ancestry = packed_identity(4)[np.arange(64) % 4]
    kwargs = {
        "in_dim": 64,
        "out_dim": 128,
        "strategy": "semantic_multiscale_balanced",
        "input_ancestry": ancestry,
        "topology_seed": 19,
        "layer_index": 2,
    }
    first = generate_dense_topology(**kwargs)
    repeated = generate_dense_topology(**kwargs)
    assert np.array_equal(first.indices, repeated.indices)
    assert not first.greedy_mask.any()
    assert np.all(first.indices[0] != first.indices[1])
    degree = np.bincount(first.indices.reshape(-1), minlength=64)
    assert np.unique(degree).tolist() == [4]


@pytest.mark.parametrize(
    ("out_dim", "expected_degrees"),
    [
        (30, [1]),
        (35, [1, 2]),
        (123, [4, 5]),
    ],
)
def test_semantic_multiscale_non_power_width_has_balanced_stage_prefixes(
    out_dim, expected_degrees
):
    result = generate_dense_topology(
        in_dim=60,
        out_dim=out_dim,
        strategy="semantic_multiscale_balanced",
        input_ancestry=packed_identity(60),
        topology_seed=7,
        layer_index=1,
    )
    degree = np.bincount(result.indices.reshape(-1), minlength=60)
    assert np.unique(degree).tolist() == expected_degrees
    assert int(degree.max() - degree.min()) <= 1


def test_semantic_multiscale_rotates_odd_width_byes():
    result = generate_dense_topology(
        in_dim=9,
        out_dim=32,
        strategy="semantic_multiscale_balanced",
        input_ancestry=packed_identity(9),
        topology_seed=2,
        layer_index=0,
    )
    degree = np.bincount(result.indices.reshape(-1), minlength=9)
    assert np.unique(degree).tolist() == [7, 8]
    assert int(degree.max() - degree.min()) == 1


def test_semantic_multiscale_first_layer_keeps_semantic_source_ordering():
    semantics = image_input_semantics(
        3, 4, 4, 3, layout="channel_interleaved"
    )
    result = generate_dense_topology(
        semantics.n_inputs,
        192,
        strategy="semantic_multiscale_balanced",
        input_semantics=semantics,
        topology_seed=7,
    )
    left_sources = semantics.source_ids[result.indices[0]]
    right_sources = semantics.source_ids[result.indices[1]]
    assert np.all(left_sources != right_sources)


def test_coverage_reuse_refinement_is_bounded_degree_preserving_and_adaptive():
    first = generate_dense_topology(
        9,
        32,
        strategy="semantic_balanced_hybrid",
        topology_seed=3,
        candidate_pool_size=8,
        swap_fraction=0.25,
    )
    ancestry = propagate_packed_ancestry(packed_identity(9), first.indices)
    base = generate_dense_topology(
        32,
        128,
        strategy="semantic_balanced_hybrid",
        topology_seed=3,
        layer_index=1,
        candidate_pool_size=8,
        swap_fraction=0.25,
    )
    no_reuse = coverage_reuse_refine(
        base.indices,
        ancestry,
        topology_seed=3,
        layer_index=1,
        change_fraction=0.25,
        candidate_pool_size=8,
        reuse_weight=0.0,
    )
    balanced = coverage_reuse_refine(
        base.indices,
        ancestry,
        topology_seed=3,
        layer_index=1,
        change_fraction=0.25,
        candidate_pool_size=8,
        reuse_weight=1.0,
    )
    repeated = coverage_reuse_refine(
        base.indices,
        ancestry,
        topology_seed=3,
        layer_index=1,
        change_fraction=0.25,
        candidate_pool_size=8,
        reuse_weight=1.0,
    )
    base_degree = np.bincount(base.indices.reshape(-1), minlength=32)
    balanced_degree = np.bincount(balanced.indices.reshape(-1), minlength=32)
    changed = np.any(balanced.indices != base.indices, axis=0)
    base_vocab = {
        tuple(sorted(pair)) for pair in base.indices.T.tolist()
    }
    no_reuse_retained = sum(
        tuple(sorted(pair)) in base_vocab
        for pair in no_reuse.indices.T.tolist()
    )
    balanced_retained = sum(
        tuple(sorted(pair)) in base_vocab
        for pair in balanced.indices.T.tolist()
    )
    assert np.array_equal(balanced.indices, repeated.indices)
    assert np.array_equal(base_degree, balanced_degree)
    assert changed.sum() <= 32
    assert changed.sum() % 2 == 0
    assert 0 < changed.sum() < 32
    assert balanced_retained > no_reuse_retained
    assert np.all(balanced.indices[0] != balanced.indices[1])


def test_coverage_reuse_zero_change_is_exact_base_and_validates_weight():
    base = np.asarray([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=np.int64)
    ancestry = packed_identity(4)
    unchanged = coverage_reuse_refine(
        base,
        ancestry,
        change_fraction=0.0,
    )
    assert np.array_equal(unchanged.indices, base)
    assert not unchanged.greedy_mask.any()
    with pytest.raises(ValueError, match="must be non-negative"):
        coverage_reuse_refine(
            base,
            ancestry,
            change_fraction=0.5,
            reuse_weight=-1.0,
        )


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


def test_classwise_ancestry_metrics_detect_group_imbalance():
    ancestry = packed_identity(8)
    balanced = np.asarray([
        [0, 2, 4, 6, 0, 2, 4, 6],
        [1, 3, 5, 7, 1, 3, 5, 7],
    ])
    collapsed = np.asarray([
        [0, 0, 0, 0, 4, 4, 4, 4],
        [1, 1, 1, 1, 5, 5, 5, 5],
    ])
    balanced_metrics = classwise_ancestry_metrics(
        ancestry, balanced, n_sources=8, output_groups=2
    )
    collapsed_metrics = classwise_ancestry_metrics(
        ancestry, collapsed, n_sources=8, output_groups=2
    )
    assert balanced_metrics["class_coverage_min"] == 1.0
    assert balanced_metrics["class_source_usage_cv_mean"] == 0.0
    assert balanced_metrics["class_distinct_ancestry_fraction_mean"] == 1.0
    assert collapsed_metrics["class_coverage_min"] == 0.25
    assert collapsed_metrics["class_source_usage_cv_mean"] > 1.0
    assert collapsed_metrics["class_distinct_ancestry_fraction_mean"] == 0.25


def test_class_conditional_refinement_is_deterministic_and_degree_preserving():
    ancestry = packed_identity(32)
    base = generate_dense_topology(
        32,
        64,
        strategy="semantic_balanced_hybrid",
        topology_seed=9,
        layer_index=3,
        candidate_pool_size=8,
        swap_fraction=0.25,
        output_groups=8,
    )
    refined = class_conditional_refine(
        base.indices,
        ancestry,
        topology_seed=9,
        layer_index=3,
        change_fraction=0.5,
        candidate_pool_size=8,
        output_groups=8,
    )
    repeated = class_conditional_refine(
        base.indices,
        ancestry,
        topology_seed=9,
        layer_index=3,
        change_fraction=0.5,
        candidate_pool_size=8,
        output_groups=8,
    )
    base_degree = np.bincount(base.indices.reshape(-1), minlength=32)
    refined_degree = np.bincount(
        refined.indices.reshape(-1), minlength=32
    )
    assert np.array_equal(refined.indices, repeated.indices)
    assert np.array_equal(base_degree, refined_degree)
    assert np.all(refined.indices[0] != refined.indices[1])
    assert refined.greedy_mask.sum() <= 32
    assert refined.greedy_mask.sum() % 2 == 0


def test_class_conditional_strategy_keeps_v3_base_when_change_is_zero():
    kwargs = dict(
        in_dim=32,
        out_dim=64,
        topology_seed=5,
        layer_index=4,
        candidate_pool_size=8,
        swap_fraction=0.25,
        output_groups=8,
        novelty_weight=1.0,
    )
    v3 = generate_dense_topology(
        **kwargs, strategy="semantic_balanced_hybrid"
    )
    head = generate_dense_topology(
        **kwargs,
        strategy="class_conditional_coverage",
        class_balance_change_fraction=0.0,
    )
    assert np.array_equal(head.indices, v3.indices)


def test_dense_class_head_preserves_v3_backbone_rng_and_cost():
    def build(classifier_method=None):
        connection_kwargs = {
            "init_method": "semantic_balanced_hybrid",
            "topology_seed": 17,
            "candidate_pool_size": 8,
            "swap_fraction": 0.25,
            "novelty_weight": 1.0,
            "class_balance_change_fraction": 0.5,
        }
        if classifier_method is not None:
            connection_kwargs["classifier_init_method"] = classifier_method
        return Dlgn(
            thresholds=torch.tensor([0.5]),
            binarization="fixed",
            binarization_kwargs={},
            in_dim=16,
            n_layers=3,
            neurons_per_layer=64,
            class_count=4,
            tau=1.0,
            connections="fixed",
            connections_kwargs=connection_kwargs,
            parametrization="raw",
            parametrization_kwargs={"weight_init": "random"},
            device="cpu",
            lut_rank=2,
        )

    torch.manual_seed(23)
    frozen_v3 = build()
    torch.manual_seed(23)
    explicit_none = build(None)
    torch.manual_seed(23)
    with_head = build("class_conditional_coverage")
    v3_layers = [
        module for module in frozen_v3 if isinstance(module, LogicDense)
    ]
    none_layers = [
        module for module in explicit_none if isinstance(module, LogicDense)
    ]
    head_layers = [
        module for module in with_head if isinstance(module, LogicDense)
    ]
    assert all(
        torch.equal(left.connections.indices, right.connections.indices)
        for left, right in zip(v3_layers, none_layers)
    )
    assert all(
        torch.equal(left.connections.indices, right.connections.indices)
        for left, right in zip(v3_layers[:-1], head_layers[:-1])
    )
    assert head_layers[-1].connections.strategy == (
        "class_conditional_coverage"
    )
    assert not torch.equal(
        v3_layers[-1].connections.indices,
        head_layers[-1].connections.indices,
    )
    assert torch.equal(
        torch.bincount(v3_layers[-1].connections.indices.flatten()),
        torch.bincount(head_layers[-1].connections.indices.flatten()),
    )
    assert sum(layer.out_dim for layer in v3_layers) == sum(
        layer.out_dim for layer in head_layers
    )
    assert sum(parameter.numel() for parameter in frozen_v3.parameters()) == (
        sum(parameter.numel() for parameter in with_head.parameters())
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            frozen_v3.parameters(), with_head.parameters()
        )
    )


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
