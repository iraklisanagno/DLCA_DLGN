import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.coverage_dlgn.prepare_table1_final import (
    final_eval_freq,
    final_seeds_for,
)
from experiments.coverage_dlgn.evaluate_table1_final import round_robin
from experiments.coverage_dlgn.summarize_table1_final import mean_ci_95
from experiments.coverage_dlgn.summarize_table2_compression_screen import (
    select_coverage_candidates,
)
from experiments.train import (
    log_linear_schedule,
    model_cost_summary,
    report_connection_strategy,
    source_manifest_files,
    source_tree_sha256,
    training_implementation_sha256,
    training_manifest_files,
)
from experiments.utils import (
    evaluate_model,
    input_threshold_count,
    split_permutation,
)
from torchlogix.layers import (
    FixedBinarization,
    GroupSum,
    LogicConv2d,
    LogicDense,
)
from torchlogix.models import (
    ClgnCifar10PaperMedium,
    ClgnCifar10PaperSmall,
    ClgnCifar10Small,
    DlgnCifar10Budget128k,
    DlgnCifar10Budget256k,
    DlgnCifar10Budget384k,
    DlgnCifar10MediumLearnable,
    DlgnCifar10SmallLearnable,
    DlgnCifar10Budget48kDepth8,
    DlgnCifar10Budget48kDepth12,
    DlgnCifar10Budget512kDepth8,
    DlgnCifar10Budget512kDepth12,
    DlgnCifar10Large,
    DlgnCifar10LargeLearnable,
    DlgnCifar10Small,
    DlgnCifar100BitLogicL,
    DlgnCifar100BitLogicM,
    DlgnCifar100BitLogicS,
    DlgnFashionMnistPaperSmall,
    DlgnFashionMnistPaperSmallLearnable,
    DlgnFashionMnistBitLogic48k,
    DlgnMnistBitLogic48k,
    DlgnFashionMnistSmall,
    DlgnMnistPaperSmall,
    DlgnMnistPaperSmallLearnable,
)
from torchlogix.connections import LearnableDenseConnections


@pytest.mark.parametrize(
    ("num_iterations", "steps_per_epoch", "expected"),
    [
        (108_000, 540, 2_000),
        (84_400, 422, 1_688),
        (42_200, 211, 844),
    ],
)
def test_table1_final_eval_frequency_divides_training(
    num_iterations, steps_per_epoch, expected
):
    eval_freq = final_eval_freq(num_iterations, steps_per_epoch)
    assert eval_freq == expected
    assert num_iterations % eval_freq == 0


def test_table1_final_seed_policy_keeps_only_mnist_exception():
    for family in ["random", "coverage_v3"]:
        assert final_seeds_for("mnist", family) == [0, 1, 2, 3, 4]
        assert final_seeds_for("fashion", family) == [0, 1, 2, 3, 4]
    for family in ["mommen", "lilogic", "bitlogic"]:
        assert final_seeds_for("mnist", family) == [0, 1, 2, 3, 4]
        assert final_seeds_for("fashion", family) == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown Table 1 cell"):
        final_seeds_for("cifar10", "random")


def test_mean_ci_95_uses_student_t_for_five_paired_seeds():
    mean, low, high = mean_ci_95([0.1, 0.2, 0.3, 0.4, 0.5])
    assert mean == pytest.approx(0.3)
    assert low == pytest.approx(0.1036757)
    assert high == pytest.approx(0.4963243)


def test_table1_test_queue_assigns_every_run_once():
    paths = [Path(f"run-{index}") for index in range(5)]
    assignments = round_robin(paths, 2)
    assert assignments == [
        [Path("run-0"), Path("run-2"), Path("run-4")],
        [Path("run-1"), Path("run-3")],
    ]


def _paper_model_kwargs(thresholds):
    return {
        "thresholds": thresholds,
        "binarization": "fixed",
        "binarization_kwargs": {},
        "connections": "fixed",
        "connections_kwargs": {"init_method": "random", "topology_seed": 0},
        "parametrization": "raw",
        "parametrization_kwargs": {"weight_init": "random"},
        "device": "cpu",
        "lut_rank": 2,
    }


def test_topology_report_strategy_uses_component_override():
    args = Namespace(
        connections_init_method="random",
        conv_connections_init_method="ancestry_channel_hybrid",
        classifier_connections_init_method="semantic_classifier_hybrid",
    )
    assert (
        report_connection_strategy(args, "conv")
        == "ancestry_channel_hybrid"
    )
    assert (
        report_connection_strategy(args, "classifier")
        == "semantic_classifier_hybrid"
    )
    args.conv_connections_init_method = None
    args.classifier_connections_init_method = None
    assert report_connection_strategy(args, "conv") == "random"
    assert report_connection_strategy(args, "classifier") == "random"
    with pytest.raises(ValueError, match="Unknown topology-report component"):
        report_connection_strategy(args, "unknown")


def test_paper_mnist_small_has_reported_gate_budget():
    model = DlgnMnistPaperSmall(**_paper_model_kwargs(torch.tensor([0.5])))
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 6
    assert sum(layer.out_dim for layer in layers) == 48_000
    assert layers[0].in_dim == 784
    assert all(layer.out_dim == 8_000 for layer in layers)
    assert all(layer.in_dim == 8_000 for layer in layers[1:])
    assert isinstance(model[-1], GroupSum)
    assert model[-1].k == 10
    assert model[-1].tau == 10.0


def test_paper_fashion_mnist_small_has_reported_gate_budget_and_encoding():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    model = DlgnFashionMnistPaperSmall(**_paper_model_kwargs(thresholds))
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert isinstance(model[0], FixedBinarization)
    assert torch.equal(model[0].get_thresholds(), thresholds)
    assert len(layers) == 6
    assert sum(layer.out_dim for layer in layers) == 48_000
    assert layers[0].in_dim == 28 * 28 * 3
    assert all(layer.out_dim == 8_000 for layer in layers)
    assert all(layer.in_dim == 8_000 for layer in layers[1:])
    assert model[-1].tau == 10.0


@pytest.mark.parametrize(
    ("model_cls", "thresholds"),
    [
        (DlgnMnistPaperSmallLearnable, torch.tensor([0.5])),
        (
            DlgnFashionMnistPaperSmallLearnable,
            torch.tensor([0.25, 0.5, 0.75]),
        ),
    ],
)
def test_table1_learnable_comparators_keep_48k_target(
    model_cls, thresholds
):
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections_kwargs"].update({
        "num_candidates": 16,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = model_cls(**kwargs)
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 6
    assert sum(layer.out_dim for layer in layers) == 48_000
    assert all(
        isinstance(layer.connections, LearnableDenseConnections)
        for layer in layers
    )
    assert all(layer.connections.num_candidates == 16 for layer in layers)
    assert all(
        layer.connections.forward_mode == "soft_mix" for layer in layers
    )


@pytest.mark.parametrize(
    ("model_cls", "thresholds"),
    [
        (DlgnMnistBitLogic48k, torch.tensor([0.2, 0.4, 0.6, 0.8])),
        (
            DlgnFashionMnistBitLogic48k,
            torch.tensor([0.2, 0.4, 0.6, 0.8]),
        ),
    ],
)
def test_table1_bitlogic_coordinate_has_48k_rank4_gates(
    model_cls, thresholds
):
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["lut_rank"] = 4
    kwargs["parametrization"] = "light"
    kwargs["connections_kwargs"].update({
        "num_candidates": 16,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = model_cls(**kwargs)
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 2
    assert sum(layer.out_dim for layer in layers) == 48_000
    assert all(layer.lut_rank == 4 for layer in layers)
    assert all(layer.out_dim == 24_000 for layer in layers)
    assert all(
        isinstance(layer.connections, LearnableDenseConnections)
        for layer in layers
    )


def test_log_linear_temperature_schedule_has_locked_endpoints():
    assert log_linear_schedule(0, 100, 1.0, 1e-4, 0.5, 0.75) == 1.0
    assert log_linear_schedule(50, 100, 1.0, 1e-4, 0.5, 0.75) == 1.0
    assert log_linear_schedule(75, 100, 1.0, 1e-4, 0.5, 0.75) == 1e-4
    assert log_linear_schedule(100, 100, 1.0, None, 0.5, 0.75) == 1.0
    assert log_linear_schedule(62, 100, 1.0, 1e-4, 0.5, 0.75) == (
        pytest.approx(10 ** -1.92)
    )
    with pytest.raises(ValueError, match="0 <= start < end <= 1"):
        log_linear_schedule(0, 100, 1.0, 1e-4, 0.8, 0.2)


def test_table1_cost_summary_separates_training_routing_from_deployment():
    kwargs = _paper_model_kwargs(torch.tensor([0.5]))
    kwargs["connections_kwargs"].update({
        "num_candidates": 16,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = DlgnMnistPaperSmallLearnable(**kwargs)
    cost = model_cost_summary(model)
    assert cost["dense_gate_count"] == 48_000
    assert cost["training_routing_parameters"] == 6 * 16 * 2 * 8_000
    assert cost["trainable_parameters"] == (
        6 * 16 * 8_000 + cost["training_routing_parameters"]
    )
    expected_routing_bits = (
        2 * 8_000 * 10
        + 5 * 2 * 8_000 * 13
    )
    assert cost["deployed_routing_bits"] == expected_routing_bits
    assert cost["deployed_routing_bytes_packed"] == (
        expected_routing_bits + 7
    ) // 8


def test_legacy_fashion_mnist_small_accepts_fixed_binarization():
    model = DlgnFashionMnistSmall(
        **_paper_model_kwargs(torch.tensor([0.25, 0.5, 0.75]))
    )
    assert isinstance(model[0], FixedBinarization)


def test_paper_cifar10_small_has_reported_gate_budget_and_encoding():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    model = DlgnCifar10Small(**_paper_model_kwargs(thresholds))
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 4
    assert sum(layer.out_dim for layer in layers) == 48_000
    assert layers[0].in_dim == 3 * 32 * 32 * 3
    assert all(layer.out_dim == 12_000 for layer in layers)
    assert all(layer.in_dim == 12_000 for layer in layers[1:])
    assert model[-1].tau == 1.0 / 0.03


def test_paper_clgn_cifar10_small_matches_logic_tree_net_s():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    model = ClgnCifar10PaperSmall(**_paper_model_kwargs(thresholds))
    conv_layers = [module for module in model if isinstance(module, LogicConv2d)]
    dense_layers = [module for module in model if isinstance(module, LogicDense)]

    assert input_threshold_count(ClgnCifar10PaperSmall) == 3
    assert input_threshold_count(ClgnCifar10Small) == 2
    assert ClgnCifar10PaperSmall.input_precision_bits == 2
    assert ClgnCifar10PaperSmall.paper_model_identifier == "S"
    assert isinstance(model[0], FixedBinarization)
    assert torch.equal(model[0].get_thresholds(), thresholds)

    assert [layer.channels for layer in conv_layers] == [9, 32, 128, 512]
    assert [layer.num_kernels for layer in conv_layers] == [32, 128, 512, 1024]
    assert [layer.tree_depth for layer in conv_layers] == [3, 3, 3, 3]
    assert [layer.receptive_field_size for layer in conv_layers] == [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]
    assert [layer.padding for layer in conv_layers] == [1, 1, 1, 1]
    assert [layer.kernel_positions for layer in conv_layers] == [
        [32, 32],
        [16, 16],
        [8, 8],
        [4, 4],
    ]
    assert [
        layer.connections.channel_group_size for layer in conv_layers
    ] == [2, 2, 2, 2]

    assert [(layer.in_dim, layer.out_dim) for layer in dense_layers] == [
        (4096, 40_960),
        (40_960, 20_480),
        (20_480, 10_240),
    ]
    learned_gate_functions = (
        sum(layer.num_kernels * 7 for layer in conv_layers)
        + sum(layer.out_dim for layer in dense_layers)
    )
    assert learned_gate_functions == 83_552
    assert isinstance(model[-1], GroupSum)
    assert model[-1].tau == 20
    assert dense_layers[-1].out_dim // 10 == 1_024


def test_paper_clgn_cifar10_medium_declares_logic_tree_net_m_scale():
    assert ClgnCifar10PaperMedium.paper_model_identifier == "M"
    assert ClgnCifar10PaperMedium.k_num == 256
    assert ClgnCifar10PaperMedium.tau == 40
    assert ClgnCifar10PaperMedium.input_precision_bits == 2
    assert input_threshold_count(ClgnCifar10PaperMedium) == 3
    assert ClgnCifar10PaperMedium.group_size == 2
    assert ClgnCifar10PaperMedium.output_gate_factor == 1


def test_paper_clgn_rejects_legacy_two_threshold_encoding():
    with pytest.raises(AssertionError, match="requires 3 input thresholds"):
        ClgnCifar10PaperSmall(
            **_paper_model_kwargs(torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
        )


def test_paper_clgn_supports_complementary_conv_and_classifier_schedules():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections_kwargs"].update({
        "conv_init_method": "ancestry_channel_hybrid",
        "classifier_init_method": "semantic_classifier_hybrid",
        "candidate_pool_size": 8,
        "swap_fraction": 0.25,
        "novelty_weight": 1.0,
    })
    model = ClgnCifar10PaperSmall(**kwargs)
    conv_layers = [module for module in model if isinstance(module, LogicConv2d)]
    dense_layers = [module for module in model if isinstance(module, LogicDense)]
    assert [
        layer.connections.strategy for layer in conv_layers
    ] == ["ancestry_channel_hybrid"] * 4
    assert [
        layer.connections.strategy for layer in dense_layers
    ] == ["semantic_classifier_hybrid"] * 3
    assert [
        torch.unique(
            torch.sort(layer.connections.channel_pairs.T, dim=1).values,
            dim=0,
        ).shape[0]
        for layer in conv_layers
    ] == [32, 128, 512, 1024]
    assert dense_layers[-1].connections.output_groups == 10
    assert sum(
        layer.num_kernels * 7 for layer in conv_layers
    ) + sum(layer.out_dim for layer in dense_layers) == 83_552


def test_paper_clgn_supports_generic_coverage_reuse_refinement():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections_kwargs"].update({
        "conv_init_method": "coverage_reuse_hybrid",
        "candidate_pool_size": 8,
        "swap_fraction": 0.25,
        "novelty_weight": 1.0,
        "reuse_change_fraction": 0.25,
        "reuse_weight": 1.0,
    })
    model = ClgnCifar10PaperSmall(**kwargs)
    conv_layers = [module for module in model if isinstance(module, LogicConv2d)]
    dense_layers = [module for module in model if isinstance(module, LogicDense)]
    assert [
        layer.connections.strategy for layer in conv_layers
    ] == ["coverage_reuse_hybrid"] * 4
    assert [
        layer.connections.strategy for layer in dense_layers
    ] == ["random"] * 3
    assert sum(
        layer.num_kernels * 7 for layer in conv_layers
    ) + sum(layer.out_dim for layer in dense_layers) == 83_552


def test_paper_clgn_sm_pilot_pairs_differ_only_in_topology_controls():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "coverage_dlgn"
        / "configs"
    )
    topology_only_keys = {
        "connections_init_method",
        "coverage_candidate_pool_size",
        "coverage_swap_fraction",
        "coverage_novelty_weight",
        "output",
    }
    for scale in ("small", "medium"):
        random_config = json.loads(
            (
                config_dir
                / f"pilot_conv_cifar10_paper_{scale}_random_seed0.json"
            ).read_text()
        )
        v4_config = json.loads(
            (
                config_dir
                / (
                    f"pilot_conv_cifar10_paper_{scale}_"
                    "semantic_channel_v4_seed0.json"
                )
            ).read_text()
        )
        assert random_config["connections_init_method"] == "random"
        assert (
            v4_config["connections_init_method"]
            == "semantic_channel_hybrid"
        )
        assert {
            key: value
            for key, value in random_config.items()
            if key not in topology_only_keys
        } == {
            key: value
            for key, value in v4_config.items()
            if key not in topology_only_keys
        }


def test_controlled_cifar10_depth_models_hold_gate_budget_and_logit_scale():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    cases = (
        (DlgnCifar10Budget48kDepth8, 8, 48_000, 36.0),
        (DlgnCifar10Budget48kDepth12, 12, 48_000, 36.0),
        (DlgnCifar10Budget512kDepth8, 8, 512_000, 128.0),
        (DlgnCifar10Budget512kDepth12, 12, 512_000, 128.0),
    )
    for model_cls, depth, budget, expected_max_logit in cases:
        model = model_cls(**_paper_model_kwargs(thresholds))
        layers = [module for module in model if isinstance(module, LogicDense)]
        assert len(layers) == depth
        assert sum(layer.out_dim for layer in layers) == budget
        assert layers[0].in_dim == 3 * 32 * 32 * 3
        assert layers[-1].out_dim % 10 == 0
        max_logit = (layers[-1].out_dim / 10) / model[-1].tau
        assert max_logit == expected_max_logit


def test_cifar10_compression_models_hold_depth_budget_and_logit_scale():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    cases = (
        (DlgnCifar10Budget128k, 128_000),
        (DlgnCifar10Budget256k, 256_000),
        (DlgnCifar10Budget384k, 384_000),
    )
    for model_cls, budget in cases:
        model = model_cls(**_paper_model_kwargs(thresholds))
        layers = [module for module in model if isinstance(module, LogicDense)]
        assert len(layers) == 4
        assert sum(layer.out_dim for layer in layers) == budget
        assert layers[0].in_dim == 3 * 32 * 32 * 3
        assert len({layer.out_dim for layer in layers}) == 1
        assert layers[-1].out_dim % 10 == 0
        max_logit = (layers[-1].out_dim / 10) / model[-1].tau
        assert max_logit == 128.0


def test_cifar10_large_matches_paper_architecture_and_five_bit_input():
    thresholds = torch.tensor([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    model = DlgnCifar10Large(**_paper_model_kwargs(thresholds))
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert model.n_input_bits == 5
    assert len(layers) == 5
    assert all(layer.out_dim == 256_000 for layer in layers)
    assert sum(layer.out_dim for layer in layers) == 1_280_000
    assert layers[0].in_dim == 3 * 32 * 32 * 5
    assert layers[-1].out_dim % 10 == 0
    assert model[-1].tau == 100.0


def test_cifar10_large_mommen_wrapper_preserves_exact_architecture():
    thresholds = torch.tensor([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections_kwargs"].update({
        "num_candidates": 8,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = DlgnCifar10LargeLearnable(**kwargs)
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 5
    assert all(layer.out_dim == 256_000 for layer in layers)
    assert sum(layer.out_dim for layer in layers) == 1_280_000
    assert layers[0].in_dim == 3 * 32 * 32 * 5
    assert all(
        isinstance(layer.connections, LearnableDenseConnections)
        for layer in layers
    )
    assert model[-1].tau == 100.0


@pytest.mark.parametrize(
    ("model_cls", "width", "budget"),
    [
        (DlgnCifar100BitLogicS, 4_000, 8_000),
        (DlgnCifar100BitLogicM, 16_000, 32_000),
        (DlgnCifar100BitLogicL, 64_000, 128_000),
    ],
)
def test_cifar100_bitlogic_rank2_ladder_matches_common_protocol(
    model_cls, width, budget
):
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    model = model_cls(**_paper_model_kwargs(thresholds))
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert model.n_input_bits == 3
    assert model.class_count == 100
    assert len(layers) == 2
    assert all(layer.out_dim == width for layer in layers)
    assert sum(layer.out_dim for layer in layers) == budget
    assert layers[0].in_dim == 3 * 32 * 32 * 3
    assert layers[0].connections.allow_partial_input_coverage
    if width == 4_000:
        assert torch.unique(layers[0].connections.indices).numel() == 8_000
    assert model[-1].k == 100
    assert model[-1].tau == 1.0


def test_cifar100_s_screen_changes_only_frozen_v3_controls():
    queue_path = (
        Path("experiments/coverage_dlgn/queues")
        / "table4_cifar100_s_screen.json"
    )
    queue = json.loads(queue_path.read_text())
    assert queue["heldout_test_used"] is False
    assert len(queue["entries"]) == 8
    for entry in queue["entries"]:
        config = json.loads(Path(entry["config"]).read_text())
        assert config["dataset"] == "cifar-100"
        assert config["architecture"] == "DlgnCifar100BitLogicS"
        assert config["augmentation"] == "bitlogic"
        assert config["batch_size"] == 128
        assert config["learning_rate"] == 0.01
        assert config["weight_decay"] == 0.0
        assert config["num_iterations"] == 5_000
        assert config["lut_rank"] == 2
        assert config["parametrization"] == "raw"
        assert config["seed"] == config["topology_seed"] == 0
        if entry["family"] == "coverage_v3":
            assert (
                config["connections_init_method"]
                == "semantic_balanced_hybrid"
            )
            assert set(config).issuperset({
                "coverage_candidate_pool_size",
                "coverage_swap_fraction",
                "coverage_novelty_weight",
            })
        else:
            assert config["connections_init_method"] == "random"


def test_cifar100_s_opt_in_supports_topk_comparators_at_exact_width():
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections"] = "learnable"
    kwargs["connections_kwargs"].update({
        "num_candidates": 32,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = DlgnCifar100BitLogicS(**kwargs)
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert all(
        isinstance(layer.connections, LearnableDenseConnections)
        for layer in layers
    )
    assert all(
        layer.connections.allow_partial_input_coverage
        for layer in layers
    )
    assert layers[0].connections.indices.shape == (32, 2, 4_000)


def test_cifar100_s_selection_is_paired_and_uses_screen_winner():
    queue_path = (
        Path("experiments/coverage_dlgn/queues")
        / "table4_cifar100_s_selection.json"
    )
    queue = json.loads(queue_path.read_text())
    assert queue["heldout_test_used"] is False
    assert queue["selected_v3_candidate"] == "swap0125"
    assert len(queue["entries"]) == 6
    for seed in (0, 1, 2):
        entries = [
            entry for entry in queue["entries"]
            if entry["seed"] == seed
        ]
        assert {entry["family"] for entry in entries} == {
            "random", "coverage_v3"
        }
        for entry in entries:
            config = json.loads(Path(entry["config"]).read_text())
            assert config["seed"] == config["topology_seed"] == seed
            assert config["num_iterations"] == 20_000
            assert config["eval_freq"] == 2_000
            assert config["parametrization"] == "raw"
            if entry["family"] == "coverage_v3":
                assert config["coverage_candidate_pool_size"] == 8
                assert config["coverage_swap_fraction"] == 0.125
                assert config["coverage_novelty_weight"] == 1.0


@pytest.mark.parametrize(
    ("model_cls", "budget", "width", "tau"),
    [
        (DlgnCifar10SmallLearnable, 48_000, 12_000, 1.0 / 0.03),
        (DlgnCifar10MediumLearnable, 512_000, 128_000, 1.0 / 0.01),
    ],
)
def test_cifar10_learnable_comparators_preserve_exact_sm_architecture(
    model_cls, budget, width, tau
):
    thresholds = torch.tensor([0.25, 0.5, 0.75])
    kwargs = _paper_model_kwargs(thresholds)
    kwargs["connections_kwargs"].update({
        "num_candidates": 8,
        "forward_mode": "soft_mix",
        "weights_init": "normal",
    })
    model = model_cls(**kwargs)
    layers = [module for module in model if isinstance(module, LogicDense)]
    assert len(layers) == 4
    assert sum(layer.out_dim for layer in layers) == budget
    assert all(layer.out_dim == width for layer in layers)
    assert all(
        isinstance(layer.connections, LearnableDenseConnections)
        for layer in layers
    )
    assert model[-1].tau == tau


def test_compression_screen_advances_ties_and_incumbent():
    def row(candidate, accuracy):
        return {
            "name": candidate,
            "family": "coverage_v3",
            "candidate": candidate,
            "best_validation_hard_accuracy": accuracy,
        }

    tied = [
        row("incumbent", 0.55),
        row("novelty050", 0.55),
        row("novelty200", 0.55),
        row("pool4", 0.54),
    ]
    assert select_coverage_candidates(tied) == [
        "incumbent",
        "novelty050",
        "novelty200",
    ]

    incumbent_third = [
        row("pool4", 0.56),
        row("swap0500", 0.55),
        row("incumbent", 0.54),
        row("novelty050", 0.53),
    ]
    assert select_coverage_candidates(incumbent_third) == [
        "pool4",
        "swap0500",
        "incumbent",
    ]


def test_seeded_split_is_deterministic_and_rng_independent():
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)
    first = split_permutation(20, seed=2027)
    actual_next = torch.rand(4)
    assert first == split_permutation(20, seed=2027)
    assert first != split_permutation(20, seed=2028)
    assert torch.equal(actual_next, expected_next)


def test_evaluation_weights_a_partial_final_batch():
    logits = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    labels = torch.tensor([1, 1, 1, 1, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
    metrics = evaluate_model(
        torch.nn.Identity(),
        loader,
        {"acc": lambda preds, y: (preds.argmax(-1) == y).float()},
        device="cpu",
    )
    assert metrics["acc"] == 0.2


def test_source_fingerprint_excludes_generated_results(tmp_path):
    source = tmp_path / "src" / "torchlogix"
    configs = tmp_path / "experiments" / "coverage_dlgn" / "configs"
    results = tmp_path / "experiments" / "coverage_dlgn" / "results" / "run"
    summary = tmp_path / "experiments" / "coverage_dlgn" / "summary"
    for path in (source, configs, results, summary):
        path.mkdir(parents=True)
    (source / "model.py").write_text("MODEL = 1\n")
    (configs / "run.json").write_text('{"seed": 0}\n')
    (results / "environment.json").write_text('{"generated": 1}\n')
    (summary / "paired.json").write_text('{"generated": 1}\n')

    manifest = {
        path.relative_to(tmp_path).as_posix()
        for path in source_manifest_files(tmp_path)
    }
    assert manifest == {
        "src/torchlogix/model.py",
        "experiments/coverage_dlgn/configs/run.json",
    }
    before = source_tree_sha256(tmp_path)
    (results / "environment.json").write_text('{"generated": 2}\n')
    (summary / "paired.json").write_text('{"generated": 2}\n')
    assert source_tree_sha256(tmp_path) == before
    (configs / "run.json").write_text('{"seed": 1}\n')
    assert source_tree_sha256(tmp_path) != before


def test_training_fingerprint_excludes_reporting_and_configs(tmp_path):
    source = tmp_path / "src" / "torchlogix"
    experiments = tmp_path / "experiments"
    configs = experiments / "coverage_dlgn" / "configs"
    for path in (source, configs):
        path.mkdir(parents=True)
    (source / "model.py").write_text("MODEL = 1\n")
    (experiments / "train.py").write_text("TRAIN = 1\n")
    (experiments / "utils.py").write_text("UTILS = 1\n")
    (experiments / "coverage_dlgn" / "summarize_results.py").write_text(
        "REPORT = 1\n"
    )
    (configs / "run.json").write_text('{"seed": 0}\n')

    manifest = {
        path.relative_to(tmp_path).as_posix()
        for path in training_manifest_files(tmp_path)
    }
    assert manifest == {
        "src/torchlogix/model.py",
        "experiments/train.py",
        "experiments/utils.py",
    }
    before = training_implementation_sha256(tmp_path)
    (experiments / "coverage_dlgn" / "summarize_results.py").write_text(
        "REPORT = 2\n"
    )
    (configs / "run.json").write_text('{"seed": 1}\n')
    assert training_implementation_sha256(tmp_path) == before
    (source / "model.py").write_text("MODEL = 2\n")
    assert training_implementation_sha256(tmp_path) != before
