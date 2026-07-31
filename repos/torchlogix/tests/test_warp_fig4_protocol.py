import json
from pathlib import Path

from torchlogix.models import ClgnCifar10Medium


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT
    / "experiments"
    / "coverage_dlgn"
    / "configs"
    / "warp_fig4_cifar10_medium"
)
PROTOCOL = (
    ROOT
    / "experiments"
    / "coverage_dlgn"
    / "protocols"
    / "warp_fig4_cifar10_medium.json"
)


def load_config(name):
    return json.loads((CONFIGS / f"{name}_seed0.json").read_text())


def test_warp_figure4_reproduction_uses_frozen_30k_medium_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["architecture"] == "ClgnCifar10Medium"
    assert ClgnCifar10Medium.n_input_bits == 2
    assert ClgnCifar10Medium.n_input_thresholds is None
    assert protocol["common_training"]["iterations"] == 30_000
    assert protocol["common_training"]["paper_curve_iterations"] == 50_000
    assert protocol["common_training"]["validation_fraction"] == 0.2
    assert protocol["heldout_test_used"] is False


def test_warp_reproduction_arms_change_only_binarization():
    uniform = load_config("warp_fixed_uniform")
    distributive = load_config("warp_fixed_distributive")
    learnable = load_config("warp_learnable")
    common_ignored = {
        "binarization",
        "binarization_init",
        "binarization_learning_rate",
        "binarization_temperature",
        "binarization_temperature_softplus",
        "binarization_forward_sampling",
        "output",
    }
    for other in [distributive, learnable]:
        assert {
            key: value
            for key, value in uniform.items()
            if key not in common_ignored
        } == {
            key: value
            for key, value in other.items()
            if key not in common_ignored
        }
    assert uniform["connections_init_method"] == "random-unique"
    assert uniform["binarization_init"] == "uniform"
    assert distributive["binarization_init"] == "distributive"
    assert learnable["binarization"] == "learnable"
    assert learnable["binarization_init"] == "distributive"
    assert learnable["binarization_learning_rate"] == 0.02
    assert learnable["binarization_temperature"] == 0.0002
    assert learnable["binarization_temperature_softplus"] == 0.01


def test_legacy_v4_pair_preserves_frozen_method_and_training_protocol():
    random = load_config("paired_random_fixed_uniform")
    v4 = load_config("legacy_v4_fixed_uniform")
    topology_fields = {
        "connections_init_method",
        "coverage_candidate_pool_size",
        "coverage_swap_fraction",
        "coverage_novelty_weight",
        "output",
    }
    assert {
        key: value
        for key, value in random.items()
        if key not in topology_fields
    } == {
        key: value
        for key, value in v4.items()
        if key not in topology_fields
    }
    assert random["connections_init_method"] == "random"
    assert v4["connections_init_method"] == "semantic_channel_hybrid"
    assert v4["coverage_candidate_pool_size"] == 8
    assert v4["coverage_swap_fraction"] == 0.25
    assert v4["coverage_novelty_weight"] == 1.0
    assert random["data_split_seed"] == v4["data_split_seed"] == 2027
    assert random["topology_seed"] == v4["topology_seed"] == 0
