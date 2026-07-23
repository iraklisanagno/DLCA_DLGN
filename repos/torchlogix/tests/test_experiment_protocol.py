import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.train import (
    source_manifest_files,
    source_tree_sha256,
    training_implementation_sha256,
    training_manifest_files,
)
from experiments.utils import evaluate_model, split_permutation
from torchlogix.layers import FixedBinarization, GroupSum, LogicDense
from torchlogix.models import (
    DlgnCifar10Budget48kDepth8,
    DlgnCifar10Budget48kDepth12,
    DlgnCifar10Budget512kDepth8,
    DlgnCifar10Budget512kDepth12,
    DlgnCifar10Small,
    DlgnFashionMnistPaperSmall,
    DlgnFashionMnistSmall,
    DlgnMnistPaperSmall,
)


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
