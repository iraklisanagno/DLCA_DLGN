import itertools
import json

import numpy as np
import pytest
import torch

from experiments.coverage_dlgn.prepare_fourth_round import METHODS, U2, matrix
from experiments.train import model_cost_summary
from torchlogix.layers import LogicConv2d, LogicDense
from torchlogix.models import ClgnCifar10PaperSmall
from torchlogix.topology import generate_dense_stack, generate_dense_topology, image_input_semantics, packed_identity


@pytest.mark.parametrize("strategy", [METHODS["balanced_random"], METHODS["nominal"]])
@pytest.mark.parametrize("inputs,outputs", [(9, 32), (60, 35), (64, 128)])
def test_new_controls_bounds_balance_determinism_and_rng(strategy, inputs, outputs):
    torch.manual_seed(11)
    state = torch.get_rng_state().clone()
    cfg = dict(strategy=strategy, topology_seed=7, input_ancestry=packed_identity(inputs))
    a = generate_dense_topology(inputs, outputs, **cfg)
    b = generate_dense_topology(inputs, outputs, **cfg)
    assert np.array_equal(a.indices, b.indices)
    assert np.all(a.indices[0] != a.indices[1])
    assert a.indices.min() >= 0 and a.indices.max() < inputs
    degree = np.bincount(a.indices.ravel(), minlength=inputs)
    assert np.ptp(degree) <= 1
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(NotImplementedError, match="rank-2"):
        generate_dense_topology(inputs, outputs, strategy=strategy, lut_rank=4)


def test_controls_share_u2_semantic_first_layer():
    semantic = image_input_semantics(3, 4, 4, 3, layout="channel_interleaved")
    first = []
    for strategy in [U2, METHODS["balanced_random"], METHODS["nominal"]]:
        layers, _ = generate_dense_stack(semantic.n_inputs, [192, 256], strategy=strategy,
                                         topology_seed=7, input_semantics=semantic)
        first.append(layers[0].indices)
    assert all(np.array_equal(first[0], x) for x in first)


def build_conv(body, head, parametrization="raw", reference=True, gumbel=False):
    class Tiny(ClgnCifar10PaperSmall):
        k_num = 8
    torch.manual_seed(41)
    return Tiny(thresholds=torch.tensor([.25, .5, .75]), binarization="fixed",
                binarization_kwargs={}, device="cpu", parametrization=parametrization,
                connections_kwargs=dict(init_method=body, conv_init_method=body,
                    classifier_init_method=head, topology_seed=17, classifier_reference_u2=reference,
                    gumbel=gumbel))


@pytest.mark.parametrize("parametrization", ["raw", "warp"])
def test_factorial_holds_head_indices_parameters_and_spatial_rng(parametrization):
    models = {(body, head): build_conv(body, head, parametrization)
              for body, head in itertools.product(["random", U2], repeat=2)}
    reference = models[("random", "random")]
    for model in models.values():
        assert all(torch.equal(a, b) for a, b in zip(reference.parameters(), model.parameters()))
        assert model_cost_summary(reference) == model_cost_summary(model)
        for a, b in zip(reference, model):
            if isinstance(a, LogicConv2d):
                assert torch.equal(a.connections.indices[0][..., :-1], b.connections.indices[0][..., :-1])
    for head in ["random", U2]:
        a, b = models[("random", head)], models[(U2, head)]
        for left, right in zip(a, b):
            if isinstance(left, LogicDense):
                assert torch.equal(left.connections.indices, right.connections.indices)
    frozen = build_conv(U2, U2, parametrization, reference=False)
    assert all(torch.equal(v, models[(U2, U2)].state_dict()[k]) for k, v in frozen.state_dict().items())


@pytest.mark.parametrize("strategy", [METHODS["balanced_random"], METHODS["nominal"]])
def test_new_conv_controls_share_initial_weights_and_spatial_samples(strategy):
    random = build_conv("random", "random", reference=False)
    rng = torch.get_rng_state().clone()
    model = build_conv(strategy, strategy, reference=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(a, b) for a, b in zip(random.parameters(), model.parameters()))
    for a, b in zip(random, model):
        if isinstance(a, LogicConv2d):
            assert torch.equal(a.connections.indices[0][..., :-1], b.connections.indices[0][..., :-1])


def test_matrix_is_bounded_and_defers_hardware_and_full_s():
    rows = matrix()
    assert len(rows) == 50
    assert sum(not row["reuse"] for row in rows) == 35
    assert len({r["name"] for r in rows}) == len(rows)
    for row in rows:
        c = row["config"]
        assert c["device"] == "cuda" and c["lut_rank"] == 2
        if row["coordinate"] == "conv_s":
            assert c["num_iterations"] == 20_000
    for phase in ["warp_pilot", "warp_confirm", "warp_conv"]:
        group = [r for r in rows if r["phase"] == phase]
        for seed in {r["seed"] for r in group}:
            configs = [dict(r["config"]) for r in group if r["seed"] == seed]
            for c in configs:
                for k in ["output", "connections_init_method", "conv_connections_init_method", "classifier_connections_init_method"]:
                    c.pop(k, None)
            assert configs[0] == configs[1]


@pytest.mark.parametrize("strategy", ["random", U2])
def test_legacy_gumbel_metadata_has_no_effect_on_fixed_models(strategy):
    a = build_conv(strategy, strategy, gumbel=False)
    rng = torch.get_rng_state().clone()
    b = build_conv(strategy, strategy, gumbel=True)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())


def test_every_reused_config_matches_archived_effective_settings():
    from experiments.coverage_dlgn.prepare_fourth_round import REPO, read
    for row in matrix():
        if row["reuse"]:
            actual = read(REPO / row["output"] / "training_config.json")
            assert all(actual.get(key) == value for key, value in row["config"].items()
                       if key not in {"config", "output", "classifier_reference_u2"})
        if row["phase"].startswith("warp_"):
            assert row["provenance"] == "ADAPTED"


def test_run_validation_rejects_nonfinite_and_incomplete_evidence(tmp_path, monkeypatch):
    from experiments.coverage_dlgn import run_fourth_round as runner
    monkeypatch.setattr(runner, "REPO", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    cfg = dict(num_iterations=20, eval_freq=10, connections="fixed")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    for name in runner.ARTIFACTS:
        (output / name).write_text("{}")
    (output / "training_config.json").write_text(json.dumps(cfg))
    history = "step,val_acc_discrete\n10,0.4\n20,0.5\n"
    (output / "metrics.csv").write_text(history)
    summary = dict(best_validation_hard_accuracy=.5, peak_gpu_memory_bytes=1024)
    (output / "run_summary.json").write_text(json.dumps(summary))
    row = dict(config="config.json", output="run", reuse=False)
    assert set(runner.validate_run(row)) == set(runner.ARTIFACTS)
    for key in summary:
        for value in [float("nan"), float("inf")]:
            (output / "run_summary.json").write_text(json.dumps(dict(summary, **{key: value})))
            with pytest.raises(RuntimeError):
                runner.validate_run(row)
    (output / "run_summary.json").write_text(json.dumps(summary))
    for bad_history in [history.replace("20,0.5", "10,0.5"), history.replace("0.5", "nan")]:
        (output / "metrics.csv").write_text(bad_history)
        with pytest.raises(RuntimeError):
            runner.validate_run(row)


def transfer_manifest():
    from experiments.coverage_dlgn import evaluate_frozen_transfer as transfer
    return dict(dataset="CIFAR-10.1", version="v6", revision=transfer.REVISION, examples=2000,
                rows=[dict(row, artifacts={"training_config.json": "0" * 64,
                                          "best_checkpoint.pt": "1" * 64})
                      for row in transfer.selected_runs()])


@pytest.mark.parametrize("mutation", ["empty", "truncated", "reordered", "missing_hash", "bad_hash", "version"])
def test_transfer_rejects_changed_selection_before_any_data_access(mutation):
    from experiments.coverage_dlgn.evaluate_frozen_transfer import verify_selection
    frozen = transfer_manifest()
    verify_selection(frozen)
    if mutation == "empty": frozen["rows"] = []
    elif mutation == "truncated": frozen["rows"].pop()
    elif mutation == "reordered": frozen["rows"].reverse()
    elif mutation == "missing_hash": frozen["rows"][0]["artifacts"].pop("best_checkpoint.pt")
    elif mutation == "bad_hash": frozen["rows"][0]["artifacts"]["best_checkpoint.pt"] = "z" * 64
    else: frozen["version"] = "v4"
    with pytest.raises(RuntimeError, match="preregistration"):
        verify_selection(frozen)


def test_transfer_validates_synthetic_encoding_and_class_range():
    from experiments.coverage_dlgn.evaluate_frozen_transfer import validate_arrays
    images = np.zeros((2000, 32, 32, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 200)
    validate_arrays(images, labels)
    for invalid in [labels - 1, labels + 1, np.zeros(2000, dtype=np.int64)]:
        with pytest.raises(RuntimeError):
            validate_arrays(images, invalid)
    with pytest.raises(RuntimeError, match="image encoding"):
        validate_arrays(images.transpose(0, 3, 1, 2), labels)


def test_preparation_never_overwrites_existing_artifacts(tmp_path):
    from experiments.coverage_dlgn.prepare_fourth_round import write_new
    path = tmp_path / "frozen.json"
    write_new(path, {"value": 1})
    original = path.read_bytes()
    write_new(path, {"value": 1})
    with pytest.raises(RuntimeError, match="refusing to change"):
        write_new(path, {"value": 2})
    assert path.read_bytes() == original


def test_summary_pairs_by_seed_and_reports_factorial_interaction():
    from experiments.coverage_dlgn.summarize_fourth_round import contrast, effects
    rows = []
    for seed in [2, 0, 1]:
        for family, value in [("random", 50), ("body", 52), ("head", 53), ("both", 56)]:
            rows.append(dict(phase="factorial", coordinate="conv_s", family=family, seed=seed,
                             best_validation_hard_pct=value + seed, topology_seconds=1., cost={"gates": 10}))
    results = {row["contrast"]: row for row in effects(rows)}
    assert results["body_with_random_head"]["paired_effect"]["mean"] == 2
    assert results["body_with_u2_head"]["paired_effect"]["mean"] == 3
    interaction = results["body_head_interaction"]
    assert interaction["paired_effect"] == dict(n=3, mean=1., sample_sd=0., ci95_low=1., ci95_high=1.)
    assert interaction["wins"] == 3
    assert [row["seed"] for row in interaction["per_seed"]] == [0, 1, 2]
    assert contrast([], "factorial", "conv_s", {"random": -1, "body": 1}, "missing") is None
    rows[0]["cost"] = {"gates": 20}
    with pytest.raises(RuntimeError, match="unmatched declared cost"):
        effects(rows)


def test_execution_freeze_rejects_changed_verification_evidence(tmp_path, monkeypatch):
    from experiments.coverage_dlgn import run_fourth_round as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "LOG", tmp_path / "logs")
    monkeypatch.setattr(runner, "FREEZE", tmp_path / "freeze.json")
    monkeypatch.setattr(runner, "implementation_hash", lambda: "implementation")
    paths = ["protocols/fourth_round.json", "logs/full_tests.log", "logs/smoke.json",
             "summary/fourth_round_transfer_freeze.json", "config.json"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    frozen = dict(implementation_sha256="implementation",
                  protocol_sha256=runner.sha(tmp_path / paths[0]),
                  full_tests_sha256=runner.sha(tmp_path / paths[1]),
                  gpu_smokes={paths[2]: runner.sha(tmp_path / paths[2])},
                  transfer_freeze_sha256=runner.sha(tmp_path / paths[3]),
                  config_hashes={paths[4]: runner.sha(tmp_path / paths[4])})
    runner.FREEZE.write_text(json.dumps(frozen))
    payload = dict(entries=[dict(config="config.json", reuse=False)], raw_warp_references=[])
    runner.verify_freeze(payload)
    for name in paths:
        (tmp_path / name).write_text("changed")
        with pytest.raises(RuntimeError, match="changed"):
            runner.verify_freeze(payload)
        (tmp_path / name).write_text("original")
