"""Archival and bit-identity regressions captured before the fourth round."""
import hashlib
import json

import pytest
import torch

from experiments.coverage_dlgn.run_gpu_queue import is_complete
from torchlogix.models import ClgnCifar10PaperSmall
from torchlogix.topology import generate_dense_topology, image_input_semantics


@pytest.mark.parametrize("strategy,expected", [
    ("semantic_balanced_hybrid", "6b1f6a09c1ac32d5bcf9f8c6b0e0621a3cce2d4f75944891e4e22eda62d3c803"),
    ("semantic_degree_balanced", "b2572332092634ce542ef6148e832c859be5e744d8da997e256373b33ea37425"),
    ("semantic_multiscale_balanced", "115196db3f827262fce2fe8266454b01ba12b0437e641deccdfb104fe6074323"),
])
def test_frozen_dense_indices_match_pre_edit_digest(strategy, expected):
    semantic = image_input_semantics(3, 4, 4, 3, layout="channel_interleaved")
    ancestry = semantic.source_ancestry()
    digest = hashlib.sha256()
    for depth, width in enumerate([192, 256, 128]):
        result = generate_dense_topology(
            ancestry.shape[0], width, strategy=strategy, input_ancestry=ancestry,
            input_semantics=semantic if depth == 0 else None, topology_seed=7,
            layer_index=depth, candidate_pool_size=8, swap_fraction=0 if depth == 0 else .25,
        )
        digest.update(result.indices.tobytes())
        ancestry = result.output_ancestry
    assert digest.hexdigest() == expected


@pytest.mark.parametrize("strategy,expected", [
    ("semantic_degree_balanced", "983f14d6cdc9eead975dc66fe950b3a7d389a9be8e9db4b5b722a498c1921d1b"),
    ("semantic_multiscale_balanced", "826770d9f89dfd8aa01ee44529ef6890533d9ecaa6425b352b16703a2c84a082"),
    ("semantic_channel_hybrid", "12336b592b73c17ce48c65550fb914d53a5e28f1cf683873957b018bb3291e10"),
])
def test_frozen_conv_state_matches_pre_edit_digest(strategy, expected):
    class Tiny(ClgnCifar10PaperSmall):
        k_num = 8
    torch.manual_seed(41)
    connections = dict(init_method=strategy, topology_seed=17, candidate_pool_size=8)
    if strategy == "semantic_multiscale_balanced":
        connections["classifier_init_method"] = strategy
    model = Tiny(thresholds=torch.tensor([.25, .5, .75]), binarization="fixed",
                 binarization_kwargs={}, connections_kwargs=connections, device="cpu")
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    assert digest.hexdigest() == expected


def test_queue_requires_nonempty_checkpoints_and_topology(tmp_path):
    for name in ["training_config.json", "environment.json", "metrics.csv", "run_summary.json"]:
        (tmp_path / name).write_text("{}")
    assert not is_complete(tmp_path)
    for name in ["best_checkpoint.pt", "final_checkpoint.pt", "topology.json"]:
        (tmp_path / name).write_bytes(b"artifact")
    assert is_complete(tmp_path)
    (tmp_path / "best_checkpoint.pt").write_bytes(b"")
    assert not is_complete(tmp_path)


def test_queue_accepts_learned_architecture_without_fixed_topology_report(tmp_path):
    for name in ["training_config.json", "environment.json", "metrics.csv", "best_checkpoint.pt", "final_checkpoint.pt"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "run_summary.json").write_text(json.dumps({"topology": [{"strategy": "learnable_topk"}]}))
    assert is_complete(tmp_path)


@pytest.mark.parametrize("bad_summary", ["null", "[]", '{"topology":42}', '{"topology":[null]}', "{"])
def test_queue_rejects_malformed_metadata(tmp_path, bad_summary):
    for name in ["training_config.json", "environment.json", "metrics.csv", "best_checkpoint.pt", "final_checkpoint.pt"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "run_summary.json").write_text(bad_summary)
    assert not is_complete(tmp_path)


def test_frozen_evaluator_rejects_missing_or_changed_hashes(tmp_path):
    from experiments.coverage_dlgn.evaluate_second_round_convolutional_final import verify_pending_artifacts
    artifacts = {}
    for name in ["training_config.json", "environment.json", "run_summary.json", "best_checkpoint.pt"]:
        (tmp_path / name).write_bytes(b"original")
        artifacts[name] = {"sha256": hashlib.sha256(b"original").hexdigest()}
    row = dict(run_dir=str(tmp_path), artifacts=artifacts)
    verify_pending_artifacts(row)
    with pytest.raises(RuntimeError, match="required artifact hashes"):
        verify_pending_artifacts(dict(row, artifacts={"environment.json": artifacts["environment.json"]}))
    (tmp_path / "best_checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_pending_artifacts(row)


def test_completed_frozen_evaluation_is_read_only(tmp_path, monkeypatch):
    from argparse import Namespace
    from experiments.coverage_dlgn import evaluate_second_round_convolutional_final as evaluator
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_metrics.json").write_text("{}")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(dict(validation_frozen=True, runs={"random": dict(run_dir=str(run_dir))})))
    logs = tmp_path / "logs"
    logs.mkdir()
    summary = logs / "test_evaluation_summary.json"
    summary.write_text("archived evidence")
    monkeypatch.setattr(evaluator, "FREEZE", freeze)
    monkeypatch.setattr(evaluator, "LOG_DIR", logs)
    monkeypatch.setattr(evaluator, "parse_args", lambda: Namespace(gpu=0, data_path=tmp_path))
    monkeypatch.setattr(evaluator, "evaluate_gpu", lambda *args: pytest.fail("must not access held-out data"))
    assert evaluator.main() == 0
    assert summary.read_text() == "archived evidence"
    assert not (logs / "started.json").exists()
