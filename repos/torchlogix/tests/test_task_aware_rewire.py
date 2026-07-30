import numpy as np
import torch

from experiments.train import get_parser, model_cost_summary
from torchlogix import Circuit
from torchlogix.layers import GroupSum, LogicDense
from torchlogix.models.dense import Dlgn
from torchlogix.task_aware import (
    TaskSignatureCollector,
    rewire_fixed_dense_model,
    task_aware_degree_preserving_refine,
)
from torchlogix.utils import set_export_mode


def test_task_aware_refine_is_deterministic_and_degree_preserving():
    indices = np.asarray([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 0],
    ])
    input_signatures = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.3, 0.7],
    ])
    output_signatures = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    first = task_aware_degree_preserving_refine(
        indices,
        input_signatures,
        output_signatures,
        topology_seed=11,
        change_fraction=1.0,
        candidate_pool_size=7,
    )
    second = task_aware_degree_preserving_refine(
        indices,
        input_signatures,
        output_signatures,
        topology_seed=11,
        change_fraction=1.0,
        candidate_pool_size=7,
    )
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.changed_mask, second.changed_mask)
    assert np.array_equal(
        np.bincount(indices.reshape(-1), minlength=8),
        np.bincount(first.indices.reshape(-1), minlength=8),
    )
    assert np.all(first.indices[0] != first.indices[1])
    assert first.score_improvement >= 0.0


def _tiny_v3():
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
        connections_kwargs={
            "init_method": "semantic_balanced_hybrid",
            "topology_seed": 7,
            "candidate_pool_size": 8,
            "swap_fraction": 0.25,
            "novelty_weight": 1.0,
        },
        parametrization="raw",
        parametrization_kwargs={"weight_init": "random"},
        device="cpu",
        lut_rank=2,
    )


def test_task_signature_collection_and_model_rewire_keep_weights_and_cost():
    torch.manual_seed(19)
    model = _tiny_v3()
    before_weights = [
        layer.weight.detach().clone()
        for layer in model if isinstance(layer, LogicDense)
    ]
    before_cost = model_cost_summary(model)
    before_degrees = [
        torch.bincount(
            layer.connections.indices.flatten(), minlength=layer.in_dim
        )
        for layer in model if isinstance(layer, LogicDense)
    ]
    x = torch.rand(12, 16)
    labels = torch.arange(12) % 4
    with TaskSignatureCollector(model) as collector:
        predictions = model(x)
        torch.nn.functional.cross_entropy(predictions, labels).backward()
        signatures = collector.signatures(labels, 4)
    report = rewire_fixed_dense_model(
        model,
        signatures,
        topology_seed=7,
        change_fraction=0.5,
        candidate_pool_size=8,
        diversity_weight=0.25,
    )
    layers = [layer for layer in model if isinstance(layer, LogicDense)]
    assert len(report) == len(layers)
    assert all(item["predecessor_degree_preserved"] for item in report)
    assert all(
        torch.equal(before, layer.weight)
        for before, layer in zip(before_weights, layers)
    )
    assert all(
        torch.equal(
            before,
            torch.bincount(
                layer.connections.indices.flatten(), minlength=layer.in_dim
            ),
        )
        for before, layer in zip(before_degrees, layers)
    )
    assert model_cost_summary(model) == before_cost


def test_task_aware_training_event_is_disabled_by_default():
    args = get_parser().parse_args([])
    assert args.task_aware_rewire_step is None
    assert args.task_aware_rewire_fraction == 0.125


def test_task_aware_rewired_model_exports_to_equivalent_circuit():
    torch.manual_seed(29)
    model = torch.nn.Sequential(
        LogicDense(
            16,
            32,
            parametrization="raw",
            parametrization_kwargs={"weight_init": "random"},
            connections_kwargs={
                "init_method": "semantic_balanced_hybrid",
                "topology_seed": 7,
                "candidate_pool_size": 8,
                "swap_fraction": 0.25,
            },
        ),
        LogicDense(
            32,
            32,
            parametrization="raw",
            parametrization_kwargs={"weight_init": "random"},
            connections_kwargs={
                "init_method": "semantic_balanced_hybrid",
                "topology_seed": 7,
                "layer_index": 1,
                "candidate_pool_size": 8,
                "swap_fraction": 0.25,
                "output_groups": 4,
            },
        ),
        GroupSum(4),
    )
    calibration = torch.randint(0, 2, (12, 16), dtype=torch.float32)
    labels = torch.arange(12) % 4
    with TaskSignatureCollector(model) as collector:
        predictions = model(calibration)
        torch.nn.functional.cross_entropy(predictions, labels).backward()
        signatures = collector.signatures(labels, 4)
    rewire_fixed_dense_model(
        model,
        signatures,
        topology_seed=7,
        change_fraction=0.5,
        candidate_pool_size=8,
        diversity_weight=0.25,
    )
    set_export_mode(model)
    inputs = torch.randint(0, 2, (3, 16), dtype=torch.bool)
    model_predictions = model(inputs)
    circuit = Circuit.from_model(model, input_shape=(16,))
    circuit_predictions = circuit(inputs)
    assert torch.equal(
        model_predictions,
        circuit_predictions.to(model_predictions.dtype),
    )
