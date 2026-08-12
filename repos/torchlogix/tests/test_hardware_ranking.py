import json
from pathlib import Path

import pytest
import torch

from experiments.marginsynth.calibrate_hardware_ranking import fit_coefficients
from experiments.marginsynth.circuit_distillation import (
    alternative_binary_action_penalties,
    changed_lut_records,
)
from experiments.marginsynth.hardware_ranking import (
    StructuralFeatureIndex,
    StructuralHardwareModel,
    cofactor_lut_id,
    constant_propagation_gains,
)
from experiments.marginsynth.select_feasible_snapshot import select_candidate
from torchlogix.layers import LogicDense


def dense(in_dim, out_dim, indices):
    layer = LogicDense(
        in_dim=in_dim,
        out_dim=out_dim,
        lut_rank=2,
        parametrization="raw",
        connections="fixed",
        connections_kwargs={"init_method": "random"},
    )
    layer.connections.indices.copy_(torch.tensor(indices))
    return layer


def model():
    return StructuralHardwareModel(
        coefficients={
            "operation_gain": 1.0,
            "constant_propagation_gain": 1.0,
            "fanout_log": 0.0,
            "downstream_influence_log": 0.0,
        },
        alternative_binary_penalty=2.0,
    )


def test_boolean_cofactors_use_repository_lut_order():
    assert cofactor_lut_id(1, 0, 0) == 0  # 0 AND b
    assert cofactor_lut_id(1, 0, 1) == 5  # 1 AND b == b
    assert cofactor_lut_id(7, 1, 0) == 3  # a OR 0 == a
    assert cofactor_lut_id(7, 1, 1) == 15  # a OR 1


def test_constant_propagation_gain_follows_a_downstream_cascade():
    first = dense(2, 2, [[0, 0], [1, 1]])
    second = dense(2, 2, [[0, 1], [1, 1]])
    third = dense(2, 1, [[0], [1]])
    ids = {
        0: torch.tensor([6, 6]),
        1: torch.tensor([1, 3]),  # AND(first[0], first[1])
        2: torch.tensor([7]),  # OR(second[0], second[1])
    }
    gains = constant_propagation_gains([first, second, third], ids)
    assert gains[0][0, 0].item() == 2.0
    assert gains[0][0, 1].item() == 1.0


def test_alternative_binary_penalty_exempts_original_binary_lut():
    penalties = alternative_binary_action_penalties(
        {0: torch.tensor([6, 1])}, [0], 2.0, torch.device("cpu")
    )
    assert penalties[0][0, 6] == 0.0
    assert penalties[0][0, 7] == 2.0
    assert penalties[0][0, 0] == 0.0
    assert penalties[0][1, 1] == 0.0


def test_hardware_ranking_prioritizes_larger_operation_gain():
    layer = dense(2, 2, [[0, 0], [1, 1]])
    original = {0: torch.tensor([6, 1])}
    with torch.no_grad():
        layer.weight.fill_(-10.0)
        layer.weight[:, 0] = torch.tensor([1.0, 2.0])
        layer.weight[:, [6, 1]] = 0.0
    index = StructuralFeatureIndex([layer], original)
    records = changed_lut_records(
        [layer],
        [0],
        original,
        "gate-count",
        activity_ranking="hardware",
        structural_index=index,
        hardware_model=model(),
    )
    assert [record["unit"] for record in records] == [0, 1]
    assert records[0]["estimated_hardware_gain"] > records[1][
        "estimated_hardware_gain"
    ]


def test_class_hardware_ranking_can_protect_lower_risk_edit():
    layer = dense(2, 2, [[0, 0], [1, 1]])
    original = {0: torch.tensor([6, 1])}
    with torch.no_grad():
        layer.weight.fill_(-10.0)
        layer.weight[:, 0] = 2.0
        layer.weight[:, [6, 1]] = 0.0
    risks = {0: torch.zeros(2, 16)}
    risks[0][0, 0] = 1.0
    risks[0][1, 0] = 0.0
    records = changed_lut_records(
        [layer],
        [0],
        original,
        "gate-count",
        activity_risks=risks,
        activity_ranking="class-fold-hardware",
        structural_index=StructuralFeatureIndex([layer], original),
        hardware_model=model(),
        hardware_rank_weight=0.25,
        activity_rank_weight=0.75,
    )
    assert [record["unit"] for record in records] == [1, 0]


def test_snapshot_selector_falls_back_when_second_pass_breaks_guard():
    candidates = [
        {
            "stage": "source",
            "repair_feasible": True,
            "calibration_feasible": True,
            "guard_feasible": True,
            "estimated_hardware_gain": 0.0,
            "cumulative_retained_changes": 0,
        },
        {
            "stage": "first",
            "repair_feasible": True,
            "calibration_feasible": True,
            "guard_feasible": True,
            "estimated_hardware_gain": 10.0,
            "cumulative_retained_changes": 100,
        },
        {
            "stage": "second",
            "repair_feasible": True,
            "calibration_feasible": True,
            "guard_feasible": False,
            "estimated_hardware_gain": 20.0,
            "cumulative_retained_changes": 200,
        },
    ]
    assert select_candidate(candidates)["stage"] == "first"


def test_nonnegative_hardware_fit_recovers_synthetic_coefficients():
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=torch.float64,
    ).numpy()
    y = x @ torch.tensor([2.0, 3.0], dtype=torch.float64).numpy()
    fitted = fit_coefficients(x, y, ridge=0.0)
    assert fitted.tolist() == pytest.approx([2.0, 3.0])


def test_main_baseline_is_held_out_from_hardware_coefficient_fit():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/marginsynth/configs/hardware_ranking_calibration_cifar_seed0.json"
        ).read_text()
    )
    unit = next(
        sample for sample in config["samples"] if sample["name"] == "unit_tying_10pct"
    )
    assert unit["fit_sample"] is False
    assert config["data_policy"]["unit_tying_used_for_fit"] is False
