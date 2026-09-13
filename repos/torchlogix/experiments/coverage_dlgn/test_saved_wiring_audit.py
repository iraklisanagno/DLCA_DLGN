from copy import deepcopy

import pytest
import torch

from experiments.coverage_dlgn.audit_saved_wiring import check_factorial, fingerprints


def arms():
    return {name: dict(body=b, head=h, spatial="spatial") for name, b, h in [
        ("random", "rbody", "rhead"), ("body", "ubody", "rhead"),
        ("head", "rbody", "uhead"), ("both", "ubody", "uhead")
    ]}


def test_independent_factorial_passes():
    assert all(check_factorial(arms()).values())


@pytest.mark.parametrize("family,field", [("body", "head"), ("both", "head"),
                                         ("head", "body"), ("both", "body"), ("random", "spatial")])
def test_rejects_a_confounded_component(family, field):
    values = arms()
    values[family][field] = "different"
    with pytest.raises(RuntimeError):
        check_factorial(values)


def test_missing_or_duplicate_treatment_is_not_independent_evidence():
    values = arms()
    values.pop("head")
    with pytest.raises(RuntimeError):
        check_factorial(values)
    values = arms()
    for row in values.values():
        row["head"] = "same"
    with pytest.raises(RuntimeError):
        check_factorial(values)


def test_fingerprints_separate_channel_spatial_and_classifier_coordinates():
    state = {}
    for layer in [1, 3, 5, 7]:
        for depth in range(3):
            state[f"{layer}.connections._indices_L{depth}"] = torch.zeros((2, 2, 3), dtype=torch.int64)
    for layer in [10, 11, 12]:
        state[f"{layer}.connections.indices"] = torch.zeros((2, 3), dtype=torch.int64)
    baseline = fingerprints(state)
    changed = deepcopy(state)
    changed["1.connections._indices_L0"][..., -1] += 1
    result = fingerprints(changed)
    assert result["body"] != baseline["body"]
    assert result["head"] == baseline["head"]
    assert result["spatial"] == baseline["spatial"]
    changed["1.connections._indices_L0"][..., 0] += 1
    assert fingerprints(changed)["spatial"] != baseline["spatial"]
    changed = deepcopy(state)
    changed["10.connections.indices"] += 1
    result = fingerprints(changed)
    assert result["head"] != baseline["head"]
    assert result["body"] == baseline["body"]
    assert result["spatial"] == baseline["spatial"]
