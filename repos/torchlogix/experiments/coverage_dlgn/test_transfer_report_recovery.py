"""Artifact-only regression tests; kept outside the frozen training test tree."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.coverage_dlgn.evaluate_frozen_transfer import REVISION, selected_runs
from experiments.coverage_dlgn.recover_transfer_report import summary_from_records


def fixture():
    rows = [dict(row, artifacts={"training_config.json": "a" * 64,
                                "best_checkpoint.pt": "b" * 64}) for row in selected_runs()]
    frozen = dict(dataset="CIFAR-10.1", version="v6", revision=REVISION, examples=2000, rows=rows)
    records = []
    for row in rows:
        count = 1000 + (10 if row["family"] == "u2" else 0)
        records.append(dict(row, predictions=[0] * 2000,
            correct=[True] * count + [False] * (2000 - count),
            scope="T", provenance="OUR" if row["family"] == "u2" else "REPRODUCED",
            examples=2000, hard_accuracy_pct=100 * count / 2000))
    return frozen, records


def test_recovers_numpy_win_serialization_failure_without_changing_records():
    # Reproduce the frozen evaluator's failure, then verify the pure-Python fix.
    with pytest.raises(TypeError):
        json.dumps({"wins": sum(x > 0 for x in [np.float64(0.5)] * 3)})
    frozen, records = fixture()
    before = deepcopy(records)
    result = summary_from_records(frozen, records, "c" * 64)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert records == before
    assert result["checkpoint_queries"] == 10
    assert result["comparisons"][0]["wins"] == 3
    assert result["comparisons"][0]["paired_effect"]["mean"] == .5
    assert result["comparisons"][1]["paired_effect"]["sample_sd"] is None


@pytest.mark.parametrize("mutation", ["missing", "reordered", "accuracy", "prediction", "correct", "hash"])
def test_rejects_invalid_saved_evidence(mutation):
    frozen, records = fixture()
    if mutation == "missing":
        records.pop()
    elif mutation == "reordered":
        records.reverse()
    elif mutation == "accuracy":
        records[0]["hard_accuracy_pct"] += 1
    elif mutation == "prediction":
        records[0]["predictions"][0] = 10
    elif mutation == "correct":
        records[0]["correct"][0] = 1
    else:
        records[0]["artifacts"] = {"training_config.json": "d" * 64, "best_checkpoint.pt": "b" * 64}
    with pytest.raises(RuntimeError):
        summary_from_records(frozen, records, "c" * 64)
