from copy import deepcopy

import pytest

from experiments.coverage_dlgn.audit_checkpoint_metadata import (
    METRICS, check_metadata, write_certificate,
)


def sample():
    history = [dict(step=step, val_acc_discrete=hard, val_acc_relaxed=.7,
                    val_loss_discrete=1.1, val_loss_relaxed=1.)
               for step, hard in [(2000, .6), (4000, .6), (6000, .5)]]
    configuration = {"seed": 0, "parametrization": "warp", "num_iterations": 6000}
    metadata = dict(step=2000, configuration=deepcopy(configuration),
                    metrics={key: history[0][key] for key in METRICS})
    return metadata, configuration, history


def test_best_uses_first_maximum_and_final_uses_last_step():
    metadata, configuration, history = sample()
    assert check_metadata(metadata, configuration, history, "best")["step"] == 2000
    metadata["step"] = 4000
    with pytest.raises(RuntimeError, match="step mismatch"):
        check_metadata(metadata, configuration, history, "best")
    metadata["step"] = 6000
    metadata["metrics"] = {key: history[-1][key] for key in METRICS}
    assert check_metadata(metadata, configuration, history, "final")["step"] == 6000


@pytest.mark.parametrize("key", METRICS)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -100.])
def test_rejects_missing_nonfinite_or_changed_metrics(key, value):
    metadata, configuration, history = sample()
    metadata["metrics"][key] = value
    with pytest.raises(RuntimeError, match="metric mismatch"):
        check_metadata(metadata, configuration, history, "best")


def test_rejects_changed_configuration_or_missing_step():
    metadata, configuration, history = sample()
    metadata["configuration"]["seed"] = 1
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        check_metadata(metadata, configuration, history, "best")
    metadata.pop("step")
    with pytest.raises(RuntimeError, match="step mismatch"):
        check_metadata(metadata, configuration, history, "best")


def test_rejects_invalid_reference_or_checkpoint_kind():
    metadata, configuration, history = sample()
    for kind, records in [("best", []), ("unknown", history)]:
        with pytest.raises(RuntimeError, match="expected best/final"):
            check_metadata(metadata, configuration, records, kind)
    history[-1]["val_acc_discrete"] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        check_metadata(metadata, configuration, history, "best")


@pytest.mark.parametrize("status,pending", [("incomplete", []), ("pass", ["run"])])
def test_incomplete_report_cannot_be_certified(status, pending):
    with pytest.raises(RuntimeError, match="cannot certify incomplete"):
        write_certificate(dict(status=status, pending=pending))
