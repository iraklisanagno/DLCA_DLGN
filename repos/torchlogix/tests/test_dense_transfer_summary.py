import json

import pytest

from experiments.marginsynth.summarize_dense_transfer import aggregate


def _record(method, *, accuracy, disagreement, nodes, live, total, feasible):
    exact_nodes = 1_000
    exact_live = 500
    return {
        "method": method,
        "guard_feasible": feasible,
        "validation_accuracy": accuracy,
        "validation_accuracy_loss": 0.8 - accuracy,
        "validation_disagreement": disagreement,
        "live_gates": live,
        "live_gate_reduction_vs_exact": 1.0 - live / exact_live,
        "abc_and_nodes": nodes,
        "abc_node_reduction_vs_exact": 1.0 - nodes / exact_nodes,
        "abc_levels": 10,
        "optimization_seconds": total,
        "method_total_seconds": total,
        "checkpoint_sha256": f"checkpoint-{method}",
        "test_used": False,
    }


def _comparison(path, seed, *, test_used=False):
    payload = {
        "status": "completed",
        "protocol_name": f"dense_cifar_seed{seed}_v1",
        "dataset": "cifar-10",
        "architecture": {"widths": [128, 128]},
        "records": [
            _record(
                "exact_simplification",
                accuracy=0.8,
                disagreement=0.0,
                nodes=1_000,
                live=500,
                total=0.0,
                feasible=True,
            ),
            _record(
                "unit_tying_10pct",
                accuracy=0.79 - 0.01 * seed,
                disagreement=0.04,
                nodes=900,
                live=450,
                total=2.0,
                feasible=None,
            ),
            _record(
                "current",
                accuracy=0.795 - 0.01 * seed,
                disagreement=0.02,
                nodes=950,
                live=475,
                total=20.0,
                feasible=True,
            ),
        ],
        "test_used": test_used,
    }
    path.write_text(json.dumps(payload))
    return path


def test_aggregate_dense_transfer_reports_paired_deltas(tmp_path):
    paths = [_comparison(tmp_path / f"seed{seed}.json", seed) for seed in range(3)]

    payload, rows = aggregate(paths)

    assert payload["seeds"] == [0, 1, 2]
    assert payload["test_used"] is False
    assert len(rows) == 9
    assert payload["aggregates"]["current"]["all_guard_feasible"] is True
    assert payload["aggregates"]["unit_tying_10pct"][
        "guard_feasibility_applicable"
    ] is False
    assert payload["aggregates"]["unit_tying_10pct"]["all_guard_feasible"] is None
    assert payload["paired_current_vs_unit_tying"][
        "validation_accuracy_delta_current_minus_unit_tying"
    ]["mean"] == pytest.approx(0.005)
    assert payload["paired_current_vs_unit_tying"][
        "validation_disagreement_delta_current_minus_unit_tying"
    ]["mean"] == pytest.approx(-0.02)
    assert payload["paired_current_vs_unit_tying"][
        "abc_node_reduction_delta_current_minus_unit_tying"
    ]["mean"] == pytest.approx(-0.05)
    assert payload["paired_current_vs_unit_tying"][
        "method_total_time_ratio_current_over_unit_tying"
    ]["mean"] == pytest.approx(10.0)


def test_aggregate_dense_transfer_rejects_test_access(tmp_path):
    paths = [
        _comparison(tmp_path / "seed0.json", 0, test_used=True),
        _comparison(tmp_path / "seed1.json", 1),
    ]

    with pytest.raises(ValueError, match="test data sealed"):
        aggregate(paths)
