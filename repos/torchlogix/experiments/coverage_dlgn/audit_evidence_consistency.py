#!/usr/bin/env python3
"""Audit canonical result values, protocol labels, and deployment evidence."""

from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary"
TABLES = ROOT / "PAPER_COMPARISON_TABLES.md"
RESULTS = ROOT / "RESULTS.md"
CONCLUSIONS = ROOT / "SECOND_ROUND_CONCLUSIONS.md"
OUTPUT = SUMMARY / "evidence_consistency_audit.json"


def load(name: str) -> dict:
    return json.loads((SUMMARY / name).read_text())


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance


def latest_checks() -> dict[str, bool]:
    """Check latest aggregates against per-run artifacts, without dataset access."""
    try:
        from . import summarize_third_round as third
    except ImportError:
        import summarize_third_round as third
    frozen = load("third_round_validation_freeze.json")
    recorded = load("third_round_results.json")
    rows = [row for group in frozen["groups"].values() for row in group]
    rebuilt = []
    hash_matches = True
    for row in rows:
        row = dict(row)
        path = Path(row["run_dir"])
        if not path.is_absolute():
            path = ROOT.parents[1] / path
        row["run_dir"] = str(path)
        for name, expected in row["artifacts"].items():
            with (path / name).open("rb") as handle:
                hash_matches &= hashlib.file_digest(handle, "sha256").hexdigest() == expected["sha256"]
        rebuilt.append(third.collect_run(row))
    medium = load("cifar10_paper_medium_u2_200k_freeze.json")
    run = ROOT / medium["run_dir"]
    log = json.loads((ROOT / "logs/cifar10_paper_medium_u2_200k/test/test_evaluation_summary.json").read_text())
    metrics = json.loads((run / "test_metrics.json").read_text())
    medium_hashes = True
    for name, key in [("best_checkpoint.pt", "checkpoint_sha256"),
                      ("training_config.json", "training_config_sha256"),
                      ("environment.json", "environment_sha256"),
                      ("metrics.csv", "metrics_sha256")]:
        with (run / name).open("rb") as handle:
            medium_hashes &= hashlib.file_digest(handle, "sha256").hexdigest() == medium[key]
    return {
        "third_round_38_runs_76_checkpoints": len(rows) == 38 and frozen["checkpoint_count"] == 76,
        "third_round_frozen_hashes_match": bool(hash_matches),
        "third_round_per_run_records_match": rebuilt == recorded["runs"],
        "third_round_aggregates_match": third.aggregate_runs(rebuilt) == recorded["groups"],
        "third_round_paired_effects_match": third.paired_effects(rebuilt) == recorded["paired_effects"],
        "third_round_dense_cross_comparisons_match": third.current_dense_cross_comparisons() == recorded["current_dense_cross_comparisons"],
        "third_round_predeclared_queries": all(r["heldout_checkpoint_queries"] == 2 and r["test_checkpoint_sha256_matches_freeze"] for r in rebuilt),
        "third_round_test_absent_at_freeze": all(not r["test_metrics_existing_at_freeze"] for r in rows),
        "medium_u2_frozen_hashes_match": bool(medium_hashes),
        "medium_u2_test_hash_matches": hashlib.sha256((run / "test_metrics.json").read_bytes()).hexdigest() == log["test_metrics_sha256"],
        "medium_u2_frozen_before_query": not medium["test_set_used"] and medium["heldout_checkpoint_queries"] == 0 and log["heldout_checkpoint_queries"] == 1 and medium["frozen_at_utc"] < log["evaluated_at_utc"],
        "medium_u2_test_71_65": close(metrics["test_hard_accuracy"], .7165) and metrics["test_hard_accuracy"] == log["test_hard_accuracy"],
        "medium_u2_selected_step_136000": medium["selected_step"] == 136000 == metrics["validation_selection_step"],
    }


def main(output: Path | None = None) -> int:
    tables = TABLES.read_text()
    dense_s = load("paper_cifar10_semantic_v3.json")["test_hard_accuracy"]
    fashion_final = load("table1_fashion_final.json")
    conv = load("convolutional_evidence_snapshot.json")
    deployment = load("convolutional_deployment.json")
    medium = load("cifar10_paper_medium_200k_paired.json")
    second_status = load("second_round_status.json")
    second_freeze = load("second_round_convolutional_validation_freeze.json")
    second_final = load("second_round_convolutional_final.json")
    second_curves = load("second_round_convolutional_curves.json")
    second_deployment = load("second_round_convolutional_deployment.json")
    second_test_log = json.loads(
        (
            ROOT
            / "logs"
            / "second_round_convolutional_final_test"
            / "test_evaluation_summary.json"
        ).read_text()
    )
    results = RESULTS.read_text()
    conclusions = CONCLUSIONS.read_text()

    fashion = {row["family"]: row for row in fashion_final["methods"]}
    checks = {
        "dense_cifar10_s_random_test_49_056": close(
            100 * dense_s["random_mean"], 49.05599876642227
        ),
        "dense_cifar10_s_v3_test_52_358": close(
            100 * dense_s["hybrid_mean"], 52.35799872279168
        ),
        "dense_cifar10_s_gain_is_test_3_302": close(
            dense_s["mean_difference_percentage_points"], 3.3019999563694022
        ),
        "table_no_longer_labels_49_692_as_test": (
            "[REPRODUCED] 49.692%" not in tables
        ),
        "table_no_longer_labels_53_116_as_final": (
            "[OUR-FINAL] 53.116%" not in tables
        ),
        "fashion_final_random_test_86_308": close(
            100 * fashion["random"]["mean_test_hard_accuracy"],
            86.30799812078476,
        ),
        "fashion_final_v3_test_87_102": close(
            100 * fashion["coverage_v3"]["mean_test_hard_accuracy"],
            87.1019980430603,
        ),
        "paper_s_and_m_are_nine_channel": (
            conv["architecture_audit"]["paper_faithful_s"][
                "boolean_input_channels"
            ]
            == 9
            == conv["architecture_audit"]["paper_faithful_m"][
                "boolean_input_channels"
            ]
        ),
        "warp_medium_is_six_channel": (
            conv["architecture_audit"]["warp_style_medium"][
                "boolean_input_channels"
            ]
            == 6
        ),
        "u1_remains_validation_only": (
            conv["paper_faithful_nine_channel"]["small_20k"][
                "heldout_policy"
            ]
            == "Do not evaluate U1 or seeds 3-4 on test during this evidence pass."
        ),
        "medium_test_query_count_is_one": (
            medium["heldout_queries_per_checkpoint"] == 1
        ),
        "medium_v4_test_69_96": close(
            100 * medium["heldout_test"]["coverage_v4"]["test_hard_accuracy"],
            69.96,
        ),
        "medium_random_test_69_57": close(
            100 * medium["heldout_test"]["fixed_random"]["test_hard_accuracy"],
            69.57,
        ),
        "deployment_complete": deployment["status"] == "COMPLETE",
        "deployment_used_no_dataset": (
            deployment["input_policy"]
            == "synthetic thresholded Boolean inputs; no dataset access"
        ),
        "all_deployment_equivalence_passed": all(
            row["synthetic_equivalence_passed"]
            and not row["heldout_test_accessed"]
            and row["checkpoint_hash_matches_freeze"]
            and row["training_config_hash_matches_freeze"]
            for group in deployment["groups"].values()
            for row in group["runs"]
        ),
        "second_round_110_of_110_complete": (
            len(second_status["runs"]) == 110
            and all(row["status"] == "complete" for row in second_status["runs"])
        ),
        "second_round_test_absent_at_freeze": all(
            not row["test_metrics_existing_at_freeze"]
            for row in second_freeze["runs"].values()
        ),
        "second_round_test_evaluated_once_without_failure": (
            second_test_log["pending_at_start"] == 4
            and second_test_log["reused_existing"] == []
            and second_test_log["failures"] == []
            and second_test_log["missing_after"] == []
            and len(second_test_log["records"]) == 4
            and all(row["return_code"] == 0 for row in second_test_log["records"])
        ),
        "u2_full_validation_is_61_000": close(
            second_final["rows"]["unified_u2"]["best_hard_validation_pct"],
            61.0,
        ),
        "u2_full_test_is_60_630": close(
            second_final["rows"]["unified_u2"]["test_hard_pct"],
            60.63,
        ),
        "u2_full_test_gain_is_3_260": close(
            second_final["rows"]["unified_u2"]["test_gain_vs_random_pp"],
            3.26,
        ),
        "full_convolutional_declared_cost_is_identical": (
            len(
                {
                    json.dumps(row["cost"], sort_keys=True)
                    for row in second_final["rows"].values()
                }
            )
            == 1
        ),
        "u2_reaches_59_5_at_34k": (
            second_curves["summaries"]["unified_u2"][
                "first_step_at_hard_validation_pct"
            ]["59.5"]
            == 34000
        ),
        "second_round_deployment_complete": (
            second_deployment["status"] == "COMPLETE"
            and second_deployment["declared_cost_identical"]
            and all(
                row["synthetic_equivalence_passed"]
                and not row["heldout_test_accessed"]
                for row in second_deployment["rows"].values()
            )
        ),
        "u2_simplified_ir_delta_is_3_686_pct": close(
            second_deployment["rows"]["unified_u2"][
                "simplified_ir_delta_vs_random_pct"
            ],
            3.6863079988613823,
        ),
        "final_docs_contain_frozen_u2_result": all(
            "60.630%" in document and "+3.260 pp" in document
            for document in (tables, results, conclusions)
        ),
        "final_docs_have_no_u2_running_marker": all(
            "U2 running" not in document and "[RUNNING]" not in document
            for document in (tables, results, conclusions)
        ),
    }
    checks.update(latest_checks())
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed": failed,
        "scope": "Legacy/second-round checks plus third-round and M U2 artifacts; not every historical experiment",
        "limitations": ["Full-S legacy freeze lacks checkpoint-content hashes", "Historical full-S export equivalence uses one synthetic example", "No exact training resume or physical hardware claim"],
        "source_files": [
            "summary/paper_cifar10_semantic_v3.json",
            "summary/table1_fashion_final.json",
            "summary/convolutional_evidence_snapshot.json",
            "summary/convolutional_deployment.json",
            "summary/cifar10_paper_medium_200k_paired.json",
            "summary/second_round_status.json",
            "summary/second_round_convolutional_validation_freeze.json",
            "summary/second_round_convolutional_final.json",
            "summary/second_round_convolutional_curves.json",
            "summary/second_round_convolutional_deployment.json",
            "logs/second_round_convolutional_final_test/test_evaluation_summary.json",
            "PAPER_COMPARISON_TABLES.md",
            "RESULTS.md",
            "SECOND_ROUND_CONCLUSIONS.md",
            "summary/third_round_validation_freeze.json",
            "summary/third_round_results.json",
            "summary/cifar10_paper_medium_u2_200k_freeze.json",
            "logs/cifar10_paper_medium_u2_200k/test/test_evaluation_summary.json",
        ],
    }
    if output is not None:
        with output.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NEW report path; never overwrite archived evidence")
    raise SystemExit(main(parser.parse_args().output))
