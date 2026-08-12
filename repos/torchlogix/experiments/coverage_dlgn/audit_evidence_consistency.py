#!/usr/bin/env python3
"""Audit canonical result values, protocol labels, and deployment evidence."""

from __future__ import annotations

import json
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


def main() -> int:
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
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed": failed,
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
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
