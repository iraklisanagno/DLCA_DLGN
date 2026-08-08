#!/usr/bin/env python3
"""Audit canonical result values, protocol labels, and deployment evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary"
TABLES = ROOT / "PAPER_COMPARISON_TABLES.md"
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
            "PAPER_COMPARISON_TABLES.md",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
