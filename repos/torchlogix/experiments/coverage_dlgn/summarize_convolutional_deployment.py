#!/usr/bin/env python3
"""Aggregate synthetic circuit-export and deployment measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPLOYMENT = ROOT / "summary" / "deployment"
OUTPUT = ROOT / "summary" / "convolutional_deployment.json"
CSV_OUTPUT = ROOT / "summary" / "convolutional_deployment.csv"
MEDIUM = ROOT / "summary" / "cifar10_paper_medium_200k_paired.json"
FREEZE = ROOT / "summary" / "convolutional_evidence_freeze.json"

GROUPS = {
    "paper_faithful_s_9ch": ["paper_s_random", "paper_s_v4", "paper_s_u1"],
    "paper_faithful_m_9ch": ["paper_m_random", "paper_m_v4"],
    "warp_style_m_6ch": ["warp_m_random", "warp_m_v4"],
}

DECLARED_COST = {
    "paper_faithful_s_9ch": {
        "learned_gate_functions": 83552,
        "spatial_gate_applications": 874496,
        "paper_reported_gate_operations_approximate": 400000,
        "trainable_gate_parameters": 1336832,
        "trainable_routing_parameters": 0,
        "routing_bits": 1945600,
        "packed_routing_bytes": 243200,
    },
    "paper_faithful_m_9ch": {
        "learned_gate_functions": 668416,
        "spatial_gate_applications": 6995968,
        "paper_reported_gate_operations_approximate": 3080000,
        "trainable_gate_parameters": 10694656,
        "trainable_routing_parameters": 0,
        "routing_bits": 19005440,
        "packed_routing_bytes": 2375680,
    },
    "warp_style_m_6ch": {
        "learned_gate_functions": 668416,
        "spatial_gate_applications": 6995968,
        "trainable_gate_parameters": 10694656,
        "trainable_routing_parameters": 0,
        "note": "Separate six-channel protocol; do not compare its accuracy directly with the nine-channel M protocol.",
    },
}


def load(name: str) -> dict:
    payload = json.loads((DEPLOYMENT / f"{name}.json").read_text())
    if payload["status"] != "COMPLETE":
        raise RuntimeError(f"incomplete deployment result: {name}")
    if not all(payload["equivalence"].values()):
        raise RuntimeError(f"equivalence failure: {name}")
    return payload


def delta_percent(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def main() -> int:
    records = {name: load(name) for names in GROUPS.values() for name in names}
    medium = json.loads(MEDIUM.read_text())
    freeze = json.loads(FREEZE.read_text())
    rows = []
    groups = {}
    for group, names in GROUPS.items():
        reference = records[names[0]]
        group_rows = []
        for name in names:
            record = records[name]
            frozen = freeze["runs"][record["run_name"]]["files"]
            checkpoint_matches = (
                record["checkpoint_sha256"]
                == frozen["best_checkpoint.pt"]["sha256"]
            )
            config_matches = (
                record["training_config_sha256"]
                == frozen["training_config.json"]["sha256"]
            )
            if not checkpoint_matches or not config_matches:
                raise RuntimeError(f"deployment input changed after freeze: {name}")
            before = record["circuit_before_simplification"]["logic_gates"]
            after = record["circuit_after_simplification"]["logic_gates"]
            benchmark = record.get("compiled_cpu_benchmark")
            row = {
                "protocol": group,
                "run_label": name,
                "architecture": record["architecture"],
                "boolean_input_channels": record["boolean_input_channels"],
                "thresholds_per_rgb_channel": record[
                    "thresholds_per_rgb_channel"
                ],
                "ir_logic_nodes_before_simplification": before,
                "ir_logic_nodes_after_simplification": after,
                "ir_reduction_percent": 100.0 * (1.0 - after / before),
                "simplified_ir_delta_vs_reference_percent": delta_percent(
                    after,
                    reference["circuit_after_simplification"]["logic_gates"],
                ),
                "trace_seconds": record["trace_seconds"],
                "simplification_seconds": record.get("simplification_seconds"),
                "peak_process_rss_bytes": record["final_peak_rss_bytes"],
                "compiled": record.get("compile", {}).get("completed", False),
                "compiled_latency_ms_batch128": (
                    benchmark["latency_milliseconds_per_batch_mean"]
                    if benchmark
                    else None
                ),
                "compiled_examples_per_second": (
                    benchmark["examples_per_second"] if benchmark else None
                ),
                "compiled_latency_delta_vs_reference_percent": (
                    delta_percent(
                        benchmark["latency_milliseconds_per_batch_mean"],
                        reference["compiled_cpu_benchmark"][
                            "latency_milliseconds_per_batch_mean"
                        ],
                    )
                    if benchmark
                    else None
                ),
                "synthetic_equivalence_passed": True,
                "heldout_test_accessed": False,
                "checkpoint_hash_matches_freeze": checkpoint_matches,
                "training_config_hash_matches_freeze": config_matches,
            }
            rows.append(row)
            group_rows.append(row)
        groups[group] = {
            "declared_matched_cost": DECLARED_COST[group],
            "runs": group_rows,
            "declared_cost_identical_within_group": True,
        }

    groups["paper_faithful_m_9ch"]["framework_gpu_inference"] = medium[
        "hardened_inference"
    ]
    payload = {
        "status": "COMPLETE",
        "input_policy": "synthetic thresholded Boolean inputs; no dataset access",
        "groups": groups,
        "definitions": {
            "learned_gate_functions": (
                "Architecture-level LUT units with trainable truth-table parameters."
            ),
            "spatial_gate_applications": (
                "Architecture-level logic applications after spatial unrolling, before "
                "learning-dependent constant/wire simplification."
            ),
            "ir_logic_nodes": (
                "Checkpoint-dependent nontrivial nodes emitted by TorchLogix Circuit; "
                "these may differ across methods even at an identical declared budget."
            ),
        },
        "limitations": [
            "Compiled CPU latency is available for paper-faithful S only.",
            "S compilation used gcc -O0 because an initial -O1 compile remained unfinished after 8.75 minutes.",
            "M is trace/equivalence/IR-size only; fully unrolled M compilation was not attempted after the S feasibility result.",
            "Latency measurements are one process-level pass and do not establish a speed advantage.",
            "Energy was not measured.",
        ],
        "compile_attempt_history": "summary/deployment/compile_attempt_history.json",
        "checkpoint_freeze_manifest": "summary/convolutional_evidence_freeze.json",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with CSV_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)
    print(CSV_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
