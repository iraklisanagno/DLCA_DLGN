#!/usr/bin/env python3
"""Aggregate full-schedule convolutional S export and CPU measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPLOYMENT = ROOT / "summary" / "deployment"
FREEZE = ROOT / "summary" / "second_round_convolutional_validation_freeze.json"
OUTPUT = ROOT / "summary" / "second_round_convolutional_deployment.json"

LABELS = {
    "random": "paper_s_random_full",
    "legacy_v4": "paper_s_v4_full",
    "unified_u1": "paper_s_u1_full",
    "unified_u2": "paper_s_u2_full",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    freeze = json.loads(FREEZE.read_text())
    rows = {}
    for method, label in LABELS.items():
        record_path = DEPLOYMENT / f"{label}.json"
        record = json.loads(record_path.read_text())
        if record["status"] != "COMPLETE":
            raise RuntimeError(f"incomplete deployment result: {record_path}")
        if not all(record["equivalence"].values()):
            raise RuntimeError(f"equivalence failed: {record_path}")
        frozen = freeze["runs"][method]
        run_dir = Path(frozen["run_dir"])
        if record["checkpoint_sha256"] != sha256(run_dir / "best_checkpoint.pt"):
            raise RuntimeError(f"checkpoint changed after freeze: {method}")
        if record["training_config_sha256"] != sha256(
            run_dir / "training_config.json"
        ):
            raise RuntimeError(f"config changed after freeze: {method}")
        benchmark = record["compiled_cpu_benchmark"]
        gpu_benchmark = json.loads(
            (run_dir / "synthetic_inference_benchmark_v2.json").read_text()
        )
        if gpu_benchmark["benchmark_implementation_version"] != 2:
            raise RuntimeError(f"obsolete GPU benchmark version: {method}")
        if gpu_benchmark["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise RuntimeError(f"GPU benchmark checkpoint mismatch: {method}")
        if gpu_benchmark["heldout_test_accessed"]:
            raise RuntimeError(f"GPU benchmark accessed held-out data: {method}")
        rows[method] = {
            "ir_logic_nodes_before_simplification": record[
                "circuit_before_simplification"
            ]["logic_gates"],
            "ir_logic_nodes_after_simplification": record[
                "circuit_after_simplification"
            ]["logic_gates"],
            "peak_export_rss_gib": record["final_peak_rss_bytes"] / 2**30,
            "compiled_cpu_batch128_ms": benchmark[
                "latency_milliseconds_per_batch_mean"
            ],
            "compiled_cpu_examples_per_second": benchmark[
                "examples_per_second"
            ],
            "synthetic_gpu_batch128_ms": gpu_benchmark[
                "milliseconds_per_batch"
            ],
            "synthetic_gpu_examples_per_second": gpu_benchmark[
                "examples_per_second"
            ],
            "synthetic_gpu_peak_memory_gib": gpu_benchmark[
                "peak_device_memory_bytes"
            ] / 2**30,
            "synthetic_equivalence_passed": True,
            "heldout_test_accessed": False,
        }
    reference = rows["random"]
    for row in rows.values():
        row["simplified_ir_delta_vs_random_pct"] = 100 * (
            row["ir_logic_nodes_after_simplification"]
            / reference["ir_logic_nodes_after_simplification"]
            - 1
        )
        row["latency_delta_vs_random_pct"] = 100 * (
            row["compiled_cpu_batch128_ms"]
            / reference["compiled_cpu_batch128_ms"]
            - 1
        )
        row["synthetic_gpu_latency_delta_vs_random_pct"] = 100 * (
            row["synthetic_gpu_batch128_ms"]
            / reference["synthetic_gpu_batch128_ms"]
            - 1
        )
    payload = {
        "status": "COMPLETE",
        "protocol": "paper-faithful LogicTreeNet-S, 9 Boolean channels, 350K",
        "input_policy": (
            "deterministic synthetic Boolean tensors for circuit/CPU and "
            "seeded synthetic RGB tensors for hardened GPU inference"
        ),
        "declared_cost_identical": True,
        "declared_cost": {
            "learned_gate_functions": 83552,
            "classifier_gate_functions": 71680,
            "spatial_gate_applications": 874496,
            "trainable_gate_parameters": 1336832,
            "trainable_routing_parameters": 0,
            "routing_bits": 1945600,
            "packed_routing_bytes": 243200,
        },
        "rows": rows,
        "limitations": [
            "Single checkpoint per method; latency is a deployment snapshot.",
            "Generated C uses gcc -O0 and includes Boolean input packing.",
            "Sub-percent GPU latency differences are treated as measurement noise.",
            "Energy was not measured.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
