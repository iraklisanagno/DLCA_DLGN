#!/usr/bin/env python3
"""Verify a MarginSynth circuit through integer RTL, Yosys, and ABC.

Only validation examples are used. The calibration partition remains reserved
for MarginSynth rewrite selection, and the held-out test partition is sealed.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model, load_dataset
try:
    from .verify_checkpoint import (
        git_revision,
        sha256_file,
        take_examples,
        tensor_sha256,
        write_artifact_manifest,
    )
except ImportError:
    from verify_checkpoint import (
        git_revision,
        sha256_file,
        take_examples,
        tensor_sha256,
        write_artifact_manifest,
    )

from torchlogix import Circuit
from torchlogix.circuit import _c_output_dtype


def command_version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def parse_yosys_cells(log: str) -> dict[str, int]:
    matches = re.findall(
        r"Number of cells:\s+(\d+)(.*?)(?=\n\n|\Z)",
        log,
        flags=re.DOTALL,
    )
    if not matches:
        return {}
    total, block = matches[-1]
    cells = {"total": int(total)}
    for name, count in re.findall(r"^\s+(\S+)\s+(\d+)\s*$", block, re.MULTILINE):
        cells[name] = int(count)
    return cells


def parse_abc_stats(log: str) -> dict[str, int]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
    matches = re.findall(
        r"i/o\s*=\s*(\d+)/\s*(\d+).*?"
        r"lat\s*=\s*(\d+).*?and\s*=\s*(\d+).*?lev\s*=\s*(\d+)",
        clean,
    )
    if not matches:
        return {}
    inputs, outputs, latches, and_nodes, levels = matches[-1]
    return {
        "inputs": int(inputs),
        "outputs": int(outputs),
        "latches": int(latches),
        "and_nodes": int(and_nodes),
        "levels": int(levels),
    }


def integer_score_predictions(scores: torch.Tensor) -> torch.Tensor:
    """Compute argmax for unsigned compiled scores unsupported by PyTorch CPU."""
    if scores.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        raise TypeError("compiled hardware scores must use an unsigned integer dtype")
    return scores.to(torch.int64).argmax(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument(
        "--pack-bits",
        type=int,
        choices=[8, 16, 32, 64],
        default=16,
    )
    parser.add_argument(
        "--compile-opt-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="GCC optimization level for semantic-equivalence builds",
    )
    parser.add_argument(
        "--verification-split",
        choices=["validation", "calibration"],
        default="validation",
        help="Held-out split used only for semantic-equivalence checks",
    )
    cli = parser.parse_args()
    if cli.examples <= 0 or cli.examples % cli.pack_bits:
        raise ValueError("--examples must be positive and divisible by --pack-bits")

    run_dir = cli.run_dir.resolve()
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"

    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _, validation_loader, calibration_loader, _ = load_dataset(
        args, include_calibration=True
    )
    verification_loader = (
        validation_loader
        if cli.verification_split == "validation"
        else calibration_loader
    )
    images, labels = take_examples(verification_loader, cli.examples)
    with torch.no_grad():
        encoded_inputs = model[0](images).bool()

    original_path = run_dir / "exact_simplified_circuit.json"
    original = Circuit.from_json_file(str(original_path))
    original_reductions = [
        original._sum_by_id[output_id] for output_id in original.outputs
    ]
    original_taus = [node.tau for node in original_reductions]
    original_betas = [node.beta for node in original_reductions]
    common_tau = original_taus[0]
    common_offset = min(original_betas)

    hardware = original.normalized_for_hardware_argmax()
    hardware_reductions = [
        hardware._sum_by_id[output_id] for output_id in hardware.outputs
    ]
    score_bits = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
    }[_c_output_dtype(hardware_reductions)]
    hardware_json_path = run_dir / "hardware_argmax_circuit.json"
    hardware_verilog_path = run_dir / "hardware_argmax_circuit.v"
    hardware_blif_path = run_dir / "hardware_argmax_circuit.blif"
    hardware.write_json(str(hardware_json_path))
    hardware.write_verilog_code(str(hardware_verilog_path))

    start = time.perf_counter()
    original.compile(
        opt_level=cli.compile_opt_level,
        pack_bits=cli.pack_bits,
    )
    original_scores = torch.from_numpy(
        original(encoded_inputs.numpy(), use_compiled=True)
    )
    hardware.compile(
        opt_level=cli.compile_opt_level,
        pack_bits=cli.pack_bits,
    )
    hardware_scores = torch.from_numpy(
        hardware(encoded_inputs.numpy(), use_compiled=True)
    )
    compiled_equivalence_seconds = time.perf_counter() - start

    expected_hardware_scores = original_scores * common_tau - common_offset
    maximum_transformation_difference = float(
        (expected_hardware_scores - hardware_scores).abs().max().item()
    )
    predictions_exact = bool(
        torch.equal(
            original_scores.argmax(dim=-1),
            integer_score_predictions(hardware_scores),
        )
    )

    yosys = shutil.which("yosys")
    abc = shutil.which("berkeley-abc") or shutil.which("abc")
    if yosys is None:
        raise RuntimeError("Yosys is required for synthesis verification")
    if abc is None:
        raise RuntimeError("Berkeley ABC is required for synthesis verification")

    yosys_script = (
        f"read_verilog -sv {hardware_verilog_path}; "
        "synth -top circuit -noabc; check; "
        f"write_blif {hardware_blif_path}"
    )
    start = time.perf_counter()
    yosys_result = subprocess.run(
        [yosys, "-p", yosys_script],
        capture_output=True,
        text=True,
        check=False,
    )
    yosys_seconds = time.perf_counter() - start
    yosys_log = yosys_result.stdout + yosys_result.stderr
    yosys_log_path = run_dir / "yosys_synthesis.log"
    yosys_log_path.write_text(yosys_log)

    abc_script = (
        f"read_blif {hardware_blif_path}; strash; balance; rewrite; "
        "refactor; rewrite; print_stats"
    )
    start = time.perf_counter()
    abc_result = subprocess.run(
        [abc, "-q", abc_script],
        capture_output=True,
        text=True,
        check=False,
    )
    abc_seconds = time.perf_counter() - start
    abc_log = abc_result.stdout + abc_result.stderr
    abc_log_path = run_dir / "abc_synthesis.log"
    abc_log_path.write_text(abc_log)

    status = (
        predictions_exact
        and maximum_transformation_difference <= 1e-5
        and yosys_result.returncode == 0
        and abc_result.returncode == 0
    )
    result = {
        "format_version": 1,
        "status": "passed" if status else "failed",
        "software": {
            "source_revision": git_revision(),
            "torch": str(torch.__version__),
            "torchlogix_circuit_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src"
                / "torchlogix"
                / "circuit.py"
            ),
            "verification_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "data_policy": {
            "verification_partition": cli.verification_split,
            "examples": cli.examples,
            "calibration_used": cli.verification_split == "calibration",
            "validation_used": cli.verification_split == "validation",
            "test_used": False,
            "verification_indices_sha256": verification_loader.split_manifest[
                "partitions"
            ][cli.verification_split]["indices_sha256"],
            "encoded_input_tensor_sha256": tensor_sha256(encoded_inputs),
            "label_tensor_sha256": tensor_sha256(labels),
        },
        "argmax_normalization": {
            "source": original_path.name,
            "source_sha256": sha256_file(original_path),
            "common_positive_tau_removed": common_tau,
            "common_offset_removed": common_offset,
            "original_betas": original_betas,
            "hardware_betas": [node.beta for node in hardware.sum_nodes],
            "predictions_exact": predictions_exact,
            "maximum_score_transformation_difference": (
                maximum_transformation_difference
            ),
        },
        "hardware_circuit": {
            "json": hardware_json_path.name,
            "json_sha256": sha256_file(hardware_json_path),
            "verilog": hardware_verilog_path.name,
            "verilog_sha256": sha256_file(hardware_verilog_path),
            "blif": hardware_blif_path.name if hardware_blif_path.exists() else None,
            "blif_sha256": (
                sha256_file(hardware_blif_path)
                if hardware_blif_path.exists()
                else None
            ),
            "logic_gates": len(hardware.gates),
            "sum_nodes": len(hardware.sum_nodes),
            "score_bits": score_bits,
        },
        "compiled_c": {
            "pack_bits": cli.pack_bits,
            "optimization_level": cli.compile_opt_level,
            "equivalence_seconds": compiled_equivalence_seconds,
        },
        "yosys": {
            "version": command_version([yosys, "-V"]),
            "command": [yosys, "-p", yosys_script],
            "returncode": yosys_result.returncode,
            "seconds": yosys_seconds,
            "log": yosys_log_path.name,
            "cells": parse_yosys_cells(yosys_log),
        },
        "abc": {
            "version": command_version([abc, "-q", "version"]),
            "command": [abc, "-q", abc_script],
            "returncode": abc_result.returncode,
            "seconds": abc_seconds,
            "log": abc_log_path.name,
            "stats": parse_abc_stats(abc_log),
        },
        "artifact_manifest": "artifact_manifest.json",
    }
    output_path = run_dir / "synthesis_verification.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not status:
        raise RuntimeError("synthesis verification failed")


if __name__ == "__main__":
    main()
