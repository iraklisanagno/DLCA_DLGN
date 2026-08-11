#!/usr/bin/env python3
"""Export and synthesize Two-Stage Unit Tying checkpoints.

The script uses the same validation partition, exact circuit simplifier,
hardware normalization, Yosys flow, and ABC flow as MarginSynth. The test
partition remains sealed.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.unit_tying import evaluate_encoded
from experiments.marginsynth.verify_checkpoint import (
    circuit_record,
    encoded_sample_shape,
    git_revision,
    score_comparison,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)
from experiments.marginsynth.verify_synthesis import (
    MAX_INTEGER_ROUNDING_RESIDUAL,
    command_version,
    integer_score_predictions,
    normalized_integer_score_comparison,
    parse_abc_stats,
    parse_yosys_cells,
)
from torchlogix import Circuit
from torchlogix.circuit import _c_output_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="Optional subset of completed ratios",
    )
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--equivalence-examples", type=int, default=32)
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
        default=0,
    )
    return parser.parse_args()


def run_command(command: list[str]) -> tuple[subprocess.CompletedProcess, float]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, time.perf_counter() - start


def main():
    cli = parse_args()
    if cli.examples <= 0 or cli.examples % cli.pack_bits:
        raise ValueError("--examples must be positive and divisible by --pack-bits")
    if (
        cli.equivalence_examples <= 0
        or cli.equivalence_examples > cli.examples
        or cli.equivalence_examples % cli.pack_bits
    ):
        raise ValueError(
            "--equivalence-examples must be positive, no larger than --examples, "
            "and divisible by --pack-bits"
        )

    run_dir = cli.run_dir.resolve()
    baseline_root = run_dir / "baselines" / "two_stage_unit_tying"
    aggregate = json.loads((baseline_root / "aggregate.json").read_text())
    available = {float(record["ratio"]): record for record in aggregate}
    ratios = sorted(available) if cli.ratios is None else cli.ratios
    if any(float(ratio) not in available for ratio in ratios):
        raise ValueError("requested ratio is not present in the completed sweep")

    yosys = shutil.which("yosys")
    abc = shutil.which("berkeley-abc") or shutil.which("abc")
    if yosys is None or abc is None:
        raise RuntimeError("both Yosys and Berkeley ABC are required")

    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    source_checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    thresholds = source_checkpoint["model_state_dict"]["0.thresholds"]
    _, validation_loader, _, _ = load_dataset(args, include_calibration=True)
    images, labels = take_examples(validation_loader, cli.examples)
    encoder_model = get_model(thresholds, args)
    encoder_model.load_state_dict(
        source_checkpoint["model_state_dict"],
        strict=True,
    )
    encoder_model.eval()
    with torch.no_grad():
        encoded_inputs = encoder_model[0](images).bool()
    del encoder_model

    aggregate_path = baseline_root / "synthesis_aggregate.json"
    existing_synthesis = []
    if aggregate_path.exists():
        existing_synthesis = json.loads(aggregate_path.read_text())
    synthesis_by_ratio = {
        float(record["ratio"]): record for record in existing_synthesis
    }
    for ratio in ratios:
        ratio = float(ratio)
        ratio_name = available[ratio]["directory"]
        ratio_dir = baseline_root / ratio_name
        output_dir = ratio_dir / "synthesis"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ratio_dir / "tied_checkpoint.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = get_model(thresholds, args)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        timings = {}
        start = time.perf_counter()
        model_scores = evaluate_encoded(
            model,
            encoded_inputs,
            int(training_config.get("batch_size", 512)),
            torch.device("cpu"),
        )
        timings["pytorch_cpu_validation_seconds"] = time.perf_counter() - start
        model_predictions = model_scores.argmax(dim=-1)
        model_accuracy = float((model_predictions == labels).float().mean())

        boolean_backend = torch.nn.Sequential(*list(model.children())[1:])
        check_inputs = encoded_inputs[: cli.equivalence_examples]
        with torch.no_grad():
            backend_scores = boolean_backend(check_inputs)
        start = time.perf_counter()
        circuit = Circuit.from_model(
            boolean_backend,
            input_shape=encoded_sample_shape(encoded_inputs),
        )
        timings["export_seconds"] = time.perf_counter() - start
        hardened_scores = circuit(check_inputs)
        hardened_path = output_dir / "hardened_circuit.json"
        circuit.write_json(str(hardened_path))
        hardened_record = circuit_record(circuit)
        hardened_record |= {
            "file": hardened_path.name,
            "sha256": sha256_file(hardened_path),
        }

        start = time.perf_counter()
        circuit.simplify()
        timings["exact_simplification_seconds"] = time.perf_counter() - start
        simplified_scores = circuit(check_inputs)
        simplified_path = output_dir / "exact_simplified_circuit.json"
        circuit.write_json(str(simplified_path))
        simplified_record = circuit_record(circuit)
        simplified_record |= {
            "file": simplified_path.name,
            "sha256": sha256_file(simplified_path),
        }

        reductions = [
            circuit._sum_by_id[output_id] for output_id in circuit.outputs
        ]
        common_tau = reductions[0].tau
        original_betas = [node.beta for node in reductions]
        common_offset = min(original_betas)
        hardware = circuit.normalized_for_hardware_argmax()
        hardware_path = output_dir / "hardware_argmax_circuit.json"
        verilog_path = output_dir / "hardware_argmax_circuit.v"
        blif_path = output_dir / "hardware_argmax_circuit.blif"
        hardware.write_json(str(hardware_path))
        hardware.write_verilog_code(str(verilog_path))

        start = time.perf_counter()
        circuit.compile(
            opt_level=cli.compile_opt_level,
            pack_bits=cli.pack_bits,
        )
        exact_compiled_scores = torch.from_numpy(
            circuit(encoded_inputs.numpy(), use_compiled=True)
        )
        timings["exact_compiled_c_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        hardware.compile(
            opt_level=cli.compile_opt_level,
            pack_bits=cli.pack_bits,
        )
        hardware_scores = torch.from_numpy(
            hardware(encoded_inputs.numpy(), use_compiled=True)
        )
        timings["hardware_compiled_c_seconds"] = time.perf_counter() - start
        normalization_comparison = normalized_integer_score_comparison(
            exact_compiled_scores,
            hardware_scores,
            common_tau,
            common_offset,
        )
        compiled_predictions_exact = bool(
            torch.equal(
                model_predictions,
                integer_score_predictions(hardware_scores),
            )
        )
        exact_compiled_comparison = score_comparison(
            model_scores,
            exact_compiled_scores,
        )

        yosys_script = (
            f"read_verilog -sv {verilog_path}; "
            "synth -top circuit -noabc; check; "
            f"write_blif {blif_path}"
        )
        yosys_result, yosys_seconds = run_command([yosys, "-p", yosys_script])
        yosys_log = yosys_result.stdout + yosys_result.stderr
        yosys_log_path = output_dir / "yosys.log"
        yosys_log_path.write_text(yosys_log)
        abc_script = (
            f"read_blif {blif_path}; strash; balance; rewrite; "
            "refactor; rewrite; print_stats"
        )
        abc_result, abc_seconds = run_command([abc, "-q", abc_script])
        abc_log = abc_result.stdout + abc_result.stderr
        abc_log_path = output_dir / "abc.log"
        abc_log_path.write_text(abc_log)

        equivalence = {
            "backend_vs_hardened": score_comparison(
                backend_scores,
                hardened_scores,
            ),
            "backend_vs_exact_simplified": score_comparison(
                backend_scores,
                simplified_scores,
            ),
            "model_vs_exact_compiled_full_validation": exact_compiled_comparison,
            "model_vs_hardware_predictions_full_validation": (
                compiled_predictions_exact
            ),
            "hardware_score_normalization": normalization_comparison,
        }
        status = (
            all(
                record["scores_close"] and record["predictions_exact"]
                for record in (
                    equivalence["backend_vs_hardened"],
                    equivalence["backend_vs_exact_simplified"],
                    exact_compiled_comparison,
                )
            )
            and compiled_predictions_exact
            and normalization_comparison["integer_scores_exact"]
            and normalization_comparison["maximum_integer_rounding_residual"]
            <= MAX_INTEGER_ROUNDING_RESIDUAL
            and yosys_result.returncode == 0
            and abc_result.returncode == 0
        )
        hardware_reductions = [
            hardware._sum_by_id[output_id] for output_id in hardware.outputs
        ]
        result = {
            "format_version": 1,
            "status": "passed" if status else "failed",
            "ratio": ratio,
            "checkpoint": {
                "file": str(checkpoint_path.relative_to(ratio_dir)),
                "sha256": sha256_file(checkpoint_path),
            },
            "data_policy": {
                "partition": "validation",
                "examples": cli.examples,
                "equivalence_examples": cli.equivalence_examples,
                "validation_indices_sha256": validation_loader.split_manifest[
                    "partitions"
                ]["validation"]["indices_sha256"],
                "encoded_inputs_sha256": tensor_sha256(encoded_inputs),
                "labels_sha256": tensor_sha256(labels),
                "calibration_used": False,
                "test_used": False,
            },
            "validation_accuracy": model_accuracy,
            "hardened_circuit": hardened_record,
            "exact_simplified_circuit": simplified_record,
            "hardware_circuit": {
                "logic_gates": len(hardware.gates),
                "sum_nodes": len(hardware.sum_nodes),
                "score_bits": {
                    "uint8_t": 8,
                    "uint16_t": 16,
                    "uint32_t": 32,
                    "uint64_t": 64,
                }[_c_output_dtype(hardware_reductions)],
                "json": hardware_path.name,
                "json_sha256": sha256_file(hardware_path),
                "verilog": verilog_path.name,
                "verilog_sha256": sha256_file(verilog_path),
                "blif": blif_path.name if blif_path.exists() else None,
                "blif_sha256": (
                    sha256_file(blif_path) if blif_path.exists() else None
                ),
            },
            "equivalence": equivalence,
            "compiled_c": {
                "pack_bits": cli.pack_bits,
                "optimization_level": cli.compile_opt_level,
            },
            "yosys": {
                "version": command_version([yosys, "-V"]),
                "returncode": yosys_result.returncode,
                "seconds": yosys_seconds,
                "cells": parse_yosys_cells(yosys_log),
                "log": yosys_log_path.name,
            },
            "abc": {
                "version": command_version([abc, "-q", "version"]),
                "returncode": abc_result.returncode,
                "seconds": abc_seconds,
                "stats": parse_abc_stats(abc_log),
                "log": abc_log_path.name,
            },
            "timings": timings,
            "peak_process_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "software": {
                "source_revision": git_revision(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        result_path = output_dir / "synthesis.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        write_artifact_manifest(output_dir)
        if not status:
            raise RuntimeError(f"synthesis verification failed for ratio {ratio}")
        compact = {
            "ratio": ratio,
            "validation_accuracy": model_accuracy,
            "exact_logic_gates": simplified_record["logic_gates"],
            "abc_and_nodes": result["abc"]["stats"].get("and_nodes"),
            "abc_levels": result["abc"]["stats"].get("levels"),
            "status": result["status"],
        }
        synthesis_by_ratio[ratio] = compact
        print(json.dumps(compact, indent=2, sort_keys=True), flush=True)

    synthesis_aggregate = [
        synthesis_by_ratio[ratio] for ratio in sorted(synthesis_by_ratio)
    ]
    aggregate_path.write_text(
        json.dumps(synthesis_aggregate, indent=2, sort_keys=True) + "\n"
    )
    write_artifact_manifest(baseline_root)
    print(
        json.dumps(
            {
                "status": "completed",
                "results": synthesis_aggregate,
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
