#!/usr/bin/env python3
"""Export and benchmark frozen convolutional checkpoints on synthetic inputs.

The benchmark never loads CIFAR data.  It removes the floating-point input
encoder, feeds already-thresholded Boolean channels to the deployable logic
submodel, checks PyTorch/export/Circuit/C agreement, and times the generated C
implementation on CPU.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model  # noqa: E402
from torchlogix import Circuit  # noqa: E402
from torchlogix.circuit import _c_output_dtype  # noqa: E402
from torchlogix.layers import GroupSum  # noqa: E402
from torchlogix.utils import set_export_mode  # noqa: E402


RUNS = {
    "paper_s_random": "pilot_conv_cifar10_paper_small_random_seed0",
    "paper_s_v4": "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed0",
    "paper_s_u1": "ablate_conv_cifar10_small_balanced_channel_no_swaps_seed0",
    "paper_m_random": "full_conv_cifar10_paper_medium_random_seed0_200k",
    "paper_m_v4": "full_conv_cifar10_paper_medium_legacy_v4_seed0",
    "warp_m_random": "warp_fig4_medium_paired_random_fixed_uniform_seed0",
    "warp_m_v4": "warp_fig4_medium_legacy_v4_fixed_uniform_seed0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=sorted(RUNS))
    parser.add_argument("--pack-bits", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--benchmark-batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compile-timeout", type=float, default=180.0)
    parser.add_argument("--opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--skip-simplify", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    return parser.parse_args()


def stage(name: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {name}", flush=True)


def persist(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def attach_compiled_library(circuit: Circuit, so_path: Path, pack_bits: int) -> None:
    """Attach a library produced from Circuit.get_c_code()."""
    ctype_map = {
        None: ctypes.c_bool,
        8: ctypes.c_uint8,
        16: ctypes.c_uint16,
        32: ctypes.c_uint32,
        64: ctypes.c_uint64,
    }
    reduction_ctype_map = {
        "float": ctypes.c_float,
        "uint8_t": ctypes.c_uint8,
        "uint16_t": ctypes.c_uint16,
        "uint32_t": ctypes.c_uint32,
        "uint64_t": ctypes.c_uint64,
    }
    in_ctype = ctype_map[pack_bits]
    output_ids = set(circuit.outputs)
    reductions = [node for node in circuit.sum_nodes if node.node_id in output_ids]
    out_ctype = (
        reduction_ctype_map[_c_output_dtype(reductions)]
        if reductions
        else in_ctype
    )
    library = ctypes.CDLL(str(so_path))
    library.circuit.argtypes = [
        ctypes.POINTER(in_ctype),
        ctypes.POINTER(out_ctype),
    ]
    library.circuit.restype = None
    library.circuit_bench.argtypes = [
        ctypes.POINTER(in_ctype),
        ctypes.POINTER(out_ctype),
        ctypes.c_int,
    ]
    library.circuit_bench.restype = None
    bool_out_ctype = out_ctype if reductions else ctypes.c_bool
    library.circuit_bench_bool.argtypes = [
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(bool_out_ctype),
        ctypes.c_int,
    ]
    library.circuit_bench_bool.restype = None
    circuit._lib = library
    circuit._pack_bits = pack_bits


def main() -> int:
    args = parse_args()
    run_name = RUNS[args.run]
    run_dir = ROOT / "results" / run_name
    output_dir = ROOT / "summary" / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.run}.json"
    checkpoint = run_dir / "best_checkpoint.pt"
    config_path = run_dir / "training_config.json"
    config = json.loads(config_path.read_text())
    config_args = Namespace(**config)
    config_args.device = "cpu"

    result = {
        "status": "RUNNING",
        "run_label": args.run,
        "run_name": run_name,
        "checkpoint": "best_checkpoint.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "training_config_sha256": sha256(config_path),
        "architecture": config["architecture"],
        "connections_init_method": config["connections_init_method"],
        "conv_connections_init_method": config.get("conv_connections_init_method"),
        "classifier_connections_init_method": config.get(
            "classifier_connections_init_method"
        ),
        "parameterization": config["parametrization"],
        "heldout_test_accessed": False,
        "input_source": "deterministic synthetic Boolean tensors only",
        "cpu": os.uname().machine,
        "torch_threads": torch.get_num_threads(),
    }
    persist(output, result)

    try:
        stage("load frozen model")
        load_start = time.perf_counter()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload["model_state_dict"]
        thresholds = state_dict["0.thresholds"]
        model = get_model(thresholds, config_args)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        result["model_load_seconds"] = time.perf_counter() - load_start
        result["thresholds_per_rgb_channel"] = int(thresholds.shape[-1])
        result["boolean_input_channels"] = 3 * int(thresholds.shape[-1])
        persist(output, result)

        children = list(model.children())
        encoder = children[0]
        deploy_model = torch.nn.Sequential(*children[1:])
        for module in deploy_model.modules():
            if isinstance(module, GroupSum):
                module.tau = 1.0
                module.beta = 0.0

        generator = torch.Generator().manual_seed(2027)
        raw = torch.rand(2, 3, 32, 32, generator=generator)
        with torch.no_grad():
            encoded = encoder(raw).to(torch.bool)
            hardened_scores = deploy_model(encoded.to(torch.float32))
            hardened_classes = hardened_scores.argmax(-1)

        stage("trace deployable Boolean circuit")
        trace_rss = peak_rss_bytes()
        trace_start = time.perf_counter()
        set_export_mode(deploy_model)
        with torch.no_grad():
            export_scores = deploy_model(encoded)
        circuit = Circuit.from_model(
            deploy_model,
            input_shape=(result["boolean_input_channels"], 32, 32),
        )
        result["trace_seconds"] = time.perf_counter() - trace_start
        result["trace_peak_rss_bytes"] = peak_rss_bytes()
        result["trace_peak_rss_delta_bytes"] = peak_rss_bytes() - trace_rss
        result["circuit_before_simplification"] = {
            "input_bits": circuit.n_inputs,
            "logic_gates": len(circuit.gates),
            "sum_reductions": len(circuit.sum_nodes),
            "outputs": len(circuit.outputs),
        }
        persist(output, result)

        stage("check PyTorch export and Python Circuit equivalence")
        python_start = time.perf_counter()
        circuit_scores = circuit(encoded[:1])
        result["python_circuit_one_example_seconds"] = time.perf_counter() - python_start
        export_match = torch.equal(
            export_scores[:1].to(torch.int64), circuit_scores.to(torch.int64)
        )
        class_match = torch.equal(
            hardened_classes[:1], circuit_scores.argmax(-1)
        )
        if not export_match or not class_match:
            raise RuntimeError(
                f"equivalence failed: export={export_match}, class={class_match}"
            )
        result["equivalence"] = {
            "hardened_class_vs_python_circuit": class_match,
            "pytorch_export_scores_vs_python_circuit": export_match,
            "synthetic_examples": 1,
        }
        persist(output, result)

        if not args.skip_simplify:
            stage("simplify circuit")
            simplify_start = time.perf_counter()
            circuit.simplify()
            result["simplification_seconds"] = time.perf_counter() - simplify_start
        result["circuit_after_simplification"] = {
            "logic_gates": len(circuit.gates),
            "sum_reductions": len(circuit.sum_nodes),
            "outputs": len(circuit.outputs),
        }
        persist(output, result)

        simplified_scores = circuit(encoded[:1])
        simplified_match = torch.equal(
            export_scores[:1].to(torch.int64), simplified_scores.to(torch.int64)
        )
        if not simplified_match:
            raise RuntimeError("simplified Python Circuit is not equivalent")
        result["equivalence"]["pytorch_export_vs_simplified_circuit"] = True

        if not args.skip_compile:
            stage("generate C and compile")
            source_start = time.perf_counter()
            c_code = circuit.get_c_code(pack_bits=args.pack_bits)
            result["generated_c_source_bytes"] = len(c_code.encode("utf-8"))
            result["c_source_generation_seconds"] = time.perf_counter() - source_start
            compile_rss = peak_rss_bytes()
            compile_start = time.perf_counter()
            compiled = False
            with tempfile.TemporaryDirectory(prefix="coverage-circuit-") as tmp:
                tmp_dir = Path(tmp)
                c_path = tmp_dir / "circuit.c"
                so_path = tmp_dir / "circuit.so"
                c_path.write_text(c_code)
                del c_code
                try:
                    completed = subprocess.run(
                        [
                            "gcc",
                            f"-O{args.opt_level}",
                            "-shared",
                            "-fPIC",
                            "-o",
                            str(so_path),
                            str(c_path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.compile_timeout,
                    )
                except subprocess.TimeoutExpired:
                    result["compile"] = {
                        "completed": False,
                        "timeout_seconds": args.compile_timeout,
                        "opt_level": args.opt_level,
                        "interpretation": (
                            "Fully unrolled generated C did not compile within "
                            "the predeclared feasibility bound."
                        ),
                    }
                else:
                    if completed.returncode != 0:
                        raise RuntimeError(
                            "C compilation failed: " + completed.stderr[-4000:]
                        )
                    attach_compiled_library(circuit, so_path, args.pack_bits)
                    compiled = True
                    result["compile"] = {
                        "completed": True,
                        "seconds": time.perf_counter() - compile_start,
                        "opt_level": args.opt_level,
                        "shared_object_bytes": so_path.stat().st_size,
                    }
                    result["compile_peak_rss_bytes"] = peak_rss_bytes()
                    result["compile_peak_rss_delta_bytes"] = (
                        peak_rss_bytes() - compile_rss
                    )
                    # The loaded mapping remains valid after TemporaryDirectory
                    # removes its directory on Linux.
            persist(output, result)

            if compiled and args.benchmark_batch % args.pack_bits:
                raise ValueError("benchmark batch must be divisible by pack bits")
            benchmark_generator = torch.Generator().manual_seed(2028)
            benchmark_input = torch.randint(
                0,
                2,
                (
                    args.benchmark_batch,
                    result["boolean_input_channels"],
                    32,
                    32,
                ),
                generator=benchmark_generator,
                dtype=torch.bool,
            ).numpy()
            if compiled:
                compiled_scores = circuit(benchmark_input, use_compiled=True)
                compiled_match = np.array_equal(
                    compiled_scores[:1].astype(np.int64),
                    circuit(torch.from_numpy(benchmark_input[:1])).numpy().astype(np.int64),
                )
                if not compiled_match:
                    raise RuntimeError("compiled C Circuit is not equivalent")
                result["equivalence"]["compiled_c_vs_python_circuit"] = True

            if compiled:
                stage("benchmark compiled CPU circuit")
                for _ in range(args.warmup):
                    circuit(benchmark_input, use_compiled=True)
                latencies = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    circuit(benchmark_input, use_compiled=True)
                    latencies.append(1000.0 * (time.perf_counter() - start))
                result["compiled_cpu_benchmark"] = {
                    "pack_bits": args.pack_bits,
                    "batch_size": args.benchmark_batch,
                    "warmup_batches": args.warmup,
                    "timed_batches": args.repeats,
                    "latency_milliseconds_per_batch_mean": float(
                        np.mean(latencies)
                    ),
                    "latency_milliseconds_per_batch_std": float(
                        np.std(latencies, ddof=1)
                    ),
                    "microseconds_per_example_mean": float(
                        1000.0 * np.mean(latencies) / args.benchmark_batch
                    ),
                    "examples_per_second": float(
                        1000.0 * args.benchmark_batch / np.mean(latencies)
                    ),
                    "includes_boolean_input_packing": True,
                    "single_process_single_generated_c_entrypoint": True,
                }

        result["final_peak_rss_bytes"] = peak_rss_bytes()
        result["status"] = (
            "COMPLETE"
            if args.skip_compile or result.get("compile", {}).get("completed")
            else "PARTIAL-COMPILE-TIMEOUT"
        )
    except Exception as exc:
        result["status"] = "FAILED"
        result["failure_type"] = type(exc).__name__
        result["failure_message"] = str(exc)
        persist(output, result)
        raise

    persist(output, result)
    stage(f"complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
