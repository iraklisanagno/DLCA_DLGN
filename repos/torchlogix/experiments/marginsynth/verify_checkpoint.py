#!/usr/bin/env python3
"""Verify and archive the hardened Circuit for a MarginSynth training run.

This command intentionally uses validation examples, never calibration or test
examples.  Calibration data is reserved for MarginSynth's rewrite search, and
the test split remains sealed until the paper protocol is frozen.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model, load_dataset

from torchlogix import Circuit


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def gate_histogram(circuit: Circuit) -> dict[str, int]:
    counts = Counter(gate.op.name for gate in circuit.gates)
    return dict(sorted(counts.items()))


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def write_artifact_manifest(run_dir: Path) -> Path:
    manifest_path = run_dir / "artifact_manifest.json"
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative_path = path.relative_to(run_dir).as_posix()
        artifacts[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "format_version": 1,
        "hash": "sha256",
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def circuit_record(circuit: Circuit) -> dict:
    return {
        "logic_gates": len(circuit.gates),
        "logic_gate_histogram": gate_histogram(circuit),
        "sum_nodes": len(circuit.sum_nodes),
        "sum_inputs": sum(len(node.input_ids) for node in circuit.sum_nodes),
        "inputs": circuit.n_inputs,
        "input_shape": circuit.input_shape,
        "outputs": len(circuit.outputs),
        "output_shape": circuit.output_shape,
        "sum_reductions": [
            {
                "node_id": node.node_id,
                "inputs": len(node.input_ids),
                "tau": node.tau,
                "beta": node.beta,
            }
            for node in circuit.sum_nodes
        ],
    }


DEFAULT_SCORE_ATOL = 1e-5


def score_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = DEFAULT_SCORE_ATOL,
) -> dict:
    """Compare scores while separating semantic and bitwise equivalence.

    Exact circuit simplification can change the floating-point addition order of
    a sum reduction without changing its Boolean function.  Keep reporting
    bitwise equality, but use a small absolute tolerance plus exact predictions
    for the semantic pass/fail decision.
    """
    reference = reference.detach().cpu()
    candidate = candidate.detach().cpu().to(reference.dtype)
    maximum_difference = float((reference - candidate).abs().max().item())
    return {
        "scores_exact": bool(torch.equal(reference, candidate)),
        "scores_close": bool(
            torch.allclose(reference, candidate, rtol=0.0, atol=atol)
        ),
        "score_absolute_tolerance": atol,
        "predictions_exact": bool(
            torch.equal(reference.argmax(dim=-1), candidate.argmax(dim=-1))
        ),
        "maximum_absolute_score_difference": maximum_difference,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--examples",
        type=int,
        default=32,
        help="Number of deterministic validation examples used for equivalence checks",
    )
    parser.add_argument(
        "--pack-bits",
        type=int,
        choices=[8, 16, 32, 64],
        default=32,
        help="Bit-packing width for the compiled-C equivalence check",
    )
    parser.add_argument(
        "--compile-opt-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="GCC optimization level used only for the compiled-C equivalence build",
    )
    parser.add_argument(
        "--verification-split",
        choices=["validation", "calibration"],
        default="validation",
        help="Held-out split used only for semantic-equivalence checks",
    )
    return parser.parse_args()


def take_examples(loader, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    image_batches = []
    label_batches = []
    remaining = count
    for images, labels in loader:
        take = min(remaining, len(images))
        image_batches.append(images[:take])
        label_batches.append(labels[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"requested {count} examples, validation partition has only "
            f"{count - remaining}"
        )
    return torch.cat(image_batches), torch.cat(label_batches)


def encoded_sample_shape(encoded_inputs: torch.Tensor) -> list[int]:
    """Return the Boolean backend's sample shape, excluding the batch axis."""
    if encoded_inputs.ndim < 2:
        raise ValueError(
            "encoded inputs must have a batch axis and at least one feature axis"
        )
    return list(encoded_inputs.shape[1:])


def main():
    cli = parse_args()
    run_dir = cli.run_dir.resolve()
    if cli.examples <= 0:
        raise ValueError("--examples must be positive")
    if cli.examples % cli.pack_bits:
        raise ValueError("--examples must be a multiple of --pack-bits")

    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"

    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    thresholds = state_dict["0.thresholds"]
    model = get_model(thresholds, args)
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
        full_model_scores = model(images)
        encoded_inputs = model[0](images).bool()
        boolean_backend = torch.nn.Sequential(*list(model.children())[1:])
        boolean_backend.eval()
        backend_scores = boolean_backend(encoded_inputs)

    timings = {}
    start = time.perf_counter()
    circuit = Circuit.from_model(
        boolean_backend,
        input_shape=encoded_sample_shape(encoded_inputs),
    )
    timings["export_seconds"] = time.perf_counter() - start

    with torch.no_grad():
        exported_backend_scores = boolean_backend(encoded_inputs)
    circuit_scores = circuit(encoded_inputs)

    hardened_path = run_dir / "hardened_circuit.json"
    circuit.write_json(str(hardened_path))
    hardened_record = circuit_record(circuit)
    hardened_record["file"] = hardened_path.name
    hardened_record["sha256"] = sha256_file(hardened_path)

    start = time.perf_counter()
    circuit.simplify()
    timings["exact_simplification_seconds"] = time.perf_counter() - start
    simplified_scores = circuit(encoded_inputs)

    simplified_path = run_dir / "exact_simplified_circuit.json"
    circuit.write_json(str(simplified_path))
    simplified_record = circuit_record(circuit)
    simplified_record["file"] = simplified_path.name
    simplified_record["sha256"] = sha256_file(simplified_path)

    start = time.perf_counter()
    circuit.compile(opt_level=cli.compile_opt_level, pack_bits=cli.pack_bits)
    timings["compiled_c_build_seconds"] = time.perf_counter() - start
    compiled_scores = torch.from_numpy(
        circuit(encoded_inputs.numpy(), use_compiled=True)
    )

    comparisons = {
        "full_model_vs_boolean_backend": score_comparison(
            full_model_scores, backend_scores
        ),
        "full_model_vs_export_mode_backend": score_comparison(
            full_model_scores, exported_backend_scores
        ),
        "boolean_backend_vs_hardened_circuit": score_comparison(
            exported_backend_scores, circuit_scores
        ),
        "boolean_backend_vs_exact_simplified_circuit": score_comparison(
            exported_backend_scores, simplified_scores
        ),
        "boolean_backend_vs_compiled_c": score_comparison(
            exported_backend_scores, compiled_scores
        ),
    }
    all_checks_passed = all(
        comparison["scores_close"] and comparison["predictions_exact"]
        for comparison in comparisons.values()
    )

    split_manifest = verification_loader.split_manifest
    result = {
        "format_version": 1,
        "status": "passed" if all_checks_passed else "failed",
        "software": {
            "source_revision": git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchlogix_circuit_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src"
                / "torchlogix"
                / "circuit.py"
            ),
            "verification_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "selected_step": checkpoint["metadata"]["step"],
            "selection_metric": "validation hard accuracy",
            "selection_value": checkpoint["metadata"]["metrics"][
                "val_acc_discrete"
            ],
        },
        "data_policy": {
            "verification_partition": cli.verification_split,
            "calibration_used": cli.verification_split == "calibration",
            "validation_used": cli.verification_split == "validation",
            "test_used": False,
            "examples": cli.examples,
            "verification_indices_sha256": split_manifest["partitions"][
                cli.verification_split
            ]["indices_sha256"],
            "input_tensor_sha256": tensor_sha256(images),
            "encoded_input_tensor_sha256": tensor_sha256(encoded_inputs),
            "label_tensor_sha256": tensor_sha256(labels),
        },
        "hardening_boundary": {
            "description": (
                "The trained input binarization is applied before the editable "
                "Boolean backend and stored separately in the checkpoint."
            ),
            "threshold_shape": list(thresholds.shape),
            "threshold_values": thresholds.detach().cpu().tolist(),
            "encoded_dtype": str(encoded_inputs.dtype),
            "encoded_input_shape": encoded_sample_shape(encoded_inputs),
        },
        "hardened_circuit": hardened_record,
        "exact_simplified_circuit": simplified_record,
        "compiled_c": {
            "pack_bits": cli.pack_bits,
            "compiler": "gcc",
            "optimization_level": cli.compile_opt_level,
        },
        "artifact_manifest": "artifact_manifest.json",
        "comparisons": comparisons,
        "timings": timings,
    }
    output_path = run_dir / "export_verification.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))

    if not all_checks_passed:
        raise RuntimeError("one or more checkpoint export checks failed")


if __name__ == "__main__":
    main()
