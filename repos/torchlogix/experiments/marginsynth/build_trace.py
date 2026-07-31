#!/usr/bin/env python3
"""Build and verify a packed MarginSynth calibration trace database."""

import argparse
import json
import resource
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
from verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)
from trace import build_trace

from torchlogix import Circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--circuit",
        default="exact_simplified_circuit.json",
        help="Circuit JSON inside the run directory",
    )
    parser.add_argument(
        "--output",
        default="calibration_trace",
        help="Trace subdirectory inside the run directory",
    )
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"

    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _, _, calibration_loader, _ = load_dataset(args, include_calibration=True)
    num_examples = len(calibration_loader.dataset)
    images, labels = take_examples(calibration_loader, num_examples)
    with torch.no_grad():
        encoded_inputs = model[0](images).bool()
        backend = torch.nn.Sequential(*list(model.children())[1:])
        backend.eval()
        backend_scores = backend(encoded_inputs).cpu()

    circuit_path = run_dir / cli.circuit
    circuit = Circuit.from_json_file(str(circuit_path))
    start = time.perf_counter()
    trace = build_trace(circuit, encoded_inputs, labels)
    construction_seconds = time.perf_counter() - start

    score_difference = float(
        np.max(np.abs(trace.scores - backend_scores.numpy()))
    )
    scores_exact = bool(np.array_equal(trace.scores, backend_scores.numpy()))
    predictions_exact = bool(
        np.array_equal(
            trace.predictions,
            backend_scores.argmax(dim=-1).numpy(),
        )
    )
    if not scores_exact or not predictions_exact:
        raise RuntimeError(
            "packed trace does not match the hardened Boolean backend: "
            f"score difference={score_difference}"
        )

    margin_quantiles = np.quantile(
        trace.winner_margins,
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
    )
    cone_sizes = np.asarray(
        [
            len(trace.affected_gate_indices(int(gate_id)))
            for gate_id in trace.gate_ids
        ],
        dtype=np.int64,
    )
    direct_fanouts = np.diff(trace.fanout_indptr)
    trace_directory = run_dir / cli.output
    extra_metadata = {
        "purpose": "MarginSynth rewrite calibration; not validation or test",
        "partition": "calibration",
        "calibration_indices_sha256": calibration_loader.split_manifest[
            "partitions"
        ]["calibration"]["indices_sha256"],
        "calibration_class_counts": calibration_loader.split_manifest[
            "partitions"
        ]["calibration"]["class_counts"],
        "test_used": False,
        "validation_used": False,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "circuit": circuit_path.name,
        "circuit_sha256": sha256_file(circuit_path),
        "encoded_input_tensor_sha256": tensor_sha256(encoded_inputs),
        "label_tensor_sha256": tensor_sha256(labels),
        "scores_exact_vs_boolean_backend": scores_exact,
        "predictions_exact_vs_boolean_backend": predictions_exact,
        "maximum_absolute_score_difference": score_difference,
        "construction_seconds": construction_seconds,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "winner_margin_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
                margin_quantiles,
            )
        },
        "zero_winner_margin_examples": int(
            np.count_nonzero(trace.winner_margins == 0.0)
        ),
        "zero_winner_margin_rate": float(
            np.mean(trace.winner_margins == 0.0)
        ),
        "affected_gate_cone_size_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                [0.0, 0.5, 0.9, 0.99, 1.0],
                np.quantile(cone_sizes, [0.0, 0.5, 0.9, 0.99, 1.0]),
            )
        },
        "direct_fanout_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                [0.0, 0.5, 0.9, 0.99, 1.0],
                np.quantile(direct_fanouts, [0.0, 0.5, 0.9, 0.99, 1.0]),
            )
        },
        "source_revision": git_revision(),
        "torch": str(torch.__version__),
        "trace_implementation_sha256": sha256_file(Path(__file__).with_name("trace.py")),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    trace.save(trace_directory, extra_metadata=extra_metadata)
    write_artifact_manifest(run_dir)

    array_bytes = sum(
        record["bytes"]
        for record in json.loads(
            (trace_directory / "metadata.json").read_text()
        )["arrays"].values()
    )
    result = {
        "status": "passed",
        "trace_directory": trace_directory.name,
        "samples": trace.num_samples,
        "nodes": len(trace.node_ids),
        "gates": trace.num_gates,
        "outputs": trace.num_outputs,
        "packed_words_per_node": trace.values.shape[1],
        "array_bytes": array_bytes,
        "construction_seconds": construction_seconds,
        "baseline_accuracy": trace.metadata["baseline_accuracy"],
        "scores_exact": scores_exact,
        "predictions_exact": predictions_exact,
        "maximum_absolute_score_difference": score_difference,
        "test_used": False,
        "validation_used": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
