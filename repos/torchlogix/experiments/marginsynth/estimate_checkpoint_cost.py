#!/usr/bin/env python3
"""Estimate ABC cost from an exactly simplified checkpoint without loading data."""

from __future__ import annotations

import argparse
import json
import platform
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

from utils import get_model

from experiments.marginsynth.cost_model import SynthCostEstimator, circuit_features
from experiments.marginsynth.verify_checkpoint import git_revision, sha256_file
from torchlogix import Circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cost-model", required=True, type=Path)
    parser.add_argument("--output", default="bayesian_cost_proxy.json")
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    method_dir = cli.method_dir.resolve()
    checkpoint_path = method_dir / cli.checkpoint
    output_path = method_dir / cli.output
    circuit_path = method_dir / "bayesian_proxy_exact_simplified_circuit.json"
    if output_path.exists() or circuit_path.exists():
        raise RuntimeError("refusing to overwrite an existing Bayesian cost proxy")
    started = time.perf_counter()

    training_config = json.loads((run_dir / "training_config.json").read_text())
    source_export_path = run_dir / "export_verification.json"
    source_export = json.loads(source_export_path.read_text())
    input_shape = source_export["hardening_boundary"]["encoded_input_shape"]
    args = Namespace(**training_config)
    args.device = "cpu"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    thresholds = state["0.thresholds"]
    model = get_model(thresholds, args)
    model.load_state_dict(state, strict=True)
    model.eval()
    boolean_backend = torch.nn.Sequential(*list(model.children())[1:])
    boolean_backend.eval()

    export_started = time.perf_counter()
    circuit = Circuit.from_model(
        boolean_backend,
        input_shape=input_shape,
    )
    export_seconds = time.perf_counter() - export_started
    simplify_started = time.perf_counter()
    circuit.simplify()
    simplify_seconds = time.perf_counter() - simplify_started
    circuit.write_json(str(circuit_path))
    features = circuit_features(circuit)
    estimator = SynthCostEstimator.from_json(cli.cost_model.resolve())
    predicted = estimator.estimate_from_features(features)
    result = {
        "format_version": 1,
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cost_model": str(cli.cost_model.resolve()),
        "cost_model_sha256": sha256_file(cli.cost_model.resolve()),
        "input_shape_source": str(source_export_path),
        "input_shape_source_sha256": sha256_file(source_export_path),
        "encoded_input_shape": input_shape,
        "exact_simplified_circuit": circuit_path.name,
        "exact_simplified_circuit_sha256": sha256_file(circuit_path),
        "features": features,
        "predicted_abc_and_nodes": predicted,
        "estimator_metadata": estimator.metadata,
        "timing": {
            "export_seconds": export_seconds,
            "exact_simplification_seconds": simplify_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "data_policy": {
            "dataset_loaded": False,
            "calibration_loaded": False,
            "validation_loaded": False,
            "test_loaded": False,
        },
        "software": {
            "source_revision": git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
