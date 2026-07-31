#!/usr/bin/env python3
"""Evaluate a fully frozen five-seed protocol on the held-out test set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.evaluate_frontier import validation_metrics
from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    take_examples,
    tensor_sha256,
)
from torchlogix import Circuit


def compiled_scores(circuit_path: Path, encoded: torch.Tensor):
    circuit = Circuit.from_json_file(str(circuit_path))
    circuit.compile(opt_level=0, pack_bits=16)
    return circuit(encoded.numpy(), use_compiled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Root containing the already-cached public dataset.",
    )
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    os.environ["DATASET_PATH"] = str(cli.dataset_path.resolve())
    freeze = json.loads(cli.freeze.read_text())
    if not freeze.get("test_access_authorized_after_this_freeze"):
        raise RuntimeError("the supplied protocol does not authorize test access")
    paired_path = Path(freeze["paired_manifest"])
    if sha256_file(paired_path) != freeze["paired_manifest_sha256"]:
        raise RuntimeError("paired manifest changed after protocol freeze")
    method_path = Path(freeze["method_config"])
    if sha256_file(method_path) != freeze["method_config_sha256"]:
        raise RuntimeError("method configuration changed after protocol freeze")
    paired = json.loads(paired_path.read_text())
    budget = float(freeze["operating_point"])
    frozen_seed_artifacts = freeze.get("seed_artifacts")
    if frozen_seed_artifacts:
        if len(frozen_seed_artifacts) != len(paired["records"]):
            raise RuntimeError("frozen seed-artifact count does not match manifest")
        # Verify every frozen input before opening even the first test example.
        for frozen_record in frozen_seed_artifacts:
            for name, artifact in frozen_record["artifacts"].items():
                artifact_path = Path(artifact["path"])
                if not artifact_path.exists():
                    raise RuntimeError(
                        f"frozen artifact is missing for seed "
                        f"{frozen_record['seed']}: {name}"
                    )
                if sha256_file(artifact_path) != artifact["sha256"]:
                    raise RuntimeError(
                        f"frozen artifact changed for seed "
                        f"{frozen_record['seed']}: {name}"
                    )
    seed_records = []
    for index, record in enumerate(paired["records"]):
        frozen_record = (
            frozen_seed_artifacts[index] if frozen_seed_artifacts else None
        )
        run_dir = Path(
            frozen_record["run_dir"] if frozen_record else record["run_dir"]
        ).resolve()
        training_config_path = (
            Path(frozen_record["artifacts"]["training_config"]["path"])
            if frozen_record
            else run_dir / "training_config.json"
        )
        training_config = json.loads(training_config_path.read_text())
        args = Namespace(**training_config)
        args.device = "cpu"
        checkpoint = torch.load(
            run_dir / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state_dict"]
        model = get_model(state["0.thresholds"], args)
        model.load_state_dict(state, strict=True)
        model.eval()
        _, _, _, test_loader = load_dataset(args, include_calibration=True)
        images, labels_tensor = take_examples(
            test_loader,
            len(test_loader.dataset),
        )
        with torch.no_grad():
            encoded = model[0](images).bool()
        labels = labels_tensor.numpy()
        baseline_path = (
            Path(frozen_record["artifacts"]["exact_baseline"]["path"])
            if frozen_record
            else run_dir / "exact_simplified_circuit.json"
        )
        baseline_scores = compiled_scores(baseline_path, encoded)
        baseline_predictions = baseline_scores.argmax(axis=1)
        baseline_accuracy = float((baseline_predictions == labels).mean())
        if frozen_record:
            selected_circuit_path = Path(
                frozen_record["artifacts"]["selected_circuit"]["path"]
            )
            selected_step = frozen_record["selected_step"]
        else:
            search_dir = run_dir / f"search_v2_frozen_seed{record['seed']}"
            summary = json.loads((search_dir / "search_summary.json").read_text())
            matches = [
                point
                for point in summary["pareto"]
                if abs(float(point["accuracy_budget"]) - budget) <= 1e-12
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"seed {record['seed']} lacks frozen budget {budget}"
                )
            point = matches[0]
            selected_circuit_path = search_dir / point["snapshot"] / "circuit.json"
            selected_step = point["selected_step"]
        method_scores = compiled_scores(selected_circuit_path, encoded)
        seed_result = {
            "seed": record["seed"],
            "selected_step": selected_step,
            "baseline_accuracy": baseline_accuracy,
            "method": validation_metrics(
                method_scores,
                labels,
                baseline_predictions,
                baseline_accuracy,
            ),
            "method_circuit_sha256": sha256_file(selected_circuit_path),
            "encoded_test_inputs_sha256": tensor_sha256(encoded),
            "test_labels_sha256": tensor_sha256(labels_tensor),
        }
        if frozen_record and "unit_tying_circuit" in frozen_record["artifacts"]:
            unit_path = Path(
                frozen_record["artifacts"]["unit_tying_circuit"]["path"]
            )
            unit_scores = compiled_scores(unit_path, encoded)
            seed_result["unit_tying"] = validation_metrics(
                unit_scores,
                labels,
                baseline_predictions,
                baseline_accuracy,
            )
            seed_result["unit_tying_ratio"] = freeze["unit_tying_ratio"]
            seed_result["unit_tying_circuit_sha256"] = sha256_file(unit_path)
        seed_records.append(seed_result)
    result = {
        "format_version": 1,
        "status": "completed",
        "protocol_freeze": str(cli.freeze),
        "protocol_freeze_sha256": sha256_file(cli.freeze),
        "operating_point": budget,
        "unit_tying_ratio": freeze.get("unit_tying_ratio"),
        "seeds": seed_records,
        "test_used": True,
        "test_opened_only_after_freeze": True,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
