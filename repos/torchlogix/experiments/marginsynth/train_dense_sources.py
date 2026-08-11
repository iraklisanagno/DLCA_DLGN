#!/usr/bin/env python3
"""Regenerate standard-random dense CIFAR sources with sealed calibration data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def repository_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed sources; partial training outputs are never resumed",
    )
    return parser.parse_args()


def validate_source(output: Path, expected: dict) -> dict:
    required = [
        output / "best_checkpoint.pt",
        output / "training_config.json",
        output / "run_summary.json",
        output / "data_split.json",
        output / "environment.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    config = json.loads((output / "training_config.json").read_text())
    invariants = {
        "dataset": "cifar-10",
        "architecture": expected["architecture"],
        "seed": int(expected["seed"]),
        "topology_seed": int(expected["seed"]),
        "data_split_seed": int(expected["data_split_seed"]),
        "connections": "fixed",
        "connections_init_method": "random",
        "calibration_set_size": float(expected["calibration_set_size"]),
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in invariants.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source configuration invariant failure: {mismatches}")
    split = json.loads((output / "data_split.json").read_text())
    calibration = split["partitions"]["calibration"]
    if not calibration["size"]:
        raise ValueError("calibration partition is empty")
    return {
        "source_run": str(output),
        "checkpoint": str(output / "best_checkpoint.pt"),
        "checkpoint_sha256": sha256_file(output / "best_checkpoint.pt"),
        "training_config_sha256": sha256_file(output / "training_config.json"),
        "run_summary_sha256": sha256_file(output / "run_summary.json"),
        "data_split_sha256": sha256_file(output / "data_split.json"),
        "calibration_size": calibration["size"],
        "calibration_indices_sha256": calibration["indices_sha256"],
        "best_validation_hard_accuracy": json.loads(
            (output / "run_summary.json").read_text()
        )["best_validation_hard_accuracy"],
        "invariants": invariants,
    }


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("dense CIFAR source training requires CUDA")
    protocol_path = cli.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    declared = protocol["sources"]
    selected = list(declared) if cli.sources is None else cli.sources
    unknown = sorted(set(selected) - set(declared))
    if unknown:
        raise ValueError(f"unknown source names: {unknown}")
    dataset_path = repository_path(protocol["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    execution_root = repository_path(protocol["execution_root"])
    execution_root.mkdir(parents=True, exist_ok=True)
    execution_path = execution_root / "source_training_execution.json"
    execution = []
    if cli.resume and execution_path.exists():
        execution = json.loads(execution_path.read_text())

    for name in selected:
        source = declared[name]
        base_config = repository_path(source["base_config"])
        historical_run = repository_path(source["historical_run"])
        output = repository_path(source["output"])
        completed = output / "best_checkpoint.pt"
        if completed.exists():
            if not cli.resume:
                raise RuntimeError(f"refusing to overwrite completed source: {output}")
            validation = validate_source(output, source)
            execution.append(
                {"source": name, "status": "reused-completed", **validation}
            )
            write_json(execution_path, execution)
            continue
        if output.exists():
            raise RuntimeError(
                f"partial source directory cannot be resumed safely: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "experiments/train.py"),
            "--config",
            str(base_config),
            "--seed",
            str(source["seed"]),
            "--topology-seed",
            str(source["seed"]),
            "--data-split-seed",
            str(source["data_split_seed"]),
            "--calibration-set-size",
            str(source["calibration_set_size"]),
            "--output",
            str(output),
        ]
        env = os.environ.copy()
        env["DATASET_PATH"] = str(dataset_path)
        started = time.perf_counter()
        output.mkdir()
        console_path = output / "console.log"
        with console_path.open("w") as handle:
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
        record = {
            "source": name,
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
            "dataset_path": str(dataset_path),
            "base_config": str(base_config),
            "base_config_sha256": sha256_file(base_config),
            "historical_run": str(historical_run),
            "historical_training_config_sha256": sha256_file(
                historical_run / "training_config.json"
            ),
            "historical_run_summary_sha256": sha256_file(
                historical_run / "run_summary.json"
            ),
            "console_log": str(console_path),
            "console_log_sha256": sha256_file(console_path),
        }
        execution.append(record)
        write_json(execution_path, execution)
        if result.returncode:
            raise RuntimeError(f"training {name} failed; see {console_path}")
        validation = validate_source(output, source)
        provenance = {
            "format_version": 1,
            "status": "completed",
            "purpose": "standard-random dense CIFAR source with calibration excluded from training",
            "source_name": name,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "historical_reference": {
                "run": str(historical_run),
                "training_config_sha256": record["historical_training_config_sha256"],
                "run_summary_sha256": record["historical_run_summary_sha256"],
                "checkpoint_available": False,
            },
            "intentional_difference": {
                "field": "calibration_set_size",
                "historical": 0.0,
                "regenerated": float(source["calibration_set_size"]),
                "reason": "post-training selection requires data excluded from source training and validation-based model selection",
            },
            "validation": validation,
        }
        write_json(output / "source_regeneration_provenance.json", provenance)
        execution[-1].update(validation)
        write_json(execution_path, execution)

    summary = {
        "format_version": 1,
        "status": "completed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selected_sources": selected,
        "executions": execution,
        "gpu": torch.cuda.get_device_name(),
        "test_used_for_selection": False,
    }
    write_json(execution_root / "source_training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
