#!/usr/bin/env python3
"""Characterize safe liveness and source-LUT activity on calibration data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENTS_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_model, load_dataset

from experiments.marginsynth.circuit_distillation import (
    stratified_optimization_repair_guard_split,
)
from experiments.marginsynth.liveness_activity import (
    collect_activity_risks,
    liveness_summary,
)
from experiments.marginsynth.margin_aware_tying import (
    stratified_fold_ids,
)
from experiments.marginsynth.unit_tying import logic_layers
from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    take_examples,
    tensor_sha256,
    write_artifact_manifest,
)


def tool_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()[:4000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("liveness/activity characterization requires CUDA")
    device = torch.device("cuda")
    run_dir = cli.run_dir.resolve()
    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text())
    output_dir = run_dir / config["output"]
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    training_config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**training_config)
    args.device = "cpu"
    checkpoint_path = run_dir / config.get("checkpoint", "best_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    thresholds = state["0.thresholds"]

    _, _, calibration_loader, _ = load_dataset(args, include_calibration=True)
    calibration_images, calibration_labels = take_examples(
        calibration_loader, len(calibration_loader.dataset)
    )
    encoder = get_model(thresholds, args)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    with torch.no_grad():
        calibration_encoded = encoder[0](calibration_images).bool().cpu()
    del encoder, calibration_images

    model = get_model(thresholds, args)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    layers = logic_layers(model)
    eligible_layers = [int(value) for value in config["eligible_logic_layers"]]
    if not eligible_layers or any(
        index < 0 or index >= len(layers) for index in eligible_layers
    ):
        raise ValueError("invalid eligible_logic_layers")
    all_ids = {
        index: layer.weight.detach().argmax(1).cpu()
        for index, layer in enumerate(layers)
    }
    graph_summary, _ = liveness_summary(layers, all_ids, eligible_layers)

    optimization, repair, guard = stratified_optimization_repair_guard_split(
        calibration_labels,
        float(config.get("optimization_fraction", 0.6)),
        float(config.get("repair_fraction", 0.2)),
        int(config.get("partition_seed", seed)) + 104729,
    )
    folds = torch.from_numpy(
        stratified_fold_ids(
            calibration_labels.numpy(),
            int(config.get("stability_folds", 4)),
            seed + 7919,
        )
    )
    activity_started = time.perf_counter()
    risks, activity_summary = collect_activity_risks(
        model,
        calibration_encoded,
        calibration_labels,
        folds,
        optimization,
        eligible_layers,
        int(config.get("activity_batch_size", 64)),
        device,
    )
    activity_summary["elapsed_seconds"] = time.perf_counter() - activity_started
    risk_path = output_dir / "activity_risks.pt"
    torch.save(
        {
            "format_version": 1,
            "risk_definition": "maximum source-to-candidate mismatch over global, class, and fold groups",
            "eligible_logic_layers": eligible_layers,
            "risks": risks,
        },
        risk_path,
    )

    sample_selection = {
        "partition": "calibration",
        "partition_indices_sha256": calibration_loader.split_manifest["partitions"]["calibration"]["indices_sha256"],
        "partition_size": len(calibration_labels),
        "optimization_size": len(optimization),
        "repair_size": len(repair),
        "guard_size": len(guard),
        "optimization_indices_sha256": tensor_sha256(optimization),
        "repair_indices_sha256": tensor_sha256(repair),
        "guard_indices_sha256": tensor_sha256(guard),
        "labels_sha256": tensor_sha256(calibration_labels),
        "fold_ids_sha256": tensor_sha256(folds),
        "validation_used": False,
        "test_used": False,
    }
    software = {
        "source_revision": git_revision(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(device),
        "yosys": tool_output(["yosys", "-V"]),
        "abc": tool_output(["berkeley-abc", "-h"]),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    resolved_config = config | {
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
    }
    summary = {
        "format_version": 1,
        "status": "completed",
        "method": "marginsynth-liveness-activity-characterization",
        "dataset": training_config["dataset"],
        "architecture": training_config["architecture"],
        "checkpoint": str(checkpoint_path.relative_to(run_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "nominal_logic_gates": sum(int(layer.out_dim) for layer in layers),
        "eligible_logic_layers": eligible_layers,
        "liveness": graph_summary,
        "activity": activity_summary,
        "sample_selection": sample_selection,
        "timing": {
            "activity_seconds": activity_summary["elapsed_seconds"],
            "total_seconds": time.perf_counter() - started,
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "activity_risks_sha256": sha256_file(risk_path),
        "software": software,
    }
    for name, payload in (
        ("config.json", resolved_config),
        ("sample_selection.json", sample_selection),
        ("liveness_analysis.json", graph_summary),
        ("activity_analysis.json", activity_summary),
        ("software.json", software),
        ("summary.json", summary),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    write_artifact_manifest(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
