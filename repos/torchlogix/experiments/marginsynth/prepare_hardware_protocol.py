#!/usr/bin/env python3
"""Freeze dense-CIFAR hardware-ranking ablation or seed-transfer protocols."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


ABLATION_OVERLAYS = {
    "focused_control": {
        "activity_ranking": "none",
        "alternative_binary_penalty": 2.0,
    },
    "hardware_aware": {
        "activity_ranking": "hardware",
        "alternative_binary_penalty": 2.0,
    },
    "class_hardware_aware": {
        "activity_ranking": "class-fold-hardware",
        "alternative_binary_penalty": 2.0,
        "hardware_rank_weight": 0.75,
        "activity_rank_weight": 0.25,
    },
}


def relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--profile", choices=["ablation", "transfer"], required=True)
    parser.add_argument("--hardware-model", required=True, type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    hardware_model = cli.hardware_model.resolve()
    if cli.output.exists():
        raise RuntimeError(f"refusing to overwrite protocol: {cli.output}")
    required = [
        source_run / "training_config.json",
        source_run / "best_checkpoint.pt",
        hardware_model,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    training = json.loads((source_run / "training_config.json").read_text())
    if training["dataset"] != "cifar-10":
        raise ValueError("hardware-ranking protocol requires CIFAR-10")
    if training["connections"] != "fixed" or training["connections_init_method"] != "random":
        raise ValueError("primary protocol requires fixed standard-random connectivity")

    config_dir = Path(__file__).resolve().parent / "configs"
    frozen_path = config_dir / "components_fashion_trial28.json"
    template_path = config_dir / "dense_cifar_transfer_template.json"
    frozen = json.loads(frozen_path.read_text())
    template = json.loads(template_path.read_text())
    architecture = training["architecture"]
    overrides = template[
        "medium_overrides"
        if architecture == "DlgnCifar10Medium"
        else "small_overrides"
    ]
    first = copy.deepcopy(frozen["first_resynthesis"])
    second = copy.deepcopy(frozen["second_resynthesis"])
    for config in (first, second):
        config.update(copy.deepcopy(overrides))
        config["eligible_logic_layers"] = template["eligible_logic_layers"]
        config["source_checkpoint"] = "best_checkpoint.pt"
        config["teacher_checkpoint"] = "best_checkpoint.pt"
        config["report_validation"] = True
        config["development_run"] = cli.profile == "ablation"
        config["hardware_ranking_model"] = relative(hardware_model)
        config["liveness_mask"] = "none"
    second.pop("lock_reference_checkpoint", None)

    freeze_payload = None
    if cli.profile == "ablation":
        if cli.freeze is not None:
            raise ValueError("--freeze is only valid for transfer")
        components = copy.deepcopy(ABLATION_OVERLAYS)
        output_root = "marginsynth_hardware_ablation/structural_rank_v1"
    else:
        if cli.freeze is None:
            raise ValueError("transfer requires --freeze")
        freeze_path = cli.freeze.resolve()
        freeze_payload = json.loads(freeze_path.read_text())
        if freeze_payload.get("status") != "frozen":
            raise ValueError("transfer freeze record is not frozen")
        selected = freeze_payload["selected_component"]
        if selected not in ABLATION_OVERLAYS:
            raise ValueError(f"unknown frozen component: {selected}")
        if freeze_payload["hardware_ranking_model_sha256"] != sha256_file(hardware_model):
            raise ValueError("hardware-ranking model differs from frozen model")
        components = {selected: copy.deepcopy(ABLATION_OVERLAYS[selected])}
        output_root = "marginsynth_hardware_transfer/structural_rank_v1"

    seed = int(training["seed"])
    protocol = {
        "format_version": 1,
        "protocol_name": f"marginsynth_hardware_{cli.profile}_{architecture}_seed{seed}_v1",
        "profile": cli.profile,
        "source_run": relative(source_run),
        "source_checkpoint": "best_checkpoint.pt",
        "output_root": output_root,
        "verification_split": "calibration",
        "verification_examples": 4992,
        "snapshot_selection": True,
        "two_pass_components": components,
        "first_resynthesis": first,
        "second_resynthesis": second,
        "frozen_method": {
            "fashion_trial28_protocol": relative(frozen_path),
            "fashion_trial28_protocol_sha256": sha256_file(frozen_path),
            "dense_template": relative(template_path),
            "dense_template_sha256": sha256_file(template_path),
            "hardware_ranking_model": relative(hardware_model),
            "hardware_ranking_model_sha256": sha256_file(hardware_model),
            "alternative_binary_policy": (
                "All 16 LUTs remain available, but a two-AIG-unit penalty is "
                "applied only to alternative binary LUTs; original LUTs are exempt."
            ),
            "snapshot_policy": (
                "Select source/first/second using calibration repair and guard "
                "feasibility plus frozen estimated hardware gain only."
            ),
            "freeze_record": None if cli.freeze is None else relative(cli.freeze),
            "freeze_record_sha256": (
                None if cli.freeze is None else sha256_file(cli.freeze.resolve())
            ),
        },
        "selection_rule": (
            "On seed 0, among guard-feasible MarginSynth variants minimize exact "
            "ABC AND nodes; break ties by guard worst-class accuracy loss, global "
            "guard accuracy loss, method time, and component name. Validation is "
            "reported but not used for selection."
        ),
        "data_policy": {
            "optimization_repair_guard": "60/20/20 stratified calibration split",
            "validation_used_for_source_checkpoint_selection": True,
            "validation_used_for_resynthesis_selection": False,
            "test_used": False,
            "coverage_connectivity_used": False,
        },
    }
    if freeze_payload is not None:
        protocol["frozen_transfer_selection"] = freeze_payload
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
