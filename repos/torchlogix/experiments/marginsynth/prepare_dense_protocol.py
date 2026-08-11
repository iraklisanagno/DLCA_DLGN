#!/usr/bin/env python3
"""Materialize a dense CIFAR MarginSynth protocol from the frozen trial-28 study."""

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


COMPONENT_OVERLAYS = {
    "current": {
        "liveness_mask": "none",
        "activity_ranking": "none",
    },
    "liveness": {
        "liveness_mask": "topological",
        "activity_ranking": "none",
    },
    "class_activity": {
        "liveness_mask": "none",
        "activity_ranking": "class-fold",
    },
    "constants_routing": {
        "action_space": "constants-routing",
        "liveness_mask": "none",
        "activity_ranking": "none",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--profile", choices=["smoke", "comparison", "transfer"], required=True)
    parser.add_argument(
        "--selected-component",
        choices=list(COMPONENT_OVERLAYS),
        default="liveness",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def relative_to_repository(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    cli = parse_args()
    source_run = cli.source_run.resolve()
    source_checkpoint = source_run / "best_checkpoint.pt"
    required = [source_run / "training_config.json", source_checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if cli.output.exists():
        raise RuntimeError(f"refusing to overwrite protocol: {cli.output}")
    training = json.loads((source_run / "training_config.json").read_text())
    if training["dataset"] != "cifar-10":
        raise ValueError("dense transfer protocol requires CIFAR-10")
    if training["connections"] != "fixed" or training["connections_init_method"] != "random":
        raise ValueError("primary dense transfer source must use fixed standard-random connectivity")
    if float(training.get("calibration_set_size", 0.0)) <= 0.0:
        raise ValueError("source training must exclude a nonempty calibration split")
    architecture = training["architecture"]
    if architecture not in {"DlgnCifar10Small", "DlgnCifar10Medium"}:
        raise ValueError(f"unsupported dense architecture: {architecture}")

    config_dir = Path(__file__).resolve().parent / "configs"
    frozen_path = config_dir / "components_fashion_trial28.json"
    template_path = config_dir / "dense_cifar_transfer_template.json"
    frozen = json.loads(frozen_path.read_text())
    template = json.loads(template_path.read_text())
    is_medium = architecture == "DlgnCifar10Medium"
    overrides = template["medium_overrides" if is_medium else "small_overrides"]

    first = copy.deepcopy(frozen["first_resynthesis"])
    second = copy.deepcopy(frozen["second_resynthesis"])
    silicon = copy.deepcopy(frozen["silicon_control"])
    for config in (first, second, silicon):
        config.update(copy.deepcopy(overrides))
        config["eligible_logic_layers"] = template["eligible_logic_layers"]
        config["source_checkpoint"] = "best_checkpoint.pt"
        config["teacher_checkpoint"] = "best_checkpoint.pt"
        config["development_run"] = cli.profile == "smoke"
        config["report_validation"] = True
    second.pop("lock_reference_checkpoint", None)

    selected = cli.selected_component
    if cli.profile == "smoke":
        components = {selected: COMPONENT_OVERLAYS[selected]}
    elif cli.profile == "comparison":
        names = ["current", "liveness", "class_activity"]
        if selected not in names:
            names.append(selected)
        components = {name: COMPONENT_OVERLAYS[name] for name in names}
    else:
        components = {selected: COMPONENT_OVERLAYS[selected]}

    seed = int(training["seed"])
    profile_root = {
        "smoke": "marginsynth_smoke/trial28_transfer_v1",
        "comparison": "marginsynth_comparison/trial28_transfer_v1",
        "transfer": "marginsynth_transfer/trial28_transfer_v1",
    }[cli.profile]
    unit_tying = copy.deepcopy(template["unit_tying"])
    unit_tying["output"] = f"{profile_root}/unit_tying"
    protocol = {
        "format_version": 1,
        "protocol_name": f"dense_cifar_{cli.profile}_{architecture}_seed{seed}_v1",
        "profile": cli.profile,
        "source_run": relative_to_repository(source_run),
        "source_checkpoint": "best_checkpoint.pt",
        "output_root": profile_root,
        "verification_split": "calibration",
        "verification_examples": 4992,
        "selected_component": selected,
        "exact_baseline": {},
        "two_pass_components": components,
        "first_resynthesis": first,
        "second_resynthesis": second,
        "frozen_transfer": {
            "fashion_protocol": relative_to_repository(frozen_path),
            "fashion_protocol_sha256": sha256_file(frozen_path),
            "dense_template": relative_to_repository(template_path),
            "dense_template_sha256": sha256_file(template_path),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "hyperparameter_policy": "trial-28 optimization hyperparameters are transferred without Bayesian retuning; only architecture-dependent evaluation/activity batch sizes and eligible layers change",
        },
        "data_policy": {
            "optimization_repair_guard": "60/20/20 stratified split of calibration",
            "validation_used_for_source_checkpoint_selection": True,
            "validation_used_for_resynthesis_selection": False,
            "test_used": False,
            "coverage_connectivity_used": False,
        },
        "continuation_criterion": {
            "selected_method_guard_feasible": True,
            "liveness_guard_feasible": True,
            "minimum_liveness_method_speedup_fraction": 0.10,
            "speed_scope": "sum of first- and second-pass total wall time, excluding one-time characterization and synthesis"
        },
    }
    if cli.profile in {"comparison", "transfer"}:
        protocol["unit_tying"] = unit_tying
    if cli.profile == "comparison":
        protocol["characterization"] = {
            "seed": 0,
            "partition_seed": 0,
            "checkpoint": "best_checkpoint.pt",
            "eligible_logic_layers": template["eligible_logic_layers"],
            "optimization_fraction": 0.6,
            "repair_fraction": 0.2,
            "guard_fraction": 0.2,
            "stability_folds": 4,
            "activity_batch_size": overrides["activity_batch_size"],
        }
        protocol["silicon_control"] = silicon

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
