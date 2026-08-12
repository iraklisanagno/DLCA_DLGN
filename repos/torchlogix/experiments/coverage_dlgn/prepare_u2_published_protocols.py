#!/usr/bin/env python3
"""Generate the frozen dense U2 and published-protocol comparison queues."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs" / "u2_published_protocols"
QUEUE_ROOT = ROOT / "queues"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def base(name: str, seed: int) -> dict:
    return {
        "dataset": "cifar-10",
        "device": "cuda",
        "seed": seed,
        "data_split_seed": 2027,
        "topology_seed": seed,
        "valid_set_size": 0.1,
        "binarization": "fixed",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_gumbel": False,
        "forward_sampling": "soft",
        "weight_init": "random",
        "output": str(RESULT_ROOT / name),
    }


def current_u2_entries() -> list[dict]:
    entries = []
    for label, architecture in (
        ("m", "DlgnCifar10Medium"),
        ("l", "DlgnCifar10Large"),
    ):
        for seed in range(3):
            name = f"third_u2_cifar10_{label}_seed{seed}"
            config = base(name, seed)
            config.update({
                "architecture": architecture,
                "batch_size": 100,
                "num_iterations": 108_000,
                "eval_freq": 2_000,
                "augmentation": "none",
                "learning_rate": 0.01,
                "binarization_num_batches": 100,
                "binarization_init": "uniform",
                "connections_init_method": "semantic_multiscale_balanced",
                "lut_rank": 2,
                "parametrization": "raw",
            })
            entries.append(entry(name, "u2", label, seed, config))
    return entries


def lilogic_entries() -> list[dict]:
    entries = []
    coordinates = (
        ("m", "DlgnCifar10LilogicM", "DlgnCifar10LilogicMTop32", 90.0),
        ("l", "DlgnCifar10LilogicL", "DlgnCifar10LilogicLTop32", 100.0),
    )
    for label, fixed_arch, top32_arch, tau in coordinates:
        for family, strategy in (
            ("random", "random"),
            ("u2", "semantic_multiscale_balanced"),
        ):
            for seed in range(3):
                name = f"third_lilogic_{label}_{family}_seed{seed}"
                config = lilogic_common(name, seed, fixed_arch, tau)
                config["connections_init_method"] = strategy
                entries.append(entry(name, family, label, seed, config))

        # The expensive comparator policy is one local seed plus the paper's
        # five-run reported distribution.  Per-run effort remains identical.
        seed = 0
        name = f"third_lilogic_{label}_top32_seed{seed}"
        config = lilogic_common(name, seed, top32_arch, tau)
        config.update({
            "connections_init_method": "random",
            "connections_num_candidates": 32,
            "connections_forward_mode": "soft_mix",
            "connections_weights_init": "normal",
            "connections_temperature": 1.0,
            "parametrization_temperature": 1.0,
        })
        entries.append(entry(name, "top32", label, seed, config))
    return entries


def lilogic_common(
    name: str,
    seed: int,
    architecture: str,
    group_sum_temperature: float,
) -> dict:
    config = base(name, seed)
    config.update({
        "architecture": architecture,
        "batch_size": 256,
        # floor(45,000 / 256) * 200 base-dataset epochs.
        "num_iterations": 35_000,
        "eval_freq": 4_375,
        "augmentation": "bitlogic",
        "learning_rate": 0.075,
        "group_sum_temperature": group_sum_temperature,
        "binarization_num_batches": 100,
        "binarization_init": "uniform",
        "connections_temperature": 1.0,
        "parametrization_temperature": 1.0,
        "lut_rank": 2,
        "parametrization": "raw",
    })
    return config


def bitlogic_entries() -> list[dict]:
    entries = []
    coordinates = (
        ("s", "DlgnCifar10BitLogicRank2S", "DlgnCifar10BitLogicBestS"),
        ("m", "DlgnCifar10BitLogicRank2M", "DlgnCifar10BitLogicBestM"),
        ("l", "DlgnCifar10BitLogicRank2L", "DlgnCifar10BitLogicBestL"),
    )
    for label, rank2_arch, best_arch in coordinates:
        for family, architecture, strategy in (
            ("random", rank2_arch, "random"),
            ("u2", rank2_arch, "semantic_multiscale_balanced"),
            ("best", best_arch, "random"),
        ):
            for seed in (0, 1):
                name = f"third_bitlogic_{label}_{family}_seed{seed}"
                config = base(name, seed)
                config.update({
                    "architecture": architecture,
                    "batch_size": 128,
                    # floor(45,000 / 128) * 100 epochs.
                    "num_iterations": 35_100,
                    "eval_freq": 3_510,
                    "augmentation": "bitlogic",
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "binarization_num_batches": 100,
                    "connections_init_method": strategy,
                })
                if family == "best":
                    config.update({
                        "binarization_init": "distributive",
                        "connections_num_candidates": 16,
                        "connections_forward_mode": "soft_mix",
                        "connections_weights_init": "normal",
                        "connections_temperature": 1.0,
                        "lut_rank": 4,
                        "parametrization": "light",
                        "parametrization_temperature": 1.0,
                    })
                else:
                    config.update({
                        "binarization_init": "uniform",
                        "lut_rank": 2,
                        "parametrization": "raw",
                    })
                entries.append(entry(name, family, label, seed, config))
    return entries


def smoke_entries() -> list[dict]:
    """One CUDA construction/training probe for each high-risk coordinate."""
    selected = [
        row for row in (
            current_u2_entries()
            + lilogic_entries()
            + bitlogic_entries()
        )
        if row["name"] in {
            "third_u2_cifar10_m_seed0",
            "third_u2_cifar10_l_seed0",
            "third_lilogic_m_u2_seed0",
            "third_lilogic_l_top32_seed0",
            "third_bitlogic_s_u2_seed0",
            "third_bitlogic_l_best_seed0",
        }
    ]
    smokes = []
    for row in selected:
        config = dict(row["config_payload"])
        name = "smoke_" + row["name"]
        config.update({
            "num_iterations": 10,
            "eval_freq": 10,
            "binarization_num_batches": 1,
            "output": str(RESULT_ROOT / name),
        })
        smokes.append(entry(
            name,
            row["family"],
            row["coordinate"],
            row["seed"],
            config,
        ))
    return smokes


def entry(
    name: str,
    family: str,
    coordinate: str,
    seed: int,
    config: dict,
) -> dict:
    return {
        "name": name,
        "family": family,
        "coordinate": coordinate,
        "seed": seed,
        "config_payload": config,
    }


def write_phase(phase: str, entries: list[dict], purpose: str) -> None:
    config_dir = CONFIG_ROOT / phase
    config_dir.mkdir(parents=True, exist_ok=True)
    queue_entries = []
    expected = set()
    for row in entries:
        row = dict(row)
        config = row.pop("config_payload")
        path = config_dir / f"{row['name']}.json"
        expected.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        queue_entries.append({
            **row,
            "config": str(path.relative_to(ROOT.parent.parent)),
            "output": config["output"],
        })
    stale = sorted(set(config_dir.glob("*.json")) - expected)
    if stale:
        raise RuntimeError(
            "Refusing to delete stale generated configs: "
            + ", ".join(str(path) for path in stale)
        )
    payload = {
        "phase": phase,
        "purpose": purpose,
        "cuda_required": True,
        "heldout_test_used": False,
        "selection_metric": "best hardened validation accuracy",
        "test_policy": (
            "evaluate predeclared best-validation and final checkpoints once "
            "after every full run in the phase completes"
        ),
        "entries": queue_entries,
    }
    path = QUEUE_ROOT / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs: {path}")


def main() -> None:
    write_phase(
        "third_u2_dense_cifar10_ml",
        current_u2_entries(),
        "Frozen U2 transfer to the existing Deep-DLGN CIFAR-10 M/L cells",
    )
    write_phase(
        "third_lilogic_cifar10",
        lilogic_entries(),
        "Direct fixed-random/U2/Top-32 comparison on LILogicNet M/L protocols",
    )
    write_phase(
        "third_bitlogic_cifar10",
        bitlogic_entries(),
        "Direct rank-2 U2 and rank-4 Pareto comparison on BitLogic Table 6",
    )
    write_phase(
        "third_u2_published_smoke",
        smoke_entries(),
        "CUDA-only construction, memory, and ten-step training probes",
    )


if __name__ == "__main__":
    main()
