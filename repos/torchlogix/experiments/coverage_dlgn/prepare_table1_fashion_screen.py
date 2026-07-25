#!/usr/bin/env python3
"""Generate the locked one-seed Table 1 Fashion-MNIST screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "table1_screen_fashion"
QUEUE_PATH = ROOT / "queues" / "table1_screen_fashion.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def common(name: str, *, batch_size: int = 100) -> dict:
    return {
        "dataset": "fashion-mnist",
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "batch_size": batch_size,
        "num_iterations": 5000,
        "eval_freq": 500,
        "valid_set_size": 0.1,
        "augmentation": "none",
        "binarization_num_batches": 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections_gumbel": False,
        "forward_sampling": "soft",
        "weight_init": "random",
        "output": str(RESULT_ROOT / name),
    }


def fixed_random() -> tuple[str, dict]:
    name = "screen_table1_fashion_random_48k_seed0"
    config = common(name)
    config.update({
        "architecture": "DlgnFashionMnistPaperSmall",
        "learning_rate": 0.01,
        "connections": "fixed",
        "connections_init_method": "random",
        "lut_rank": 2,
        "parametrization": "raw",
    })
    return name, config


def coverage_configs() -> list[tuple[str, dict]]:
    candidates = [
        ("incumbent", 8, 0.25, 1.0, "raw"),
        ("pool16", 16, 0.25, 1.0, "raw"),
        ("pool32", 32, 0.25, 1.0, "raw"),
        ("swap0125", 8, 0.125, 1.0, "raw"),
        ("swap0500", 8, 0.5, 1.0, "raw"),
        ("novelty050", 8, 0.25, 0.5, "raw"),
        ("novelty200", 8, 0.25, 2.0, "raw"),
        ("light", 8, 0.25, 1.0, "light"),
        ("warp", 8, 0.25, 1.0, "warp"),
    ]
    configs = []
    for label, pool, swap, novelty, parametrization in candidates:
        name = f"screen_table1_fashion_v3_{label}_48k_seed0"
        config = common(name)
        config.update({
            "architecture": "DlgnFashionMnistPaperSmall",
            "learning_rate": 0.01,
            "connections": "fixed",
            "connections_init_method": "semantic_balanced_hybrid",
            "coverage_candidate_pool_size": pool,
            "coverage_swap_fraction": swap,
            "coverage_novelty_weight": novelty,
            "lut_rank": 2,
            "parametrization": parametrization,
        })
        configs.append((name, config))
    return configs


def learnable_architecture(depth: int) -> str:
    return {
        2: "DlgnFashionMnistLearnable48kDepth2",
        3: "DlgnFashionMnistLearnable48kDepth3",
        6: "DlgnFashionMnistPaperSmallLearnable",
    }[depth]


def mommen_configs() -> list[tuple[str, dict]]:
    configs = []
    for candidate_count in (8, 16):
        for depth in (2, 3, 6):
            name = (
                f"screen_table1_fashion_mommen_nc{candidate_count}_"
                f"depth{depth}_48k_seed0"
            )
            config = common(name)
            config.update({
                "architecture": learnable_architecture(depth),
                "learning_rate": 0.01,
                "connections": "fixed",
                "connections_init_method": "random",
                "connections_num_candidates": candidate_count,
                "connections_forward_mode": "soft_mix",
                "connections_weights_init": "normal",
                "connections_temperature": 1.0,
                "connections_temperature_final": 0.0001,
                "connections_temperature_anneal_start": 0.6666666667,
                "connections_temperature_anneal_end": 0.8333333333,
                "parametrization_temperature": 1.0,
                "parametrization_temperature_final": 0.0001,
                "parametrization_temperature_anneal_start": 0.8333333333,
                "parametrization_temperature_anneal_end": 1.0,
                "lut_rank": 2,
                "parametrization": "raw",
            })
            configs.append((name, config))
    return configs


def lilogic_configs() -> list[tuple[str, dict]]:
    configs = []
    for depth in (2, 3, 6):
        for group_sum_temperature in (15, 20, 25, 30):
            name = (
                f"screen_table1_fashion_lilogic_top32_depth{depth}_"
                f"tau{group_sum_temperature}_48k_seed0"
            )
            config = common(name, batch_size=256)
            config.update({
                "architecture": learnable_architecture(depth),
                "learning_rate": 0.075,
                "group_sum_temperature": float(group_sum_temperature),
                "connections": "fixed",
                "connections_init_method": "random",
                "connections_num_candidates": 32,
                "connections_forward_mode": "soft_mix",
                "connections_weights_init": "normal",
                "connections_temperature": 1.0,
                "parametrization_temperature": 1.0,
                "lut_rank": 2,
                "parametrization": "raw",
            })
            configs.append((name, config))
    return configs


def bitlogic_config() -> tuple[str, dict]:
    name = "screen_table1_fashion_bitlogic_48k_seed0"
    config = common(name, batch_size=128)
    config.update({
        "architecture": "DlgnFashionMnistBitLogic48k",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "binarization_num_batches": 100,
        "binarization_init": "distributive",
        "connections": "fixed",
        "connections_init_method": "random",
        "connections_num_candidates": 16,
        "connections_forward_mode": "soft_mix",
        "connections_weights_init": "normal",
        "connections_temperature": 1.0,
        "lut_rank": 4,
        "parametrization": "light",
        "parametrization_temperature": 1.0,
    })
    return name, config


def main() -> None:
    entries = (
        [fixed_random()]
        + coverage_configs()
        + mommen_configs()
        + lilogic_configs()
        + [bitlogic_config()]
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []
    for name, config in entries:
        path = CONFIG_DIR / f"{name}.json"
        expected_paths.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        queue.append({
            "name": name,
            "config": str(path.relative_to(ROOT.parent.parent)),
            "output": config["output"],
        })
    stale = sorted(set(CONFIG_DIR.glob("*.json")) - expected_paths)
    if stale:
        raise RuntimeError(
            "Refusing to delete stale generated configs: "
            + ", ".join(str(path) for path in stale)
        )
    payload = {
        "phase": "table1_screen_fashion",
        "implementation_parent": "44e6126",
        "source_revision_policy": (
            "exact launch revision is recorded by each environment.json"
        ),
        "selection_metric": "best_validation_hard_accuracy",
        "test_set_used": False,
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
