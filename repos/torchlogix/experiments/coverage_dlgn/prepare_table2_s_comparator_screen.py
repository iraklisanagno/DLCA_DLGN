#!/usr/bin/env python3
"""Generate the reduced CIFAR-10 S comparator screen.

This screen does not touch CoverageDLGN. Mommen compares only Nc=8 and Nc=16
on the exact 48K S architecture. LILogicNet has one fixed matched-budget
adaptation and is included only for implementation/feasibility validation.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "table2_s_comparator_screen"
QUEUE_PATH = ROOT / "queues" / "table2_s_comparator_screen.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def common(name: str, batch_size: int = 100) -> dict:
    return {
        "dataset": "cifar-10",
        "architecture": "DlgnCifar10SmallLearnable",
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "batch_size": batch_size,
        "num_iterations": 5_000,
        "eval_freq": 500,
        "valid_set_size": 0.1,
        "augmentation": "none",
        "binarization_num_batches": 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_init_method": "random",
        "connections_gumbel": False,
        "connections_forward_mode": "soft_mix",
        "connections_weights_init": "normal",
        "connections_temperature": 1.0,
        "forward_sampling": "soft",
        "weight_init": "random",
        "lut_rank": 2,
        "parametrization": "raw",
        "parametrization_temperature": 1.0,
        "output": str(RESULT_ROOT / name),
    }


def entries() -> list[tuple[str, str, dict]]:
    generated = []
    for candidate_count in (8, 16):
        name = f"screen_table2_cifar10_s_mommen_nc{candidate_count}_seed0"
        config = common(name)
        config.update({
            "learning_rate": 0.01,
            "connections_num_candidates": candidate_count,
            "connections_temperature_final": 0.0001,
            "connections_temperature_anneal_start": 0.6666666667,
            "connections_temperature_anneal_end": 0.8333333333,
            "parametrization_temperature_final": 0.0001,
            "parametrization_temperature_anneal_start": 0.8333333333,
            "parametrization_temperature_anneal_end": 1.0,
        })
        generated.append((name, "mommen", config))

    name = "screen_table2_cifar10_s_lilogic_top32_tau30_seed0"
    config = common(name, batch_size=256)
    config.update({
        "learning_rate": 0.075,
        "group_sum_temperature": 30.0,
        "connections_num_candidates": 32,
    })
    generated.append((name, "lilogic", config))
    return generated


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []
    for name, family, config in entries():
        path = CONFIG_DIR / f"{name}.json"
        expected_paths.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        queue.append({
            "name": name,
            "family": family,
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
        "phase": "table2_s_comparator_screen",
        "selection_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "reduced_search_policy": {
            "mommen": "one seed each for Nc=8 and Nc=16",
            "lilogic": "one fixed paper-derived matched-budget recipe",
        },
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
