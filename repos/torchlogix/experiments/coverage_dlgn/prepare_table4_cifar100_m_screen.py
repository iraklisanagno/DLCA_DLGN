#!/usr/bin/env python3
"""Generate the diagnostic one-seed dense CIFAR-100 M topology screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_m_screen"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_m_screen.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")

COVERAGE_CANDIDATES = (
    ("incumbent", 8, 0.25, 1.0),
    ("pool4", 4, 0.25, 1.0),
    ("pool16", 16, 0.25, 1.0),
    ("swap0125", 8, 0.125, 1.0),
    ("swap0500", 8, 0.50, 1.0),
    ("novelty050", 8, 0.25, 0.50),
    ("novelty200", 8, 0.25, 2.0),
)


def common(name: str) -> dict:
    return {
        "dataset": "cifar-100",
        "architecture": "DlgnCifar100BitLogicM",
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "batch_size": 128,
        "num_iterations": 5_000,
        "eval_freq": 500,
        "valid_set_size": 0.1,
        "augmentation": "bitlogic",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "binarization_num_batches": 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_gumbel": False,
        "forward_sampling": "soft",
        "weight_init": "random",
        "lut_rank": 2,
        "parametrization": "raw",
        "output": str(RESULT_ROOT / name),
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = []

    random_name = "screen_table4_cifar100_m_random_seed0"
    random_config = common(random_name)
    random_config["connections_init_method"] = "random"
    generated.append({
        "name": random_name,
        "family": "random",
        "candidate": "random",
        "config_payload": random_config,
    })

    for label, pool, swap, novelty in COVERAGE_CANDIDATES:
        name = f"screen_table4_cifar100_m_v3_{label}_seed0"
        config = common(name)
        config.update({
            "connections_init_method": "semantic_balanced_hybrid",
            "coverage_candidate_pool_size": pool,
            "coverage_swap_fraction": swap,
            "coverage_novelty_weight": novelty,
        })
        generated.append({
            "name": name,
            "family": "coverage_v3",
            "candidate": label,
            "config_payload": config,
        })

    expected_paths = set()
    queue = []
    for entry in generated:
        config = entry.pop("config_payload")
        path = CONFIG_DIR / f"{entry['name']}.json"
        expected_paths.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        queue.append({
            **entry,
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
        "phase": "table4_cifar100_m_screen",
        "purpose": (
            "one-seed 5K diagnostic after undersubscribed S failed; not final"
        ),
        "selection_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "s_promotion_condition_met": False,
        "diagnostic_rationale": (
            "M has 32K first-layer slots for 9,216 encoded inputs"
        ),
        "architecture": {
            "name": "DlgnCifar100BitLogicM",
            "depth": 2,
            "width": 16_000,
            "gate_count": 32_000,
            "lut_rank": 2,
            "input_thresholds": 3,
            "class_count": 100,
        },
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
