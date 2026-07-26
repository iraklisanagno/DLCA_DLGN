#!/usr/bin/env python3
"""Generate the reduced one-seed CIFAR-10 L CoverageDLGN screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "table2_l_screen"
QUEUE_PATH = ROOT / "queues" / "table2_l_screen.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")

# Existing V3 controls only. This reduced set retains the incumbent, the
# smaller candidate pool, the successful 0.50 swap setting, and both published
# gate parameterizations without introducing a new topology score or branch.
COVERAGE_CANDIDATES = (
    ("incumbent", 8, 0.25, "raw"),
    ("pool4", 4, 0.25, "raw"),
    ("swap0500", 8, 0.50, "raw"),
    ("light", 8, 0.25, "light"),
    ("warp", 8, 0.25, "warp"),
)


def common(name: str) -> dict:
    return {
        "dataset": "cifar-10",
        "architecture": "DlgnCifar10Large",
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "batch_size": 100,
        "num_iterations": 5_000,
        "eval_freq": 500,
        "valid_set_size": 0.1,
        "augmentation": "none",
        "learning_rate": 0.01,
        "binarization_num_batches": 100,
        "binarization": "fixed",
        "binarization_init": "uniform",
        "binarization_per": "global",
        "connections": "fixed",
        "connections_gumbel": False,
        "forward_sampling": "soft",
        "weight_init": "random",
        "lut_rank": 2,
        "output": str(RESULT_ROOT / name),
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = []

    random_name = "screen_table2_cifar10_l_random_seed0"
    random_config = common(random_name)
    random_config.update({
        "connections_init_method": "random",
        "parametrization": "raw",
    })
    generated.append({
        "name": random_name,
        "family": "random",
        "candidate": "random",
        "config_payload": random_config,
    })

    for label, pool, swap, parametrization in COVERAGE_CANDIDATES:
        name = f"screen_table2_cifar10_l_v3_{label}_seed0"
        config = common(name)
        config.update({
            "connections_init_method": "semantic_balanced_hybrid",
            "coverage_candidate_pool_size": pool,
            "coverage_swap_fraction": swap,
            "coverage_novelty_weight": 1.0,
            "parametrization": parametrization,
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
        "phase": "table2_l_screen",
        "purpose": "one-seed 5K validation screen; not a final result",
        "selection_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "architecture": {
            "name": "DlgnCifar10Large",
            "depth": 5,
            "width": 256_000,
            "gate_count": 1_280_000,
            "input_bits": 5,
        },
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
