#!/usr/bin/env python3
"""Generate the paired three-seed CIFAR-100 S 20K selection queue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "summary" / "table4_cifar100_s_screen.json"
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_s_selection"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_s_selection.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def common(name: str, seed: int) -> dict:
    return {
        "dataset": "cifar-100",
        "architecture": "DlgnCifar100BitLogicS",
        "device": "cuda",
        "seed": seed,
        "data_split_seed": 2027,
        "topology_seed": seed,
        "batch_size": 128,
        "num_iterations": 20_000,
        "eval_freq": 2_000,
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
    screen = json.loads(SCREEN.read_text())
    winner = next(
        row for row in screen["rows"]
        if row["name"] == screen["selected_raw_v3"]
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    for seed in (0, 1, 2):
        random_name = f"select_table4_cifar100_s_random_seed{seed}"
        random_config = common(random_name, seed)
        random_config["connections_init_method"] = "random"
        generated.append({
            "name": random_name,
            "family": "random",
            "candidate": "random",
            "seed": seed,
            "config_payload": random_config,
        })

        v3_name = (
            f"select_table4_cifar100_s_v3_{winner['candidate']}_seed{seed}"
        )
        v3_config = common(v3_name, seed)
        v3_config.update({
            "connections_init_method": "semantic_balanced_hybrid",
            "coverage_candidate_pool_size": winner[
                "coverage_candidate_pool_size"
            ],
            "coverage_swap_fraction": winner["coverage_swap_fraction"],
            "coverage_novelty_weight": winner[
                "coverage_novelty_weight"
            ],
        })
        generated.append({
            "name": v3_name,
            "family": "coverage_v3",
            "candidate": winner["candidate"],
            "seed": seed,
            "config_payload": v3_config,
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
        "phase": "table4_cifar100_s_selection",
        "purpose": "paired three-seed 20K validation selection",
        "selection_metric": "mean best hardened validation accuracy",
        "heldout_test_used": False,
        "screen_source": str(SCREEN.relative_to(ROOT.parent.parent)),
        "selected_v3_candidate": winner["candidate"],
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
