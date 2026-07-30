#!/usr/bin/env python3
"""Generate one-seed screens for the deeper dense CIFAR-100 coordinates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "table4_dense_cifar100_deep.json"
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_deep_screen"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_screen.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def common(label: str, architecture: dict) -> dict:
    name = f"screen_table4_cifar100_{label}_random_seed0"
    paper = architecture["paper_training"]
    return {
        "dataset": "cifar-100",
        "architecture": architecture["class"],
        "device": "cuda",
        "seed": 0,
        "data_split_seed": 2027,
        "topology_seed": 0,
        "batch_size": paper["batch_size"],
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
        "parametrization": "raw",
        "output": str(RESULT_ROOT / name),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    for label, architecture in protocol["architectures"].items():
        random_name = f"screen_table4_cifar100_{label}_random_seed0"
        random_config = common(label, architecture)
        random_config["connections_init_method"] = "random"
        generated.append({
            "name": random_name,
            "architecture_label": label,
            "family": "random",
            "candidate": "random",
            "config_payload": random_config,
        })
        for candidate in protocol["method"][
            "controls_screened_per_architecture"
        ]:
            name = (
                f"screen_table4_cifar100_{label}_v3_"
                f"{candidate['label']}_seed0"
            )
            config = common(label, architecture)
            config.update({
                "connections_init_method": "semantic_balanced_hybrid",
                "coverage_candidate_pool_size": candidate[
                    "coverage_candidate_pool_size"
                ],
                "coverage_swap_fraction": candidate[
                    "coverage_swap_fraction"
                ],
                "coverage_novelty_weight": candidate[
                    "coverage_novelty_weight"
                ],
                "output": str(RESULT_ROOT / name),
            })
            generated.append({
                "name": name,
                "architecture_label": label,
                "family": "coverage_v3",
                "candidate": candidate["label"],
                "config_payload": config,
            })

    expected_paths = set()
    entries = []
    for entry in generated:
        config = entry.pop("config_payload")
        path = CONFIG_DIR / f"{entry['name']}.json"
        expected_paths.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        entries.append({
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
        "phase": "table4_cifar100_deep_screen",
        "purpose": (
            "one-seed 5K validation control screen on two published deep "
            "dense CIFAR-100 coordinates"
        ),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
