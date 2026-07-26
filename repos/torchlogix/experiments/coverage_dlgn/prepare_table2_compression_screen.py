#!/usr/bin/env python3
"""Generate the locked CIFAR-10 compression-ladder screening queue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "table2_screen_cifar10_compression"
QUEUE_PATH = ROOT / "queues" / "table2_screen_cifar10_compression.json"
PROTOCOL_PATH = ROOT / "protocols" / "table2_dense_cifar10.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")

COMPRESSION_CELLS = {
    "128k": ("DlgnCifar10Budget128k", 128_000),
    "256k": ("DlgnCifar10Budget256k", 256_000),
    "384k": ("DlgnCifar10Budget384k", 384_000),
}

COVERAGE_CANDIDATES = (
    ("incumbent", 8, 0.25, 1.0, "raw"),
    ("pool4", 4, 0.25, 1.0, "raw"),
    ("pool16", 16, 0.25, 1.0, "raw"),
    ("swap0125", 8, 0.125, 1.0, "raw"),
    ("swap0500", 8, 0.5, 1.0, "raw"),
    ("novelty050", 8, 0.25, 0.5, "raw"),
    ("novelty200", 8, 0.25, 2.0, "raw"),
    ("light", 8, 0.25, 1.0, "light"),
    ("warp", 8, 0.25, 1.0, "warp"),
)


def common(name: str, architecture: str) -> dict:
    return {
        "dataset": "cifar-10",
        "architecture": architecture,
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


def entries() -> list[dict]:
    generated = []
    for cell, (architecture, gate_count) in COMPRESSION_CELLS.items():
        random_name = f"screen_table2_cifar10_{cell}_random_seed0"
        random_config = common(random_name, architecture)
        random_config.update({
            "connections_init_method": "random",
            "parametrization": "raw",
        })
        generated.append({
            "name": random_name,
            "family": "random",
            "cell": cell,
            "gate_count": gate_count,
            "candidate": "random",
            "config_payload": random_config,
        })
        for label, pool, swap, novelty, parametrization in COVERAGE_CANDIDATES:
            name = f"screen_table2_cifar10_{cell}_v3_{label}_seed0"
            config = common(name, architecture)
            config.update({
                "connections_init_method": "semantic_balanced_hybrid",
                "coverage_candidate_pool_size": pool,
                "coverage_swap_fraction": swap,
                "coverage_novelty_weight": novelty,
                "parametrization": parametrization,
            })
            generated.append({
                "name": name,
                "family": "coverage_v3",
                "cell": cell,
                "gate_count": gate_count,
                "candidate": label,
                "config_payload": config,
            })
    return generated


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []
    for entry in entries():
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
        "phase": "table2_screen_cifar10_compression",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": protocol["selection"]["metric"],
        "heldout_test_used": False,
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
