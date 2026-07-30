#!/usr/bin/env python3
"""Generate the fixed-budget CIFAR-100 depth pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    ROOT / "protocols" / "table4_dense_cifar100_depth384k.json"
)
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_depth384k_pilot"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_depth384k_pilot.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    constants = protocol["controlled_constants"]
    method = protocol["method"]
    pilot = protocol["pilot"]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    expected_paths = set()
    for label, architecture in protocol["architectures"].items():
        for family in ("random", "coverage_v3"):
            name = (
                f"pilot_table4_cifar100_384k_{label}_{family}_seed0"
            )
            config = {
                "dataset": protocol["dataset"],
                "architecture": architecture["class"],
                "device": "cuda",
                "seed": pilot["seed"],
                "data_split_seed": constants["data_split_seed"],
                "topology_seed": pilot["seed"],
                "batch_size": constants["batch_size"],
                "num_iterations": pilot["iterations"],
                "eval_freq": pilot["eval_frequency"],
                "valid_set_size": constants["validation_fraction"],
                "augmentation": constants["augmentation"],
                "learning_rate": constants["learning_rate"],
                "binarization_num_batches": 100,
                "binarization": "fixed",
                "binarization_init": "uniform",
                "binarization_per": "global",
                "connections": "fixed",
                "connections_gumbel": False,
                "forward_sampling": "soft",
                "weight_init": "random",
                "lut_rank": constants["lut_rank"],
                "parametrization": constants["parametrization"],
                "connections_init_method": (
                    "random"
                    if family == "random"
                    else "semantic_balanced_hybrid"
                ),
                "output": str(RESULT_ROOT / name),
            }
            if family == "coverage_v3":
                config.update({
                    "coverage_candidate_pool_size": method[
                        "coverage_candidate_pool_size"
                    ],
                    "coverage_swap_fraction": method[
                        "coverage_swap_fraction"
                    ],
                    "coverage_novelty_weight": method[
                        "coverage_novelty_weight"
                    ],
                })
            path = CONFIG_DIR / f"{name}.json"
            expected_paths.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            entries.append({
                "name": name,
                "architecture_label": label,
                "depth": architecture["depth"],
                "width_per_layer": architecture["width_per_layer"],
                "family": family,
                "seed": pilot["seed"],
                "config": str(path.relative_to(ROOT.parent.parent)),
                "output": config["output"],
            })
    stale = sorted(set(CONFIG_DIR.glob("*.json")) - expected_paths)
    if stale:
        raise RuntimeError(
            "Refusing to delete stale generated configs: "
            + ", ".join(str(path) for path in stale)
        )
    queue = {
        "phase": "table4_cifar100_depth384k_pilot",
        "purpose": "paired seed-0 20K fixed-384K depth pilot",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": pilot["metric"],
        "heldout_test_used": False,
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(queue, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
