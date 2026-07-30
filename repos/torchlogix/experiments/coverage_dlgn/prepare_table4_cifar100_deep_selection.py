#!/usr/bin/env python3
"""Generate paired three-seed confirmations for deep CIFAR-100 screens."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "table4_dense_cifar100_deep.json"
SCREEN_PATH = ROOT / "summary" / "table4_cifar100_deep_screen.json"
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_deep_selection"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_selection.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def common(
    name: str, seed: int, architecture: dict
) -> dict:
    return {
        "dataset": "cifar-100",
        "architecture": architecture["class"],
        "device": "cuda",
        "seed": seed,
        "data_split_seed": 2027,
        "topology_seed": seed,
        "batch_size": architecture["paper_training"]["batch_size"],
        "num_iterations": 20_000,
        "eval_freq": 2_000,
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
    screen = json.loads(SCREEN_PATH.read_text())
    candidates = {
        row["label"]: row
        for row in protocol["method"]["controls_screened_per_architecture"]
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    skipped_architectures = []
    for label, architecture in protocol["architectures"].items():
        selection = screen["selections"][label]
        if not selection["advance_to_confirmation"]:
            skipped_architectures.append(label)
            continue
        selected = candidates[selection["selected_v3_candidate"]]
        for seed in (0, 1, 2):
            random_name = (
                f"select_table4_cifar100_{label}_random_seed{seed}"
            )
            random_config = common(random_name, seed, architecture)
            random_config["connections_init_method"] = "random"
            generated.append({
                "name": random_name,
                "architecture_label": label,
                "family": "random",
                "candidate": "random",
                "seed": seed,
                "config_payload": random_config,
            })

            v3_name = (
                f"select_table4_cifar100_{label}_v3_"
                f"{selected['label']}_seed{seed}"
            )
            v3_config = common(v3_name, seed, architecture)
            v3_config.update({
                "connections_init_method": "semantic_balanced_hybrid",
                "coverage_candidate_pool_size": selected[
                    "coverage_candidate_pool_size"
                ],
                "coverage_swap_fraction": selected[
                    "coverage_swap_fraction"
                ],
                "coverage_novelty_weight": selected[
                    "coverage_novelty_weight"
                ],
            })
            generated.append({
                "name": v3_name,
                "architecture_label": label,
                "family": "coverage_v3",
                "candidate": selected["label"],
                "seed": seed,
                "config_payload": v3_config,
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
        "phase": "table4_cifar100_deep_selection",
        "purpose": "paired three-seed 20K validation confirmation",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT.parent.parent)),
        "screen_source": str(SCREEN_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "mean best hardened validation accuracy",
        "heldout_test_used": False,
        "skipped_architectures": skipped_architectures,
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(f"skipped={skipped_architectures}")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
