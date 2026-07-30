#!/usr/bin/env python3
"""Generate paper-schedule 6x64K CIFAR-100 random/V3 runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / "summary" / "table4_cifar100_deep_selection.json"
SELECTION_CONFIG_DIR = (
    ROOT / "configs" / "table4_cifar100_deep_selection"
)
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_deep_final"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_deep_final.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    result = selection["selections"]["64k"]
    if selection["test_set_used"]:
        raise RuntimeError("selection unexpectedly used held-out test")
    if not result["promote_to_paper_schedule"]:
        raise RuntimeError("64K confirmation did not authorize full schedule")
    candidate = result["selected_v3_candidate"]
    sources = {
        "random": (
            SELECTION_CONFIG_DIR
            / "select_table4_cifar100_64k_random_seed0.json"
        ),
        "coverage_v3": (
            SELECTION_CONFIG_DIR
            / f"select_table4_cifar100_64k_v3_{candidate}_seed0.json"
        ),
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    entries = []
    for seed in (0, 1, 2):
        for family, source_path in sources.items():
            source = json.loads(source_path.read_text())
            label = "random" if family == "random" else f"v3_{candidate}"
            name = f"final_table4_cifar100_64k_{label}_seed{seed}"
            config = dict(source)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "valid_set_size": 0.2,
                "num_iterations": 40_000,
                "eval_freq": 2_000,
                "output": str(RESULT_ROOT / name),
            })
            path = CONFIG_DIR / f"{name}.json"
            expected_paths.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            entries.append({
                "name": name,
                "architecture_label": "64k",
                "family": family,
                "candidate": "random" if family == "random" else candidate,
                "seed": seed,
                "selection_source": source_path.stem,
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
        "phase": "table4_cifar100_deep_final",
        "purpose": (
            "three paired seeds at the scalability paper's 100-epoch "
            "6x64K schedule"
        ),
        "selection_summary": str(
            SELECTION.relative_to(ROOT.parent.parent)
        ),
        "validation_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "paired_seeds": [0, 1, 2],
        "paper_schedule": {
            "train_examples": 40_000,
            "batch_size": 100,
            "steps_per_epoch": 400,
            "epochs": 100,
            "num_iterations": 40_000,
            "validation_fraction": 0.2,
            "augmentation": "none",
            "optimizer": "Adam",
            "learning_rate": 0.01,
        },
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
