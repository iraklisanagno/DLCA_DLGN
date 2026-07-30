#!/usr/bin/env python3
"""Generate only the missing class-head arms of the paired 20K pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "table4_cifar100_class_head.json"
SOURCE_CONFIG_DIR = (
    ROOT / "configs" / "table4_cifar100_deep_selection"
)
CONFIG_DIR = ROOT / "configs" / "table4_cifar100_class_head"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_class_head.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    method = protocol["method"]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    expected_paths = set()
    for seed in protocol["pilot"]["paired_seeds"]:
        source_path = (
            SOURCE_CONFIG_DIR
            / f"select_table4_cifar100_64k_v3_swap0125_seed{seed}.json"
        )
        source = json.loads(source_path.read_text())
        if source["connections_init_method"] != "semantic_balanced_hybrid":
            raise RuntimeError(f"unexpected V3 source: {source_path}")
        if source["coverage_swap_fraction"] != 0.125:
            raise RuntimeError(f"unexpected V3 swap fraction: {source_path}")
        name = f"select_table4_cifar100_64k_v3_class_head_seed{seed}"
        config = dict(source)
        config.update({
            "classifier_connections_init_method": (
                method["classifier_strategy"]
            ),
            "class_balance_change_fraction": (
                method["class_balance_change_fraction"]
            ),
            "output": str(RESULT_ROOT / name),
        })
        path = CONFIG_DIR / f"{name}.json"
        expected_paths.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        entries.append({
            "name": name,
            "family": "coverage_v3_class_head",
            "seed": seed,
            "selection_source": str(
                source_path.relative_to(ROOT.parent.parent)
            ),
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
        "phase": "table4_cifar100_class_head",
        "purpose": (
            "missing class-head arms for the paired three-seed 20K pilot"
        ),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT.parent.parent)),
        "controls_reused": {
            "random": protocol["controls"]["random_pattern"],
            "coverage_v3": protocol["controls"]["v3_pattern"],
        },
        "selection_metric": protocol["pilot"]["selection_metric"],
        "heldout_test_used": False,
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
