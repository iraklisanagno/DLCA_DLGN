#!/usr/bin/env python3
"""Generate only the task-aware arms of the paired CIFAR-10 M pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols" / "cifar10_medium_task_aware.json"
CONFIG_DIR = ROOT / "configs" / "cifar10_medium_task_aware"
QUEUE = ROOT / "queues" / "cifar10_medium_task_aware.json"
SOURCE_DIR = ROOT / "configs"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    method = protocol["method"]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    expected = set()
    source_path = (
        SOURCE_DIR / "pilot_cifar10_medium_semantic_balanced_v3_seed0.json"
    )
    source = json.loads(source_path.read_text())
    if source["connections_init_method"] != "semantic_balanced_hybrid":
        raise RuntimeError(f"not a frozen V3 source: {source_path}")
    for seed in protocol["pilot"]["paired_seeds"]:
        name = f"pilot_cifar10_medium_v3_task_aware_seed{seed}"
        config = dict(source)
        config.update({
            "seed": seed,
            "topology_seed": seed,
            "task_aware_rewire_step": method["rewire_step"],
            "task_aware_rewire_fraction": method["rewire_fraction"],
            "task_aware_rewire_candidate_pool_size": (
                method["candidate_pool_size"]
            ),
            "task_aware_rewire_diversity_weight": (
                method["diversity_weight"]
            ),
            "output": str(RESULT_ROOT / name),
        })
        path = CONFIG_DIR / f"{name}.json"
        expected.add(path)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        entries.append({
            "name": name,
            "family": "coverage_v3_task_aware",
            "seed": seed,
            "source_v3_config": str(
                source_path.relative_to(ROOT.parent.parent)
            ),
            "config": str(path.relative_to(ROOT.parent.parent)),
            "output": config["output"],
        })
    stale = sorted(set(CONFIG_DIR.glob("*.json")) - expected)
    if stale:
        raise RuntimeError(
            "Refusing to delete stale configs: "
            + ", ".join(str(path) for path in stale)
        )
    payload = {
        "phase": protocol["phase"],
        "protocol": str(PROTOCOL.relative_to(ROOT.parent.parent)),
        "controls_reused_not_rerun": True,
        "heldout_test_used": False,
        "entries": entries,
    }
    QUEUE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE)


if __name__ == "__main__":
    main()
