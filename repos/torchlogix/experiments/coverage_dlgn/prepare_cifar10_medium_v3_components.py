#!/usr/bin/env python3
"""Generate only the missing arms of the frozen V3 component ablation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols" / "cifar10_medium_v3_components.json"
SOURCE = ROOT / "configs" / "pilot_cifar10_medium_random_v3_seed0.json"
CONFIG_DIR = ROOT / "configs" / "cifar10_medium_v3_components"
QUEUE = ROOT / "queues" / "cifar10_medium_v3_components.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    source = json.loads(SOURCE.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    expected = set()
    missing_arms = {
        arm["family"]: arm for arm in protocol["arms"]
        if "reused_pattern" not in arm
    }
    for family, arm in missing_arms.items():
        for seed in protocol["training"]["paired_seeds"]:
            name = f"ablate_cifar10_medium_{family}_seed{seed}"
            config = dict(source)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "connections_init_method": arm["connections_init_method"],
                "coverage_candidate_pool_size": 8,
                "coverage_swap_fraction": arm.get(
                    "coverage_swap_fraction", 0.25
                ),
                "coverage_novelty_weight": 1.0,
                "output": str(RESULT_ROOT / name),
            })
            path = CONFIG_DIR / f"{name}.json"
            expected.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            entries.append({
                "name": name,
                "family": family,
                "seed": seed,
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
        "purpose": "train only missing V3 component-ablation arms",
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
