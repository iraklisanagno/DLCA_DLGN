#!/usr/bin/env python3
"""Prepare only seeds 3/4 for the unified CIFAR-10 S promotion test."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols" / "cifar10_conv_small_unified_five_seed.json"
CONFIG_DIR = ROOT / "configs" / "cifar10_conv_small_unified_five_seed"
QUEUE = ROOT / "queues" / "cifar10_conv_small_unified_five_seed.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")
SOURCE_PATTERNS = {
    "random": ROOT / "configs" / (
        "pilot_conv_cifar10_paper_small_random_seed0.json"
    ),
    "frozen_v4": ROOT / "configs" / (
        "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed0.json"
    ),
    "unified_candidate": ROOT / "configs" / (
        "cifar10_conv_small_v4_components"
        "/ablate_conv_cifar10_small_balanced_channel_no_swaps_seed0.json"
    ),
}
NAME_PATTERNS = {
    "random": "pilot_conv_cifar10_paper_small_random_seed{seed}",
    "frozen_v4": (
        "pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{seed}"
    ),
    "unified_candidate": (
        "pilot_conv_cifar10_paper_small_semantic_degree_balanced_seed{seed}"
    ),
}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    expected = set()

    # Interleaving by seed assigns every method for seed 3 to GPU 0 and every
    # method for seed 4 to GPU 1 when run_gpu_queue uses GPUs [0, 1].
    for family in ("random", "frozen_v4", "unified_candidate"):
        source = json.loads(SOURCE_PATTERNS[family].read_text())
        arm = protocol["arms"][family]
        for seed in protocol["training"]["new_seeds"]:
            name = NAME_PATTERNS[family].format(seed=seed)
            config = dict(source)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "connections_init_method": arm["connections_init_method"],
                "output": str(RESULT_ROOT / name),
            })
            if family == "unified_candidate":
                # These fields are recorded explicitly even though U1 ignores
                # swap/candidate controls by construction.
                config["coverage_swap_fraction"] = 0.0
                config["coverage_candidate_pool_size"] = 8
                config["coverage_novelty_weight"] = 1.0
            path = CONFIG_DIR / f"{name}.json"
            expected.add(path)
            path.write_text(json.dumps(config, indent=2) + "\n")
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
        "purpose": protocol["purpose"],
        "protocol": str(PROTOCOL.relative_to(ROOT.parent.parent)),
        "historical_seeds_reused_not_rerun": [0, 1, 2],
        "heldout_test_used": False,
        "entries": entries,
    }
    QUEUE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE)


if __name__ == "__main__":
    main()
