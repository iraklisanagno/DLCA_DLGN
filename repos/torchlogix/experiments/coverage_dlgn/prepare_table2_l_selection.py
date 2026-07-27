#!/usr/bin/env python3
"""Generate paired three-seed 20K CIFAR-10 L raw/WARP selection runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCREEN_SUMMARY = ROOT / "summary" / "table2_l_screen.json"
SCREEN_CONFIG_DIR = ROOT / "configs" / "table2_l_screen"
CONFIG_DIR = ROOT / "configs" / "table2_l_selection"
QUEUE_PATH = ROOT / "queues" / "table2_l_selection.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def strip_coverage_controls(config: dict) -> None:
    for key in (
        "coverage_candidate_pool_size",
        "coverage_swap_fraction",
        "coverage_novelty_weight",
    ):
        config.pop(key, None)


def main() -> None:
    screen = json.loads(SCREEN_SUMMARY.read_text())
    if screen["test_set_used"]:
        raise RuntimeError("screen unexpectedly used held-out test")

    raw_random_source = screen["raw_random"]
    raw_v3_source = screen["selected_raw_v3"]
    warp_v3_source = screen["selected_warp_v3"]
    sources = {
        "raw_random": (
            raw_random_source,
            "random",
            "raw",
        ),
        "raw_v3": (
            raw_v3_source,
            "coverage_v3",
            "raw",
        ),
        "warp_random": (
            warp_v3_source,
            "random",
            "warp",
        ),
        "warp_v3": (
            warp_v3_source,
            "coverage_v3",
            "warp",
        ),
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    entries = []
    for family, (source, topology, parametrization) in sources.items():
        source_path = SCREEN_CONFIG_DIR / f"{source}.json"
        source_config = json.loads(source_path.read_text())
        for seed in (0, 1, 2):
            name = f"select_table2_cifar10_l_{family}_seed{seed}"
            config = dict(source_config)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "num_iterations": 20_000,
                "eval_freq": 2_000,
                "parametrization": parametrization,
                "output": str(RESULT_ROOT / name),
            })
            if topology == "random":
                config["connections_init_method"] = "random"
                strip_coverage_controls(config)
            else:
                config["connections_init_method"] = (
                    "semantic_balanced_hybrid"
                )
            path = CONFIG_DIR / f"{name}.json"
            expected_paths.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            entries.append({
                "name": name,
                "family": family,
                "topology": topology,
                "parametrization": parametrization,
                "seed": seed,
                "screen_source": source,
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
        "phase": "table2_l_selection",
        "screen_summary": str(SCREEN_SUMMARY.relative_to(ROOT.parent.parent)),
        "selection_metric": "mean best hardened validation accuracy",
        "heldout_test_used": False,
        "paired_seeds": [0, 1, 2],
        "primary_claim": ["raw_random", "raw_v3"],
        "secondary_combined_claim": ["warp_random", "warp_v3"],
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
