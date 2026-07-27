#!/usr/bin/env python3
"""Generate five-seed full CIFAR-10 L raw random/V3 final runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SELECTION_SUMMARY = ROOT / "summary" / "table2_l_selection.json"
SELECTION_CONFIG_DIR = ROOT / "configs" / "table2_l_selection"
CONFIG_DIR = ROOT / "configs" / "table2_l_final"
QUEUE_PATH = ROOT / "queues" / "table2_l_final.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    selection = json.loads(SELECTION_SUMMARY.read_text())
    if selection["test_set_used"]:
        raise RuntimeError("selection unexpectedly used held-out test")
    if selection["frozen_primary_finalists"] != ["raw_random", "raw_v3"]:
        raise RuntimeError("raw primary finalists are not frozen")

    sources = {
        "random": (
            SELECTION_CONFIG_DIR
            / "select_table2_cifar10_l_raw_random_seed0.json"
        ),
        "coverage_v3": (
            SELECTION_CONFIG_DIR
            / "select_table2_cifar10_l_raw_v3_seed0.json"
        ),
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    entries = []
    for family, source_path in sources.items():
        source_config = json.loads(source_path.read_text())
        for seed in range(5):
            label = "random" if family == "random" else "v3_swap0500"
            name = f"final_table2_cifar10_l_{label}_seed{seed}"
            config = dict(source_config)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "num_iterations": 108_000,
                "eval_freq": 2_000,
                "output": str(RESULT_ROOT / name),
            })
            path = CONFIG_DIR / f"{name}.json"
            expected_paths.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            entries.append({
                "name": name,
                "family": family,
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
        "phase": "table2_l_final",
        "selection_summary": str(
            SELECTION_SUMMARY.relative_to(ROOT.parent.parent)
        ),
        "validation_metric": "best hardened validation accuracy",
        "heldout_test_used": False,
        "paired_seeds": list(range(5)),
        "entries": entries,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
