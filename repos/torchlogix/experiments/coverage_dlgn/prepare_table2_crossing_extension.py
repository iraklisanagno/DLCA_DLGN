#!/usr/bin/env python3
"""Extend the selected compression crossing from three to five paired seeds."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table2_cifar10_compression_final3.json"
BASE_QUEUE = ROOT / "queues" / "table2_final_cifar10_compression.json"
CONFIG_DIR = ROOT / "configs" / "table2_final_cifar10_compression"
QUEUE_PATH = ROOT / "queues" / "table2_crossing_extension_cifar10.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    base_queue = json.loads(BASE_QUEUE.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("full-effort validation unexpectedly used test data")
    crossing = summary["selected_crossing_for_five_seeds"]
    if crossing is None:
        raise RuntimeError("no compression cell met the frozen crossing rule")

    queue = []
    for family in ("random", "coverage_v3"):
        source_entry = next(
            entry for entry in base_queue["entries"]
            if (
                entry["cell"] == crossing
                and entry["family"] == family
                and entry["name"].endswith("_seed0")
            )
        )
        source_path = ROOT.parent.parent / source_entry["config"]
        source_config = json.loads(source_path.read_text())
        for seed in (3, 4):
            name = source_entry["name"].removesuffix("seed0") + f"seed{seed}"
            config = dict(source_config)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "output": str(RESULT_ROOT / name),
            })
            path = CONFIG_DIR / f"{name}.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            queue.append({
                "name": name,
                "family": family,
                "cell": crossing,
                "candidate": source_entry["candidate"],
                "extension_of": source_entry["name"],
                "config": str(path.relative_to(ROOT.parent.parent)),
                "output": config["output"],
            })

    payload = {
        "phase": "table2_crossing_extension_cifar10",
        "validation_summary": str(SUMMARY_PATH.relative_to(ROOT.parent.parent)),
        "selected_crossing": crossing,
        "heldout_test_used": False,
        "extension_seeds": [3, 4],
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs for crossing {crossing}")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
