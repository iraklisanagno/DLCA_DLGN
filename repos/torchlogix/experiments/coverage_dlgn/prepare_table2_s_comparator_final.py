#!/usr/bin/env python3
"""Generate three-seed full-effort CIFAR-10 S comparator runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table2_s_comparator_screen.json"
SCREEN_CONFIG_DIR = ROOT / "configs" / "table2_s_comparator_screen"
CONFIG_DIR = ROOT / "configs" / "table2_s_comparator_final"
QUEUE_PATH = ROOT / "queues" / "table2_s_comparator_final.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("screen unexpectedly used held-out test data")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []
    for family in ("mommen", "lilogic"):
        screen_name = summary["selected_for_full_training"][family]
        source = json.loads(
            (SCREEN_CONFIG_DIR / f"{screen_name}.json").read_text()
        )
        candidate = screen_name.removeprefix("screen_").removesuffix("_seed0")
        for seed in (0, 1, 2):
            name = f"final_{candidate}_seed{seed}"
            config = dict(source)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "num_iterations": 108_000 if family == "mommen" else 42_200,
                "eval_freq": 2_000 if family == "mommen" else 844,
                "output": str(RESULT_ROOT / name),
            })
            path = CONFIG_DIR / f"{name}.json"
            expected_paths.add(path)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            queue.append({
                "name": name,
                "family": family,
                "screen_source": screen_name,
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
        "phase": "table2_s_comparator_final",
        "screen_summary": str(SUMMARY_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "frozen best hardened validation checkpoint",
        "heldout_test_used": False,
        "seeds": [0, 1, 2],
        "matched_training_examples": {
            "mommen": 10_800_000,
            "lilogic": 10_803_200,
            "relative_difference": 0.0002962963,
        },
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
