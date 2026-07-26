#!/usr/bin/env python3
"""Generate the paired three-seed, 20K CIFAR-10 compression selection queue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table2_cifar10_compression_screen.json"
SCREEN_CONFIG_DIR = ROOT / "configs" / "table2_screen_cifar10_compression"
CONFIG_DIR = ROOT / "configs" / "table2_select_cifar10_compression"
QUEUE_PATH = ROOT / "queues" / "table2_select_cifar10_compression.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("screen summary unexpectedly used the held-out test set")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []

    for cell in ("128k", "256k", "384k"):
        selected = summary["selected_for_20k"][cell]
        for method in ("random", "coverage_v3"):
            for screen_name in selected[method]:
                screen_path = SCREEN_CONFIG_DIR / f"{screen_name}.json"
                screen_config = json.loads(screen_path.read_text())
                candidate = screen_name.removeprefix(
                    f"screen_table2_cifar10_{cell}_"
                ).removesuffix("_seed0")
                for seed in (0, 1, 2):
                    name = f"select_table2_cifar10_{cell}_{candidate}_seed{seed}"
                    config = dict(screen_config)
                    config.update({
                        "seed": seed,
                        "topology_seed": seed,
                        "num_iterations": 20_000,
                        "eval_freq": 2_000,
                        "output": str(RESULT_ROOT / name),
                    })
                    path = CONFIG_DIR / f"{name}.json"
                    expected_paths.add(path)
                    path.write_text(
                        json.dumps(config, indent=2, sort_keys=True) + "\n"
                    )
                    queue.append({
                        "name": name,
                        "family": method,
                        "cell": cell,
                        "candidate": candidate,
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
        "phase": "table2_select_cifar10_compression",
        "screen_summary": str(SUMMARY_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "mean best hardened validation accuracy",
        "heldout_test_used": False,
        "paired_seeds": [0, 1, 2],
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
