#!/usr/bin/env python3
"""Generate the locked three-seed, 20K MNIST selection queue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table1_mnist_screen.json"
CONFIG_DIR = ROOT / "configs" / "table1_select_mnist"
QUEUE_PATH = ROOT / "queues" / "table1_select_mnist.json"
SCREEN_CONFIG_DIR = ROOT / "configs" / "table1_screen_mnist"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []
    selected = summary["selected_for_20k"]
    family_order = [
        "random",
        "coverage_v3",
        "mommen",
        "lilogic",
        "bitlogic",
    ]
    for method in family_order:
        for screen_name in selected[method]:
            screen_config_path = SCREEN_CONFIG_DIR / f"{screen_name}.json"
            screen_config = json.loads(screen_config_path.read_text())
            candidate = screen_name.removeprefix("screen_").removesuffix(
                "_seed0"
            )
            for seed in (0, 1, 2):
                name = f"select_{candidate}_seed{seed}"
                config = dict(screen_config)
                config.update({
                    "seed": seed,
                    "topology_seed": seed,
                    "num_iterations": 20000,
                    "eval_freq": 1000,
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
        "phase": "table1_select_mnist",
        "screen_summary": str(SUMMARY_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "mean best hardened validation accuracy",
        "test_set_used": False,
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
