#!/usr/bin/env python3
"""Generate full-effort paired CIFAR-10 compression runs after selection.

All three compression cells receive three paired 108K-step seeds. The
noninferiority crossing is determined only from this full-effort validation
stage; its selected random/V3 pair is then extended to five seeds separately.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary" / "table2_cifar10_compression_selection.json"
SELECTION_QUEUE = ROOT / "queues" / "table2_select_cifar10_compression.json"
SELECTION_CONFIG_DIR = ROOT / "configs" / "table2_select_cifar10_compression"
CONFIG_DIR = ROOT / "configs" / "table2_final_cifar10_compression"
QUEUE_PATH = ROOT / "queues" / "table2_final_cifar10_compression.json"
RESULT_ROOT = Path("experiments/coverage_dlgn/results")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    selection_queue = json.loads(SELECTION_QUEUE.read_text())
    if summary["test_set_used"]:
        raise RuntimeError("selection unexpectedly used the held-out test set")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    queue = []

    for cell in ("128k", "256k", "384k"):
        winner = summary["selected_for_full_training"][cell]
        selected_candidates = (("random", "random"), ("coverage_v3", winner))
        for family, candidate in selected_candidates:
            source_entry = next(
                entry for entry in selection_queue["entries"]
                if (
                    entry["cell"] == cell
                    and entry["family"] == family
                    and entry["candidate"] == candidate
                    and entry["name"].endswith("_seed0")
                )
            )
            source_path = ROOT.parent.parent / source_entry["config"]
            source_config = json.loads(source_path.read_text())
            for seed in (0, 1, 2):
                name = f"final_table2_cifar10_{cell}_{candidate}_seed{seed}"
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
                path.write_text(
                    json.dumps(config, indent=2, sort_keys=True) + "\n"
                )
                queue.append({
                    "name": name,
                    "family": family,
                    "cell": cell,
                    "candidate": candidate,
                    "selection_source": source_entry["name"],
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
        "phase": "table2_final_cifar10_compression",
        "selection_summary": str(SUMMARY_PATH.relative_to(ROOT.parent.parent)),
        "selection_metric": "frozen best hardened validation checkpoint",
        "heldout_test_used": False,
        "iterations": 108_000,
        "paired_seeds": [0, 1, 2],
        "entries": queue,
    }
    QUEUE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(queue)} configs")
    print(QUEUE_PATH)


if __name__ == "__main__":
    main()
