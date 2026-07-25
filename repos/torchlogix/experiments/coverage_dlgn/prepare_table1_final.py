#!/usr/bin/env python3
"""Generate five-seed, 200-effective-epoch Table 1 final queues."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path("experiments/coverage_dlgn/results")
TRAIN_EXAMPLES = 54_000
FINAL_EPOCHS = 200
FINAL_SEEDS = range(5)
PREFERRED_EVAL_FREQ = 2_000
FALLBACK_EVAL_EPOCHS = 4


def final_eval_freq(num_iterations: int, steps_per_epoch: int) -> int:
    """Choose a valid final-evaluation interval near the frozen cadence."""
    if num_iterations % PREFERRED_EVAL_FREQ == 0:
        return PREFERRED_EVAL_FREQ
    eval_freq = steps_per_epoch * FALLBACK_EVAL_EPOCHS
    if num_iterations % eval_freq != 0:
        raise ValueError(
            f"{num_iterations=} is not divisible by {eval_freq=}"
        )
    return eval_freq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=["mnist", "fashion"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell = args.cell
    summary_path = ROOT / "summary" / f"table1_{cell}_selection.json"
    selection_dir = ROOT / "configs" / f"table1_select_{cell}"
    final_dir = ROOT / "configs" / f"table1_final_{cell}"
    queue_path = ROOT / "queues" / f"table1_final_{cell}.json"

    summary = json.loads(summary_path.read_text())
    selected = summary["selected_for_final"]
    final_dir.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    entries = []
    family_order = [
        "random",
        "coverage_v3",
        "mommen",
        "lilogic",
        "bitlogic",
    ]
    for family in family_order:
        candidate = selected[family]
        source_path = selection_dir / f"select_{candidate}_seed0.json"
        source = json.loads(source_path.read_text())
        batch_size = source["batch_size"]
        steps_per_epoch = math.ceil(TRAIN_EXAMPLES / batch_size)
        num_iterations = steps_per_epoch * FINAL_EPOCHS
        eval_freq = final_eval_freq(num_iterations, steps_per_epoch)
        for seed in FINAL_SEEDS:
            name = f"final_{candidate}_seed{seed}"
            config = dict(source)
            config.update({
                "seed": seed,
                "topology_seed": seed,
                "num_iterations": num_iterations,
                "eval_freq": eval_freq,
                "output": str(RESULT_ROOT / name),
            })
            path = final_dir / f"{name}.json"
            expected_paths.add(path)
            path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
            entries.append({
                "name": name,
                "family": family,
                "selected_candidate": candidate,
                "selection_summary": str(
                    summary_path.relative_to(ROOT.parent.parent)
                ),
                "batch_size": batch_size,
                "train_examples": TRAIN_EXAMPLES,
                "steps_per_epoch": steps_per_epoch,
                "epochs": FINAL_EPOCHS,
                "eval_freq": eval_freq,
                "num_evaluations": num_iterations // eval_freq,
                "config": str(path.relative_to(ROOT.parent.parent)),
                "output": config["output"],
            })

    stale = sorted(set(final_dir.glob("*.json")) - expected_paths)
    if stale:
        raise RuntimeError(
            "Refusing to delete stale generated configs: "
            + ", ".join(str(path) for path in stale)
        )
    payload = {
        "phase": f"table1_final_{cell}",
        "selection_summary": str(
            summary_path.relative_to(ROOT.parent.parent)
        ),
        "training_effort": {
            "policy": "equal effective epochs",
            "train_examples": TRAIN_EXAMPLES,
            "epochs": FINAL_EPOCHS,
            "iterations_depend_on_batch_size": True,
        },
        "validation_metric": "best hardened validation accuracy",
        "test_set_used": False,
        "entries": entries,
    }
    queue_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} configs")
    print(queue_path)


if __name__ == "__main__":
    main()
