#!/usr/bin/env python3
"""Prepare reproducible five-seed training and paired-method configurations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-base", required=True, type=Path)
    parser.add_argument("--search-base", required=True, type=Path)
    parser.add_argument("--unit-tying-base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--reuse-seed0-run",
        type=Path,
        help="Use an existing fully compatible seed-0 run directory",
    )
    parser.add_argument(
        "--reuse-seed0-search-output",
        help="Reuse an existing search directory inside the seed-0 run.",
    )
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    training_base = json.loads(cli.training_base.read_text())
    search_base = json.loads(cli.search_base.read_text())
    tying_base = json.loads(cli.unit_tying_base.read_text())
    records = []
    for seed in cli.seeds:
        run_id = f"paper_fashion_mnist_raw_seed{seed}"
        run_dir = (
            cli.reuse_seed0_run
            if seed == 0 and cli.reuse_seed0_run is not None
            else Path("experiments/marginsynth/results") / run_id
        )
        training = copy.deepcopy(training_base)
        training["seed"] = seed
        training["topology_seed"] = seed
        training["output"] = str(run_dir)
        search = copy.deepcopy(search_base)
        search["search_seed"] = seed
        search["output"] = (
            cli.reuse_seed0_search_output
            if seed == 0 and cli.reuse_seed0_search_output
            else f"search_v2_frozen_seed{seed}"
        )
        tying = copy.deepcopy(tying_base)
        tying["calibration_seed"] = seed
        training_path = cli.output / f"training_seed{seed}.json"
        search_path = cli.output / f"search_seed{seed}.json"
        tying_path = cli.output / f"unit_tying_seed{seed}.json"
        for path, payload in (
            (training_path, training),
            (search_path, search),
            (tying_path, tying),
        ):
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        records.append(
            {
                "seed": seed,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "training_config": str(training_path),
                "training_config_sha256": sha256_file(training_path),
                "search_config": str(search_path),
                "search_config_sha256": sha256_file(search_path),
                "unit_tying_config": str(tying_path),
                "unit_tying_config_sha256": sha256_file(tying_path),
            }
        )
    manifest = {
        "format_version": 1,
        "seeds": cli.seeds,
        "paired_starting_checkpoint_required": True,
        "test_policy": "sealed until a separate frozen-protocol manifest exists",
        "records": records,
    }
    manifest_path = cli.output / "paired_study_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
