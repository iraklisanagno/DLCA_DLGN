#!/usr/bin/env python3
"""Add semantic topology diagnostics to existing frozen checkpoints."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import get_model
from torchlogix.topology import analyze_model_topology, write_topology_report


def analyze_run(run_dir: Path):
    config = json.loads((run_dir / "training_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"
    payload = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
    state_dict = payload["model_state_dict"]
    model = get_model(state_dict["0.thresholds"], args)
    model.load_state_dict(state_dict, strict=True)
    rows = analyze_model_topology(model)
    write_topology_report(
        rows,
        run_dir,
        stem="semantic_topology",
        metadata={
            "checkpoint": "best_checkpoint.pt",
            "strategy": config["connections_init_method"],
            "topology_seed": config.get("topology_seed"),
        },
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        rows = analyze_run(run_dir)
        first = rows[0]
        final = rows[-1]
        print(
            f"{run_dir.name}: same-source={first['same_source_pair_fraction']:.4f}, "
            f"spatial-p50={first['spatial_manhattan_p50']:.1f}, "
            f"final-source-ancestry={final['mean_source_ancestry']:.2f}, "
            f"fanout-cv={final['fanout_cv']:.3f}"
        )


if __name__ == "__main__":
    main()
