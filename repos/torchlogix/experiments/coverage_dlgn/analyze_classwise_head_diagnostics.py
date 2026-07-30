#!/usr/bin/env python3
"""Compare class-wise ancestry in frozen CIFAR-10 and CIFAR-100 runs."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchlogix.topology import (
    classwise_ancestry_metrics,
    image_input_semantics,
    propagate_packed_ancestry,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "summary" / "classwise_head_diagnostics.json"
RUNS = {
    "cifar10_s_random": (
        ROOT / "results" / "paper_cifar10_small_random_seed0"
    ),
    "cifar10_s_v3": (
        ROOT / "results"
        / "paper_cifar10_small_semantic_balanced_v3_seed0"
    ),
    "cifar10_l_random": (
        ROOT / "results" / "final_table2_cifar10_l_random_seed0"
    ),
    "cifar10_l_v3": (
        ROOT / "results" / "final_table2_cifar10_l_v3_swap0500_seed0"
    ),
    "cifar100_64k_random": (
        ROOT / "results" / "final_table4_cifar100_64k_random_seed0"
    ),
    "cifar100_64k_v3": (
        ROOT / "results"
        / "final_table4_cifar100_64k_v3_swap0125_seed0"
    ),
}


def analyze(run_dir: Path) -> dict[str, object]:
    config = json.loads((run_dir / "training_config.json").read_text())
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint["model_state_dict"]
    index_keys = sorted(
        (key for key in state if key.endswith("connections.indices")),
        key=lambda key: int(key.split(".", 1)[0]),
    )
    indices = [state[key].detach().cpu().numpy() for key in index_keys]
    if not indices:
        raise RuntimeError(f"no dense topology in {run_dir}")
    threshold_count = indices[0].max(initial=0)
    del threshold_count  # dimensions below are read from the saved topology
    in_dim = int(indices[0].max()) + 1
    # Fixed dense topologies cover the complete encoded input, so the first
    # layer's declared in_dim is recovered from its topology report.
    topology = json.loads((run_dir / "topology.json").read_text())
    in_dim = int(topology["layers"][0]["in_dim"])
    threshold_bits = in_dim // (3 * 32 * 32)
    if threshold_bits * 3 * 32 * 32 != in_dim:
        raise RuntimeError(f"unexpected CIFAR input dimension {in_dim}")
    semantics = image_input_semantics(
        3,
        32,
        32,
        threshold_bits,
        layout="channel_interleaved",
    )
    ancestry = semantics.source_ancestry()
    for layer_indices in indices[:-1]:
        ancestry = propagate_packed_ancestry(ancestry, layer_indices)
    class_count = 100 if config["dataset"] == "cifar-100" else 10
    metrics = classwise_ancestry_metrics(
        ancestry,
        indices[-1],
        n_sources=semantics.n_sources,
        output_groups=class_count,
    )
    return {
        "run_dir": str(run_dir),
        "dataset": config["dataset"],
        "architecture": config["architecture"],
        "strategy": config["connections_init_method"],
        "topology_seed": config["topology_seed"],
        "checkpoint_step": checkpoint["metadata"]["step"],
        "metrics": metrics,
    }


def main() -> None:
    rows = {label: analyze(path) for label, path in RUNS.items()}
    payload = {
        "status": "OFFLINE-DIAGNOSTIC",
        "heldout_test_evaluated": False,
        "purpose": (
            "Measure class-wise ancestry balance in already frozen "
            "topologies; checkpoint weights and labels are not used."
        ),
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)
    for label, row in rows.items():
        metrics = row["metrics"]
        print(
            f"{label}: coverage_min={metrics['class_coverage_min']:.6f} "
            f"usage_cv={metrics['class_source_usage_cv_mean']:.6f} "
            f"within_jaccard={metrics['within_class_jaccard_mean']:.6f} "
            f"between_jaccard={metrics['between_class_jaccard_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
