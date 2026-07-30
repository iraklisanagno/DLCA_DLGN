#!/usr/bin/env python3
"""Audit the reused-control class-head pilot and apply its frozen gate."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import torch

from torchlogix.topology import (
    classwise_ancestry_metrics,
    image_input_semantics,
    propagate_packed_ancestry,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "table4_cifar100_class_head.json"
QUEUE_PATH = ROOT / "queues" / "table4_cifar100_class_head.json"
QUEUE_SUMMARY = (
    ROOT / "logs" / "table4_cifar100_class_head" / "queue_summary.json"
)
JSON_PATH = ROOT / "summary" / "table4_cifar100_class_head.json"
CSV_PATH = ROOT / "summary" / "table4_cifar100_class_head.csv"
T_CRITICAL_DF2 = 4.302652729696142


def load_row(
    *,
    family: str,
    seed: int,
    run_dir: Path,
) -> dict[str, object]:
    run = json.loads((run_dir / "run_summary.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    if (run_dir / "test_metrics.json").exists():
        raise RuntimeError(f"pilot control touched held-out test: {run_dir}")
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
    ancestry = image_input_semantics(
        3, 32, 32, 3, layout="channel_interleaved"
    ).source_ancestry()
    for layer_indices in indices[:-1]:
        ancestry = propagate_packed_ancestry(ancestry, layer_indices)
    class_metrics = classwise_ancestry_metrics(
        ancestry,
        indices[-1],
        n_sources=3 * 32 * 32,
        output_groups=100,
    )
    return {
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_validation_hard_accuracy": (
            run["best_validation_hard_accuracy"]
        ),
        "wall_seconds": run["wall_seconds"],
        "topology_construction_seconds": sum(
            layer["construction_seconds"] for layer in run["topology"]
        ),
        "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
        "dense_gate_count": run["cost"]["dense_gate_count"],
        "trainable_parameters": run["cost"]["trainable_parameters"],
        "deployed_routing_bits": run["cost"]["deployed_routing_bits"],
        "connections_init_method": config["connections_init_method"],
        "classifier_connections_init_method": config.get(
            "classifier_connections_init_method"
        ),
        "class_coverage_min": class_metrics["class_coverage_min"],
        "class_source_usage_cv_mean": class_metrics[
            "class_source_usage_cv_mean"
        ],
        "within_class_jaccard_mean": class_metrics[
            "within_class_jaccard_mean"
        ],
        "between_class_jaccard_mean": class_metrics[
            "between_class_jaccard_mean"
        ],
        "training_implementation_sha256": environment[
            "training_implementation_sha256"
        ],
    }


def paired_stats(
    rows: list[dict[str, object]],
    left: str,
    right: str,
) -> dict[str, object]:
    by_key = {(row["family"], row["seed"]): row for row in rows}
    gains = [
        by_key[(right, seed)]["best_validation_hard_accuracy"]
        - by_key[(left, seed)]["best_validation_hard_accuracy"]
        for seed in (0, 1, 2)
    ]
    mean = statistics.mean(gains)
    std = statistics.stdev(gains)
    half_width = T_CRITICAL_DF2 * std / math.sqrt(3)
    return {
        "left": left,
        "right": right,
        "per_seed_gain_pp": [100 * gain for gain in gains],
        "paired_mean_gain_pp": 100 * mean,
        "paired_95ci_pp": [
            100 * (mean - half_width),
            100 * (mean + half_width),
        ],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    queue = json.loads(QUEUE_PATH.read_text())
    audit = json.loads(QUEUE_SUMMARY.read_text())
    if audit["failed"]:
        raise RuntimeError(f"class-head queue failed: {audit['failed']}")
    expected = {entry["name"] for entry in queue["entries"]}
    observed = set(audit["skipped"])
    observed.update(row["name"] for row in audit["finished"])
    if observed != expected:
        raise RuntimeError(
            f"queue mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rows = []
    for seed in protocol["pilot"]["paired_seeds"]:
        rows.append(load_row(
            family="random",
            seed=seed,
            run_dir=ROOT / "results"
            / f"select_table4_cifar100_64k_random_seed{seed}",
        ))
        rows.append(load_row(
            family="coverage_v3",
            seed=seed,
            run_dir=ROOT / "results"
            / f"select_table4_cifar100_64k_v3_swap0125_seed{seed}",
        ))
        rows.append(load_row(
            family="coverage_v3_class_head",
            seed=seed,
            run_dir=ROOT / "results"
            / f"select_table4_cifar100_64k_v3_class_head_seed{seed}",
        ))

    for row in rows:
        if row["dense_gate_count"] != 384_000:
            raise RuntimeError(f"gate-budget mismatch: {row}")
        if row["trainable_parameters"] != 6_144_000:
            raise RuntimeError(f"parameter-budget mismatch: {row}")
    costs = {
        (
            row["dense_gate_count"],
            row["trainable_parameters"],
            row["deployed_routing_bits"],
        )
        for row in rows
    }
    if len(costs) != 1:
        raise RuntimeError(f"deployment cost mismatch: {costs}")

    family_stats = {}
    for family in (
        "random",
        "coverage_v3",
        "coverage_v3_class_head",
    ):
        values = [
            row["best_validation_hard_accuracy"]
            for row in rows if row["family"] == family
        ]
        family_stats[family] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "n": len(values),
            "class_coverage_min_mean": statistics.mean(
                row["class_coverage_min"]
                for row in rows if row["family"] == family
            ),
            "class_source_usage_cv_mean": statistics.mean(
                row["class_source_usage_cv_mean"]
                for row in rows if row["family"] == family
            ),
            "within_class_jaccard_mean": statistics.mean(
                row["within_class_jaccard_mean"]
                for row in rows if row["family"] == family
            ),
            "between_class_jaccard_mean": statistics.mean(
                row["between_class_jaccard_mean"]
                for row in rows if row["family"] == family
            ),
        }
    versus_random = paired_stats(
        rows, "random", "coverage_v3_class_head"
    )
    versus_v3 = paired_stats(
        rows, "coverage_v3", "coverage_v3_class_head"
    )
    promotion = protocol["promotion"]
    promoted = (
        versus_random["paired_mean_gain_pp"]
        >= promotion["required_mean_gain_over_random_pp"]
        and versus_v3["paired_mean_gain_pp"]
        >= promotion["required_mean_gain_over_v3_pp"]
    )
    payload = {
        "phase": queue["phase"],
        "provenance": "TRIED",
        "validation_metric": queue["selection_metric"],
        "test_set_used": False,
        "controls_reused_not_rerun": True,
        "training_implementation_sha256_by_family": {
            family: sorted({
                row["training_implementation_sha256"]
                for row in rows if row["family"] == family
            })
            for family in family_stats
        },
        "family_stats": family_stats,
        "class_head_vs_random": versus_random,
        "class_head_vs_v3": versus_v3,
        "promotion_thresholds_pp": {
            "over_random": promotion[
                "required_mean_gain_over_random_pp"
            ],
            "over_v3": promotion["required_mean_gain_over_v3_pp"],
        },
        "promote": promoted,
        "rows": sorted(rows, key=lambda row: (row["family"], row["seed"])),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(JSON_PATH)
    for family, stats in family_stats.items():
        print(
            f"{family}: {100 * stats['mean']:.3f}% "
            f"+/- {100 * stats['sample_std']:.3f}%"
        )
    print(
        "head-random="
        f"{versus_random['paired_mean_gain_pp']:+.3f} pp; "
        "head-v3="
        f"{versus_v3['paired_mean_gain_pp']:+.3f} pp; "
        f"promote={promoted}"
    )


if __name__ == "__main__":
    main()
