#!/usr/bin/env python3
"""Collect live, provenance-preserving status for DATE second-round queues."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUEUE_ROOT = ROOT / "queues"
SUMMARY_ROOT = ROOT / "summary"
REQUIRED_RUN_FILES = {
    "training_config.json",
    "environment.json",
    "metrics.csv",
    "run_summary.json",
}


def output_status(output: Path) -> str:
    if not output.exists():
        return "pending"
    present = {path.name for path in output.iterdir()}
    if REQUIRED_RUN_FILES.issubset(present):
        return "complete"
    return "incomplete"


def group_name(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def collect_queue(queue_path: Path) -> list[dict]:
    queue = json.loads(queue_path.read_text())
    rows = []
    for entry in queue["entries"]:
        config = json.loads(Path(entry["config"]).read_text())
        output = Path(entry["output"])
        status = output_status(output)
        row = {
            "phase": queue["phase"],
            "name": entry["name"],
            "group": group_name(entry["name"]),
            "status": status,
            "dataset": config["dataset"],
            "architecture": config["architecture"],
            "seed": config["seed"],
            "iterations": config["num_iterations"],
            "connections": config.get("connections_init_method"),
            "conv_connections": config.get("conv_connections_init_method"),
            "best_hard_validation_pct": None,
            "final_hard_validation_pct": None,
            "wall_minutes": None,
            "peak_gpu_gib": None,
            "dense_gate_count": None,
            "trainable_parameters": None,
            "training_routing_parameters": None,
            "deployed_routing_bits": None,
            "topology_construction_seconds": None,
            "source_revision": None,
            "source_tree_sha256": None,
            "training_implementation_sha256": None,
            "torch": None,
            "cuda_build": None,
            "gpu_names": None,
            "output": str(output),
        }
        if status == "complete":
            summary = json.loads((output / "run_summary.json").read_text())
            environment = json.loads((output / "environment.json").read_text())
            final_metrics = summary.get("final_metrics", {})
            cost = summary.get("cost") or {}
            topology = summary.get("topology") or []
            construction = [
                layer["construction_seconds"]
                for layer in topology
                if layer.get("construction_seconds") is not None
            ]
            row.update({
                "best_hard_validation_pct": (
                    100 * summary["best_validation_hard_accuracy"]
                ),
                "final_hard_validation_pct": (
                    100 * final_metrics["val_acc_discrete"]
                    if final_metrics.get("val_acc_discrete") is not None
                    else None
                ),
                "wall_minutes": summary.get("wall_seconds", 0) / 60,
                "peak_gpu_gib": (
                    summary["peak_gpu_memory_bytes"] / 2**30
                    if summary.get("peak_gpu_memory_bytes") is not None
                    else None
                ),
                "dense_gate_count": cost.get("dense_gate_count"),
                "trainable_parameters": cost.get("trainable_parameters"),
                "training_routing_parameters": cost.get(
                    "training_routing_parameters"
                ),
                "deployed_routing_bits": cost.get("deployed_routing_bits"),
                "topology_construction_seconds": (
                    sum(construction) if construction else None
                ),
                "source_revision": environment.get("source_revision"),
                "source_tree_sha256": environment.get("source_tree_sha256"),
                "training_implementation_sha256": environment.get(
                    "training_implementation_sha256"
                ),
                "torch": environment.get("torch"),
                "cuda_build": environment.get("cuda_build"),
                "gpu_names": "; ".join(environment.get("gpu_names", [])),
            })
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["phase"], row["group"]), []).append(row)
    output = []
    for (phase, name), group in sorted(groups.items()):
        complete = [row for row in group if row["status"] == "complete"]
        accuracies = [row["best_hard_validation_pct"] for row in complete]
        peak_values = [
            row["peak_gpu_gib"]
            for row in complete
            if row["peak_gpu_gib"] is not None
        ]
        output.append({
            "phase": phase,
            "group": name,
            "complete": len(complete),
            "expected": len(group),
            "incomplete": sum(row["status"] == "incomplete" for row in group),
            "pending": sum(row["status"] == "pending" for row in group),
            "best_hard_validation_mean_pct": (
                statistics.mean(accuracies) if accuracies else None
            ),
            "best_hard_validation_std_pct": (
                statistics.stdev(accuracies) if len(accuracies) > 1 else None
            ),
            "wall_minutes_mean": (
                statistics.mean(row["wall_minutes"] for row in complete)
                if complete
                else None
            ),
            "peak_gpu_gib_max": (
                max(peak_values) if peak_values else None
            ),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queues", nargs="*", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=SUMMARY_ROOT / "second_round_status",
    )
    args = parser.parse_args()
    queue_paths = args.queues or sorted(
        QUEUE_ROOT.glob("second_round_*.json")
    )
    rows = [row for path in queue_paths for row in collect_queue(path)]
    groups = aggregate(rows)
    payload = {
        "queues": [str(path) for path in queue_paths],
        "runs": rows,
        "groups": groups,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_csv(
        args.output_prefix.with_name(
            args.output_prefix.name + "_groups"
        ).with_suffix(".csv"),
        groups,
    )
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"second-round status: {complete}/{len(rows)} complete")


if __name__ == "__main__":
    main()
