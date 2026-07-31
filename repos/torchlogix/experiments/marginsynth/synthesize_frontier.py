#!/usr/bin/env python3
"""Synthesize selected MarginSynth frontier snapshots with one frozen flow."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import (
    git_revision,
    sha256_file,
    write_artifact_manifest,
)
from experiments.marginsynth.verify_synthesis import (
    command_version,
    parse_abc_stats,
    parse_yosys_cells,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--search", required=True)
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    search_dir = run_dir / cli.search
    summary = json.loads((search_dir / "search_summary.json").read_text())
    yosys = shutil.which("yosys")
    abc = shutil.which("berkeley-abc") or shutil.which("abc")
    if yosys is None or abc is None:
        raise RuntimeError("both Yosys and Berkeley ABC are required")

    records = []
    seen_steps = set()
    for pareto_point in summary["pareto"]:
        step = pareto_point["selected_step"]
        if step in seen_steps:
            continue
        seen_steps.add(step)
        snapshot_dir = search_dir / pareto_point["snapshot"]
        verilog_path = snapshot_dir / "circuit.v"
        synthesis_dir = snapshot_dir / "synthesis"
        synthesis_dir.mkdir(exist_ok=True)
        blif_path = synthesis_dir / "circuit.blif"

        yosys_script = (
            f"read_verilog -sv {verilog_path}; "
            "synth -top circuit -noabc; check; "
            f"write_blif {blif_path}"
        )
        start = time.perf_counter()
        yosys_result = subprocess.run(
            [yosys, "-p", yosys_script],
            capture_output=True,
            text=True,
            check=False,
        )
        yosys_seconds = time.perf_counter() - start
        yosys_log = yosys_result.stdout + yosys_result.stderr
        yosys_log_path = synthesis_dir / "yosys.log"
        yosys_log_path.write_text(yosys_log)

        abc_script = (
            f"read_blif {blif_path}; strash; balance; rewrite; "
            "refactor; rewrite; print_stats"
        )
        start = time.perf_counter()
        abc_result = subprocess.run(
            [abc, "-q", abc_script],
            capture_output=True,
            text=True,
            check=False,
        )
        abc_seconds = time.perf_counter() - start
        abc_log = abc_result.stdout + abc_result.stderr
        abc_log_path = synthesis_dir / "abc.log"
        abc_log_path.write_text(abc_log)

        if yosys_result.returncode or abc_result.returncode:
            raise RuntimeError(
                f"synthesis failed for Pareto step {step}: "
                f"Yosys={yosys_result.returncode}, ABC={abc_result.returncode}"
            )
        records.append(
            {
                "step": step,
                "budgets_selecting_step": [
                    point.get("budget", point.get("accuracy_budget"))
                    for point in summary["pareto"]
                    if point["selected_step"] == step
                ],
                "snapshot": pareto_point["snapshot"],
                "global_loss": pareto_point.get(
                    "global_loss",
                    pareto_point.get("accuracy_loss"),
                ),
                "accuracy_loss": pareto_point.get("accuracy_loss"),
                "disagreement": pareto_point.get("disagreement"),
                "maximum_per_class_loss": pareto_point.get(
                    "maximum_per_class_loss",
                    pareto_point.get("maximum_per_class_accuracy_loss"),
                ),
                "maximum_per_class_disagreement": pareto_point.get(
                    "maximum_per_class_disagreement"
                ),
                "live_gates": pareto_point["live_gates"],
                "proxy_cost": pareto_point.get(
                    "proxy_cost",
                    pareto_point.get("estimated_abc_nodes"),
                ),
                "verilog_sha256": sha256_file(verilog_path),
                "blif_sha256": sha256_file(blif_path),
                "yosys": {
                    "returncode": yosys_result.returncode,
                    "seconds": yosys_seconds,
                    "cells": parse_yosys_cells(yosys_log),
                    "log": str(yosys_log_path.relative_to(search_dir)),
                },
                "abc": {
                    "returncode": abc_result.returncode,
                    "seconds": abc_seconds,
                    "stats": parse_abc_stats(abc_log),
                    "log": str(abc_log_path.relative_to(search_dir)),
                },
            }
        )

    result = {
        "format_version": 1,
        "status": "passed",
        "development_run": True,
        "search": cli.search,
        "source_revision": git_revision(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "search_summary_sha256": sha256_file(
            search_dir / "search_summary.json"
        ),
        "flow": {
            "yosys_version": command_version([yosys, "-V"]),
            "abc_version": command_version([abc, "-q", "version"]),
            "yosys_commands": (
                "read_verilog -sv; synth -top circuit -noabc; check; write_blif"
            ),
            "abc_commands": (
                "read_blif; strash; balance; rewrite; refactor; rewrite; "
                "print_stats"
            ),
        },
        "points": records,
        "test_used": False,
        "validation_used": False,
    }
    output_path = search_dir / "frontier_synthesis.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
