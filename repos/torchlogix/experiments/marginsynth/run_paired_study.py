#!/usr/bin/env python3
"""Run or resume the frozen paired MarginSynth/Two-Stage study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def run_logged(command, log_path: Path, environment: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with {result.returncode}; see {log_path}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cost-model", required=True, type=Path)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "export", "search", "unit-tying", "synthesize"),
        default=("train", "export", "search", "unit-tying", "synthesize"),
    )
    parser.add_argument("--dataset-path", default="/tmp/torchlogix-datasets")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional subset of manifest seeds to run or resume.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed stages instead of preserving their artifacts.",
    )
    cli = parser.parse_args()
    manifest = json.loads(cli.manifest.read_text())
    environment = dict(os.environ)
    environment["DATASET_PATH"] = cli.dataset_path
    environment["CUDA_VISIBLE_DEVICES"] = cli.cuda_visible_devices
    python = sys.executable
    margin_dir = Path(__file__).resolve().parent
    records = []
    selected_seeds = set(cli.seeds or manifest["seeds"])
    known_seeds = {int(record["seed"]) for record in manifest["records"]}
    unknown_seeds = selected_seeds - known_seeds
    if unknown_seeds:
        raise ValueError(f"seeds are not present in manifest: {sorted(unknown_seeds)}")
    for seed_record in manifest["records"]:
        seed = int(seed_record["seed"])
        if seed not in selected_seeds:
            continue
        run_dir = Path(seed_record["run_dir"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        executed = []
        skipped = []
        if "train" in cli.stages and not (run_dir / "best_checkpoint.pt").exists():
            command = [
                python,
                str(margin_dir.parent / "train.py"),
                "--config",
                seed_record["training_config"],
            ]
            run_logged(command, run_dir / "paper_train.console.log", environment)
            executed.append("train")
        elif "train" in cli.stages:
            skipped.append("train")
        if "export" in cli.stages:
            export_outputs = (
                run_dir / "exact_simplified_circuit.json",
                run_dir / "synthesis_verification.json",
                run_dir / "calibration_trace" / "metadata.json",
            )
            if cli.force or not all(path.exists() for path in export_outputs):
                run_logged(
                    [
                        python,
                        str(margin_dir / "verify_checkpoint.py"),
                        str(run_dir),
                        "--examples",
                        "6000",
                        "--pack-bits",
                        "16",
                        "--compile-opt-level",
                        "0",
                    ],
                    run_dir / "paper_export.console.log",
                    environment,
                )
                run_logged(
                    [
                        python,
                        str(margin_dir / "verify_synthesis.py"),
                        str(run_dir),
                        "--examples",
                        "6000",
                        "--pack-bits",
                        "16",
                        "--compile-opt-level",
                        "0",
                    ],
                    run_dir / "paper_baseline_synthesis.console.log",
                    environment,
                )
                run_logged(
                    [python, str(margin_dir / "build_trace.py"), str(run_dir)],
                    run_dir / "paper_trace.console.log",
                    environment,
                )
                executed.append("export")
            else:
                skipped.append("export")
        frozen_cost = run_dir / "synth_cost_model.json"
        if not frozen_cost.exists():
            shutil.copy2(cli.cost_model, frozen_cost)
        if "search" in cli.stages:
            search_config = json.loads(
                Path(seed_record["search_config"]).read_text()
            )
            search_name = search_config["output"]
            search_dir = run_dir / search_name
            search_outputs = (
                search_dir / "search_summary.json",
                search_dir / "search_verification.json",
                search_dir / "validation_frontier.json",
            )
            if cli.force or not all(path.exists() for path in search_outputs):
                run_logged(
                    [
                        python,
                        str(margin_dir / "search_v2.py"),
                        str(run_dir),
                        "--config",
                        seed_record["search_config"],
                    ],
                    run_dir / "paper_search.console.log",
                    environment,
                )
                run_logged(
                    [
                        python,
                        str(margin_dir / "verify_search_v2.py"),
                        str(run_dir),
                        "--search",
                        search_name,
                    ],
                    run_dir / "paper_search_verify.console.log",
                    environment,
                )
                run_logged(
                    [
                        python,
                        str(margin_dir / "evaluate_frontier.py"),
                        str(run_dir),
                        "--search",
                        search_name,
                    ],
                    run_dir / "paper_search_validation.console.log",
                    environment,
                )
                executed.append("search")
            else:
                skipped.append("search")
        if "unit-tying" in cli.stages:
            unit_output = (
                run_dir / "baselines" / "two_stage_unit_tying" / "aggregate.json"
            )
            if cli.force or not unit_output.exists():
                run_logged(
                    [
                        python,
                        str(margin_dir / "unit_tying.py"),
                        str(run_dir),
                        "--config",
                        seed_record["unit_tying_config"],
                    ],
                    run_dir / "paper_unit_tying.console.log",
                    environment,
                )
                executed.append("unit-tying")
            else:
                skipped.append("unit-tying")
        if "synthesize" in cli.stages:
            search_config = json.loads(
                Path(seed_record["search_config"]).read_text()
            )
            synthesis_outputs = (
                run_dir
                / search_config["output"]
                / "frontier_synthesis.json",
                run_dir
                / "baselines"
                / "two_stage_unit_tying"
                / "synthesis_aggregate.json",
            )
            if cli.force or not all(path.exists() for path in synthesis_outputs):
                run_logged(
                    [
                        python,
                        str(margin_dir / "synthesize_frontier.py"),
                        str(run_dir),
                        "--search",
                        search_config["output"],
                    ],
                    run_dir / "paper_search_synthesis.console.log",
                    environment,
                )
                run_logged(
                    [
                        python,
                        str(margin_dir / "synthesize_unit_tying.py"),
                        str(run_dir),
                    ],
                    run_dir / "paper_unit_tying_synthesis.console.log",
                    environment,
                )
                executed.append("synthesize")
            else:
                skipped.append("synthesize")
        records.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "executed_stages": executed,
                "skipped_completed_stages": skipped,
                "best_checkpoint_sha256": (
                    sha256_file(run_dir / "best_checkpoint.pt")
                    if (run_dir / "best_checkpoint.pt").exists()
                    else None
                ),
                "test_used": False,
            }
        )
    result = {
        "format_version": 1,
        "paired_manifest": str(cli.manifest),
        "paired_manifest_sha256": sha256_file(cli.manifest),
        "frozen_cost_model": str(cli.cost_model),
        "frozen_cost_model_sha256": sha256_file(cli.cost_model),
        "records": records,
        "selected_seeds": sorted(selected_seeds),
        "test_used": False,
    }
    output = cli.manifest.parent / "paired_execution.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
