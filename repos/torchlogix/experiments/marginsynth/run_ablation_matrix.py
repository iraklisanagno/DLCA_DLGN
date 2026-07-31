#!/usr/bin/env python3
"""Materialize and optionally execute the frozen MarginSynth v2 ablations."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import (
    sha256_file,
    write_artifact_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of variant names",
    )
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    matrix = json.loads(cli.matrix.read_text())
    base_path = cli.matrix.parent / matrix["base_config"]
    base = json.loads(base_path.read_text())
    selected = set(cli.variants or matrix["variants"].keys())
    unknown = selected - set(matrix["variants"])
    if unknown:
        raise ValueError(f"unknown ablation variants: {sorted(unknown)}")
    config_dir = run_dir / "ablation_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, overrides in matrix["variants"].items():
        if name not in selected:
            continue
        config = copy.deepcopy(base)
        config.update(overrides)
        config["max_accepted_rewrites"] = int(
            matrix["development_max_accepted_rewrites"]
        )
        config["output"] = f"ablations/v2_{name}_seed0"
        config_path = config_dir / f"v2_{name}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        command = [
            sys.executable,
            str(Path(__file__).with_name("search_v2.py")),
            str(run_dir),
            "--config",
            str(config_path),
        ]
        record = {
            "variant": name,
            "config": str(config_path.relative_to(run_dir)),
            "config_sha256": sha256_file(config_path),
            "command": command,
            "status": "materialized",
        }
        if cli.run:
            log_path = config_dir / f"v2_{name}.console.log"
            with log_path.open("w") as handle:
                result = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            record["returncode"] = result.returncode
            record["log"] = str(log_path.relative_to(run_dir))
            record["log_sha256"] = sha256_file(log_path)
            record["status"] = "completed" if result.returncode == 0 else "failed"
            records.append(record)
            if result.returncode:
                break
        else:
            records.append(record)
    result = {
        "format_version": 1,
        "matrix": str(cli.matrix),
        "matrix_sha256": sha256_file(cli.matrix),
        "base_config_sha256": sha256_file(base_path),
        "executed": cli.run,
        "records": records,
        "test_used": False,
    }
    output = config_dir / "ablation_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if any(record["status"] == "failed" for record in records):
        raise RuntimeError("an ablation failed")


if __name__ == "__main__":
    main()
