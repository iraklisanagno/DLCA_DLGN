#!/usr/bin/env python3
"""Run the predeclared seed-0 circuit-distillation ablation matrix."""

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

from experiments.marginsynth.verify_checkpoint import sha256_file


ABLATIONS = {
    "mse_objective": {"objective": "mse"},
    "constants_only": {"action_space": "constants"},
    "gate_count_proxy": {"cost_kind": "gate-count"},
    "no_group_robustness": {"robust_groups": False},
    "no_exact_repair": {"repair": False},
}


def nested_update(config: dict, changes: dict) -> dict:
    result = copy.deepcopy(config)
    for key, value in changes.items():
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    base_path = cli.base_config.resolve()
    base = json.loads(base_path.read_text())
    matrix_dir = run_dir / "distillation" / "ablation_matrix_v3_seed0"
    if matrix_dir.exists():
        raise RuntimeError(f"refusing to overwrite ablation matrix: {matrix_dir}")
    config_dir = matrix_dir / "configs"
    config_dir.mkdir(parents=True)
    script = Path(__file__).with_name("circuit_distillation.py")
    records = []
    for name, overrides in ABLATIONS.items():
        config = nested_update(base, overrides)
        config["ablation"] = name
        config["output"] = f"distillation/ablation_matrix_v3_seed0/{name}"
        config_path = config_dir / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        log_path = matrix_dir / f"{name}.console.log"
        with log_path.open("w") as handle:
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(run_dir),
                    "--config",
                    str(config_path),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record = {
            "ablation": name,
            "overrides": overrides,
            "config": str(config_path.relative_to(run_dir)),
            "config_sha256": sha256_file(config_path),
            "console_log": str(log_path.relative_to(run_dir)),
            "console_log_sha256": sha256_file(log_path),
            "returncode": process.returncode,
        }
        result_dir = run_dir / config["output"]
        summary_path = result_dir / "summary.json"
        if process.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            changes = json.loads((result_dir / "learned_changes.json").read_text())
            retained = int(summary["retained_changes"])
            selected = changes[:retained]
            record.update(
                {
                    "status": "completed",
                    "learned_changes": summary["learned_changes"],
                    "retained_changes": retained,
                    "nonconstant_retained_changes": sum(
                        int(item["new_lut"] not in (0, 15)) for item in selected
                    ),
                    "proxy_aig_reduction": float(
                        sum(item["proxy_benefit"] for item in selected)
                    ),
                    "calibration_accuracy": summary["calibration"]["accuracy"],
                    "calibration_disagreement": summary["calibration"]["decision_flip_rate"],
                    "validation_accuracy": summary["validation"]["accuracy"],
                    "validation_disagreement": summary["validation"]["decision_flip_rate"],
                    "repair_holdout_feasible": summary["repair_holdout_feasible"],
                    "calibration_feasible": summary["calibration_feasible"],
                    "gpu_seconds": summary["timing"]["total_seconds"],
                    "summary": str(summary_path.relative_to(run_dir)),
                    "summary_sha256": sha256_file(summary_path),
                }
            )
        else:
            record["status"] = "failed"
        records.append(record)
        (matrix_dir / "ablation_summary.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(record, sort_keys=True), flush=True)
    manifest = {
        "format_version": 1,
        "status": "completed" if all(x["status"] == "completed" for x in records) else "failed",
        "base_config": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "test_used": False,
        "validation_used_for_selection": False,
        "ablations": records,
    }
    (matrix_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
