#!/usr/bin/env python3
"""Run the frozen short-recovery objective/locking ablation matrix."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--base-config", type=Path, required=True)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    base = json.loads(cli.base_config.read_text())
    base["steps"] = 3000
    base["snapshot_steps"] = [0, 250, 500, 1000, 2000, 3000]
    matrix = {
        "label_only": {
            "loss_weights": {
                "labels": 1.0,
                "decision": 0.0,
                "class_worst": 0.0,
                "fold_worst": 0.0,
                "hardware_ceiling": 0.0,
                "entropy": 0.0,
            }
        },
        "label_margin_no_hardware_ceiling": {
            "loss_weights": base["loss_weights"] | {"hardware_ceiling": 0.0}
        },
        "soft_forward": {"forward_sampling": "soft"},
        "unlocked_first_pass": {"lock_source_changes": False},
    }
    config_dir = run_dir / "recovery" / "ablation_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("recovery_finetune.py")
    manifest = []
    for name, overrides in matrix.items():
        config = copy.deepcopy(base)
        config.update(overrides)
        config["method"] = f"locked-rewrite-short-recovery-ablation-{name}"
        config["output"] = f"recovery/ablations/{name}_seed{config['seed']}"
        config_path = config_dir / f"{name}_seed{config['seed']}.json"
        if config_path.exists():
            raise RuntimeError(f"refusing to overwrite {config_path}")
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        command = [
            sys.executable,
            str(script),
            str(run_dir),
            "--config",
            str(config_path),
        ]
        result = subprocess.run(command, check=False)
        manifest.append(
            {
                "name": name,
                "config": str(config_path),
                "output": config["output"],
                "returncode": result.returncode,
            }
        )
        if result.returncode:
            break
    manifest_path = config_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not manifest or manifest[-1]["returncode"]:
        raise RuntimeError(f"recovery ablation failed; see {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
