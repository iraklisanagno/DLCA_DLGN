#!/usr/bin/env python3
"""Run identical export/Yosys/ABC measurement for distillation ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    cli = parser.parse_args()
    run_dir = cli.run_dir.resolve()
    matrix_dir = run_dir / "distillation" / "ablation_matrix_v3_seed0"
    matrix = json.loads((matrix_dir / "ablation_summary.json").read_text())
    exporter = Path(__file__).with_name("export_tied_method.py")
    results = []
    for record in matrix:
        name = record["ablation"]
        method_dir = matrix_dir / name
        export_summary = method_dir / "export_summary.json"
        log_path = matrix_dir / f"{name}.synthesis.console.log"
        if export_summary.exists():
            returncode = 0
            reused = True
        else:
            with log_path.open("w") as handle:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(exporter),
                        str(run_dir),
                        str(method_dir),
                        "--checkpoint",
                        "distilled_checkpoint.pt",
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            returncode = process.returncode
            reused = False
        output = {
            "ablation": name,
            "returncode": returncode,
            "reused_existing_export": reused,
        }
        if returncode == 0 and export_summary.exists():
            synthesis = json.loads(export_summary.read_text())
            output.update(
                {
                    "status": "completed",
                    "abc_and_nodes": synthesis["abc_and_nodes"],
                    "abc_levels": synthesis["abc_levels"],
                    "live_gates": synthesis["live_gates"],
                    "export_summary": str(export_summary.relative_to(run_dir)),
                    "export_summary_sha256": sha256_file(export_summary),
                }
            )
        else:
            output["status"] = "failed"
        if log_path.exists():
            output["console_log_sha256"] = sha256_file(log_path)
        results.append(output)
        (matrix_dir / "synthesis_summary.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(output, sort_keys=True), flush=True)
    if any(item["status"] != "completed" for item in results):
        raise RuntimeError("one or more ablation synthesis runs failed")


if __name__ == "__main__":
    main()
