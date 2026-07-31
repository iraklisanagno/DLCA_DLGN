#!/usr/bin/env python3
"""Export and synthesize a tied-method checkpoint in an isolated child run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def run(command: list[str], log_path: Path) -> None:
    with log_path.open("w") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("method_dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="tied_checkpoint.pt",
        help="Method checkpoint filename (supports independent distillation runs)",
    )
    parser.add_argument("--prepare-residual-trace", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a failed child export without deleting its diagnostic logs.",
    )
    cli = parser.parse_args()
    source_run = cli.source_run.resolve()
    method_dir = cli.method_dir.resolve()
    export_dir = method_dir / "export_run"
    if export_dir.exists() and not cli.resume:
        raise RuntimeError(
            f"refusing to overwrite existing export run: {export_dir}"
        )
    export_dir.mkdir(parents=True, exist_ok=cli.resume)
    checkpoint = method_dir / cli.checkpoint
    required = [source_run / "training_config.json", checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    shutil.copy2(source_run / "training_config.json", export_dir / "training_config.json")
    tied_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_payload = torch.load(
        source_run / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    source_metadata = source_payload.get("metadata", {})
    tied_metadata = tied_payload.setdefault("metadata", {})
    for key in ("step", "metrics", "configuration", "topology"):
        if key in source_metadata and key not in tied_metadata:
            tied_metadata[key] = source_metadata[key]
    torch.save(tied_payload, export_dir / "best_checkpoint.pt")
    python = sys.executable
    script_dir = Path(__file__).resolve().parent
    log_tag = ".resume" if cli.resume else ""
    run(
        [
            python,
            str(script_dir / "verify_checkpoint.py"),
            str(export_dir),
            "--examples", "6000",
            "--pack-bits", "16",
            "--compile-opt-level", "0",
        ],
        export_dir / f"export{log_tag}.console.log",
    )
    run(
        [
            python,
            str(script_dir / "verify_synthesis.py"),
            str(export_dir),
            "--examples", "6000",
            "--pack-bits", "16",
            "--compile-opt-level", "0",
        ],
        export_dir / f"synthesis{log_tag}.console.log",
    )
    if cli.prepare_residual_trace:
        run(
            [python, str(script_dir / "build_trace.py"), str(export_dir)],
            export_dir / f"trace{log_tag}.console.log",
        )
    synthesis = json.loads((export_dir / "synthesis_verification.json").read_text())
    payload = {
        "format_version": 1,
        "status": "completed",
        "source_run": str(source_run),
        "method_dir": str(method_dir),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint": cli.checkpoint,
        "export_dir": str(export_dir),
        "exact_circuit_sha256": sha256_file(
            export_dir / "exact_simplified_circuit.json"
        ),
        "live_gates": synthesis["hardware_circuit"]["logic_gates"],
        "abc_and_nodes": synthesis["abc"]["stats"]["and_nodes"],
        "abc_levels": synthesis["abc"]["stats"]["levels"],
        "residual_trace_prepared": cli.prepare_residual_trace,
        "test_used": False,
    }
    (method_dir / "export_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
