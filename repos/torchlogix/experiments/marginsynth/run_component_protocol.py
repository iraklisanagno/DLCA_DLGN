#!/usr/bin/env python3
"""Execute a frozen MarginSynth component protocol with resumable logging."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.marginsynth.verify_checkpoint import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Optional subset of component names declared by the protocol",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolved_repository_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") in {"completed", "passed"}
    except (json.JSONDecodeError, OSError):
        return False


def execute_stage(
    name: str,
    command: list[str],
    expected_summary: Path,
    log_dir: Path,
    execution: list[dict],
    resume: bool,
) -> None:
    if resume and completed(expected_summary):
        record = {
            "stage": name,
            "status": "reused-completed",
            "command": command,
            "summary": str(expected_summary),
            "summary_sha256": sha256_file(expected_summary),
        }
        execution.append(record)
        write_json(log_dir.parent / "execution.json", execution)
        return
    log_path = log_dir / f"{name}.console.log"
    started = time.perf_counter()
    with log_path.open("w") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record = {
        "stage": name,
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "command": command,
        "console_log": str(log_path),
        "console_log_sha256": sha256_file(log_path),
        "summary": str(expected_summary),
        "summary_sha256": (
            sha256_file(expected_summary) if expected_summary.exists() else None
        ),
    }
    execution.append(record)
    write_json(log_dir.parent / "execution.json", execution)
    if result.returncode:
        raise RuntimeError(f"stage {name} failed; see {log_path}")
    if not completed(expected_summary):
        raise RuntimeError(f"stage {name} did not create a completed summary")


def stage_config(
    base: dict,
    overlay: dict,
    output: str,
    method: str,
) -> dict:
    result = copy.deepcopy(base)
    result.update(copy.deepcopy(overlay))
    result["output"] = output
    result["method"] = method
    return result


def main() -> None:
    cli = parse_args()
    protocol_path = cli.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    two_pass_components = protocol.get("two_pass_components", {})
    available_components = []
    if "exact_baseline" in protocol:
        available_components.append("exact_baseline")
    if "unit_tying" in protocol:
        available_components.append("unit_tying")
    if "characterization" in protocol:
        available_components.append("characterization")
    available_components.extend(two_pass_components)
    if "silicon_control" in protocol:
        available_components.append("silicon_control")
    selected_components = (
        available_components if cli.components is None else cli.components
    )
    unknown = sorted(set(selected_components) - set(available_components))
    if unknown:
        raise ValueError(
            f"components not declared by protocol: {unknown}; "
            f"available: {available_components}"
        )
    source_run = resolved_repository_path(protocol["source_run"])
    checkpoint = source_run / protocol.get("source_checkpoint", "best_checkpoint.pt")
    required = [source_run / "training_config.json", checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    output_root_relative = protocol["output_root"]
    root = source_run / output_root_relative
    if root.exists() and not cli.resume:
        raise RuntimeError(f"refusing to overwrite protocol directory: {root}")
    root.mkdir(parents=True, exist_ok=cli.resume)
    log_dir = root / "orchestration_logs"
    config_dir = root / "input_configs"
    log_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    write_json(root / "protocol.json", protocol)
    provenance = {
        "format_version": 1,
        "protocol_source": str(protocol_path),
        "protocol_source_sha256": sha256_file(protocol_path),
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "python": sys.executable,
        "selected_components": selected_components,
        "test_used": False,
    }
    write_json(root / "provenance.json", provenance)
    execution = []
    execution_path = root / "execution.json"
    if cli.resume and execution_path.exists():
        execution = json.loads(execution_path.read_text())
    script_dir = Path(__file__).resolve().parent

    if "exact_baseline" in selected_components:
        exact_dir = root / "exact_baseline"
        exact_dir.mkdir(exist_ok=True)
        exact_checkpoint = exact_dir / "source_checkpoint.pt"
        if not exact_checkpoint.exists():
            shutil.copy2(checkpoint, exact_checkpoint)
        execute_stage(
            "exact_baseline",
            [
                sys.executable,
                str(script_dir / "export_tied_method.py"),
                str(source_run),
                str(exact_dir),
                "--checkpoint",
                exact_checkpoint.name,
                "--verification-split",
                protocol.get("verification_split", "calibration"),
                "--examples",
                str(protocol.get("verification_examples", 6000)),
            ],
            exact_dir / "export_summary.json",
            log_dir,
            execution,
            cli.resume,
        )

    if "unit_tying" in selected_components:
        unit_config = copy.deepcopy(protocol["unit_tying"])
        unit_config_path = config_dir / "unit_tying.json"
        write_json(unit_config_path, unit_config)
        ratio = float(unit_config["selected_ratio"])
        ratio_name = f"ratio_{int(round(100 * ratio)):02d}"
        ratio_dir = source_run / unit_config["output"] / ratio_name
        execute_stage(
            "unit_tying",
            [
                sys.executable,
                str(script_dir / "unit_tying.py"),
                str(source_run),
                "--config",
                str(unit_config_path),
                "--ratios",
                str(ratio),
            ],
            ratio_dir / "summary.json",
            log_dir,
            execution,
            cli.resume,
        )
        execute_stage(
            "unit_tying_export",
            [
                sys.executable,
                str(script_dir / "export_tied_method.py"),
                str(source_run),
                str(ratio_dir),
                "--checkpoint",
                "tied_checkpoint.pt",
                "--verification-split",
                protocol.get("verification_split", "calibration"),
                "--examples",
                str(protocol.get("verification_examples", 6000)),
            ],
            ratio_dir / "export_summary.json",
            log_dir,
            execution,
            cli.resume,
        )

    if "characterization" in selected_components:
        output = f"{output_root_relative}/characterization"
        config = copy.deepcopy(protocol["characterization"])
        config["output"] = output
        config_path = config_dir / "characterization.json"
        write_json(config_path, config)
        execute_stage(
            "characterization",
            [
                sys.executable,
                str(script_dir / "analyze_liveness_activity.py"),
                str(source_run),
                "--config",
                str(config_path),
            ],
            source_run / output / "summary.json",
            log_dir,
            execution,
            cli.resume,
        )

    for component, overlay in two_pass_components.items():
        if component not in selected_components:
            continue
        component_root = f"{output_root_relative}/{component}"
        first_output = f"{component_root}/first_resynthesis"
        first = stage_config(
            protocol["first_resynthesis"],
            overlay,
            first_output,
            f"trial28-{component}-first-resynthesis",
        )
        first_path = config_dir / f"{component}_first.json"
        write_json(first_path, first)
        execute_stage(
            f"{component}_first",
            [
                sys.executable,
                str(script_dir / "circuit_distillation.py"),
                str(source_run),
                "--config",
                str(first_path),
            ],
            source_run / first_output / "summary.json",
            log_dir,
            execution,
            cli.resume,
        )

        second_output = f"{component_root}/second_resynthesis"
        first_checkpoint = f"{first_output}/distilled_checkpoint.pt"
        second_overlay = overlay | {
            "source_checkpoint": first_checkpoint,
            "lock_reference_checkpoint": first_checkpoint,
        }
        second = stage_config(
            protocol["second_resynthesis"],
            second_overlay,
            second_output,
            f"trial28-{component}-second-resynthesis",
        )
        second_path = config_dir / f"{component}_second.json"
        write_json(second_path, second)
        execute_stage(
            f"{component}_second",
            [
                sys.executable,
                str(script_dir / "circuit_distillation.py"),
                str(source_run),
                "--config",
                str(second_path),
            ],
            source_run / second_output / "summary.json",
            log_dir,
            execution,
            cli.resume,
        )
        method_dir = source_run / second_output
        execute_stage(
            f"{component}_export",
            [
                sys.executable,
                str(script_dir / "export_tied_method.py"),
                str(source_run),
                str(method_dir),
                "--checkpoint",
                "distilled_checkpoint.pt",
                "--verification-split",
                protocol.get("verification_split", "calibration"),
                "--examples",
                str(protocol.get("verification_examples", 6000)),
            ],
            method_dir / "export_summary.json",
            log_dir,
            execution,
            cli.resume,
        )

    if "silicon_control" in selected_components:
        output = f"{output_root_relative}/silicon_control"
        config = copy.deepcopy(protocol["silicon_control"])
        config["output"] = output
        config_path = config_dir / "silicon_control.json"
        write_json(config_path, config)
        execute_stage(
            "silicon_control",
            [
                sys.executable,
                str(script_dir / "circuit_distillation.py"),
                str(source_run),
                "--config",
                str(config_path),
            ],
            source_run / output / "summary.json",
            log_dir,
            execution,
            cli.resume,
        )
        method_dir = source_run / output
        execute_stage(
            "silicon_control_export",
            [
                sys.executable,
                str(script_dir / "export_tied_method.py"),
                str(source_run),
                str(method_dir),
                "--checkpoint",
                "distilled_checkpoint.pt",
                "--verification-split",
                protocol.get("verification_split", "calibration"),
                "--examples",
                str(protocol.get("verification_examples", 6000)),
            ],
            method_dir / "export_summary.json",
            log_dir,
            execution,
            cli.resume,
        )

    summary = {
        "format_version": 1,
        "status": "completed",
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": sha256_file(protocol_path),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "selected_components": selected_components,
        "stages": execution,
        "test_used": False,
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
