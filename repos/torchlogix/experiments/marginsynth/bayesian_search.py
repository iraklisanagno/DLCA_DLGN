#!/usr/bin/env python3
"""Run the four prespecified constrained multi-objective MarginSynth studies.

Each trial is an isolated, resumable subprocess chain.  The optimizer sees two
objectives (calibration-guard accuracy loss and exact ABC AND nodes) plus the
prespecified feasibility residuals.  Validation and test data are never used
for trial selection.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

TORCHLOGIX_ROOT = Path(__file__).resolve().parents[2]
if str(TORCHLOGIX_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLOGIX_ROOT))

from experiments.marginsynth.bayesian_protocol import (
    METHOD_AGGRESSIVE,
    METHOD_GUARDED,
    STUDY_CASES,
    active_budgets,
    aggressive_recovery_configs,
    canonical_json,
    constraint_names,
    constraint_values,
    flatten_record,
    guarded_two_pass_configs,
    is_feasible,
    load_json,
    object_sha256,
    pareto_records,
    suggest_parameters,
    validate_protocol,
)


class TrialStageError(RuntimeError):
    """A logged subprocess stage failed without invalidating the study."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_output(command: list[str], cwd: Path) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "returncode": None,
            "seconds": time.perf_counter() - started,
            "error": repr(error),
        }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(output_root: Path, repository_root: Path, protocol: dict) -> None:
    environment_dir = output_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    probes = {
        "git_revision": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short", "--branch"],
        "git_diff_stat": ["git", "diff", "--stat"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze", "--all"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "yosys_version": ["yosys", "-V"],
        "abc_version": ["berkeley-abc", "-q", "version"],
    }
    results = {}
    for name, command in probes.items():
        result = command_output(command, repository_root)
        results[name] = result
        text = result.get("stdout", "") + result.get("stderr", "")
        (environment_dir / f"{name}.txt").write_text(text)
    selected_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "DATASET_PATH",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
        if os.environ.get(name) is not None
    }
    manifest = {
        "format_version": 1,
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "optuna",
                "pandas",
                "scikit-learn",
            )
        },
        "selected_environment": selected_environment,
        "protocol_sha256": object_sha256(protocol),
        "probes": {
            name: {
                "command": value["command"],
                "returncode": value["returncode"],
                "seconds": value["seconds"],
                "artifact": f"{name}.txt",
                "artifact_sha256": sha256_file(environment_dir / f"{name}.txt"),
                **({"error": value["error"]} if "error" in value else {}),
            }
            for name, value in results.items()
        },
    }
    write_json(environment_dir / "manifest.json", manifest)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def numeric_leaves(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from numeric_leaves(value[key], child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_leaves(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def export_study_tables(study_dir: Path) -> None:
    records = []
    for path in sorted((study_dir / "trials").glob("trial_*/trial_record.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    write_json(study_dir / "trials.json", records)
    write_csv(study_dir / "trials.csv", [flatten_record(record) for record in records])

    parameter_rows = []
    constraint_rows = []
    stage_rows = []
    metric_rows = []
    for record in records:
        number = record["trial_number"]
        for name, value in sorted(record.get("parameters", {}).items()):
            parameter_rows.append(
                {"trial_number": number, "parameter": name, "value": value}
            )
        names = record.get("constraint_names", [])
        values = record.get("constraint_values", [])
        for name, value in zip(names, values):
            constraint_rows.append(
                {
                    "trial_number": number,
                    "constraint": name,
                    "residual": value,
                    "satisfied": value <= 0.0,
                }
            )
        commands_path = study_dir / "trials" / f"trial_{number:05d}" / "commands.jsonl"
        if commands_path.exists():
            for line in commands_path.read_text().splitlines():
                if line.strip():
                    stage_rows.append(json.loads(line))
        for path, value in numeric_leaves(record.get("metrics", {})):
            metric_rows.append(
                {"trial_number": number, "metric": path, "value": value}
            )
    write_csv(study_dir / "parameters_long.csv", parameter_rows)
    write_csv(study_dir / "constraints_long.csv", constraint_rows)
    write_csv(study_dir / "stages.csv", stage_rows)
    write_csv(study_dir / "metrics_long.csv", metric_rows)

    pareto = pareto_records(records)
    write_json(study_dir / "pareto.json", pareto)
    write_csv(study_dir / "pareto.csv", [flatten_record(record) for record in pareto])
    statuses = {}
    for record in records:
        statuses[record.get("status", "unknown")] = (
            statuses.get(record.get("status", "unknown"), 0) + 1
        )
    summary = {
        "format_version": 1,
        "updated_at_utc": utc_now(),
        "trial_count": len(records),
        "status_counts": statuses,
        "feasible_exact_trial_count": sum(
            record.get("feasible") is True
            and record.get("objective_fidelity") == "exact_abc"
            for record in records
        ),
        "pareto_trial_numbers": [record["trial_number"] for record in pareto],
        "validation_used": False,
        "test_used": False,
    }
    write_json(study_dir / "study_summary.json", summary)


class StudyExecutor:
    def __init__(
        self,
        run_dir: Path,
        repository_root: Path,
        protocol_path: Path,
        protocol: dict,
        output_root: Path,
        case_name: str,
        smoke: bool,
        synthesize_infeasible: bool,
    ) -> None:
        self.run_dir = run_dir
        self.repository_root = repository_root
        self.protocol_path = protocol_path
        self.protocol = protocol
        self.output_root = output_root
        self.case_name = case_name
        self.method, self.policy = STUDY_CASES[case_name]
        self.smoke = smoke
        self.synthesize_infeasible = synthesize_infeasible
        self.study_dir = output_root / case_name
        self.trials_dir = self.study_dir / "trials"
        self.events_path = self.study_dir / "events.jsonl"
        self.script_dir = Path(__file__).resolve().parent
        self.python = sys.executable
        self.templates = {
            name: load_json((protocol_path.parent / relative).resolve())
            for name, relative in protocol["templates"].items()
        }
        self.budgets = active_budgets(protocol, self.policy)
        self.maximum_recovery_steps = int(
            protocol["execution"]["maximum_primary_recovery_steps"]
        )
        self.cost_model = run_dir / protocol["execution"]["cost_model"]
        self.auto_synthesize_feasible = bool(
            protocol["execution"]["auto_synthesize_feasible"]
        )
        self.stage_timeout = float(protocol["execution"]["stage_timeout_seconds"])

    def event(self, event: str, **values) -> None:
        append_jsonl(
            self.events_path,
            {"timestamp_utc": utc_now(), "event": event, **values},
        )

    def run_stage(
        self,
        trial_number: int,
        trial_dir: Path,
        stage: str,
        command: list[str],
    ) -> dict:
        log_path = trial_dir / "logs" / f"{stage}.console.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_utc = utc_now()
        started = time.perf_counter()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.event("stage_started", trial_number=trial_number, stage=stage)
        timed_out = False
        with log_path.open("w") as handle:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.repository_root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self.stage_timeout,
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired as error:
                timed_out = True
                returncode = None
                handle.write(f"\nTIMEOUT: {error!r}\n")
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = {
            "trial_number": trial_number,
            "stage": stage,
            "command": canonical_json(command),
            "working_directory": str(self.repository_root),
            "started_at_utc": started_utc,
            "finished_at_utc": utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "child_user_seconds": after.ru_utime - before.ru_utime,
            "child_system_seconds": after.ru_stime - before.ru_stime,
            "child_max_rss_kib": after.ru_maxrss,
            "returncode": returncode,
            "timed_out": timed_out,
            "console_log": str(log_path.relative_to(self.study_dir)),
            "console_log_sha256": sha256_file(log_path),
        }
        append_jsonl(trial_dir / "commands.jsonl", record)
        self.event(
            "stage_finished",
            trial_number=trial_number,
            stage=stage,
            returncode=returncode,
            wall_seconds=record["wall_seconds"],
        )
        if returncode != 0:
            raise TrialStageError(
                f"trial {trial_number} stage {stage} failed; see {log_path}"
            )
        return record

    def write_trial_record(self, trial_dir: Path, record: dict) -> None:
        write_json(trial_dir / "trial_record.json", record)
        export_study_tables(self.study_dir)

    def synthesize(
        self,
        number: int,
        trial_dir: Path,
        method_dir: Path,
        checkpoint: str,
    ) -> dict:
        self.run_stage(
            number,
            trial_dir,
            "exact_export_yosys_abc",
            [
                self.python,
                str(self.script_dir / "export_tied_method.py"),
                str(self.run_dir),
                str(method_dir),
                "--checkpoint",
                checkpoint,
                "--verification-split",
                "calibration",
            ],
        )
        return load_json(method_dir / "export_summary.json")

    def estimate_cost(
        self,
        number: int,
        trial_dir: Path,
        method_dir: Path,
        checkpoint: str,
    ) -> dict:
        self.run_stage(
            number,
            trial_dir,
            "operation_aware_cost_proxy",
            [
                self.python,
                str(self.script_dir / "estimate_checkpoint_cost.py"),
                str(self.run_dir),
                str(method_dir),
                "--checkpoint",
                checkpoint,
                "--cost-model",
                str(self.cost_model),
            ],
        )
        return load_json(method_dir / "bayesian_cost_proxy.json")

    def guarded_pipeline(
        self,
        number: int,
        trial_dir: Path,
        trial_relative: str,
        parameters: dict,
    ) -> tuple[dict, dict, Path, str]:
        first, second = guarded_two_pass_configs(
            self.templates["first_resynthesis"],
            self.templates["second_resynthesis"],
            parameters,
            self.budgets,
            trial_relative,
            int(self.protocol["execution"]["model_seed"]),
            self.smoke,
        )
        config_dir = trial_dir / "input_configs"
        first_path = config_dir / "first_resynthesis.json"
        second_path = config_dir / "second_resynthesis.json"
        write_json(first_path, first)
        write_json(second_path, second)
        self.run_stage(
            number,
            trial_dir,
            "first_resynthesis",
            [
                self.python,
                str(self.script_dir / "circuit_distillation.py"),
                str(self.run_dir),
                "--config",
                str(first_path),
            ],
        )
        self.run_stage(
            number,
            trial_dir,
            "second_resynthesis",
            [
                self.python,
                str(self.script_dir / "circuit_distillation.py"),
                str(self.run_dir),
                "--config",
                str(second_path),
            ],
        )
        first_dir = self.run_dir / first["output"]
        second_dir = self.run_dir / second["output"]
        self.run_stage(
            number,
            trial_dir,
            "calibration_guard_evaluation",
            [
                self.python,
                str(self.script_dir / "evaluate_calibration_guard.py"),
                str(self.run_dir),
                str(second_dir),
                "--checkpoint",
                "distilled_checkpoint.pt",
                "--partition-config",
                str(first_path),
                "--reporting-folds",
                str(self.protocol["execution"]["reporting_folds"]),
            ],
        )
        summaries = {
            "first_resynthesis": load_json(first_dir / "summary.json"),
            "second_resynthesis": load_json(second_dir / "summary.json"),
            "calibration_guard_evaluation": load_json(
                second_dir / "bayesian_guard_evaluation.json"
            ),
        }
        behavior = summaries["calibration_guard_evaluation"]["guard"]
        internal_guard = summaries["second_resynthesis"].get("guard_holdout")
        if internal_guard is None:
            raise TrialStageError("guarded two-pass trial did not produce guard metrics")
        for key in (
            "accuracy",
            "accuracy_loss",
            "decision_flip_rate",
            "maximum_per_class_accuracy_loss",
            "maximum_per_class_disagreement",
        ):
            if abs(float(behavior[key]) - float(internal_guard[key])) > 1e-9:
                raise TrialStageError(
                    f"independent guard evaluation mismatch for {key}"
                )
        return summaries, behavior, second_dir, "distilled_checkpoint.pt"

    def aggressive_pipeline(
        self,
        number: int,
        trial_dir: Path,
        trial_relative: str,
        parameters: dict,
    ) -> tuple[dict, dict, Path, str, int, int]:
        first, recovery = aggressive_recovery_configs(
            self.templates["first_resynthesis"],
            self.templates["recovery"],
            parameters,
            self.budgets,
            trial_relative,
            int(self.protocol["execution"]["model_seed"]),
            self.smoke,
        )
        config_dir = trial_dir / "input_configs"
        first_path = config_dir / "aggressive_resynthesis.json"
        recovery_path = config_dir / "short_recovery.json"
        write_json(first_path, first)
        write_json(recovery_path, recovery)
        self.run_stage(
            number,
            trial_dir,
            "aggressive_resynthesis",
            [
                self.python,
                str(self.script_dir / "circuit_distillation.py"),
                str(self.run_dir),
                "--config",
                str(first_path),
            ],
        )
        self.run_stage(
            number,
            trial_dir,
            "short_recovery",
            [
                self.python,
                str(self.script_dir / "recovery_finetune.py"),
                str(self.run_dir),
                "--config",
                str(recovery_path),
            ],
        )
        first_dir = self.run_dir / first["output"]
        recovery_dir = self.run_dir / recovery["output"]
        self.run_stage(
            number,
            trial_dir,
            "calibration_guard_evaluation",
            [
                self.python,
                str(self.script_dir / "evaluate_calibration_guard.py"),
                str(self.run_dir),
                str(recovery_dir),
                "--checkpoint",
                "recovered_checkpoint.pt",
                "--partition-config",
                str(first_path),
                "--reporting-folds",
                str(self.protocol["execution"]["reporting_folds"]),
            ],
        )
        summaries = {
            "aggressive_resynthesis": load_json(first_dir / "summary.json"),
            "short_recovery": load_json(recovery_dir / "summary.json"),
            "calibration_guard_evaluation": load_json(
                recovery_dir / "bayesian_guard_evaluation.json"
            ),
        }
        recovery_summary = summaries["short_recovery"]
        selected_step = int(recovery_summary["selected_step"])
        snapshots = load_json_list(recovery_dir / "snapshot_metrics.json")
        selected_records = [row for row in snapshots if int(row["step"]) == selected_step]
        if len(selected_records) != 1:
            raise TrialStageError("selected recovery snapshot is not uniquely logged")
        locked_violations = int(selected_records[0]["locked_row_violations"])
        behavior = summaries["calibration_guard_evaluation"]["guard"]
        return (
            summaries,
            behavior,
            recovery_dir,
            "recovered_checkpoint.pt",
            selected_step,
            locked_violations,
        )

    def __call__(self, trial) -> tuple[float, float]:
        number = int(trial.number)
        trial_dir = self.trials_dir / f"trial_{number:05d}"
        if trial_dir.exists():
            raise TrialStageError(f"refusing to overwrite existing trial {trial_dir}")
        trial_dir.mkdir(parents=True)
        started = time.perf_counter()
        record = {
            "format_version": 1,
            "trial_number": number,
            "study_case": self.case_name,
            "method": self.method,
            "disagreement_policy": self.policy,
            "status": "running",
            "started_at_utc": utc_now(),
            "protocol_sha256": object_sha256(self.protocol),
            "model_seed": int(self.protocol["execution"]["model_seed"]),
            "sampler_seed": int(self.protocol["sampler"]["seed"]),
            "smoke": self.smoke,
            "validation_used": False,
            "test_used": False,
        }
        self.write_trial_record(trial_dir, record)
        self.event("trial_started", trial_number=number)
        try:
            parameters = suggest_parameters(
                trial, self.protocol["search_spaces"][self.method]
            )
            record["parameters"] = parameters
            write_json(trial_dir / "suggested_parameters.json", parameters)
            trial_relative = str(trial_dir.relative_to(self.run_dir))
            if self.method == METHOD_GUARDED:
                summaries, behavior, method_dir, checkpoint = self.guarded_pipeline(
                    number, trial_dir, trial_relative, parameters
                )
                selected_step = 0
                locked_violations = 0
            elif self.method == METHOD_AGGRESSIVE:
                (
                    summaries,
                    behavior,
                    method_dir,
                    checkpoint,
                    selected_step,
                    locked_violations,
                ) = self.aggressive_pipeline(
                    number, trial_dir, trial_relative, parameters
                )
            else:  # pragma: no cover - fixed study matrix prevents this.
                raise ValueError(self.method)

            cost_proxy = self.estimate_cost(
                number, trial_dir, method_dir, checkpoint
            )
            summaries["operation_aware_cost_proxy"] = cost_proxy
            proxy = float(cost_proxy["predicted_abc_and_nodes"])

            names, residuals = constraint_values(
                behavior,
                self.budgets,
                self.policy,
                self.method,
                locked_row_violations=locked_violations,
                selected_recovery_step=selected_step,
                maximum_recovery_steps=self.maximum_recovery_steps,
            )
            feasible = is_feasible(residuals)
            trial.set_user_attr("constraint_names", names)
            trial.set_user_attr("constraint_values", residuals)
            trial.set_user_attr("behavior_feasible", feasible)
            trial.set_user_attr("predicted_abc_and_nodes", proxy)
            trial.set_user_attr("trial_record", str((trial_dir / "trial_record.json").relative_to(self.output_root)))

            synthesis = None
            if not self.smoke and (
                (feasible and self.auto_synthesize_feasible)
                or self.synthesize_infeasible
            ):
                synthesis = self.synthesize(
                    number, trial_dir, method_dir, checkpoint
                )
                fidelity = "exact_abc"
            elif self.smoke:
                fidelity = "smoke_proxy"
                self.event(
                    "synthesis_skipped",
                    trial_number=number,
                    reason="smoke_mode",
                )
            else:
                fidelity = "calibrated_proxy"
                self.event(
                    "synthesis_skipped",
                    trial_number=number,
                    reason=(
                        "behavioral_constraints_violated"
                        if not feasible
                        else "deferred_to_exact_promotion"
                    ),
                )

            accuracy_objective = float(behavior["accuracy_loss"])
            hardware_objective = proxy
            record.update(
                {
                    "status": "completed",
                    "finished_at_utc": utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "parameters": parameters,
                    "accuracy_loss": accuracy_objective,
                    "abc_and_nodes": (
                        None if synthesis is None else int(synthesis["abc_and_nodes"])
                    ),
                    "hardware_objective": hardware_objective,
                    "hardware_proxy": proxy,
                    "optimizer_hardware_fidelity": (
                        "calibrated_operation_aware_proxy"
                    ),
                    "objective_fidelity": fidelity,
                    "constraint_names": names,
                    "constraint_values": residuals,
                    "feasible": feasible,
                    "selected_recovery_step": (
                        selected_step if self.method == METHOD_AGGRESSIVE else None
                    ),
                    "locked_row_violations": locked_violations,
                    "metrics": {
                        "selection_guard": behavior,
                        "stages": summaries,
                        "synthesis": synthesis,
                    },
                    "artifacts": {
                        "trial_directory": str(trial_dir),
                        "method_directory": str(method_dir),
                        "checkpoint": checkpoint,
                        "checkpoint_sha256": sha256_file(method_dir / checkpoint),
                    },
                    "validation_used": False,
                    "test_used": False,
                }
            )
            if synthesis is not None:
                record["unit_tying_abc_node_delta"] = int(
                    synthesis["abc_and_nodes"]
                    - self.protocol["references"]["unit_tying"]["abc_and_nodes"]
                )
            self.write_trial_record(trial_dir, record)
            self.event(
                "trial_completed",
                trial_number=number,
                feasible=feasible,
                objective_fidelity=fidelity,
                wall_seconds=record["wall_seconds"],
            )
            return accuracy_objective, hardware_objective
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "finished_at_utc": utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "traceback": traceback.format_exc(),
                    "parameters": record.get("parameters", dict(trial.params)),
                    "feasible": False,
                    "validation_used": False,
                    "test_used": False,
                }
            )
            trial.set_user_attr("failure_type", type(error).__name__)
            trial.set_user_attr("failure_message", str(error))
            trial.set_user_attr("trial_record", str((trial_dir / "trial_record.json").relative_to(self.output_root)))
            self.write_trial_record(trial_dir, record)
            self.event(
                "trial_failed",
                trial_number=number,
                failure_type=type(error).__name__,
                failure_message=str(error),
            )
            raise


def load_json_list(path: Path) -> list:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--study",
        choices=["all", *STUDY_CASES],
        default="all",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Additional trials per selected study (defaults to the protocol budget).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two-update GPU chains and use a marked proxy instead of ABC.",
    )
    parser.add_argument(
        "--synthesize-infeasible",
        action="store_true",
        help="Also run exact ABC for behaviorally infeasible trials.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen protocol and source artifacts without trials.",
    )
    return parser.parse_args()


def make_sampler(optuna, protocol: dict, method: str, policy: str):
    expected_names = constraint_names(policy, method)

    def constraints_func(frozen_trial):
        values = frozen_trial.user_attrs.get("constraint_values")
        if values is None or len(values) != len(expected_names):
            return [1.0e6] * len(expected_names)
        return [float(value) for value in values]

    settings = protocol["sampler"]
    if settings["name"] != "motpe":
        raise ValueError("this frozen protocol currently supports sampler.name=motpe")
    return optuna.samplers.TPESampler(
        seed=int(settings["seed"]),
        n_startup_trials=int(settings["n_startup_trials"]),
        multivariate=bool(settings.get("multivariate", True)),
        group=bool(settings.get("group", True)),
        constant_liar=bool(settings.get("constant_liar", False)),
        constraints_func=constraints_func,
    )


def main() -> None:
    cli = parse_args()
    run_dir = cli.run_dir.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = load_json(protocol_path)
    validate_protocol(protocol, protocol_path, run_dir)
    repository_root = TORCHLOGIX_ROOT
    output_name = protocol["protocol_name"] + ("_smoke" if cli.smoke else "")
    output_root = run_dir / "bayesian_search" / output_name
    if cli.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "protocol": str(protocol_path),
                    "protocol_sha256": object_sha256(protocol),
                    "source_run": str(run_dir),
                    "studies": list(STUDY_CASES),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if output_root.exists() and not cli.resume:
        raise RuntimeError(
            f"refusing to reuse Bayesian output without --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_root / "protocol.snapshot.json"
    if snapshot_path.exists():
        snapshot = load_json(snapshot_path)
        if object_sha256(snapshot) != object_sha256(protocol):
            raise RuntimeError("protocol changed since the resumable study was created")
    else:
        write_json(snapshot_path, protocol)
        capture_environment(output_root, repository_root, protocol)
        write_json(
            output_root / "source_artifacts.json",
            {
                "run_dir": str(run_dir),
                "training_config": {
                    "path": str(run_dir / "training_config.json"),
                    "sha256": sha256_file(run_dir / "training_config.json"),
                },
                "teacher_checkpoint": {
                    "path": str(run_dir / "best_checkpoint.pt"),
                    "sha256": sha256_file(run_dir / "best_checkpoint.pt"),
                },
                "cost_model": {
                    "path": str(run_dir / protocol["execution"]["cost_model"]),
                    "sha256": sha256_file(
                        run_dir / protocol["execution"]["cost_model"]
                    ),
                },
                "validation_used": False,
                "test_used": False,
            },
        )

    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is missing; install experiments/requirements.txt in torchlogix/venv"
        ) from error
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optimizer_log = output_root / "optimizer.log"
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optimizer_log))
    storage = f"sqlite:///{(output_root / 'study.sqlite3').resolve()}"
    selected_cases = list(STUDY_CASES) if cli.study == "all" else [cli.study]
    n_trials = int(
        cli.n_trials
        if cli.n_trials is not None
        else protocol["sampler"]["trials_per_study"]
    )
    if n_trials <= 0:
        raise ValueError("n-trials must be positive")

    for case_name in selected_cases:
        method, policy = STUDY_CASES[case_name]
        study_dir = output_root / case_name
        if study_dir.exists() and not cli.resume:
            raise RuntimeError(f"refusing to overwrite study {study_dir}")
        study_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            study_dir / "study_manifest.json",
            {
                "format_version": 1,
                "case": case_name,
                "method": method,
                "disagreement_policy": policy,
                "objectives": protocol["objectives"],
                "optimizer_objectives": protocol["optimizer_objectives"],
                "directions": ["minimize", "minimize"],
                "constraint_names": constraint_names(policy, method),
                "active_budgets": active_budgets(protocol, policy),
                "search_space": protocol["search_spaces"][method],
                "sampler": protocol["sampler"],
                "smoke": cli.smoke,
                "synthesize_infeasible": cli.synthesize_infeasible,
                "validation_used": False,
                "test_used": False,
            },
        )
        sampler = make_sampler(optuna, protocol, method, policy)
        study_name = f"{output_name}:{case_name}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            directions=["minimize", "minimize"],
            load_if_exists=cli.resume,
        )
        if not study.trials and protocol["sampler"].get("enqueue_reference", True):
            study.enqueue_trial(protocol["reference_parameters"][method])
        executor = StudyExecutor(
            run_dir,
            repository_root,
            protocol_path,
            protocol,
            output_root,
            case_name,
            cli.smoke,
            cli.synthesize_infeasible,
        )

        def callback(_study, _trial):
            export_study_tables(study_dir)

        study.optimize(
            executor,
            n_trials=n_trials,
            callbacks=[callback],
            catch=(TrialStageError,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
        export_study_tables(study_dir)
    write_json(
        output_root / "execution_summary.json",
        {
            "format_version": 1,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "selected_cases": selected_cases,
            "additional_trials_per_study": n_trials,
            "storage": str(output_root / "study.sqlite3"),
            "optimizer_log": str(optimizer_log),
            "validation_used": False,
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
