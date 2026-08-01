"""Frozen search-space and metric helpers for MarginSynth Bayesian studies.

This module deliberately has no Optuna or PyTorch import.  Protocol validation,
configuration materialization, constraint construction, and Pareto extraction
therefore remain testable on CPU-only machines and after the optimizer API
changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METHOD_GUARDED = "guarded_two_pass"
METHOD_AGGRESSIVE = "aggressive_recovery"
POLICY_CONSTRAINED = "constrained"
POLICY_UNCONSTRAINED = "unconstrained"

STUDY_CASES = {
    "guarded_constrained": (METHOD_GUARDED, POLICY_CONSTRAINED),
    "guarded_unconstrained": (METHOD_GUARDED, POLICY_UNCONSTRAINED),
    "aggressive_constrained": (METHOD_AGGRESSIVE, POLICY_CONSTRAINED),
    "aggressive_unconstrained": (METHOD_AGGRESSIVE, POLICY_UNCONSTRAINED),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_protocol(protocol: dict, protocol_path: Path, run_dir: Path) -> None:
    required = {
        "format_version",
        "protocol_name",
        "sampler",
        "objectives",
        "optimizer_objectives",
        "budgets",
        "study_cases",
        "templates",
        "search_spaces",
        "execution",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    if int(protocol["format_version"]) != 1:
        raise ValueError("only Bayesian protocol format_version=1 is supported")
    if protocol["objectives"] != ["accuracy_loss", "abc_and_nodes"]:
        raise ValueError("the frozen objectives must be accuracy loss and ABC AND nodes")
    if protocol["optimizer_objectives"] != [
        "accuracy_loss",
        "predicted_abc_and_nodes",
    ]:
        raise ValueError(
            "optimizer objectives must be accuracy loss and calibrated ABC prediction"
        )
    declared = protocol["study_cases"]
    if sorted(declared) != sorted(STUDY_CASES):
        raise ValueError("study_cases must declare exactly the four prespecified cases")
    for name, (method, policy) in STUDY_CASES.items():
        if declared[name] != {"method": method, "disagreement_policy": policy}:
            raise ValueError(f"study case {name} does not match its frozen semantics")
    budgets = protocol["budgets"]
    for key in ("accuracy_loss", "per_class_accuracy_loss"):
        if key not in budgets:
            raise ValueError(f"missing mandatory budget {key}")
    for key in ("disagreement", "per_class_disagreement"):
        if key not in budgets:
            raise ValueError(f"missing constrained-study budget {key}")
    for method in (METHOD_GUARDED, METHOD_AGGRESSIVE):
        if method not in protocol["search_spaces"]:
            raise ValueError(f"missing search space for {method}")
        for name, spec in protocol["search_spaces"][method].items():
            validate_search_dimension(name, spec)
    for name, relative in protocol["templates"].items():
        template = (protocol_path.parent / relative).resolve()
        if not template.is_file():
            raise FileNotFoundError(f"missing template {name}: {template}")
    required_inputs = [
        run_dir / "training_config.json",
        run_dir / "best_checkpoint.pt",
        run_dir / "export_verification.json",
        run_dir / protocol["execution"]["cost_model"],
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("missing source-run inputs: " + ", ".join(missing_inputs))


def validate_search_dimension(name: str, spec: dict) -> None:
    kind = spec.get("type")
    if kind == "float":
        low, high = float(spec["low"]), float(spec["high"])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"invalid float range for {name}")
        if spec.get("log", False) and low <= 0:
            raise ValueError(f"log-scaled range for {name} must be positive")
    elif kind == "int":
        if int(spec["low"]) > int(spec["high"]):
            raise ValueError(f"invalid integer range for {name}")
    elif kind == "categorical":
        if not spec.get("choices"):
            raise ValueError(f"categorical dimension {name} has no choices")
    else:
        raise ValueError(f"unsupported search dimension type for {name}: {kind}")


def suggest_parameters(trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """Materialize a protocol-defined mixed search space through an Optuna trial."""
    values = {}
    for name in sorted(search_space):
        spec = search_space[name]
        if spec["type"] == "float":
            values[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
            )
        elif spec["type"] == "int":
            values[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif spec["type"] == "categorical":
            values[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:  # pragma: no cover - validation prevents this branch.
            raise ValueError(f"unsupported search dimension {name}")
    return values


def active_budgets(protocol: dict, policy: str) -> dict[str, float]:
    budgets = protocol["budgets"]
    active = {
        "accuracy_loss": float(budgets["accuracy_loss"]),
        "per_class_accuracy_loss": float(budgets["per_class_accuracy_loss"]),
    }
    if policy == POLICY_CONSTRAINED:
        active.update(
            {
                "disagreement": float(budgets["disagreement"]),
                "per_class_disagreement": float(
                    budgets["per_class_disagreement"]
                ),
            }
        )
    elif policy != POLICY_UNCONSTRAINED:
        raise ValueError(f"unknown disagreement policy: {policy}")
    return active


def _clean_template(template: dict) -> dict:
    cleaned = copy.deepcopy(template)
    for key in ("source_config", "source_config_sha256", "ablation"):
        cleaned.pop(key, None)
    return cleaned


def _smoke_steps(value: int, smoke: bool) -> int:
    return min(int(value), 2) if smoke else int(value)


def guarded_two_pass_configs(
    first_template: dict,
    second_template: dict,
    parameters: dict,
    budgets: dict,
    trial_relative: str,
    seed: int,
    smoke: bool,
) -> tuple[dict, dict]:
    """Build an independent first pass and locked, guarded second pass."""
    first = _clean_template(first_template)
    first_steps = _smoke_steps(parameters["first_steps"], smoke)
    first.update(
        {
            "method": "bayes-guarded-two-pass-first-resynthesis",
            "teacher_checkpoint": "best_checkpoint.pt",
            "source_checkpoint": "best_checkpoint.pt",
            "action_space": "all",
            "objective": "margin",
            "robust_groups": True,
            "cost_kind": parameters["cost_kind"],
            "optimization_fraction": 0.6,
            "repair_fraction": 0.2,
            "guard_fraction": 0.2,
            "partition_seed": int(seed),
            "steps": first_steps,
            "learning_rate": parameters["first_learning_rate"],
            "initial_logit_gap": parameters["first_initial_logit_gap"],
            "temperature_end": parameters["first_temperature_end"],
            "margin_retention": parameters["margin_retention"],
            "margin_reserve": parameters["margin_reserve"],
            "cost_warmup_steps": max(
                1,
                int(round(first_steps * parameters["first_cost_warmup_fraction"])),
            ),
            "loss_weights": {
                "decision": parameters["decision_weight"],
                "class_worst": parameters["first_class_weight"],
                "fold_worst": parameters["first_fold_weight"],
                "labels": parameters["first_label_weight"],
                "hardware": parameters["first_hardware_weight"],
            },
            "budgets": copy.deepcopy(budgets),
            "selection_budgets": copy.deepcopy(budgets),
            "repair": True,
            "repair_scan": int(parameters["repair_scan"]),
            "report_validation": False,
            "log_every": 1 if smoke else max(10, first_steps // 20),
            "seed": int(seed),
            "output": f"{trial_relative}/first_resynthesis",
        }
    )

    second = _clean_template(second_template)
    second_steps = _smoke_steps(parameters["second_steps"], smoke)
    first_checkpoint = f"{first['output']}/distilled_checkpoint.pt"
    second.update(
        {
            "method": "bayes-guarded-two-pass-second-resynthesis",
            "teacher_checkpoint": "best_checkpoint.pt",
            "source_checkpoint": first_checkpoint,
            "lock_reference_checkpoint": first_checkpoint,
            "lock_source_changes": True,
            "action_space": "all",
            "objective": "margin",
            "robust_groups": True,
            "cost_kind": parameters["cost_kind"],
            "optimization_fraction": 0.6,
            "repair_fraction": 0.2,
            "guard_fraction": 0.2,
            "partition_seed": int(seed),
            "steps": second_steps,
            "learning_rate": parameters["second_learning_rate"],
            "initial_logit_gap": parameters["second_initial_logit_gap"],
            "temperature_end": parameters["second_temperature_end"],
            "margin_retention": parameters["margin_retention"],
            "margin_reserve": parameters["margin_reserve"],
            "cost_warmup_steps": max(
                1,
                int(round(second_steps * parameters["second_cost_warmup_fraction"])),
            ),
            "loss_weights": {
                "decision": parameters["decision_weight"],
                "class_worst": parameters["second_class_weight"],
                "fold_worst": parameters["second_fold_weight"],
                "labels": parameters["second_label_weight"],
                "hardware": parameters["second_hardware_weight"],
            },
            "budgets": copy.deepcopy(budgets),
            "selection_budgets": copy.deepcopy(budgets),
            "repair": True,
            "repair_scan": int(parameters["repair_scan"]),
            "report_validation": False,
            "log_every": 1 if smoke else max(10, second_steps // 20),
            "seed": int(seed + 1009),
            "output": f"{trial_relative}/second_resynthesis",
        }
    )
    return first, second


def aggressive_recovery_configs(
    first_template: dict,
    recovery_template: dict,
    parameters: dict,
    budgets: dict,
    trial_relative: str,
    seed: int,
    smoke: bool,
) -> tuple[dict, dict]:
    """Build aggressive unrepaired resynthesis and locked short recovery."""
    first = _clean_template(first_template)
    first_steps = _smoke_steps(parameters["first_steps"], smoke)
    first.update(
        {
            "method": "bayes-aggressive-first-resynthesis",
            "teacher_checkpoint": "best_checkpoint.pt",
            "source_checkpoint": "best_checkpoint.pt",
            "action_space": "all",
            "objective": "margin",
            "robust_groups": True,
            "cost_kind": parameters["cost_kind"],
            "optimization_fraction": 0.6,
            "repair_fraction": 0.2,
            "guard_fraction": 0.2,
            "partition_seed": int(seed),
            "steps": first_steps,
            "learning_rate": parameters["first_learning_rate"],
            "initial_logit_gap": parameters["first_initial_logit_gap"],
            "temperature_end": parameters["first_temperature_end"],
            "margin_retention": parameters["margin_retention"],
            "margin_reserve": parameters["margin_reserve"],
            "cost_warmup_steps": max(
                1,
                int(round(first_steps * parameters["first_cost_warmup_fraction"])),
            ),
            "loss_weights": {
                "decision": parameters["decision_weight"],
                "class_worst": parameters["first_class_weight"],
                "fold_worst": parameters["first_fold_weight"],
                "labels": parameters["first_label_weight"],
                "hardware": parameters["first_hardware_weight"],
            },
            "budgets": copy.deepcopy(budgets),
            "selection_budgets": copy.deepcopy(budgets),
            "repair": False,
            "repair_scan": 0,
            "report_validation": False,
            "log_every": 1 if smoke else max(10, first_steps // 20),
            "seed": int(seed),
            "output": f"{trial_relative}/aggressive_resynthesis",
        }
    )

    recovery = _clean_template(recovery_template)
    recovery_steps = _smoke_steps(parameters["recovery_steps"], smoke)
    schedule = [0, 250, 500, 1000, 2000, 3000, 5000]
    snapshots = sorted({value for value in schedule if value <= recovery_steps} | {recovery_steps})
    recovery.update(
        {
            "method": "bayes-aggressive-margin-synth-short-recovery",
            "teacher_checkpoint": "best_checkpoint.pt",
            "source_checkpoint": f"{first['output']}/distilled_checkpoint.pt",
            "lock_source_changes": True,
            "steps": recovery_steps,
            "snapshot_steps": snapshots,
            "learning_rate": parameters["recovery_learning_rate"],
            "initial_logit_gap": parameters["recovery_initial_logit_gap"],
            "temperature_end": parameters["recovery_temperature_end"],
            "margin_retention": parameters["margin_retention"],
            "margin_reserve": parameters["margin_reserve"],
            "cost_kind": parameters["cost_kind"],
            "hardware_ceiling_tolerance": parameters[
                "hardware_ceiling_tolerance"
            ],
            "loss_weights": {
                "labels": parameters["recovery_label_weight"],
                "decision": parameters["recovery_decision_weight"],
                "class_worst": parameters["recovery_class_weight"],
                "fold_worst": parameters["recovery_fold_weight"],
                "hardware_ceiling": parameters["recovery_hardware_weight"],
                "entropy": parameters["recovery_entropy_weight"],
            },
            "selection_budgets": copy.deepcopy(budgets),
            "report_budgets": copy.deepcopy(budgets),
            "report_calibration": False,
            "report_validation": False,
            "log_every": 1 if smoke else max(50, recovery_steps // 30),
            "seed": int(seed + 2017),
            "output": f"{trial_relative}/short_recovery",
        }
    )
    return first, recovery


def constraint_names(policy: str, method: str) -> list[str]:
    names = ["accuracy_loss", "maximum_per_class_accuracy_loss"]
    if policy == POLICY_CONSTRAINED:
        names.extend(["disagreement", "maximum_per_class_disagreement"])
    if method == METHOD_AGGRESSIVE:
        names.extend(["locked_row_violations", "recovery_step_budget"])
    return names


def constraint_values(
    metrics: dict,
    budgets: dict,
    policy: str,
    method: str,
    locked_row_violations: int = 0,
    selected_recovery_step: int = 0,
    maximum_recovery_steps: int = 3000,
) -> tuple[list[str], list[float]]:
    names = constraint_names(policy, method)
    residuals = {
        "accuracy_loss": float(metrics["accuracy_loss"])
        - float(budgets["accuracy_loss"]),
        "maximum_per_class_accuracy_loss": float(
            metrics["maximum_per_class_accuracy_loss"]
        )
        - float(budgets["per_class_accuracy_loss"]),
        "disagreement": float(metrics["decision_flip_rate"])
        - float(budgets.get("disagreement", 0.0)),
        "maximum_per_class_disagreement": float(
            metrics["maximum_per_class_disagreement"]
        )
        - float(budgets.get("per_class_disagreement", 0.0)),
        "locked_row_violations": float(locked_row_violations),
        "recovery_step_budget": float(
            selected_recovery_step - maximum_recovery_steps
        ),
    }
    return names, [residuals[name] for name in names]


def is_feasible(values: list[float]) -> bool:
    return all(math.isfinite(value) and value <= 0.0 for value in values)


def dominates(left: dict, right: dict) -> bool:
    """Return whether a feasible completed record Pareto-dominates another."""
    left_values = (left["accuracy_loss"], left["abc_and_nodes"])
    right_values = (right["accuracy_loss"], right["abc_and_nodes"])
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def pareto_records(records: list[dict]) -> list[dict]:
    eligible = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("feasible") is True
        and record.get("objective_fidelity") == "exact_abc"
        and record.get("abc_and_nodes") is not None
    ]
    frontier = [
        record
        for record in eligible
        if not any(
            dominates(other, record)
            for other in eligible
            if other["trial_number"] != record["trial_number"]
        )
    ]
    return sorted(
        frontier,
        key=lambda record: (
            record["accuracy_loss"],
            record["abc_and_nodes"],
            record["trial_number"],
        ),
    )


def _proxy_dominates(left: dict, right: dict) -> bool:
    left_values = (left["accuracy_loss"], left["hardware_proxy"])
    right_values = (right["accuracy_loss"], right["hardware_proxy"])
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def proxy_pareto_records(records: list[dict]) -> list[dict]:
    eligible = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("feasible") is True
        and record.get("hardware_proxy") is not None
    ]
    frontier = [
        record
        for record in eligible
        if not any(
            _proxy_dominates(other, record)
            for other in eligible
            if other["trial_number"] != record["trial_number"]
        )
    ]
    return sorted(
        frontier,
        key=lambda record: (
            record["accuracy_loss"],
            record["hardware_proxy"],
            record["trial_number"],
        ),
    )


def select_promotion_records(records: list[dict], limit: int) -> list[dict]:
    """Choose feasible proxy-Pareto points plus deterministic diverse fillers."""
    if limit <= 0:
        raise ValueError("promotion limit must be positive")
    eligible = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("feasible") is True
        and record.get("hardware_proxy") is not None
    ]
    if len(eligible) <= limit:
        return sorted(eligible, key=lambda record: record["trial_number"])
    frontier = proxy_pareto_records(eligible)
    if len(frontier) > limit:
        if limit == 1:
            return [min(frontier, key=lambda record: record["hardware_proxy"])]
        positions = {
            int(round(index * (len(frontier) - 1) / (limit - 1)))
            for index in range(limit)
        }
        return [frontier[index] for index in sorted(positions)]

    chosen = list(frontier)
    chosen_numbers = {record["trial_number"] for record in chosen}
    by_accuracy = {
        record["trial_number"]: rank
        for rank, record in enumerate(
            sorted(
                eligible,
                key=lambda item: (
                    item["accuracy_loss"], item["hardware_proxy"], item["trial_number"]
                ),
            )
        )
    }
    by_hardware = {
        record["trial_number"]: rank
        for rank, record in enumerate(
            sorted(
                eligible,
                key=lambda item: (
                    item["hardware_proxy"], item["accuracy_loss"], item["trial_number"]
                ),
            )
        )
    }
    fillers = sorted(
        (record for record in eligible if record["trial_number"] not in chosen_numbers),
        key=lambda record: (
            by_accuracy[record["trial_number"]]
            + by_hardware[record["trial_number"]],
            max(
                by_accuracy[record["trial_number"]],
                by_hardware[record["trial_number"]],
            ),
            record["trial_number"],
        ),
    )
    chosen.extend(fillers[: limit - len(chosen)])
    return sorted(chosen, key=lambda record: record["trial_number"])


def flatten_record(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings for stable, tidy CSV exports."""
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, dict):
            output.update(flatten_record(item, child))
        elif isinstance(item, (list, tuple)):
            output[child] = canonical_json(item)
        else:
            output[child] = item
    return output
