"""Verify saved transfer predictions and repair reporting, without model inference.

The frozen evaluator completed all ten queries but its NumPy-valued win count
failed JSON serialization. This separate recovery leaves that frozen source and
the original records untouched. Default execution is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.coverage_dlgn.prepare_fourth_round import ROOT, read, write_new
from experiments.coverage_dlgn.run_fourth_round import sha, verify_freeze
from experiments.coverage_dlgn.evaluate_frozen_transfer import FREEZE, LOG, verify_selection
from experiments.coverage_dlgn.summarize_third_round import mean_sd_ci


def summary_from_records(frozen, records, freeze_hash):
    verify_selection(frozen)
    if len(records) != len(frozen["rows"]):
        raise RuntimeError("incomplete saved transfer records")
    for row, selected in zip(records, frozen["rows"]):
        if any(row.get(k) != v for k, v in selected.items()):
            raise RuntimeError("saved record selection or hashes changed")
        if (row.get("scope") != "T" or row.get("examples") != 2000
                or row.get("provenance") != ("OUR" if row["family"] == "u2" else "REPRODUCED")):
            raise RuntimeError("invalid transfer metadata")
        predictions, correct = row.get("predictions", []), row.get("correct", [])
        if (len(predictions) != 2000 or len(correct) != 2000
                or any(type(p) is not int or not 0 <= p < 10 for p in predictions)
                or any(type(c) is not bool for c in correct)):
            raise RuntimeError("invalid saved predictions or correctness")
        accuracy = 100 * sum(correct) / 2000
        if not math.isclose(row["hard_accuracy_pct"], accuracy, abs_tol=1e-10):
            raise RuntimeError("saved accuracy disagrees with correctness")
    comparisons = []
    for coordinate in ["dense_m", "conv_s", "conv_m"]:
        pairs = []
        for seed in sorted({r["seed"] for r in records if r["coordinate"] == coordinate}):
            arms = {r["family"]: r for r in records
                    if r["coordinate"] == coordinate and r["seed"] == seed}
            a, b = arms["random"]["correct"], arms["u2"]["correct"]
            pairs.append(dict(seed=seed,
                gain_pp=float(arms["u2"]["hard_accuracy_pct"] - arms["random"]["hard_accuracy_pct"]),
                u2_only_correct=sum(y and not x for x, y in zip(a, b)),
                random_only_correct=sum(x and not y for x, y in zip(a, b))))
        comparisons.append(dict(coordinate=coordinate, per_seed=pairs,
            paired_effect=mean_sd_ci([r["gain_pp"] for r in pairs]),
            wins=sum(r["gain_pp"] > 0 for r in pairs)))
    return dict(status="complete", freeze_sha256=freeze_hash, scope="T",
        rows=[{k: v for k, v in r.items() if k not in {"predictions", "correct"}} for r in records],
        comparisons=comparisons, no_adaptation=True, checkpoint_queries=len(records))


def verify_saved():
    import numpy as np
    execution = verify_freeze(read(ROOT / "protocols/fourth_round.json"))
    frozen, started = read(FREEZE), read(LOG / "started.json")
    freeze_hash = sha(FREEZE)
    if (started["freeze_sha256"] != freeze_hash
            or started["implementation_sha256"] != execution["implementation_sha256"]):
        raise RuntimeError("evaluation did not use the frozen selection/implementation")
    if datetime.fromisoformat(frozen["frozen_at_utc"]) >= datetime.fromisoformat(started["started_at_utc"]):
        raise RuntimeError("selection was not frozen before access")
    receipt = read(LOG / "dataset_receipt.json")
    if receipt["revision"] != frozen["revision"] or set(receipt["files"]) != {"data", "labels"}:
        raise RuntimeError("dataset receipt changed")
    for record in receipt["files"].values():
        if sha(record["path"]) != record["sha256"]:
            raise RuntimeError("downloaded dataset differs from its receipt")
    # Read only already-downloaded labels to audit recorded predictions. No
    # images are decoded, no model is loaded, and no new predictions are made.
    labels = np.load(receipt["files"]["labels"]["path"], allow_pickle=False)
    if (labels.shape != (2000,) or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels, return_counts=True),
                                  np.array([np.arange(10), np.full(10, 200)]))):
        raise RuntimeError("invalid saved class-balanced labels")
    labels = labels.tolist()
    records, hashes = [], {}
    for selected in frozen["rows"]:
        path = LOG / f"{selected['name']}.json"
        row = read(path)
        hashes[path.name] = sha(path)
        for name, expected in selected["artifacts"].items():
            if sha(ROOT / "results" / selected["name"] / name) != expected:
                raise RuntimeError("checkpoint/config changed")
        if row["correct"] != [p == y for p, y in zip(row["predictions"], labels)]:
            raise RuntimeError("predictions disagree with saved labels/correctness")
        if datetime.fromisoformat(row["evaluated_at_utc"]) <= datetime.fromisoformat(started["started_at_utc"]):
            raise RuntimeError("prediction record predates started marker")
        records.append(row)
    expected_files = set(hashes) | {"started.json", "dataset_receipt.json", "recovery.json", "completed.json"}
    if {p.name for p in LOG.glob("*.json")} - expected_files:
        raise RuntimeError("unexpected transfer history; manual audit required")
    summary = summary_from_records(frozen, records, freeze_hash)
    # Prove serialization before writing any recovery artifact.
    json.dumps(summary, allow_nan=False)
    output = ROOT / "summary/fourth_round_transfer_results.json"
    if output.exists() and read(output) != summary:
        raise RuntimeError("existing summary differs from saved predictions")
    completed_path = LOG / "completed.json"
    if completed_path.exists():
        completed, recovery = read(completed_path), read(LOG / "recovery.json")
        if (completed.get("checkpoint_queries") != 10
                or completed.get("recovered_from_saved_predictions") is not True
                or completed.get("summary_sha256") != sha(output)
                or completed.get("recovery_sha256") != sha(LOG / "recovery.json")
                or recovery.get("saved_prediction_hashes") != hashes
                or recovery.get("additional_inference_queries") != 0
                or recovery.get("started_sha256") != sha(LOG / "started.json")
                or recovery.get("receipt_sha256") != sha(LOG / "dataset_receipt.json")
                or recovery.get("recovery_source_sha256") != sha(Path(__file__))
                or recovery.get("regression_test_sha256") != sha(ROOT / "test_transfer_report_recovery.py")):
            raise RuntimeError("recovery/completion audit hashes changed")
        if any(datetime.fromisoformat(r["evaluated_at_utc"]) >=
               datetime.fromisoformat(completed["completed_at_utc"]) for r in records):
            raise RuntimeError("completion marker predates saved evaluations")
    return summary, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="repair missing aggregate/completion only")
    args = parser.parse_args()
    summary, hashes = verify_saved()
    if args.write:
        if (LOG / "completed.json").exists():
            raise RuntimeError("already completed; use read-only verification")
        timestamp = datetime.now(timezone.utc).isoformat()
        recovery = dict(recovered_at_utc=timestamp, checkpoint_queries=10,
            additional_inference_queries=0, saved_prediction_hashes=hashes,
            started_sha256=sha(LOG / "started.json"), receipt_sha256=sha(LOG / "dataset_receipt.json"),
            recovery_source_sha256=sha(Path(__file__)),
            regression_test_sha256=sha(ROOT / "test_transfer_report_recovery.py"),
            original_failure="TypeError: Object of type int64 is not JSON serializable; aggregate write after all ten saved records",
            frozen_evaluator_unchanged=True)
        write_new(LOG / "recovery.json", recovery)
        write_new(ROOT / "summary/fourth_round_transfer_results.json", summary)
        write_new(LOG / "completed.json", dict(completed_at_utc=timestamp, checkpoint_queries=10,
            recovered_from_saved_predictions=True, recovery_sha256=sha(LOG / "recovery.json"),
            summary_sha256=sha(ROOT / "summary/fourth_round_transfer_results.json")))
    print(json.dumps(dict(status="verified", checkpoint_queries=10, additional_inference_queries=0,
                         comparisons=summary["comparisons"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
