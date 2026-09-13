"""Verify saved checkpoint selection and metadata without inference or data access.

This supplemental auditor is outside the frozen training implementation. Default
execution is read-only; a final certificate requires the entire promoted matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.coverage_dlgn.prepare_fourth_round import ROOT, REPO, read, write_new
from experiments.coverage_dlgn.run_fourth_round import sha
from experiments.coverage_dlgn.summarize_fourth_round import build_report

METRICS = ("val_acc_discrete", "val_acc_relaxed", "val_loss_discrete", "val_loss_relaxed")


def check_metadata(metadata, configuration, history, kind):
    if kind not in {"best", "final"} or not history:
        raise RuntimeError("expected best/final checkpoint and nonempty history")
    for record in history:
        if any(not math.isfinite(float(record[key])) for key in ("step", *METRICS)):
            raise RuntimeError("nonfinite checkpoint reference history")
    # The training callback uses strict improvement: ties retain the first max.
    selected = (max(history, key=lambda row: row["val_acc_discrete"])
                if kind == "best" else history[-1])
    if metadata.get("step") != selected["step"]:
        raise RuntimeError(f"{kind} checkpoint selected/final step mismatch")
    if metadata.get("configuration") != configuration:
        raise RuntimeError(f"{kind} checkpoint configuration mismatch")
    saved = metadata.get("metrics", {})
    for key in METRICS:
        value = saved.get(key)
        if (value is None or not math.isfinite(float(value))
                or abs(float(value) - selected[key]) > 1e-7):
            raise RuntimeError(f"{kind} checkpoint metric mismatch: {key}")
    return dict(kind=kind, step=int(selected["step"]),
                metrics={key: selected[key] for key in METRICS})


def audit():
    import torch
    report = build_report()  # Verify frozen source, configs and completed artifacts first.
    unique = {}
    for row in report["rows"] + report["raw_warp_references"]:
        if row["output"] in unique:
            if unique[row["output"]]["artifact_hashes"] != row["artifact_hashes"]:
                raise RuntimeError("reused checkpoint aliases disagree")
        else:
            unique[row["output"]] = row
    records = []
    for output, row in sorted(unique.items()):
        directory = REPO / output
        configuration = read(directory / "training_config.json")
        for kind in ("best", "final"):
            filename = f"{kind}_checkpoint.pt"
            path = directory / filename
            expected = row["artifact_hashes"][filename]
            if sha(path) != expected:
                raise RuntimeError("checkpoint changed before metadata audit")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checked = check_metadata(checkpoint["metadata"], configuration,
                                     row["validation_curve"], kind)
            del checkpoint
            if sha(path) != expected:
                raise RuntimeError("checkpoint changed during metadata audit")
            records.append(dict(output=output, checkpoint_sha256=expected, **checked))
    return dict(status="pass" if report["status"] == "complete" else "incomplete",
                scope="completed_matrix_and_raw_references_metadata_only_no_inference_or_dataset_access",
                preregistration_sha256=report["preregistration_sha256"],
                auditor_sha256=sha(Path(__file__)),
                regression_test_sha256=sha(ROOT / "test_checkpoint_metadata_audit.py"),
                completed_matrix_cells=len(report["rows"]), unique_runs=len(unique),
                checkpoints=len(records), pending=report["pending"], records=records)


def write_certificate(report):
    if report["status"] != "pass" or report["pending"]:
        raise RuntimeError("cannot certify incomplete checkpoint metadata")
    write_new(ROOT / "summary/fourth_round_checkpoint_metadata_audit.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = audit()
    if args.write:
        write_certificate(report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
