"""Check factorial independence in saved fixed wiring; no model/data evaluation.

This supplemental artifact audit does not modify the frozen experiment code.
Default execution is read-only; an incomplete factorial cannot be certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.coverage_dlgn.prepare_fourth_round import ROOT, REPO, read, write_new
from experiments.coverage_dlgn.run_fourth_round import FREEZE, LOG, sha, validate_run, verify_freeze


def fingerprints(state):
    head = sorted(k for k in state if k.endswith(".connections.indices"))
    body = sorted(k for k in state if ".connections._indices_L" in k)
    spatial = sorted(k for k in body if k.endswith("_indices_L0"))
    if len(head) != 3 or len(body) != 12 or len(spatial) != 4:
        raise RuntimeError("unexpected paper-S saved wiring layout")

    def digest(keys, spatial_only=False):
        result = hashlib.sha256()
        for key in keys:
            value = state[key].detach().cpu()
            if spatial_only:
                if value.shape[-1] != 3:
                    raise RuntimeError("unexpected spatial/channel coordinate axis")
                value = value[..., :-1]
            result.update(key.encode())
            result.update(str((tuple(value.shape), value.dtype)).encode())
            result.update(value.contiguous().numpy().tobytes())
        return result.hexdigest()

    return dict(head=digest(head), body=digest(body), spatial=digest(spatial, True))


def check_factorial(arms):
    if set(arms) != {"random", "body", "head", "both"}:
        raise RuntimeError("all four factorial arms are required")
    checks = dict(
        matched_spatial=len({r["spatial"] for r in arms.values()}) == 1,
        random_head_held_fixed=arms["random"]["head"] == arms["body"]["head"],
        u2_head_held_fixed=arms["head"]["head"] == arms["both"]["head"],
        random_body_held_fixed=arms["random"]["body"] == arms["head"]["body"],
        u2_body_held_fixed=arms["body"]["body"] == arms["both"]["body"],
        distinct_body_treatments=arms["random"]["body"] != arms["body"]["body"],
        distinct_head_treatments=arms["random"]["head"] != arms["head"]["head"],
    )
    if not all(checks.values()):
        raise RuntimeError(f"factorial saved wiring confounded: {checks}")
    return checks


def audit(smokes=False):
    import torch
    payload = read(ROOT / "protocols/fourth_round.json")
    frozen = verify_freeze(payload)
    rows = [r for r in payload["entries"] if r["phase"] == "factorial"]
    if smokes:
        rows = [r for r in rows if r["seed"] == 0]
    records, pending = [], []
    for row in rows:
        if smokes:
            name = "smoke_" + row["name"]
            output = ROOT / "results" / name
            expected = read(LOG / f"{name}.json")["artifacts"]["best_checkpoint.pt"]
        else:
            name, output = row["name"], REPO / row["output"]
            stamp = LOG / f"{name}.complete.json"
            if not row["reuse"] and not stamp.exists():
                pending.append(name)
                continue
            expected_all = (frozen["reused_artifacts"][row["output"]] if row["reuse"]
                            else read(stamp)["artifacts"])
            if validate_run(row) != expected_all:
                raise RuntimeError("completed factorial artifacts changed")
            expected = expected_all["best_checkpoint.pt"]
        checkpoint = output / "best_checkpoint.pt"
        if sha(checkpoint) != expected:
            raise RuntimeError("checkpoint changed before wiring audit")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        records.append(dict(name=name, seed=row["seed"], family=row["family"],
                            checkpoint_sha256=expected, fingerprints=fingerprints(state)))
        del state
    checks = []
    for seed in sorted({r["seed"] for r in records}):
        arms = {r["family"]: r["fingerprints"] for r in records if r["seed"] == seed}
        if len(arms) == 4:
            checks.append(dict(seed=seed, checks=check_factorial(arms)))
    return dict(status="incomplete" if pending else "pass", smoke_only=smokes,
        scope="saved_wiring_only_no_inference_or_dataset_access", preregistration_sha256=sha(FREEZE),
        auditor_sha256=sha(Path(__file__)), regression_test_sha256=sha(ROOT / "test_saved_wiring_audit.py"),
        records=records, per_seed=checks, pending=pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smokes", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = audit(args.smokes)
    if args.write:
        if report["status"] != "pass":
            raise RuntimeError("cannot certify incomplete factorial wiring")
        suffix = "smoke_" if args.smokes else ""
        write_new(ROOT / f"summary/fourth_round_factorial_{suffix}wiring_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
