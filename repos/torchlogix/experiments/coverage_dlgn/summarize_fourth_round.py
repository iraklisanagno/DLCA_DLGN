"""Read verified validation artifacts; write a complete fourth-round report once."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coverage_dlgn.prepare_fourth_round import ROOT, REPO, read, write_new
from experiments.coverage_dlgn.run_fourth_round import LOG, FREEZE, sha, validate_run, verify_freeze
from experiments.coverage_dlgn.summarize_third_round import mean_sd_ci


def collect(row, frozen):
    output = REPO / row["output"]
    hashes = validate_run(row)
    expected = (frozen["reused_artifacts"][row["output"]] if row["reuse"]
                else read(LOG / f"{row['name']}.complete.json")["artifacts"])
    if hashes != expected:
        raise RuntimeError(f"artifact changed: {row['name']}")
    config = read(REPO / row["config"])
    summary = read(output / "run_summary.json")
    with (output / "metrics.csv").open() as handle:
        history = [{key: float(value) for key, value in record.items() if value}
                   for record in csv.DictReader(handle)]
    best = max(history, key=lambda r: r["val_acc_discrete"])
    last = history[-1]
    # Historical pilots predate the cost field; use this exact architecture's
    # preflight measurement, not an inference from file size or physical area.
    cost = frozen["costs"][row["coordinate"] + "|" + config["parametrization"]]
    if summary.get("cost", cost) != cost:
        raise RuntimeError(f"declared cost changed: {row['name']}")
    return dict(row, config=row["config"], artifact_hashes=hashes,
                architecture=config["architecture"], parametrization=config["parametrization"],
                steps=config["num_iterations"], best_validation_hard_pct=100 * best["val_acc_discrete"],
                best_step=int(best["step"]), best_step_relaxed_pct=100 * best["val_acc_relaxed"],
                best_step_gap_pp=100 * (best["val_acc_relaxed"] - best["val_acc_discrete"]),
                final_validation_hard_pct=100 * last["val_acc_discrete"],
                final_validation_relaxed_pct=100 * last["val_acc_relaxed"],
                final_gap_pp=100 * (last["val_acc_relaxed"] - last["val_acc_discrete"]),
                wall_minutes=summary["wall_seconds"] / 60,
                peak_gpu_gib=summary["peak_gpu_memory_bytes"] / 2**30,
                topology_seconds=sum(layer.get("construction_seconds", 0.) for layer in summary["topology"]),
                cost=cost, validation_curve=history)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["coordinate"], row["family"])].append(row)
    metrics = ["best_validation_hard_pct", "final_validation_hard_pct", "best_step_gap_pp",
               "final_gap_pp", "wall_minutes", "peak_gpu_gib", "topology_seconds"]
    return [dict(phase=key[0], coordinate=key[1], family=key[2], scope="V",
                 provenance=group[0]["provenance"], seeds=sorted(r["seed"] for r in group),
                 metrics={metric: mean_sd_ci([r[metric] for r in group]) for metric in metrics})
            for key, group in sorted(groups.items())]


def contrast(rows, phase, coordinate, weights, label):
    arms = {family: {r["seed"]: r for r in rows
                     if r["phase"] == phase and r["coordinate"] == coordinate and r["family"] == family}
            for family in weights}
    seeds = sorted(set.intersection(*(set(arm) for arm in arms.values())))
    if not seeds:
        return None
    effects = []
    for seed in seeds:
        costs = [arm[seed]["cost"] for arm in arms.values()]
        if any(cost != costs[0] for cost in costs):
            raise RuntimeError(f"unmatched declared cost: {label}")
        effects.append(dict(seed=seed,
            gain_pp=sum(weights[family] * arms[family][seed]["best_validation_hard_pct"] for family in weights),
            construction_delta_seconds=sum(weights[family] * arms[family][seed]["topology_seconds"] for family in weights)))
    return dict(phase=phase, coordinate=coordinate, contrast=label, scope="V", weights=weights,
                per_seed=effects, paired_effect=mean_sd_ci([r["gain_pp"] for r in effects]),
                construction_delta_seconds=mean_sd_ci([r["construction_delta_seconds"] for r in effects]),
                wins=sum(r["gain_pp"] > 0 for r in effects), equal_declared_cost=True)


def effects(rows):
    output = []
    for coordinate in ["dense_m", "conv_s"]:
        for control in ["random", "balanced_random", "nominal"]:
            output.append(contrast(rows, "mechanism", coordinate, {"u2": 1, control: -1}, f"u2_minus_{control}"))
    for label, weights in [
        ("body_with_random_head", {"body": 1, "random": -1}),
        ("body_with_u2_head", {"both": 1, "head": -1}),
        ("head_with_random_body", {"head": 1, "random": -1}),
        ("head_with_u2_body", {"both": 1, "body": -1}),
        ("body_head_interaction", {"both": 1, "body": -1, "head": -1, "random": 1}),
    ]:
        output.append(contrast(rows, "factorial", "conv_s", weights, label))
    for phase, coordinate in [("warp_pilot", "dense_m"), ("warp_confirm", "dense_m"), ("warp_conv", "conv_s")]:
        output.append(contrast(rows, phase, coordinate, {"u2": 1, "random": -1}, "u2_minus_random"))
    return [row for row in output if row is not None]


def build_report():
    payload = read(ROOT / "protocols/fourth_round.json")
    frozen = verify_freeze(payload)
    promotion_path = LOG / "warp_promotion.json"
    promotion = read(promotion_path) if promotion_path.exists() else None
    rows, pending, skipped = [], [], []
    for row in payload["entries"]:
        if row["phase"] in {"warp_confirm", "warp_conv"} and promotion is not None and not promotion["passed"]:
            if (LOG / f"{row['name']}.complete.json").exists():
                raise RuntimeError("unpromoted arm was run")
            skipped.append(dict(name=row["name"], reason="predeclared WARP promotion failed"))
        elif row["reuse"] or (LOG / f"{row['name']}.complete.json").exists():
            rows.append(collect(row, frozen))
        else:
            pending.append(row["name"])
    pilots = {r["family"]: r for r in rows if r["phase"] == "warp_pilot"}
    if promotion is not None:
        if set(pilots) != {"random", "u2"}:
            raise RuntimeError("promotion lacks completed pilots")
        expected = pilots["random"]["best_validation_hard_pct"] >= 40 and pilots["u2"]["best_validation_hard_pct"] > pilots["random"]["best_validation_hard_pct"]
        if promotion["passed"] != expected:
            raise RuntimeError("promotion disagrees with frozen criterion")
    references = [collect(row, frozen) for row in payload["raw_warp_references"]]
    raw_comparisons = []
    for row in rows:
        if not row["phase"].startswith("warp_"):
            continue
        reference_phase = "raw_reference" if row["phase"] == "warp_confirm" else "mechanism"
        matched = [r for r in rows + references if r["phase"] == reference_phase
                   and all(r[k] == row[k] for k in ["coordinate", "family", "seed", "steps"])]
        if len(matched) == 1:
            raw_comparisons.append(dict(name=row["name"], raw_reference=matched[0]["name"], scope="V", provenance="ADAPTED",
                                        warp_minus_raw_pp=row["best_validation_hard_pct"] - matched[0]["best_validation_hard_pct"]))
    return dict(status="complete" if not pending and promotion is not None else "incomplete",
                scope="V", preregistration_sha256=sha(FREEZE), rows=rows, groups=aggregate(rows),
                paired_effects=effects(rows), pending=pending, skipped=skipped,
                warp_promotion=promotion, raw_warp_references=references, warp_vs_raw=raw_comparisons,
                limitations=["Three-seed exploratory paired intervals are not multiplicity-adjusted",
                             "Wall time may reflect shared-host contention; topology construction is offline",
                             "No new CIFAR-10 test queries; no physical hardware claim"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-final", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_final:
        if report["status"] != "complete":
            raise RuntimeError("cannot freeze an incomplete experiment report")
        write_new(ROOT / "summary/fourth_round_results.json", report)
        fields = ["name", "phase", "coordinate", "family", "seed", "provenance", "scope",
                  "best_validation_hard_pct", "final_validation_hard_pct", "final_gap_pp",
                  "wall_minutes", "peak_gpu_gib", "topology_seconds"]
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["rows"])
        path = ROOT / "summary/fourth_round_runs.csv"
        if path.exists():
            if path.read_text() != handle.getvalue():
                raise RuntimeError("refusing to change existing CSV")
        else:
            with path.open("x") as output:
                output.write(handle.getvalue())
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"rows", "raw_warp_references"}}, indent=2))


if __name__ == "__main__":
    main()
