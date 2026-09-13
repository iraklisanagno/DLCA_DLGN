"""Fail-closed fourth-round verification and serial CUDA execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coverage_dlgn.prepare_fourth_round import ROOT, REPO, U2, prepare, read, write_new
from experiments.coverage_dlgn.run_gpu_queue import is_complete

LOG = ROOT / "logs/fourth_round"
FREEZE = ROOT / "summary/fourth_round_preregistration.json"
ARTIFACTS = ["training_config.json", "environment.json", "metrics.csv", "run_summary.json",
             "best_checkpoint.pt", "final_checkpoint.pt", "topology.json"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def implementation_hash():
    paths = list((REPO / "src/torchlogix").rglob("*.py"))
    paths += [REPO / "experiments/train.py", REPO / "experiments/utils.py"]
    paths += [ROOT / "run_gpu_queue.py"]
    paths += list((REPO / "tests").rglob("*.py"))
    paths += list(ROOT.glob("*fourth_round*.py"))
    paths += [ROOT / "evaluate_frozen_transfer.py", ROOT / "FOURTH_ROUND_PROTOCOL.md"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_run(row):
    output = REPO / row["output"]
    if not is_complete(output):
        raise RuntimeError(f"missing complete artifacts: {output}")
    cfg = read(REPO / row["config"])
    actual = read(output / "training_config.json")
    ignored = {"config", "output", "classifier_reference_u2"}
    for key, value in cfg.items():
        if key not in ignored and actual.get(key) != value:
            raise RuntimeError(f"configuration mismatch: {output}: {key}")
    if not row.get("reuse") and actual.get("classifier_reference_u2", False) != cfg.get("classifier_reference_u2", False):
        raise RuntimeError("factorial classifier-reference mismatch")
    with (output / "metrics.csv").open() as handle:
        records = list(csv.DictReader(handle))
    expected = list(range(cfg["eval_freq"], cfg["num_iterations"] + 1, cfg["eval_freq"]))
    if [int(r["step"]) for r in records] != expected:
        raise RuntimeError(f"incomplete/duplicated validation history: {output}")
    for record in records:
        for key, value in record.items():
            if value and not math.isfinite(float(value)):
                raise RuntimeError(f"nonfinite {key}: {output}")
    summary = read(output / "run_summary.json")
    best = max(float(r["val_acc_discrete"]) for r in records)
    reported_best = float(summary["best_validation_hard_accuracy"])
    if not math.isfinite(reported_best) or abs(best - reported_best) > 1e-7:
        raise RuntimeError(f"best validation mismatch: {output}")
    memory = float(summary["peak_gpu_memory_bytes"])
    if not math.isfinite(memory) or memory <= 0:
        raise RuntimeError(f"missing GPU evidence: {output}")
    return {name: sha(output / name) for name in ARTIFACTS}


def command(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        subprocess.run(args, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def model_signature(config):
    import torch
    from experiments.train import get_parser, model_cost_summary
    from experiments.utils import get_model
    from torchlogix.layers import LogicDense, LogicConv2d
    args = get_parser().parse_args([])
    vars(args).update(config)
    torch.manual_seed(args.seed)
    model = get_model(torch.tensor([.25, .5, .75], device="cuda"), args)
    def digest(tensors):
        h = hashlib.sha256()
        for tensor in tensors:
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    signature = dict(cost=model_cost_summary(model), parameters=digest(model.parameters()),
                     cpu_rng=digest([torch.get_rng_state()]),
                     cuda_rng=digest([torch.cuda.get_rng_state()]),
                     spatial=digest(m.connections.indices[0][..., :-1] for m in model if isinstance(m, LogicConv2d)),
                     classifier=digest(m.connections.indices for m in model if isinstance(m, LogicDense)))
    del model
    torch.cuda.empty_cache()
    return signature


def preflight(payload):
    import torch
    if FREEZE.exists():
        verify_freeze(payload)
        print("Existing complete verification and preregistration still match", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.ones(1, device="cuda")
    from experiments.coverage_dlgn import evaluate_frozen_transfer as transfer
    transfer.freeze()
    LOG.mkdir(parents=True, exist_ok=True)
    command([sys.executable, "-m", "pytest", "tests", "-q"], LOG / "full_tests.log")
    signatures = {}
    smokes = []
    representatives = {}
    for row in payload["entries"]:
        cfg = read(REPO / row["config"])
        key = (row["coordinate"], cfg["parametrization"], cfg["connections_init_method"],
               cfg.get("classifier_connections_init_method"), cfg["classifier_reference_u2"])
        if row["seed"] == 0:
            representatives.setdefault(key, row)
    for key, row in representatives.items():
        cfg = read(REPO / row["config"])
        signature = model_signature(cfg)
        baseline = signatures.setdefault((key[0], key[1]), signature)
        for field in ["cost", "parameters", "cpu_rng", "cuda_rng", "spatial"]:
            if signature[field] != baseline[field]:
                raise RuntimeError(f"unpaired {field}: {row['name']}")
        if key[-1]:
            head_key = (key[0], key[1], key[3], "head")
            previous = signatures.setdefault(head_key, signature)
            if previous["classifier"] != signature["classifier"]:
                raise RuntimeError("factorial classifier wiring changed with body")
        name = "smoke_" + row["name"]
        output = ROOT / "results" / name
        smoke = dict(cfg, num_iterations=10, eval_freq=10, output=str(output.relative_to(REPO)))
        path = ROOT / "configs/fourth_round/smokes" / f"{name}.json"
        write_new(path, smoke)
        smoke_row = dict(row, reuse=False, config=str(path.relative_to(REPO)), output=smoke["output"])
        stamp = LOG / f"{name}.json"
        if stamp.exists():
            previous = read(stamp)
            if previous["implementation_sha256"] != implementation_hash() or previous["artifacts"] != validate_run(smoke_row):
                raise RuntimeError("stale smoke; archive failed attempt before retry")
        else:
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"unclassified smoke output: {output}")
            print(f"GPU smoke {name}", flush=True)
            command([sys.executable, "experiments/train.py", "--config", str(path)], LOG / f"{name}.log")
            write_new(stamp, dict(implementation_sha256=implementation_hash(), artifacts=validate_run(smoke_row), signature=signature))
        smokes.append(str(stamp.relative_to(REPO)))
    transfer_smoke = LOG / "transfer_smoke.json"
    write_new(transfer_smoke, transfer.smoke_checkpoints())
    smokes.append(str(transfer_smoke.relative_to(REPO)))
    all_rows = payload["entries"] + payload["raw_warp_references"]
    reuse = {r["output"]: validate_run(r) for r in all_rows if r["reuse"]}
    write_new(FREEZE, dict(implementation_sha256=implementation_hash(),
                          config_hashes={r["config"]: sha(REPO / r["config"]) for r in all_rows},
                          costs={"|".join(key): value["cost"] for key, value in signatures.items() if len(key) == 2},
                          reused_artifacts=reuse, gpu_smokes={path: sha(REPO / path) for path in smokes},
                          full_tests_sha256=sha(LOG / "full_tests.log"),
                          transfer_freeze_sha256=sha(transfer.FREEZE),
                          protocol_sha256=sha(ROOT / "protocols/fourth_round.json")))
    print("Full tests, paired construction, GPU smokes and reuse validation passed; frozen", flush=True)


def verify_freeze(payload):
    frozen = read(FREEZE)
    if implementation_hash() != frozen["implementation_sha256"]:
        raise RuntimeError("implementation changed after verification")
    if sha(ROOT / "protocols/fourth_round.json") != frozen["protocol_sha256"]:
        raise RuntimeError("protocol changed after freeze")
    if sha(LOG / "full_tests.log") != frozen["full_tests_sha256"]:
        raise RuntimeError("full-test evidence changed after freeze")
    if sha(ROOT / "summary/fourth_round_transfer_freeze.json") != frozen["transfer_freeze_sha256"]:
        raise RuntimeError("transfer selection changed after freeze")
    for path, expected in frozen["gpu_smokes"].items():
        if sha(REPO / path) != expected:
            raise RuntimeError("GPU smoke evidence changed after freeze")
    for row in payload["entries"] + payload["raw_warp_references"]:
        if sha(REPO / row["config"]) != frozen["config_hashes"][row["config"]]:
            raise RuntimeError("configuration changed after freeze")
        if row["reuse"] and validate_run(row) != frozen["reused_artifacts"][row["output"]]:
            raise RuntimeError("reused artifacts changed after freeze")
    return frozen


def run(payload):
    import torch
    verify_freeze(payload)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.ones(1, device="cuda")
    for phase in ["mechanism", "factorial", "warp_pilot", "warp_confirm", "warp_conv"]:
        if phase in {"warp_confirm", "warp_conv"}:
            pilot = {r["family"]: read(REPO / r["output"] / "run_summary.json")["best_validation_hard_accuracy"]
                     for r in payload["entries"] if r["phase"] == "warp_pilot"}
            passed = pilot["random"] >= .4 and pilot["u2"] > pilot["random"]
            write_new(LOG / "warp_promotion.json", dict(passed=passed, scope="V", pilot=pilot))
            if not passed:
                print(f"Not promoted: {phase}", flush=True)
                continue
        for row in payload["entries"]:
            if row["phase"] != phase or row["reuse"]:
                continue
            stamp = LOG / f"{row['name']}.complete.json"
            if stamp.exists():
                if read(stamp)["artifacts"] != validate_run(row):
                    raise RuntimeError("completed output changed")
                continue
            output = REPO / row["output"]
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"unclassified output, do not restart automatically: {output}")
            print(f"Starting {row['name']}", flush=True)
            command([sys.executable, "experiments/train.py", "--config", row["config"]], LOG / f"{row['name']}.log")
            write_new(stamp, dict(artifacts=validate_run(row), scope="V"))
            print(f"Completed {row['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "preflight", "run"])
    args = parser.parse_args()
    os.chdir(REPO)
    payload = prepare()
    if args.action == "preflight": preflight(payload)
    if args.action == "run": run(payload)


if __name__ == "__main__":
    main()
