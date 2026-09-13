"""Exactly-once, predeclared CIFAR-10.1 v6 evaluation; no retraining or tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coverage_dlgn.prepare_fourth_round import ROOT, REPO, read, write_new
from experiments.coverage_dlgn.run_fourth_round import sha, implementation_hash

REVISION = "d9982abb0bfc4846b8d13a11e66b887d946205d0"
FREEZE = ROOT / "summary/fourth_round_transfer_freeze.json"
LOG = ROOT / "logs/fourth_round_transfer"


def utc():
    return datetime.now(timezone.utc).isoformat()


def selected_runs():
    rows = []
    for seed in range(3):
        for family, name in [("random", f"paper_cifar10_medium_random_seed{seed}"),
                             ("u2", f"third_u2_cifar10_m_seed{seed}")]:
            rows.append(dict(coordinate="dense_m", seed=seed, family=family, name=name))
    for coordinate, names in [
        ("conv_s", ["second_full_conv_cifar10_s_random_seed0", "second_final_u2_conv_cifar10_s_seed0"]),
        ("conv_m", ["full_conv_cifar10_paper_medium_random_seed0_200k", "full_conv_cifar10_paper_medium_u2_seed0_200k"]),
    ]:
        rows.extend(dict(coordinate=coordinate, seed=0, family=family, name=name)
                    for family, name in zip(["random", "u2"], names))
    return rows


def freeze():
    if FREEZE.exists():
        frozen = read(FREEZE)
        verify_selection(frozen)
        return frozen
    if LOG.exists():
        raise RuntimeError("transfer history already exists; cannot declare untouched access")
    rows = []
    for row in selected_runs():
        path = ROOT / "results" / row["name"]
        rows.append(dict(row, artifacts={name: sha(path / name) for name in
                                        ["training_config.json", "best_checkpoint.pt"]}))
    payload = dict(dataset="CIFAR-10.1", version="v6", revision=REVISION,
                   examples=2000, frozen_at_utc=utc(), dataset_accessed=False,
                   checkpoint_selection="existing best hardened CIFAR-10 validation; all cells predeclared",
                   preprocessing="NHWC uint8 to NCHW float32 / 255; checkpoint thresholds; no adaptation",
                   rows=rows)
    write_new(FREEZE, payload)
    return payload


def verify_selection(frozen):
    expected = selected_runs()
    rows = frozen.get("rows", [])
    if (frozen.get("dataset") != "CIFAR-10.1" or frozen.get("version") != "v6"
            or frozen.get("revision") != REVISION or frozen.get("examples") != 2000
            or len(rows) != len(expected)):
        raise RuntimeError("transfer selection differs from preregistration")
    for row, selection in zip(rows, expected):
        artifacts = row.get("artifacts", {})
        if (row != dict(selection, artifacts=artifacts)
                or set(artifacts) != {"training_config.json", "best_checkpoint.pt"}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in artifacts.values())):
            raise RuntimeError("transfer selection or required hashes differ from preregistration")


def validate_arrays(images, labels):
    import numpy as np
    if images.shape != (2000, 32, 32, 3) or images.dtype != np.uint8:
        raise RuntimeError("unexpected CIFAR-10.1 v6 image encoding")
    if labels.shape != (2000,) or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError("unexpected CIFAR-10.1 v6 label encoding")
    if (np.any(labels < 0) or np.any(labels > 9)
            or not np.array_equal(np.bincount(labels.astype(np.int64), minlength=10), np.full(10, 200))):
        raise RuntimeError("expected ten balanced classes in v6")


def load_frozen_model(row):
    import torch
    from experiments.train import get_parser
    from experiments.utils import get_model
    path = ROOT / "results" / row["name"]
    for name, expected in row["artifacts"].items():
        if sha(path / name) != expected:
            raise RuntimeError("transfer checkpoint/config changed after freeze")
    args = get_parser().parse_args([])
    vars(args).update(read(path / "training_config.json"))
    args.device = "cuda"
    checkpoint = torch.load(path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model = get_model(state["0.thresholds"], args)
    model.load_state_dict(state, strict=True)
    return model.to("cuda").eval()


def smoke_checkpoints():
    """Validate every selected checkpoint with synthetic inputs, never data."""
    import torch
    frozen = read(FREEZE)
    verify_selection(frozen)
    results = []
    for row in frozen["rows"]:
        model = load_frozen_model(row)
        with torch.inference_mode():
            x = torch.arange(2 * 3 * 32 * 32, device="cuda").reshape(2, 3, 32, 32).remainder(256).float() / 255
            output = model(x)
        if output.shape != (2, 10) or not torch.isfinite(output).all():
            raise RuntimeError(f"invalid hardened transfer smoke: {row['name']}")
        results.append(dict(name=row["name"], shape=list(output.shape), finite=True))
        del model, output, x
        torch.cuda.empty_cache()
    return dict(freeze_sha256=sha(FREEZE), synthetic_only=True, rows=results)


def evaluate(data_dir):
    import numpy as np
    import torch
    from experiments.coverage_dlgn.run_fourth_round import verify_freeze
    from experiments.coverage_dlgn.summarize_third_round import mean_sd_ci
    # The same verified implementation used for the new runs is mandatory.
    verify_freeze(read(ROOT / "protocols/fourth_round.json"))
    frozen = read(FREEZE)
    verify_selection(frozen)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen transfer")
    torch.ones(1, device="cuda")
    for row in frozen["rows"]:
        for name, expected in row["artifacts"].items():
            if sha(ROOT / "results" / row["name"] / name) != expected:
                raise RuntimeError("transfer checkpoint/config changed after freeze")
    if (LOG / "started.json").exists():
        raise RuntimeError("transfer already started; inspect saved results, never re-query automatically")
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / "started.json").open("x") as handle:
        json.dump(dict(started_at_utc=utc(), freeze_sha256=sha(FREEZE),
                       implementation_sha256=implementation_hash()), handle)
    # Started marker precedes any image/label download or read, even on failure.
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind in ["data", "labels"]:
        filename = f"cifar10.1_v6_{kind}.npy"
        destination = data_dir / filename
        if destination.exists():
            raise RuntimeError("unclassified transfer dataset already exists; audit before use")
        url = f"https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/{REVISION}/datasets/{filename}"
        with urllib.request.urlopen(url, timeout=120) as response, destination.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        paths[kind] = destination
    write_new(LOG / "dataset_receipt.json", dict(revision=REVISION,
              files={kind: dict(path=str(path), sha256=sha(path)) for kind, path in paths.items()}))
    images = np.load(paths["data"], allow_pickle=False)
    labels = np.load(paths["labels"], allow_pickle=False)
    validate_arrays(images, labels)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255)
    results = []
    for row in frozen["rows"]:
        model = load_frozen_model(row)
        predictions = []
        with torch.inference_mode():
            for batch in x.split(128):
                predictions.extend(model(batch.to("cuda")).argmax(-1).cpu().tolist())
        correct = (np.asarray(predictions) == labels).tolist()
        record = dict(row, scope="T", provenance="OUR" if row["family"] == "u2" else "REPRODUCED",
                      examples=len(labels), hard_accuracy_pct=100 * np.mean(correct),
                      predictions=predictions, correct=correct, evaluated_at_utc=utc())
        write_new(LOG / f"{row['name']}.json", record)
        results.append(record)
        print(f"{row['coordinate']} {row['family']} seed {row['seed']}: {record['hard_accuracy_pct']:.3f}%", flush=True)
        del model
        torch.cuda.empty_cache()
    comparisons = []
    for coordinate in ["dense_m", "conv_s", "conv_m"]:
        pairs = []
        for seed in sorted({r["seed"] for r in results if r["coordinate"] == coordinate}):
            r = {r["family"]: r for r in results if r["coordinate"] == coordinate and r["seed"] == seed}
            a, b = np.asarray(r["random"]["correct"]), np.asarray(r["u2"]["correct"])
            pairs.append(dict(seed=seed, gain_pp=r["u2"]["hard_accuracy_pct"] - r["random"]["hard_accuracy_pct"],
                              u2_only_correct=int((b & ~a).sum()), random_only_correct=int((a & ~b).sum())))
        comparisons.append(dict(coordinate=coordinate, per_seed=pairs,
                                paired_effect=mean_sd_ci([r["gain_pp"] for r in pairs]),
                                wins=sum(r["gain_pp"] > 0 for r in pairs)))
    summary = dict(status="complete", freeze_sha256=sha(FREEZE), scope="T",
                   rows=[{k: v for k, v in r.items() if k not in {"predictions", "correct"}} for r in results],
                   comparisons=comparisons, no_adaptation=True, checkpoint_queries=len(results))
    write_new(ROOT / "summary/fourth_round_transfer_results.json", summary)
    write_new(LOG / "completed.json", dict(completed_at_utc=utc(), checkpoint_queries=len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/torchlogix-datasets/cifar10.1-frozen-fourth"))
    args = parser.parse_args()
    os.chdir(REPO)
    if args.action == "freeze":
        print(json.dumps(freeze(), indent=2))
    else:
        evaluate(args.data_dir)
