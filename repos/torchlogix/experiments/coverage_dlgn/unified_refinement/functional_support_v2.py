"""Read-only local LUT relevance and conservative path support; no data access."""
from __future__ import annotations

import argparse
import csv
import io
import numpy as np
import torch

from torchlogix.parametrization import RawLUTParametrization
from torchlogix.topology import image_input_semantics, packed_popcount
from .common import ROOT, OLD, REPO, read, sha, verify_incumbent, write_json


def relevance(luts):
    luts = np.asarray(luts)
    if luts.ndim != 2 or luts.shape[1] != 4 or not np.isin(luts, [0, 1]).all():
        raise ValueError("expected Boolean truth tables ordered 00,01,10,11")
    return np.stack(((luts[:, 0] != luts[:, 2]) | (luts[:, 1] != luts[:, 3]),
                     (luts[:, 0] != luts[:, 1]) | (luts[:, 2] != luts[:, 3])))


def layer_analysis(indices, luts, source_ids):
    """Return metrics and supports; path union is NOT exact global dependence."""
    source_ids = np.asarray(source_ids)
    if (source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype, np.integer)
            or len(source_ids) == 0 or source_ids.min() < 0):
        raise ValueError("invalid source mapping")
    if len(indices) != len(luts) or not indices:
        raise ValueError("matching nonempty layer lists required")
    n_sources = int(source_ids.max()) + 1
    support = np.zeros((len(source_ids), (n_sources + 63) // 64), dtype=np.uint64)
    support[np.arange(len(source_ids)), source_ids // 64] = np.left_shift(
        np.uint64(1), (source_ids % 64).astype(np.uint64))
    structural = support.copy()
    records, dependencies, widths = [], [], [len(source_ids)]
    for depth, (parents, tables) in enumerate(zip(indices, luts)):
        parents = np.asarray(parents)
        if (parents.ndim != 2 or parents.shape[0] != 2 or parents.shape[1] < 1
                or not np.issubdtype(parents.dtype, np.integer)
                or parents.min() < 0 or parents.max() >= len(support)):
            raise ValueError("invalid checkpoint rank-two indices")
        # Native random may repeat a parent. Retain its actual wiring; local
        # relevance over free LUT inputs remains a conservative path bound.
        relevant = relevance(tables)
        if relevant.shape != parents.shape:
            raise ValueError("LUT/gate count differs")
        dependencies.append(relevant)
        widths.append(parents.shape[1])
        next_support = np.zeros((parents.shape[1], support.shape[1]), dtype=np.uint64)
        for slot in (0, 1):
            active = relevant[slot]
            next_support[active] |= support[parents[slot, active]]
        structural = structural[parents[0]] | structural[parents[1]]
        support = next_support
        active_degree = np.bincount(parents[relevant], minlength=widths[-2])
        count = relevant.sum(axis=0)
        structural_sizes = packed_popcount(structural)
        support_sizes = packed_popcount(support)
        records.append(dict(depth=depth, gates=parents.shape[1],
            constant_fraction=float(np.mean(count == 0)),
            unary_fraction=float(np.mean(count == 1)),
            binary_fraction=float(np.mean(count == 2)),
            locally_relevant_edges=int(count.sum()),
            predecessor_active_degree_min=int(active_degree.min()),
            predecessor_active_degree_max=int(active_degree.max()),
            predecessor_active_degree_mean=float(active_degree.mean()),
            unused_predecessor_fraction=float(np.mean(active_degree == 0)),
            structural_sources_mean=float(structural_sizes.mean()),
            dependency_path_sources_mean=float(support_sizes.mean()),
            empty_dependency_support_fraction=float(np.mean(support_sizes == 0))))
    active = np.ones(widths[-1], dtype=bool)
    for depth in reversed(range(len(indices))):
        records[depth]["output_path_reachable_gate_fraction"] = float(active.mean())
        before = np.zeros(widths[depth], dtype=bool)
        parents = np.asarray(indices[depth])
        for slot in (0, 1):
            before[parents[slot, active & dependencies[depth][slot]]] = True
        active = before
    return dict(layers=records, n_sources=n_sources,
                input_bit_path_reachable_fraction=float(active.mean()),
                raw_source_path_reachable_fraction=float(np.unique(source_ids[active]).size / n_sources)), support


def freeze_selection():
    verify_incumbent()
    report = read(OLD / "summary/fourth_round_results.json")
    rows = [r for r in report["rows"] if r["phase"] == "mechanism" and r["coordinate"] == "dense_m"]
    if len(rows) != 12:
        raise RuntimeError("expected all twelve dense mechanism rows")
    selected = []
    for row in rows:
        for kind in ("best", "final"):
            filename = f"{kind}_checkpoint.pt"
            selected.append(dict(name=row["name"], family=row["family"], seed=row["seed"],
                output=row["output"], kind=kind, checkpoint_sha256=row["artifact_hashes"][filename],
                configuration_sha256=row["artifact_hashes"]["training_config.json"],
                validation_hard_pct=row[f"{'best' if kind == 'best' else 'final'}_validation_hard_pct"]))
    value = dict(scope="checkpoint_only_no_dataset_no_inference", rows=selected,
        sources={name: sha(ROOT / name) for name in ["functional_support_v2.py", "construction.py", "common.py"]},
        protocol_sha256=sha(ROOT / "PROTOCOL.md"),
        definitions="Exact local LUT relevance; dependency-pruned path unions upper-bound global support; GroupSum inputs are reachability roots; constants remain contributions; associations descriptive")
    write_json(ROOT / "analysis_selection_v2.json", value)
    return value


def analyze():
    frozen = read(ROOT / "analysis_selection_v2.json")
    verify_incumbent()
    for name, expected in frozen["sources"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("analysis implementation changed after selection freeze")
    if sha(ROOT / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("analysis protocol changed")
    results = []
    for row in frozen["rows"]:
        output = REPO / row["output"]
        checkpoint = output / f"{row['kind']}_checkpoint.pt"
        if sha(checkpoint) != row["checkpoint_sha256"] or sha(output / "training_config.json") != row["configuration_sha256"]:
            raise RuntimeError("saved input changed")
        cfg = read(output / "training_config.json")
        if cfg["architecture"] != "DlgnCifar10Medium" or cfg["parametrization"] != "raw" or cfg["lut_rank"] != 2:
            raise RuntimeError("unsupported architecture or LUT semantics")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved["model_state_dict"]
        keys = sorted(k for k in state if k.endswith(".connections.indices"))
        if len(keys) != 4 or state["0.thresholds"].shape[-1] != 3:
            raise RuntimeError("unexpected dense checkpoint layout")
        indices, tables = [], []
        param = RawLUTParametrization(2)
        for key in keys:
            weight = state[key.removesuffix("connections.indices") + "weight"]
            if weight.ndim != 2 or weight.shape[1] != 16 or not torch.isfinite(weight).all():
                raise RuntimeError("invalid raw gate parameters")
            indices.append(state[key].numpy())
            tables.append(param.get_luts(weight).numpy())
        semantics = image_input_semantics(3, 32, 32, 3, layout="channel_interleaved")
        metrics, _ = layer_analysis(indices, tables, semantics.source_ids)
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed during analysis")
        results.append(dict(row, **metrics))
        print("Analyzed",row["name"],row["kind"],flush=True)
        del saved, state, indices, tables
    report = dict(status="complete", selection_sha256=sha(ROOT / "analysis_selection_v2.json"),
                  scope=frozen["scope"], checkpoint_count=len(results), rows=results,
                  limitations=[frozen["definitions"], "No causal or physical-cost conclusion; n=3, paired checkpoints are not independent replicates"])
    write_json(ROOT / "summary/functional_support.json", report)
    flat = [dict(name=r["name"], family=r["family"], seed=r["seed"], kind=r["kind"], **layer)
            for r in results for layer in r["layers"]]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(flat)
    path = ROOT / "summary/functional_support.csv"
    if path.exists() and path.read_text() != handle.getvalue():
        raise RuntimeError("refusing to replace analysis CSV")
    if not path.exists():
        with path.open("x") as stream: stream.write(handle.getvalue())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "analyze"])
    args = parser.parse_args()
    result = freeze_selection() if args.action == "freeze" else analyze()
    print("Selected/analyzed",len(result["rows"]),"checkpoints")


if __name__ == "__main__":
    main()

