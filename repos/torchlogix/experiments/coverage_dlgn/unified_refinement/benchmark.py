"""Scoped dense index-stack benchmark, NOT end-to-end model construction."""
from __future__ import annotations
import argparse
import json
import resource
import subprocess
import sys
import time
from .common import ROOT, REPO, write_json
from .construction import nominal_indices, fingerprint


def worker(mode):
    from torchlogix.topology import generate_dense_topology, image_input_semantics, propagate_packed_ancestry
    semantics = image_input_semantics(3,32,32,3,layout='channel_interleaved')
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.perf_counter()
    ancestry = semantics.source_ancestry() if mode != 'ur1_index_only' else None
    n, hashes, degrees = 9216, [], []
    import numpy as np
    for depth in range(4):
        if mode == 'legacy_with_ancestry':
            result = generate_dense_topology(n,128000,strategy='semantic_multiscale_nominal',
                topology_seed=0,layer_index=depth,input_ancestry=ancestry,
                input_semantics=semantics if depth == 0 else None)
            indices, ancestry = result.indices, result.output_ancestry
        else:
            indices = nominal_indices(n,128000,layer_index=depth,topology_seed=0,
                input_semantics=semantics if depth == 0 else None)
            if ancestry is not None: ancestry = propagate_packed_ancestry(ancestry,indices)
        hashes.append(fingerprint(indices))
        fanout = np.bincount(indices.ravel(),minlength=n)
        degrees.append(dict(min=int(fanout.min()),max=int(fanout.max())))
        n = 128000
    elapsed = time.perf_counter()-started
    return dict(mode=mode,seconds=elapsed,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        preconstruction_peak_rss_bytes=baseline,indices_sha256=hashes,fanout=degrees,
        ancestry_output_sha256=fingerprint(ancestry) if ancestry is not None else None)


def benchmark():
    records=[]
    for repeat in range(3):
        for mode in ['legacy_with_ancestry','ur1_index_only','ur1_with_ancestry']:
            command=[sys.executable,'-m','experiments.coverage_dlgn.unified_refinement.benchmark','--worker',mode]
            result=subprocess.run(command,cwd=REPO,text=True,capture_output=True,check=True)
            row=json.loads(result.stdout)
            records.append(dict(repeat=repeat,**row))
            print(mode,repeat,round(row['seconds'],3),'s',flush=True)
    if any(r['indices_sha256'] != records[0]['indices_sha256'] for r in records):
        raise RuntimeError('benchmark index identity failed')
    ancestry=[r['ancestry_output_sha256'] for r in records if r['ancestry_output_sha256'] is not None]
    if len(set(ancestry)) != 1: raise RuntimeError('ancestry output identity failed')
    report=dict(status='complete',scope='CPU dense-M index-stack construction; NOT end-to-end model construction',
        cache_policy='fresh interpreter per observation; imports and semantic metadata setup excluded equally; no topology warmup',
        seed=0,repeats=3,rows=records,
        limitations=['Peak RSS includes interpreter/imports, not isolated allocator memory',
        'No model initialization, spatial sampling, GPU transfer or full optional topology diagnostics measured',
        'Legacy builder propagates ancestry; index-only row intentionally omits it; use ancestry-matched rows for that comparison',
        'No accuracy or physical area/energy claim'])
    write_json(ROOT/'summary/construction_benchmark.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['legacy_with_ancestry','ur1_index_only','ur1_with_ancestry'])
    args=parser.parse_args()
    if args.worker: print(json.dumps(worker(args.worker)))
    else: benchmark()
