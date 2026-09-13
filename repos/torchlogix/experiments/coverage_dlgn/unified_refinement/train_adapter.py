"""Process-local factorial adapter; the historical training implementation is untouched."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
import torch
from torchlogix.layers import LogicDense
from .common import ROOT, REPO, read, sha, source_hash, verify_incumbent, write_json
from .construction import factorial_indices, fingerprint


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dataset_integrity():
    """Require an explicit verified existing dataset; never trigger downloads."""
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity
    configured = os.environ.get('DATASET_PATH')
    if not configured: raise RuntimeError('Set DATASET_PATH to the existing dataset parent (normally /tmp/torchlogix-datasets)')
    root = Path(configured).resolve() / 'data-cifar' / CIFAR10.base_folder
    expected = list(CIFAR10.train_list) + list(CIFAR10.test_list) + [(CIFAR10.meta['filename'], CIFAR10.meta['md5'])]
    hashes = {}
    for name, md5 in expected:
        path = root / name
        if not check_integrity(str(path), md5): raise RuntimeError('Missing/corrupt dataset; refusing automatic download: '+str(path))
        hashes[name] = sha(path)
    return dict(root=str(root),sha256=hashes,scope='file integrity only; no predictions or test evaluation')


def intervene(model, arm, topology_seed):
    """Build once, substitute indices only, retain weights and ALL RNG states."""
    layers = [m for m in model if isinstance(m, LogicDense)]
    if len(layers) != 4 or any(m.connections.lut_rank != 2 for m in layers):
        raise ValueError('UR1 factorial requires the four-layer rank-two dense model')
    reference = [m.connections.indices.detach().cpu().numpy().copy() for m in layers]
    started = time.perf_counter()
    indices, records = factorial_indices(reference, [m.connections.in_dim for m in layers], arm, topology_seed)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        for layer, edges, record in zip(layers, indices, records):
            layer.connections.indices.copy_(torch.as_tensor(edges, device=layer.connections.indices.device))
            if not record['structured']:
                if not record['changed_pair_multiset']:
                    raise RuntimeError('null failed to change actual pair multiset')
                layer.connections.strategy = 'ur1_degree_null'
    # Historical per-layer metadata describe reference construction; the extra
    # null cost is recorded separately, not hidden or attributed repeatedly.
    return dict(arm=arm, topology_seed=topology_seed, layers=records,
                reference_indices_sha256=[fingerprint(a) for a in reference],
                null_construction_seconds=seconds,
                cost_note='Reference construction remains in layer metadata; add null_construction_seconds once')


def analyze_intervened(model):
    """Reuse frozen metric formulas on actual indices, then restore new labels.

    The old analyzer canonicalizes strategy names for its label field only.
    Temporarily supply a known label, never alter indices or ancestry, and
    relabel every returned row to its actual intervention before serialization.
    """
    from torchlogix.topology import analyze_model_topology
    layers = [m for m in model.modules() if isinstance(m, LogicDense)]
    labels = [m.connections.strategy for m in layers]
    try:
        for layer, label in zip(layers, labels):
            if label == 'ur1_degree_null': layer.connections.strategy = 'semantic_multiscale_balanced'
        rows = analyze_model_topology(model)
    finally:
        for layer, label in zip(layers, labels): layer.connections.strategy = label
    if len(rows) != len(layers): raise RuntimeError('unexpected dense metric rows')
    for row, label in zip(rows, labels): row['strategy'] = label
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = read(args.config)
    dataset = dataset_integrity()
    output = (REPO / cfg['output']).resolve()
    if not output.is_relative_to(ROOT / 'results') or output == ROOT / 'results':
        raise RuntimeError('training output must be a new UR1 results child')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('refusing an existing output; no implicit restart')
    if cfg.get('device') != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU training prohibited')
    torch.ones(1, device='cuda')
    verify_incumbent()
    from experiments import train
    original_parser, original_model, original_analyzer = train.get_parser, train.get_model, train.analyze_model_topology
    def get_parser():
        result = original_parser()
        result.add_argument('--ur1_arm', choices=['identity', 'SS', 'SR', 'RS', 'RR'], default='identity')
        return result
    def get_model(thresholds, settings):
        model = original_model(thresholds, settings)
        before = tensor_digest(model.parameters())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        arm = settings.ur1_arm
        if arm == 'identity':
            receipt = dict(arm=arm, null_construction_seconds=0)
        else:
            if (settings.architecture != 'DlgnCifar10Medium' or settings.connections != 'fixed'
                    or settings.connections_init_method != 'semantic_multiscale_balanced'
                    or settings.parametrization != 'raw' or settings.lut_rank != 2):
                raise RuntimeError('unsupported factorial recipe')
            receipt = intervene(model, arm, settings.topology_seed)
        if (before != tensor_digest(model.parameters()) or not torch.equal(cpu_rng, torch.get_rng_state())
                or not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
            raise RuntimeError('intervention changed weights or Torch RNG')
        from experiments.train import model_cost_summary
        write_json(output / 'ur1_construction.json', dict(receipt,
            dataset_integrity=dataset,
            ur1_source_sha256=source_hash(), initial_parameters_sha256=before,
            cpu_rng_sha256=tensor_digest([cpu_rng]), cuda_rng_sha256=tensor_digest([cuda_rng]),
            cost=model_cost_summary(model)))
        return model
    train.get_parser, train.get_model, train.analyze_model_topology = get_parser, get_model, analyze_intervened
    try:
        train.main(['--config', str(args.config.resolve())])
    finally:
        train.get_parser, train.get_model, train.analyze_model_topology = original_parser, original_model, original_analyzer


if __name__ == '__main__':
    main()
