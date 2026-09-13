"""Fail-closed CUDA preflight, archival reuse and run auditing for UR1."""
from __future__ import annotations
import csv
import os
import sys
import torch
import numpy as np
from experiments.coverage_dlgn.run_fourth_round import command, validate_run
from experiments.coverage_dlgn.audit_checkpoint_metadata import check_metadata
from .common import ROOT, OLD, REPO, PRESERVATION, read, sha, source_hash, verify_incumbent, write_json
from .construction import fingerprint, nominal_indices
from .train_adapter import intervene, tensor_digest, dataset_integrity
from .prepare import prepare


def audit_run(row):
    artifacts = validate_run(row)
    directory = REPO / row['output']
    cfg = read(directory / 'training_config.json')
    with (directory / 'metrics.csv').open() as stream:
        history = [{k: float(v) for k, v in r.items() if v} for r in csv.DictReader(stream)]
    checks = []
    construction = None if row.get('reuse') else read(directory / 'ur1_construction.json')
    for kind in ['best', 'final']:
        saved = torch.load(directory / f'{kind}_checkpoint.pt', map_location='cpu', weights_only=False)
        checks.append(check_metadata(saved['metadata'], cfg, history, kind))
        if construction and construction['arm'] != 'identity':
            keys = sorted(k for k in saved['model_state_dict'] if k.endswith('.connections.indices'))
            actual = [fingerprint(saved['model_state_dict'][k].numpy()) for k in keys]
            expected = [r['output_sha256'] for r in construction['layers']]
            if actual != expected:
                raise RuntimeError('saved wiring differs from intervention receipt')
        del saved
    if construction:
        if construction['ur1_source_sha256'] != source_hash():
            raise RuntimeError('adapter source changed since execution')
        if construction['cost'] != read(directory / 'run_summary.json')['cost']:
            raise RuntimeError('initial/final declared cost mismatch')
        artifacts['ur1_construction.json'] = sha(directory / 'ur1_construction.json')
    return dict(artifacts=artifacts, metadata=checks)


def args_from(cfg):
    from experiments.train import get_parser
    args = get_parser().parse_args([])
    vars(args).update(cfg)
    return args


def graph_preflight(payload):
    from experiments.utils import get_model
    from experiments.train import model_cost_summary
    from torchlogix.layers import LogicDense, LogicConv2d
    from torchlogix.topology import image_input_semantics
    records = []
    for row in payload['entries']:
        if not row['reuse']: continue
        cfg = read(REPO / row['config'])
        torch.manual_seed(cfg['seed'])
        model = get_model(torch.tensor([.25, .5, .75], device='cuda'), args_from(cfg))
        layers = [m for m in model if isinstance(m, LogicDense)]
        originals = [m.connections.indices.clone() for m in layers]
        old = torch.load(REPO / row['output'] / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        keys = sorted(k for k in old['model_state_dict'] if k.endswith('.connections.indices'))
        if len(keys) != 4 or any(not torch.equal(a.cpu(), old['model_state_dict'][k]) for a,k in zip(originals,keys)):
            raise RuntimeError('SS does not reproduce archived U2 wiring')
        del old
        params, cost = tensor_digest(model.parameters()), model_cost_summary(model)
        rng = tensor_digest([torch.get_rng_state(), torch.cuda.get_rng_state()])
        arms = {}
        for arm in ['SS', 'SR', 'RS', 'RR']:
            for layer, original in zip(layers, originals):
                layer.connections.indices.copy_(original)
                layer.connections.strategy = 'semantic_multiscale_balanced'
            receipt = intervene(model, arm, cfg['topology_seed'])
            if params != tensor_digest(model.parameters()) or cost != model_cost_summary(model):
                raise RuntimeError('unpaired parameters/cost')
            if rng != tensor_digest([torch.get_rng_state(), torch.cuda.get_rng_state()]):
                raise RuntimeError('Torch RNG side effect')
            arms[arm] = receipt
        for depth in range(4):
            for left,right in ([('SS','SR'),('RS','RR')] if depth == 0 else [('SS','RS'),('SR','RR')]):
                if arms[left]['layers'][depth]['output_sha256'] != arms[right]['layers'][depth]['output_sha256']:
                    raise RuntimeError('factorial component drift')
        records.append(dict(seed=cfg['seed'], cost=cost, initial_parameters_sha256=params, arms=arms,
            archived_reference_indices_match=True, reuse=audit_run(row)))
        del model, layers, originals
        torch.cuda.empty_cache()
    nominal = []
    for coordinate in ['dense_m', 'conv_s']:
        for seed in range(3):
            directory = OLD / 'results' / f'fourth_mechanism_{coordinate}_nominal_seed{seed}'
            cfg = read(directory / 'training_config.json')
            torch.manual_seed(seed)
            model = get_model(torch.tensor([.25,.5,.75], device='cuda'), args_from(cfg))
            hashes = []
            for layer in model:
                if isinstance(layer, LogicConv2d):
                    c = layer.connections
                    semantics = image_input_semantics(3,1,1,3,layout='channel_interleaved') if c.layer_index == 0 else None
                    actual = nominal_indices(c.channels,c.num_kernels,layer_index=c.layer_index,topology_seed=seed,input_semantics=semantics)
                    target = c.channel_pairs.detach().cpu().numpy()
                elif isinstance(layer, LogicDense):
                    c = layer.connections
                    semantics = (image_input_semantics(3,32,32,3,layout='channel_interleaved') if coordinate == 'dense_m' and c.layer_index == 0 else
                        image_input_semantics(1024,2,2,1,layout='channel_interleaved') if coordinate == 'conv_s' and c.layer_index == 4 else None)
                    actual = nominal_indices(c.in_dim,c.out_dim,layer_index=c.layer_index,topology_seed=seed,input_semantics=semantics)
                    target = c.indices.detach().cpu().numpy()
                else: continue
                np.testing.assert_array_equal(actual,target)
                hashes.append(dict(layer_index=c.layer_index,shape=list(actual.shape),sha256=fingerprint(actual)))
            old = torch.load(directory / 'best_checkpoint.pt',map_location='cpu',weights_only=False)['model_state_dict']
            for key, tensor in model.state_dict().items():
                if '.connections.' in key and ('indices' in key):
                    if not torch.equal(tensor.cpu(),old[key]): raise RuntimeError('nominal model does not match saved wiring')
            nominal.append(dict(coordinate=coordinate,seed=seed,layers=hashes,archived_wiring_match=True))
            del model, old
            torch.cuda.empty_cache()
    return dict(status='pass',factorial=records,nominal=nominal)


def preflight():
    dataset = dataset_integrity()
    payload = prepare()
    freeze = ROOT / 'execution_freeze.json'
    if freeze.exists(): return verify_freeze()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.ones(1,device='cuda')
    torch.set_num_threads(1)
    log = ROOT / 'logs'
    # Each named attempt is exclusive. Failed outputs/logs stay for diagnosis.
    command([sys.executable,'-m','pytest','experiments/coverage_dlgn/unified_refinement','-q','-p','no:cacheprovider'],log/'unit_tests.log')
    graph = graph_preflight(payload)
    write_json(log/'graph_preflight.json',graph)
    smoke_rows = []
    for arm in ['SS','SR','RS','RR']:
        base = next(r for r in payload['entries'] if r['phase']=='factorial' and r['seed']==0 and r['arm']==arm)
        smoke_rows.append((base,arm))
    for family in ['random','u2']:
        base = next(r for r in payload['entries'] if r['phase']=='full_s' and r['seed']==1 and r['family']==family)
        smoke_rows.append((base,'identity'))
    smokes = {}
    for base,arm in smoke_rows:
        name = 'smoke_'+base['name']
        cfg = dict(read(REPO/base['config']),ur1_arm=arm,num_iterations=10,eval_freq=10,
            output=str((ROOT/'results'/name).relative_to(REPO)))
        path = ROOT/'configs/smokes'/f'{name}.json'
        write_json(path,cfg)
        row = dict(base,name=name,reuse=False,output=cfg['output'],config=str(path.relative_to(REPO)))
        print('GPU smoke',name,flush=True)
        command([sys.executable,'-m','experiments.coverage_dlgn.unified_refinement.train_adapter','--config',str(path)],log/f'{name}.log')
        smokes[name] = audit_run(row)
        if arm == 'SS':
            old_dir = OLD/'results/smoke_fourth_mechanism_dense_m_u2_seed0'
            for kind in ['best','final']:
                a = torch.load(old_dir/f'{kind}_checkpoint.pt',map_location='cpu',weights_only=False)['model_state_dict']
                b = torch.load(REPO/row['output']/f'{kind}_checkpoint.pt',map_location='cpu',weights_only=False)['model_state_dict']
                if a.keys() != b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
                    raise RuntimeError('SS 10-update replay differs from archived smoke; reuse decision required')
            smokes[name]['archived_10_update_state_bit_identity'] = True
        write_json(log/f'{name}.json',smokes[name])
    # Real training setup (including binarization/data-loader initialization)
    # must leave paired initial parameters, RNG states and cost identical.
    for prefix in ['smoke_ur1_factorial_dense_m_', 'smoke_ur1_full_conv_s_']:
        receipts=[read(ROOT/'results'/name/'ur1_construction.json') for name in smokes if name.startswith(prefix)]
        for receipt in receipts[1:]:
            for field in ['cost','initial_parameters_sha256','cpu_rng_sha256','cuda_rng_sha256']:
                if receipt[field] != receipts[0][field]: raise RuntimeError('unpaired smoke initialization: '+field)
    baseline_log = PRESERVATION/'baseline_full_tests.log'
    if '3455 passed, 3038 skipped' not in baseline_log.read_text():
        raise RuntimeError('missing original full regression pass')
    value = dict(status='pass',source_sha256=source_hash(),matrix_sha256=sha(ROOT/'matrix.json'),
        dataset_integrity=dataset,
        protocol_sha256=sha(ROOT/'PROTOCOL.md'),incumbent=verify_incumbent(),
        original_full_tests_sha256=sha(baseline_log),
        unit_tests_sha256=sha(log/'unit_tests.log'),graph_preflight_sha256=sha(log/'graph_preflight.json'),
        config_hashes={r['config']:sha(REPO/r['config']) for r in payload['entries']},
        reused={r['name']:audit_run(r) for r in payload['entries'] if r['reuse']},
        gpu_smokes=smokes)
    write_json(freeze,value)
    print('UR1 execution freeze passed',flush=True)
    return value


def verify_freeze():
    frozen = read(ROOT/'execution_freeze.json')
    verify_incumbent()
    if dataset_integrity() != frozen['dataset_integrity']: raise RuntimeError('dataset path/content changed after freeze')
    for path,key in [(ROOT/'matrix.json','matrix_sha256'),(ROOT/'PROTOCOL.md','protocol_sha256'),
                     (ROOT/'logs/unit_tests.log','unit_tests_sha256'),(ROOT/'logs/graph_preflight.json','graph_preflight_sha256'),
                     (PRESERVATION/'baseline_full_tests.log','original_full_tests_sha256')]:
        if sha(path) != frozen[key]: raise RuntimeError('frozen evidence changed: '+str(path))
    if source_hash() != frozen['source_sha256']: raise RuntimeError('UR1 source changed after freeze')
    for path,expected in frozen['config_hashes'].items():
        if sha(REPO/path) != expected: raise RuntimeError('frozen config changed')
    for row in read(ROOT/'matrix.json')['entries']:
        if row['reuse'] and audit_run(row) != frozen['reused'][row['name']]: raise RuntimeError('reused run changed')
    for name,record in frozen['gpu_smokes'].items():
        for path,expected in record['artifacts'].items():
            if sha(ROOT/'results'/name/path) != expected: raise RuntimeError('smoke changed')
    return frozen


if __name__ == '__main__':
    preflight()
