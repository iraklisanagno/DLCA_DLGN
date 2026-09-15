"""Read existing experiment evidence; write only inside this paper directory."""
from pathlib import Path
import hashlib
import json
import math
import statistics as st

PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent
TORCHLOGIX = EXP.parents[1]
DEST = PAPER / 'evidence'
DEST.mkdir(exist_ok=True)
SOURCES = {}


def read(relative):
    path = EXP / relative
    raw = path.read_bytes()
    SOURCES[relative] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)


def stat(values):
    return {'raw': values, 'n': len(values), 'mean': st.mean(values),
            'sample_sd': st.stdev(values) if len(values) > 1 else None}


def paired(a, b):
    values = [x-y for x, y in zip(a, b, strict=True)]
    result = stat(values)
    assert len(values) == 3
    half = 4.302652729911275 * result['sample_sd'] / math.sqrt(3)
    result.update(ci95=[result['mean']-half, result['mean']+half],
                  wins=sum(v > 0 for v in values))
    return result


second = read('summary/second_round_final_dense.json')
third = read('summary/third_round_results.json')
dense = []
for coord, label, gates in [('mnist_8k', 'MNIST', 8000),
                            ('fashion_16k', 'Fashion-MNIST', 16000),
                            ('dense_cifar10_s', 'CIFAR-10 S', 48000)]:
    cell = second['cells'][coord]
    row = {'coordinate': coord, 'label': label, 'gates': gates, 'scope': 'T', 'seeds': [0, 1, 2]}
    for method in ['random', 'v3', 'u2']:
        row[method] = stat(cell[method]['test_hard_pct'])
        row[method]['time_min'] = cell[method]['mean_training_wall_minutes']
        row[method]['gpu_gib'] = cell[method]['maximum_peak_gpu_gib']
        row[method]['construction_s'] = cell[method]['mean_topology_construction_seconds']
    row['effect'] = paired(row['u2']['raw'], row['random']['raw'])
    dense.append(row)
for cell in third['current_dense_cross_comparisons']:
    coord = cell['coordinate']
    row = {'coordinate': 'dense_cifar10_'+coord, 'label': 'CIFAR-10 '+coord.upper(),
           'gates': 512000 if coord == 'm' else 1280000, 'scope': 'T', 'seeds': [0, 1, 2]}
    for method, values in cell['best_test_hard_pct'].items():
        row[method] = stat(values['per_seed'])
    row['effect'] = paired(row['u2']['raw'], row['random']['raw'])
    dense.append(row)

full_s_old = read('summary/second_round_convolutional_final.json')['rows']
full_s_new = read('unified_refinement/summary/full_s_results.json')
factorial = read('unified_refinement/summary/factorial_results.json')
for phase in [full_s_new, factorial]:
    assert phase['status'] == 'complete' and phase['test_queries'] == 0
    for row in phase['rows']:
        folder = TORCHLOGIX / row['output']
        for name in ['metrics.csv', 'run_summary.json', 'training_config.json']:
            raw = (folder/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row['artifacts'][name], (folder, name)
        assert all((folder/name).is_file() for name in row['artifacts'])
        if not row['reuse']:
            receipt = read('unified_refinement/logs/'+row['name']+'.complete.json')
            assert receipt['artifacts'] == row['artifacts']

full_s = {'scope': 'V', 'seeds': [0, 1, 2], 'new_test_queries': 0}
for method, key in [('random', 'random'), ('u2', 'unified_u2')]:
    old = full_s_old[key]
    new = sorted([r for r in full_s_new['rows'] if r['family'] == method], key=lambda r:r['seed'])
    full_s[method] = stat([old['best_hard_validation_pct']]+[r['best_validation_hard_pct'] for r in new])
    full_s[method]['final_hard'] = stat([old['final_hard_validation_pct']]+[r['final_validation_hard_pct'] for r in new])
    full_s[method]['time_h'] = st.mean([old['training_wall_hours']]+[r['wall_seconds']/3600 for r in new])
    full_s[method]['gpu_gib'] = max([old['peak_gpu_gib']]+[r['peak_gpu_memory_bytes']/2**30 for r in new])
full_s['effect'] = paired(full_s['u2']['raw'], full_s['random']['raw'])

conv_test = []
for method, key in [('random', 'random'), ('u2', 'unified_u2')]:
    old = full_s_old[key]
    conv_test.append({'architecture':'S', 'method':method, 'seed':0, 'scope':'T',
                      'accuracy':old['test_hard_pct'], 'time_h':old['training_wall_hours'],
                      'gpu_gib':old['peak_gpu_gib'], 'gates':83552})
medium_old = read('summary/cifar10_paper_medium_200k_paired.json')
medium_new = read('logs/cifar10_paper_medium_u2_200k/test/test_evaluation_summary.json')
for method in ['random','u2']:
    name = 'full_conv_cifar10_paper_medium_'+method+'_seed0_200k'
    run = read('results/'+name+'/run_summary.json')
    accuracy = medium_old['heldout_test']['fixed_random']['test_hard_accuracy'] if method == 'random' else medium_new['test_hard_accuracy']
    conv_test.append({'architecture':'M', 'method':method, 'seed':0, 'scope':'T',
                      'accuracy':100*accuracy, 'time_h':run['wall_seconds']/3600,
                      'gpu_gib':run['peak_gpu_memory_bytes']/2**30, 'gates':668416})

arms = {}
for arm in ['SS','SR','RS','RR']:
    rows = sorted([r for r in factorial['rows'] if r['arm']==arm],key=lambda r:r['seed'])
    arms[arm] = stat([r['best_validation_hard_pct'] for r in rows])
    arms[arm]['time_min'] = st.mean([r['wall_seconds']/60 for r in rows])
    arms[arm]['gpu_gib'] = max(r['peak_gpu_memory_bytes']/2**30 for r in rows)

resources = []
for group in third['groups']:
    if group['phase'] != 'third_lilogic_cifar10': continue
    resources.append({key: group[key] for key in ['coordinate','family','seeds','trainable_parameters',
                      'training_routing_parameters','best_test_hard_pct','training_wall_minutes',
                      'training_peak_gpu_gib','reported_reference']})

support = {}
for name in ['fourth_round_factorial','fourth_round_dense_mechanism','fourth_round_conv_mechanism',
             'fourth_round_warp_dense','fourth_round_warp_conv','fourth_round_transfer_results',
             'second_round_u2_pilot']:
    support[name] = read('summary/'+name+'.json')

freeze = read('summary/second_round_final_validation_freeze.json')
configs = {}
for name, rows in freeze['groups'].items():
    if name in ['mnist_8k_u2','fashion_16k_u2','dense_cifar10_s_u2']:
        configs[name] = read('results/'+rows[0]['name']+'/training_config.json')
for name in ['third_u2_cifar10_m_seed0','third_u2_cifar10_l_seed0',
             'full_conv_cifar10_paper_medium_u2_seed0_200k']:
    configs[name] = read('results/'+name+'/training_config.json')
configs['full_s'] = read('unified_refinement/configs/ur1_full_conv_s_u2_seed1.json')

# Supplementary resource rows and platform provenance, without additional runs.
support['third_round_all_groups'] = third
platforms = {}
for name in ['second_final_u2_mnist_8k_seed0','third_u2_cifar10_m_seed0',
             'third_lilogic_m_top32_seed0','full_conv_cifar10_paper_medium_u2_seed0_200k']:
    env = read('results/'+name+'/environment.json')
    platforms[name] = {k:env[k] for k in ['gpu_names','python','torch','platform'] if k in env}
for name in ['third_lilogic_m_top32_seed0','third_lilogic_l_top32_seed0',
             'third_lilogic_m_random_seed0','third_lilogic_l_u2_seed0']:
    configs[name] = read('results/'+name+'/training_config.json')
support['archived_platforms'] = platforms
for relative in ['../../src/torchlogix/topology.py','../../src/torchlogix/models/dense.py',
                 '../../src/torchlogix/models/conv.py','../utils.py']:
    raw = (EXP/relative).read_bytes()
    SOURCES[relative] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

result = dict(snapshot_date='2026-09-11', dense=dense, conv_test=conv_test, full_s=full_s,
              factorial_arms=arms, factorial_contrasts=factorial['contrasts'],
              resources=resources, support=support, configs=configs,
              provenance={'random':'REPRODUCED','u2':'OUR','v3':'OUR','top32':'REPRODUCED'},
              sources=SOURCES,
              audit_scope='UR1 receipts, artifact existence, metrics/config/summary hashes checked; '
                          'historical summaries read and source-hashed, not all historical checkpoint binaries rehashed.')
(DEST/'paper_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Supporting raw results', '', 'All values are percentages. Seeds are ordered 0, 1, 2. SD is the sample standard deviation.', '',
       '## Dense held-out tests', '', '| Model | Random | V3 | U2 | U2 gain, pp | Paired 95% CI | Wins |', '|---|---|---|---|---:|---|---|']
fmt=lambda v:', '.join(f'{x:.3f}' for x in v)
for row in dense:
    e=row['effect'];lines.append(f"| {row['label']} | {fmt(row['random']['raw'])} | {fmt(row['v3']['raw'])} | {fmt(row['u2']['raw'])} | {e['mean']:.3f} | {fmt(e['ci95'])} | {e['wins']}/3 |")
lines += ['', '## Full-S validation', '', '| Method | Best hard seeds | Final hard seeds |', '|---|---|---|']
for method in ['random','u2']:lines.append(f"| {method} | {fmt(full_s[method]['raw'])} | {fmt(full_s[method]['final_hard']['raw'])} |")
lines += ['', '## Dense factorial validation', '', '| Arm | Best hard seeds | Mean | SD |', '|---|---|---:|---:|']
for arm,row in arms.items():lines.append(f"| {arm} | {fmt(row['raw'])} | {row['mean']:.3f} | {row['sample_sd']:.3f} |")
lines += ['', '| Contrast | Paired seed differences, pp | Mean | Paired 95% CI | Wins |', '|---|---|---:|---|---|']
for name,e in factorial['contrasts']['best_validation_hard_pct'].items():lines.append(f"| {name} | {fmt(e['raw_pp'])} | {e['mean_pp']:.3f} | {fmt(e['ci95_pp'])} | {e['wins']}/3 |")
lines += ['', 'All five factorial intervals are exploratory and unadjusted. Randomized factorial arms preserve exact per-node, per-slot degrees and are not native-random baselines.',
          '', 'Complete resource rows, supporting studies, saved training configurations, and source hashes are in [paper_evidence.json](paper_evidence.json). The source registry uses paths relative to the parent coverage_dlgn directory. No inference or training was performed to assemble these tables.']
(DEST/'raw_results.md').write_text('\n'.join(lines)+'\n')
print('Collected',len(SOURCES),'source records;',len(dense),'dense coordinates; 16 audited UR1 logical rows.')
