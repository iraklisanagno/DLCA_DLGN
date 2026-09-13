"""Additive UR1 reports; never mutate incumbent tables or frozen run artifacts."""
from __future__ import annotations
import argparse
import csv
import io
import math
import statistics
from .common import ROOT, OLD, REPO, read, sha, write_json


def paired(values):
    values=[float(x) for x in values]
    if len(values)!=3 or not all(math.isfinite(x) for x in values): raise ValueError('three finite paired seeds required')
    mean,sd=statistics.mean(values),statistics.stdev(values)
    half=4.302652729911275*sd/math.sqrt(3)
    return dict(raw_pp=values,n=3,mean_pp=mean,sample_sd_pp=sd,ci95_pp=[mean-half,mean+half],
        wins=sum(x>0 for x in values),ties=sum(x==0 for x in values),losses=sum(x<0 for x in values))


def save_text(path,content):
    if path.exists():
        if path.read_text()!=content: raise RuntimeError('refusing to replace report '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: stream.write(content)


def functional():
    report=read(ROOT/'summary/functional_support.json')
    families=['random','balanced_random','nominal','u2']
    lines=['# Hardened dense circuit diagnostics (OUR, V-associated; no inference)','','All 24 preregistered checkpoints analyzed. Local LUT relevance is exact;','dependency-pruned path support is only an upper bound on global dependence.','','| Family | Best-checkpoint validation mean (%) | Final-layer structural sources | Final-layer path sources | Final-layer binary gates (%) |','|---|---:|---:|---:|---:|']
    for family in families:
        rows=[r for r in report['rows'] if r['family']==family and r['kind']=='best']
        assert len(rows)==3
        mean=lambda fn:statistics.mean(fn(r) for r in rows)
        lines.append(f"| {family} | {mean(lambda r:r['validation_hard_pct']):.3f} | {mean(lambda r:r['layers'][-1]['structural_sources_mean']):.3f} | {mean(lambda r:r['layers'][-1]['dependency_path_sources_mean']):.3f} | {100*mean(lambda r:r['layers'][-1]['binary_fraction']):.2f} |")
    lines += ['','All layers, raw seeds, best/final differences, unused predecessors and reachability','are retained in functional_support.json and functional_support.csv. Three training','replicates per family: gates/checkpoints are not additional independent replicates.','These are descriptive associations, not a causal claim, a routing objective, or','physical pruning/area evidence. Constants can still contribute to classifier sums.','']
    save_text(ROOT/'summary/FUNCTIONAL_SUPPORT.md','\n'.join(lines))
    # SVG is a repo-native deterministic plot, no bitmap generation needed.
    colors=dict(random='#777777',balanced_random='#b77616',nominal='#3567b1',u2='#14835c')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="460" viewBox="0 0 960 460">','<rect width="960" height="460" fill="white"/>','<g font-family="sans-serif" font-size="12">']
    svg.append('<text x="30" y="25" font-size="16">Dependency-pruned raw-source path support: per-gate layer means (n=3)</text>')
    for panel,kind in enumerate(['best','final']):
        x0=70+panel*465;y0=370
        selected=[r for r in report['rows'] if r['kind']==kind]
        ymax=max(l['dependency_path_sources_mean'] for r in selected for l in r['layers'])*1.1
        svg.append(f'<text x="{x0}" y="55">{kind} checkpoint; upper bound, not exact global support</text>')
        for tick in range(5):
            value=ymax*tick/4;y=y0-280*tick/4
            svg.append(f'<path d="M{x0},{y:.2f} h340" stroke="#dddddd"/><text x="{x0-35}" y="{y+4:.2f}">{value:.1f}</text>')
        for depth in range(4):svg.append(f'<text x="{x0+depth*105}" y="395">L{depth+1}</text>')
        for family in families:
            rows=[r for r in selected if r['family']==family]
            for r in rows:
                pts=' '.join(f"{x0+d*105},{y0-280*l['dependency_path_sources_mean']/ymax:.2f}" for d,l in enumerate(r['layers']))
                svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[family]}" stroke-opacity="0.4"/>')
            pts=' '.join(f"{x0+d*105},{y0-280*statistics.mean(r['layers'][d]['dependency_path_sources_mean'] for r in rows)/ymax:.2f}" for d in range(4))
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[family]}" stroke-width="3"/>')
    for i,family in enumerate(families):svg.append(f'<text x="{70+i*215}" y="440" fill="{colors[family]}">{family}</text>')
    svg.append('</g></svg>')
    save_text(ROOT/'summary/functional_support.svg','\n'.join(svg)+'\n')


def training(phase):
    from .verify import audit_run,verify_freeze
    verify_freeze()
    entries=[r for r in read(ROOT/'matrix.json')['entries'] if r['phase']==phase]
    rows=[]
    for entry in entries:
        if not entry['reuse'] and not (ROOT/'logs'/f"{entry['name']}.complete.json").exists():
            raise RuntimeError('matrix incomplete; do not write a final report')
        checked=audit_run(entry)
        if not entry['reuse'] and checked!=read(ROOT/'logs'/f"{entry['name']}.complete.json"):
            raise RuntimeError('completion receipt differs')
        directory=REPO/entry['output'];summary=read(directory/'run_summary.json')
        rows.append(dict(entry,**checked,best_validation_hard_pct=100*summary['best_validation_hard_accuracy'],
            final_validation_hard_pct=100*summary['final_metrics']['val_acc_discrete'],
            final_validation_relaxed_pct=100*summary['final_metrics']['val_acc_relaxed'],
            cost=summary['cost'],wall_seconds=summary['wall_seconds'],peak_gpu_memory_bytes=summary['peak_gpu_memory_bytes']))
    report=dict(status='complete',phase=phase,provenance='OUR',scope='V',rows=rows,
        execution_freeze_sha256=sha(ROOT/'execution_freeze.json'),test_queries=0)
    if phase=='factorial':
        if len(rows)!=12:raise RuntimeError('incomplete factorial')
        contrasts={'first_given_structured_deeper':{'SS':1,'RS':-1},'first_given_randomized_deeper':{'SR':1,'RR':-1},
            'deeper_given_structured_first':{'SS':1,'SR':-1},'deeper_given_randomized_first':{'RS':1,'RR':-1},
            'interaction':{'SS':1,'SR':-1,'RS':-1,'RR':1}}
        by={(r['seed'],r['arm']):r for r in rows}
        report['contrasts']={endpoint:{name:paired([sum(coef*by[seed,arm][endpoint] for arm,coef in spec.items()) for seed in range(3)])
            for name,spec in contrasts.items()} for endpoint in ['best_validation_hard_pct','final_validation_hard_pct','final_validation_relaxed_pct']}
        report['limitations']=['Five exploratory unadjusted paired intervals; n=3','Exact per-node per-slot null is not native random or a uniform graph sampler','Conditional effects, not semantic ordering alone']
    else:
        report['limitations']=['Four new full-S validation runs only; old seed-0 test evidence is unchanged','No new test queries; select/freeze new checkpoints before any later test evaluation']
    write_json(ROOT/'summary'/f'{phase}_results.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['functional','factorial','full_s'])
    args=parser.parse_args()
    if args.action=='functional':functional()
    else:training(args.action)
