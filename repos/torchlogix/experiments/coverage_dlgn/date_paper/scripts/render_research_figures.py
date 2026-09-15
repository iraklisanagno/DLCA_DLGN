"""Generate vector research plots from frozen evidence; never run experiments."""
from pathlib import Path
import hashlib
import json
import math
import statistics as st

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'output' / 'figures'
D = json.loads((PAPER / 'evidence/paper_evidence.json').read_text())
REGISTRY = {}
COLORS = {'random': 'controlgray', 'u2': 'methodblue', 'v3': 'accentorange', 'top32': 'accentorange'}
NAMES = {'random': 'Random', 'u2': 'LC', 'v3': 'LC-D', 'top32': 'Top-32'}

def axis(options, body, second=False):
    common = r'width=89mm,height=50mm,axis lines=left,tick align=outside,grid=major,grid style={gray!15},scaled ticks=false,tick label style={font=\fontsize{11}{13}\selectfont},label style={font=\fontsize{11}{13}\selectfont},title style={font=\fontsize{11}{13}\selectfont},legend style={font=\fontsize{11}{13}\selectfont,draw=none,fill=white,inner sep=1pt}'
    position = r',at={(9.25cm,0)},anchor=south west' if second else ''
    return '\\begin{axis}[' + common + position + ',' + options + ']\n' + body + '\\end{axis}\n'

def points(coords, color='methodblue', mark='*', connect=False, error=None, legend=None):
    opts = f'color={color},mark={mark},mark size=2pt,line width=0.9pt,mark options={{solid,fill={color},draw={color}}}'
    if not connect: opts += ',only marks'
    if error: opts += ',error bars/.cd,' + error + ' dir=both,' + error + ' explicit'
    text = '\\addplot+[' + opts + '] coordinates {'
    for row in coords:
        x,y = row[:2]
        text += f'({x:.8g},{y:.8g})'
        if error: text += f' +- ({row[2] if error=="x" else 0:.8g},{row[2] if error=="y" else 0:.8g})'
    text += '};\n'
    if legend: text += '\\addlegendentry{' + legend + '}\n'
    return text

def forest(rows, labels, title, xmin, xmax, second=False):
    # Show complete uncertainty intervals, including negative control outcomes.
    low=min(r[1] if r[1] is not None else r[0] for r in rows)
    high=max(r[2] if r[2] is not None else r[0] for r in rows)
    if low<xmin:xmin=math.floor(low-.25)
    if high>xmax:xmax=math.ceil(high+.25)
    body = f'\\draw[dashed,gray] (axis cs:0,-.5)--(axis cs:0,{len(rows)-.5});\n'
    for i,r in enumerate(rows):
        mean,lo,hi = r[:3]
        body += points([(mean,len(rows)-1-i,(hi-lo)/2)] if lo is not None else [(mean,len(rows)-1-i)],
                       color=r[3] if len(r)>3 else 'methodblue',error='x' if lo is not None else None)
    ticklabels=','.join('{'+x+'}' for x in reversed(labels))
    opts=f'title={{{title}}},xmin={xmin},xmax={xmax},ymin=-.5,ymax={len(rows)-.5},ytick={{{",".join(str(i) for i in range(len(rows)))}}},yticklabels={{{ticklabels}}},xlabel={{Accuracy difference (pp)}}'
    return axis(opts,body,second)

def write(name, panels, caption, source, wide=True):
    pre = r'''\documentclass[10pt]{article}
\usepackage[paperwidth=198mm,paperheight=59mm,margin=1mm]{geometry}
\usepackage{newtxtext,newtxmath,tikz,pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{methodblue}{HTML}{216A85}
\definecolor{controlgray}{HTML}{555555}
\definecolor{accentorange}{HTML}{A64B1D}
\pagestyle{empty}
\begin{document}\noindent\begin{tikzpicture}
'''
    (OUT/(name+'_plot.tex')).write_text(pre+panels+'\\end{tikzpicture}\\end{document}\n')
    env='figure*' if wide else 'figure'
    wrapper='\\begin{'+env+'}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/'+name+'_plot.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+name+'}\n\\end{'+env+'}\n'
    (OUT/(name+'.tex')).write_text(wrapper)
    REGISTRY[name]={'caption_tex':caption,'source':source,'standalone_pdf':'output/figures/'+name+'_plot.pdf'}

# Main dense comparison and paired effects.
cifar=D['dense'][2:]
body=''
for method in ['random','u2','v3']:
    body+=points([(r['gates']/1000,r[method]['mean'],r[method]['sample_sd']) for r in cifar],COLORS[method],{'random':'square*','u2':'*','v3':'triangle*'}[method],True,'y',NAMES[method])
a=axis(r'title={(a) Dense CIFAR-10 (T)},xmode=log,log basis x=10,xmin=35,xmax=1650,ymin=47,ymax=64,xtick={48,512,1280},xticklabels={48,512,1280},xlabel={Gate count (thousands)},ylabel={Accuracy (\%)},legend pos=south east',body)
b=forest([(r['effect']['mean'],*r['effect']['ci95']) for r in D['dense']],['MNIST','Fashion','C10 S','C10 M','C10 L'],'(b) LC minus random (T)',-2,6.4,True)
write('dense_results',a+b,r'Dense accuracy at fixed budgets. (a) Mean $\pm$ sample SD over three seeds. LC-D is the separate dense specialization. (b) Paired 95\% confidence intervals for LC minus random; all three CIFAR-10 intervals exclude zero.', ['dense'])

# Two resource coordinates, no line interpolating disparate architectures.
panels=''
for idx,(metric,xlabel,xlim) in enumerate([('training_peak_gpu_gib','Peak GPU allocation (GiB)',(0.3,40)),('training_wall_minutes','Observed training time (min)',(10,600))]):
    body=''
    for method in ['random','u2','top32']:
        rows=sorted([r for r in D['resources'] if r['family']==method],key=lambda r:r['coordinate'],reverse=True)
        body+=points([(r[metric]['mean'],r['best_test_hard_pct']['mean'],r['best_test_hard_pct']['sample_sd'] or 0) for r in rows],COLORS[method],{'random':'square*','u2':'*','top32':'triangle*'}[method],False,'y',NAMES[method])
        for r in rows:
            anchor='south east' if method=='random' else 'south west'
            body+=f"\\node[font=\\fontsize{{11}}{{13}}\\selectfont,anchor={anchor}] at (axis cs:{r[metric]['mean']},{r['best_test_hard_pct']['mean']+.25}) {{{r['coordinate'].upper()}}};\n"
    panels+=axis(f'title={{({"ab"[idx]}) LILogic architectures (T)}},xmode=log,xmin={xlim[0]},xmax={xlim[1]},ymin=47,ymax=65,xlabel={{{xlabel}}},ylabel={{Accuracy (\\%)}},legend pos=south east',body,idx==1)
write('routing_tradeoff',panels,r'Accuracy versus (a) peak GPU allocation and (b) observed training time under the LILogic M/L configurations. LC and random: three seeds, mean $\pm$ sample SD. Top-32: one seed. M/L identify architectures; wall-time ratios are descriptive execution measurements.', ['resources'])

body=''
for i,arm in enumerate(['SS','SR','RS','RR']):
    r=D['factorial_arms'][arm]
    body+=points([(i,r['mean'],r['sample_sd'])],color='methodblue' if arm=='SS' else 'controlgray',error='y')
    body+=points([(i+(j-1)*.1,v) for j,v in enumerate(r['raw'])],color='accentorange',mark='x')
a=axis(r'title={(a) Dense M factorial (V)},xmin=-.5,xmax=3.5,ymin=53.5,ymax=60.5,xtick={0,1,2,3},xticklabels={SS,SR,RS,RR},xlabel={First / deeper structure},ylabel={Accuracy (\%)}',body)
keys=['first_given_structured_deeper','first_given_randomized_deeper','deeper_given_structured_first','deeper_given_randomized_first','interaction']
rows=[]
for i,k in enumerate(keys):
    r=D['factorial_contrasts']['best_validation_hard_pct'][k];rows.append((r['mean_pp'],*r['ci95_pp'],'methodblue' if i<2 else 'controlgray'))
b=forest(rows,['SS$-$RS','SR$-$RR','SS$-$SR','RS$-$RR','Interaction'],'(b) Paired effects (V)',-3.5,6.4,True)
write('structural_ablation',a+b,r'Dense M attribution at 20K updates. S retains LC structure; R randomizes pairs while preserving each predecessor degree in each input slot. (a) Three seed values (crosses) and mean $\pm$ sample SD. (b) All five paired 95\% intervals, exploratory and unadjusted. The first two contrasts isolate first-layer structure.', ['factorial_arms','factorial_contrasts'])

body=''
for method in ['random','u2']:
    r=D['full_s'][method]
    body+=points(list(enumerate(r['raw'])),COLORS[method],connect=True,legend=NAMES[method])
a=axis(r'title={(a) Full-S validation replication},xmin=-.15,xmax=2.15,ymin=57.5,ymax=63,xtick={0,1,2},xlabel={Training seed},ylabel={Accuracy (\%)},legend pos=south east',body)
rows=[]
for size in ['S','M']:
    vals={r['method']:r['accuracy'] for r in D['conv_test'] if r['architecture']==size};rows.append((vals['u2']-vals['random'],None,None))
r=D['full_s']['effect'];rows.append((r['mean'],*r['ci95'],'accentorange'))
b=forest(rows,['S test, n=1','M test, n=1','S val., n=3'],'(b) LC minus random',-4.5,7,True)
write('conv_replication',a+b,r'Convolutional replication and measurement scope. Single-seed tests have no error bars. Full-S validation uses three paired seeds with a 95\% interval; its third seed reverses the sign of the gain. Validation and test observations are not pooled.', ['conv_test','full_s'])

# Constructor resources for all five principal dense configurations.
resource_rows=[]
for r in D['dense']:
    if 'construction_s' in r['u2']: q=r['u2']
    else:
        g=next(g for g in D['support']['third_round_all_groups']['groups'] if g['phase']=='third_u2_dense_cifar10_ml' and g['coordinate']==r['coordinate'][-1])
        q={'construction_s':g['topology_construction_seconds']['mean'],'time_min':g['training_wall_minutes']['mean'],'gpu_gib':g['training_peak_gpu_gib']['mean']}
    resource_rows.append({'label':r['label'],**q})
body=points([(i,r['construction_s']) for i,r in enumerate(resource_rows)])
a=axis(r'title={(a) LC CPU construction},xmin=-.5,xmax=4.5,ymode=log,xtick={0,1,2,3,4},xticklabels={MN,F-M,S,M,L},xlabel={Dense configuration},ylabel={Construction time (s)}',body)
body=points([(i,100*r['construction_s']/(60*r['time_min'])) for i,r in enumerate(resource_rows)])
b=axis(r'title={(b) Construction / training time},xmin=-.5,xmax=4.5,ymin=0,ymax=.7,xtick={0,1,2,3,4},xticklabels={MN,F-M,S,M,L},xlabel={Dense configuration},ylabel={Observed duration ratio (\%)}',body,True)
write('construction_cost',a+b,r'Offline LC construction for MNIST (MN), Fashion-MNIST (F-M), and CIFAR-10 S/M/L. Ratios use recorded training duration, including evaluation; they do not measure a controlled end-to-end speedup.', ['dense','support.third_round_all_groups.groups'])

panels=''
for i,k in enumerate(['fourth_round_dense_mechanism','fourth_round_conv_mechanism']):
    rows=[]
    for e in D['support'][k]['paired_effects']:
        r=e['paired_effect'];rows.append((r['mean'],r['ci95_low'],r['ci95_high']))
    panels+=forest(rows,['Random','Balanced','Nominal'],f'({"ab"[i]}) '+('Dense M' if i==0 else 'Conv. S')+' (V)',-3.2,6.7,i==1)
write('mechanism_controls',panels,r'LC minus each stage-selection control at 20K updates, three paired seeds and exploratory 95\% intervals. Neither coordinate establishes an incremental benefit over the nominal schedule. The balanced control is distinct from native random wiring.', ['support.fourth_round_dense_mechanism','support.fourth_round_conv_mechanism'])

rows=[]
for e in D['support']['fourth_round_factorial']['paired_effects']:
    r=e['paired_effect'];rows.append((r['mean'],r['ci95_low'],r['ci95_high']))
a=forest(rows,['Body / R head','Body / LC head','Head / R body','Head / LC body','Interaction'],'(a) Conv. factorial effects (V)',-5.5,7)
body=''
for i,family in enumerate(['both','body','head','random']):
    g=next(g for g in D['support']['fourth_round_factorial']['groups'] if g['family']==family);r=g['metrics']['best_validation_hard_pct']
    body+=points([(i,r['mean'],r['sample_sd'])],error='y')
b=axis(r'title={(b) Conv. S factorial (V)},xmin=-.5,xmax=3.5,ymin=54,ymax=61,xtick={0,1,2,3},xticklabels={Both,Body,Head,Random},xlabel={Structured components},ylabel={Accuracy (\%)}',body,True)
write('conv_factorial',a+b,r'Convolutional body/classifier attribution at 20K updates, three seeds. The structured classifier retains reference-body ancestry, so body effects are conditional on that classifier. (a) All five exploratory paired 95\% intervals. (b) Mean $\pm$ sample SD.', ['support.fourth_round_factorial'])

rows=[]
for k in ['fourth_round_warp_dense','fourth_round_warp_conv']:
    r=D['support'][k]['paired_effects'][0]['paired_effect'];rows.append((r['mean'],r['ci95_low'],r['ci95_high']))
a=forest(rows,['Dense M','Conv. S'],'(a) Adapted WARP: LC minus random',-2,6)
body=''
for i,k in enumerate(['fourth_round_warp_dense','fourth_round_warp_conv']):
    for j,family in enumerate(['random','u2']):
        g=next(g for g in D['support'][k]['groups'] if g['family']==family);r=g['metrics']['best_validation_hard_pct']
        body+=points([(i+(j-.5)*.2,r['mean'],r['sample_sd'])],COLORS[family],error='y',legend=NAMES[family] if i==0 else None)
b=axis(r'title={(b) Adapted WARP accuracy (V)},xmin=-.5,xmax=1.5,ymin=40,ymax=61,xtick={0,1},xticklabels={Dense M,Conv. S},ylabel={Accuracy (\%)},legend pos=north east',body,True)
write('warp_compatibility',a+b,r'Compatibility with adapted WARP training. Dense M uses 108K updates; convolutional S uses 20K. Three seeds; paired 95\% intervals in (a), mean $\pm$ sample SD in (b). Both adapted recipes trail matched raw training; these data do not reproduce WARP\textquotesingle s published headline results.', ['support.fourth_round_warp_dense','support.fourth_round_warp_conv'])

rows=[]
for e in D['support']['fourth_round_transfer_results']['comparisons']:
    r=e['paired_effect'];rows.append((r['mean'],r['ci95_low'],r['ci95_high']))
a=forest(rows,['Dense M, n=3','Conv. S, n=1','Conv. M, n=1'],'(a) Frozen CIFAR-10.1 tests',-1,5.5)
body=''
for i,e in enumerate(D['support']['fourth_round_transfer_results']['comparisons']):
    body+=points([(i+(j-(len(e['per_seed'])-1)/2)*.12,s['gain_pp']) for j,s in enumerate(e['per_seed'])],mark='x')
b=axis(r'title={(b) Individual seed gains (T)},xmin=-.5,xmax=2.5,ymin=-1,ymax=5,xtick={0,1,2},xticklabels={Dense M,Conv. S,Conv. M},ylabel={LC minus random (pp)}',body,True)
write('transfer',a+b,r'Frozen CIFAR-10.1 evaluation without adaptation or repeat inference. Dense M has a three-seed paired 95\% interval; convolutional points each use one seed. (b) All available paired seed differences.', ['support.fourth_round_transfer_results'])

body=''
for seed in range(3):
    body+=points([(i+(seed-1)*.12,r['effect']['raw'][seed]) for i,r in enumerate(D['dense'])],['methodblue','accentorange','controlgray'][seed],['*','triangle*','square*'][seed],legend='Seed '+str(seed))
a=axis(r'title={(a) Dense paired test gains},xmin=-.5,xmax=4.5,ymin=0,ymax=6,xtick={0,1,2,3,4},xticklabels={MN,F-M,S,M,L},xlabel={Dense configuration},ylabel={LC minus random (pp)},legend pos=north west',body)
body=''
for method in ['random','u2','v3']:
    body+=points([(i,r[method]['sample_sd']) for i,r in enumerate(D['dense'])],COLORS[method],connect=True,legend=NAMES[method])
b=axis(r'title={(b) Between-seed variation (T)},xmin=-.5,xmax=4.5,ymin=0,ymax=.85,xtick={0,1,2,3,4},xticklabels={MN,F-M,S,M,L},xlabel={Dense configuration},ylabel={Sample SD (pp)},legend pos=north west',body,True)
write('dense_seed_gains',a+b,r'All 15 paired dense LC test differences and the sample standard deviations of all three fixed constructions. MN/F-M denote MNIST/Fashion-MNIST; S/M/L denote CIFAR-10 sizes. Positive seed differences do not guarantee that every three-seed confidence interval excludes zero.', ['dense'])

(PAPER/'evidence/figure_registry.json').write_text(json.dumps({'evidence_sha256':hashlib.sha256((PAPER/'evidence/paper_evidence.json').read_bytes()).hexdigest(),'figures':REGISTRY,'dense_construction_rows':resource_rows},indent=2)+'\n')
print('Generated',len(REGISTRY),'two-panel vector figure sources from frozen measurements.')
