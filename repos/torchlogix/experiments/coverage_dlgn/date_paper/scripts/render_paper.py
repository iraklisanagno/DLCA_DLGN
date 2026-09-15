"""Render the paper's controlled Markdown dialect and evidence-driven tables."""
from pathlib import Path
import json
import re
import statistics
import math

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'output'
D = json.loads((PAPER/'evidence/paper_evidence.json').read_text())
(OUT/'tables').mkdir(exist_ok=True)

def prose(text):
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.S)
    text = re.sub(r'^#+ .*$', '', text, flags=re.M)
    text = re.sub(r'\{\{.*?\}\}|\[@.*?\]|\$.*?\$|\\ref\{.*?\}', '', text)
    return text.replace('**','').replace('- ','').strip()

def render(text, abstract=False):
    text = re.sub(r'\[@(.*?)\]', lambda m: r'\cite{'+','.join(x.strip() for x in m[1].split(';'))+'}', text)
    text = re.sub(r'\{\{(figure|algorithm|table):(.*?)\}\}',lambda m:r'\input{'+('tables' if m[1]=='table' else 'figures')+'/'+m[2]+'}',text)
    text = re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m[1]+'}',text)
    text = re.sub(r'^## (.*)$',lambda m:r'\subsection{'+m[1]+'}',text,flags=re.M)
    text = re.sub(r'^# (.*)$',lambda m:'' if abstract else r'\section{'+m[1]+'}',text,flags=re.M)
    text = re.sub(r'(?<!\\)%',r'\%',text)
    lines=[]; listing=False
    for line in text.splitlines():
        if line.startswith('- '):
            if not listing: lines.append(r'\begin{itemize}');listing=True
            lines.append(r'\item '+line[2:])
        elif listing and line.strip():
            lines.append(r'\end{itemize}');listing=False;lines.append(line)
        else: lines.append(line)
    if listing:lines.append(r'\end{itemize}')
    return '\n'.join(lines).strip()+'\n'

metrics={}
project=(PAPER/'project.yaml').read_text()
words_per_page=float(re.search(r'words_per_page: ([0-9.]+)',project)[1])
tolerance=float(re.search(r'section_tolerance: ([0-9.]+)',project)[1])
targets={name:float(weight)*words_per_page for name,weight in re.findall(r'^    (0[1-5]_\w+): ([0-9.]+)',project,re.M)}
editorial_ranges={'00_abstract':[170,200],'01_introduction':[450,550],'02_related_work':[350,450],'03_methodology':[850,1050],'04_experiments':[900,1150],'05_conclusion':[150,210]}
for i,p in enumerate(sorted((PAPER/'sections').glob('*.md'))):
    raw=p.read_text(); clean=prose(raw)
    cadence=prose(re.sub(r'^\*\*[^*]+\*\* ', '', raw, flags=re.M))
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',cadence.replace('et al.','et al').replace('Fig.','Fig'))
    lengths=[len(s.split()) for s in sentences if s.strip()]
    words=len(clean.split())
    target=180 if i==0 else round(targets[p.stem])
    metrics[p.stem]={'words':words,'target':target,'range':[170,200] if i==0 else [math.ceil(target*(1-tolerance)-1e-9),math.floor(target*(1+tolerance)+1e-9)],
                     'editorial_range':editorial_ranges[p.stem],
                     'budget_exception':'Figure-rich six-content-page limit; see paper_story.md' if i>0 else None,
                     'approx_mean_sentence_words_excluding_math':round(statistics.mean(lengths),1),
                     'sentences_over_35_words':[s for s,n in zip(sentences,lengths) if n>35],
                     'em_dashes':raw.count('\u2014'),'banned_terms':[x for x in ['delve','brittle','embark','intricate','multifaceted','admittedly','testament','seamless','synergistically'] if re.search(r'\b'+x+r'\b',raw,re.I)]}
    rendered=render(raw,i==0)
    assets=['figures/overview','tables/protocol','tables/dense','tables/conv','tables/routing','figures/dense_results','figures/routing_tradeoff','figures/structural_ablation']
    if p.stem in ['03_methodology','04_experiments']:
        for asset in assets: rendered=rendered.replace(r'\input{'+asset+'}', '')
    if p.stem=='03_methodology':
        rendered='\\input{figures/overview}\n'+rendered
        rendered+='\\input{tables/protocol}\n\\input{tables/dense}\n'
    if p.stem=='04_experiments':
        rendered=''.join('\\input{'+asset+'}\n' for asset in assets[3:])+rendered

    (OUT/(p.stem+'.tex')).write_text(rendered)
(OUT/'quality_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
title=re.search(r'^  title: "(.*)"$',(PAPER/'project.yaml').read_text(),re.M)[1]
(OUT/'title.tex').write_text('\\title{'+title+'}\n')

def row(cells): return ' & '.join(cells)+r' \\'+'\n'
def pm(s,bold=False):
    value=f"{s['mean']:.2f}"
    if bold:value=r'\text{\bfseries '+value+'}'
    return '$'+value+(r'\pm'+f"{s['sample_sd']:.2f}$" if s.get('sample_sd') is not None else '$')
def ci(v):return f'$[{v[0]:.2f},{v[1]:.2f}]$'
def table(name,caption,cols,header,rows,foot='',wide=True):
    env='table*' if wide else 'table'
    text='\\begin{'+env+('}[t]\n' if wide else '}[!htb]\n')+'\\centering\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n'
    text+=r'\tabfont'+'\n'+r'\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{tabular}{@{}'+cols+r'@{}}'+'\n\\toprule\n'+row(header)+'\\midrule\n'+''.join(rows)+'\\bottomrule\n\\end{tabular}\n'
    if foot:text+='\\par\\smallskip\n\\begin{minipage}{\\'+('textwidth' if wide else 'columnwidth')+'}\\normalsize '+foot+'\\end{minipage}\n'
    text+='\\end{'+env+'}\n'
    (OUT/'tables'/f'{name}.tex').write_text(text.replace('U2','LC').replace('V3','LC-D'))

rows=[row(['MNIST',r'$5\times1{,}334+1{,}330$','8K','1','108K','100']),
      row(['Fashion-MNIST',r'$5\times2{,}666+2{,}670$','16K','3','108K','100']),
      row(['CIFAR-10 S',r'$4\times12$K','48K','3','108K','100']),
      row(['CIFAR-10 M',r'$4\times128$K','512K','3','108K','100']),
      row(['CIFAR-10 L',r'$5\times256$K','1,280K','5','108K','100']),
      row(['Conv. S/M',r'$(k,4k,16k,32k)$',r'83,552 / 668,416','3','350K / 200K','128'])]
table('protocol',r'Main evaluation configurations. $b$: thresholds per raw value; K denotes 1,000. Dense widths enumerate gates per layer; convolutional widths enumerate output channels with $k=32/256$ for S/M.', 'llrcrr', ['Dataset/model','Layer widths','Gate functions','$b$','Updates','Batch'],rows,
      r'Dense: Adam, learning rate 0.01, no augmentation. Convolutional: AdamW, learning rate 0.02, decay 0.002, random crops with four-pixel padding and horizontal flips. All use raw gate logits and validation every 2K updates. Convolutional S/M share gate functions across 874,496/6,995,968 spatial gate applications; each classifier uses three dense layers of widths $(1280k,640k,320k)$.',True)
reported=[None,None,(51.27,0.26),(57.39,0.13),(60.78,0.12)]
rows=[]
for r,ref in zip(D['dense'],reported):
 e=r['effect'];rows.append(row([r['label'],f"{r['gates']/1000:,.0f}",pm(r['random']),pm(r['u2'],r['u2']['mean']>=r['v3']['mean']),pm(r['v3'],r['v3']['mean']>r['u2']['mean']),f"$+{e['mean']:.2f}$",ci(e['ci95']),r'\NA' if ref is None else f'${ref[0]:.2f}\\pm{ref[1]:.2f}$']))
table('dense',r'Dense held-out accuracy (T, \%). Local entries are mean $\pm$ sample SD over seeds 0, 1, 2. U2 gains are percentage points against the paired random baseline.', 'lrrrrrrr', ['Model','Gates, K','Random (REP)','LC (OUR)','LC-D (OUR)',r'$\Delta$ U2',r'Paired 95\% CI','Random (R)'],rows,
      r'LC wins 3/3 pairs at each coordinate. LC-D: separate dense specialization. Bold: largest local mean. R: published random baseline \cite{petersen2022}; no exact-budget MNIST/Fashion reference.')
rows=[]
for r in D['conv_test']:
 ref={'S':60.38,'M':71.01}[r['architecture']] if r['method']=='random' else None
 rows.append(row([r['architecture'], 'Random (REP)' if r['method']=='random' else 'U2 (OUR)','T / 1',f"{r['accuracy']:.2f}",f"{r['time_h']:.2f}",f"{r['gpu_gib']:.3f}",r'\NA' if ref is None else f'{ref:.2f}']))
rows.append(r'\midrule'+'\n')
for method in ['random','u2']:
 r=D['full_s'][method]
 rows.append(row(['S','Random (REP)' if method=='random' else 'U2 (OUR)','V / 3',pm(r),f"{r['time_h']:.2f}",f"{r['gpu_gib']:.3f}",r'\NA']))
table('conv','Convolutional CIFAR-10 accuracy and measured training cost. Single-seed test results and three-seed validation replications are separate evidence.', 'llcrrrr', ['Size','Method','Scope / seeds',r'Accuracy, \%','Time, h','GPU, GiB','Random T (R)'],rows,
      r'R: published S/M results \cite{petersen2024}. Full-S V: paired gain $+1.23$ pp, 95\% CI $[-3.36,5.82]$, 2/3 wins; extra seeds have no test results. Time: mean; memory: maximum peak allocation.')
rows=[]
for coord in ['m','l']:
 for method in ['random','u2','top32']:
  r=next(x for x in D['resources'] if x['coordinate']==coord and x['family']==method)
  ref=r['reported_reference']
  if method=='random' and ref is not None: ref=dict(ref,spread_pct={'m':0.14,'l':0.27}[coord])
  reftext=r'\NA' if ref is None else f"{ref['accuracy_pct']:.2f}"+(r'$\pm$'+f"{ref['spread_pct']:.2f}" if 'spread_pct' in ref else '')
  rows.append(row([coord.upper(),{'random':'Random (REP)','u2':'U2 (OUR)','top32':'Top-32 (REP)'}[method],str(len(r['seeds'])),pm(r['best_test_hard_pct']),f"{r['trainable_parameters']/1e6:.3f}",f"{r['training_wall_minutes']['mean']:.2f}",f"{r['training_peak_gpu_gib']['mean']:.3f}",reftext]))
 if coord=='m':rows.append(r'\midrule'+'\n')
table('routing',r'CIFAR-10 routing trade-off under LILogic architectures: M is $1\times64$K gates; L is $2\times128$K gates. Both use seven input thresholds. Local accuracy is T; R retains the published statistic.', 'llcrrrrr', ['Size','Method','$n$',r'Accuracy, \%','Params., M','Time, min','GPU, GiB','Accuracy (R)'],rows,
      r'Local: 35K updates, batch 256, Adam at 0.075, reflected-padding crops and horizontal flips. R: LILogic v2 \cite{fojcik2026}. Top-32 adds 4.096M/16.384M routing parameters; its single-seed resource comparison is descriptive.')
rows=[]
for arm in ['SS','SR','RS','RR']:
 r=D['factorial_arms'][arm]
 rows.append(row([arm,', '.join(f'{v:.2f}' for v in r['raw']),pm(r)]))
table('factorial','Dense M factorial ablation (20K updates, V). Arm letters indicate first/deeper layers; S is U2 structure and R is a degree-preserving randomization.', 'llr', ['Arm','Seeds 0, 1, 2',r'Mean $\pm$ SD'],rows,
      r'All arms use 1.123 GiB peak allocation. Figure \ref{fig:effects} shows all five paired contrasts. The first-layer effects and SS$-$SR win 3/3 seed pairs; RS$-$RR wins 1/3 and the interaction is positive for 2/3.',False)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='sentences_over_35_words'} for k,v in metrics.items()},indent=2))
