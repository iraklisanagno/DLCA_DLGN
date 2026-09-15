"""Create the detailed evidence atlas from the same frozen paper snapshot."""
from pathlib import Path
import json
import math
import statistics as st

PAPER=Path(__file__).resolve().parents[1]
D=json.loads((PAPER/'evidence/paper_evidence.json').read_text())
F=json.loads((PAPER/'evidence/figure_registry.json').read_text())
OUT=PAPER/'output'
NAME={'random':'Random','u2':'LogicConnect','v3':'LogicConnect-D','top32':'Top-32','best':'Learned rank-4'}
sections=[]

def num(x,n=3): return '—' if x is None else f'{x:.{n}f}'
def pm(r): return num(r['mean'])+(' ± '+num(r['sample_sd']) if r.get('sample_sd') is not None else '')
def add(title,columns,rows,note): sections.append(dict(title=title,columns=columns,rows=[[str(x) for x in row] for row in rows],note=note))

rows=[]
for d in D['dense']:
    for method in ['random','u2','v3']:
        r=d[method];rows.append([d['label'],d['gates'],NAME[method],pm(r),', '.join(num(x) for x in r['raw']),'REP' if method=='random' else 'OUR'])
add('Dense held-out accuracy: all methods and seeds',['Model','Gates','Method','Mean ± SD (%)','Seeds 0, 1, 2 (%)','Source'],rows,'T throughout; three seeds. LogicConnect-D is the distinct dense specialization. A best-of-variants test maximum is not an evaluated method-selection policy.')

rows=[]
for d in D['dense']:
    for method in ['u2','v3']:
        vals=[a-b for a,b in zip(d[method]['raw'],d['random']['raw'],strict=True)];mean=st.mean(vals);half=4.302652729911275*st.stdev(vals)/math.sqrt(3)
        rows.append([d['label'],NAME[method],num(mean),f'[{mean-half:.3f}, {mean+half:.3f}]',sum(x>0 for x in vals),', '.join(num(x) for x in vals)])
add('Dense paired effects against native random',['Model','Method','Gain (pp)','Paired 95% CI','Wins / 3','Raw gains (pp)'],rows,'T, paired over the complete three-seed set. Intervals are not adjusted for multiple comparisons. Both variants are retained.')

rows=[]
for r in D['conv_test']:rows.append([r['architecture'],NAME[r['method']],'T',1,num(r['accuracy']),num(r['time_h']),num(r['gpu_gib']),r['gates']])
for method in ['random','u2']:
    r=D['full_s'][method];rows.append(['S',NAME[method],'V',3,pm(r),num(r['time_h']),num(r['gpu_gib']),83552])
add('Convolutional accuracy and training cost',['Size','Method','Scope','n','Accuracy (%)','Time (h)','GPU (GiB)','Gate functions'],rows,'S/M are nine-channel, paper-faithful architectures. V and T are separate observations. Recorded wall times include evaluation and depend on execution conditions. GPU is peak PyTorch allocation.')

rows=[]
reported={('m','random'):(49.17,.14,2.9),('m','top32'):(57.28,.30,15.3),('l','random'):(54.76,.27,16.3),('l','top32'):(60.98,.19,297.2)}
for coord in ['m','l']:
    for method in ['random','u2','top32']:
        r=next(r for r in D['resources'] if r['coordinate']==coord and r['family']==method);ref=reported.get((coord,method))
        rows.append([coord.upper(),NAME[method],len(r['seeds']),pm(r['best_test_hard_pct']),num(r['training_wall_minutes']['mean']),num(r['training_peak_gpu_gib']['mean']),num(r['trainable_parameters']/1e6),f'{ref[0]:.2f} ± {ref[1]:.2f}' if ref else '—',f'{ref[2]:.1f}' if ref else '—'])
add('LILogic architectures: achieved and reported values',['Size','Method','n','Local T (%)','Local min','Local GiB','Params (M)','Reported T (%)','Reported min'],rows,'LILogic v2 (2026), seven thresholds. Published time is contextual because execution platforms and implementations differ. Reported peak GPU allocation is unavailable. Top-32 local n=1; fixed methods n=3.')

rows=[]
for coord in ['m','l']:
    lc=next(r for r in D['resources'] if r['coordinate']==coord and r['family']=='u2');top=next(r for r in D['resources'] if r['coordinate']==coord and r['family']=='top32');rand=next(r for r in D['resources'] if r['coordinate']==coord and r['family']=='random')
    lca=lc['best_test_hard_pct']['mean'];ta=top['best_test_hard_pct']['mean'];ra=rand['best_test_hard_pct']['mean']
    rows.append([coord.upper(),num(ta-lca),num(top['training_peak_gpu_gib']['mean']/lc['training_peak_gpu_gib']['mean']),num(100*(1-lc['training_peak_gpu_gib']['mean']/top['training_peak_gpu_gib']['mean'])),num(top['training_wall_minutes']['mean']/lc['training_wall_minutes']['mean']),num(100*(1-lc['trainable_parameters']/top['trainable_parameters'])),num(100*(lca-ra)/(ta-ra))])
add('Derived routing trade-offs',['Size','Accuracy cost (pp)','Top-32 / LC GPU','GPU saved (%)','Observed time ratio','Params saved (%)','Top-32 gain recovered (%)'],rows,'Ratios use locally observed means. These quantities describe an accuracy/resource trade-off, not a claim of universal dominance or a controlled timing speedup.')

rows=[]
for r in F['dense_construction_rows']:rows.append([r['label'],num(r['construction_s']),num(r['time_min']),num(r['gpu_gib']),num(100*r['construction_s']/(60*r['time_min']))])
add('Unified dense construction overhead',['Model','CPU construction (s)','Training (min)','GPU (GiB)','Construction / training (%)'],rows,'All rows use the unified constructor; no dense-specialization construction times are substituted. Training includes evaluation.')

rows=[]
for arm,r in D['factorial_arms'].items():rows.append([arm,pm(r),', '.join(num(v) for v in r['raw']),num(r['time_min']),num(r['gpu_gib'])])
add('Dense first/deeper factorial: complete seed table',['Arm','Mean ± SD (%)','Seeds 0, 1, 2 (%)','Time (min)','GPU (GiB)'],rows,'V, dense M at 20K updates. S retains LogicConnect; R preserves each predecessor degree in each input slot while randomizing pair identities. Native random is not the RR control.')

rows=[]
for key,r in D['factorial_contrasts']['best_validation_hard_pct'].items():rows.append([key.replace('_',' '),num(r['mean_pp']),f"[{r['ci95_pp'][0]:.3f}, {r['ci95_pp'][1]:.3f}]",r['wins'],', '.join(num(v) for v in r['raw_pp'])])
add('Dense factorial: all five contrasts',['Contrast','Gain (pp)','Paired 95% CI','Wins / 3','Raw gains (pp)'],rows,'V, exploratory and unadjusted. First-layer effects have positive intervals. The additional deeper effect and interaction remain inconclusive.')

for key,title in [('fourth_round_dense_mechanism','Dense M stage-selection controls'),('fourth_round_conv_mechanism','Convolutional S stage-selection controls'),('fourth_round_factorial','Convolutional body/classifier effects'),('fourth_round_warp_dense','Dense M adapted-WARP effect'),('fourth_round_warp_conv','Convolutional S adapted-WARP effect')]:
    rows=[]
    for e in D['support'][key]['paired_effects']:
        r=e['paired_effect'];label=e['contrast'].replace('u2','LC').replace('_',' ')
        rows.append([label,num(r['mean']),f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]",r['n'],e['wins']])
    add(title,['Comparison','Gain (pp)','Paired 95% CI','n','Wins'],rows,'V, exploratory. The body/classifier effects condition on a fixed reference-ancestry classifier. WARP is an adapted recipe and trails the matched raw representation. These controls do not establish an isolated benefit from ancestry scoring.')

rows=[]
for e in D['support']['fourth_round_transfer_results']['comparisons']:
    r=e['paired_effect'];rows.append([e['coordinate'],num(r['mean']),f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]" if r['n']>1 else '—',r['n'],e['wins']])
add('Frozen CIFAR-10.1 transfer',['Model','LC gain (pp)','Paired 95% CI','n','Wins'],rows,'T, no adaptation. One-time frozen evaluation. Single-seed convolutional points have no across-seed interval.')

rows=[]
for g in D['support']['third_round_all_groups']['groups']:
    family='BitLogic-adapted' if 'bitlogic' in g['phase'] else 'LILogic' if 'lilogic' in g['phase'] else 'Original dense'
    rows.append([family,g['coordinate'].upper(),NAME.get(g['family'],g['family']),g['rank'],len(g['seeds']),pm(g['best_test_hard_pct']),num(g['training_wall_minutes']['mean']),num(g['training_peak_gpu_gib']['mean']),g['provenance']])
add('Third-round inventory, including negative reproductions',['Protocol','Size','Method','Rank','n','Local T (%)','Time (min)','GPU (GiB)','Provenance'],rows,'The BitLogic-related local reproductions failed to reproduce the published accuracy. They are preserved as negative evidence; low local comparator accuracy is not evidence of superiority over published BitLogic. Rank and protocol differences prevent topology-only attribution across arbitrary rows.')

md=['# LogicConnect Evidence Atlas','','Detailed supporting results for the DATE manuscript. This atlas is an internal review and presentation artifact; its availability does not imply DATE accepts it as supplementary material.','','LC is the unified constructor; LC-D is the distinct dense specialization. Raw identifiers remain in the frozen evidence for traceability. Older broad results are preserved in [the historical catalog](evidence/historical_comparison_tables.md). Published source locations are in [reported_sources.md](evidence/reported_sources.md).','']
for sec in sections:
    md+=['## '+sec['title'],'',sec['note'],'','| '+' | '.join(sec['columns'])+' |','| '+' | '.join(['---']*len(sec['columns']))+' |']
    md+=['| '+' | '.join(row)+' |' for row in sec['rows']];md+=['']
md+=['## Standalone vector plots','']
for name,r in F['figures'].items():md.append('- ['+name.replace('_',' ')+']('+r['standalone_pdf']+') — source: `'+', '.join(r['source'])+'`.')
(PAPER/'evidence_atlas.md').write_text('\n'.join(md)+'\n')

def tex(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#').replace('±',r'$\pm$').replace('—','--')

header=r'''\documentclass[10pt,a4paper]{article}
\usepackage[landscape,margin=15mm]{geometry}
\usepackage{newtxtext,newtxmath,graphicx,booktabs,array}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={LogicConnect Evidence Atlas},pdfauthor={}}
\begin{document}
\section*{LogicConnect: Evidence Atlas}
Detailed supporting results and plots for internal review and presentation. This artifact is separate from the DATE submission. LC denotes the unified construction; LC-D is a distinct dense specialization. All local values come from the frozen evidence snapshot. Reported literature values retain separate labels and source records. No new training or inference was performed.
'''
parts=[header]
for sec in sections:
    parts+=['\\clearpage\n\\section*{'+tex(sec['title'])+'}\n',tex(sec['note'])+'\n\\par\\bigskip\n']
    n=len(sec['columns']);firstwidth=57 if n<=6 else 29;rest=(260-firstwidth-3*n)/(n-1)
    cols='@{}p{'+str(firstwidth)+'mm}'+('p{'+f'{rest:.1f}'+'mm}')*(n-1)+'@{}'
    for start in range(0,len(sec['rows']),10):
        if start:parts.append('\\clearpage\n\\section*{'+tex(sec['title'])+' (continued)}\n')
        parts+=['\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{'+cols+'}\n\\toprule\n',' & '.join(tex(c) for c in sec['columns'])+r' \\ \midrule'+'\n']
        for row in sec['rows'][start:start+10]:parts.append(' & '.join(tex(c) for c in row)+r' \\'+'\n')
        parts.append('\\bottomrule\n\\end{tabular}\n')
for name,r in F['figures'].items():
    parts+=['\\clearpage\n\\section*{'+tex(name.replace('_',' ').title())+'}\n',r'\begin{center}\includegraphics[width=184mm]{figures/'+name+r'_plot.pdf}\end{center}'+'\n',r['caption_tex']+'\n']
parts.append('\\end{document}\n')
(OUT/'evidence_atlas.tex').write_text(''.join(parts))
(PAPER/'evidence/atlas_tables.json').write_text(json.dumps(sections,indent=2)+'\n')
print('Generated',len(sections),'detailed tables and an atlas of',len(F['figures']),'plots.')
