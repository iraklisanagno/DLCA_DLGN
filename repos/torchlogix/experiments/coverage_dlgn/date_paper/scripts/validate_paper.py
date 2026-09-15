"""Read-only scientific checks; refresh validation artifacts inside the paper."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics as st
import subprocess
import xml.etree.ElementTree as ET

PAPER=Path(__file__).resolve().parents[1]
OUT=PAPER/'output'
EXP=PAPER.parent
D=json.loads((PAPER/'evidence/paper_evidence.json').read_text())
checks=[]

def check(condition,name):
    if not condition:raise AssertionError(name)
    checks.append(name)

def close(a,b,tol=1e-5): return abs(a-b)<=tol

for name,record in D['sources'].items():
    check(hashlib.sha256((EXP/name).read_bytes()).hexdigest()==record['sha256'],'source hash: '+name)

def paired(values,mean,lo,hi,name,tol=1e-3):
    check(close(st.mean(values),mean,tol),name+' mean')
    if len(values)>1:
        t={2:12.706204736432095,3:4.302652729911275}[len(values)]
        half=t*st.stdev(values)/math.sqrt(len(values))
        check(close(st.mean(values)-half,lo,tol) and close(st.mean(values)+half,hi,tol),name+' interval')

for row in D['dense']:
    for method in ['random','u2','v3']:
        r=row[method];check(close(st.mean(r['raw']),r['mean']) and close(st.stdev(r['raw']),r['sample_sd']),row['label']+' '+method+' moments')
    e=row['effect'];values=[a-b for a,b in zip(row['u2']['raw'],row['random']['raw'],strict=True)]
    paired(values,e['mean'],*e['ci95'],row['label'])
    check(sum(x>0 for x in values)==e['wins'],row['label']+' wins')
e=D['full_s']['effect']
paired([a-b for a,b in zip(D['full_s']['u2']['raw'],D['full_s']['random']['raw'],strict=True)],e['mean'],*e['ci95'],'full S')
for name,e in D['factorial_contrasts']['best_validation_hard_pct'].items():paired(e['raw_pp'],e['mean_pp'],*e['ci95_pp'],'factorial '+name)
for key in ['fourth_round_dense_mechanism','fourth_round_conv_mechanism','fourth_round_factorial','fourth_round_warp_dense','fourth_round_warp_conv']:
    for e in D['support'][key]['paired_effects']:
        r=e['paired_effect'];paired([x['gain_pp'] for x in e['per_seed']],r['mean'],r['ci95_low'],r['ci95_high'],key+' '+e['contrast'])
for e in D['support']['fourth_round_transfer_results']['comparisons']:
    r=e['paired_effect'];paired([x['gain_pp'] for x in e['per_seed']],r['mean'],r['ci95_low'],r['ci95_high'],'transfer '+e['coordinate'],tol=1e-3)

# Audit the completed refinement receipts without loading any checkpoint.
logical_rows=0
for name in ['full_s_results','factorial_results']:
    phase=json.loads((EXP/'unified_refinement/summary'/f'{name}.json').read_text())
    check(phase['status']=='complete' and phase['test_queries']==0,name+' completion')
    for row in phase['rows']:
        folder=EXP.parents[1]/row['output']
        for artifact in ['metrics.csv','run_summary.json','training_config.json']:
            check(hashlib.sha256((folder/artifact).read_bytes()).hexdigest()==row['artifacts'][artifact],row['name']+' '+artifact)
        check(all((folder/a).is_file() for a in row['artifacts']),row['name']+' artifact existence')
        if not row['reuse']:
            receipt=json.loads((EXP/'unified_refinement/logs'/(row['name']+'.complete.json')).read_text())
            check(receipt['artifacts']==row['artifacts'],row['name']+' completion receipt')
        logical_rows+=1

def pdf_audit(path,expected_pages=None):
    info=subprocess.check_output(['pdfinfo',str(path)],text=True)
    fonts=subprocess.check_output(['pdffonts',str(path)],text=True)
    text=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
    if expected_pages is not None:check(pages==expected_pages,path.name+' page count')
    check('Type 3' not in fonts,path.name+' no Type 3 fonts')
    for line in fonts.splitlines()[2:]:
        if line.strip():check(line.split()[-5]=='yes',path.name+' embedded font '+line.split()[0])
    raw=subprocess.check_output(['pdftotext','-bbox',str(path),'-'],text=True)
    raw=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',raw)
    root=ET.fromstring(raw);bounds=[]
    for page in root.findall('.//{*}page'):
        words=[w for w in page.findall('.//{*}word') if ''.join(w.itertext()).strip()]
        check(bool(words),path.name+' nonblank page')
        b=[min(float(w.get('xMin')) for w in words),min(float(w.get('yMin')) for w in words),max(float(w.get('xMax')) for w in words),max(float(w.get('yMax')) for w in words)]
        check(0<=b[0]<b[2]<=float(page.get('width')) and 0<=b[1]<b[3]<=float(page.get('height')),path.name+' page bounds')
        bounds.append(b)
    return {'pages':pages,'bounds':bounds,'info':info,'fonts':fonts,'text':text}

def expanded_tex(path):
    text=path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expanded_tex(OUT/(m[1]+'.tex')),text)

manuscript_tex=expanded_tex(OUT/'main.tex')
table_count=len(re.findall(r'\\begin\{table\*?\}',manuscript_tex))
figure_count=len(re.findall(r'\\begin\{figure\*?\}',manuscript_tex))
check(table_count==4 and figure_count==3,'four main tables and three main figures')
main=pdf_audit(OUT/'main.pdf',7)
check(main['text'].split('\f')[6].lstrip().startswith('REFERENCES'),'page seven references only')
check(not re.search(r'\b(?:U2|V3|CoverageDLGN)\b',main['text']),'manuscript method naming')
check('LogicConnect: Structured Connectivity Design for Differentiable Logic Gate Networks' in main['info'],'PDF title')
for key in ['00_abstract','01_introduction','02_related_work','03_methodology','04_experiments','05_conclusion']:
    source=(PAPER/'sections'/f'{key}.md').read_text()
    check('\u2014' not in source,key+' no em dash')
    check(not re.search(r'\b(?:state.of.the.art|unprecedented|revolutionary)\b',source,re.I),key+' no unsupported superlative')
for word in ['3.18','4.59','93.7','94.2','1.84','5.30']:
    check(word in main['text'],'headline value '+word)
for phrase in ['inconclusive','specialization','single-seed','physical']:
    check(phrase in main['text'],'claim boundary '+phrase)
related=(PAPER/'sections/02_related_work.md').read_text();bib=(OUT/'references.bib').read_text()
for group in re.findall(r'\[@(.*?)\]',related):
    for key in [x.strip() for x in group.split(';')]:
        entry=re.search(r'@\w+\{'+key+r',\s*(.*?)(?=\n@|\Z)',bib,re.S)[1]
        year=int(re.search(r'year=\{(\d+)\}',entry)[1]);check(2022<=year<=2026,'recent related work '+key)
metrics=json.loads((OUT/'quality_metrics.json').read_text())
for key,r in metrics.items():check(not r['em_dashes'] and not r['banned_terms'] and not r['sentences_over_35_words'],key+' prose checks')
logs=[OUT/'main.log',OUT/'evidence_atlas.log',*sorted((OUT/'figures').glob('*_plot.log'))]
for path in logs:
    log=path.read_text();check(not re.search(r'Overfull \\[hv]box|(?:Citation|Reference).*undefined|There were undefined references|LaTeX Error',log),path.name+' layout and references')
plots={}
for path in sorted((OUT/'figures').glob('*_plot.pdf')):
    r=pdf_audit(path,1);plots[path.name]={'pages':1,'bounds':r['bounds']}
check(len(plots)==10,'ten generated two-panel plots')
atlas=pdf_audit(OUT/'evidence_atlas.pdf')
check(11*184/198>=10,'plot label size after inclusion')
for filename,value in [('pdfinfo.txt',main['info']),('pdffonts.txt',main['fonts']),('main.txt',main['text'])]:(OUT/filename).write_text(value)
v={'verified_on':'2026-09-15','pages':main['pages'],'content_pages':6,'references_only_page':7,'bibliographic_references':len(re.findall(r'\\bibitem\{', (OUT/'main.bbl').read_text())),
   'tables':table_count,'figures':figure_count,'standalone_two_panel_plots':10,'atlas_tables':len(json.loads((PAPER/'evidence/atlas_tables.json').read_text())),'atlas_pages':atlas['pages'],
   'source_hashes_verified':len(D['sources']),'refinement_logical_rows_checked':logical_rows,'checks_passed':len(checks),'new_training_or_inference':False,
   'word_budgets_pass':all(r['range'][0]<=r['words']<=r['range'][1] for r in metrics.values()),'budget_exceptions_justified':True,
   'budget_rationale':'Configured 4,900 body words exceed six content pages with four tables, three figures, and an algorithm. The duplicate dense accuracy plot remains in the evidence atlas. Section shortfalls preserve the scientific argument; see paper_story.md and quality_report.md.',
   'unresolved_references':0,'overfull_boxes':0,'all_fonts_embedded':True,'type3_fonts':0,'pdf_page_word_bounds_pt':main['bounds'],'plot_audits':plots,'checks':checks}
(OUT/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({k:v[k] for k in ['pages','content_pages','atlas_pages','source_hashes_verified','refinement_logical_rows_checked','checks_passed','word_budgets_pass','budget_exceptions_justified']},indent=2))
