"""Render the standalone factorial plot from the archived paired contrasts."""
from pathlib import Path
import json
PAPER = Path(__file__).resolve().parents[1]
D = json.loads((PAPER/'evidence/paper_evidence.json').read_text())
tex = '\\documentclass[10pt]{article}\n\\usepackage[paperwidth=94mm,paperheight=59mm,margin=1mm]{geometry}\n\\usepackage{newtxtext,newtxmath,tikz,pgfplots}\n\\pgfplotsset{compat=1.18}\n\\definecolor{methodblue}{HTML}{216A85}\n\\definecolor{controlgray}{HTML}{555555}\n\\definecolor{accentorange}{HTML}{A64B1D}\n\\pagestyle{empty}\n\\begin{document}\n\\noindent\\begin{tikzpicture}\n\\begin{axis}[width=88mm,height=59mm,xmin=-3.5,xmax=6.4,ymin=-0.5,ymax=4.5,\n xtick={-2,0,2,4,6},ytick={0,1,2,3,4},\n yticklabels={Interaction,$RS-RR$,$SS-SR$,$SR-RR$,$SS-RS$},\n tick label style={font=\\fontsize{11}{13}\\selectfont},label style={font=\\fontsize{11}{13}\\selectfont},\n xlabel={Accuracy difference (pp)},axis x line*=bottom,axis y line*=left,\n xmajorgrids=true,grid style={gray!20},tick align=outside]\n\\draw[dashed,gray] (axis cs:0,-.5)--(axis cs:0,4.5);\n'
keys=['first_given_structured_deeper','first_given_randomized_deeper',
      'deeper_given_structured_first','deeper_given_randomized_first','interaction']
for i,key in enumerate(keys):
    r=D['factorial_contrasts']['best_validation_hard_pct'][key]
    color='methodblue' if i<2 else 'controlgray' if i<4 else 'accentorange'
    half=(r['ci95_pp'][1]-r['ci95_pp'][0])/2
    tex += (r'\addplot+[only marks,mark=*,mark size=2pt,color='+color+
            r',mark options={fill='+color+',draw='+color+'}'+
            r',error bars/.cd,x dir=both,x explicit,error bar style={line width=.8pt},error mark options={rotate=90,mark size=3pt}] coordinates {'+
            f"({r['mean_pp']},{4-i}) +- ({half},0)"+'};\n')
tex += r"\end{axis}\end{tikzpicture}\end{document}"+'\n'
(PAPER/'output/figures/factorial_effects.tex').write_text(tex)
print('Rendered five paired effects from the evidence snapshot.')
