# DATE paper draft

The completed six-page anonymous manuscript uses the title from `project.yaml` and the voice guidance from `voice_profile.yaml`. It presents LogicConnect (experimental identifier U2) as a unified dense/convolutional construction and retains V3, learned-routing trade-offs, reported references, and inconclusive findings where they affect the claim.

- [Compiled paper](output/main.pdf)
- [LaTeX entry point](output/main.tex) and [bibliography](output/references.bib)
- [Quality and evidence review](output/quality_report.md)
- [Claims and section budgets](notes/claims_and_budget.md)
- Editable section prose: [abstract](sections/00_abstract.md), [introduction](sections/01_introduction.md), [related work](sections/02_related_work.md), [methodology](sections/03_methodology.md), [evaluation](sections/04_experiments.md), [conclusion](sections/05_conclusion.md)
- Section briefs are in `notes/`; manuscript prose is in `sections/`.
- [Raw seed results](evidence/raw_results.md), [machine-readable evidence](evidence/paper_evidence.json), and [reported-value provenance](evidence/reported_sources.md)
- [Broader historical dataset/topology/method catalog](evidence/historical_comparison_tables.md). This preserves the earlier comparison document; its entire historical inventory was not re-audited for the six-page manuscript.
- [Standalone factorial plot](output/figures/factorial_effects.pdf)

## Rebuild

From this directory:

```bash
bash scripts/build.sh
```

The build reads the frozen `evidence/paper_evidence.json`, regenerates section LaTeX, tables, and the statistical plot, then compiles both PDFs using the local Tectonic binary. It runs without network access after the initial package cache has been populated. The repository Python environment is used; no new Python dependency is needed. All generated files, compiler caches, and temporary files remain in this paper directory.

Edit prose in `sections/*.md`; generated section `.tex` files and table files will be overwritten by the renderer. The controlled Markdown dialect supports headings, paragraphs, contribution bullets, bold labels, `$...$` math, display equations, `[@citation_key]` citations, LaTeX references, and `{{table:name}}`/`{{figure:name}}`/`{{algorithm:name}}` asset markers. Float enqueue points are adjusted in the renderer for the current six-page layout. The source retains URLs while the printed bibliography suppresses them for readability.

To deliberately refresh the evidence from existing experiment records, run:

```bash
../../../venv/bin/python -B scripts/collect_evidence.py
bash scripts/build.sh
```

The collector only reads saved artifacts. It does not run training, access datasets, or evaluate checkpoints. After a refresh, review changed numbers and the claim boundaries before using the manuscript.

## Scope and remaining evidence

The paper includes five principal dense coordinates across MNIST, Fashion-MNIST, and CIFAR-10; convolutional S/M; LILogic M/L routing comparisons; first/deeper-layer ablations; and the recorded compatibility/transfer boundaries. The broader historical catalog preserves additional architectures, methods, compression studies, and reported values that do not fit the main six-page narrative.

The writing and build are complete. The main scientific limitations are the inconclusive three-seed full-S validation result, single-seed convolutional held-out and Top-32 comparisons, execution-dependent wall times, and the absence of matched physical synthesis. The text states these limitations rather than proposing unsupported conclusions. This directory is an internal working package; the anonymous submission artifact is `output/main.pdf`, while source evidence and historical records retain their repository provenance.

The official DATE template and its source manifest are under `latex_templates/ieee/`. The local compiler provenance is in `.tools/compiler_source.json`. The title follows the latest user-approved LogicConnect wording; the voice-profile files were preserved.
