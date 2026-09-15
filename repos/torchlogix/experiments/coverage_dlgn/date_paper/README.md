# LogicConnect DATE manuscript

The current draft is **LogicConnect: Structured Connectivity Design for Differentiable Logic Gate Networks**. Following manuscript feedback, the sections were rewritten after rereading all six papers in `previous_pdfs/`. The revision develops the Boolean-circuit context, design rationale, and experimental interpretation in connected research prose.

- [Main paper PDF](output/main.pdf): six content pages plus one references-only page; anonymous DATE format.
- [Paper story and claim–evidence map](paper_story.md): established before section drafting.
- [Quality report](output/quality_report.md) and [section review log](output/section_review_log.md).
- [Review against the author's papers](output/research_prose_review.md), with the updated [voice profile](voice_profile.yaml).
- [Evidence atlas PDF](output/evidence_atlas.pdf) and [detailed Markdown tables](evidence_atlas.md): fifteen tables and ten two-panel vector plots for internal review and presentations.
- Editable manuscript: [abstract](sections/00_abstract.md), [introduction](sections/01_introduction.md), [related work](sections/02_related_work.md), [methodology](sections/03_methodology.md), [evaluation](sections/04_experiments.md), and [conclusion](sections/05_conclusion.md).
- [LaTeX entry point](output/main.tex) and [bibliography](output/references.bib).
- [Frozen evidence](evidence/paper_evidence.json), [raw seeds](evidence/raw_results.md), [reported-value sources](evidence/reported_sources.md), and [recent-literature review](evidence/literature_review_20260915.md).
- [Figure registry](evidence/figure_registry.json), [atlas tables](evidence/atlas_tables.json), and [validation record](output/validation.json).

The main paper contains four tables, three figures, one algorithm, and fifteen references. The dense accuracy plot remains in the atlas; its values and paired intervals are reported in the main accuracy table. Related work uses eleven papers from 2022–2026. Older dataset citations identify the benchmarks.

## Naming and evidence

LogicConnect is the frozen unified construction, archived as U2. LogicConnect-D is the distinct dense specialization, archived as V3. The methodology describes only LogicConnect. Comparative results retain both variants, highlight the largest local dense mean, and preserve measurement provenance. Specialized results are not attributed to the unified constructor.

The text emphasizes dense accuracy and training-memory benefits while retaining the learned-routing accuracy advantage, inconclusive convolutional replication, uncertain deeper/ancestry effects, and transfer limitations. Equal logical cost is not a physical area or energy result.

The atlas is a separate internal artifact; its creation does not imply DATE permits it as a submission supplement. The [historical catalog](evidence/historical_comparison_tables.md) preserves the broader earlier inventory and is not represented as an entirely re-audited experiment set.

## Rebuild and validate

From this directory:

    bash scripts/build.sh

The build uses the repository Python environment and existing local Tectonic cache. It generates the manuscript, ten two-panel vector plots, atlas, and validation records without network access or new experiments. All writes remain inside this directory.

For validation alone:

    ../../../venv/bin/python -B scripts/validate_paper.py

Validation checks 49 frozen source hashes, 16 refinement logical rows and their completion artifacts, paired statistics, method naming, citation years, fonts, references, and PDF bounds. It does not load checkpoints or access datasets.

Edit prose in sections/*.md; generated section LaTeX and tables are overwritten by the renderer. Figure and atlas generators read the frozen evidence. The nominal project.yaml prose targets exceed the space available beside the figures and tables; justified shortfalls are recorded in the quality report.

## Preserved first effort

The earlier manuscript and supporting sources are preserved in [output/revisions/before_story_20260915/](output/revisions/before_story_20260915/). Commit `4b51235` preserves the draft immediately before the revision from `previous_pdfs/`. Previous pass1–pass3 snapshots remain historical. Original section notes remain research input; paper_story.md is the current narrative anchor.

Compiler and official-template provenance remain in .tools/ and latex_templates/. No experiment implementation, checkpoint, or result directory was changed.
