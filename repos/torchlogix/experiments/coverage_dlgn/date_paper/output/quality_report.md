# Quality Report

## Manuscript Summary

- Title: **LogicConnect: Structured Connectivity Design for Differentiable Logic Gate Networks**.
- Venue: DATE 2027, anonymous research-paper draft.
- Actual length: **six content pages plus one references-only page**.
- Main evidence: **four tables, four figures, one algorithm, fifteen references**.
- Companion material: **ten two-panel vector plots and fifteen detailed tables**, assembled into a 28-page atlas.
- Build: successful with the local Tectonic compiler and cached dependencies.
- Scope: all changes remain within date_paper; no new training, inference, dataset access, or experiment-code changes.
- Earlier manuscript preserved under revisions/before_story_20260915/.

This is a completed manuscript draft and editorial audit. It has not undergone external peer review; the audit does not establish acceptance or supply missing experimental evidence.

## Scientific Narrative

The problem is restricted predecessor access at a fixed DLGN depth and gate budget. Gate training cannot create an absent connection, while learning connections adds training state. The insight is to design signal combinations before training using semantic input pairs and regular multiscale matchings.

[paper_story.md](../paper_story.md) was created before drafting. The introduction establishes the constraint, degree-preserving observation, insight, and three contributions. The methodology motivates the design through the constraints it addresses. Evaluation answers questions about accuracy, resources, attribution, and transfer. The conclusion synthesizes those findings without introducing another claim.

## Claim–Evidence Audit

| Claim | Design and evidence | Status | Boundary |
|---|---|---|---|
| C1: shared fixed-connectivity construction | Semantic pairing, degree-first stage selection, ancestry tie criterion, channel/classifier adapters; Fig. 1, Algorithm 1, inspected source/configurations | Supported for evaluated rank-2 architectures | Equal gate budgets do not imply equal physical area or wire cost. |
| C2: improved dense accuracy | Table II and Fig. 2: CIFAR-10 gains +3.18/+4.56/+4.59 pp, n=3, positive paired intervals | Supported | Fashion-MNIST interval includes zero; the dense specialization can be better. |
| C2: training-resource trade-off | Table IV and Fig. 3: 93.7–94.2% less peak GPU memory, 80% fewer trainable parameters, 1.84–5.30 pp accuracy cost against Top-32 | Supported as a measured trade-off | Top-32 n=1; wall-time ratios do not isolate algorithmic speedup. |
| C2: convolutional application | Table III: S/M test gains +3.26/+2.08 pp, n=1; full-S validation replication retained | Supported compatibility and descriptive gains | Full-S n=3 validation interval [-3.36, 5.82] is inconclusive. |
| C3: first-layer effect beyond degree | Fig. 4: per-predecessor, per-slot degree-preserving randomization; first-layer effects +4.49/+3.69 pp | Supported at dense M, 20K updates | Deeper +0.52 pp interval includes zero; all five contrasts are exploratory and unadjusted. |
| C3: mechanism and transfer boundaries | Nominal controls, conditional convolutional attribution, adapted WARP, frozen CIFAR-10.1; atlas plots | Supported bounded characterization | No isolated ancestry-selection gain, universal compatibility, or universal transfer improvement. |

## Naming and Selection

LogicConnect names the frozen unified method. LogicConnect-D names the distinct dense specialization. Only the unified construction appears in the methodology. Table II retains both variants and highlights the largest observed local mean. No best-of-variants test maximum is represented as an evaluated selection policy or attributed to an algorithm that did not produce it.

The abstract and headline gains use unified-method measurements. Raw U2/V3 identifiers remain internal provenance, script keys, notes, and preserved drafts.

## Section Audit

Counts exclude headings, asset markers, citations, math, captions, tables, and bibliography text.

| Section | Scientific role and evidence | Style and continuity | Words | Nominal target | Budget |
|---|---|---|---:|---:|---|
| Abstract | Problem, method, scope, resource cost, ablation | Written last; every result developed in the body | 174 | 170–200 range | Pass |
| Introduction | Constraint → cost → observation → insight → contributions | Direct technical opening; bounded headline results | 449 | 700 | Justified shortfall |
| Related work | Graph, routing, optimization, simplification | Synthesized research choices; eleven 2022–2026 sources | 317 | 490 | Justified shortfall |
| Methodology | Graph, semantic rationale, stage decision, convolution, cost | Equations explained; procedural detail specifies the evaluated algorithm | 795 | 1610 | Justified shortfall |
| Evaluation | RQ1–RQ4, matching, costs, controls, limits | Patterns and implications; losing coordinates retained | 894 | 1610 | Justified shortfall |
| Conclusion | Mechanism, dense gain, resource trade-off, scope | Synthesis without new results | 117 | 490 | Justified shortfall |

The body totals **2,572 prose words** against 4,900 nominal words in the user-edited configuration. Those targets cannot all fit beside four tables, four figures, equations, and an algorithm in six content pages at the required type size. Targets were preserved; repeated framing and interpretation were removed. The validation record reports word_budgets_pass=false and budget_exceptions_justified=true. No padding, smaller body fonts, or compressed line spacing was used.

## Explicit Style and Anti-Manual Review

style.yaml guided the evidence-bounded tone, concrete subjects, explicit measurements, and paragraph continuity. voice_profile.yaml supplied complementary author preferences.

The introduction starts from an architectural constraint. Each mechanism explains the purpose of semantic distinctions, disjoint matchings, degree priority, or channel adaptation. Results follow question/comparison/observation/implication rather than run order. No repository inventory, API walkthrough, or chronological implementation narrative remains.

No authored em dash, listed banned term, or prose-only sentence over the 35-word review threshold remains. Names, mathematical definitions, comparison scope, and uncertainty stay consistent. Native-random and degree-preserving controls are distinguished. Accuracy benefits retain the associated resource cost or uncertainty.

Approximate sentence means excluding math and labels are 15.8/18.0/13.6/14.5/13.3/13.0 words from abstract through conclusion. Several sections are below the soft 18–24-word guide. Contextual review accepts these shorter result and mathematical sentences because joining them would blur comparison, condition, or limitation. This is an explicit editorial judgment, not a claim of exact stylistic imitation.

Scientific and continuity checks followed each section, with complete-text reviews after concision passes. [section_review_log.md](section_review_log.md) records the decisions. The candidate passed the scientific and anti-manual review within three major prose passes.

## Numerical and Source Validation

[validate_paper.py](../scripts/validate_paper.py) performs the following without running a model:

- Verifies all **49 source hashes** in the frozen experimental registry.
- Checks all **16 refinement logical rows**, including artifact existence, metrics/configuration/summary hashes, and applicable completion receipts.
- Recomputes dense moments and paired intervals, full-S replication, dense and convolutional factorial effects, mechanism controls, adapted-WARP effects, and transfer statistics.
- Allows 0.001 pp in interval recomputation for rounded Student-t constants in archived summaries; printed conclusions are unaffected.
- Checks headline values, scope qualifiers, method naming, related-work years, font embedding, references, and page bounds.
- Verifies ten single-page standalone plots and the references-only seventh manuscript page.

Reported values remain separate, with PDF locations in [reported_sources.md](../evidence/reported_sources.md). Every plot has a source entry in [figure_registry.json](../evidence/figure_registry.json). Negative interval endpoints remain visible. Poor local BitLogic-related reproductions remain negative evidence in the atlas and do not justify superiority over published BitLogic.

The broad historical catalog is preserved; this review does not claim to re-audit every historical run or rehash all checkpoint binaries.

## Literature and Citations

All eleven related-work sources date from 2022–2026. Added Mind the Gap, scalability, and multilinear-fitting references were checked against supplied PDFs and primary metadata. Preprints retain their status. Supplied BitLogic and unit-tying PDFs establish their publication metadata; older sources identify datasets only. [The literature record](../evidence/literature_review_20260915.md) documents verification.

No missing manuscript citation remains. The paper makes no firstness or state-of-the-art claim.

## Build and Visual Checks

The seven-page A4 manuscript uses the official DATE IEEEtran template and 184 mm × 239 mm text area. Page seven contains only references. There is no author block, acknowledgment, page number, or copyright notice.

Body, abstract, captions, tables, and bibliography use 10-point Times-compatible text. Plot labels use 11-point source text in a 198 mm canvas, displayed at 184 mm, retaining approximately 10.22 points. Mathematical subscripts use standard math typography.

All fonts are embedded. No Type 3 fonts, unresolved citations/references, or overfull boxes remain in the main paper, atlas, or ten plots. Each standalone plot has exactly one nonblank page. All extracted word boxes stay inside PDF bounds. Every main page and plot panel was visually reviewed for clipping, readability, equations, labels, uncertainty, and reference placement.

An initial compiler failure during bold-math loading was resolved by using ordinary bold text within math for highlighted means. The atlas uses cached base table support instead of unavailable optional packages. Neither adjustment changes numbers or claims. Benign class/font initialization and legacy-package notices remain in logs.

[build.sh](../scripts/build.sh) compiles both PDFs and runs the audit offline. [validation.json](validation.json), [pdfinfo.txt](pdfinfo.txt), and [pdffonts.txt](pdffonts.txt) record the final checks.

## Potential Overclaims and Remaining Scientific Issues

Excluded claims include universal best accuracy; attributing the dense specialization to the unified constructor; an independently established balancing/ancestry cause; a general deeper-layer benefit; replicated convolutional held-out superiority; hardware-independent timing speedups; physical area/energy gains from gate count; and superiority inferred from failed local BitLogic reproductions.

Remaining evidence needs are replicated convolutional held-out accuracy, additional learned-routing seeds, an isolated ancestry-selection benefit, and matched physical implementation measurements. These require further experiments, not stronger prose. No indispensable author-dependent placeholder remains.
