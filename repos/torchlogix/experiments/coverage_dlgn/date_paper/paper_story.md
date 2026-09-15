# Paper Story

Prepared on 2026-09-15 before the section rewrite. The previous manuscript, sources, and reports are preserved in `output/revisions/before_story_20260915/`.

## Manuscript revision after reading the author's papers

The user found the prose in commit `4b51235` too similar to a research report. That commit preserves the starting manuscript for this revision. All six PDFs in `previous_pdfs/` were reread before changing the sections; their hashes match the corpus recorded in `voice_profile.yaml`.

The scientific question and C1–C3 evidence map below remain unchanged. The revision changes how that argument is developed: introduce Boolean inference and the connectivity constraint before the numerical motivation; explain semantic pairing and multiscale construction through their design rationale; then interpret comparisons in connected paragraphs. RQ identifiers remain internal planning aids. Descriptive subsection titles and ordinary statistical language replace report labels in the manuscript. Numerical results, equations, published-reference distinctions, and method identities are preserved.

The working allocation is approximately 500–550 introduction words, 330–400 related-work words, 900–1050 method words, 1000–1150 evaluation words, and 120–160 conclusion words. These are editorial estimates within the existing six-content-page requirement, not replacements for the user's nominal configuration. The prose must remain explanatory when layout is adjusted.

## Central Research Question

How much accuracy can a differentiable logic gate network recover by designing its fixed connections before training, without paying for trainable routing or increasing the declared gate budget?

## Problem

A two-input gate sees only its assigned predecessors. Gate-function training cannot introduce a missing connection into a fixed graph. This makes the connection graph an architectural constraint on the combinations of signals available at a given depth and width. Learned routing expands the search but adds candidate-signal processing and training memory, creating a practical design choice for resource-constrained training.

## Gap

Recent DLGN research improves gate relaxation, learns connections, extends architectures, and simplifies trained circuits. These advances do not answer the narrower question of what a deliberately structured, fixed graph provides under an otherwise matched architecture and training recipe. A shared design must also respect the difference between dense input coordinates and shared convolutional channel groups.

The gap is a measured architectural trade-off, not an unsupported claim that no prior work has used structured connections. Native dense random already balances fan-out. Degree balance alone therefore cannot explain the measured gains.

## Key Observation

The existing experiments show that predecessor identity matters beyond predecessor degree. At dense CIFAR-10 M, randomizing first-layer wiring while preserving each predecessor's degree in each input slot reduces hardened validation accuracy by 4.49 points when deeper wiring remains structured. The corresponding paired 95% confidence interval is [3.90, 5.09] over three seeds. Conversely, the additional deeper-layer effect has an interval containing zero. These are findings from a completed ablation, not measurements used to fit the constructor.

## Core Insight

Treat the graph as a design variable before training: use image semantics to constrain meaningful input pairs and regular multiscale matchings to organize subsequent signal combinations. The resulting inductive bias uses the ordinary fixed-index representation. It trades an offline construction cost for a more useful graph without introducing a second learned routing problem.

## Proposed Approach

LogicConnect is the frozen unified construction recorded as U2 / `semantic_multiscale_balanced` in the experiments. It uses semantic input pairing; degree-first selection of whole multiscale matching stages; an ancestry-overlap tie criterion; and a convolutional adapter that changes channel pairs while retaining spatial sampling and weight sharing. No accuracy-driven selection or individual-edge swapping occurs during construction. Training learns the gate functions on the resulting fixed graph.

The methodology describes this construction only. The older dense specialization is named LogicConnect-D in comparative results and is explicitly identified as a distinct butterfly-and-swap variant (archived V3 / `semantic_balanced_hybrid`). Its measurements are never attributed to the unified constructor. Both columns are retained, and the largest observed local mean in a matched row may be highlighted. A maximum chosen after viewing test results is not presented as an evaluated selection policy or a new confidence interval.

## Central Claim

Offline connectivity design offers a useful DLGN accuracy/resource operating point: LogicConnect improves the three evaluated dense CIFAR-10 models over matched fixed random connections, preserves the declared architecture, and uses substantially less training memory than the evaluated learned-routing comparator. The convolutional implementation demonstrates shared applicability and positive single-seed held-out results, with replicated validation evidence that remains inconclusive.

## Contributions

- C1: A shared fixed-connectivity construction for dense and convolutional DLGNs, grounded in semantic input pairs and regular multiscale predecessor matchings.
- C2: A measured accuracy/resource trade-off: dense gains under matched budgets, convolutional evidence with its replication boundary, and a quantified comparison with learned routing.
- C3: Structural attribution that separates predecessor identity from degree and locates the strongest measured dense contribution in the first layer, while testing the limits of deeper selection and transfer.

## Claim–Evidence Map

### C1 — Connectivity design before training

Claim: One construction serves dense gates, convolutional channel groups, and the convolutional classifier without extra trainable routing state or inference operators.

Design: semantic raw-source labels; staged matchings; lexicographic degree/overlap selection; channel and flattening adapters.

Evidence: `../../src/torchlogix/topology.py`, dense/conv model code, saved configurations in `evidence/paper_evidence.json`; overview figure and construction algorithm.

Interpretation: compatibility follows from the actual adapters and fixed-index representation. Equal gate count and operator type are architectural properties, not physical area or energy measurements.

Status: supported for the evaluated rank-2 architectures. No rank-r or large-model generalization is claimed.

### C2 — Accuracy and training-resource trade-off

Claim: LogicConnect improves dense CIFAR-10 accuracy under matched architecture and training settings and supplies a lower-memory operating point than learned Top-32 routing.

Evidence and planned assets:

- Dense table and accuracy/gain plot: S/M/L test gains +3.183/+4.557/+4.593 points, paired 95% intervals [0.772, 5.595], [3.781, 5.332], [3.721, 5.466], three seeds per method and model.
- MNIST/Fashion-MNIST rows: gains +0.663/+0.520 points; Fashion-MNIST interval includes zero. All five unified dense comparisons have three positive seed differences.
- Convolutional table: S/M test accuracy 60.63/71.65%, versus random 57.37/69.57%, one seed each. Full-S validation 61.053% versus 59.820%, gain +1.233 with interval [-3.356, 5.823], two wins in three seeds.
- Routing table and memory/time plots: LILogic M/L accuracy 52.543/60.193% versus Top-32 57.84/62.03%. Top-32 uses 17.1/16.0 times the peak GPU memory; LogicConnect uses 80% fewer trainable parameters. Top-32 has one local seed, fixed methods three.
- Dense resource plot: architecture-matched peak allocations; offline construction and observed training times shown separately. Full-M convolutional training is 35.69 h for LogicConnect versus 25.47 h for random; execution conditions prevent isolating a topology-induced timing effect.
- Dense specialization column: all V3 measurements retained as LogicConnect-D. Its 61.073% at dense L exceeds the unified 60.463%; no substitution is made.

Baselines: locally reproduced fixed random; locally reproduced Top-32; explicit dense specialization; published values retained as reported references, not paired observations.

Interpretation: superiority is specific to a matched accuracy or resource coordinate. The evidence does not establish universal best accuracy, a universal Pareto frontier, or a controlled wall-time speedup.

Status: supported with the stated scope; replicated convolutional held-out gains require additional experiments.

### C3 — What structure contributes

Claim: First-layer predecessor identity explains the strongest measured dense structural effect after controlling per-slot degrees; deeper selection has no separately established gain.

Evidence and planned assets:

- Dense factorial plot and full-seed table: SS/SR/RS/RR validation means 59.133/58.613/54.640/54.927%; first-layer contrasts +4.493/+3.687 points; deeper SS-SR +0.520, interval [-0.523, 1.563]. All five contrasts, including interaction, retained.
- Mechanism-control plot: unified-minus-nominal intervals include zero in both dense M and convolutional S at 20K updates. Ancestry selection is a defined heuristic, not a causally established accuracy mechanism.
- Convolutional body/classifier control: body effect +2.140 points with a fixed reference-structured classifier, interval [0.674, 3.606]. The classifier uses reference-body ancestry, so the effect is conditional.
- Transfer plot: frozen CIFAR-10.1 dense M gain +3.733, interval [2.801, 4.666]; convolutional S/M -0.050/+1.950, one seed each. CIFAR-100 pilots do not establish a gain.
- Adapted WARP plot: dense topology gain +4.700 points; convolutional effect inconclusive and both adapted recipes trail their matched raw parameterizations.

Interpretation: semantic structural access is useful in the evaluated dense model; the data do not show that maximum ancestry coverage, degree balancing alone, or every deeper-layer decision causes an improvement.

Status: supported as a bounded characterization. Exploratory intervals are unadjusted for multiple comparisons.

## Evaluation Questions

- RQ1: Does structured fixed connectivity improve hardened accuracy at the same architectural budget?
- RQ2: What accuracy, training-memory, parameter, and wall-time trade-offs does it offer relative to learned routing?
- RQ3: Which structural choices contribute after preserving predecessor degrees?
- RQ4: Where do the benefits weaken under architecture, gate-parameterization, and distribution changes?

## Important Limitations

- Three training seeds give wide intervals; the convolutional and Top-32 held-out comparisons have one seed each.
- Validation and test measurements must remain distinct, and no new dataset access is part of this writing task.
- LogicConnect-D is a different construction. Selecting its best test number and labeling it as the unified LogicConnect result would invalidate methodological attribution.
- Reported numbers use their authors' hardware and protocols. Missing exact-budget references and peak-memory measurements remain missing.
- Wall times include evaluation and reflect execution conditions. Physical synthesis, area, delay, and energy were not measured under a matched flow.
- No general benefit is established for ancestry-based tie selection, deeper structure, CIFAR-100, or arbitrary training parameterizations.

## Evidence Policy and Literature Window

The frozen `evidence/paper_evidence.json` snapshot supplies the existing main results. Its 49 registered source hashes match the repository on 2026-09-15. Derived tables and plots must be generated from this snapshot or from additional explicitly hash-linked saved summaries. Checkpoints are not rerun. Raw experiment identifiers remain in provenance files.

Related work uses relevant 2022–2026 papers, within five years of this writing date. The collection is in the root `README.md` and `pdfs/`; `DLCA/README.md` describes the cellular-automata notebooks. Original older dataset citations are retained solely to identify the benchmarks. Preprints are labeled as preprints, and archived versions determine quoted numerical references.

## Section Roles and Budget

The current user-edited configuration gives seven total pages and nominal body targets of 700/490/1610/1610/490 words. DATE permits six content pages plus one references-only page. These nominal prose targets cannot all be filled alongside the requested evidence-rich figures and tables at 10-point text. Preserve the configuration and record justified shortfalls rather than pad prose or violate the page limit.

| Section | Role | Working prose allocation |
|---|---|---:|
| Introduction | Problem, evidence-backed observation, insight, and three contributions | 450–550 |
| Related work | Distinguish graph design from learned routing, gate training, and simplification | 350–450 |
| Methodology | Explain the unified design through challenges, rationale, equations, and an algorithm | 850–1050 |
| Evaluation | Answer RQ1–RQ4 with comparisons, costs, ablations, and interpretation | 900–1150 |
| Conclusion | Synthesize the mechanism, strongest results, and limits | 150–210 |
| Abstract | Write last, using only results developed in the body | 170–200 |

Each section is drafted and audited before the next. After each completed section, reread the manuscript accumulated so far and its neighboring source sections for scientific consistency, terminology, voice, and continuity. Keep an audit record in `output/section_review_log.md`. The final quality report distinguishes editorial completion from unresolved scientific evidence.

## Figure and Table Strategy

The revised main paper uses figures for the overview, routing trade-offs, and structural attribution. The dense scaling plot remains in the evidence atlas, because the main dense table already reports its means and paired intervals. Additional standalone figures cover seed variability, construction overhead, mechanism controls, convolutional attribution, transfer, and gate-parameterization compatibility. The separate atlas retains all ten plots and detailed comparison tables; it is not assumed to be an admissible DATE supplement. Every plotted point must have a machine-readable source, scope, and seed count.

## Narrative

Restricted gate inputs constrain representable signal combinations
→ learned routing adds training cost
→ predecessor identity matters even at preserved degree
→ design semantic pairs and regular matchings before training
→ retain ordinary gates and fixed indices across architectures
→ measure accuracy, resources, attribution, and transfer
→ establish a useful, bounded operating point for DLGN design.
