# Claims, evidence, and section budgets

The manuscript uses the exact title in `project.yaml`. The singular final word is preserved as supplied. No author information was supplied, and the DATE 2027 instructions require anonymous review. The paper therefore omits author names, affiliations, acknowledgments, page numbers, and copyright notices.

The official template is stored under `latex_templates/ieee/`; the source URL and archive hash accompany it. DATE allows six content pages and an optional seventh page for references only. This project targets six total pages. Do not compress line spacing or shrink the body below 10 points.

## Claims map

| ID | Introduction claim | Method support | Evidence and boundary |
|---|---|---|---|
| C1 | One offline construction covers dense and convolutional LGNs | Semantic input pairing, matching-stage selection, convolutional channel adapter, fixed-index export | U2 implementation in `src/torchlogix/` (relative to the TorchLogix root) is inspected directly; construction changes channel pairing but preserves spatial sampling, tree shape, and classifier dimensions |
| C2 | Structured fixed connectivity improves dense CIFAR-10 accuracy at unchanged declared architectural cost | Two inputs per gate, same gate functions/parameters, no learned routing | Three-seed S/M/L held-out comparisons; V3 remains better at L; Fashion-MNIST interval includes zero |
| C3 | The construction also applies to convolutional LGNs, with qualified empirical benefit | Same pair builder for channels and classifier; channel ancestry differs from dense pixel ancestry | S/M single-seed held-out gains; full-S three-seed validation gain is inconclusive; do not imply universal convolutional robustness |
| C4 | Controlled ablations locate the dense gain primarily in first-layer structure | Separately replace first/deeper indices while preserving per-node, per-slot degrees | Dense M 20K, SS/SR/RS/RR, n=3; deeper contribution and interaction intervals cross zero |

C1-C4 form three contribution bullets: construction; accuracy/resource evaluation; controlled attribution. No contribution claims maximum coverage, proven novelty-selection benefit, physical area savings, universal gains, or state-of-the-art accuracy.

## Prose budget before drafting

Derived from the user-supplied 450 words/page and 15% tolerance. Counts exclude headings, equations, captions, tables, references, and research notes. The abstract has a separate 140-180-word target. Target pages are planning weights; floats and references determine the actual page layout.

| Section | Page weight | Target words | Accepted interval |
|---|---:|---:|---:|
| 01 Introduction | 1.0 | 450 | 383-517 |
| 02 Related work | 1.0 | 450 | 383-517 |
| 03 Methodology | 2.0 | 900 | 765-1035 |
| 04 Experiments | 1.3 | 585 | 498-672 |
| 05 Conclusion | 0.7 | 315 | 268-362 |

The relatively large conclusion allocation is used for synthesis, applicability boundaries, and remaining scientific limitations. Experimental protocol and raw seeds also appear in tables and the local evidence notes rather than being omitted to fit the prose budget.

## Quality loop

1. Draft all five section briefs before writing the manuscript. Read all section prose before every revision pass.
2. Pass 1: verify claims, define notation, generate tables from source summaries, and check budgets and transitions.
3. Pass 2: revise only failing prose, review long sentences and generic wording, and inspect compiled layout.
4. Pass 3 if necessary: resolve remaining prose or layout failures. Record remaining research limitations rather than inventing supporting evidence.

The quality report must distinguish a completed writing/build review from evidence still needed before submission. No new training, test access, or synthesis is part of this writing task.
