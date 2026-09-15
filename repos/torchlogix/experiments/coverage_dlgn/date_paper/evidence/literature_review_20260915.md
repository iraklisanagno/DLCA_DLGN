# Literature verification for the narrative rewrite

Related work covers 2022–2026, within five years of the 2026-09-15 writing date. The paper list is in the repository-root `README.md`; `DLCA/README.md` documents the cellular-automata notebooks. The 21 supplied PDFs were indexed and extracted without altering the originals. Their hashes and extraction paths are in `literature/local_corpus_20260915.json`.

| Citation | Year / source status used | Role in the argument | Verification source |
|---|---|---|---|
| Deep Differentiable Logic Gate Networks | 2022, NeurIPS | Fixed predecessor graphs and differentiable gate functions; original reported dense baseline | Supplied PDF, especially Table 9; [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html) |
| Convolutional Differentiable Logic Gate Networks | 2024, NeurIPS | Shared logic trees, spatial structure, and reported S/M references | Supplied PDF; [primary record](https://arxiv.org/abs/2411.04732) |
| A Method for Optimizing Connections in Differentiable Logic Gate Networks | 2025, preprint | Candidate-predecessor distributions as a learned alternative | Supplied PDF abstract/method; [primary record](https://arxiv.org/abs/2507.06173), checked 2026-09-15 |
| LILogic Net | Archived v2, 2026, preprint | Sparse Top-K routing and basis projection; local resource comparison and separate reported values | Supplied v2 PDF, especially p. 14; [versioned record](https://arxiv.org/abs/2511.12340v2), checked 2026-09-15 |
| Mind the Gap | 2025, NeurIPS | Gumbel/straight-through training addresses hardening rather than fixed graph design | Supplied v2 PDF; [primary record](https://arxiv.org/abs/2506.07500v2), acceptance and author metadata checked 2026-09-15 |
| Light Differentiable Logic Gate Networks | 2025, preprint | Gate reparameterization as a separate design axis | Supplied PDF; [primary record](https://arxiv.org/abs/2510.03250) |
| WARP Logic Neural Networks | 2026, preprint | Walsh–Hadamard gate representation; distinguishes adapted local controls from published claims | Supplied PDF; [primary record](https://arxiv.org/abs/2602.03527), checked 2026-09-15 |
| Fitting Multilinear Polynomials for Logic Gate Networks | 2026, preprint | Redundant directions in soft gate mixtures motivate an orthogonal gate-optimization line | Supplied PDF abstract/method; [primary record](https://arxiv.org/abs/2605.08657), checked 2026-09-15 |
| BitLogic | 2026, TMLR | Encoding, fan-in, connectivity, and training must be separated when attributing gains | Supplied PDF header and method. [OpenReview](https://openreview.net/forum?id=ZbsSZAfDod) returned a browser challenge on this check; publication metadata comes from the supplied PDF, not an inferred web result. |
| From MNIST to ImageNet | 2025, preprint | Output aggregation and temperature bound class-count transfer claims | Supplied PDF; [primary record](https://arxiv.org/abs/2509.25933), author/year metadata checked 2026-09-15 |
| Two-Stage Unit Tying | 2026, ICML, PMLR 306 | Post-training functional simplification differs from initial graph construction | Supplied PDF p. 1 explicitly identifies the proceedings and year; [author-supplied source link](https://openreview.net/pdf/bcb00a589c7c9b97b68e4e13f2fc1696b46557e3.pdf) |

The literature section synthesizes these design choices. It does not claim priority over every structured graph design, repeat unverified headline speedups, or assert successful reproduction of BitLogic. The older MNIST, Fashion-MNIST, CIFAR, and CIFAR-10.1 references identify datasets in the evaluation; they are not related-work comparators subject to the five-year window.

Published accuracy and training-time cells retain the versions and locations in `reported_sources.md`. No reported GPU-memory number was invented. The local poor BitLogic-related reproductions remain negative evidence in the atlas and cannot justify superiority over the published method.

The [DATE 2027 call for papers](https://www.date-conference.com/call-for-papers) was checked on 2026-09-15: six content pages plus an optional references-only seventh page, anonymous review, and at least 10-point Times-compatible text. The main PDF follows that format; the evidence atlas is a separate internal artifact, with no assumption that it is admissible as a conference supplement.
