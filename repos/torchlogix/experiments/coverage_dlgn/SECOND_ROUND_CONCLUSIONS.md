# CoverageDLGN DATE second-round conclusions

## Outcome

The nine-step second round is complete. All 110 planned CUDA runs finished;
completed artifacts were skipped rather than rerun. The evidence supports a
concrete paper story: deterministic degree-balanced semantic connectivity
improves hardened accuracy without learned routing, additional gates, or
additional training effort. Frozen V3 is the strongest dense specialization;
the separately frozen U2 rule is the generic dense/convolutional formulation.

The strongest new convolutional result is paper-faithful nine-channel
LogicTreeNet-S U2: **60.630% hard test**, versus 57.370% fixed random,
58.930% legacy V4, and 58.800% U1 under the same seed, architecture, 350K
updates, validation selection, and circuit budget. This is **+3.260 pp over
random**, +1.700 pp over V4, +1.830 pp over U1, and 0.250 pp above the
paper-reported 60.38% LogicTreeNet-S test result. The full-resource cohort is
one seed; the corresponding 20K pilot has three seeds and gains +2.173 pp
over primary random with 3/3 wins and 95% CI [+1.647, +2.700].

## Nine-step disposition

| Step | Outcome | Evidence |
|---:|---|---|
| 1 | Corrected comparison scope and labels | BitLogic/WARP-LUT two-layer ladders are no longer presented as dense S/M/L reproductions; reported, adapted, reproduced, and our-result provenance is explicit. |
| 2 | Exact-depth comparators complete | 18 CUDA runs: Mommen, LILogicNet, and BitLogic on MNIST/Fashion, six layers x 8K gates, three seeds each. |
| 3 | Compression ladders complete | 48 CUDA runs across eight new MNIST/Fashion budgets; V3 is positive in all eight cells and wins 22/24 pairs. |
| 4 | Full convolutional S controls complete | Random, frozen V4, and U1 completed 350K updates with matched resources. |
| 5 | CIFAR-100 strengthened | Baseline recipe screen, three-seed 3 x 128K cohort, two allocation ablations, frozen one-time test, and depth failures documented. |
| 6 | Generic U2 implemented and audited | `semantic_multiscale_balanced` applies the same semantic ordering, degree-first deterministic multiscale matching, and zero-learned-routing rule to dense and convolutional networks. V3/V4/U1 remain frozen. |
| 7 | Five-coordinate U2 pilot complete | 15 CUDA runs; positive on MNIST, Fashion, dense CIFAR-10, and convolutional CIFAR-10; rejected on CIFAR-100. |
| 8 | Promotions and trade-offs complete | Three dense coordinates at full effort plus full 350K convolutional S, one-time held-out tests, curves, CUDA inference, and compiled circuit measurements. |
| 9 | Fallback disposition and consolidation complete | Calibrated fallback was correctly not run because convolutional U2 passed; all outcomes, failures, costs, and limitations are retained. |

## Main accuracy evidence

| Domain / architecture | Selected rule | Random hard test | CoverageDLGN hard test | Gain | Replication |
|---|---|---:|---:|---:|---|
| MNIST, dense 6 x 1.33K, 8K gates | U2 | 91.273 +/- 0.217% | **91.937 +/- 0.137%** | **+0.663 pp** | n=3, CI [+0.015, +1.311], 3/3 |
| Fashion-MNIST, dense 6 x 2.67K, 16K gates | V3 | 85.197 +/- 0.261% | **85.913 +/- 0.356%** | +0.717 pp | n=3, 3/3; CI crosses zero |
| CIFAR-10 dense S, 6 x 8K, 48K gates | V3 | 49.056 +/- 0.356% | **52.358 +/- 0.282%** | **+3.302 pp** | n=5 |
| CIFAR-10 dense M, 512K gates | V3 | 54.028% | **58.284%** | **+4.256 pp** | n=5 |
| CIFAR-10 dense L, 1.28M gates | V3 | 55.960 +/- 0.251% | **61.020 +/- 0.336%** | **+5.060 pp** | n=5 |
| CIFAR-100 dense, 3 x 128K | V3 | 20.923 +/- 0.352% | **21.467 +/- 0.410%** | +0.543 pp | n=3, 3/3; test CI crosses zero; validation gain +0.840 pp is significant |
| CIFAR-10 LogicTreeNet-S, 9 channels, 350K | U2 | 57.370% | **60.630%** | **+3.260 pp** | full n=1; supporting 20K n=3 pilot |

V3 also beats architecture-matched Mommen and LILogicNet on MNIST while
using fewer trainable parameters, less training time, and less GPU memory.
The matched six-layer BitLogic transfer converged to chance on both MNIST and
Fashion; it is a reproduced-negative architecture transfer, not a failed job.

## Cost and learning-efficiency trade-off

The full convolutional cohort is exact-cost matched: every method has 83,552
learned LUT functions, 874,496 spatial gate applications, 1,336,832 trainable
LUT parameters, zero trainable routing parameters, and 1,945,600 deployed
routing bits. U2 trains in 4.951 hours with 1.831 GiB peak GPU allocation,
versus 4.975 hours and 1.831 GiB for random. Offline topology construction is
1.372 seconds for U2 versus 0.217 seconds for random.

U2's hardened validation-curve mean is 59.562%, versus 57.233% random,
58.510% V4, and 58.344% U1. It reaches 59.5% at 34K updates; V4 reaches that
threshold at 212K, U1 at 252K, and random never reaches it by 350K.

Synthetic hardened CUDA inference at batch 128 is 6.835 ms for U2 and 6.852
ms for random, with 0.3462 GiB peak allocation for both. The difference is
measurement noise, so the claim is matched runtime rather than acceleration.
Compiled `gcc -O0` CPU circuit latency is 3.185 ms for U2 versus 3.230 ms for
random, also a snapshot rather than a speed claim. U2's simplified circuit is
262,260 IR nodes versus 252,936 random (**+3.686%**), and peak export RSS is
1.138 versus 1.114 GiB. This modest post-simplification-size increase is the
main observed cost of the accuracy gain. Circuit equivalence passed; energy
was not measured.

## Defensible DATE claim

The defensible contribution is not that one topology dominates every dataset.
It is that a systematic topology-design principle—semantic source ordering,
degree-balanced fixed routing, and deterministic multiscale pairing—provides:

- a strong dense specialization (V3), with gains that increase from CIFAR-10
  S to M to L at exact gate budgets;
- a single generic instantiation (U2) that is positive on compressed MNIST,
  compressed Fashion-MNIST, dense CIFAR-10, and convolutional CIFAR-10 without
  dataset-specific rule changes; and
- no learned-routing, deployed-routing, gate-count, optimizer-update, GPU
  memory, or measured-runtime penalty, apart from offline construction and a
  checkpoint-dependent +3.686% simplified-IR trade-off on full convolutional S.

The negative CIFAR-100 U2 result is part of the scope boundary: U2 is not
claimed to replace V3 everywhere. CIFAR-100's best retained matched result is
V3, whose 3 x 128K validation gain is significant but whose test confidence
interval crosses zero.

## Limitations before submission

- Full 350K LogicTreeNet-S accuracy/resource evidence has one seed per method;
  multi-seed support is currently the 20K pilot. A final paper should add at
  least two full seeds for random and U2 if compute permits.
- The local LogicTreeNet-S training protocol uses a 45K/5K train/validation
  split and 350K updates; it is architecture-faithful but not identical to
  every detail of the published 50K-training protocol. The reported 60.38%
  comparison must retain this qualifier.
- Dense CIFAR-100 gains are modest and absolute accuracy remains below stronger
  reported multilinear parameterizations; it supports a boundary analysis,
  not the headline accuracy claim.
- Simplified IR and CPU latency are checkpoint/backend snapshots. Energy and
  optimized hardware synthesis are not measured.
- LogicTreeNet-M/L full multi-seed confirmation is not included in this round
  because of their prohibitive training cost.

## Machine-readable sources

- `summary/second_round_status.{json,csv}` and grouped CSV;
- `summary/second_round_u2_pilot.json`;
- `summary/second_round_final_validation_freeze.json`;
- `summary/second_round_final_dense.json`;
- `summary/second_round_convolutional_validation_freeze.json`;
- `summary/second_round_convolutional_final.json`;
- `summary/second_round_convolutional_curves.{json,csv}`;
- `summary/second_round_convolutional_deployment.json`.
