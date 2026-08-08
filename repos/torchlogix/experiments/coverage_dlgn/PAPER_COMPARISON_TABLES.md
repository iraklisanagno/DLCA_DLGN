# CoverageDLGN paper comparison tables and experiment plan

**Created:** July 24, 2026  
**Status:** active experiment ledger; populate in place as matched runs finish

This is the canonical working document for the accuracy tables planned for the
CoverageDLGN paper. It distinguishes results produced by our controlled
TorchLogix protocol from numbers reported under other papers' protocols.

## Notation and comparison rules

Every accuracy value must carry one of these provenance labels directly in
its table cell:

- `[TRIED]` is a valid local exploratory, pilot, short-schedule, or superseded
  result that is not eligible as the final comparison value.
- `[TRIED-ONE-SEED]` is a completed local protocol with one seed; it is useful
  for feasibility and paired diagnostics but has no uncertainty estimate and
  is not eligible as a final statistical claim.
- `[REPRODUCED]` is a local result from a paper-faithful implementation and
  protocol.
- `[ADAPTED]` is a local result from a published method adapted to our common
  target architecture or protocol.
- `[OUR-FINAL]` is a frozen final CoverageDLGN result.
- `[REPORTED]` is copied from the cited paper and was not produced locally.
- `[PENDING]` means that the planned local experiment has not finished.
- `[N/A]` means that no valid value exists for that side of the comparison.

`A / R` columns retain the achieved-local/reported-paper ordering, but the
labels are mandatory even when that ordering appears obvious. A local failed
or interrupted attempt is recorded in the experiment log and never receives
an accuracy value. Each `[REPRODUCED]`, `[ADAPTED]`, or `[OUR-FINAL]` value
must point to its machine-readable run summary when populated. Each
`[REPORTED]` value must point to a PDF table or figure in the provenance
ledger.

- A reported architecture or protocol that is not an exact match is context,
  not evidence that one method beats another.
- `[N/A]` on the reported side means that no paper-reported result exists for
  that method and cell.
- Gate count means total deployed logic gates or, for convolutional networks,
  spatially instantiated gate operations. It never means layer width.
- A primary improvement claim may compare only `A` values produced with the
  same data split, encoding, augmentation, gate budget, training effort,
  checkpoint-selection rule, and hardened evaluator.
- Final central cells report mean, standard deviation, and paired 95%
  confidence intervals. Reported values retain the statistic used by their
  source paper.
- Each dataset--architecture cell contains one CoverageDLGN row: the best
  configuration selected by hardened validation accuracy for that cell.
  Alternative configurations remain in the experiment history, not the paper
  table.
- S/M/L labels are local to an architecture family. Gate counts are always
  printed because size labels are not comparable across papers.
- Bold type in the final manuscript will identify the best protocol-matched
  `A` result, never the largest `R-only` value.

## Table 1: Dense MNIST and Fashion-MNIST

Target architecture: six layers, 48K total rank-2 gates, and 768K raw gate
logits during training.

| Method | Matched gates | MNIST A / R | Fashion-MNIST A / R | Match status |
|---|---:|---:|---:|---|
| Deep DLGN fixed random | 48K | **[REPRODUCED] 97.090 +/- 0.180% (n=5) / [REPORTED] 97.69%** | **[REPRODUCED] 86.308 +/- 0.186% (n=5) / [REPORTED] 87.17%** | Exact architecture; sources: `summary/table1_mnist_final.json`, `summary/table1_fashion_final.json` |
| Mommen, \(N_c=16\) | 48K | [ADAPTED] 98.084 +/- 0.066% (n=5) / [REPORTED] 98.14% `[12K, nonmatched]` | [ADAPTED] 87.260 +/- 0.282% (n=3) / [REPORTED] 87.16% `[8K, nonmatched]` | Adapted to 48K; Fashion selected \(N_c=8\), depth 3; final sources as above |
| LILogicNet Top-32 | 48K | [ADAPTED] 98.124 +/- 0.029% (n=5) / [REPORTED] 98.95 +/- 0.09% `[32K, nonmatched]` | [ADAPTED] 88.437 +/- 0.159% (n=3) / [REPORTED] 90.26 +/- 0.11% `[64K, nonmatched]` | Adapted to 48K; Fashion Top-32, depth 2, tau 30; final sources as above |
| BitLogic best-of-space | 48K | [ADAPTED] 98.204 +/- 0.042% (n=5) / [REPORTED] 97.84 +/- 0.04% `[128K total, nonmatched]` | [ADAPTED] 89.740 +/- 0.243% (n=3) / [REPORTED] 89.16 +/- 0.08% `[128K total, nonmatched]` | Adapted to 48K; final sources as above |
| **CoverageDLGN** | **48K** | **[OUR-FINAL] 97.500 +/- 0.099% (n=5) / [N/A]** | **[OUR-FINAL] 87.102 +/- 0.357% (n=5) / [N/A]** | Exact target; final sources as above |

Mommen \(N_c=16\) adds approximately 32 routing logits per gate during
training but retains one hard predecessor per gate input after training.
The selected Fashion-MNIST \(N_c=8\) variant uses 1.536M total trainable
parameters, including 768K training-only routing parameters; its three final
seeds averaged 24.04 minutes and peaked at 0.498 GiB allocated GPU memory.
The selected Fashion-MNIST LILogicNet variant uses 3.84M total trainable
parameters, including 3.072M training-only routing parameters; its three
final seeds averaged 70.84 minutes and peaked at 4.676 GiB.
The adapted Fashion-MNIST BitLogic variant also uses 3.84M total trainable
parameters, including 3.072M training-only routing parameters; its three
final seeds averaged 88.94 minutes and peaked at 3.557 GiB.
On the locked MNIST test set, CoverageDLGN improves its paired fixed-random
baseline by 0.410 percentage points (95% Student-t confidence interval
[+0.108, +0.712] pp). All five methods were evaluated once using the
checkpoint selected solely by hardened validation accuracy.
On the locked Fashion-MNIST test set, CoverageDLGN improves its paired
fixed-random baseline by 0.794 percentage points (95% Student-t confidence
interval [+0.471, +1.117] pp). The selected checkpoint for every method was
again determined solely from hardened validation accuracy before the
exactly-once test evaluation.

## Table 2: Dense CIFAR-10 S/M/L

| Architecture | Target gates | Raw training parameters | Method | A / R accuracy | Reported configuration |
|---|---:|---:|---|---:|---|
| S: 4 x 12K | 48K | 0.768M | Deep DLGN random | **[REPRODUCED] 49.056 +/- 0.356% test (n=5) / [REPORTED] 51.27%** | Exact 48K architecture; one-time held-out test; source: `summary/paper_cifar10_semantic_v3.json` |
|  |  | 1.536M | Mommen learned connectivity | **[ADAPTED] 50.950 +/- 0.244% (n=3) / [N/A]** | Exact-48K \(N_c=8\) adaptation; one-time held-out test; source: `summary/table2_s_comparator_final.json` |
|  |  | 3.840M | LILogicNet | **[ADAPTED] 50.743 +/- 0.574% (n=3) / [REPORTED] 55.11%** | Exact-48K Top-32 adaptation; one-time held-out test; reported value uses 8K nonmatched gates; local source as above |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates under the BitLogic protocol, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] 52.358 +/- 0.282% test (n=5) / [N/A]** | Exact 48K target; +3.302 pp paired test gain, 95% CI [+2.767, +3.837] |
| M: 4 x 128K | 512K | 8.192M | Deep DLGN random | **[REPRODUCED] 54.028% (n=5) / [REPORTED] 57.39%** | Exact 512K architecture; one-time held-out test |
|  |  | 16.384M | Mommen learned connectivity | **[TRIED] 54.420% test (n=1) / [N/A]** | Exact-512K \(N_c=8\) adaptation; conditional policy stops at one seed; one-time held-out test |
|  |  |  | LILogicNet | [PENDING] / [REPORTED] 57.66 +/- 0.17% | 64K gates, nonmatched |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] 58.284% (n=5) / [N/A]** | Exact 512K target; +4.256 pp paired test gain |
| L: 5 x 256K | 1.28M | 20.48M | Deep DLGN random | **[REPRODUCED] 55.960 +/- 0.251% test (n=5) / [REPORTED] 60.78%** | Exact 1.28M architecture; one-time held-out test; source: `summary/table2_l_final.json` |
|  |  | 40.960M | Mommen learned connectivity | **[TRIED] 54.340% test (n=1) / [N/A]** | Exact-1.28M \(N_c=8\) adaptation; conditional policy stops at one seed; one-time held-out test |
|  |  |  | LILogicNet-L | [PENDING] / [REPORTED] 60.98 +/- 0.19% | 256K gates, nonmatched |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] raw V3 swap-0.50 61.020 +/- 0.336% test (n=5) / [N/A]** | Exact 1.28M target; +5.060 pp paired, 95% CI [+4.555, +5.565]; one-time held-out test |

The earlier 49.692% random and 53.116% CoverageDLGN S values are mean best
hardened **validation** accuracies, not held-out test accuracies. They remain
in `summary/paper_cifar10_semantic_v3.json` and the experiment history, but
the primary S cells above now use the exactly-once test means of 49.056% and
52.358%. This correction changes the paired gain from +3.424 validation pp to
+3.302 test pp; it does not change the selected checkpoints or method.

The one-seed 5K compression screen is complete; these are validation
selection values, not held-out test or paper-final values:

The dense L one-seed 5K screen selected raw V3 swap-0.50 for the primary
topology-only comparison: 58.780% versus 53.040% raw random (+5.740 pp).
V3+WARP reached 59.080%, but it advances only with a newly generated matched
random-topology+WARP control and remains a secondary combined configuration.
Light reached 36.020% and was rejected. Source:
`summary/table2_l_screen.json`.

At the paired 20K stage, raw V3 reached 60.907% versus 55.627% raw random
(+5.280 pp, 95% Student-t CI [+3.769, +6.791], all three seeds positive).
V3+WARP reached 59.813% versus 54.207% matched random+WARP (+5.607 pp,
95% CI [+4.769, +6.444]). Raw V3 is therefore frozen for the primary
five-seed final rather than selecting the lower-accuracy combined variant.
Source: `summary/table2_l_selection.json`.

The full 108K five-seed validation confirms the primary result: raw V3
reached 61.748% versus 56.468% raw random. Every paired difference is
positive; the mean gain is +5.280 pp with a 95% Student-t interval of
[+4.843, +5.717]. The held-out test remained locked until the one-seed L
Mommen comparator was completed and all validation checkpoints were committed.
The completed one-seed Mommen adaptation reached 54.340% validation in
13.39 hours, below both fixed random and V3, while adding 20.48M
training-only routing parameters. It is retained as `[TRIED, n=1]` and is not
promoted. Source: `summary/table2_l_mommen_final.json`.

After every L validation choice was committed, all 11 frozen checkpoints were
evaluated exactly once on held-out test. Raw V3 reached 61.020% versus
55.960% random. All five paired test gains are positive; the mean is
+5.060 pp with a 95% Student-t interval of [+4.555, +5.565]. Mommen reached
54.340% for its single adapted seed. These L checkpoints will not be queried
on test again.

| Target gates | Random screen | Best CoverageDLGN screen | Paired 20K selection | Full 108K validation | Held-out test |
|---:|---:|---:|---|---|---|
| 128K | [TRIED] 49.780% | [TRIED] 54.300% (raw, pool 4) | [TRIED] raw swap 0.50: 54.853% vs random 50.000% (n=3) | [OUR-FINAL] 55.140% vs [REPRODUCED] 50.760% (n=5, +4.380 pp) | **[OUR-FINAL] 53.910% vs [REPRODUCED] 49.748% (n=5, +4.162 pp, 95% CI [+3.759, +4.565])** |
| 256K | [TRIED] 51.600% | [TRIED] 55.940% (three-way raw tie) | [TRIED] raw incumbent: 57.513% vs random 52.567% (n=3) | **[TRIED] 57.800% vs 53.073% (n=3, +4.727 pp, 95% CI [+3.120, +6.333])** | **[OUR-FINAL] 56.903 +/- 0.134% vs [REPRODUCED] 52.253 +/- 0.058% (n=3, +4.650 pp, 95% CI [+4.174, +5.126])** |
| 384K | [TRIED] 52.280% | [TRIED] 57.520% (WARP) | [TRIED] raw incumbent: 58.980% vs random 54.400% (n=3) | **[TRIED] 59.313% vs 54.920% (n=3, +4.393 pp, 95% CI [+3.184, +5.603])** | **[OUR-FINAL] 58.143 +/- 0.153% vs [REPRODUCED] 53.657 +/- 0.328% (n=3, +4.487 pp, 95% CI [+3.515, +5.458])** |

The frozen 256K and 384K checkpoints were evaluated exactly once on held-out
test on July 29, 2026. All 12 evaluations completed successfully; these
checkpoints are now closed to further test queries. Machine-readable source:
`summary/table2_cifar10_compression_remaining_test.json`.

### CIFAR-10 M mechanism ablation

The paired three-seed, 20K component study reused the existing random and full
V3 controls and trained only the two missing arms:

| Component arm | Validation accuracy | Paired incremental effect |
|---|---:|---:|
| Fixed random | 54.820 +/- 0.530% | reference |
| Balanced butterfly fan-out | 58.980 +/- 0.548% | +4.160 pp over random, 95% CI [+3.988, +4.332] |
| Semantic first layer, no ancestry swaps | 59.253 +/- 0.153% | +0.273 pp, 95% CI [-0.780, +1.326] |
| Full frozen V3 | 59.293 +/- 0.214% | +0.040 pp from ancestry swaps, 95% CI [-0.434, +0.514] |

Full V3 remains +4.473 pp over random (95% CI [+3.624, +5.323]), but balanced
fan-out accounts for most of that gain at this coordinate. A separate
task-aware one-shot extension reached 59.093 +/- 0.234%: +4.273 pp over
random but -0.200 pp versus V3. It failed the frozen promotion gate and was
not evaluated on held-out test. Sources:
`summary/cifar10_medium_v3_components.json` and
`summary/cifar10_medium_task_aware.json`.

The screen and selection machine-readable sources are
`summary/table2_cifar10_compression_screen.json` and
`summary/table2_cifar10_compression_selection.json`. The 20K stage is
complete; its values select topology settings only. The held-out CIFAR-10
test set remains locked while the winners receive full matched training.

The crossing was selected using the predeclared three-seed result: 128K
CoverageDLGN reached 54.927% validation, only 0.157 pp below the existing
512K random reference of 55.084%, satisfying the frozen 0.3 pp
noninferiority rule. Its subsequent five-seed validation result is 55.140%.
The frozen checkpoints were then evaluated exactly once on the held-out test.

## Table 3: Convolutional CIFAR-10 S/M/L

Paper-faithful S and M both use 2-bit RGB precision represented by three
thermometer thresholds per channel, or nine Boolean input channels. They use
the same four depth-3 convolutional-stage pattern, raw LUT parameterization,
group size 2, and classifier pattern. S uses `k_num=32`, tau 20; M uses
`k_num=256`, tau 40. The WARP-style study below is a separate six-channel
protocol and is never pooled with these rows.

The cost column prints three definitions when available: the source paper's
approximate gate-operation value, locally learned LUT units, and locally
spatially instantiated gate applications. These definitions are not
interchangeable.

| Architecture | Input | Cost definitions | Method | Local achieved accuracy | Paper-reported accuracy | Provenance / match status |
|---|---:|---|---|---:|---:|---|
| LogicTreeNet-S | 3 thresholds/RGB = 9 channels | paper ~0.40M; 83,552 LUTs; 874,496 spatial applications | Original fixed routing | [TRIED-SELECTION] 56.864% best hard validation (n=5); [TRIED-HISTORICAL-TEST] 56.140% hard test (n=3) | **[REPORTED] 60.38% test** | Exact architecture; 20K local schedule; five-seed cohort is validation-only |
|  |  | ~0.70 x source gates | Two-stage unit tying, 30% | [N/A] | [REPORTED] 56.70 +/- 0.08% validation | Exact S architecture, different gate count and validation metric |
|  |  | 0.57M reported | Conv. TTNet-S | [N/A] | [REPORTED] 50.10% | Different truth-table architecture |
|  |  | same declared S budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared S budget | WARP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared S budget | **CoverageDLGN-Channel (frozen V4)** | [TRIED-SELECTION] 57.448% validation (n=5); [TRIED-HISTORICAL-TEST] 56.367% test (n=3) | [N/A] | +0.584 validation pp, 95% CI [-0.335, +1.503]; historical test +0.227 pp, CI crosses zero |
|  |  | same declared S budget | **Unified semantic degree-balanced (U1)** | **[TRIED-STOPPED] 57.624% validation (n=5)** | [N/A] | +0.760 pp, 95% CI [-0.700, +2.220], wins 4/5; no test query; failed +1 pp gate |
|  |  | same declared S budget | Channel-spatial leaf pairing | [TRIED-STOPPED] 57.033% validation (n=3) | [N/A] | -0.153 pp vs V4; failed both promotion gates |
| LogicTreeNet-M | 3 thresholds/RGB = 9 channels | paper ~3.08M; 668,416 LUTs; 6,995,968 spatial applications | Original fixed routing | **[TRIED-ONE-SEED] 70.68% best hard validation; 69.57% hard test** | **[REPORTED] 71.01% test** | Exact architecture; best-validation checkpoint from matched 200K run |
|  |  | ~0.70 x source gates | Two-stage unit tying, 30% | [N/A] | [REPORTED] 70.77 +/- 0.07% validation | Exact M architecture, different gate count and validation metric |
|  |  | approximately 3.08M reported | Scalability-boundaries CDLGN-M | [N/A] | [REPORTED] 65.23% | Minimally modified M protocol |
|  |  | 189M reported | Conv. TTNet-L | [N/A] | [REPORTED] 70.75% | Different, much larger architecture |
|  |  | same declared M budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared M budget | WARP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared M budget | **CoverageDLGN-Channel (frozen V4)** | **[TRIED-ONE-SEED] 71.26% best hard validation; 69.96% hard test** | [N/A] | +0.58 validation pp and +0.39 test pp; V4 led 97/100 validation evaluations |
| LogicTreeNet-L | 5-bit paper input | paper ~28.9M | Original fixed routing | [PENDING] | **[REPORTED] 84.99% test** | Exact replication requires 5-bit input and teacher |
|  |  | 189M reported | Conv. TTNet-L | [N/A] | [REPORTED] 70.75% | Different architecture |
|  |  | M architecture | Scalability-boundaries CDLGN-M | [N/A] | [REPORTED] 65.23% | Not LogicTreeNet-L |
|  |  | same declared L budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | New L adaptation |
|  |  | same declared L budget | WARP-LogicTreeNet | [PENDING] | [N/A] | New L adaptation |
|  |  | same declared L budget | **CoverageDLGN-Channel** | [PENDING] | [N/A] | Requires a paper-faithful L baseline first |

### WARP-style CIFAR-10 Medium compatibility study

This is a separate reconstruction of Figure 4 in the WARP paper. It uses the
public TorchLogix `ClgnCifar10Medium` configuration with two thresholds per
RGB channel (six Boolean input channels), raw rank-2 gates, no augmentation,
and 30K updates. It is not the exact nine-channel LogicTreeNet-M architecture
in the rows above and does not reproduce its reported 71.01% test result.
The WARP paper provides three-seed validation curves rather than exact table
values; the reported endpoints below are approximate readings of its 50K-step
plot. Local results are seed 0 only and the held-out test set was not queried.

| Method | Threshold/routing setting | Local best hard validation | Approx. WARP Figure 4 endpoint | Status |
|---|---|---:|---:|---|
| WARP fixed uniform | Public `random-unique`; fixed uniform thresholds | **[TRIED-PARTIAL] 65.35%** | [REPORTED-PLOT] approximately 64.0% | One seed at 30K |
| WARP fixed distributive | Public `random-unique`; fixed distributive thresholds | **[TRIED-PARTIAL] 66.12%** | [REPORTED-PLOT] approximately 65.0% | One seed at 30K |
| WARP learnable | Public `random-unique`; learnable thresholds | **[TRIED-PARTIAL] 65.88%** | [REPORTED-PLOT] approximately 66.6% | One seed at 30K |
| Matched random | V4 pilot's random sampler; fixed uniform thresholds | **[TRIED-PARTIAL] 64.58%** | [N/A] | Seed-0 control for Legacy V4 |
| **CoverageDLGN Legacy V4** | Frozen `semantic_channel_hybrid`; fixed uniform thresholds | **[TRIED-PARTIAL] 66.23%** | [N/A] | Seed 0; **+1.65 pp** over matched random |

The fixed-uniform and fixed-distributive jobs were originally launched with
50K updates and interrupted just after the common 30K validation boundary,
after both had reached the approximate plotted endpoints. No later validation
was evaluated, and 30K was frozen before observing the Medium Legacy V4
result. These results therefore show that the reconstructed WARP accuracies
are reached within 30K; they are not an exact 50K reproduction. The direct
V4 claim uses only its matched-random row because the public WARP arms use a
different `random-unique` routing sampler. Seeds 1 and 2 remain pending.

Machine-readable source:
`summary/warp_fig4_cifar10_medium.json`.

### Convolutional circuit and deployment accounting

All rows below use synthetic thresholded Boolean inputs and therefore make no
validation or test-set query. Functional equivalence passed from hardened
PyTorch class output through export mode, Python `Circuit`, circuit
simplification, and, for S, compiled C. Declared LUT/spatial budgets are equal
within every matched architecture; simplified IR nodes are checkpoint-dependent
because constant, wire, and duplicate functions can be removed after training.

| Protocol | Method | Simplified IR nodes | Peak export RSS | Compiled CPU, batch 128 | Interpretation |
|---|---|---:|---:|---:|---|
| Paper-faithful S, 9 channels | Fixed random | 197,851 | 1.071 GiB | 3.057 ms / 41.87K examples/s | Reference |
|  | Frozen V4 | 202,827 (+2.52%) | 1.073 GiB | 3.076 ms / 41.61K examples/s | +0.62% latency; no speed claim |
|  | U1 | 214,883 (+8.61%) | 1.074 GiB | 3.095 ms / 41.35K examples/s | +1.26% latency; no speed claim |
| Paper-faithful M, 9 channels | Fixed random | 1,676,852 | 6.103 GiB | [N/A] | Trace/equivalence only |
|  | Frozen V4 | 1,702,350 (+1.52%) | 6.092 GiB | [N/A] | Trace/equivalence only |
| WARP-style M, 6 channels | Matched random | 1,101,364 | 4.070 GiB | [N/A] | Separate protocol; trace/equivalence only |
|  | Legacy V4 | 1,129,547 (+2.56%) | 4.077 GiB | [N/A] | Separate protocol; trace/equivalence only |

The S circuits were compiled with bit packing at 64 examples per machine word
and `gcc -O0`; timings include Boolean input packing. An initial unbounded
`gcc -O1` random-S attempt was stopped after approximately 8.75 minutes. The
bounded `-O0` compiles completed in 37.75, 39.75, and 44.72 seconds for
random, V4, and U1. Fully unrolled M compilation was not attempted after this
feasibility result. This backend result does not imply that optimized FPGA or
specialized bit-packed kernels have the same limitation.

Machine-readable sources:

- `summary/convolutional_deployment.json` and CSV export;
- `summary/deployment/*.json` per-run records;
- `summary/deployment/compile_attempt_history.json`.

## Table 4: Dense CIFAR-100 S/M/L

The initial compact CIFAR-100 ladder follows BitLogic's two-layer common
protocol. The paper reports width per layer, so total gate count is twice its
reported width. The S coordinate has 8,000 first-layer input slots for 9,216
encoded inputs; paired local random and CoverageDLGN runs therefore use the
same opt-in, uniform maximal-subset constraint without changing the two-by-4K
paper architecture.

| Architecture | Total gates | Fan-in | Method | A / R accuracy | Source status |
|---|---:|---:|---|---:|---|
| S: 2 x 4K | 8K | 2 | DiffLogic | [PENDING] / [REPORTED] 7.49 +/- 0.21% | BitLogic common-protocol reproduction |
|  | 8K | 2 | WARP-LUT | [PENDING] / [REPORTED] 7.00 +/- 0.04% | BitLogic reproduction; C100 extrapolation |
|  | 8K | 2 | LILogicNet | [PENDING] / [REPORTED] 7.63 +/- 0.01% | BitLogic reproduction; C100 extrapolation |
|  | 8K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 10.19 +/- 0.06%** | Exact BitLogic configuration |
|  | **8K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates; lower fan-in |
| M: 2 x 16K | 32K | 2 | DiffLogic | [PENDING] / [REPORTED] 10.61 +/- 0.08% | BitLogic reproduction |
|  | 32K | 2 | WARP-LUT | [PENDING] / [REPORTED] 10.46 +/- 0.00% | BitLogic reproduction |
|  | 32K | 2 | LILogicNet | [PENDING] / [REPORTED] 10.62 +/- 0.12% | BitLogic reproduction |
|  | 32K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 14.06 +/- 0.04%** | Exact BitLogic configuration |
|  | **32K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates |
| L: 2 x 64K | 128K | 2 | DiffLogic | [PENDING] / [REPORTED] 14.64 +/- 0.09% | BitLogic reproduction |
|  | 128K | 2 | WARP-LUT | [PENDING] / [REPORTED] 14.43% | Single seed in BitLogic |
|  | 128K | 2 | LILogicNet | [PENDING] / [REPORTED] 14.54 +/- 0.04% | BitLogic reproduction |
|  | 128K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 18.82 +/- 0.09%** | Exact BitLogic configuration |
|  | **128K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates |

Live validation status (not table-final accuracy): the CIFAR-100 S one-seed
5K screen was followed by the paired three-seed 20K confirmation. Fixed
random reached `[TRIED]` 8.780 +/- 0.381%; the best frozen V3 control
(`swap_fraction=0.125`) reached `[TRIED]` 7.940 +/- 0.302%. The paired gap
is -0.840 pp, 95% CI [-1.404, -0.276], so S did not promote to full effort.
M receives only a diagnostic screen because its first layer is no longer
undersubscribed. That screen reached `[TRIED]` 9.880% for random and
`[TRIED]` 10.380% for frozen V3 `swap_fraction=0.5` (+0.500 pp), so M
advanced to paired three-seed 20K confirmation. The confirmation reached
`[TRIED]` 10.060 +/- 0.220% for random and `[TRIED]` 9.707 +/- 0.076% for
V3, a paired -0.353 pp (95% CI [-1.058, +0.352]). M did not promote; L is
reported-only and no CIFAR-100 held-out test was used.

Additional reported-only dense CIFAR-100 references:

| Method | Accuracy | Gates or parameters | Status |
|---|---:|---:|---|
| Scalability-boundaries dense DLGN | **[REPRODUCED, topology-adapted] 20.677 +/- 0.522% test (n=3)** / [REPORTED] 22.54 +/- 0.26% | 6 x 64K = 384K gates; 6.144M trainable LUT parameters | Exact architecture/schedule; independent TorchLogix routing differs from canonical difflogic generator; see `CIFAR100_BASELINE_AUDIT.md` |
| **CoverageDLGN V3** | **[OUR-FINAL] 21.010 +/- 0.131% test (n=3)** / [N/A] | **6 x 64K = 384K gates; 6.144M trainable LUT parameters** | Frozen V3 `swap_fraction=0.125`; paired +0.333 pp, 95% CI [-0.816, +1.483]; one-time held-out test |
| V3 + class-conditional head | [TRIED] 21.593 +/- 0.133% validation at 20K / [N/A] | 6 x 64K = 384K gates; 6.144M trainable LUT parameters | Validation-only negative result; +0.013 pp over V3 and +0.553 pp over random; no held-out test |
| Multilinear Soft-Mix | [TRIED] 11.680% validation at 5K (random); best frozen V3 11.060% / [REPORTED] 27.92 +/- 0.43% | 6 x 256K = 1.536M gates; 24.576M trainable LUT parameters | Exact 31-threshold architecture; screen only, not comparable to the reported full result |
| Multilinear-CovJac | [REPORTED] 28.37 +/- 0.22% | 1.536M gates; 4 parameters/gate | Reported only |
| Multilinear-CovJac large | [REPORTED] 32.72 +/- 0.09% | 6 x 1.28M = 7.68M gates; 4 parameters/gate | Reported only |

The deep 6-by-64K screen selected the existing V3
`swap_fraction=0.125` control at +0.640 pp. Its three-seed 20K confirmation
was +0.540 pp, with all three paired point estimates nonnegative, and thus
authorized the full paper-length run. Full validation was 21.577 +/- 0.067%
for V3 versus 20.943 +/- 0.311% for random (+0.633 pp). The locked
best-validation checkpoints were then evaluated once on the held-out test
set, producing the table-final values above. The local random result is
1.863 pp below the reported scalability result, so it is marked
`[REPRODUCED]` as an exact-architecture local reproduction, not an exact
numerical replication.

For 6-by-256K, fixed random reached 11.680% validation at 5K. Frozen V3
swap fractions 0.125, 0.25, and 0.5 reached 10.320%, 10.180%, and 11.060%.
Because every existing V3 control was negative, the predefined rule stopped
this branch before multi-seed confirmation, full training, or held-out test.
The reported Soft-Mix/CovJac accuracies remain reported-only full-effort
references and must not be compared directly with this 5K diagnostic.

Controlled 384K-gate CIFAR-100 depth ablation (seed 0, 20K steps,
validation-only):

| Architecture | Random | Frozen CoverageDLGN V3 | Paired gain | Decision |
|---|---:|---:|---:|---|
| 3 x 128K | `[TRIED]` 21.080% | `[TRIED]` 21.860% | +0.780 pp | Positive, below predeclared +1 pp confirmation threshold |
| 6 x 64K | `[TRIED]` 21.040% | `[TRIED]` 21.580% | +0.540 pp | Earlier three-seed 20K mean, shown as the reference rather than a seed-0 cell |
| 12 x 32K | `[TRIED]` 1.180% | `[TRIED]` 1.180% | +0.000 pp | Chance-level optimization failure |
| 24 x 16K | `[TRIED]` 1.200% | `[TRIED]` 1.200% | +0.000 pp | Chance-level optimization failure |

The 3-, 12-, and 24-layer cells have the same 384K gates, 6.144M trainable
LUT parameters, three-threshold encoding, temperature 10, split, batch,
optimizer, and effort. No cell crossed the frozen +1 pp seed-0 promotion
threshold, so no additional seeds or held-out tests were run. The reference
6-by-64K entry is a three-seed mean from its earlier confirmation and is not
used as though it were a matched seed-0 depth cell.

Topology diagnostics explain why added depth does not automatically help V3.
At 12 layers, the final gate already contains about 2,262 raw image sources
under random and 2,289 under V3, while both cover all 3,072 RGB spatial
sources globally. At 24 layers, every final gate contains all 3,072 sources
and cross-gate ancestry Jaccard is 1.0 for both methods. Thus ancestry
completely saturates while optimization collapses to chance.

A subsequent class-conditional final-layer refinement directly reduced
CIFAR-100's remaining per-class source-usage imbalance. Across three seeds,
mean source-usage CV fell from 0.25655 for V3 to 0.23475 at identical
deployment cost. Accuracy remained tied: 21.593% versus 21.580% V3
validation at 20K. This branch failed its frozen +2 pp over random and +1 pp
over V3 gates, so it has no full or test value and is excluded from the
paper-final accuracy claim.

## Table 5: Convolutional CIFAR-100 S/M/L

These are transferred LogicTreeNet scales. No local paper reports exact
CIFAR-100 S and L results. Estimated operation counts apply Light's tenfold
final-layer expansion for 100 classes to the rounded CIFAR-10 gate counts.

| Architecture | Estimated gate operations | Method | A / R accuracy | Status |
|---|---:|---|---:|---|
| C100 LogicTreeNet-S | approximately 0.49M | Original/Soft-Mix routing | [PENDING] / [N/A] | New S transfer |
|  | approximately 0.49M | Light/IWP | [PENDING] / [N/A] | No published C100-S result |
|  | approximately 0.49M | WARP | [PENDING] / [N/A] | No published C100-S result |
|  | **approximately 0.49M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | New matched experiment |
| C100 LogicTreeNet-M | approximately 3.82M | Original/Soft-Mix | [PENDING] / [REPORTED] 29.0 +/- 0.6% | R value is fivefold-depth M, not base M |
|  | approximately 5 x M depth | Light/IWP | [PENDING] / **[REPORTED] 38.2 +/- 0.3%** | Fivefold-depth M, nonmatched |
|  | M-derived | Scalability-boundaries CDLGN | [N/A] / [REPORTED] 30.96% | Minimally modified M |
|  | approximately 3.82M | WARP | [PENDING] / [N/A] | No published exact M result |
|  | **approximately 3.82M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | New matched experiment |
| C100 LogicTreeNet-L | approximately 34.8M | Original/Soft-Mix | [PENDING] / [N/A] | New L transfer |
|  | approximately 34.8M | Light/IWP | [PENDING] / [N/A] | No published C100-L result |
|  | approximately 34.8M | WARP | [PENDING] / [N/A] | No published C100-L result |
|  | **approximately 34.8M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | Requires teacher/input protocol decision |

Convolutional CIFAR-100 S/L therefore cannot yet contain three exact
paper-reported architecture matches. Any dense reported numbers shown beside
them in the manuscript must remain explicitly `R-only`.

## Reported-value provenance

- Deep DLGN: MNIST Table 4, CIFAR-10 Table 5, and architecture Table 6.
- Mommen connection optimization: MNIST and Fashion-MNIST Figures 3 and 4.
- LILogicNet: Tables 3 and 6.
- BitLogic: shared-protocol Table 6 and method mapping Table 8.
- Convolutional DLGN: CIFAR-10 Table 1 and architecture Appendix A.1.
- Two-stage unit tying: CIFAR-10(M) Table 1 and CIFAR-10(S) Table 7.
- Light DLGN: CIFAR-100 fivefold-depth comparison in Figure 8.
- Scalability-boundaries study: CIFAR-10/100 Tables 11 and 12.
- Multilinear-CovJac: cross-dataset Table 1 and scaling Table 15.

Before using a reported value in a manuscript, record its PDF filename, table
or figure number, metric definition, seed count, and whether its gate count is
width, learned gates, spatial operations, or synthesized gates.

## Experiment plan

The goal is one best CoverageDLGN result per dataset and architecture, not
multiple reporting variants. Hyperparameters may be selected independently
for every cell, but the implemented CoverageDLGN mechanism must not change.

### Frozen run-count policy

- Fixed random and CoverageDLGN use five full seeds on every promoted central
  cell.
- Locally reproduced or adapted paper-derived comparators use three full
  seeds on S/M cells. A paper-prescribed recipe is preferred; otherwise a
  one-seed short screen advances only one setting to the three long runs.
- On deep/L cells, an expensive locally adapted comparator initially receives
  one full seed and is labelled `[TRIED, n=1]`. It is promoted to three full
  seeds only when that run is competitive with the central fixed-routing
  methods or is scientifically necessary for a paper claim. Otherwise the
  one-seed local value is retained alongside the clearly labelled
  `[REPORTED]` paper value.
- The already-running MNIST Table 1 queue is the only locked exception: all
  five methods retain five final seeds. Fashion-MNIST uses five seeds for
  random/CoverageDLGN and three for Mommen/LILogicNet/BitLogic.
- Dense CIFAR-10 and CIFAR-100 run Mommen for three seeds at each feasible
  S scale. CIFAR-10 M uses one full Mommen seed first and promotes it to
  three only if competitive or scientifically necessary; the conditional
  one-to-three-seed policy above also applies to L. LILogicNet is run locally
  for three seeds on S and is reported-only on M/L. BitLogic and WARP-LUT use
  reported values. CIFAR-100 promotion decisions remain subject to its
  dataset-specific plan and timing feasibility.
- Promoted convolutional cells run fixed random and CoverageDLGN for five
  seeds and Light/IWP and WARP for three seeds. Mommen, LILogicNet, and
  BitLogic are not treated as convolutional comparators unless a separately
  justified generic extension is implemented.
- CIFAR-10 compression points use three paired random/CoverageDLGN seeds.
  Only the selected Pareto crossing is extended to five final seeds.
- Convolutional L and CIFAR-100 M/L obey the promotion/kill conditions below.
- Every numeric table value carries the provenance labels defined above.

### Selection protocol used for every CoverageDLGN cell

1. Rebuild the topology for the exact dataset, encoding, width, depth, and
   topology seed.
2. Search only existing method controls, such as candidate-pool size, swap or
   long-range fraction, novelty/overlap weights, and the supported raw,
   Light/IWP, or WARP gate parameterization. Record the selected
   parameterization in the table.
3. Do not introduce a new score term, dataset-coded branch, layer-specific
   exception, or topology mechanism during the search.
4. Do not select a favorable topology or training seed. Every candidate uses
   the same predeclared paired pilot seeds.
5. Use topology-only analysis to reject invalid, excessively slow, or
   degenerate configurations, but select accuracy configurations by mean
   hardened validation accuracy rather than maximum ancestry coverage.
6. Use a multi-fidelity search:
   - one-seed CUDA smoke and short screen for the broad candidate set;
   - three paired 20K validation seeds for the leading candidates;
   - freeze the best mean hardened-validation configuration; and
   - train the frozen winner for the final seed set and full schedule.
7. Keep the held-out test set locked until the winner and every comparator are
   frozen.
8. Preserve every attempted configuration, learning curve, failure, and
   selection decision in the experiment history. Only the winner populates
   the CoverageDLGN row.
9. Give reproduced competitors a documented validation search or their
   source paper's frozen recipe. Do not give CoverageDLGN more selection seeds
   or optimizer updates than a matched trainable comparator without reporting
   the difference.
10. After the Table 1 Fashion-MNIST cell, use a paper-prescribed fixed
    LILogicNet recipe when directly applicable. Otherwise use the short screen
    to choose one LILogicNet setting and run three paired 20K seeds only for
    that setting. Report this smaller comparator-search budget explicitly;
    do not multiply or reduce the trainable-parameter count based on the
    number of independent runs.

### Plan for Table 1: Dense MNIST and Fashion-MNIST

1. Audit the existing 48K random and CoverageDLGN artifacts; retain an
   achieved value only when its protocol matches the final table.
2. Implement and test Mommen \(N_c=8/16\), LILogicNet Top-32, and the matched
   BitLogic coordinates.
3. Use MNIST CUDA smokes to validate every new implementation.
4. Search CoverageDLGN independently on MNIST 48K and Fashion-MNIST 48K using
   the common selection protocol above.
5. Run the same pilot and selection effort for the reproduced comparators.
6. Retain five final seeds for every frozen MNIST row. On Fashion-MNIST,
   train five final random/CoverageDLGN seeds and three final
   Mommen/LILogicNet/BitLogic seeds.
7. Evaluate test once and replace the current CoverageDLGN values only when
   the newly selected configurations are protocol-complete.

**Completion condition:** each dataset has achieved results for fixed random,
at least three paper-derived comparators, and the single best CoverageDLGN
configuration.

### Plan for Table 2: Dense CIFAR-10 S/M/L and compression

1. Retain the existing S and M results as incumbent CoverageDLGN candidates.
2. Add and test 128K, 256K, and 384K four-layer compression architectures.
3. Search CoverageDLGN independently for S, 128K, 256K, 384K, M, and L.
4. Run fixed random/CoverageDLGN for five seeds. Run Mommen for three seeds
   on S; on M and L, start with one full seed and promote to three only when
   competitive or scientifically necessary. Run LILogicNet for three seeds
   on S; retain LILogicNet M/L, WARP-LUT, and BitLogic as explicitly
   reported-only values.
5. Use three paired 20K seeds for selection at every budget.
6. Run five full fixed-random/CoverageDLGN seeds for S, M, the selected
   compression crossing, and L after a timing/memory feasibility run.
7. Select the smallest CoverageDLGN budget that matches the larger random
   model within a predeclared 0.3-percentage-point non-inferiority margin.

**Completion condition:** S/M/L contain the five-seed local primary
comparison, the policy-compliant Mommen adaptation, and at least three
clearly labelled local or reported paper-derived comparisons. The table or
companion figure demonstrates the best observed accuracy--gate-count Pareto
frontier.

### Plan for Table 3: Convolutional CIFAR-10 S/M/L

1. Finish a full-schedule paper-faithful fixed-routing S baseline.
2. Search the unchanged convolutional CoverageDLGN mechanism independently
   for S, M, and L. V4 remains frozen; V5, coverage--reuse, and channel-spatial
   leaf pairing are separate negative methods. The unified no-swap U1
   candidate completed five seeds at +0.760 pp and four wins, but failed its
   predeclared +1 pp gate and is not promoted.
3. Compare raw, Light/IWP, and WARP parameterizations within the allowed
   CoverageDLGN search and record the winning choice.
4. Reproduce Light/IWP-LogicTreeNet and WARP-LogicTreeNet for three final
   seeds on each promoted scale.
5. Retain two-stage unit tying as a clearly labelled reported value unless a
   later paper need justifies local reproduction.
6. Run three paired 20K selection seeds, five final random/CoverageDLGN
   seeds, and three final Light/IWP/WARP seeds for S and M.
7. Start L only after reproducing its 5-bit input, preprocessing, teacher
   supervision, and gate accounting, followed by a timing/memory feasibility
   run.

**Promotion condition for L:** M must show a positive mean CoverageDLGN gain
over its matched fixed-routing baseline. Otherwise L remains reported-only.

**Current stop decision:** the U1 S gate failed, so its M, L, and
convolutional CIFAR-100 rows remain unrun. Restarting those scales requires a
separately justified, preregistered method rather than post-hoc continuation
of U1.

### Plan for Table 4: Dense CIFAR-100 S/M/L

1. Audit the BitLogic two-layer S/M/L common protocol and gate accounting.
2. Retain DiffLogic, WARP-LUT, and BitLogic best-of-space as reported values;
   run a three-seed local LILogicNet adaptation on S and retain its M/L
   reported values.
3. Search rank-2 CoverageDLGN independently at 8K, 32K, and 128K total gates.
4. Keep fan-in and operation count visible because BitLogic best-of-space is
   rank 4.
5. Use five final seeds for fixed random/CoverageDLGN and three for Mommen at
   each scale, subject to timing feasibility.
6. Retain the 384K and Multilinear-CovJac values as reported-only unless their
   exact architectures are separately reproduced.

**Completion condition:** all S/M/L rows contain a five-seed local primary
comparison, a three-seed Mommen adaptation, and at least three clearly
labelled local or reported paper-derived comparators.

### Plan for Table 5: Convolutional CIFAR-100 S/M/L

1. Finalize the 100-class output-head definition and replace estimated gate
   counts with exact learned-gate and spatial-operation counts.
2. Implement and smoke-test the transferred S architecture.
3. Search CoverageDLGN independently for S using random, raw/Light/WARP
   parameterization candidates and three paired 20K validation seeds.
4. Populate five-seed fixed-routing/CoverageDLGN rows and three-seed
   Light/IWP/WARP adapted comparator rows.
5. Promote M only after S produces a positive signal; promote L only after M.
6. For every promoted scale, run five final random/CoverageDLGN seeds, three
   final Light/IWP/WARP seeds, and one locked test evaluation after selection.
7. Leave unpromoted scales and nonmatching literature results explicitly
   reported-only.

**Kill condition:** if a scale fails to improve its matched fixed-routing
baseline after the allowed search, do not launch the next, substantially more
expensive scale.

### Supporting analyses after the five tables

1. On the selected dense CIFAR-10 central cell, compare random, butterfly,
   pure greedy coverage, and full CoverageDLGN.
2. Ablate overlap, fan-out preservation, multiscale semantic pairing, and raw
   source versus bit ancestry without using those ablations to rename or
   redefine the selected method.
3. Evaluate all frozen winners on held-out test exactly once.
4. Report mean, standard deviation, paired 95% intervals, learning curves,
   gates, training parameters, routing bits, construction time, peak GPU
   memory, hardened latency, and circuit-export equivalence.
5. Generate accuracy--gate-count and topology-metric--accuracy figures from
   machine-readable summaries.

## Immediate next work

1. Freeze the per-cell selection protocol and search budget in a
   machine-readable manifest.
2. Implement and smoke-test the Table 1 comparators on MNIST.
3. Search and populate Table 1 on MNIST and Fashion-MNIST.
4. Continue with the Table 2 dense CIFAR-10 compression and S/M/L searches.
5. Do not start convolutional CIFAR-10 L or any convolutional CIFAR-100 scale
   until the preceding promotion condition is satisfied.
