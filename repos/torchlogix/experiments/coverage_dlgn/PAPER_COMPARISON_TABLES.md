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
- Training time is the arithmetic mean wall time of one completed training
  run unless a row is explicitly labelled one-seed or approximate. Peak GPU
  memory is the maximum PyTorch device allocation recorded in the local
  cohort, converted from bytes to GiB. Neither value includes hyperparameter
  search, queue time, held-out evaluation, circuit export, or compilation.
  CoverageDLGN topology construction is an offline, CPU-side preprocessing
  cost and is reported separately when it was instrumented. `N/R` means the
  resource was not recorded; it never denotes zero cost.

## Table 1: Dense MNIST and Fashion-MNIST

Target architecture: six layers, 48K total rank-2 gates, and 768K raw gate
logits during training.

| Method | Matched gates | MNIST A / R | MNIST train / peak GPU | Fashion-MNIST A / R | Fashion train / peak GPU | Match status |
|---|---:|---:|---:|---:|---:|---|
| Deep DLGN fixed random | 48K | **[REPRODUCED] 97.090 +/- 0.180% (n=5) / [REPORTED] 97.69%** | 15.03 min / 0.099 GiB | **[REPRODUCED] 86.308 +/- 0.186% (n=5) / [REPORTED] 87.17%** | 14.97 min / 0.099 GiB | Exact architecture; sources: `summary/table1_mnist_final.json`, `summary/table1_fashion_final.json` |
| Mommen, \(N_c=16\) | 48K | [ADAPTED] 98.084 +/- 0.066% (n=5) / [REPORTED] 98.14% `[12K, nonmatched]` | 35.92 min / 1.002 GiB | [ADAPTED] 87.260 +/- 0.282% (n=3) / [REPORTED] 87.16% `[8K, nonmatched]` | 24.04 min / 0.498 GiB | Adapted to 48K; Fashion selected \(N_c=8\), depth 3; final sources as above |
| LILogicNet Top-32 | 48K | [ADAPTED] 98.124 +/- 0.029% (n=5) / [REPORTED] 98.95 +/- 0.09% `[32K, nonmatched]` | 82.02 min / 4.187 GiB | [ADAPTED] 88.437 +/- 0.159% (n=3) / [REPORTED] 90.26 +/- 0.11% `[64K, nonmatched]` | 70.84 min / 4.676 GiB | Adapted to 48K; Fashion Top-32, depth 2, tau 30; final sources as above |
| BitLogic best-of-space | 48K | [ADAPTED] 98.204 +/- 0.042% (n=5) / [REPORTED] 97.84 +/- 0.04% `[128K total, nonmatched]` | 89.18 min / 3.557 GiB | [ADAPTED] 89.740 +/- 0.243% (n=3) / [REPORTED] 89.16 +/- 0.08% `[128K total, nonmatched]` | 88.94 min / 3.557 GiB | Adapted to 48K; final sources as above |
| **CoverageDLGN** | **48K** | **[OUR-FINAL] 97.500 +/- 0.099% (n=5) / [N/A]** | **15.02 min / 0.099 GiB** | **[OUR-FINAL] 87.102 +/- 0.357% (n=5) / [N/A]** | **15.02 min / 0.099 GiB** | Exact target; final sources as above |

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

### Second-round six-layer architecture-matched validation controls

These local controls hold the six-by-8K, 48K-gate architecture and 200-epoch
effort fixed. Values are best hardened **validation** accuracy, not held-out
test accuracy. Random and V3 reuse their completed five-seed cohorts; learned-
connectivity controls use three seeds. BitLogic retains its published rank-4
gate/4-bit input parameterization, so its reproduced-negative result measures
depth transfer rather than a rank-2 parameterization match.

| Method | Training parameters (MNIST / Fashion) | MNIST matched V | MNIST time / GPU | Fashion matched V | Fashion time / GPU | Provenance |
|---|---:|---:|---:|---:|---:|---|
| Fixed-random DLGN | 0.768M / 0.768M | [REPRODUCED] 97.157 +/- 0.043% (n=5) | 15.03 min / 0.099 GiB | [REPRODUCED] 87.477 +/- 0.183% (n=5) | 14.97 min / 0.099 GiB | Exact frozen control; reused, not rerun |
| Mommen learned connectivity | 2.304M / 1.536M | [ADAPTED] 95.683 +/- 0.404% (n=3) | 51.90 min / 0.809 GiB | [ADAPTED] 87.400 +/- 0.928% (n=3) | 29.09 min / 0.453 GiB | Exact depth/budget; method-specific candidate count |
| LILogicNet Top-32 | 3.840M / 3.840M | [ADAPTED] 95.717 +/- 0.351% (n=3) | 92.44 min / 3.706 GiB | [ADAPTED] 84.267 +/- 1.636% (n=3) | 92.05 min / 3.706 GiB | Exact depth/budget; 200 epochs |
| BitLogic rank-4 | 3.840M / 3.840M | [ADAPTED] 11.417 +/- 0.000% (n=3) | 112.75 min / 3.527 GiB | [ADAPTED] 10.867 +/- 0.000% (n=3) | 112.74 min / 3.527 GiB | Reproduced-negative six-layer transfer on both datasets |
| **CoverageDLGN V3** | **0.768M / 0.768M** | **[OUR-FINAL] 97.403 +/- 0.114% (n=5)** | **15.02 min / 0.099 GiB** | **[OUR-FINAL] 87.873 +/- 0.271% (n=5)** | **15.02 min / 0.099 GiB** | Frozen topology-only method; reused, not rerun |

The complete-cohort means place V3 +0.473 pp above matched Mommen and +3.607 pp
above matched LILogicNet on Fashion while retaining the fixed-random training
parameter count. The common-seed paired Fashion effects are +0.483 pp (95% CI
[-1.597, +2.563]) and +3.617 pp ([-0.016, +7.249]), respectively, so neither is
a conclusive three-seed superiority claim. On MNIST, the corresponding paired
effects are +1.706 pp over Mommen ([+0.666, +2.745]) and +1.672 pp over
LILogicNet ([+0.952, +2.392]). The BitLogic collapse is consistent on all three
Fashion seeds and is retained as negative depth-transfer evidence, not
presented as a paper-faithful reproduction of BitLogic's two-layer result.
Relative to matched Mommen, V3 uses 3x fewer MNIST and 2x fewer Fashion
training parameters, trains 3.46x/1.94x faster, and peaks at 8.17x/4.58x less
GPU memory. Relative to matched LILogicNet it uses 5x fewer training
parameters, trains 6.15x/6.13x faster, and peaks at 37.43x less GPU memory on
both datasets. These are training-resource trade-offs at an identical 48K
deployed gate budget; they are not inference-speed claims.
Machine-readable sources are the `second_exact_*` result directories and
`summary/second_round_status.{json,csv}`.

### Dense MNIST/Fashion-MNIST gate-budget compression

All rows use six layers, three paired seeds, the fixed split, and 200 effective
epochs. Accuracy is best hardened validation; held-out test remains locked.

| Dataset | Total gates | Training parameters | Fixed random V | CoverageDLGN V3 V | Paired effect | 95% CI | Wins | Time random / V3 | Peak GPU random / V3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 4K | 0.064M | [REPRODUCED] 85.539 +/- 0.495% | **[OUR] 86.067 +/- 0.159%** | +0.528 pp | [-1.087, +2.142] | 2/3 | 14.78 / 14.92 min | 0.009 / 0.009 GiB |
| MNIST | 8K | 0.128M | [REPRODUCED] 91.461 +/- 0.286% | **[OUR] 91.956 +/- 0.113%** | +0.494 pp | [-0.479, +1.467] | 3/3 | 15.06 / 15.09 min | 0.017 / 0.017 GiB |
| MNIST | 16K | 0.256M | [REPRODUCED] 95.100 +/- 0.161% | **[OUR] 95.478 +/- 0.444%** | +0.378 pp | [-0.925, +1.681] | 3/3 | 15.06 / 15.01 min | 0.034 / 0.034 GiB |
| MNIST | 32K | 0.512M | [REPRODUCED] 96.694 +/- 0.135% | **[OUR] 97.011 +/- 0.129%** | +0.317 pp | [-0.334, +0.967] | 3/3 | 15.09 / 14.97 min | 0.066 / 0.066 GiB |
| Fashion-MNIST | 8K | 0.128M | [REPRODUCED] 83.433 +/- 0.148% | **[OUR] 83.644 +/- 0.158%** | **+0.211 pp** | **[+0.085, +0.338]** | 3/3 | 15.09 / 15.10 min | 0.017 / 0.017 GiB |
| Fashion-MNIST | 16K | 0.256M | [REPRODUCED] 86.194 +/- 0.250% | **[OUR] 86.883 +/- 0.192%** | +0.689 pp | [-0.344, +1.722] | 3/3 | 15.09 / 14.96 min | 0.034 / 0.034 GiB |
| Fashion-MNIST | 32K | 0.512M | [REPRODUCED] 87.461 +/- 0.234% | **[OUR] 87.778 +/- 0.327%** | +0.317 pp | [-1.071, +1.704] | 2/3 | 15.06 / 15.08 min | 0.066 / 0.066 GiB |
| Fashion-MNIST | 64K | 1.024M | [REPRODUCED] 87.333 +/- 0.557% | **[OUR] 88.100 +/- 0.200%** | +0.767 pp | [-0.524, +2.057] | 3/3 | 15.09 / 14.89 min | 0.132 / 0.132 GiB |

V3's mean effect is positive in all eight new cells and in 22/24 paired runs.
Only Fashion-MNIST 8K has a positive cell-wise 95% interval at n=3; the other
cells are directional compression evidence, not individual superiority
claims. The reused 48K validation references remain positive (+0.247 pp MNIST,
+0.396 pp Fashion, n=5). V3 changes fixed connectivity only, so gate count,
training parameter count, training time, and peak allocation match random.

## Table 2: Dense CIFAR-10 S/M/L

| Architecture | Target gates | Raw training parameters | Method | A / R accuracy | Mean train time / peak GPU | Reported configuration |
|---|---:|---:|---|---:|---:|---|
| S: 4 x 12K | 48K | 0.768M | Deep DLGN random | **[REPRODUCED] 49.056 +/- 0.356% test (n=5) / [REPORTED] 51.27%** | 12.24 min / 0.106 GiB | Exact 48K architecture; one-time held-out test; source: `summary/paper_cifar10_semantic_v3.json` |
|  |  | 1.536M | Mommen learned connectivity | **[ADAPTED] 50.950 +/- 0.244% (n=3) / [N/A]** | 26.45 min / 0.480 GiB | Exact-48K \(N_c=8\) adaptation; one-time held-out test; source: `summary/table2_s_comparator_final.json` |
|  |  | 3.840M | LILogicNet | **[ADAPTED] 50.743 +/- 0.574% (n=3) / [REPORTED] 55.11%** | 86.87 min / 3.949 GiB | Exact-48K Top-32 adaptation; one-time held-out test; reported value uses 8K nonmatched gates; local source as above |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] 52.358 +/- 0.282% test (n=5) / [N/A]** | **12.26 min / 0.106 GiB** | Exact 48K target; +3.302 pp paired test gain, 95% CI [+2.767, +3.837] |
|  |  | 0.768M | Unified U2 | **[OUR-TRANSFER] 52.097 +/- 0.630% test (n=3) / [N/A]** | 12.11 min / 0.106 GiB | Same cross-domain rule as convolutional U2; +3.183 pp vs three-seed random, CI [+0.772, +5.595]; V3 remains selected |
| M: 4 x 128K | 512K | 8.192M | Deep DLGN random | **[REPRODUCED] 54.028% (n=5) / [REPORTED] 57.39%** | 41.55 min / 1.123 GiB | Exact 512K architecture; one-time held-out test |
|  |  | 16.384M | Mommen learned connectivity | **[TRIED] 54.420% test (n=1) / [N/A]** | 4.84 h / 4.918 GiB | Exact-512K \(N_c=8\) adaptation; conditional policy stops at one seed; one-time held-out test |
|  |  |  | LILogicNet | [REPRODUCED] Top-32 57.840% (n=1) / [REPORTED] 57.28 +/- 0.30% | 37.81 min / 8.100 GiB | 64K gates, nonmatched; direct U2 protocol table below |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] 58.284% (n=5) / [N/A]** | **41.48 min / 1.123 GiB** | Exact 512K target; +4.256 pp paired test gain |
| L: 5 x 256K | 1.28M | 20.48M | Deep DLGN random | **[REPRODUCED] 55.960 +/- 0.251% test (n=5) / [REPORTED] 60.78%** | 1.89 h / 2.717 GiB | Exact 1.28M architecture; one-time held-out test; source: `summary/table2_l_final.json` |
|  |  | 40.960M | Mommen learned connectivity | **[TRIED] 54.340% test (n=1) / [N/A]** | 13.39 h / 11.904 GiB | Exact-1.28M \(N_c=8\) adaptation; conditional policy stops at one seed; one-time held-out test |
|  |  |  | LILogicNet-L | [REPRODUCED] Top-32 62.030% (n=1) / [REPORTED] 60.98 +/- 0.19% | 350.64 min / 24.861 GiB | 256K gates, nonmatched; direct U2 protocol table below |
|  |  |  | **CoverageDLGN** | **[OUR-FINAL] raw V3 swap-0.50 61.020 +/- 0.336% test (n=5) / [N/A]** | **1.89 h / 2.717 GiB** | Exact 1.28M target; +5.060 pp paired, 95% CI [+4.555, +5.565]; one-time held-out test |

WARP-LUT and BitLogic use a separate two-layer common backbone in the
BitLogic paper, not the Deep-DLGN S/M/L architectures above. Their complete
reported CIFAR-10 ladder is therefore recorded separately:

| Reported backbone | Total gates | WARP-LUT reported | BitLogic rank-4 reported | Local status |
|---|---:|---:|---:|---|
| 2 x 4K | 8K | 33.86 +/- 0.10% | 38.93 +/- 0.19% | Direct local protocol transfer below |
| 2 x 16K | 32K | 42.92 +/- 0.29% | 49.22 +/- 0.26% | Direct local protocol transfer below |
| 2 x 64K | 128K | 52.12 +/- 0.01% | 58.06 +/- 0.14% | Direct local protocol transfer below |

### U2 transfer to dense M/L and published connectivity protocols

All local rows below are hardened held-out test accuracy from the immutable
third-round freeze. `+/-` is sample standard deviation. U2 is unchanged from
the unified dense/convolutional rule; it uses fixed rank-2 routing and zero
training-only routing parameters.

| Protocol / scale | Method | Gates / rank | Training params / routing params | Hard test accuracy | Gain vs paired random | Train time / peak GPU | CUDA ms/batch-128 | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Current dense M, 4 x 128K | U2 | 512K / 2 | 8.192M / 0 | **58.653 +/- 0.168% (n=3)** | **+4.557 pp**, CI [+3.781, +5.332], 3/3 | 41.39 min / 1.123 GiB | 8.879 | [OUR-TRANSFER] |
| Current dense L, 5 x 256K | U2 | 1.28M / 2 | 20.480M / 0 | **60.463 +/- 0.348% (n=3)** | **+4.593 pp**, CI [+3.721, +5.466], 3/3 | 113.53 min / 2.717 GiB | 26.073 | [OUR-TRANSFER] |
| LILogic M, 1 x 64K | Fixed random | 64K / 2 | 1.024M / 0 | 49.010 +/- 0.426% (n=3) | -- | 14.62 min / 0.474 GiB | 0.899 | [REPRODUCED] |
|  | **U2** | 64K / 2 | 1.024M / 0 | **52.543 +/- 0.296% (n=3)** | **+3.533 pp**, CI [+1.797, +5.270], 3/3 | 14.59 min / 0.474 GiB | 0.934 | [OUR-TRANSFER] |
|  | Top-32 | 64K / 2 | 5.120M / 4.096M | 57.840% (n=1) | +8.830 pp | 37.81 min / 8.100 GiB | 7.552 | [REPRODUCED] / [REPORTED] 57.28 +/- 0.30% |
| LILogic L, 2 x 128K | Fixed random | 256K / 2 | 4.096M / 0 | 55.333 +/- 0.469% (n=3) | -- | 15.20 min / 1.557 GiB | 4.586 | [REPRODUCED] |
|  | **U2** | 256K / 2 | 4.096M / 0 | **60.193 +/- 0.286% (n=3)** | **+4.860 pp**, CI [+3.083, +6.637], 3/3 | 18.47 min / 1.557 GiB | 4.303 | [OUR-TRANSFER] |
|  | Top-32 | 256K / 2 | 20.480M / 16.384M | 62.030% (n=1) | +6.697 pp | 350.64 min / 24.861 GiB | 20.893 | [REPRODUCED] / [REPORTED] 60.98 +/- 0.19% |

U2 preserves the exact fixed-random gate and parameter budgets. Relative to
Top-32, U2 uses 5x fewer trainable parameters; peak allocated training memory
is 17.1x lower on M and 16.0x lower on L, while hardened inference is 8.1x and
4.9x faster locally. Top-32 remains more accurate by 5.297 pp on M and 1.837
pp on L, so this is an accuracy--resource Pareto result, not absolute accuracy
superiority.

| BitLogic protocol | Method | Gates / rank | Training params / routing params | Hard test accuracy | Gain vs paired rank-2 random | Train time / peak GPU | CUDA ms/batch-128 | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S, 2 x 4K | Rank-2 random | 8K / 2 | 0.128M / 0 | 26.175 +/- 0.445% (n=2) | -- | 8.50 min / 0.028 GiB | 0.425 | [REPRODUCED] |
|  | **U2** | 8K / 2 | 0.128M / 0 | **28.435 +/- 1.648% (n=2)** | +2.260 pp, 2/2; CI inconclusive | 8.47 min / 0.028 GiB | 0.514 | [OUR-TRANSFER] |
|  | Rank-4 learnable-16 | 8K / 4 | 0.640M / 0.512M | 27.500 +/- 0.113% (n=2) | -- | 9.92 min / 0.611 GiB | 2.305 | [REPRODUCED-NEGATIVE] / [REPORTED] 38.93 +/- 0.19% |
| M, 2 x 16K | Rank-2 random | 32K / 2 | 0.512M / 0 | 25.945 +/- 0.870% (n=2) | -- | 8.58 min / 0.104 GiB | 0.741 | [REPRODUCED] |
|  | U2 | 32K / 2 | 0.512M / 0 | 26.040 +/- 0.014% (n=2) | +0.095 pp, 1/2; inconclusive | 8.64 min / 0.104 GiB | 0.738 | [OUR-TRANSFER] |
|  | Rank-4 learnable-16 | 32K / 4 | 2.560M / 2.048M | 16.625 +/- 0.078% (n=2) | -- | 41.21 min / 2.372 GiB | 6.407 | [REPRODUCED-NEGATIVE] / [REPORTED] 49.22 +/- 0.26% |
| L, 2 x 64K | Rank-2 random | 128K / 2 | 2.048M / 0 | 25.160 +/- 0.156% (n=2) | -- | 8.52 min / 0.414 GiB | 1.778 | [REPRODUCED] |
|  | U2 | 128K / 2 | 2.048M / 0 | 25.930 +/- 2.871% (n=2) | +0.770 pp, 1/2; inconclusive | 8.56 min / 0.414 GiB | 1.729 | [OUR-TRANSFER] |
|  | Rank-4 learnable-16 | 128K / 4 | 10.240M / 8.192M | 13.160 +/- 0.240% (n=2) | -- | 164.47 min / 9.456 GiB | 15.972 | [REPRODUCED-NEGATIVE] / [REPORTED] 58.06 +/- 0.14% |

The BitLogic rank-4 transfers learn relaxed signal but fail when hardened. For
example, final relaxed test accuracy averages 57.60% at M while final hard
accuracy is 14.70%. These cells document an unresolved implementation or
protocol mismatch and are not faithful BitLogic reproductions. The defensible
U2 evidence on this ladder is S; M/L remain directional and underpowered.

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

Measured full-run resource scaling for the matched fixed-random and raw-V3
arms is:

| Gate budget | Random train time | V3 train time | Random / V3 peak GPU |
|---:|---:|---:|---:|
| 48K | 12.24 min | 12.26 min | 0.106 / 0.106 GiB |
| 128K | 12.44 min | 12.44 min | 0.281 / 0.281 GiB |
| 256K | 19.14 min | 19.17 min | 0.562 / 0.562 GiB |
| 384K | 29.83 min | 29.82 min | 0.845 / 0.845 GiB |
| 512K | 41.55 min | 41.48 min | 1.123 / 1.123 GiB |
| 1.28M | 1.89 h | 1.89 h | 2.717 / 2.717 GiB |

These measurements show no material GPU-training overhead from V3 once its
fixed indices have been constructed. At 48K, the recorded offline topology
construction means were 0.05 seconds for random and 9.33 seconds for V3. At
512K, the optimized exact builder required about 5 seconds for random and
107 seconds for V3. The original 209-second V3 construction was superseded
by the bit-identical lazy-heap implementation. Final L runs used cached
topologies, so a comparable uncached L construction time is not recorded.

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
| LogicTreeNet-S | 3 thresholds/RGB = 9 channels | paper ~0.40M; 83,552 LUTs; 874,496 spatial applications | Original fixed routing | [TRIED-SELECTION] 56.864% best hard validation (n=5) at 20K; **[REPRODUCED-FULL] 58.680% V / 57.370% T (n=1) at 350K**; [TRIED-HISTORICAL-TEST] 56.140% hard test (n=3) | **[REPORTED] 60.38% test** | Exact architecture; full best-V checkpoint frozen before one-time test |
|  |  | ~0.70 x source gates | Two-stage unit tying, 30% | [N/A] | [REPORTED] 56.70 +/- 0.08% validation | Exact S architecture, different gate count and validation metric |
|  |  | 0.57M reported | Conv. TTNet-S | [N/A] | [REPORTED] 50.10% | Different truth-table architecture |
|  |  | same declared S budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared S budget | WARP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared S budget | **CoverageDLGN-Channel (frozen V4)** | [TRIED-SELECTION] 57.448% validation (n=5) at 20K; **[OUR-FULL] 59.860% V / 58.930% T (n=1) at 350K**; [TRIED-HISTORICAL-TEST] 56.367% test (n=3) | [N/A] | Full schedule: +1.180 V / +1.560 T pp over matched random with identical cost/time/memory |
|  |  | same declared S budget | **Unified semantic degree-balanced (U1)** | [TRIED-SELECTION] 57.624% validation (n=5) at 20K; **[OUR-FULL] 59.880% V / 58.800% T (n=1) at 350K** | [N/A] | Full +1.200 V / +1.430 T pp versus matched random at identical cost/time/memory |
|  |  | same declared S budget | **Unified multiscale (U2)** | **[OUR-PILOT] 58.847 +/- 0.600% V (n=3) at 20K; [OUR-FULL] 61.000% V / 60.630% T (n=1) at 350K** | [N/A] | **Pilot +2.173 pp, CI [+1.647, +2.700], 3/3; full +2.320 V / +3.260 T pp; 0.25 pp above reported S test** |
|  |  | same declared S budget | Channel-spatial leaf pairing | [TRIED-STOPPED] 57.033% validation (n=3) | [N/A] | -0.153 pp vs V4; failed both promotion gates |
| LogicTreeNet-M | 3 thresholds/RGB = 9 channels | paper ~3.08M; 668,416 LUTs; 6,995,968 spatial applications | Original fixed routing | **[TRIED-ONE-SEED] 70.68% best hard validation; 69.57% hard test** | **[REPORTED] 71.01% test** | Exact architecture; best-validation checkpoint from matched 200K run |
|  |  | ~0.70 x source gates | Two-stage unit tying, 30% | [N/A] | [REPORTED] 70.77 +/- 0.07% validation | Exact M architecture, different gate count and validation metric |
|  |  | approximately 3.08M reported | Scalability-boundaries CDLGN-M | [N/A] | [REPORTED] 65.23% | Minimally modified M protocol |
|  |  | 189M reported | Conv. TTNet-L | [N/A] | [REPORTED] 70.75% | Different, much larger architecture |
|  |  | same declared M budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared M budget | WARP-LogicTreeNet | [PENDING] | [N/A] | Exact adaptation not run |
|  |  | same declared M budget | **CoverageDLGN-Channel (frozen V4)** | **[TRIED-ONE-SEED] 71.26% best hard validation; 69.96% hard test** | [N/A] | +0.58 validation pp and +0.39 test pp; V4 led 97/100 validation evaluations |
|  |  | same exact M budget | **Unified multiscale U2** | **[OUR-ONE-SEED] 72.38% best hard validation; 71.65% hard test** | [N/A] | **+1.70 V / +2.08 T pp vs random; +1.12 V / +1.69 T pp vs V4; +0.64 T pp above reported M** |
| LogicTreeNet-L | 5-bit paper input | paper ~28.9M | Original fixed routing | [PENDING] | **[REPORTED] 84.99% test** | Exact replication requires 5-bit input and teacher |
|  |  | 189M reported | Conv. TTNet-L | [N/A] | [REPORTED] 70.75% | Different architecture |
|  |  | M architecture | Scalability-boundaries CDLGN-M | [N/A] | [REPORTED] 65.23% | Not LogicTreeNet-L |
|  |  | same declared L budget | Light/IWP-LogicTreeNet | [PENDING] | [N/A] | New L adaptation |
|  |  | same declared L budget | WARP-LogicTreeNet | [PENDING] | [N/A] | New L adaptation |
|  |  | same declared L budget | **CoverageDLGN-Channel** | [PENDING] | [N/A] | Requires a paper-faithful L baseline first |

Measured training resources for the locally completed paper-faithful arms
are shown separately to keep validation and test metrics readable:

| Architecture | Method | Training wall time / run | Peak training GPU | Offline topology construction |
|---|---|---:|---:|---:|
| LogicTreeNet-S, 20K | Fixed random | 17.03 min | 1.831 GiB | 0.176 s |
|  | Frozen V4 | 17.04 min | 1.831 GiB | 0.413 s |
|  | Unified U1 | 17.03 min | 1.831 GiB | 0.165 s |
| LogicTreeNet-S, 350K | Fixed random | 4.975 h | 1.831 GiB | 0.217 s |
|  | Frozen V4 | 4.957 h | 1.831 GiB | 0.449 s |
|  | Unified U1 | 4.980 h | 1.831 GiB | 0.206 s |
|  | Unified U2 | 4.951 h | 1.831 GiB | 1.372 s |
| LogicTreeNet-M, 200K | Fixed random | 25.47 h | 14.615 GiB | 2.478 s |
|  | Frozen V4 | approximately 25.49 h | N/R | 6.182 s |
|  | Unified U2 | 35.686 h | 14.614 GiB | 18.908 s |

The S resource values summarize the newly instrumented seeds 3 and 4; the
older seeds predate complete resource fields. The V4 M wall time is the
artifact span and is approximate. Its training-memory peak was not finalized
because the run ended by controlled `SIGINT`; the matched fixed-random run
recorded 14.615 GiB. V4's measured one-time M topology overhead relative to
random is 3.704 seconds. U2's measured offline topology overhead relative to
random is 16.430 seconds. Its wall-clock training was collected under different
concurrent machine load and is reported as measured, not attributed to the
fixed topology rule.

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

| Method | Training wall time / run | Peak training GPU | Resource status |
|---|---:|---:|---|
| WARP fixed uniform | N/R | N/R | Controlled interruption at the frozen 30K boundary |
| WARP fixed distributive | N/R | N/R | Controlled interruption at the frozen 30K boundary |
| WARP learnable | 4.29 h | 14.619 GiB | Complete seed-0 run |
| Matched random | 3.87 h | 14.614 GiB | Complete seed-0 run |
| **CoverageDLGN Legacy V4** | **3.85 h** | **14.614 GiB** | Complete seed-0 run |

The fixed-uniform and fixed-distributive jobs were originally launched with
50K updates and interrupted just after the common 30K validation boundary,
after both had reached the approximate plotted endpoints. No later validation
was evaluated, and 30K was frozen before observing the Medium Legacy V4
result. These results therefore show that the reconstructed WARP accuracies
are reached within 30K; they are not an exact 50K reproduction. The direct
V4 claim uses only its matched-random row because the public WARP arms use a
different `random-unique` routing sampler. Seeds 1 and 2 remain pending.

The separate paper-faithful nine-channel S U2 pilot is now complete. At the
same 20K effort and exact circuit budget, U2 reached `[OUR]` 58.847 +/- 0.600%
best hardened validation (n=3). It gains +2.173 pp over the original primary
fixed-random cohort (95% CI [+1.647, +2.700]), +1.660 pp over frozen V4, and
+0.833 pp over U1, winning all three pairs in every comparison. Against the
separate explicit controlled-random cohort it gains +1.707 pp with 3/3 wins;
that interval crosses zero. Both random provenances are reported. Training
time is 17.02 minutes per seed, peak allocation is 1.831 GiB, topology
construction is 1.380 seconds, and routing adds no trainable or deployed
cost. The unchanged full-effort seed-0 checkpoint reached 61.000% hardened
validation and 60.630% one-time held-out hard test, +3.260 pp over its
matched full random control. This is 0.250 pp above the reported 60.38% S
test value, but the full-resource cohort is one seed; the three-seed pilot is
the method-level replication evidence.

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
| Paper-faithful S, 9 channels, 20K checkpoints | Fixed random | 197,851 | 1.071 GiB | 3.057 ms / 41.87K examples/s | Historical pilot snapshot |
|  | Frozen V4 | 202,827 (+2.52%) | 1.073 GiB | 3.076 ms / 41.61K examples/s | +0.62% latency; no speed claim |
|  | U1 | 214,883 (+8.61%) | 1.074 GiB | 3.095 ms / 41.35K examples/s | +1.26% latency; no speed claim |
| Paper-faithful S, 9 channels, 350K frozen cohort | Fixed random | 252,936 | 1.114 GiB | 3.230 ms / 39.63K examples/s | Full-schedule reference |
|  | Frozen V4 | 241,262 (-4.62%) | 1.115 GiB | 3.163 ms / 40.46K examples/s | Snapshot; no speed claim |
|  | U1 | 251,693 (-0.49%) | 1.124 GiB | 3.136 ms / 40.82K examples/s | Snapshot; no speed claim |
|  | **U2** | **262,260 (+3.69%)** | **1.138 GiB** | **3.185 ms / 40.19K examples/s** | **+3.260 test pp at modest simplified-IR cost** |
| Paper-faithful M, 9 channels | Fixed random | 1,676,852 | 6.103 GiB | [N/A] | Trace/equivalence only |
|  | Frozen V4 | 1,702,350 (+1.52%) | 6.092 GiB | [N/A] | Trace/equivalence only |
| WARP-style M, 6 channels | Matched random | 1,101,364 | 4.070 GiB | [N/A] | Separate protocol; trace/equivalence only |
|  | Legacy V4 | 1,129,547 (+2.56%) | 4.077 GiB | [N/A] | Separate protocol; trace/equivalence only |

Corrected hardened CUDA inference for the full S cohort is
6.852/6.856/6.855/6.835 ms per batch 128 for random/V4/U1/U2, with 0.3462
GiB peak device allocation in every case. These sub-percent differences are
treated as matched runtime, not speed claims.

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

Second-round, architecture-matched 3 x 128K validation and one-time test evidence:

| Method | Accuracy | Gates / parameters | Paired evidence | Training / peak GPU | Offline topology |
|---|---:|---:|---:|---:|---:|
| Fixed random | [REPRODUCED] 21.093 +/- 0.101% V; 20.923 +/- 0.352% T (n=3) | 384K / 6.144M | -- | 5.35 min / 0.890 GiB | 3.36 s |
| **CoverageDLGN V3** | **[OUR] 21.933 +/- 0.110% V; 21.467 +/- 0.410% T (n=3)** | **384K / 6.144M** | **V +0.840 pp, CI [+0.351, +1.329]; T +0.543 pp, 3/3 wins** | **5.34 min / 0.890 GiB** | 71.03 s |

This cohort uses the original temperature-10, learning-rate-0.01,
no-augmentation recipe for all six runs so the previously completed seed 0
remains comparable. A separate baseline-only 5K screen selected temperature
20 at 22.700% validation; it is marked `[TRIED-SCREEN]` and is not mixed into
the paired cohort. The validation selections were frozen before the single
held-out query. The n=3 test CI [-0.141, +1.227] crosses zero.

Same-384K output-allocation ablations (seed 0, validation only):

| Allocation | Fixed random | V3 | Gain | Decision |
|---|---:|---:|---:|---|
| 3 x 128K | 21.080% | **21.860%** | +0.780 pp | Seed-0 member of the promoted three-seed cohort |
| 96K + 96K + 192K | 19.960% | **20.620%** | +0.660 pp | Lower absolute accuracy; reject allocation |
| 64K + 64K + 256K | **20.380%** | 20.240% | -0.140 pp | Reject allocation |

Measured resource cost for the locally executed dense CIFAR-100 coordinates:

| Architecture / schedule | Method | Training wall time / run | Peak training GPU | Offline topology construction |
|---|---|---:|---:|---:|
| 6 x 64K, full | Fixed random | 11.37 min | 0.795 GiB | 2.07 s |
|  | CoverageDLGN V3 | 11.49 min | 0.795 GiB | 37.46 s |
| 6 x 64K, 20K ablation | CoverageDLGN V3 | 5.67 min | 0.795 GiB | 37.47 s |
|  | V3 + class-conditional head | 5.60 min | 0.795 GiB | 49.50 s |
| 6 x 256K, 5K screen | Fixed random | 32.03 min | 14.181 GiB | 34.60 s |
|  | Best screened V3 (`swap_fraction=0.5`) | 31.59 min | 14.181 GiB | 554.86 s |

The 6-by-256K values are short-screen resource measurements and must not be
presented as the cost of a full paper-length training run. No local resource
claim is made for reported-only Multilinear-CovJac rows.

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

Controlled 384K-gate CIFAR-100 depth ablation (20K steps,
validation-only):

| Architecture | Random | Frozen CoverageDLGN V3 | Paired gain | Decision |
|---|---:|---:|---:|---|
| 3 x 128K | `[REPRODUCED]` 21.093 +/- 0.101% (n=3) | `[OUR]` **21.933 +/- 0.110% (n=3)** | **+0.840 pp; 95% CI [+0.351, +1.329]** | Three-seed positive validation; frozen test +0.543 pp, 3/3 wins |
| 6 x 64K | `[TRIED]` 21.040% | `[TRIED]` 21.580% | +0.540 pp | Earlier three-seed 20K mean, shown as the reference rather than a seed-0 cell |
| 12 x 32K | `[TRIED]` 1.180% | `[TRIED]` 1.180% | +0.000 pp | Chance-level optimization failure |
| 24 x 16K | `[TRIED]` 1.200% | `[TRIED]` 1.200% | +0.000 pp | Chance-level optimization failure |

The 3-, 12-, and 24-layer cells have the same 384K gates, 6.144M trainable
LUT parameters, three-threshold encoding, temperature 10, split, batch,
optimizer, and effort. The second round completed the missing 3 x 128K seeds;
all three paired gains are positive. The one-time held-out test is now
complete as reported above. The
reference 6-by-64K entry is a three-seed mean from its earlier confirmation
and is not used as though it were a matched seed-0 depth cell.

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
