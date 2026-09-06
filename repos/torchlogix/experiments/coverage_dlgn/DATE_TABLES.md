The complete paper-facing tables are below. They are also preserved in [PAPER_COMPARISON_TABLES.md](/ssd1/users/ianagno/my-repos/DLCA_DLGN_coverageDLGN/repos/torchlogix/experiments/coverage_dlgn/PAPER_COMPARISON_TABLES.md).

Notation:

- `A` = locally achieved; `R` = paper-reported.
- `T` = hardened test; `V` = hardened validation.
- `†` = nonmatched architecture, budget, fan-in, or protocol.
- Training time is the mean wall time for one run, not the total across seeds.
- GPU memory is peak PyTorch allocation. Additional hardware headroom is advisable.
- Topology construction is offline CPU preprocessing and is not included in GPU training time.
- `N/R` = not recorded; `—` = not available or not run.

## Current CoverageDLGN method status

All V3 rows below remain valid. V3 (`semantic_balanced_hybrid`) is the strongest
dense specialization. The unified paper candidate is U2
(`semantic_multiscale_balanced`): the same semantic, degree-first multiscale
fixed-routing rule is used in dense and convolutional networks, with zero
learned routing. U2 has not replaced or overwritten V3, V4, or U1.

| U2 evidence | Hardened result | Matched effect | Status |
|---|---:|---:|---|
| Dense CIFAR-10 M, 4 x 128K | 58.653 +/- 0.168 T (3) | +4.557 pp vs random, 3/3 | Unified transfer; tied with V3 within CI |
| Dense CIFAR-10 L, 5 x 256K | 60.463 +/- 0.348 T (3) | +4.593 pp vs random, 3/3 | Unified transfer; V3 remains 0.610 pp higher |
| LILogic M protocol, 64K | 52.543 +/- 0.296 T (3) | +3.533 pp vs random, 3/3 | Top-32 is more accurate but much more expensive |
| LILogic L protocol, 256K | 60.193 +/- 0.286 T (3) | +4.860 pp vs random, 3/3 | Top-32 is +1.837 pp, with 5x training parameters |
| LogicTreeNet-S, 350K | 60.630 T (1) | +3.260 pp vs random | Full one seed; 20K support is n=3, +2.173 pp |
| LogicTreeNet-M, 200K | 71.650 T (1) | +2.080 pp vs random | Full one seed; +0.64 pp vs reported 71.01, not a statistical SOTA claim |
| Dense CIFAR-100, 3 x 128K pilot | U2 +0.100 pp | Did not promote | Retain V3 result; U2 is not universal |

Current U2 evidence uses raw rank-two LUTs. The WARP-style table below also
uses raw LUT parameterization and Legacy V4, so it is not a U2+WARP result.
U2 under TorchLogix WARP/Light/Gumbel and rank-four U2 are explicitly pending.

## Table I. Dense MNIST and Fashion-MNIST

Six layers, 48K rank-2 gates, and 0.768M raw gate logits.

| Method | Trainable parameters | MNIST A / R | MNIST time / GPU | Fashion-MNIST A / R | Fashion time / GPU | Provenance |
|---|---:|---:|---:|---:|---:|---|
| Fixed-random DLGN | 0.768M | 97.090 ± 0.180 T (5) / 97.69 R | 15.03 min / 0.099 GiB | 86.308 ± 0.186 T (5) / 87.17 R | 14.97 min / 0.099 GiB | Reproduced, exact local architecture |
| Mommen connectivity | 2.304M MNIST; 1.536M Fashion | 98.084 ± 0.066 T (5) / 98.14 R† | 35.92 min / 1.002 GiB | 87.260 ± 0.282 T (3) / 87.16 R† | 24.04 min / 0.498 GiB | Adapted to 48K; reported results use 12K/8K |
| LILogicNet | 3.840M | 98.124 ± 0.029 T (5) / 98.95 ± 0.09 R† | 82.02 min / 4.187 GiB | 88.437 ± 0.159 T (3) / 90.26 ± 0.11 R† | 70.84 min / 4.676 GiB | Adapted to 48K; reported results use 32K/64K |
| BitLogic | 3.840M | **98.204 ± 0.042 T (5)** / 97.84 ± 0.04 R† | 89.18 min / 3.557 GiB | **89.740 ± 0.243 T (3)** / 89.16 ± 0.08 R† | 88.94 min / 3.557 GiB | Adapted to 48K; reported results use 128K |
| CoverageDLGN V3 | 0.768M | 97.500 ± 0.099 T (5) / — | **15.02 min / 0.099 GiB** | 87.102 ± 0.357 T (5) / — | **15.02 min / 0.099 GiB** | Final topology-only method |

### Paired CoverageDLGN improvement

| Dataset | Random T | CoverageDLGN T | Gain | Paired 95% CI |
|---|---:|---:|---:|---:|
| MNIST | 97.090% | 97.500% | +0.410 pp | [+0.108, +0.712] |
| Fashion-MNIST | 86.308% | 87.102% | +0.794 pp | [+0.471, +1.117] |

CoverageDLGN uses no training-only routing parameters, whereas the adapted learned-connectivity methods use between 0.768M and 3.072M additional routing parameters.

### Six-layer architecture-matched validation controls

Six-by-8K, 48K-gate, 200-epoch controls. These are best hardened validation
values; they are not held-out test results. BitLogic remains rank-4 with its
method-specific four-bit input encoding.

| Method | Training parameters (MNIST / Fashion) | MNIST V | MNIST time / GPU | Fashion V | Fashion time / GPU |
|---|---:|---:|---:|---:|---:|
| Fixed random | 0.768M / 0.768M | 97.157 +/- 0.043 (5) | 15.03 min / 0.099 GiB | 87.477 +/- 0.183 (5) | 14.97 min / 0.099 GiB |
| Mommen | 2.304M / 1.536M | 95.683 +/- 0.404 (3) | 51.90 min / 0.809 GiB | 87.400 +/- 0.928 (3) | 29.09 min / 0.453 GiB |
| LILogicNet | 3.840M / 3.840M | 95.717 +/- 0.351 (3) | 92.44 min / 3.706 GiB | 84.267 +/- 1.636 (3) | 92.05 min / 3.706 GiB |
| BitLogic rank-4 | 3.840M / 3.840M | 11.417 +/- 0.000 (3) | 112.75 min / 3.527 GiB | 10.867 +/- 0.000 (3) | 112.74 min / 3.527 GiB |
| **CoverageDLGN V3** | **0.768M / 0.768M** | **97.403 +/- 0.114 (5)** | **15.02 min / 0.099 GiB** | **87.873 +/- 0.271 (5)** | **15.02 min / 0.099 GiB** |

V3's three common MNIST seeds gain +1.706 pp over Mommen (95% CI
[+0.666, +2.745]) and +1.672 pp over LILogicNet ([+0.952, +2.392]). Fashion
common-seed effects are +0.483 pp and +3.617 pp, but both three-seed intervals
touch/cross zero. V3 has no training-only routing parameters. BitLogic is a
reproduced-negative six-layer transfer result; its published two-layer result
remains valid context.

### Gate-budget compression validation

Six-layer fixed-random and frozen-V3 models use identical gates, LUT
parameters, splits, and 200-epoch effort. Values are best hardened validation
over three paired seeds; time is mean wall time per run and GPU memory is the
maximum peak allocation. Topology construction is offline and excluded from
training time.

| Dataset | Total gates | LUT parameters | Random V | CoverageDLGN V3 V | Paired gain | 95% CI | Seed wins | Random / V3 time | Random / V3 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 4K | 0.064M | 85.539 +/- 0.495 | **86.067 +/- 0.159** | +0.528 pp | [-1.087, +2.142] | 2/3 | 14.78 / 14.92 min | 0.009 / 0.009 GiB |
| MNIST | 8K | 0.128M | 91.461 +/- 0.286 | **91.956 +/- 0.113** | +0.494 pp | [-0.479, +1.467] | 3/3 | 15.06 / 15.09 min | 0.017 / 0.017 GiB |
| MNIST | 16K | 0.256M | 95.100 +/- 0.161 | **95.478 +/- 0.444** | +0.378 pp | [-0.925, +1.681] | 3/3 | 15.06 / 15.01 min | 0.034 / 0.034 GiB |
| MNIST | 32K | 0.512M | 96.694 +/- 0.135 | **97.011 +/- 0.129** | +0.317 pp | [-0.334, +0.967] | 3/3 | 15.09 / 14.97 min | 0.066 / 0.066 GiB |
| Fashion-MNIST | 8K | 0.128M | 83.433 +/- 0.148 | **83.644 +/- 0.158** | **+0.211 pp** | **[+0.085, +0.338]** | 3/3 | 15.09 / 15.10 min | 0.017 / 0.017 GiB |
| Fashion-MNIST | 16K | 0.256M | 86.194 +/- 0.250 | **86.883 +/- 0.192** | +0.689 pp | [-0.344, +1.722] | 3/3 | 15.09 / 14.96 min | 0.034 / 0.034 GiB |
| Fashion-MNIST | 32K | 0.512M | 87.461 +/- 0.234 | **87.778 +/- 0.327** | +0.317 pp | [-1.071, +1.704] | 2/3 | 15.06 / 15.08 min | 0.066 / 0.066 GiB |
| Fashion-MNIST | 64K | 1.024M | 87.333 +/- 0.557 | **88.100 +/- 0.200** | +0.767 pp | [-0.524, +2.057] | 3/3 | 15.09 / 14.89 min | 0.132 / 0.132 GiB |

V3 has a positive mean effect in all eight new cells and wins 22 of 24 paired
runs. Fashion-MNIST 8K is individually significant at n=3; the other
cell-wise intervals remain inconclusive. The reused 48K validation references
are also positive: +0.247 pp on MNIST and +0.396 pp on Fashion-MNIST (n=5).
The nonpromoted ladder checkpoints remain validation-only. The predeclared
MNIST-8K and Fashion-16K coordinates were frozen and evaluated once on test:

| Dataset / gates | Random T | V3 T | Unified U2 T | U2 gain vs random | U2 95% CI | Wins |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / 8K | 91.273 +/- 0.217 | 91.907 +/- 0.307 | **91.937 +/- 0.137** | **+0.663 pp** | **[+0.015, +1.311]** | **3/3** |
| Fashion / 16K | 85.197 +/- 0.261 | **85.913 +/- 0.356** | 85.717 +/- 0.453 | +0.520 pp | [-1.016, +2.056] | 3/3 |

U2 uses the same gate/parameter/routing budget and is much faster to construct
than V3, but V3 remains the selected Fashion result.

## Table II. Dense CIFAR-10 S/M/L

| Scale | Method | Gates | Training parameters | Local test A | Reported R | Time / peak GPU | Comparison status |
|---|---|---:|---:|---:|---:|---:|---|
| S | Fixed-random DLGN | 48K | 0.768M | 49.056 ± 0.356 (5) | 51.27 | 12.24 min / 0.106 GiB | Exact local architecture |
| S | Mommen | 48K | 1.536M | 50.950 ± 0.244 (3) | — | 26.45 min / 0.480 GiB | Exact-budget adaptation |
| S | LILogicNet | 48K | 3.840M | 50.743 ± 0.574 (3) | 55.11†, 8K | 86.87 min / 3.949 GiB | Exact-budget local adaptation |
| S | CoverageDLGN V3 | 48K | 0.768M | **52.358 ± 0.282 (5)** | — | **12.26 min / 0.106 GiB** | Exact; +3.302 pp |
| M | Fixed-random DLGN | 512K | 8.192M | 54.028 (5) | 57.39 | 41.55 min / 1.123 GiB | Exact local architecture |
| M | Mommen | 512K | 16.384M | 54.420 (1) | — | 4.84 h / 4.918 GiB | Exact-budget adaptation; one seed |
| M | LILogicNet | 64K† | 5.120M / 4.096M | 57.840 (1) | 57.28 ± 0.30† | 37.81 min / 8.100 GiB | Reproduced Top-32; direct U2 table below |
| M | CoverageDLGN V3 | 512K | 8.192M | **58.284 (5)** | — | **41.48 min / 1.123 GiB** | Exact; +4.256 pp |
| L | Fixed-random DLGN | 1.28M | 20.480M | 55.960 ± 0.251 (5) | 60.78 | 1.89 h / 2.717 GiB | Exact local architecture |
| L | Mommen | 1.28M | 40.960M | 54.340 (1) | — | 13.39 h / 11.904 GiB | Exact-budget adaptation; one seed |
| L | LILogicNet-L | 256K† | 20.480M / 16.384M | 62.030 (1) | 60.98 ± 0.19† | 350.64 min / 24.861 GiB | Reproduced Top-32; direct U2 table below |
| L | CoverageDLGN V3 | 1.28M | 20.480M | **61.020 ± 0.336 (5)** | — | **1.89 h / 2.717 GiB** | Exact; +5.060 pp |

The BitLogic paper uses a separate two-layer common backbone; its results do
not correspond to the S/M/L rows above. The complete reported CIFAR-10 ladder
is retained separately:

| Reported backbone | Gates | WARP-LUT R | BitLogic rank-4 R | Status |
|---|---:|---:|---:|---|
| 2 × 4K | 8K | 33.86 ± 0.10 | 38.93 ± 0.19 | Direct local transfer below |
| 2 × 16K | 32K | 42.92 ± 0.29 | 49.22 ± 0.26 | Direct local transfer below |
| 2 × 64K | 128K | 52.12 ± 0.01 | 58.06 ± 0.14 | Direct local transfer below |

### Unified U2 on dense and published-connectivity protocols

| Protocol | Method | Gates / rank | Train params / routing | Hard test A | Reported R | Time / peak GPU | CUDA ms/batch-128 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Current dense M | U2 | 512K / 2 | 8.192M / 0 | **58.653 +/- 0.168 (3)**; +4.557 pp | -- | 41.39 min / 1.123 GiB | 8.879 | Our transfer |
| Current dense L | U2 | 1.28M / 2 | 20.480M / 0 | **60.463 +/- 0.348 (3)**; +4.593 pp | -- | 113.53 min / 2.717 GiB | 26.073 | Our transfer |
| LILogic M | Fixed random | 64K / 2 | 1.024M / 0 | 49.010 +/- 0.426 (3) | 49.17 | 14.62 min / 0.474 GiB | 0.899 | Reproduced |
|  | **U2** | 64K / 2 | 1.024M / 0 | **52.543 +/- 0.296 (3)**; +3.533 pp, 3/3 | -- | 14.59 min / 0.474 GiB | 0.934 | Our transfer |
|  | Top-32 | 64K / 2 | 5.120M / 4.096M | 57.840 (1) | 57.28 +/- 0.30 | 37.81 min / 8.100 GiB | 7.552 | Reproduced |
| LILogic L | Fixed random | 256K / 2 | 4.096M / 0 | 55.333 +/- 0.469 (3) | 54.76 | 15.20 min / 1.557 GiB | 4.586 | Reproduced |
|  | **U2** | 256K / 2 | 4.096M / 0 | **60.193 +/- 0.286 (3)**; +4.860 pp, 3/3 | -- | 18.47 min / 1.557 GiB | 4.303 | Our transfer |
|  | Top-32 | 256K / 2 | 20.480M / 16.384M | 62.030 (1) | 60.98 +/- 0.19 | 350.64 min / 24.861 GiB | 20.893 | Reproduced |

U2 keeps the fixed-random circuit and training budgets. Top-32 is more
accurate, but uses 5x more trainable parameters, 16--17x more peak allocated
training memory, and 4.9--8.1x more hardened inference time locally.

| BitLogic ladder | Rank-2 random T | U2 T | Paired U2 gain | Rank-4 local T / reported R | Interpretation |
|---|---:|---:|---:|---:|---|
| S, 8K gates | 26.175 +/- 0.445 (2) | **28.435 +/- 1.648 (2)** | +2.260 pp, 2/2 | 27.500 +/- 0.113 / 38.93 +/- 0.19 | Positive, underpowered U2 transfer |
| M, 32K gates | 25.945 +/- 0.870 (2) | 26.040 +/- 0.014 (2) | +0.095 pp, 1/2 | 16.625 +/- 0.078 / 49.22 +/- 0.26 | U2 inconclusive; rank-4 mismatch |
| L, 128K gates | 25.160 +/- 0.156 (2) | 25.930 +/- 2.871 (2) | +0.770 pp, 1/2 | 13.160 +/- 0.240 / 58.06 +/- 0.14 | U2 inconclusive; rank-4 mismatch |

Machine-readable source: `summary/third_round_results.json`. Validation was
frozen before test; all 38 runs evaluated both predeclared checkpoints once
with zero failures.

## Table III. CIFAR-10 accuracy–gate-count frontier

| Gate budget | Fixed random T | CoverageDLGN V3 T | Gain | Paired 95% CI | Random time | V3 time | Random / V3 GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48K | 49.056 ± 0.356 (5) | **52.358 ± 0.282 (5)** | +3.302 pp | [+2.767, +3.837] | 12.24 min | 12.26 min | 0.106 / 0.106 GiB |
| 128K | 49.748 (5) | **53.910 (5)** | +4.162 pp | [+3.759, +4.565] | 12.44 min | 12.44 min | 0.281 / 0.281 GiB |
| 256K | 52.253 ± 0.058 (3) | **56.903 ± 0.134 (3)** | +4.650 pp | [+4.174, +5.126] | 19.14 min | 19.17 min | 0.562 / 0.562 GiB |
| 384K | 53.657 ± 0.328 (3) | **58.143 ± 0.153 (3)** | +4.487 pp | [+3.515, +5.458] | 29.83 min | 29.82 min | 0.845 / 0.845 GiB |
| 512K | 54.028 (5) | **58.284 (5)** | +4.256 pp | [+3.851, +4.661] | 41.55 min | 41.48 min | 1.123 / 1.123 GiB |
| 1.28M | 55.960 ± 0.251 (5) | **61.020 ± 0.336 (5)** | +5.060 pp | [+4.555, +5.565] | 1.89 h | 1.89 h | 2.717 / 2.717 GiB |

### Offline topology overhead

| Gate budget | Random construction | CoverageDLGN construction | Notes |
|---:|---:|---:|---|
| 48K | 0.05 s | 9.33 s | Five-seed recorded mean |
| 512K | ≈5 s | ≈107 s | Optimized exact builder |
| 1.28M | N/R | N/R | Final runs used cached topologies |

The first 512K V3 implementation required 209 seconds. A bit-identical lazy-heap implementation reduced this to approximately 107 seconds without changing the method or generated topology.

## Table IV. CoverageDLGN V3 component ablation

CIFAR-10 M, 512K gates, three seeds, 20K updates, hardened validation.

| Construction | Validation accuracy | Incremental effect | Train time/run | Peak GPU | Topology construction |
|---|---:|---:|---:|---:|---:|
| Fixed random | 54.820 ± 0.530 | Reference | 7.69 min | 1.123 GiB | 5.46 s |
| Degree-balanced butterfly | 58.980 ± 0.548 | +4.160 pp over random; CI [+3.988, +4.332] | 7.67 min | 1.123 GiB | 5.54 s |
| + semantic first layer, no swaps | 59.253 ± 0.153 | +0.273 pp; CI [−0.780, +1.326] | 7.70 min | 1.123 GiB | 12.25 s |
| + ancestry swaps: full V3 | **59.293 ± 0.214** | +0.040 pp; CI [−0.434, +0.514] | 7.70 min | 1.123 GiB | 107.87 s |

Full V3 gains +4.473 pp over random, with a 95% CI of [+3.624, +5.323]. Degree-balanced fan-out accounts for most of the improvement.

## Table V. Paper-faithful convolutional CIFAR-10

Both local architectures use nine Boolean input channels: three thermometer thresholds for each RGB channel.

| Architecture | Method | Circuit cost | Local V | Local T | Reported | Train time / GPU | Status |
|---|---|---:|---:|---:|---:|---:|---|
| LogicTreeNet-S | Fixed routing | ≈0.40M paper ops; 83,552 LUTs; 874,496 spatial ops | 56.864 (5) | 56.140 (3) | 60.38 T | 17.03 min / 1.831 GiB | Exact architecture; local 20K schedule |
| S | Two-stage unit tying, 30% | ≈70% source gates | — | — | 56.70 ± 0.08 V | — | Reported only; different gate count |
| S | Conv. TTNet-S | 0.57M reported | — | — | 50.10 | — | Different truth-table architecture |
| S | Light/IWP-LogicTreeNet | Same declared S budget | — | — | — | — | Pending |
| S | WARP-LogicTreeNet | Same declared S budget | — | — | — | — | Pending |
| S | CoverageDLGN V4 | Same declared S budget | 57.448 (5) | **56.367 (3)** | — | 17.04 min / 1.831 GiB | +0.584 V pp; test +0.227 pp |
| S | Unified no-swap U1 | Same declared S budget | **57.624 (5)** | Not queried | — | 17.03 min / 1.831 GiB | +0.760 pp; CI crosses zero |
| S | Unified multiscale U2 | Same exact S budget | **58.847 ± 0.600 (3)** | Locked | — | 17.02 min / 1.831 GiB | **+2.173 pp vs primary random; CI [+1.647, +2.700]; 3/3 wins** |
| S | Channel-spatial pairing | Same declared S budget | 57.033 (3) | Not queried | — | 17.08 min / 1.831 GiB | −0.153 pp versus V4 |
| S, 350K | Fixed routing | 83,552 total LUTs; 71,680 classifier; 1.337M parameters | 58.680 (1) | 57.370 (1) | 60.38 T | 4.975 h / 1.831 GiB | Frozen V before one-time T |
| S, 350K | CoverageDLGN V4 | Same exact cost | 59.860 (1) | 58.930 (1) | — | 4.957 h / 1.831 GiB | +1.180 V / +1.560 T pp |
| S, 350K | Unified U1 | Same exact cost | 59.880 (1) | 58.800 (1) | — | 4.980 h / 1.831 GiB | +1.200 V / +1.430 T pp |
| S, 350K | **Unified U2** | **Same exact cost** | **61.000 (1)** | **60.630 (1)** | — | **4.951 h / 1.831 GiB** | **+2.320 V / +3.260 T pp; 0.25 pp above reported S** |
| LogicTreeNet-M | Fixed routing | ≈3.08M paper ops; 668,416 LUTs; 6,995,968 spatial ops | 70.68 (1) | 69.57 (1) | 71.01 T | 25.47 h / 14.615 GiB | Exact architecture; 200K updates |
| M | Two-stage unit tying, 30% | ≈70% source gates | — | — | 70.77 ± 0.07 V | — | Reported only; different gate count |
| M | Scalability CDLGN-M | ≈3.08M reported | — | — | 65.23 | — | Reported only |
| M | Conv. TTNet-L | 189M reported | — | — | 70.75† | — | Much larger architecture |
| M | CoverageDLGN V4 | Same declared M budget | **71.26 (1)** | **69.96 (1)** | — | ≈25.49 h / N/R | +0.58 V pp; +0.39 T pp |
| M | **Unified U2** | **Same exact M budget** | **72.38 (1)** | **71.65 (1)** | — | **35.686 h / 14.614 GiB** | **+1.70 V / +2.08 T pp vs random; +1.12 V / +1.69 T pp vs V4; +0.64 T pp above reported M** |
| LogicTreeNet-L | Fixed routing | ≈28.9M paper ops | — | — | 84.99 T | — | Requires 5-bit input and teacher |
| L | CoverageDLGN | Same declared L budget | — | — | — | — | Pending faithful baseline |

### Convolutional topology overhead

| Architecture | Method | Offline topology construction |
|---|---|---:|
| LogicTreeNet-S | Fixed random | 0.176 s |
| S | V4 | 0.413 s |
| S | U1 | 0.165 s |
| S | U2 | 1.380 s |
| S, full 350K | Fixed random | 0.217 s |
| S, full 350K | V4 | 0.449 s |
| S, full 350K | U1 | 0.206 s |
| S, full 350K | U2 | 1.372 s |
| LogicTreeNet-M | Fixed random | 2.478 s |
| M | V4 | 6.182 s |
| M | U2 | 18.908 s |

The M V4 topology adds 3.704 seconds relative to its random control, while U2
adds 16.430 seconds. V4 peak training allocation was not finalized because the
run ended through controlled interruption. U2 recorded 14.614 GiB. Its longer
wall-clock training occurred under different concurrent machine load, so only
the offline construction time is treated as method-specific overhead.

## Table VI. WARP-style CIFAR-10 Medium compatibility

Six Boolean channels, no augmentation, 30K updates, validation only, one seed.

| Method | Routing and thresholds | Local V | Approximate WARP endpoint | Train time | Peak GPU | Status |
|---|---|---:|---:|---:|---:|---|
| WARP fixed uniform | Random-unique; uniform | 65.35 | ≈64.0 | N/R | N/R | Interrupted after frozen 30K boundary |
| WARP fixed distributive | Random-unique; distributive | 66.12 | ≈65.0 | N/R | N/R | Interrupted after frozen 30K boundary |
| WARP learnable | Random-unique; learnable | 65.88 | ≈66.6 | 4.29 h | 14.619 GiB | Complete seed 0 |
| Matched fixed random | V4 sampler; uniform | 64.58 | — | 3.87 h | 14.614 GiB | V4-matched control |
| CoverageDLGN Legacy V4 | Semantic-channel hybrid; uniform | **66.23** | — | **3.85 h** | **14.614 GiB** | +1.65 pp over matched random |

The public WARP rows and Legacy V4 use different routing samplers. The valid Legacy V4 comparison is against the matched-random row, not directly against public WARP.

## Table VII. Convolutional circuit and deployment cost

These measurements use synthetic Boolean inputs and make no validation or test-set query.

| Protocol | Method | Simplified IR nodes | Peak export RSS | Compiled CPU, batch 128 | Interpretation |
|---|---|---:|---:|---:|---|
| Paper-faithful S, 9 channels, 20K checkpoints | Fixed random | 197,851 | 1.071 GiB | 3.057 ms; 41.87K examples/s | Historical pilot snapshot |
| | V4 | 202,827 (+2.52%) | 1.073 GiB | 3.076 ms; 41.61K/s | +0.62% latency; no speed claim |
| | U1 | 214,883 (+8.61%) | 1.074 GiB | 3.095 ms; 41.35K/s | +1.26% latency; no speed claim |
| Paper-faithful S, 9 channels, 350K frozen cohort | Fixed random | 252,936 | 1.114 GiB | 3.230 ms; 39.63K examples/s | Full-schedule reference |
| | V4 | 241,262 (−4.62%) | 1.115 GiB | 3.163 ms; 40.46K/s | Snapshot; no speed claim |
| | U1 | 251,693 (−0.49%) | 1.124 GiB | 3.136 ms; 40.82K/s | Snapshot; no speed claim |
| | **U2** | **262,260 (+3.69%)** | **1.138 GiB** | **3.185 ms; 40.19K/s** | **+3.260 test pp; modest simplified-IR cost** |
| Paper-faithful M, 9 channels | Fixed random | 1,676,852 | 6.103 GiB | N/R | Trace and equivalence only |
| | V4 | 1,702,350 (+1.52%) | 6.092 GiB | N/R | Trace and equivalence only |
| WARP-style M, 6 channels | Matched random | 1,101,364 | 4.070 GiB | N/R | Separate protocol |
| | Legacy V4 | 1,129,547 (+2.56%) | 4.077 GiB | N/R | Separate protocol |

The declared LUT and spatial-operation budgets are identical within each matched architecture. Simplified IR nodes vary because checkpoint-dependent constants, wires, and duplicate functions can be removed. Corrected hardened CUDA inference for the full cohort is 6.852/6.856/6.855/6.835 ms per batch 128 for random/V4/U1/U2, with 0.3462 GiB peak device allocation for all four; these sub-percent differences are matched-runtime evidence, not speed claims.

## Table VIII. Dense CIFAR-100 BitLogic-protocol ladder

These local experiments did not promote to final test evaluation. Reported results are retained as comparison context.

| Scale | Method | Total gates | Fan-in | Local A | Reported R | Status |
|---|---|---:|---:|---:|---:|---|
| S, 2 × 4K | DiffLogic | 8K | 2 | Pending | 7.49 ± 0.21 | Reported comparison |
| S | WARP-LUT | 8K | 2 | Pending | 7.00 ± 0.04 | Reported comparison |
| S | LILogicNet | 8K | 2 | Pending | 7.63 ± 0.01 | Reported comparison |
| S | BitLogic | 8K | 4 | Pending | **10.19 ± 0.06** | Exact reported BitLogic coordinate |
| S | CoverageDLGN | 8K | 2 | 7.940 ± 0.302 V at 20K | — | Random reached 8.780 ± 0.381; did not promote |
| M, 2 × 16K | DiffLogic | 32K | 2 | Pending | 10.61 ± 0.08 | Reported comparison |
| M | WARP-LUT | 32K | 2 | Pending | 10.46 ± 0.00 | Reported comparison |
| M | LILogicNet | 32K | 2 | Pending | 10.62 ± 0.12 | Reported comparison |
| M | BitLogic | 32K | 4 | Pending | **14.06 ± 0.04** | Exact reported BitLogic coordinate |
| M | CoverageDLGN | 32K | 2 | 9.707 ± 0.076 V at 20K | — | Random reached 10.060 ± 0.220; did not promote |
| L, 2 × 64K | DiffLogic | 128K | 2 | Not run | 14.64 ± 0.09 | Reported only |
| L | WARP-LUT | 128K | 2 | Not run | 14.43 | Reported single seed |
| L | LILogicNet | 128K | 2 | Not run | 14.54 ± 0.04 | Reported only |
| L | BitLogic | 128K | 4 | Not run | **18.82 ± 0.09** | Exact reported BitLogic coordinate |
| L | CoverageDLGN | 128K | 2 | Not run | — | Stopped after S/M failed promotion |

No CIFAR-100 held-out test was queried for this two-layer ladder.

## Table IX. Dense CIFAR-100 deep architectures

| Method | Architecture | Gates | Parameters | Local accuracy | Reported accuracy | Train time / GPU | Topology time | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Fixed random | 3 × 128K | 384K | 6.144M | 21.093 ± 0.101 V; 20.923 ± 0.352 T (3) | — | 5.35 min / 0.890 GiB | 3.36 s | Matched 20K control; one-time test |
| CoverageDLGN V3 | 3 × 128K | 384K | 6.144M | **21.933 ± 0.110 V; 21.467 ± 0.410 T (3)** | — | 5.34 min / 0.890 GiB | 71.03 s | **V +0.840 pp, CI [+0.351, +1.329]; T +0.543 pp, 3/3 wins** |
| Fixed-random DLGN | 6 × 64K | 384K | 6.144M | 20.677 ± 0.522 T (3) | 22.54 ± 0.26 | 11.37 min / 0.795 GiB | 2.07 s | Exact architecture; local routing differs |
| CoverageDLGN V3 | 6 × 64K | 384K | 6.144M | **21.010 ± 0.131 T (3)** | — | 11.49 min / 0.795 GiB | 37.46 s | +0.333 pp; inconclusive |
| V3 + class head | 6 × 64K | 384K | 6.144M | 21.593 ± 0.133 V at 20K | — | 5.60 min / 0.795 GiB | 49.50 s | Negative ablation; no test |
| Fixed random | 6 × 256K | 1.536M | 24.576M | 11.680 V at 5K | 27.92 ± 0.43† | 32.03 min / 14.181 GiB | 34.60 s | Short screen only |
| Best screened V3 | 6 × 256K | 1.536M | 24.576M | 11.060 V at 5K | — | 31.59 min / 14.181 GiB | 554.86 s | Did not promote |
| Multilinear-CovJac | 6 × 256K | 1.536M | 4 parameters/gate | — | 28.37 ± 0.22 | — | — | Reported only |
| Multilinear-CovJac large | 6 × 1.28M | 7.68M | 4 parameters/gate | — | 32.72 ± 0.09 | — | — | Reported only |

CoverageDLGN’s 6 × 64K paired test gain is +0.333 pp with a 95% CI of
[−0.816, +1.483], so that earlier test coordinate is not statistically
conclusive. The separate 3 × 128K validation coordinate is positive by
+0.840 pp with a 95% CI of [+0.351, +1.329]. Its frozen one-time held-out
test gain is +0.543 pp with 3/3 wins; the n=3 CI [-0.141, +1.227] crosses zero.

## Table X. CIFAR-100 controlled allocation and depth study

All architectures use 384K gates, 6.144M gate parameters, three-threshold encoding, and 20K validation updates.

| Architecture | Random V | V3 V | Gain | Random/V3 time | Random/V3 GPU | Random/V3 topology time | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 × 128K | 21.093 ± 0.101 | **21.933 ± 0.110** | **+0.840 pp; CI [+0.351, +1.329]** | 5.35 / 5.34 min | 0.890 / 0.890 GiB | 3.36 / 71.03 s | **Three-seed positive result** |
| 96K + 96K + 192K | 19.960 | **20.620** | +0.660 pp | 5.54 / 5.52 min | 0.966 / 0.966 GiB | 2.72 / 83.35 s | Seed-0 allocation ablation; lower absolute accuracy |
| 64K + 64K + 256K | **20.380** | 20.240 | −0.140 pp | 5.97 / 5.95 min | 1.060 / 1.060 GiB | 1.90 / 89.29 s | Seed-0 allocation ablation; rejected |
| 6 × 64K | 21.040 | **21.580** | +0.540 pp | 5.61 / 5.67 min | 0.795 / 0.795 GiB | 2.07 / 37.47 s | Three-seed reference |
| 12 × 32K | 1.180 | 1.180 | 0.000 pp | 6.82 / 6.85 min | 0.750 / 0.750 GiB | 1.12 / 33.79 s | Chance-level failure |
| 24 × 16K | 1.200 | 1.200 | 0.000 pp | 9.55 / 9.57 min | 0.723 / 0.723 GiB | 0.56 / 42.43 s | Chance-level failure |

At 24 layers, final-gate ancestry saturates all 3,072 image sources and
cross-gate ancestry Jaccard reaches 1.0 for both methods. Greater ancestry
coverage therefore cannot repair the optimization collapse. At fixed 384K
gates, balanced 3 × 128K is also better than both class-head-heavy allocations.

### Dense hardened GPU deployment snapshot

Representative seed-0 checkpoints use deterministic synthetic inputs, batch
128, and no dataset access. Values are milliseconds per batch / peak device
GiB on the RTX PRO 6000 Blackwell GPU.

| Coordinate | Random | V3 | U2 | Interpretation |
|---|---:|---:|---:|---|
| MNIST, 8K | 1.138 / 0.0049 | 1.140 / 0.0049 | 1.137 / 0.0049 | Matched |
| Fashion, 16K | 1.134 / 0.0094 | 1.136 / 0.0094 | 1.137 / 0.0094 | Matched |
| Dense CIFAR-10 S | 0.770 / 0.0412 | 0.765 / 0.0412 | 0.767 / 0.0412 | Matched |
| Dense CIFAR-100, 3 × 128K | 3.589 / 0.4082 | 3.527 / 0.4076 | — | Matched; U2 rejected |

These single-checkpoint sub-2% timing differences are not speed claims. They
support unchanged runtime and memory at an identical hardened circuit budget.

## Table XI. Convolutional CIFAR-100 planned comparison

No matched local result is currently available, so this is not yet a manuscript-ready result table.

| Architecture | Estimated operations | Method | Local A | Reported R | Status |
|---|---:|---|---:|---:|---|
| LogicTreeNet-S | ≈0.49M | Original/Soft-Mix | Pending | — | New transfer |
| S | ≈0.49M | Light/IWP | Pending | — | No published exact S result |
| S | ≈0.49M | WARP | Pending | — | No published exact S result |
| S | ≈0.49M | CoverageDLGN-Channel | Pending | — | Requires new matched experiment |
| LogicTreeNet-M | ≈3.82M | Original/Soft-Mix | Pending | 29.0 ± 0.6† | Reported result uses fivefold depth |
| M | ≈5× M depth | Light/IWP | Pending | **38.2 ± 0.3†** | Fivefold-depth nonmatch |
| M | M-derived | Scalability CDLGN | — | 30.96 | Minimally modified M |
| M | ≈3.82M | WARP | Pending | — | No published exact M result |
| M | ≈3.82M | CoverageDLGN-Channel | Pending | — | New matched experiment |
| LogicTreeNet-L | ≈34.8M | Original/Soft-Mix | Pending | — | New transfer |
| L | ≈34.8M | Light/IWP | Pending | — | No published exact L result |
| L | ≈34.8M | WARP | Pending | — | No published exact L result |
| L | ≈34.8M | CoverageDLGN-Channel | Pending | — | Requires input/teacher decision |

## Pending controlled U2 evidence

These are preregistration targets, not achieved results. Parameterization and
fan-in are separate axes; do not run their full Cartesian product.

| Question | Required cells | Status |
|---|---|---|
| Full convolutional replication | LogicTreeNet-S random/U2 seeds 1-2; promote matched M seeds 1-2 only if S remains positive | Pending |
| Rank-2 parameterization independence | Raw random/U2 (existing) vs WARP random/U2 on dense CIFAR-10 M and LogicTreeNet-S | Pending WARP cells |
| Rank-four topology | Light rank-4 fixed random vs rank-4 U2 vs BitLogic learned-16 | Not implemented |
| Rank-four coordinates | BitLogic 2 x 16K and 2 x 64K CIFAR-10 protocols | Pending after implementation/tests |
| Physical cost | Matched synthesis/place-route for selected random/U2 checkpoints | Pending; simplified IR alone is not area |

The existing tables and U2 results are committed historical evidence. New
cells must retain validation/test and provenance labels and must not alter the
frozen V3/V4/U1/U2 definitions.
