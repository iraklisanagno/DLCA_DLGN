The complete paper-facing tables are below. They are also preserved in [PAPER_COMPARISON_TABLES.md](/ssd1/users/ianagno/my-repos/DLCA_DLGN_coverageDLGN/repos/torchlogix/experiments/coverage_dlgn/PAPER_COMPARISON_TABLES.md).

Notation:

- `A` = locally achieved; `R` = paper-reported.
- `T` = hardened test; `V` = hardened validation.
- `†` = nonmatched architecture, budget, fan-in, or protocol.
- Training time is the mean wall time for one run, not the total across seeds.
- GPU memory is peak PyTorch allocation. Additional hardware headroom is advisable.
- Topology construction is offline CPU preprocessing and is not included in GPU training time.
- `N/R` = not recorded; `—` = not available or not run.

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

## Table II. Dense CIFAR-10 S/M/L

| Scale | Method | Gates | Training parameters | Local test A | Reported R | Time / peak GPU | Comparison status |
|---|---|---:|---:|---:|---:|---:|---|
| S | Fixed-random DLGN | 48K | 0.768M | 49.056 ± 0.356 (5) | 51.27 | 12.24 min / 0.106 GiB | Exact local architecture |
| S | Mommen | 48K | 1.536M | 50.950 ± 0.244 (3) | — | 26.45 min / 0.480 GiB | Exact-budget adaptation |
| S | LILogicNet | 48K | 3.840M | 50.743 ± 0.574 (3) | 55.11†, 8K | 86.87 min / 3.949 GiB | Exact-budget local adaptation |
| S | WARP-LUT | 128K† | — | — | 52.12 ± 0.01† | — | Reported only |
| S | BitLogic rank 4 | 128K† | — | — | 58.06 ± 0.14† | — | Reported only |
| S | CoverageDLGN V3 | 48K | 0.768M | **52.358 ± 0.282 (5)** | — | **12.26 min / 0.106 GiB** | Exact; +3.302 pp |
| M | Fixed-random DLGN | 512K | 8.192M | 54.028 (5) | 57.39 | 41.55 min / 1.123 GiB | Exact local architecture |
| M | Mommen | 512K | 16.384M | 54.420 (1) | — | 4.84 h / 4.918 GiB | Exact-budget adaptation; one seed |
| M | LILogicNet | 64K† | — | — | 57.66 ± 0.17† | — | Reported only |
| M | WARP-LUT | 128K† | — | — | 52.12 ± 0.01† | — | Reported only |
| M | BitLogic rank 4 | 128K† | — | — | 58.06 ± 0.14† | — | Reported only |
| M | CoverageDLGN V3 | 512K | 8.192M | **58.284 (5)** | — | **41.48 min / 1.123 GiB** | Exact; +4.256 pp |
| L | Fixed-random DLGN | 1.28M | 20.480M | 55.960 ± 0.251 (5) | 60.78 | 1.89 h / 2.717 GiB | Exact local architecture |
| L | Mommen | 1.28M | 40.960M | 54.340 (1) | — | 13.39 h / 11.904 GiB | Exact-budget adaptation; one seed |
| L | LILogicNet-L | 256K† | — | — | 60.98 ± 0.19† | — | Reported only |
| L | WARP-LUT | 128K† | — | — | 52.12 ± 0.01† | — | Reported only |
| L | BitLogic rank 4 | 128K† | — | — | 58.06 ± 0.14† | — | Reported only |
| L | CoverageDLGN V3 | 1.28M | 20.480M | **61.020 ± 0.336 (5)** | — | **1.89 h / 2.717 GiB** | Exact; +5.060 pp |

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
| S | Channel-spatial pairing | Same declared S budget | 57.033 (3) | Not queried | — | 17.08 min / 1.831 GiB | −0.153 pp versus V4 |
| LogicTreeNet-M | Fixed routing | ≈3.08M paper ops; 668,416 LUTs; 6,995,968 spatial ops | 70.68 (1) | 69.57 (1) | 71.01 T | 25.47 h / 14.615 GiB | Exact architecture; 200K updates |
| M | Two-stage unit tying, 30% | ≈70% source gates | — | — | 70.77 ± 0.07 V | — | Reported only; different gate count |
| M | Scalability CDLGN-M | ≈3.08M reported | — | — | 65.23 | — | Reported only |
| M | Conv. TTNet-L | 189M reported | — | — | 70.75† | — | Much larger architecture |
| M | CoverageDLGN V4 | Same declared M budget | **71.26 (1)** | **69.96 (1)** | — | ≈25.49 h / N/R | +0.58 V pp; +0.39 T pp |
| LogicTreeNet-L | Fixed routing | ≈28.9M paper ops | — | — | 84.99 T | — | Requires 5-bit input and teacher |
| L | CoverageDLGN | Same declared L budget | — | — | — | — | Pending faithful baseline |

### Convolutional topology overhead

| Architecture | Method | Offline topology construction |
|---|---|---:|
| LogicTreeNet-S | Fixed random | 0.176 s |
| S | V4 | 0.413 s |
| S | U1 | 0.165 s |
| LogicTreeNet-M | Fixed random | 2.478 s |
| M | V4 | 6.182 s |

The M V4 topology adds 3.704 seconds relative to its random control. Its peak training allocation was not finalized because the run ended through controlled interruption.

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
| Paper-faithful S, 9 channels | Fixed random | 197,851 | 1.071 GiB | 3.057 ms; 41.87K examples/s | Reference |
| | V4 | 202,827 (+2.52%) | 1.073 GiB | 3.076 ms; 41.61K/s | +0.62% latency; no speed claim |
| | U1 | 214,883 (+8.61%) | 1.074 GiB | 3.095 ms; 41.35K/s | +1.26% latency; no speed claim |
| Paper-faithful M, 9 channels | Fixed random | 1,676,852 | 6.103 GiB | N/R | Trace and equivalence only |
| | V4 | 1,702,350 (+1.52%) | 6.092 GiB | N/R | Trace and equivalence only |
| WARP-style M, 6 channels | Matched random | 1,101,364 | 4.070 GiB | N/R | Separate protocol |
| | Legacy V4 | 1,129,547 (+2.56%) | 4.077 GiB | N/R | Separate protocol |

The declared LUT and spatial-operation budgets are identical within each matched architecture. Simplified IR nodes vary because checkpoint-dependent constants, wires, and duplicate functions can be removed.

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
| Fixed-random DLGN | 6 × 64K | 384K | 6.144M | 20.677 ± 0.522 T (3) | 22.54 ± 0.26 | 11.37 min / 0.795 GiB | 2.07 s | Exact architecture; local routing differs |
| CoverageDLGN V3 | 6 × 64K | 384K | 6.144M | **21.010 ± 0.131 T (3)** | — | 11.49 min / 0.795 GiB | 37.46 s | +0.333 pp; inconclusive |
| V3 + class head | 6 × 64K | 384K | 6.144M | 21.593 ± 0.133 V at 20K | — | 5.60 min / 0.795 GiB | 49.50 s | Negative ablation; no test |
| Fixed random | 6 × 256K | 1.536M | 24.576M | 11.680 V at 5K | 27.92 ± 0.43† | 32.03 min / 14.181 GiB | 34.60 s | Short screen only |
| Best screened V3 | 6 × 256K | 1.536M | 24.576M | 11.060 V at 5K | — | 31.59 min / 14.181 GiB | 554.86 s | Did not promote |
| Multilinear-CovJac | 6 × 256K | 1.536M | 4 parameters/gate | — | 28.37 ± 0.22 | — | — | Reported only |
| Multilinear-CovJac large | 6 × 1.28M | 7.68M | 4 parameters/gate | — | 32.72 ± 0.09 | — | — | Reported only |

CoverageDLGN’s 384K paired test gain is +0.333 pp with a 95% CI of [−0.816, +1.483], so it is not statistically conclusive.

## Table X. CIFAR-100 controlled depth study

All architectures use 384K gates, 6.144M gate parameters, three-threshold encoding, and 20K validation updates.

| Architecture | Random V | V3 V | Gain | Random/V3 time | Random/V3 GPU | Random/V3 topology time | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 × 128K | 21.080 | **21.860** | +0.780 pp | 5.33 / 5.35 min | 0.889 / 0.889 GiB | 3.34 / 47.98 s | Positive but below +1 pp threshold |
| 6 × 64K | 21.040 | **21.580** | +0.540 pp | 5.61 / 5.67 min | 0.795 / 0.795 GiB | 2.07 / 37.47 s | Three-seed reference |
| 12 × 32K | 1.180 | 1.180 | 0.000 pp | 6.82 / 6.85 min | 0.750 / 0.750 GiB | 1.12 / 33.79 s | Chance-level failure |
| 24 × 16K | 1.200 | 1.200 | 0.000 pp | 9.55 / 9.57 min | 0.723 / 0.723 GiB | 0.56 / 42.43 s | Chance-level failure |

At 24 layers, final-gate ancestry saturates all 3,072 image sources and cross-gate ancestry Jaccard reaches 1.0 for both methods. Greater ancestry coverage therefore cannot repair the optimization collapse.

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

The Markdown update remains uncommitted.
