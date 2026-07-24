# CoverageDLGN implementation and experiment history

**Updated:** July 24, 2026
**Status:** Dense semantic-balanced v3 and convolutional semantic-channel v4
are implemented. The original five-seed 48K study, a five-seed 512K
escalation, a two-budget/three-depth study, and a three-seed convolutional
pilot have frozen validation/test artifacts. V3 improves 512K dense CIFAR-10
held-out accuracy by **+4.256 pp**; v4 improves the convolutional pilot by
**+2.000 pp**. Component ablations, long-training convolutional reproduction,
and protocol-identical named-method comparisons remain before a DATE claim.

## Paper-architecture convolutional correction

An audit against pages 6, 14, and 15 of the convolutional DLGN paper found
that the legacy `ClgnCifar10Small`/`Medium` classes use two thresholds (six RGB
Boolean channels), while the paper states that its 2-bit S/M input is encoded
with three thresholds (nine channels). Separate
`ClgnCifar10PaperSmall`/`Medium` classes now implement the corrected encoding;
legacy classes were not changed, preserving the original v4 pilot.

The corrected S class still has exactly 83,552 learned gate functions and
874,496 spatial gate applications. Thus, random and v4 receive the same
architecture correction without changing their paired gate budget.

Paired CUDA smoke runs passed for S and M. Both saved `[1, 3]` thresholds,
random/v4 state tensor shapes were identical, and all four convolutional
spatial-coordinate hashes matched. Peak memory was 508.25/508.27 MiB for S
and 1178.63/1178.83 MiB for M (random/v4). S ran two updates plus hard
validation; M ran one update without evaluation. These are execution tests,
not accuracy measurements.

A 20-step M timing run at batch 128 used 14.61 GiB peak GPU memory and took
9.49 seconds (0.475 seconds/step including setup). A 20K M run is projected at
about 2.6 hours before evaluation overhead, or roughly eight hours for the
three-seed paired queue on two GPUs.

The corrected S three-seed 20K pilot and frozen held-out evaluation are
complete:

| Seed | Random hard val | V4 hard val | Val gain | Random hard test | V4 hard test | Test gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 56.10% | 57.60% | +1.50 pp | 55.15% | 57.73% | +2.58 pp |
| 1 | 57.40% | 57.80% | +0.40 pp | 56.35% | 56.21% | -0.14 pp |
| 2 | 56.52% | 56.16% | -0.36 pp | 56.92% | 55.16% | -1.76 pp |
| Mean | **56.673%** | **57.187%** | **+0.513 pp** | **56.140%** | **56.367%** | **+0.227 pp** |

The paired 95% interval is [-1.810, +2.836] pp on validation and
[-5.221, +5.675] pp on held-out test. Both are inconclusive. S therefore
provides a positive mean validation signal but not a statistically supported
accuracy claim. The published S result is 60.38%; this short 20K pilot is not
a numerical reproduction of the paper's long schedule.

### Ancestry-aware v5 and complementary classifier-tail study

The next revision was intentionally topology-only and preserved v3 and v4 as
separate strategies. `ancestry_channel_hybrid` propagates packed channel
ancestry through all four convolution blocks, starts from balanced
round-robin channel pairs, and applies degree-preserving swaps scored by
ancestry and predecessor novelty. It never changes spatial coordinates,
learned gates, or the paper-S architecture. `semantic_classifier_hybrid`
adapts v3 to the shrinking dense classifier: its first classifier layer uses
the semantic schedule, while later layers use exact balanced matchings before
v3 ancestry swaps.

A controlled seed-0 5K screen used identical initialization weights, data,
training effort, and 83,552-gate budget:

| Strategy | Best hard validation | Gain over random |
|---|---:|---:|
| Controlled random | 52.52% | — |
| Frozen v4 + v3 classifier tail | 53.94% | +1.42 pp |
| Ancestry v5, random classifier | **55.36%** | **+2.84 pp** |
| Ancestry v5 + v3 classifier tail | 55.16% | +2.64 pp |

The classifier tail was therefore not added to v5. The selected ancestry-only
candidate was frozen and run for 20K steps on three paired seeds:

| Seed | Random hard val | Ancestry v5 hard val | Paired gain |
|---:|---:|---:|---:|
| 0 | 55.00% | 56.88% | +1.88 pp |
| 1 | 57.96% | 56.98% | -0.98 pp |
| 2 | 58.46% | 57.30% | -1.16 pp |
| Mean | **57.140%** | **57.053%** | **-0.087 pp** |

The paired 95% Student-t interval is [-4.324, +4.150] pp. V5 therefore fails
the requested +3 pp criterion and is effectively tied with random. It was not
evaluated on held-out test data because it failed validation selection.
Frozen v4 remains the primary convolutional mechanism; v3 remains the primary
dense mechanism. V5 is retained as a reproducible negative-result/diagnostic
extension rather than presented as an accuracy improvement.

The topology hypothesis itself was achieved. Mean distinct channel groups
changed from `[8, 31, 127, 511]` to `[32, 128, 512, 1024]`; mean raw-channel
ancestry at the last block rose from 5.414 to 8.195 of nine inputs; and final
predecessor Jaccard overlap fell from 0.575 to 0.488. Fan-out CV became zero
after the first block. Mean convolutional topology construction increased
from 0.175 to 0.363 seconds, an offline +0.189-second cost; peak training GPU
allocation remained 1,874.6 MiB for both variants. The accuracy result shows
that maximizing these diagnostics too aggressively can remove useful channel
reuse.

Exact values are in
`summary/paper_conv_small_ancestry_v5_pilot.json`. Historical V5 artifacts
have correct per-layer strategy labels, but their top-level
`conv_topology.json` metadata says `random` because it used the global
fallback label. This reporting-only defect was found after the runs and fixed
with a regression test; model indices and all accuracy results are unaffected.
The exact three-seed queue is reproduced with:

```bash
bash experiments/coverage_dlgn/run_conv_cifar10_paper_small_ancestry_v5_pilot_two_gpus.sh
```

### Coverage--reuse-balanced refinement

V5 established that maximum ancestry diversity can remove useful repeated
channel motifs. A generic follow-up, `coverage_reuse_hybrid`, therefore starts
from the frozen v3/v4 topology and applies only score-improving two-edge swaps.
Its score combines ancestry coverage, within-gate overlap, cross-gate novelty,
and the frequency of predecessor pairs in the base topology. Every swap
preserves the exact predecessor degree sequence. Convolutional use changes
channel groups only and retains bit-identical spatial coordinates.

One parameter set was frozen before the S/M screens: candidate pool 8, v4 base
swap fraction 0.25, maximum refinement change fraction 0.25, novelty weight
1.0, and reuse weight 1.0. No layer- or dataset-specific schedule was used.

| Architecture / protocol | V4 best/current hard val | Coverage--reuse best/current hard val | Difference |
|---|---:|---:|---:|
| Paper S, 5K steps, best | 54.92% | **55.40%** | +0.48 pp |
| Paper M, 1K steps, current; eval every 500 | 35.78% | **38.92%** | +3.14 pp |
| Paper M confirmation, 5K steps, best/current | **61.56%** | 58.44% | -3.12 pp |

The M confirmation learning curve was:

| Step | V4 hard val | Coverage--reuse hard val | Difference |
|---:|---:|---:|---:|
| 1,000 | 35.98% | 35.52% | -0.46 pp |
| 2,000 | 45.84% | 46.84% | +1.00 pp |
| 3,000 | 52.86% | 50.04% | -2.82 pp |
| 4,000 | 56.78% | 55.36% | -1.42 pp |
| 5,000 | **61.56%** | 58.44% | **-3.12 pp** |

The earlier +3.14 pp M observation was protocol-sensitive: that 1K screen
evaluated every 500 steps, while the confirmatory run evaluated every 1,000
steps. Only iteration count, evaluation frequency, and output directory
differed between those protocols; each paired comparison itself remained
controlled. The early observation is therefore not treated as confirmation.

Paper M used 668,416 learned gates in both cases. Peak GPU allocation was
identical at 15,692,077,568 bytes. Convolutional topology construction took
4.884 seconds for v4 and 15.498 seconds for coverage--reuse, and total wall
time was 2,314.5 versus 2,334.2 seconds. This is a modest one-time construction
cost, but it was not justified by accuracy.

The requested stable +3 pp success criterion was not met. The planned
multi-seed 20K escalation and held-out test evaluation were stopped by design,
preventing selection leakage and unnecessary computation. V4 remains the
retained convolutional mechanism; the generic coverage--reuse implementation
is preserved as a reproducible negative result. Exact values and artifact
paths are in `summary/paper_conv_coverage_reuse_screen.json`.

The legacy `ClgnCifar10Large` has `k=512`, so its scale corresponds to paper B,
not L, but it is not a faithful B model. B additionally needs a doubled final
output layer, fixed edge/curvature preprocessing, and teacher supervision.
Because the PDF does not fully specify the detector implementation, no
paper-exact B result is claimed.

An attempt to detach the eight-hour M queue with `nohup` from the managed
command sandbox was terminated when that command session exited; the sandbox
uses parent-lifetime process cleanup. No training update or checkpoint was
produced, so this is a launch-environment failure rather than a model result.
The two startup logs and queue metadata are retained under
`results/failed/paper_medium_async_launch_attempt/`. Run the documented M
script in a persistent foreground terminal instead.

## Untouched baseline first

Before source changes:

1. The TorchLogix environment was recreated with Python 3.12.13 and
   PyTorch 2.9.0+cu130. Python 3.10 could not parse `circuit.py` because this
   checkout uses Python 3.12 f-string grammar.
2. The untouched test suite passed: **3,248 passed, 3,038 skipped, one warning**
   in 147 seconds.
3. `DlgnMnistTiny`, fixed legacy-random routing, seed 0, batch 128, 1,000 GPU
   steps reached **84.34% hard** and **84.65% soft** validation accuracy.
   Artifacts are in `results/prechange_baseline_smoke_seed0/`.
4. That run exposed a pre-existing reporting defect: the correct best checkpoint
   was saved at 84.34%, but the terminal summary printed 0.0%. The training
   runner now updates and reports the actual best hard validation accuracy.

Exact pre-change command:

```bash
DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python experiments/train.py \
  --dataset mnist --architecture DlgnMnistTiny --device cuda --seed 0 \
  --batch-size 128 --num-iterations 1000 --eval-freq 500 \
  --binarization-num-batches 1 --connections fixed \
  --connections-init-method random \
  --output experiments/coverage_dlgn/results/prechange_baseline_smoke_seed0
```

## Paper-scale MNIST-small reproduction

After the initial smoke study, an explicit `DlgnMnistPaperSmall` architecture
was added to match Table 6 of Petersen et al. (2022): six rank-2 logic layers,
8,000 gates per layer, 48,000 deployed gates, 768,000 raw training logits, and
temperature 10. This avoids the pre-existing `DlgnMnistSmall` naming ambiguity:
that class contains five layers and 40,000 gates.

The paired protocol used raw parametrization with standard-normal initialization,
fixed random or hybrid-v2 connections, Adam at 0.01, batch size 100, and 108,000
steps. With 54,000 training examples, this is exactly 200 epochs. The data split
was frozen with split seed 2027; training and topology seeds were 0, 1, and 2.
Validation and test loaders included every example without shuffling. The held-out
test set was evaluated once, only after the topology fraction, all training runs,
and best-validation checkpoints were frozen.

| Seed | Random best hard val | Hybrid-v2 best hard val | Val difference | Random hard test | Hybrid hard test | Test difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97.1167% | 97.1667% | +0.05 pp | 97.33% | 97.29% | -0.04 pp |
| 1 | 97.1167% | 97.1667% | +0.05 pp | 97.00% | 97.15% | +0.15 pp |
| 2 | 97.2167% | 97.2167% | +0.00 pp | 97.22% | 97.32% | +0.10 pp |
| Mean | **97.1500%** | **97.1833%** | **+0.033 pp** | **97.1833%** | **97.2533%** | **+0.070 pp** |

The validation paired Student-t interval is +0.033 ± 0.072 pp; the held-out test
interval is +0.070 ± 0.245 pp. Both are inconclusive. The corrected hybrid
therefore **matches** the paper-scale random baseline but does not establish an
accuracy gain. The random reproduction is 0.51 pp below the paper's 97.69% mean;
our frozen 10% validation split leaves 10% fewer training examples, and TorchLogix
is an independent implementation, so this is close but not an exact numerical
reproduction.

Topology moved in the intended direction: final mean ancestry increased from
61.49 to 62.51 inputs and mean overlap decreased from 1.25 to 0.96. Both methods
used the same 1.2 million routing-index bits and approximately 102 MiB peak GPU
memory. Mean training time was about 15 minutes. Mean topology construction time
was 0.025 seconds for random and 1.94 seconds for hybrid; this is a one-time,
offline cost and does not change the deployed gate count.

Exact configurations and one-shot test metrics are under `configs/` and the six
`results/paper_mnist_small_*` directories. Generated paired reports are under
`summary/paired_paper_mnist_*`.

## Semantic-balanced v3: five-seed central result

### Diagnostic basis and mechanism revision

The v2 bit-level metric was misleading for thermometer-encoded images. It
treated the three correlated threshold bits from one raw pixel/channel-pixel as
three independent sources. Re-analysis of the frozen v2 checkpoints found:

- Fashion v2 paired two thresholds from the same pixel in **22.0%** of
  first-layer gates and reduced final raw-pixel ancestry from **61.51** for
  random to **55.15**, despite increasing bit ancestry;
- CIFAR v2 changed median first-layer spatial distance from about 20--21 pixels
  to one pixel, same-channel pairs from 34.7% to 75.8%, and same-threshold pairs
  from 33.6% to 85.8%; this structured image locality, rather than the
  negligible bit-ancestry increase, plausibly explains its CIFAR gain; and
- replacing individual butterfly gates raised final fan-out CV from zero to
  0.222 on Fashion and 0.161 on CIFAR, leaving about 81 Fashion predecessors
  unused.

V3 addresses those observations without trainable routing:

1. It tracks raw-source ancestry, collapsing correlated threshold bits from the
   same scalar input.
2. Its first layer uses a deterministic tensor-aware butterfly over
   `(channel, y, x, threshold)` coordinates. Spatial scales and semantic axes
   are interleaved, same-source pairs are forbidden, and bounded look-ahead
   keeps the predecessor degrees balanced.
3. Deeper layers start from the affine butterfly and accept only
   degree-preserving two-edge swaps. The score combines normalized source
   ancestry, predecessor overlap, and bounded cross-gate novelty. Because each
   swap reuses the same four predecessor occurrences, the complete fan-out
   sequence is invariant.
4. The first layer is not swapped, preserving its image structure. The frozen
   controls are candidate pool 8, swap fraction 0.25, and novelty weight 1.0.

A 64-candidate full-scale construction profile exceeded one minute and was
stopped. The bounded pool of 8 completed paper-scale construction in about
5 seconds on Fashion and 9 seconds on CIFAR, and was frozen before the central
runs.

### Selection protocol

Three paired 20,000-step CIFAR validation pilots selected the fixed v3
mechanism: random averaged 49.04%, v3 averaged 52.19%, a +3.15 pp difference
with paired 95% CI [+0.98, +5.32] pp. The settings were then frozen.

The same settings were used for the full CIFAR runs and applied to
Fashion-MNIST without retuning. All runs use the same architecture, split seed,
training/topology seed pairing, batch size, optimizer, parameterization,
48,000-gate budget, and 108,000 steps as their random controls. Best checkpoints
were selected only by hardened validation accuracy. Held-out test evaluation
was performed after the central checkpoints were frozen.

### Hardened accuracy

| Dataset / seed | Random best val | V3 best val | Difference | Random test | V3 test | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Fashion 0 | 87.6667% | 88.2500% | +0.583 pp | 86.58% | 87.37% | +0.79 pp |
| Fashion 1 | 87.4667% | 87.8833% | +0.417 pp | 86.22% | 86.73% | +0.51 pp |
| Fashion 2 | 87.4000% | 88.3167% | +0.917 pp | 86.11% | 87.43% | +1.32 pp |
| Fashion 3 | 87.2167% | 88.4000% | +1.183 pp | 86.41% | 87.10% | +0.69 pp |
| Fashion 4 | 87.6333% | 87.9833% | +0.350 pp | 86.22% | 87.16% | +0.94 pp |
| **Fashion mean** | **87.4767%** | **88.1667%** | **+0.690 pp** | **86.3080%** | **87.1580%** | **+0.850 pp** |
| CIFAR 0 | 50.1600% | 52.8600% | +2.70 pp | 48.84% | 52.40% | +3.56 pp |
| CIFAR 1 | 49.0400% | 53.4200% | +4.38 pp | 49.29% | 52.68% | +3.39 pp |
| CIFAR 2 | 49.3400% | 53.0600% | +3.72 pp | 48.61% | 52.09% | +3.48 pp |
| CIFAR 3 | 49.8200% | 53.1400% | +3.32 pp | 49.51% | 52.05% | +2.54 pp |
| CIFAR 4 | 50.1000% | 53.1000% | +3.00 pp | 49.03% | 52.57% | +3.54 pp |
| **CIFAR mean** | **49.6920%** | **53.1160%** | **+3.424 pp** | **49.0560%** | **52.3580%** | **+3.302 pp** |

Paired 95% Student-t intervals:

- Fashion validation: **+0.690 pp**, CI **[+0.253, +1.127] pp**;
- Fashion test: **+0.850 pp**, CI **[+0.470, +1.230] pp**;
- CIFAR validation: **+3.424 pp**, CI **[+2.611, +4.237] pp**;
- CIFAR test: **+3.302 pp**, CI **[+2.767, +3.837] pp**.

V3 clears the +0.3 pp mean threshold on both datasets, every paired validation
seed is positive, and all four five-seed intervals exclude zero. The official
test sets had previously been evaluated for v2; the v3 mechanism and settings
were based on topology and validation diagnostics, not on v3 test results, but
this prior test visibility remains an honest experimental-history caveat.

### Topology, cost, and runtime

| Dataset / metric | Random mean | V3 mean | V3 minus random |
|---|---:|---:|---:|
| Fashion final raw-source ancestry | 61.5108 | 62.0446 | +0.5339 |
| CIFAR final raw-source ancestry | 15.9611 | 15.9743 | +0.0131 |
| Fashion final fan-out CV | 0 | 0 | 0 |
| CIFAR final fan-out CV | 0 | 0 | 0 |
| Fashion unused final predecessors | 0 | 0 | 0 |
| CIFAR unused final predecessors | 0 | 0 | 0 |
| Fashion topology construction | 0.029 s | 5.051 s | +5.022 s |
| CIFAR topology construction | 0.048 s | 9.334 s | +9.287 s |
| Fashion topology temporary peak | 15.38 MiB | 1.71 MiB | -13.67 MiB |
| CIFAR topology temporary peak | 34.61 MiB | 8.97 MiB | -25.63 MiB |
| Fashion peak GPU allocation | 101.75 MiB | 101.75 MiB | 0 |
| CIFAR peak GPU allocation | 108.66 MiB | 108.66 MiB | 0 |
| Fashion training time | 903.8 s | 891.1 s | -12.7 s |
| CIFAR training time | 734.4 s | 735.9 s | +1.5 s |
| Fashion hardened inference | 1.1364 ms/batch | 1.1359 ms/batch | -0.0005 ms |
| CIFAR hardened inference | 0.7856 ms/batch | 0.7849 ms/batch | -0.0007 ms |

Gate count, LUT logits, routing bits, packed circuit storage, and inference
operations are identical to random. V3 adds only an offline construction
procedure; the deployed model still stores the same fixed index tensors.

Machine-readable five-seed reports are
`summary/paper_fashion_mnist_semantic_v3.json` and
`summary/paper_cifar10_semantic_v3.json`. Learning curves include random, v2,
and v3. The reproducibility audit records two broad source-hash cohorts for v3:
reporting and queue/config files were added between seeds 0--2 and 3--4, while
training implementation files and resolved protocol fields were unchanged.
Future manifests additionally record a training-only implementation hash so
reporting edits cannot create this ambiguity.

## Fashion-MNIST and CIFAR-10 v2 paper-architecture results

This section is retained as the superseded negative result that motivated v3.

The Fashion-MNIST experiment uses the fixed-connectivity architecture reported
by Mommen et al. (2025): six rank-2 layers of 8,000 gates, 48,000 gates total,
three fixed uniform thresholds at 0.25, 0.5, and 0.75, GroupSum temperature 10,
Adam at 0.01, batch size 100, and 200 epochs. With the frozen 54,000-example
training subset this is 108,000 optimizer steps. A separate
`DlgnFashionMnistPaperSmall` class was added because the pre-existing
`DlgnFashionMnistSmall` has five layers and only 40,000 gates. The audit also
found that the Fashion base class replaced the supported `fixed` binarizer name
with the unsupported string `uniform`; this defect was removed and covered by a
regression test.

The central fraction was frozen at 0.25 before these runs. All central
checkpoints were selected only by hardened validation accuracy. Held-out test
evaluation was run once after all 16 training runs and the fraction sweep had
finished.

### Hardened accuracy

| Dataset / seed | Random best hard val | Hybrid best hard val | Val difference | Random hard test | Hybrid hard test | Test difference |
|---|---:|---:|---:|---:|---:|---:|
| Fashion 0 | 87.6667% | 87.5667% | -0.10 pp | 86.58% | 86.37% | -0.21 pp |
| Fashion 1 | 87.4667% | 87.6667% | +0.20 pp | 86.22% | 86.25% | +0.03 pp |
| Fashion 2 | 87.4000% | 87.1833% | -0.217 pp | 86.11% | 86.43% | +0.32 pp |
| **Fashion mean** | **87.5111%** | **87.4722%** | **-0.039 pp** | **86.3033%** | **86.3500%** | **+0.047 pp** |
| CIFAR-10 0 | 50.1600% | 51.4000% | +1.24 pp | 48.84% | 50.78% | +1.94 pp |
| CIFAR-10 1 | 49.0400% | 51.0000% | +1.96 pp | 49.29% | 50.43% | +1.14 pp |
| CIFAR-10 2 | 49.3400% | 51.8600% | +2.52 pp | 48.61% | 50.80% | +2.19 pp |
| **CIFAR-10 mean** | **49.5133%** | **51.4200%** | **+1.907 pp** | **48.9133%** | **50.6700%** | **+1.757 pp** |

The paired 95% Student-t intervals are:

- Fashion validation: **-0.039 pp**, CI **[-0.573, +0.495] pp**;
- Fashion test: **+0.047 pp**, CI **[-0.613, +0.706] pp**;
- CIFAR-10 validation: **+1.907 pp**, CI **[+0.313, +3.501] pp**;
- CIFAR-10 test: **+1.757 pp**, CI **[+0.394, +3.119] pp**.

The CIFAR gain is consistent across all three seeds and its three-seed interval
excludes zero. This is encouraging pilot evidence, not the final DATE claim:
the specification requires five central seeds. Fashion is a tie and does not
meet the required +0.3 pp mean improvement. Both methods use the same gate and
routing budgets, so the alternative Pareto condition is not met.

The original deep-DLGN paper reports 51.27% test accuracy for this small CIFAR
architecture. Our random mean is 48.91%, 2.36 pp lower, while the hybrid mean is
50.67%. This is an architecture reproduction, not a protocol-identical
numerical reproduction: the local protocol reserves 10% of the 50,000 training
examples for frozen validation, whereas the paper trains on the complete
training set. The paper uses 200 epochs; 108,000 local steps on 45,000 examples
correspond to 240 epochs. This mismatch must be resolved before comparison with
the published 51.27% result.

### Fashion fraction sweep

Only validation accuracy was used for the seed-0 fraction sweep:

| Long-range fraction | Best hard val | Final mean ancestry | Final overlap |
|---:|---:|---:|---:|
| 0.00 | 87.4667% | 63.1206 | 0.4561 |
| 0.25 | 87.5667% | 63.4958 | 0.3125 |
| 0.50 | 87.3000% | 63.6898 | 0.3103 |
| 0.75 | 87.7000% | 63.8860 | 0.1140 |
| 1.00 | 87.5167% | 64.0000 | 0.0000 |

Coverage and overlap improve monotonically overall, but accuracy does not.
Fraction 0.75 is only +0.033 pp above the paired seed-0 random baseline, so the
sweep does not rescue Fashion and does not justify post-hoc retuning.

Two seed-1 attempts were terminated externally with signal 15, first near step
24,000 and then near step 72,000. Neither emitted a Python traceback, CUDA error,
or failed metric. Their partial artifacts are retained under `results/failed/`.
Long experiments were subsequently detached from the command session and
redirected to per-run `console.log` files; both GPUs were then confirmed active
by `nvidia-smi`.

A topology-only attempt passed a full training JSON file to
`analyze_topology.py`; the command correctly rejected unrelated training keys.
The analysis was rerun with explicit dimensions and topology arguments. On the
Fashion paper architecture, increasing the hybrid fraction from 0 to 1 increased
final mean ancestry from 63.12 to the theoretical maximum 64.00 and reduced mean
overlap from 0.46 to 0.00.

The dependent CIFAR-10 queue used the Petersen et al. (2022) small architecture:
four rank-2 layers of 12,000 gates, 48,000 gates total, three uniform color
thresholds, GroupSum temperature `1/0.03`, batch size 100, Adam at 0.01, and
108,000 steps. It deliberately disables augmentation because the published
experiment states that no augmentation or dropout was used. It waited for all
Fashion pairs and fraction ablations, ran one complete fixed-random seed first,
completed all three random seeds, and only then ran the three matched hybrid
seeds using the predeclared fraction 0.25.

### Topology, circuit cost, runtime, and memory

Both datasets use exactly 48,000 rank-2 gates and 768,000 trainable LUT logits.
Random and hybrid have identical deployed storage and operation counts:

| Dataset | Routing bits | Gate-function bits | Total packed circuit bytes | PyTorch index-buffer bytes |
|---|---:|---:|---:|---:|
| Fashion-MNIST | 1,232,000 | 192,000 | 178,000 | 768,000 |
| CIFAR-10 | 1,344,000 | 192,000 | 192,000 | 768,000 |

The packed totals exclude thresholds and small classifier metadata. Hybrid adds
no trainable or deployed routing parameter.

| Dataset / metric | Random mean | Hybrid mean | Hybrid minus random |
|---|---:|---:|---:|
| Fashion final ancestry | 63.1593 | 63.5172 | +0.3579 |
| Fashion final overlap | 0.4260 | 0.3083 | -0.1177 |
| CIFAR final ancestry | 15.9885 | 15.9943 | +0.0058 |
| CIFAR final overlap | 0.00608 | 0.00450 | -0.00158 |
| Fashion training time | 910.0 s | 902.2 s | -7.8 s |
| CIFAR training time | 731.5 s | 737.3 s | +5.8 s |
| Fashion peak GPU allocation | 101.75 MiB | 101.75 MiB | 0 |
| CIFAR peak GPU allocation | 108.66 MiB | 108.66 MiB | 0 |
| Fashion topology construction | 0.030 s | 2.468 s | +2.439 s |
| CIFAR topology construction | 0.045 s | 4.828 s | +4.783 s |
| Fashion topology temporary peak | 15.38 MiB | 4.64 MiB | -10.74 MiB |
| CIFAR topology temporary peak | 34.61 MiB | 26.55 MiB | -8.06 MiB |

Random has perfectly balanced final-layer fan-out (CV 0); hybrid CV is 0.222
on Fashion and 0.161 on CIFAR. Hybrid construction is slower but is a one-time,
offline cost and uses less temporary memory than the current random-analysis
implementation.

Hardened PyTorch GPU forward-pass benchmarks use one fixed test batch of 100,
20 warm-up batches, 100 timed batches, and exclude input transfer:

| Dataset | Random | Hybrid | Paired difference |
|---|---:|---:|---:|
| Fashion-MNIST | 1.1352 ms/batch | 1.1359 ms/batch | +0.0006 ms |
| CIFAR-10 | 0.7783 ms/batch | 0.7776 ms/batch | -0.0007 ms |

These are framework-level GPU measurements, not exported/compiled circuit
latencies, and must not be compared directly with the paper's nanosecond
circuit benchmarks. They confirm no runtime penalty at equal tensor shapes.

### Learning curves

Every paper-architecture run has 54 validation records (steps 2,000 through
108,000). Fashion curves nearly overlap. On CIFAR, the hybrid advantage is
+1.16 pp at step 2,000, +1.27 pp at step 10,000, +2.09 pp at step 54,000, and
+1.86 pp at the final step. The gain is not caused only by selecting a favorable
early checkpoint. Regenerable mean/std data and SVGs are in
`summary/learning_curves.csv`, `summary/learning_curves_fashion_mnist.svg`, and
`summary/learning_curves_cifar10.svg`.

## Controlled seed-0 smoke results

All post-change rows use the same MNIST split, training seed, topology seed,
architecture, gate budget, optimizer, batch size, and 1,000 training steps.
Accuracy is validation accuracy; the test set was not used.

| Schedule / attempt | Hard | Soft | Final mean ancestry | Final overlap | Final fan-out CV | Outcome |
|---|---:|---:|---:|---:|---:|---|
| paired random | 83.14% | 83.92% | 31.36 | 0.297 | 0.000 | control |
| random unique | 81.28% | 82.39% | 31.09 | 0.565 | 0.695 | below control |
| local cyclic v1 | 38.71% | 46.64% | 12.23 | 9.358 | 0.000 | failed mixing |
| butterfly v1, unshuffled | 42.78% | 52.00% | 28.79 | 0.214 | 0.000 | output-order failure |
| pure coverage greedy | 81.76% | 82.64% | 32.00 | 0.000 | 0.205 | near control, lower |
| hybrid v1, fraction 0.25 | 60.53% | 75.80% | 30.33 | 0.166 | 0.238 | regular-base failure |
| hybrid v1, fraction 0.75 | 81.27% | 81.81% | 31.95 | — | — | recovered, lower |
| hybrid v1, fraction 0.90 | 83.17% | 83.90% | 31.96 | — | — | tied, weak regularity claim |
| butterfly v2, affine ordered | 83.08% | 83.64% | 31.78 | 0.121 | 0.000 | ordering fix worked |
| hybrid v2, fraction 0.25 | 82.97% | 84.04% | 31.68 | 0.224 | 0.238 | intended mechanism, near control |

The v1 to v2 change does not alter the regular pair set or fan-out. It applies
a deterministic affine ordering to distribute regular pairs across consecutive
output and class groups. This isolated a major confound in flattened-image
DLGNs: a mathematically regular topology can perform badly solely because gate
ordering interacts with the grouped classifier.

## Three paired smoke seeds

The corrected 25% hybrid was compared with independent-seed random routing for
seeds 0, 1, and 2:

| Seed | Random hard | Hybrid-v2 hard | Hybrid minus random |
|---:|---:|---:|---:|
| 0 | 83.14% | 82.97% | -0.17 pp |
| 1 | 82.64% | 83.02% | +0.37 pp |
| 2 | 83.31% | 82.46% | -0.85 pp |
| Mean | 83.03% | 82.81% | **-0.22 pp** |

The paired-difference standard deviation is 0.61 pp. The smoke-only normal
approximation interval is -0.22 ± 0.69 pp. This is inconclusive and is not a
paper claim. Raw values and the generated paired report are under `summary/`.

Topology changed in the intended direction: mean final ancestry increased from
31.38 (random, three-seed mean) to 31.71 (hybrid-v2), while accuracy did not
improve at this short budget. This directly matches the brief's first stated
risk: increased coverage may not increase accuracy.

## 512K-gate CIFAR-10 escalation

The frozen v3 settings (candidate pool 8, swap fraction 0.25, novelty weight
1.0) first passed a paired three-seed, 20K-step `DlgnCifar10Medium` pilot with
a +4.473 pp mean hard-validation gain. This triggered the predeclared five-seed,
108K-step escalation.

| Seed | Random hard val | V3 hard val | Val gain | Random hard test | V3 hard test | Test gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.40% | 59.98% | +4.58 pp | 53.90% | 58.45% | +4.55 pp |
| 1 | 55.34% | 59.34% | +4.00 pp | 54.21% | 58.07% | +3.86 pp |
| 2 | 54.74% | 59.44% | +4.70 pp | 54.18% | 58.55% | +4.37 pp |
| 3 | 55.00% | 59.52% | +4.52 pp | 53.86% | 58.40% | +4.54 pp |
| 4 | 54.94% | 59.52% | +4.58 pp | 53.99% | 57.95% | +3.96 pp |
| Mean | **55.084%** | **59.560%** | **+4.476 pp** | **54.028%** | **58.284%** | **+4.256 pp** |

Paired 95% Student-t intervals are [+4.136, +4.816] pp on validation and
[+3.851, +4.661] pp on held-out test. Hardened inference remains the same
computation: about 3.57 ms/random versus 3.53 ms/v3 per 100 examples. V3 adds
a one-time offline construction cost: 107 seconds on average versus 5 seconds
for seeded random, while deployed gates and index tensor shapes are identical.

The initial medium v3 construction took 209 seconds, including 114 seconds in
the first layer. An exact lazy-heap implementation of the exhausted-stage
selection rule reduced this to 106 seconds, including 11 seconds in the first
layer. Frozen small and medium indices were bit-identical before and after.

## Controlled gate-budget/depth study

This 20K-step study uses three paired seeds and exact 48K/512K budgets. Width is
adjusted with depth, and GroupSum temperature preserves maximum class-logit
scale within each budget. The 4-layer cells reuse protocol-identical pilots.

| Gates | Layers | Random mean hard val | V3 mean hard val | Paired gain | Paired 95% t interval |
|---:|---:|---:|---:|---:|---:|
| 48K | 4 | 49.040% | 52.193% | +3.153 pp | [+0.985, +5.322] |
| 48K | 8 | 44.907% | 47.553% | +2.647 pp | [+0.013, +5.280] |
| 48K | 12 | 14.087% | 12.367% | -1.720 pp | [-15.325, +11.885] |
| 512K | 4 | 54.820% | 59.293% | +4.473 pp | [+3.624, +5.323] |
| 512K | 8 | 48.413% | 52.007% | +3.593 pp | [+2.151, +5.035] |
| 512K | 12 | 11.807% | 11.260% | -0.547 pp | [-3.873, +2.779] |

V3 is robust at four and eight layers for both budgets. Both twelve-layer cells
are unstable or near chance; v3 does not repair this general optimization
failure. Depth is an architecture variable, not a v3 parameter.

## Convolutional semantic-channel v4 pilot

Only after the dense depth study completed, the frozen butterfly/swap rule was
applied across the two-channel groups of each `LogicConv2d`. For every paired
seed, all four spatial-coordinate hashes and all initial gate weights are
identical; only channel IDs change. Dense tail routing is also paired.

`ClgnCifar10Small` has four depth-3 logic-tree convolution blocks with OR
pooling, followed by three dense logic layers and GroupSum: 83,552 learned gate
functions and 874,496 spatial gate applications. The 20K-step pilot uses
standard crop/flip augmentation, AdamW, batch 128, learning rate 0.02, weight
decay 0.002, residual initialization, and the frozen 45K/5K split.

| Seed | Random hard val | V4 hard val | Val gain | Random hard test | V4 hard test | Test gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.30% | 56.86% | +2.56 pp | 54.79% | 56.85% | +2.06 pp |
| 1 | 55.82% | 57.80% | +1.98 pp | 56.03% | 58.78% | +2.75 pp |
| 2 | 55.42% | 56.18% | +0.76 pp | 55.44% | 56.63% | +1.19 pp |
| Mean | **55.180%** | **56.947%** | **+1.767 pp** | **55.420%** | **57.420%** | **+2.000 pp** |

The validation interval is inconclusive at [-0.516, +4.049] pp; the held-out
interval is narrowly positive at [+0.058, +3.942] pp. This is pilot evidence,
not a long-training published-baseline reproduction. Mean hardened latency is
unchanged: 6.868 ms/random and 6.869 ms/v4 per 128 examples; peak GPU memory is
1.831 GiB for both.

Random versus v4 distinct channel groups are 5 versus 14, 31 versus about 96,
127 versus about 461, and 511 versus 1,024 across the four convolution blocks.
V4 channel fan-out CV is 0.044 in the six-channel input block and zero
thereafter, compared with 0.285, 0.136, 0.068, and 0.035 for random.

## Failures and fixes retained for history

1. **Python 3.10 import failure.** `circuit.py` requires Python 3.12 grammar.
   Fixed by recreating `venv` with `uv` and Python 3.12; research code was not
   downgraded.
2. **Best-accuracy terminal report always 0.0.** The local accumulator was never
   updated. Fixed and covered by actual run summaries.
3. **Tiny hybrid construction stall.** A candidate pool of 64 requested more
   unique long-range pairs than an eight-node cycle contains. Candidate search
   is now bounded and permits duplicate candidates after exhaustion; regression
   test added.
4. **Topology config ignored required output.** Argparse still required the CLI
   flag even when JSON supplied it. Output is now validated after config defaults.
5. **Training module import failure.** `from utils` worked only for direct script
   execution. Both package import and direct execution are now supported.
6. **Unshuffled regular schedule collapse.** Consecutive butterfly outputs fed
   consecutive `GroupSum` class blocks, yielding 42.78% hard accuracy. A
   deterministic affine ordering raised the unchanged pair set to 83.08%.
7. **Initial 25% hybrid collapse.** It inherited the unshuffled butterfly issue
   and reached 60.53%. With only the ordering fix it reached 82.97%.
8. **Fan-out coefficient magnitude sweep.** `gamma=1.0` and `gamma=0.25` produced
   bit-identical pure-greedy topologies and checkpoints. Any positive penalty
   resolved the same ties here. A meaningful final ablation must compare
   `gamma=0` against a positive value.
9. **Fraction sweep.** Fractions 0.75 and 0.90 recovered accuracy, but 0.90 makes
   the regularity claim weak. The corrected v2 schedule permits retaining the
   intended fraction 0.25 for subsequent pilots.
10. **Rich checkpoint safe-load failure.** Early rich smoke checkpoints retained
    NumPy scalars and PyTorch's `TorchVersion` object in their metadata, so the
    modern default `torch.load(..., weights_only=True)` rejected them. Checkpoint
    construction now recursively converts metadata to primitive types and has a
    save/load regression test. Historical rich checkpoints require the explicitly
    unsafe loader for trusted local files; their parallel raw state dicts do not.
11. **Incomplete early manifests.** Environment/source fingerprinting was added
    after the pre-change and earliest smoke runs. Those directories are preserved
    as history, not presented as final archival evidence; all future training runs
    write an environment manifest before optimization begins.
12. **First frozen-test evaluator import failure.** Direct execution did not put
    `experiments/` on `sys.path`; both parallel processes stopped before loading
    a dataset or producing a test metric. The evaluator now adds its own parent
    directory explicitly.
13. **Second frozen-test evaluator import failure.** `experiments/utils.py`
    implicitly relied on `experiments/train.py` importing `torchlogix.models`.
    Standalone evaluation therefore stopped during model construction, again
    before dataset evaluation. The shared utility now imports its dependency
    explicitly. The third attempt evaluated all frozen checkpoints successfully.
14. **Fashion seed-1 jobs received external SIGTERM.** Two attempts stopped
    without a Python traceback or CUDA error, first near step 24,000 and then
    near step 72,000. Their partial directories are preserved under
    `results/failed/`. The queues were relaunched detached from the command
    session with per-run logs; every expected run then completed.
15. **GPU hidden from the held-out evaluator sandbox.** Two concurrent
    evaluation attempts failed at CUDA initialization before loading data,
    checkpoints, or writing test metrics. An independently approved CUDA
    allocation and matrix-multiplication check succeeded, and the exact
    evaluators then completed with direct GPU access. These failures did not
    inspect or influence test results.
16. **Slow first CIFAR-10 acquisition.** The one-time 170 MB download made the
    dependent queue appear idle. It completed normally; the archive MD5 is
    `c58f30108f718f92721af3b95e74349a`, and all extracted Python batches were
    verified before training continued.
17. **Unstable historical source fingerprints.** The original manifest hashed
    generated JSON below the experiment directory. Consequently, unchanged
    source could receive a different hash after another run wrote results.
    The manifest scope now excludes `results/` and `summary/`, records the exact
    file list, and has a regression test. For the already completed runs, the
    exact pre-analysis source is archived as
    `summary/training_source_pre_analysis.tar.gz` with SHA-256
    `84dd8a2ac53ee14fba59c582606eb25a29ecb801b37d89dc27f8f58f07053c47`.
18. **Bit ancestry selected the wrong image topology.** The original diagnostic
    counted thermometer bits independently. Frozen-checkpoint re-analysis
    showed that v2 often paired correlated thresholds from one raw source and
    actually reduced Fashion raw-pixel ancestry. Semantic diagnostics now
    retain `(channel, y, x, threshold)` coordinates and report both bit and
    raw-source ancestry, spatial distance, semantic-axis pairing, group
    coverage, predecessor overlap, and cross-gate adjacency similarity.
19. **Full-scale 64-candidate construction was impractical.** A CIFAR profile
    with 64 swap candidates per output exceeded 60 seconds and was manually
    stopped before training. No metric from that attempt was used. Candidate
    pool 8 reduced deterministic construction to about 9 seconds on CIFAR and
    5 seconds on Fashion; this value was frozen for every v3 pilot and central
    run.
20. **Broad source hashes split the v3 seed cohorts.** Reporting, queue, and
    configuration files were added after seeds 0--2, so the old broad-tree
    fingerprint differs for seeds 3--4 although the training implementation
    and resolved protocol do not. The audit records both cohorts explicitly.
    New manifests additionally hash a fixed training manifest
    (`src/torchlogix`, `experiments/train.py`, and `experiments/utils.py`) so
    future report-only edits do not change the implementation identity.
21. **Uniform 48K/12-layer allocation was infeasible.** A first draft used
    4,000 gates per layer, but rank-2 input coverage requires at least 4,608
    first-layer gates for 9,216 encoded CIFAR inputs. The regression test failed
    before training. Widths were redistributed to
    `[4608, 3946, 3946, 3945 × 8, 3940]`, preserving exactly 48K gates and the
    classifier scale.
22. **Float input to the convolutional export test failed.** Circuit export
    expects hardened Boolean activations; the first new test supplied floats
    and exposed the existing dtype contract. The test was corrected to use
    Boolean input. Training and export implementation were unchanged.
23. **Naive v3 classifier-tail reuse broke balanced fan-out.** Directly applying
    the equal-width dense schedule to paper-S's shrinking classifier produced
    fan-out CVs of 0.240 and 0.285. The classifier-specific schedule now uses
    exact round-robin matchings in shrinking layers; regression tests require
    exact predecessor degree balance.
24. **V5 top-level report label used the fallback strategy.** Per-layer
    topology rows correctly recorded `ancestry_channel_hybrid`, but the JSON
    metadata said `random` when a convolution-specific override was used.
    Effective convolution and classifier strategies are now resolved
    independently and covered by a regression test. This was reporting-only.
25. **Coverage--reuse M early gain did not confirm.** A seed-0 1K screen with
    evaluation every 500 steps showed +3.14 pp, but a preplanned 5K
    confirmation with evaluation every 1,000 steps ended at -3.12 pp. The
    method was not promoted to multi-seed or held-out-test evaluation.

## Implemented artifacts

- Fixed-strategy interface and all six requested dense schedules.
- Packed-bitset ancestry propagation and greedy scoring.
- Semantic-balanced v3 routing with tensor-aware, source-safe first-layer
  pairing and degree-preserving deeper-layer swaps.
- Semantic-channel v4 routing with paired independent RNGs and unchanged
  convolutional spatial coordinates.
- Ancestry-channel v5 routing with cross-block packed ancestry, balanced
  channel pairings, degree-preserving novelty swaps, and unchanged spatial
  coordinates.
- Generic coverage--reuse refinement with strictly improving,
  degree-preserving swaps, shared S/M parameters, and both dense and
  convolutional integration. Its failed M promotion result is retained.
- A classifier-specific v3 extension for shrinking convolutional classifier
  tails, without modifying or replacing the frozen dense v3 strategy.
- Per-depth coverage, overlap, fan-out, distinct-pair, reachability, storage,
  construction-time, and temporary-memory metrics.
- Image-semantic diagnostics for raw-source ancestry, spatial and semantic-axis
  pairing, group coverage, and adjacency diversity, including analysis of
  already frozen checkpoints.
- Topology-only CLI and JSON configuration support.
- Independent topology seeds that do not advance the PyTorch training RNG.
- Self-describing checkpoints and environment/source fingerprints.
- Fashion-MNIST registry support and training-only CIFAR-10 augmentation.
- Deterministic unit, bounds, expected-example, brute-force ancestry, RNG, tiny
  graph, semantic-layout, degree-preservation, integration, source-fingerprint,
  and checkpoint-index tests.
- Complete five-seed medium, two-budget/three-depth, and convolutional pilot
  cohorts with a dedicated 52-record follow-up audit.
- Held-out evaluation of every random, v2, and v3 central best-validation
  checkpoint.
- Hardened GPU inference benchmarks, learning curves, paired confidence
  intervals, circuit/storage accounting, and a machine-readable
  reproducibility audit.
- Reproducible table/CSV/SVG generation, plus a checksummed snapshot of the
  exact source used for training.

## Not completed; required before a DATE claim

- A protocol-identical numerical reproduction of the published CIFAR-10
  baseline. The architecture is exact, but the current frozen-validation
  protocol uses 45,000 rather than all 50,000 training images and 240 rather
  than 200 effective epochs.
- Butterfly and pure-coverage schedules on every budget/depth cell. The paired
  random/v3 matrix is complete, but the specification's full four-schedule
  matrix is not.
- Candidate-pool, swap-fraction, and novelty-weight ablations. Pool 8, fraction
  0.25, and weight 1.0 were frozen for the central study after the bounded
  three-seed pilot; they were not exhaustively optimized.
- Overlap-off, fan-out-off, and distance/locality ablations at pilot scale.
- WARP/Light repetition.
- Mommen partial-learnable, LILogic Top-K, BitLogic, and RigL comparisons under
  identical splits, budgets, and training effort.
- Long-training/five-seed convolutional reproduction. The implemented v4
  extension is currently a positive three-seed, 20K-step pilot and does not
  reproduce the paper's long LogicTreeNet result.
- Exported-circuit equivalence and compiled CPU latency/energy measurements;
  the completed inference benchmark is framework-level PyTorch GPU timing.
- A true accuracy/cost Pareto improvement. The present methods have identical
  gates, operations, and deployed routing storage.

The superseded fraction-0.25 v2 hybrid failed the formal kill criterion.
Semantic-balanced v3 clears the continuation criterion, remains positive at
four and eight layers across both budgets, and transfers positively to the
channel-only v4 pilot. The responsible next steps are component ablations, the
remaining schedules in the budget/depth matrix, long-training convolutional
reproduction, and protocol-identical named baselines. These results justify
continuing the project; they do not alone constitute a complete DATE claim.

## Final verification

Final verification after regenerating the summary artifacts:

- the reproducibility audit reports `pass` for expected-run completeness,
  54-point learning curves, paired protocol equality, held-out metrics,
  inference benchmarks, environment consistency, and the documented v3 source
  cohorts;
- the follow-up audit reports `pass` for 52 medium/depth/convolutional records,
  paired protocol equality, and bit-identical convolutional spatial hashes;
- the July 24 ancestry/classifier focused suite passes with **1,862 passed and
  1,660 skipped**;
- the complete TorchLogix suite passes with **3,302 passed, 3,038 skipped, and
  one pre-existing warning** in 161.99 seconds;
- all three generated SVGs parse as valid XML;
- the pre-analysis source archive passes its recorded SHA-256 check; and
- `nvidia-smi` reports both RTX PRO 6000 GPUs at 0% utilization with no running
  compute processes after the queues completed.
