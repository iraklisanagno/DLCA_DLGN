# CoverageDLGN implementation and experiment history

**Updated:** August 3, 2026
**Status:** Dense semantic-balanced v3 and convolutional semantic-channel v4
remain frozen. The separate unified semantic-degree-balanced U1 candidate
completed its five-seed convolutional S gate but was not promoted. The
original five-seed 48K study, a five-seed 512K
escalation, a two-budget/three-depth study, and a three-seed convolutional
pilot have frozen validation/test artifacts. V3 improves 512K dense CIFAR-10
held-out accuracy by **+4.256 pp**; v4 improves the convolutional pilot by
**+2.000 pp**. Component ablations and a one-seed 200K nine-channel
LogicTreeNet-M comparison are complete; multi-seed long convolutional
confirmation and protocol-identical named-method comparisons remain before a
DATE claim.

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
26. **Dense CIFAR-100 compact scales did not promote.** Exact two-by-4K S has
    fewer first-layer slots than encoded inputs. Its best frozen V3 control
    trailed random by -0.840 pp across three paired 20K seeds (95% CI
    [-1.404, -0.276]). A documented M diagnostic removed the undersubscribed
    first layer and screened at +0.500 pp, but its three-seed 20K confirmation
    was -0.353 pp (95% CI [-1.058, +0.352]), with every seed negative. No
    full S/M, L, or CIFAR-100 held-out evaluation was run.
27. **Deep CIFAR-100 separates a small positive regime from a stopped
    large-width regime.** On the exact 6-by-64K architecture, frozen V3
    `swap_fraction=0.125` passed the 5K and three-seed 20K gates. At the
    paper-length schedule it improved validation by +0.633 pp and held-out
    test by +0.333 pp (21.010% versus 20.677%, n=3), but the test 95% CI
    [-0.816, +1.483] crosses zero. On exact 6-by-256K, all three existing V3
    controls trailed random at 5K; the best was -0.620 pp, so the frozen
    protocol stopped the branch before full effort or test access. This is
    evidence that unchanged V3 is not uniformly beneficial as width,
    threshold count, and temperature change.
28. **The deep screen recovered from one external termination without
    duplicating completed work.** Three complete runs were retained, two
    incomplete attempts were quarantined under `results/failed/`, and the
    resumed queue executed only the missing entries. A separate selection
    generator field-name error occurred before training and produced no
    scientific result. Both incidents and their resolutions are recorded in
    `EXPERIMENT_LOG.md`.
29. **Depth alone does not increase V3's CIFAR-100 benefit at fixed 384K
    gates.** A controlled seed-0 20K ablation produced +0.780 pp for
    3-by-128K, but this missed the frozen +1 pp confirmation threshold.
    Both 12-by-32K and 24-by-16K remained at approximately 1.2% chance-level
    accuracy for random and V3. At depth 24, every final gate already has all
    3,072 raw RGB spatial sources in its ancestry under both methods.
    Therefore additional depth simultaneously saturates V3's ancestry
    objective and creates an optimization failure under the unchanged
    training recipe; it is not evidence that V3 improves monotonically with
    depth.
30. **Class-conditional ancestry balance is not the missing CIFAR-100
    mechanism.** A separate fixed final-layer refinement reduced V3's mean
    per-class source-usage CV from 0.25655 to 0.23475 across three seeds while
    preserving its backbone, exact predecessor degrees, gate/parameter
    budgets, routing bits, and inference path. Nevertheless, 20K hardened
    validation was 21.593% versus 21.580% for V3: only +0.013 pp
    (95% CI [-0.385, +0.412]). Its +0.553 pp gain over random also missed the
    frozen +2 pp requirement. No full, transfer, convolutional, or held-out
    experiment was run.

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
- A separate class-conditional coverage head that preserves V3/V4 and exact
  deployment cost. Its successful diagnostic change and failed CIFAR-100
  accuracy promotion are retained as a negative result.
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

## July 29 dense follow-up

The remaining frozen CIFAR-10 compression checkpoints were evaluated once on
held-out test. At 256K gates, CoverageDLGN reached 56.903% +/- 0.134% versus
52.253% +/- 0.058% for random, a paired +4.650 pp with 95% CI
[+4.174, +5.126]. At 384K, CoverageDLGN reached 58.143% +/- 0.153% versus
53.657% +/- 0.328%, a paired +4.487 pp with 95% CI
[+3.515, +5.458]. Every paired gain was positive. These 12 checkpoints are
closed to further test queries.

The CIFAR-10 M V3 component ablation reused the old random/full-V3 controls
and trained only six missing arms. Balanced butterfly fan-out contributes
+4.160 pp over random (95% CI [+3.988, +4.332]). Adding the semantic first
layer contributes +0.273 pp (95% CI [-0.780, +1.326]), and the ancestry-swap
stage contributes +0.040 pp (95% CI [-0.434, +0.514]). Full V3 remains
+4.473 pp over random (95% CI [+3.624, +5.323]). Thus balanced fan-out is the
dominant measured component at this coordinate; the smaller semantic and
ancestry increments are inconclusive.

A separate one-shot task-aware extension was implemented without changing
V3. It preserved the exact predecessor-degree vector and deployment cost,
used one ordinary step-10K batch, added no optimizer steps, and changed about
11.3% of gates. Across three seeds it reached 59.093% +/- 0.234%, which is
+4.273 pp over random but -0.200 pp versus V3. It failed the predeclared +1 pp
over-V3 gate, so it has no full, held-out, transfer, or convolutional result.

The CIFAR-100 baseline audit found a material reproduction mismatch:
canonical difflogic random routing uses two Torch permutations, whereas the
paired local study uses an independent NumPy topology seed. Architecture and
training schedule match, but the local 20.677% result is therefore labeled
`[REPRODUCED, topology-adapted]` beside the paper's reported 22.54%.

Machine-readable sources:

- `summary/table2_cifar10_compression_remaining_test.json`;
- `summary/cifar10_medium_v3_components.json`;
- `summary/cifar10_medium_task_aware.json`; and
- `summary/cifar100_baseline_audit.json`.

## July 29 convolutional revision

The specification-compliant channel-spatial adapter preserved frozen V4's
channel pairs and spatial-coordinate tensor, changed only bottom-level channel
assignment, and added no gates, routing entries, or trainable parameters.
Across three CIFAR-10 S validation seeds it reached 57.033%, versus 56.673%
random (+0.360 pp, 95% CI [-0.501, +1.221]) and 57.187% V4
(-0.153 pp, 95% CI [-2.714, +2.407]). It failed both promotion gates, so no M
or held-out run followed.

The paired component arm with balanced channel routing and swaps disabled
reached 58.013%: +1.340 pp over random and +0.827 pp over V4. Both intervals
cross zero. This is the best current convolutional diagnostic, not a promoted
replacement for V4.

The initial explicit-classifier attempt is excluded because its independent
dense topology RNG changed later parameter initialization. Its artifacts and
explanation are retained under
`results/failed/cifar10_conv_small_explicit_classifier_rng_attempt1`.

Machine-readable sources:

- `summary/cifar10_conv_small_v4_components.json`;
- `summary/cifar10_conv_small_channel_spatial.json`.

## July 30 unified degree-balanced U1 result

The exact historical V4/no-swap checkpoints were reanalyzed per layer. V4
preserved predecessor fan-out and spatial indices while changing 8.33%,
20.31%, 10.68%, and 2.08% of output pairs across the four convolutional
layers on average. It reduced duplicate pairs and increased span, but the
largest mean predecessor-Jaccard change was only 0.00246 and mean raw ancestry
was effectively unchanged. The no-swap learning curve overtook V4 after 12K.
The useful base is an affine-ordered balanced butterfly, not round-robin.

A separate `semantic_degree_balanced` strategy (U1) was implemented for dense
and convolutional networks. It uses semantic input order, deterministic
balanced butterfly pairs, the exact base fan-out schedule, no forced
convolutional leaf pairing, and no ancestry swaps. V3 and V4 paths were not
edited. Full-model regression proves U1 is bitwise identical to the historical
convolutional no-swap arm, allowing seeds 0-2 to be reused.

Only missing seeds 3 and 4 were trained for random, frozen V4, and U1. All six
runs completed with zero failures:

| Method | Five-seed best hard validation | Gain vs random | Paired 95% CI | Positive pairs |
|---|---:|---:|---:|---:|
| Random | 56.864% | -- | -- | -- |
| Frozen V4 | 57.448% | +0.584 pp | [-0.335, +1.503] | 4/5 |
| U1 | **57.624%** | **+0.760 pp** | **[-0.700, +2.220]** | **4/5** |

U1 passed the four-of-five consistency condition but failed the predeclared
+1.0 pp mean-gain condition. It was not promoted. Consequently,
convolutional M, CIFAR-100 transfer, full schedules, held-out test accuracy,
and extended cost/runtime/memory evaluation were not run.

For new seeds 3/4, U1 topology construction averaged 0.165 seconds versus
0.414 seconds for V4. Mean wall time and peak GPU allocation were effectively
identical, and all recorded gate, parameter, and routing-bit costs match.

Machine-readable sources:

- `summary/cifar10_conv_small_no_swap_diagnostics.json`;
- `summary/cifar10_conv_small_unified_five_seed.json`.

## July 31 WARP-style CIFAR-10 Medium result

A separate experiment reconstructed the WARP Figure 4 Medium setting before
combining it with frozen Legacy V4. This uses legacy
`ClgnCifar10Medium`, two thresholds per RGB channel (six Boolean channels),
raw rank-2 gate parameterization, no augmentation, and 30K updates. It is not
the paper-faithful nine-channel `ClgnCifar10PaperMedium` architecture and must
not be presented as a reproduction of the original LogicTreeNet-M 71.01%
test result.

The hardened validation results completed so far are:

| Method | Seed-0 best hard validation | Approx. WARP Figure 4 endpoint |
|---|---:|---:|
| WARP fixed uniform | 65.35% | approximately 64.0% |
| WARP fixed distributive | 66.12% | approximately 65.0% |
| WARP learnable | 65.88% | approximately 66.6% |
| Matched random, fixed uniform | 64.58% | -- |
| Frozen Legacy V4, fixed uniform | **66.23%** | -- |

Legacy V4 is +1.65 pp over its matched random control on seed 0. This is a
single paired result, not a mean or confidence interval. The public WARP arms
use `random-unique` routing, while the Legacy V4 attribution pair uses the
same random sampler as the original V4 pilot; therefore the primary topology
effect is 66.23% versus 64.58%, not a direct comparison against a WARP arm.
Frozen V3 and V4 implementations were not changed.

The fixed-uniform and fixed-distributive WARP jobs began with the plotted 50K
budget. Both were interrupted just after their common 30K validation boundary
once they had reached the approximate plotted endpoints. No validation after
30K was evaluated. The 30K budget was frozen before any Medium Legacy V4
accuracy was observed, and the remaining seed-0 arms used the same budget.
Accordingly, the result supports “reached the reconstructed WARP accuracy
within 30K updates,” not an exact 50K replication. All values above are
validation accuracy; the held-out CIFAR-10 test set has not been queried.
Seeds 1 and 2 remain pending.

Machine-readable sources and protocol history:

- `summary/warp_fig4_cifar10_medium.json`;
- `summary/warp_fig4_cifar10_medium.csv`;
- `protocols/warp_fig4_cifar10_medium.json`;
- `WARP_FIG4_REPRODUCTION.md`.

## August 3 nine-channel LogicTreeNet-M matched 200K result

Frozen Legacy V4 and original fixed-random routing were compared on the exact
nine-channel `ClgnCifar10PaperMedium` architecture for 200K updates. Both use
seed 0, split seed 2027, topology seed 0, the same 45K/5K split, batch 128,
AdamW at 0.02 with 0.002 weight decay, standard training-only crop/flip, raw
rank-2 gates, residual probability 0.951, and validation every 2K updates.
Only the convolutional fixed-topology rule differs; both dense classifier
tails use fixed random routing.

The original V4 job was configured for 350K updates, but its trajectory had
plateaued. It was stopped by a guarded `SIGINT` only after the 200K checkpoint,
metrics, and thresholds were present. The random control then ran normally to
200K. Both produced 100 matched validation evaluations:

| Statistic | Legacy V4 | Fixed random | V4 gain |
|---|---:|---:|---:|
| Best hardened validation | **71.26% at 194K** | 70.68% at 164K | **+0.58 pp** |
| Final 200K hardened validation | 70.28% | 70.10% | +0.18 pp |
| Best relaxed validation, any step | 72.96% | 72.56% | +0.40 pp |
| Hardened held-out test at best-hard checkpoint | **69.96%** | 69.57% | **+0.39 pp** |
| Relaxed held-out test at the same checkpoint | **72.18%** | 71.28% | **+0.90 pp** |

V4 led on 97 of 100 hardened validation evaluations. Its mean and median
advantages across the full trajectory were +1.055 and +0.960 pp, respectively,
and it first reached 70% hardened validation at 36K rather than random's 66K.
This supports a one-seed optimization/learning-curve signal, but the +0.39 pp
held-out gain is not a statistical paper claim.

The two best-hard-validation checkpoints were SHA-256 frozen before the test
set was accessed, then each was evaluated exactly once on all 10,000 held-out
examples. The reported LogicTreeNet-M accuracy is 71.01% test: local V4 is
1.05 pp below it and local random is 1.44 pp below it. The local protocol uses
a 45K/5K selection split and explicitly adapted crop/flip augmentation, so it
is a controlled topology comparison rather than an exact numerical
reproduction of the paper's training protocol.

The topology is constructed offline. V4 required 6.182 seconds versus 2.478
seconds for random, an added 3.704 seconds over approximately 25.5 hours of
training. A matched hardened PyTorch/CUDA timing pass measured 74.573 versus
74.891 ms per batch of 128; the sub-percent difference is noise, not a speed
claim. Peak inference allocations were 2,949,915,136 and 2,949,698,048 bytes.
The architecture and deployed costs match: approximately 3.08M reported gate
operations, 10,694,656 trainable gate parameters, zero trainable routing
parameters, and 2,375,680 packed deployed-routing bytes. The random run
recorded 15,691,860,480 bytes peak training allocation; V4's exact peak was not
emitted because its normal finalizer was bypassed by the controlled stop.

Machine-readable sources:

- `summary/cifar10_paper_medium_200k_freeze.json`;
- `summary/cifar10_paper_medium_200k_paired.json`;
- `summary/cifar10_paper_medium_200k_curve.csv`;
- `protocols/cifar10_paper_medium_200k_paired.json`.

### August 14 unified U2 extension at LogicTreeNet-M scale

Unified U2 was subsequently trained for the same 200K updates on the exact
nine-channel `ClgnCifar10PaperMedium` architecture, seed, split, optimizer,
augmentation, and gate parameterization. The generic, convolutional, and
classifier fixed-routing initializers are all `semantic_multiscale_balanced`;
no V3 or Legacy V4 implementation was modified.

| Statistic | Unified U2 | Legacy V4 | Fixed random | U2 vs V4 | U2 vs random |
|---|---:|---:|---:|---:|---:|
| Best hardened validation | **72.38% at 136K** | 71.26% at 194K | 70.68% at 164K | **+1.12 pp** | **+1.70 pp** |
| Final 200K hardened validation | **71.64%** | 70.28% | 70.10% | **+1.36 pp** | **+1.54 pp** |
| Hardened held-out test | **71.65%** | 69.96% | 69.57% | **+1.69 pp** | **+2.08 pp** |
| Relaxed held-out test | **72.95%** | 72.18% | 71.28% | **+0.77 pp** | **+1.67 pp** |

U2 led random in 97 of 100 hardened validation evaluations and Legacy V4 in
87 of 100. Its mean curve gains were +1.7876 and +0.7328 pp, respectively.
The best-validation checkpoint was SHA-256 frozen before any test access and
then evaluated exactly once on all 10,000 held-out examples. Its 71.65% hard
test accuracy is numerically +0.64 pp above the reported LogicTreeNet-M 71.01%,
but the paper value remains an external reference rather than a matched local
control.

U2 preserves the exact gate, parameter, and deployed-routing budgets: 668,416
learned LUT units including convolutional kernels, 10,694,656 trainable gate
parameters, zero trainable routing parameters, and 2,375,680 packed routing
bytes. Offline topology construction took 18.908 seconds. Training took 35.686
hours and recorded 14.614 GiB peak PyTorch CUDA allocation. The longer wall
time than the earlier random/V4 jobs was measured under different concurrent
machine load and is not attributed to fixed routing; the offline construction
time is the controlled method-specific overhead.

Machine-readable sources:

- `summary/cifar10_paper_medium_u2_200k_freeze.json`;
- `results/full_conv_cifar10_paper_medium_u2_seed0_200k/test_metrics.json`;
- `logs/cifar10_paper_medium_u2_200k/test/test_evaluation_summary.json`;
- `LOGICTREENET_M_U2_PROTOCOL.md`.

## August 4 convolutional evidence and deployment snapshot

The completed convolutional artifacts were frozen by SHA-256 and consolidated
without loading a dataset, launching accuracy training, or making another
test-set query. The snapshot covers 22 valid run directories and keeps two
protocol families separate:

- paper-faithful LogicTreeNet-S/M use three thermometer thresholds per RGB
  channel, nine Boolean channels, four depth-3 convolutional stages, raw LUTs,
  and group size 2; S uses `k_num=32`, tau 20, while M uses `k_num=256`, tau
  40;
- WARP-style Medium uses the legacy `ClgnCifar10Medium` class with two
  thresholds per RGB channel, six Boolean channels, and no augmentation.

The paper-faithful S cohort has two distinct evidence scopes. The five-seed
validation-only promotion cohort is random 56.864%, V4 57.448%, and U1
57.624%. U1 is +0.760 pp over random, positive on four of five seeds, but its
95% interval [-0.700, +2.220] crosses zero and it failed the predeclared +1 pp
mean gate. Historical random/V4 seeds 0--2 had already been evaluated once on
test before the U1 decision: random averaged 56.140% and V4 56.367%, a mixed
+0.227 pp paired effect with 95% interval [-5.221, +5.675]. U1 and S seeds 3/4
remain unqueried on test. These cohorts are now labelled separately.

Mean S learning curves show that V4 and U1 reached 55% hardened validation at
8K updates versus 10K for random, and reached 57% at 18K while the random mean
never reached 57% within 20K. These are descriptive validation-curve metrics,
not additional selection criteria.

Synthetic Boolean circuit export was then measured on frozen seed-0
checkpoints. No CIFAR data was loaded. All checked models passed hardened
class, PyTorch export, Python `Circuit`, simplified-Circuit, and, where
compiled, generated-C equivalence.

| Protocol | Method | Declared learned LUTs | Spatial gate applications | Simplified IR nodes | Peak process RSS | Compiled CPU batch-128 |
|---|---|---:|---:|---:|---:|---:|
| Paper S, 9ch | Random | 83,552 | 874,496 | 197,851 | 1.071 GiB | 3.057 ms |
|  | V4 | 83,552 | 874,496 | 202,827 | 1.073 GiB | 3.076 ms |
|  | U1 | 83,552 | 874,496 | 214,883 | 1.074 GiB | 3.095 ms |
| Paper M, 9ch | Random | 668,416 | 6,995,968 | 1,676,852 | 6.103 GiB | not compiled |
|  | V4 | 668,416 | 6,995,968 | 1,702,350 | 6.092 GiB | not compiled |
| WARP M, 6ch | Matched random | 668,416 | 6,995,968 | 1,101,364 | 4.070 GiB | not compiled |
|  | Legacy V4 | 668,416 | 6,995,968 | 1,129,547 | 4.077 GiB | not compiled |

Declared architecture budgets, trainable gate parameters, zero trainable
routing parameters, and routing-entry counts are identical within each
matched group. Simplified IR nodes are not required to be identical because
the learned truth tables determine which constants, wires, duplicates, and
dead nodes the circuit compiler can remove. Relative to random, simplified IR
size is +2.52% for S V4, +8.61% for S U1, +1.52% for paper-M V4, and +2.56%
for WARP-style M V4. The S CPU latency changes are +0.62% and +1.26% for V4
and U1 and do not support a speed claim.

An initial fully unrolled random-S `gcc -O1` compilation was interrupted after
approximately 8.75 minutes. The bounded `gcc -O0`, 64-way bit-packed compiles
completed in 37.75, 39.75, and 44.72 seconds for random, V4, and U1. M circuit
construction and equivalence were feasible, but M compilation was not
attempted after the S scaling result. Energy remains unmeasured.

Machine-readable sources:

- `summary/convolutional_evidence_freeze.json`;
- `summary/convolutional_evidence_snapshot.json` and curve CSV;
- `summary/convolutional_deployment.json` and CSV;
- `summary/deployment/*.json` and `compile_attempt_history.json`.

## August 9, 2026 DATE second-round status

Architecture-matched six-by-8K learned-connectivity controls use 48K gates,
the fixed data split (`data_split_seed=2027`), and 200 effective epochs. The
table reports best hardened validation, not held-out test accuracy.

| Dataset | Method | Best hard validation | Runs | Mean time | Peak GPU | Status |
|---|---|---:|---:|---:|---:|---|
| MNIST | Fixed random | 97.157 +/- 0.043% | 5 | 15.03 min | 0.099 GiB | Reused exact control |
|  | Mommen | 95.683 +/- 0.404% | 3 | 51.90 min | 0.809 GiB | Complete exact-depth adaptation |
|  | LILogicNet | 95.717 +/- 0.351% | 3 | 92.44 min | 3.706 GiB | Complete exact-depth adaptation |
|  | BitLogic rank-4 | 11.417 +/- 0.000% | 3 | 112.75 min | 3.527 GiB | Reproduced-negative depth transfer |
|  | CoverageDLGN V3 | **97.403 +/- 0.114%** | 5 | **15.02 min** | **0.099 GiB** | Reused frozen method |
| Fashion-MNIST | Fixed random | 87.477 +/- 0.183% | 5 | 14.97 min | 0.099 GiB | Reused exact control |
|  | Mommen | 87.400 +/- 0.928% | 3 | 29.09 min | 0.453 GiB | Complete exact-depth adaptation |
|  | LILogicNet | 84.267 +/- 1.636% | 3 | 92.05 min | 3.706 GiB | Complete exact-depth adaptation |
|  | BitLogic rank-4 | 10.867 +/- 0.000% | 3 | 112.74 min | 3.527 GiB | Reproduced-negative depth transfer |
|  | CoverageDLGN V3 | **87.873 +/- 0.271%** | 5 | **15.02 min** | **0.099 GiB** | Reused frozen method |

V3's three common MNIST seeds gain +1.706 pp over Mommen (95% CI
[+0.666, +2.745]) and +1.672 pp over LILogicNet ([+0.952, +2.392]). On Fashion,
the common-seed paired effects are +0.483 pp ([-1.597, +2.563]) and +3.617 pp
([-0.016, +7.249]); those intervals do not support a conclusive superiority
claim. V3 uses no training-only routing parameters. BitLogic's two-layer
reproduction remains valid; the chance-level six-layer result is specifically
a negative depth-transfer result.

At the identical 48K deployed gate budget, V3 uses 3x/2x fewer training
parameters than matched Mommen on MNIST/Fashion, trains 3.46x/1.94x faster,
and peaks at 8.17x/4.58x less GPU memory. Against matched LILogicNet it uses 5x
fewer training parameters, trains 6.15x/6.13x faster, and peaks at 37.43x less
GPU memory. These ratios describe training resources, not inference speed.

The initial U2 convolutional smoke exposed a pre-pilot structural defect:
truncated non-power-of-two cyclic stages left 12,288 and 5,112 classifier
inputs unused. U2 alone was corrected to use deterministic matching stages,
degree-first selection, normalized-ancestry scale selection, and rotating odd-
width byes. Fresh dense and convolutional CUDA smokes completed. The corrected
40,960 -> 20,480 and 20,480 -> 10,240 classifier reductions both have
`fanout_cv=0.0` and zero unused inputs; all convolutional channel topologies
also have zero unused inputs. Frozen V3, V4, and U1 regression tests pass.

The complete three-seed compression ladder finished on CUDA without rerunning
the reused 48K cohorts. Values below are best hardened validation; held-out
test remains locked.

| Dataset | Total gates | Random | CoverageDLGN V3 | Paired gain | 95% CI | Wins |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 4K | 85.539 +/- 0.495% | 86.067 +/- 0.159% | +0.528 pp | [-1.087, +2.142] | 2/3 |
| MNIST | 8K | 91.461 +/- 0.286% | 91.956 +/- 0.113% | +0.494 pp | [-0.479, +1.467] | 3/3 |
| MNIST | 16K | 95.100 +/- 0.161% | 95.478 +/- 0.444% | +0.378 pp | [-0.925, +1.681] | 3/3 |
| MNIST | 32K | 96.694 +/- 0.135% | 97.011 +/- 0.129% | +0.317 pp | [-0.334, +0.967] | 3/3 |
| Fashion-MNIST | 8K | 83.433 +/- 0.148% | 83.644 +/- 0.158% | +0.211 pp | [+0.085, +0.338] | 3/3 |
| Fashion-MNIST | 16K | 86.194 +/- 0.250% | 86.883 +/- 0.192% | +0.689 pp | [-0.344, +1.722] | 3/3 |
| Fashion-MNIST | 32K | 87.461 +/- 0.234% | 87.778 +/- 0.327% | +0.317 pp | [-1.071, +1.704] | 2/3 |
| Fashion-MNIST | 64K | 87.333 +/- 0.557% | 88.100 +/- 0.200% | +0.767 pp | [-0.524, +2.057] | 3/3 |

V3 has a positive mean effect in all eight new cells and wins 22/24 paired
runs. Fashion-MNIST 8K is the only individually significant n=3 cell; the
other intervals remain inconclusive. The reused 48K validation gains are also
positive (+0.247 pp MNIST and +0.396 pp Fashion, n=5). Mean training time and
peak GPU allocation are matched within measurement noise at each budget;
there are no training-only routing parameters. Machine-readable status and
provenance are in `summary/second_round_status.{json,csv}`.

### Strengthened dense CIFAR-100 result

A baseline-only, seed-0 training-recipe screen was run for 5K updates on the
same 3 x 128K architecture. It is screening evidence and is not mixed with the
matched topology cohort:

| Temperature | Learning rate | Augmentation | Best hard validation |
|---:|---:|---|---:|
| 1 | 0.01 | None | 11.020% |
| 5 | 0.01 | None | 17.180% |
| **20** | **0.01** | **None** | **22.700%** |
| 10 | 0.02 | None | 19.940% |
| 10 | 0.01 | Standard | 18.280% |
| 10 | 0.02 | Standard | 18.180% |

The screen identifies temperature 20 as the strongest short-run baseline
recipe. To preserve an exact comparison with the already-completed seed 0,
the missing seeds of the original temperature-10, learning-rate-0.01,
no-augmentation 20K random/V3 pair were then completed without changing that
recipe:

| Method | Best hard validation (n=3) | Paired gain | 95% CI | Wins | Train time | Peak GPU | Topology time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed random | 21.093 +/- 0.101% | -- | -- | -- | 5.35 min | 0.890 GiB | 3.36 s |
| Frozen V3 | **21.933 +/- 0.110%** | **+0.840 pp** | **[+0.351, +1.329]** | **3/3** | 5.34 min | 0.890 GiB | 71.03 s |

This is a statistically positive validation result at identical 384K gate,
6.144M LUT-parameter, optimizer-update, memory, and deployed-routing budgets.
V3 changes only the offline fixed topology and adds no training-only routing
parameters. After validation was frozen, the checkpoints were evaluated once
on held-out test: random reached 20.923 +/- 0.352% and V3 reached
**21.467 +/- 0.410%**. The paired test gain is +0.543 pp with 3/3 wins; its
n=3 95% CI [-0.141, +1.227] remains inconclusive.

Two seed-0, same-384K allocation ablations moved capacity toward the
100-class output layer. The 96K/96K/192K split reached 19.960% random and
20.620% V3 (+0.660 pp); the more extreme 64K/64K/256K split reached 20.380%
random and 20.240% V3 (-0.140 pp). Both are below the balanced 3 x 128K
architecture. Thus the positive result is not explained by adding gates or by
silently enlarging the class head.

### Frozen unified U2 pilot

The corrected `semantic_multiscale_balanced` U2 rule was frozen before its
three-seed pilots. It applies the same deterministic degree-balanced,
semantic-order, multiscale matching rule to dense layers and convolutional
channel routing. It has no learned routing, swaps, or deployment overhead.
All 15 CUDA runs completed without failures; held-out test remained locked.

| Coordinate | Random | U2 | Gain | 95% CI | Wins | Comparison with frozen predecessor |
|---|---:|---:|---:|---:|---:|---|
| MNIST, 8K, 20K | 90.956% | **91.711%** | **+0.756 pp** | **[+0.029, +1.483]** | 3/3 | +0.044 pp vs V3; tied |
| Fashion-MNIST, 16K, 20K | 85.739% | **86.339%** | +0.600 pp | [-0.261, +1.461] | 3/3 | -0.089 pp vs V3; tied |
| Dense CIFAR-10 S, 20K | 49.040% | **52.333%** | **+3.293 pp** | **[+1.803, +4.784]** | 3/3 | +0.140 pp vs V3; tied |
| Dense CIFAR-100, 3 x 128K, 20K | 21.093% | 21.193% | +0.100 pp | [-0.473, +0.673] | 2/3 | **-0.740 pp vs V3, 0/3; reject** |
| Conv. CIFAR-10 S, 20K | 56.673% | **58.847%** | **+2.173 pp** | **[+1.647, +2.700]** | 3/3 | +1.660 pp vs V4 and +0.833 pp vs U1; 3/3 each |

The convolutional row uses the original primary random cohort. Against the
separate explicit controlled-random cohort, U2 gains +1.707 pp with 3/3 wins;
the n=3 interval is wide and crosses zero. U2 also wins all three pairs versus
V4 (+1.660 pp, CI [-0.137, +3.457]) and U1 (+0.833 pp, CI
[-1.516, +3.183]). These comparisons and both random provenances are retained
rather than selecting whichever random cohort gives the larger gain.

U2's mean offline construction time is 0.101 seconds on MNIST-8K, 0.225
seconds on Fashion-16K, 0.917 seconds on dense CIFAR-10 S, 13.346 seconds on
CIFAR-100 3 x 128K, and 1.380 seconds on convolutional S. Gate count, LUT
parameters, training-only routing parameters (zero), deployed routing bits,
training time, and peak GPU allocation match the corresponding fixed-routing
budget. The positive MNIST, Fashion, dense CIFAR-10, and convolutional
coordinates promoted to the unchanged full-effort rule. CIFAR-100 U2 did not;
the statistically positive frozen V3 result remains the retained method there.

Machine-readable source: `summary/second_round_u2_pilot.json`.

### Promoted dense full-effort and one-time held-out test

The promoted dense U2 coordinates were trained for the unchanged 108K
schedule with three seeds. Validation selections were frozen before any new
test query; 27 previously unevaluated checkpoints were then evaluated once.
The six pre-existing CIFAR-10 random/V3 test records were reused, not rerun.

| Coordinate | Method | Best hard validation | Hard test | Test gain vs random | 95% CI | Wins |
|---|---|---:|---:|---:|---:|---:|
| MNIST, 8K | Random | 91.461 +/- 0.286% | 91.273 +/- 0.217% | -- | -- | -- |
|  | Frozen V3 | 91.956 +/- 0.113% | 91.907 +/- 0.307% | +0.633 pp | [-0.290, +1.557] | 3/3 |
|  | Unified U2 | 91.839 +/- 0.304% | **91.937 +/- 0.137%** | **+0.663 pp** | **[+0.015, +1.311]** | **3/3** |
| Fashion-MNIST, 16K | Random | 86.194 +/- 0.250% | 85.197 +/- 0.261% | -- | -- | -- |
|  | Frozen V3 | **86.883 +/- 0.192%** | **85.913 +/- 0.356%** | +0.717 pp | [-0.284, +1.717] | 3/3 |
|  | Unified U2 | 86.606 +/- 0.256% | 85.717 +/- 0.453% | +0.520 pp | [-1.016, +2.056] | 3/3 |
| Dense CIFAR-10 S | Random | 49.513 +/- 0.580% | 48.913 +/- 0.346% | -- | -- | -- |
|  | Frozen V3 | **53.113 +/- 0.284%** | **52.390 +/- 0.295%** | **+3.477 pp** | **[+3.265, +3.688]** | **3/3** |
|  | Unified U2 | 52.833 +/- 0.300% | 52.097 +/- 0.630% | +3.183 pp | [+0.772, +5.595] | 3/3 |

U2 is therefore a generic positive fixed-topology rule versus random in all
three promoted dense coordinates, but it does not supersede frozen V3 on
Fashion-MNIST or dense CIFAR-10. On MNIST test it ties V3 (+0.030 pp, CI
[-0.394, +0.454]). It reduces mean offline construction relative to V3 from
0.866 to 0.100 seconds on MNIST, 3.358 to 0.231 seconds on Fashion, and 9.313
to 0.916 seconds on dense CIFAR-10 S while preserving the exact circuit and
training budgets. Machine-readable sources are
`summary/second_round_final_validation_freeze.json` and
`summary/second_round_final_dense.json`.

The generic source-tree hash differs for a few sequential U2 runs because
documentation and orchestration files were updated while the queue was
active. The separately recorded training-implementation SHA-256 is identical
for every frozen dense checkpoint; no model, topology, optimizer, or data
code changed between those runs.

Representative seed-0 hardened GPU inference used deterministic synthetic
inputs, batch 128, 20 warmups, and 100 timed batches on GPU 1. It never loaded
a dataset:

| Coordinate | Random | V3 | U2 | Peak device memory | Interpretation |
|---|---:|---:|---:|---:|---|
| MNIST, 8K | 1.138 ms | 1.140 ms | 1.137 ms | 0.0049 GiB | Matched |
| Fashion, 16K | 1.134 ms | 1.136 ms | 1.137 ms | 0.0094 GiB | Matched |
| Dense CIFAR-10 S | 0.770 ms | 0.765 ms | 0.767 ms | 0.0412 GiB | Matched |
| Dense CIFAR-100, 3 x 128K | 3.589 ms | 3.527 ms | -- | 0.4082 / 0.4076 GiB | Matched; U2 rejected |

Sub-percent differences, and the single 1.7% CIFAR-100 difference, are
single-pass measurement variation rather than speed claims. The defensible
trade-off is unchanged hardened runtime and memory at identical circuit cost.

### Full-schedule LogicTreeNet-S validation and held-out test

The exact nine-input-channel LogicTreeNet-S fixed-random, frozen legacy-V4,
unified-U1, and frozen unified-U2 seed-0 runs completed all 350K updates. All
four best hardened-validation selections were recorded in
`summary/second_round_convolutional_validation_freeze.json`; the manifest
confirms that none had a test record at freeze time. The four checkpoints
were then evaluated once on the held-out 10,000-example test set.

| Method | Best / final hard V | Hard / relaxed T | Gain vs random T | Training time | Peak GPU | Topology time |
|---|---:|---:|---:|---:|---:|---:|
| Fixed random | 58.680 / 58.500% | 57.370 / 60.410% | -- | 4.975 h | 1.831 GiB | 0.217 s |
| Frozen legacy V4 | 59.860 / 58.460% | 58.930 / 62.450% | +1.560 pp | 4.957 h | 1.831 GiB | 0.449 s |
| Unified U1 | 59.880 / 58.580% | 58.800 / 61.300% | +1.430 pp | 4.980 h | 1.831 GiB | 0.206 s |
| **Unified U2** | **61.000 / 60.280%** | **60.630 / 63.580%** | **+3.260 pp** | **4.951 h** | **1.831 GiB** | **1.372 s** |

U2 also gains +1.700 test pp over legacy V4 and +1.830 pp over U1. Its
2.320 validation-point gain and +3.260 test-point gain therefore survive the
full matched schedule. Its 60.630% hard test result is 0.250 pp above the
paper-reported 60.38% LogicTreeNet-S result, but the local full-schedule
cohort has only one seed; the method-level multi-seed evidence remains the
three-seed 20K pilot (+2.173 pp versus primary random, 3/3 wins).

Learning-curve aggregation over all 350K updates gives U2 a 59.562% mean hard
validation accuracy versus 57.233% random, 58.510% V4, and 58.344% U1. U2
first reaches 58.0% at 14K updates, 59.0% at 34K, and 59.5% at 34K. Random
first reaches 58.0% at 82K and never reaches 59.0%; V4 reaches 59.5% at 212K
and U1 at 252K. These are matched learning-efficiency measurements, not a
shortened-training claim.

The 83,552 count is the total number of learned LUT functions. The
`dense_gate_count=71,680` field in `run_summary.json` counts classifier gates
only; 874,496 is the distinct spatial gate-application count. These three
definitions must not be interchanged.

All four methods have exactly 1,336,832 trainable LUT parameters, zero
trainable routing parameters, 1,945,600 deployed routing bits, and the same
83,552 learned LUT functions / 874,496 spatial gate applications. Corrected
synthetic CUDA inference at batch 128 is 6.852/6.856/6.855/6.835 ms for
random/V4/U1/U2, with 0.3462 GiB peak device allocation in every case; these
sub-percent differences are treated as matched runtime, not a speed claim.

Compiled `gcc -O0` circuit measurements at batch 128 are 3.230, 3.163,
3.136, and 3.185 ms for random/V4/U1/U2. U2's simplified circuit has 262,260
IR nodes versus 252,936 random (+3.686%), while peak export RSS is 1.138 GiB
versus 1.114 GiB. Thus the central trade-off is +3.260 pp hard-test accuracy
at exact declared gate/routing/training cost and matched measured runtime,
with 1.155 seconds of additional offline topology construction and a 3.686%
larger post-simplification IR. Functional equivalence passed for every
exported circuit. Energy was not measured.

Machine-readable sources:

- `summary/second_round_convolutional_final.json`;
- `summary/second_round_convolutional_curves.json` and CSV;
- `summary/second_round_convolutional_deployment.json`;
- `logs/second_round_convolutional_final_test/test_evaluation_summary.json`.

## Third round: U2 on dense M/L and published connectivity protocols

All 38 predeclared CUDA trainings completed. Before held-out access,
`summary/third_round_validation_freeze.json` hashed each run configuration,
environment, summary, and both best-validation and final checkpoints (76
checkpoint hashes). Both checkpoints were evaluated once per run on the
10,000-example CIFAR-10 test set using two GPUs. The evaluation summary records
38/38 successes, zero failures, and zero missing results.

### Existing dense M/L transfer

| Scale | U2 hard test | Paired gain vs random | Paired U2 vs V3 | Time | Peak GPU | Gates / parameters |
|---|---:|---:|---:|---:|---:|---:|
| M, 4 x 128K | **58.653 +/- 0.168% (n=3)** | **+4.557 pp**, CI [+3.781, +5.332], 3/3 | +0.297 pp, CI [-0.580, +1.174], 2/3 | 41.39 min | 1.123 GiB | 512K / 8.192M |
| L, 5 x 256K | **60.463 +/- 0.348% (n=3)** | **+4.593 pp**, CI [+3.721, +5.466], 3/3 | -0.610 pp, CI [-0.983, -0.237], 0/3 | 113.53 min | 2.717 GiB | 1.28M / 20.480M |

U2 therefore transfers strongly versus fixed random at both scales without a
method change. It is statistically tied with frozen V3 on M and trails V3 on
L. V3 remains the best dense specialization; U2 remains the unified method
shared with convolutional networks.

### LILogicNet protocol

| Coordinate | Fixed random T | U2 T | Paired gain | Top-32 local / reported | U2 / Top-32 training parameters | U2 / Top-32 peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| M, 1 x 64K | 49.010 +/- 0.426% | **52.543 +/- 0.296%** | **+3.533 pp**, CI [+1.797, +5.270], 3/3 | 57.840% / 57.28 +/- 0.30% | 1.024M / 5.120M | 0.474 / 8.100 GiB |
| L, 2 x 128K | 55.333 +/- 0.469% | **60.193 +/- 0.286%** | **+4.860 pp**, CI [+3.083, +6.637], 3/3 | 62.030% / 60.98 +/- 0.19% | 4.096M / 20.480M | 1.557 / 24.861 GiB |

All random/U2 cells use three paired seeds; Top-32 follows the frozen one-seed
expensive-comparator policy. U2 has the same gates, parameters, memory, and
training effort as fixed random. It uses 5x fewer trainable parameters than
Top-32, 16--17x less peak allocated training memory, and 4.9--8.1x lower local
hardened inference latency. Top-32 is 5.297 pp more accurate on M and 1.837 pp
on L. This is a clear accuracy--resource Pareto result rather than absolute
accuracy dominance.

The local fixed controls are close to the paper (49.010 versus reported 49.17
on M; 55.333 versus 54.76 on L), and the one-seed Top-32 results are also close
or higher than reported. These agreements support the protocol mapping.

### BitLogic two-layer protocol

| Scale | Rank-2 random T | U2 T | Paired gain | Rank-4 local T | Rank-4 reported |
|---|---:|---:|---:|---:|---:|
| S, 8K | 26.175 +/- 0.445% | **28.435 +/- 1.648%** | +2.260 pp, 2/2; CI inconclusive | 27.500 +/- 0.113% | 38.93 +/- 0.19% |
| M, 32K | 25.945 +/- 0.870% | 26.040 +/- 0.014% | +0.095 pp, 1/2; inconclusive | 16.625 +/- 0.078% | 49.22 +/- 0.26% |
| L, 128K | 25.160 +/- 0.156% | 25.930 +/- 2.871% | +0.770 pp, 1/2; inconclusive | 13.160 +/- 0.240% | 58.06 +/- 0.14% |

The S U2 result is directionally positive on both seeds, while M/L are
inconclusive. The rank-4 local arms are not faithful reproductions: relaxed
models learn signal but hardening collapses. At M, for example, final relaxed
test accuracy averages 57.60% while final hard accuracy is 14.70%. The local
rank-4 values are retained as `[REPRODUCED-NEGATIVE]` protocol-transfer data,
and the paper values remain separately labeled `[REPORTED]`.

Machine-readable sources are `summary/third_round_results.json`,
`summary/third_round_runs.csv`, and `summary/third_round_groups.csv`. Every
synthetic benchmark and held-out checkpoint hash matches the validation
freeze. Focused topology/protocol verification passes with 145 tests; the six
new third-round-specific provenance/aggregation tests also pass. The complete
TorchLogix suite passes with 3,412 passed, 3,038 skipped, and one pre-existing
warning in 257.62 seconds.

## Remaining limitations before submission

- A protocol-identical numerical reproduction of the published CIFAR-10
  baseline. The architecture is exact, but the current frozen-validation
  protocol uses 45,000 rather than all 50,000 training images and 240 rather
  than 200 effective epochs.
- Butterfly and pure-coverage schedules on every budget/depth cell. The
  CIFAR-10 M butterfly component is now complete, but the specification's
  full four-schedule matrix is not.
- Candidate-pool, swap-fraction, and novelty-weight ablations. Pool 8, fraction
  0.25, and weight 1.0 were frozen for the central study after the bounded
  three-seed pilot; they were not exhaustively optimized.
- Overlap-off and distance/locality ablations at pilot scale. The balanced
  fan-out component has now been isolated on CIFAR-10 M.
- WARP/Light repetition.
- RigL remains unimplemented. Six-layer 48K Mommen, LILogic, and BitLogic
  controls are complete on MNIST/Fashion under the common split and epoch
  budget. BitLogic's chance-level outcome is a reproduced-negative six-layer
  transfer result, not a failure to finish the runs.
- Multi-seed long-training convolutional confirmation. The full 350K S cohort
  is positive by +3.260 pp on held-out test and exceeds the paper-reported S
  test value by 0.250 pp, but has one full seed per method. Its supporting
  20K method-level pilot has three seeds. The one-seed 200K nine-channel M
  comparison is positive by +0.39 pp on held-out test but is 1.05 pp below
  the paper's reported test accuracy and has no confidence interval.
- Optimized compiled CPU latency for M and direct energy measurements. Circuit
  equivalence and an `-O0` compiled CPU measurement are complete for S; M has
  trace, simplification, equivalence, and framework-level PyTorch GPU timing.
- Direct energy and optimized hardware-synthesis measurements. The present
  methods have identical declared gates, operations, and deployed routing
  storage; U2 improves accuracy at matched measured CUDA/CPU runtime but has
  a 3.686% larger simplified IR in the full S checkpoint snapshot.

The superseded fraction-0.25 v2 hybrid failed the formal kill criterion.
Frozen V3 remains the strongest dense specialization, while U2 is the
separate generic rule shared by dense and convolutional architectures. U2's
full S result clears the requested +3 pp test-gain target without changing
gate count, routing storage, training effort, memory, or measured runtime.
The responsible next additions are two more full S random/U2 seeds,
protocol-identical published-baseline reproduction, and hardware energy/
synthesis evidence. `SECOND_ROUND_CONCLUSIONS.md` gives the complete claim,
trade-off, scope, and nine-step disposition.

## Final verification

Final verification after regenerating the summary artifacts:

- the August 10 evidence-consistency audit reports `pass` for all legacy and
  second-round checks, including 110/110 completion, frozen-before-test
  provenance, exact declared cost, U2 accuracy, learning curve, and deployment
  equivalence;
- the second-round focused topology/protocol suite passes with **129 passed**
  in 86.07 seconds;
- the reproducibility audit reports `pass` for expected-run completeness,
  54-point learning curves, paired protocol equality, held-out metrics,
  inference benchmarks, environment consistency, and the documented v3 source
  cohorts;
- the follow-up audit reports `pass` for 52 medium/depth/convolutional records,
  paired protocol equality, and bit-identical convolutional spatial hashes;
- the July 24 ancestry/classifier focused suite passes with **1,862 passed and
  1,660 skipped**;
- the July 29 topology/protocol/circuit/task-aware focused suite passes with
  **157 passed** in 208.18 seconds, including post-rewire circuit equivalence;
- the complete TorchLogix suite passes with **3,367 passed, 3,038 skipped, and
  one pre-existing warning** in 244.56 seconds;
- all three generated SVGs parse as valid XML;
- the pre-analysis source archive passes its recorded SHA-256 check; and
- `nvidia-smi` reports both RTX PRO 6000 GPUs at 0% utilization with no running
  compute processes after the queues completed.
