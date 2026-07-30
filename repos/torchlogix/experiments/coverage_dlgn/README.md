# CoverageDLGN experiments

This directory contains the rank-2 CoverageDLGN implementation package. It
follows the staged scope in `ideas/date_ideas/coverage_dlgn.md`: dense fixed
connections and topology metrics were stabilized first, validated at two gate
budgets and three depths, and only then extended across convolutional channels.

The paper-architecture comparison is recorded in `ARCHITECTURE_AUDIT.md`.
Published and independently reproduced CIFAR-10 accuracy values, together with
their comparability restrictions, are recorded in
`CIFAR10_BASELINE_REFERENCE.md`.
Legacy convolutional classes and results are preserved; new
`ClgnCifar10PaperSmall` and `ClgnCifar10PaperMedium` classes correct the
published S/M input encoding to three thermometer thresholds without changing
their gate budgets.

## Environment

Use only the repository virtual environment.  The validated local stack is:

- Python 3.12.13;
- PyTorch 2.9.0+cu130 and torchvision 0.24.0+cu130;
- NVIDIA driver 580.159.03 and CUDA 13.0;
- NVIDIA RTX PRO 6000 Blackwell.

Installation with `uv`:

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python \
  torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python venv/bin/python -e . pytest
```

Never install dependencies outside `repos/torchlogix/venv`.

## Implemented schedules

All strategies use the existing fixed `indices` buffer and add no trainable
routing parameters:

- `random`: legacy behavior, or an independent local RNG when a topology seed is supplied;
- `random_unique`: unique unordered predecessor pairs when feasible;
- `local_cyclic`: cyclic pairs within a bounded radius;
- `butterfly`: increasing-stride regular stages with deterministic affine output ordering;
- `coverage_greedy`: bounded candidate search using ancestry union, overlap, fan-out, and distance;
- `coverage_hybrid`: affine-ordered butterfly/local base with a fixed fraction of greedy long-range pairs;
- `semantic_balanced_hybrid`: a tensor-aware first layer followed by
  degree-preserving, ancestry/overlap/novelty-scored swaps.
- `semantic_channel_hybrid`: the frozen v3 butterfly/swap rule applied only to
  convolutional channel groups; spatial receptive-field coordinates are left
  bit-identical to the matched random run.
- `ancestry_channel_hybrid`: balanced convolutional channel pairs followed by
  cross-block ancestry/novelty swaps; spatial coordinates remain bit-identical
  to the matched control.
- `semantic_classifier_hybrid`: a classifier-tail adaptation of v3 whose
  shrinking layers use balanced round-robin matchings before ancestry swaps.
- `coverage_reuse_hybrid`: a generic degree-preserving refinement of the
  frozen v3/v4 base topology that trades ancestry novelty against reuse of
  predecessor motifs. It is retained as an experimental negative result after
  its CIFAR-10 M confirmation failed.

The affine ordering is important.  It leaves the regular predecessor pair set
and fan-out unchanged but prevents consecutive output/class groups from seeing
only consecutive regions of a flattened image.

Ancestry is represented by packed `uint64` bitsets during construction and
analysis.  It is released after model construction.  Only connection indices
remain in deployed state.

For image inputs, semantic analysis maps every encoded bit back to
`(channel, y, x, threshold)` and collapses correlated threshold bits into a
single raw source. The v3 first layer interleaves spatial scales and semantic
axes, forbids pairing two thresholds from the same source, and balances
predecessor use. Deeper-layer two-edge swaps preserve the exact predecessor
degree sequence while improving normalized raw-source ancestry, overlap, and
cross-gate novelty. The first layer is not swapped so its image structure is
retained.

## Topology-only analysis

No dataset or model training is needed:

```bash
venv/bin/python experiments/coverage_dlgn/analyze_topology.py \
  --config experiments/coverage_dlgn/configs/topology_hybrid_mnist_tiny.json
```

This writes per-depth CSV and JSON reports containing input coverage, gate
ancestry, predecessor overlap, fan-out statistics, distinct pairs, reachability,
routing bits, construction time, and temporary generator memory. For image
models, the report also includes raw-source ancestry, spatial-distance and
channel/threshold pairing statistics, group coverage, and adjacency diversity.

The paper-scale v3 topology configurations are:

```bash
venv/bin/python experiments/coverage_dlgn/analyze_topology.py \
  --config experiments/coverage_dlgn/configs/topology_semantic_balanced_fashion_paper.json
venv/bin/python experiments/coverage_dlgn/analyze_topology.py \
  --config experiments/coverage_dlgn/configs/topology_semantic_balanced_cifar_paper.json
```

Frozen random/v2 checkpoints can be re-analyzed with
`analyze_checkpoint_topology.py`; this is how the bit-versus-source ancestry
failure was diagnosed without retraining.

## Training

Configuration values are parser defaults; explicit command-line values override
them.  Dataset files may be placed outside the repository:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/smoke_coverage_hybrid.json
```

Runs made by the finalized runner contain:

- `training_config.json`;
- `environment.json` with revision, source-tree hash and manifest, packages,
  CUDA, and GPU;
- `metrics.csv` and `thresholds.csv`;
- `topology.csv` and `topology.json`;
- `conv_topology.csv` and `conv_topology.json` for convolutional models;
- legacy raw state dicts plus self-describing best/final checkpoints;
- `run_summary.json` with hard accuracy, time, and peak GPU memory.

The pre-change baseline and some early smoke directories predate the environment
manifest. Their exact configurations and metrics are retained, but they must not
be mistaken for final archival runs. Likewise, rich checkpoints produced before
the safe-metadata fix require `torch.load(..., weights_only=False)` and should be
opened only when they are trusted local artifacts; the parallel raw state-dict
files remain loadable with the safe default. Newly generated rich checkpoints
contain only tensors and primitive metadata and are covered by a safe-load test.
The current manifest records both a broad source fingerprint and a
training-only implementation fingerprint over `src/torchlogix`,
`experiments/train.py`, and `experiments/utils.py`. This prevents reporting,
queue, or configuration edits from changing the identity of future training
code. Historical Fashion/CIFAR manifests predate that correction. The checksummed
`summary/training_source_pre_analysis.tar.gz` archive preserves the exact source
state that produced those runs.

For CIFAR-10, `augmentation: standard` applies random crop and horizontal flip
only to the training subset.  Validation uses deterministic `ToTensor` input.

### Paper-scale MNIST reproduction

The explicit paper-small model has six 8,000-gate layers. Run the matched seed-0
pair with:

```bash
DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/paper_mnist_small_random_seed0.json
DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/paper_mnist_small_hybrid_v2_seed0.json
```

The seed-1 and seed-2 configuration files repeat the same protocol. All runs use
split seed 2027, 108,000 steps (200 epochs of the 54,000-example training split),
batch size 100, paper-style random-normal gate initialization, and best hardened
validation accuracy for model selection.

Only after all settings and checkpoints are frozen, held-out test evaluation is:

```bash
DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python \
  experiments/coverage_dlgn/evaluate_checkpoint.py RUN_DIR [RUN_DIR ...]
```

The corresponding fixed-batch hardened GPU benchmark is:

```bash
DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python \
  experiments/coverage_dlgn/benchmark_checkpoint.py RUN_DIR [RUN_DIR ...]
```

The three-seed result is 97.15% versus 97.18% mean hard validation accuracy and
97.18% versus 97.25% mean hard test accuracy for random versus hybrid-v2. The
paired gains (+0.033 pp validation and +0.070 pp test) are statistically
inconclusive and are not presented as a positive claim.

## Recreating tables and the figure

```bash
venv/bin/python experiments/coverage_dlgn/summarize_results.py
venv/bin/python experiments/coverage_dlgn/summarize_followup.py
```

Only complete runs with all required artifacts enter aggregate tables.
Incomplete attempts under `results/failed/` are excluded. Outputs under
`experiments/coverage_dlgn/summary/` include:

- per-run and paper-architecture CSV tables;
- paired validation and held-out-test JSON reports with Student-t intervals;
- Fashion fraction-sweep data;
- mean/std learning-curve CSV and SVG files;
- the coverage-versus-accuracy SVG;
- a machine-readable reproducibility audit; and
- the checksummed pre-analysis source archive.

The follow-up summarizer adds `followup_runs.csv` and
`followup_summary.json`. It audits 52 medium/depth/convolutional run records,
paired protocol equality, and identical convolutional spatial-coordinate
hashes.

## Protocol boundaries

- The validation split is used for the smoke fraction sweep.
- Smoke settings never used held-out test accuracy. Frozen paper-scale MNIST,
  Fashion-MNIST, and CIFAR-10 central checkpoints were evaluated once on test
  after all validation work ended.
- The three-seed, 1,000-step MNIST runs are mechanism smokes, not the DATE result.
- The completed v2 Fashion/CIFAR study is a superseded three-seed
  kill-criterion pilot. V3 has five central seeds, clears the two-dataset
  continuation criterion, and has now been tested at 48K/512K gates and
  4/8/12 layers. The 12-layer cells expose an optimization failure rather than
  a reliable routing gain. Component ablations and protocol-identical named
  baselines are still required for a DATE claim.
- The CIFAR architecture matches the published small model, but the local
  90/10 split and 108,000 steps equal 240 effective epochs. It is therefore not
  yet a protocol-identical numerical reproduction of the paper's full-training,
  200-epoch result.
- The convolutional v4 experiment is a 20K-step, three-seed pilot of
  `ClgnCifar10Small`, not a numerical reproduction of the paper's long
  LogicTreeNet training. It uses the paper hyperparameters (standard
  augmentation, AdamW, batch 128, learning rate 0.02, weight decay 0.002) and
  selects only on hardened validation accuracy.

## Medium, depth, and convolutional follow-up

The 512K-gate `DlgnCifar10Medium` escalation uses five seeds and 108K steps:

```bash
bash experiments/coverage_dlgn/run_cifar_medium_v3_full_two_gpus.sh
bash experiments/coverage_dlgn/run_cifar_medium_v3_evaluate_two_gpus.sh
```

Mean held-out hardened accuracy is 54.028% for random routing and 58.284% for
v3. The paired gain is +4.256 percentage points with a 95% Student-t interval
of [+3.851, +4.661]. Mean hardened latency is approximately 3.57 versus
3.53 ms per 100 examples. V3 topology construction is a one-time offline cost
of about 107 seconds versus 5 seconds for seeded random construction.

The controlled 20K-step study holds total gates and maximum GroupSum logit
scale fixed while varying depth:

```bash
bash experiments/coverage_dlgn/run_cifar_depth_v3_two_gpus.sh
```

At 48K gates, mean paired v3 gains at depths 4/8/12 are +3.153, +2.647, and
−1.720 points. At 512K gates they are +4.473, +3.593, and −0.547 points.
The 4- and 8-layer intervals are above zero; both 12-layer cells are
unstable/near chance.

The convolutional channel extension and its frozen evaluation are reproduced
with:

```bash
bash experiments/coverage_dlgn/run_conv_cifar10_small_v4_pilot_two_gpus.sh
bash experiments/coverage_dlgn/run_conv_cifar10_small_v4_evaluate_two_gpus.sh
venv/bin/python experiments/coverage_dlgn/analyze_conv_checkpoint_topology.py \
  experiments/coverage_dlgn/results/pilot_conv_cifar10_small_*_seed?
```

V4 improves all three held-out seed pairs by +2.06, +2.75, and +1.19 points.
The mean changes from 55.42% to 57.42%; the paired 95% interval is
[+0.058, +3.942]. Hardened latency is unchanged at 6.87 ms per 128 examples.
The model has 83,552 learned gate functions and 874,496 spatial gate
applications. Both variants have identical counts, tensor shapes, spatial
hashes, and initial weights.

## Paper-faithful convolutional S/M pilot

The next convolutional stage uses the paper-specific S/M classes. Both retain
the four depth-3 blocks, OR pooling, paper widths and temperatures, and the
two-channel receptive-field restriction. Unlike the legacy pilot, 2-bit RGB
is encoded by the three thresholds stated on page 6 of the paper.

Paired smoke configurations have completed on CUDA:

```bash
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/smoke_conv_cifar10_paper_small_random_seed0.json
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/smoke_conv_cifar10_paper_small_semantic_channel_v4_seed0.json
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/smoke_conv_cifar10_paper_medium_random_seed0.json
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/smoke_conv_cifar10_paper_medium_semantic_channel_v4_seed0.json
```

The three-seed, 20K-step paired pilots and frozen held-out evaluations run
across both GPUs. Run S first, inspect its paired result, and then run M:

```bash
bash experiments/coverage_dlgn/run_conv_cifar10_paper_sm_pilot_two_gpus.sh small
bash experiments/coverage_dlgn/run_conv_cifar10_paper_sm_evaluate_two_gpus.sh small
bash experiments/coverage_dlgn/run_conv_cifar10_paper_sm_pilot_two_gpus.sh medium
bash experiments/coverage_dlgn/run_conv_cifar10_paper_sm_evaluate_two_gpus.sh medium
```

Smoke accuracies are not evidence. Their acceptance criteria are successful
CUDA forward/backward, hard evaluation for S, checkpoint creation, three saved
thresholds, identical random/v4 tensor shapes, and identical spatial hashes.

The S 20K pilot is complete. Mean hardened validation changes from 56.673% to
57.187% (+0.513 pp), while held-out test changes from 56.140% to 56.367%
(+0.227 pp). Three-seed confidence intervals include zero and individual test
effects are mixed, so this is not paper-level evidence. Exact values are in
`summary/paper_conv_small_pilot.json`. M remains necessary to test whether the
stronger topology diversification at larger width translates into a stable
accuracy gain.

The complementary ancestry/classifier-tail revision is also complete on
paper-S. A 5K seed-0 screen selected ancestry-only v5 (+2.84 pp) over random,
v4+v3-tail (+1.42 pp), and ancestry-v5+v3-tail (+2.64 pp). The frozen 20K
three-seed pilot did not confirm the screen: paired validation gains were
+1.88, -0.98, and -1.16 pp, for a mean of -0.087 pp and a 95% interval of
[-4.324, +4.150]. V5 was therefore not evaluated on held-out test data.

The exact paired pilot is launched on both GPUs with:

```bash
bash experiments/coverage_dlgn/run_conv_cifar10_paper_small_ancestry_v5_pilot_two_gpus.sh
```

The runner refuses to append to existing result directories.

V5 nevertheless supplies the intended topology diagnostic: last-block mean
raw-channel ancestry rises from 5.414 to 8.195 of nine inputs, distinct groups
rise from 511 to 1,024, and predecessor overlap falls from 0.575 to 0.488 at
identical gate and spatial-operation budgets. Frozen v4 remains the retained
convolutional method, while V5 remains available as a documented
negative-result mechanism. Exact values are in
`summary/paper_conv_small_ancestry_v5_pilot.json`.

The subsequent coverage--reuse-balanced revision is documented in
`COVERAGE_REUSE_METHOD.md`. It preserved exact fan-out, spatial coordinates,
gate budgets, routing storage, and training operations. On seed 0 it improved
paper-S best hard validation from 54.92% to 55.40% in 5K steps (+0.48 pp).
An early paper-M 1K screen showed +3.14 pp, but the decisive 5K confirmation
reversed the outcome: frozen v4 reached 61.56% and coverage--reuse reached
58.44% (-3.12 pp). No multi-seed or held-out test escalation was performed.
The generic implementation remains available for controlled ablation, while
v4 remains the selected convolutional method. Machine-readable values are in
`summary/paper_conv_coverage_reuse_screen.json`.

Paper B follows only after S/M. The existing `ClgnCifar10Large` is merely the
same `k=512` scale: it lacks the doubled output layer, fixed edge/curvature
preprocessing, and teacher protocol, so it must not be labeled a published B
reproduction.

## Paper-architecture Fashion-MNIST and CIFAR-10 study

The Fashion-MNIST comparison uses `DlgnFashionMnistPaperSmall` (six layers by
8,000 gates), not the legacy five-layer `DlgnFashionMnistSmall`. The dependent
CIFAR-10 comparison uses the published dense `DlgnCifar10Small` architecture
(four layers by 12,000 gates). Both have 48,000 gates, three fixed uniform
thresholds, no augmentation, and a frozen 90/10 split with split seed 2027.

The completed long runs were file-logged and scheduled across both GPUs by:

```bash
bash experiments/coverage_dlgn/run_fashion_gpu0_queue.sh
bash experiments/coverage_dlgn/run_fashion_gpu1_queue.sh
bash experiments/coverage_dlgn/run_cifar_after_fashion_queue.sh
```

The CIFAR queue waited for all three paired Fashion random/hybrid runs and
completed all fixed-random seeds before starting hybrid runs. In total, all 16
expected runs completed: ten Fashion runs (three central pairs plus a five-point
seed-0 fraction sweep, with fraction 0.25 shared) and six CIFAR central runs.
Incomplete attempts are retained under `results/failed/` and never enter
aggregate tables.

### Superseded v2 pilot outcome

| Dataset | Random hard test | Hybrid hard test | Paired difference |
|---|---:|---:|---:|
| Fashion-MNIST | 86.3033% | 86.3500% | +0.047 pp |
| CIFAR-10 | 48.9133% | 50.6700% | +1.757 pp |

The CIFAR gain is consistent across the three seeds, but Fashion is a tie.
Because the gate, operation, and routing-storage budgets are identical, the
fraction-0.25 hybrid fails the specification's required improvement on both
datasets and does not meet its alternative Pareto condition. Full numerical
details, confidence intervals, topology/resource metrics, and failure history
are in `RESULTS.md`.

## Semantic-balanced v3 study

The v3 settings were selected with three paired 20,000-step CIFAR validation
pilots, then frozen: candidate pool 8, degree-preserving swap fraction 0.25,
and novelty weight 1.0. The same settings were transferred to Fashion-MNIST
without retuning. Full runs use the same 48,000-gate architectures, split,
batch size, optimizer, parametrization, and 108,000-step effort as random.

The paired seed 0--2 pilot can be launched across both GPUs with:

```bash
bash experiments/coverage_dlgn/run_cifar_v3_pilot_two_gpus.sh
```

The five-seed central queues are:

```bash
bash experiments/coverage_dlgn/run_cifar_v3_full_two_gpus.sh
bash experiments/coverage_dlgn/run_fashion_v3_full_two_gpus.sh
bash experiments/coverage_dlgn/run_v3_seeds3_4_two_gpus.sh
```

The final hardened results are:

| Dataset | Random validation | V3 validation | Difference | Random test | V3 test | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 87.4767% | 88.1667% | +0.690 pp | 86.3080% | 87.1580% | +0.850 pp |
| CIFAR-10 | 49.6920% | 53.1160% | +3.424 pp | 49.0560% | 52.3580% | +3.302 pp |

All values are five-seed means. Paired 95% intervals exclude zero for
validation and held-out test on both datasets. Gate count, LUT parameters,
routing storage, framework-level hardened inference time, and peak training
GPU allocation are unchanged; v3 adds only deterministic offline topology
construction (about 5 seconds for Fashion and 9 seconds for CIFAR).

This cleared the specification's continuation criterion. The subsequent
budget/depth study and CIFAR-10 M component ablation are now complete;
WARP/Light, named sparse-routing baselines, and deployment Pareto measurements
remain—not yet a standalone DATE claim. Exact per-seed results and
experimental-history caveats are in `RESULTS.md`.

## Deep dense CIFAR-100 extension

The exact 6-by-64K scalability architecture and 6-by-256K multilinear
architecture are declared in
`protocols/table4_dense_cifar100_deep.json`. This extension changes no V3
mechanism. It screens only the already used V3 swap fractions 0.125, 0.25,
and 0.5 against matched fixed random.

Generate and summarize each frozen stage with:

```bash
venv/bin/python experiments/coverage_dlgn/prepare_table4_cifar100_deep_screen.py
venv/bin/python experiments/coverage_dlgn/summarize_table4_cifar100_deep_screen.py
venv/bin/python experiments/coverage_dlgn/prepare_table4_cifar100_deep_selection.py
venv/bin/python experiments/coverage_dlgn/summarize_table4_cifar100_deep_selection.py
venv/bin/python experiments/coverage_dlgn/prepare_table4_cifar100_deep_final.py
venv/bin/python experiments/coverage_dlgn/summarize_table4_cifar100_deep_final.py
```

The queues produced by the preparation scripts contain the exact training
commands and output directories. After the full 6-by-64K validation winner
was frozen, its six locked checkpoints were evaluated once with:

```bash
venv/bin/python experiments/coverage_dlgn/evaluate_table4_cifar100_deep_final.py
```

The resulting held-out test means are 20.677% for random and 21.010% for
CoverageDLGN V3 (`swap_fraction=0.125`), a paired +0.333 pp over three seeds.
The exact 6-by-256K branch stopped at the 5K screen because all three frozen
V3 controls trailed random; it has no multi-seed or held-out-test result.
Machine-readable summaries are
`summary/table4_cifar100_deep_{screen,selection,final}.json`, while the
interruption and recovery history is recorded in `EXPERIMENT_LOG.md`.

### Fixed-384K depth ablation

The controlled depth experiment is frozen in
`protocols/table4_dense_cifar100_depth384k.json`. It compares 3-by-128K,
12-by-32K, and 24-by-16K networks at exactly 384K gates using unchanged V3.
Generate and summarize its validation-only queue with:

```bash
venv/bin/python experiments/coverage_dlgn/prepare_table4_cifar100_depth384k_pilot.py
venv/bin/python experiments/coverage_dlgn/run_gpu_queue.py \
  --queue experiments/coverage_dlgn/queues/table4_cifar100_depth384k_pilot.json \
  --gpus 0 1 --data-path /tmp/torchlogix-datasets
venv/bin/python experiments/coverage_dlgn/summarize_table4_cifar100_depth384k_pilot.py
```

At 20K steps, 3-by-128K gains +0.780 pp (21.860% versus 21.080%), below the
predeclared +1 pp threshold. The 12- and 24-layer pairs remain at chance,
and depth 24 gives every final gate complete raw-source ancestry under both
random and V3. No extra seed or held-out test was run.

### Class-conditional coverage head

`CLASS_CONDITIONAL_HEAD.md` documents the separate final-layer refinement,
its invariants, and its stopped CIFAR-100 result. It does not alter frozen V3
or V4. The three-seed 20K pilot reduced mean class source-usage CV from
0.25655 to 0.23475 but produced only +0.013 pp over V3 and +0.553 pp over
random. It failed both predeclared promotion gates, so no full, transfer,
convolutional, or held-out-test run followed.

### CIFAR-10 compression completion and V3 components

The frozen 256K and 384K checkpoints were evaluated exactly once with:

```bash
venv/bin/python experiments/coverage_dlgn/evaluate_table2_compression_remaining.py \
  --gpus 0 1 --data-path /tmp/torchlogix-datasets
venv/bin/python experiments/coverage_dlgn/summarize_table2_compression_remaining_test.py
```

At 256K gates, V3 reaches 56.903% +/- 0.134% test versus
52.253% +/- 0.058% random (+4.650 pp, 95% CI [+4.174, +5.126]). At 384K,
V3 reaches 58.143% +/- 0.153% versus 53.657% +/- 0.328% (+4.487 pp,
95% CI [+3.515, +5.458]). These checkpoints must not be queried on test
again.

The CIFAR-10 M component ablation is reproduced with:

```bash
venv/bin/python experiments/coverage_dlgn/prepare_cifar10_medium_v3_components.py
venv/bin/python experiments/coverage_dlgn/run_gpu_queue.py \
  --queue experiments/coverage_dlgn/queues/cifar10_medium_v3_components.json \
  --gpus 0 1 --data-path /tmp/torchlogix-datasets
venv/bin/python experiments/coverage_dlgn/summarize_cifar10_medium_v3_components.py
```

It reuses the existing random/full-V3 controls. Balanced fan-out contributes
+4.160 pp over random; semantic first-layer scheduling adds +0.273 pp and
ancestry swaps add +0.040 pp, with the latter two intervals crossing zero.
Full V3 remains +4.473 pp over random.

### Task-aware rewiring negative result

`TASK_AWARE_REWIRING.md` documents the optional one-shot
activation-gradient extension. It is disabled by default and does not change
frozen V3. The three-seed 20K CIFAR-10 M pilot reached 59.093% +/- 0.234%:
+4.273 pp over random but -0.200 pp versus V3. It failed its strict promotion
gate, so no full schedule, held-out test, transfer, or convolutional run
followed.

### CIFAR-100 baseline audit

`CIFAR100_BASELINE_AUDIT.md` records why the local 20.677% random result is
`[REPRODUCED, topology-adapted]` rather than bit-faithful to the reported
22.54%. The architecture and schedule match, but independent TorchLogix
topology seeding differs from canonical difflogic's two-`randperm` routing.
The audit did not retrain or re-query any checkpoint.

### Convolutional V4 component and channel-spatial study

`CONV_CHANNEL_SPATIAL_ADAPTER.md` defines the separate convolutional
channel-spatial adapter, its common-RNG invariants, and promotion gate. The
corrected three-seed CIFAR-10 S study reuses historical random/V4 controls.

Balanced channel routing without swaps reached 58.013% validation
(+1.340 pp over random, +0.827 pp over V4), but both intervals cross zero.
The channel-spatial adapter reached 57.033% (+0.360 pp over random,
-0.153 pp versus V4) and failed both promotion thresholds. No M or held-out
run followed. The invalid explicit-classifier RNG attempt is preserved and
excluded in `results/failed/cifar10_conv_small_explicit_classifier_rng_attempt1`.
