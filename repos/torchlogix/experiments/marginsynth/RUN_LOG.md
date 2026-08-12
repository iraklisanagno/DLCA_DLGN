# MarginSynth experiment log

This ledger records every training, calibration, rewriting, compilation, and
synthesis run used during MarginSynth development. Generated run directories
contain the complete resolved configuration, environment fingerprint, exact
dataset indices and hashes, metrics, checkpoints, and console output.

Development runs are not paper results. A run may enter a paper table only
after the protocol and hyperparameters are frozen, all required seeds complete,
and the held-out test policy in
[PAPER_EVALUATION_REFERENCE.md](PAPER_EVALUATION_REFERENCE.md) is satisfied.

## Run registry

| Run ID | Purpose | Dataset | Model | Seed | Status | Configuration | Artifacts |
|---|---|---|---|---:|---|---|---|
| `dev_mnist_tiny_raw_seed0` | First trained circuit for export, rewrite, and incremental-simulation development | MNIST | `DlgnMnistTiny`, rank-2 raw, fixed random connections | 0 | Completed and verified (development only) | [config](configs/dev_mnist_tiny_raw_seed0.json) | `results/dev_mnist_tiny_raw_seed0/` |
| `pilot_fashion_mnist_paper_small_raw_seed0` | First dense scalability pilot and direct Two-Stage Unit Tying baseline | Fashion-MNIST | `DlgnFashionMnistPaperSmall`, rank-2 raw, fixed random connections; 6 × 8,000 = 48,000 nominal gates | 0 | Completed and verified; test sealed | [training config](configs/pilot_fashion_mnist_paper_small_raw_seed0.json), [unit-tying config](configs/two_stage_unit_tying_fashion_seed0.json) | `results/pilot_fashion_mnist_paper_small_raw_seed0/` |
| `margin_all_aig_holdout_v3_seed0` | Independent whole-circuit LUT resynthesis central pilot | Fashion-MNIST | Same frozen checkpoint; four internal layers jointly optimized on GPU | 0 | Completed and synthesized; seed-0 promotion failed | [config](configs/circuit_distillation_fashion_seed0_v3.json) | `results/pilot_fashion_mnist_paper_small_raw_seed0/distillation/margin_all_aig_holdout_v3_seed0/` |
| `ablation_matrix_v3_seed0` | Margin/MSE, action-space, proxy, group-robustness, and repair ablations | Fashion-MNIST | Same frozen checkpoint and partitions | 0 | Five ablations completed and synthesized; test sealed | [runner](run_circuit_distillation_ablations.py) | `results/pilot_fashion_mnist_paper_small_raw_seed0/distillation/ablation_matrix_v3_seed0/` |
| `marginsynth_bayesian_fashion_seed0_v2` | Four-case constrained MOTPE exploration of guarded two-pass and aggressive/recovery MarginSynth | Fashion-MNIST | Same frozen six-layer, 48,000-gate checkpoint; four editable internal layers | 0 | 160/160 acquisition trials and 27/27 exact promotions completed; validation/test sealed | [protocol](configs/bayesian_exploration_fashion_seed0.json) | `results/pilot_fashion_mnist_paper_small_raw_seed0/bayesian_search/marginsynth_bayesian_fashion_seed0_v2/` |

## `dev_mnist_tiny_raw_seed0`

Purpose:

- create a meaningful but inexpensive hardened circuit for MarginSynth
  correctness development;
- validate deterministic train/validation/calibration isolation;
- validate archival checkpoint and environment recording; and
- provide the first real checkpoint for model/backend/Circuit equivalence.

This run is deliberately smaller than the paper architectures and must not be
used to claim state-of-the-art accuracy or compression.

Frozen protocol:

- official MNIST training split partitioned into 80% training, 10% validation,
  and 10% calibration;
- official MNIST test split untouched;
- data split seed 2027;
- training and topology seed 0;
- five dense layers with 1,000 rank-2 gates per layer;
- raw 16-gate parametrization;
- fixed random connectivity;
- residual initialization with probability 0.951;
- Adam, learning rate 0.01, batch size 128;
- 10,000 GPU optimizer iterations; and
- hard and relaxed validation every 1,000 iterations.

Launch commands:

```bash
mkdir -p experiments/marginsynth/results/dev_mnist_tiny_raw_seed0

script -e -q -c \
  "env DATASET_PATH=/tmp/torchlogix-datasets venv/bin/python experiments/train.py --config experiments/marginsynth/configs/dev_mnist_tiny_raw_seed0.json" \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0/console.log
```

Verification command:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/verify_checkpoint.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --examples 6000 \
  --pack-bits 16
```

Synthesis-verification command:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/verify_synthesis.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --examples 6000 \
  --pack-bits 16
```

Calibration-trace command:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/build_trace.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0
```

Non-mutating candidate smoke command:

```bash
venv/bin/python experiments/marginsynth/smoke_candidates.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --per-kind 32 \
  --full-checks 14
```

Greedy development-search commands:

```bash
venv/bin/python experiments/marginsynth/search.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --config experiments/marginsynth/configs/dev_search_mnist_seed0.json

venv/bin/python experiments/marginsynth/verify_search.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --search search_dev_decision_seed0
```

Generated artifacts:

- `console.log`;
- `training_config.json`;
- `environment.json`;
- `data_split.json`;
- `metrics.csv`;
- `thresholds.csv`;
- `topology.csv` and `topology.json`;
- `best_model.pt` and `best_checkpoint.pt`;
- `final_model.pt` and `final_checkpoint.pt`;
- `run_summary.json`;
- `hardened_circuit.json`;
- `exact_simplified_circuit.json`;
- `export_verification.json`; and
- `hardware_argmax_circuit.json`, `.v`, and `.blif`;
- `synthesis_verification.json`;
- `yosys_synthesis.log` and `abc_synthesis.log`; and
- `calibration_trace/`, containing packed values, scores, margins, labels,
  fan-out indices, transitive cone bitsets, and its metadata; and
- `candidate_smoke.json`, containing candidate-space counts, individual
  non-mutating evaluations, and incremental/full checks; and
- `search_dev_decision_seed0/`, containing the frozen search configuration,
  provenance, deterministic rewrite log, search summary, replay verification,
  and 26 JSON/C/Verilog snapshots; and
- `artifact_manifest.json`, containing byte counts and SHA-256 hashes for every
  other run artifact.

Results (2026-07-30 UTC):

- 80,000 trainable parameters and 10,000 optimizer steps;
- NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition;
- PyTorch 2.9.0+cu130, CUDA build 13.0;
- training wall time 75.795 seconds;
- peak allocated GPU memory 14,439,424 bytes;
- best hard validation accuracy 0.877833 at step 10,000;
- final relaxed validation accuracy 0.880500;
- final hard/relaxed validation loss 0.590356/0.588533; and
- final training loss 0.218915.

Data isolation:

| Partition | Examples | Index SHA-256 |
|---|---:|---|
| Training | 48,000 | `45761be8173c626cd55429725e9eaccfb1f443cb0cdf3c7eaad064f6bddaca66` |
| Validation | 6,000 | `07e7b56cea7caad06689f4d87ae9f0da0a180ee6c75c2bbe7f0040e9853bd70a` |
| Calibration | 6,000 | `928b1229fc37699abd53335a6e5d2b40e6512d37911a66375627b407cacbdb35` |
| Test | 10,000 | `9e1c19b3fdc185411bd1a987deb6a9fda2531116aac060a84c4d878854d1c099` |

Training used only the training partition. Model selection used only
validation. The calibration partition was not used during training or export
verification, and the test partition was not evaluated.

Export verification:

- the best archival checkpoint loaded strictly and has SHA-256
  `43734682d81973f0f25c6c23ac850cdacdede5429efdff1b742c1302b4803c3a`;
- the learned input boundary is a one-bit threshold of 0.5, applied before the
  editable Boolean backend;
- full-model, Boolean-backend, hardened-Circuit, exact-simplified-Circuit, and
  packed compiled-C scores and predictions were exactly equal across the full
  6,000-example validation partition (maximum absolute difference 0.0);
- hardening exported 3,364 Boolean primitive gates and 1,000 sum inputs;
- the exact function-preserving baseline reduced this to 2,009 primitive gates
  and 999 sum inputs, a 40.3% primitive-gate reduction; and
- export, exact simplification, and packed-C compilation took approximately
  0.22, 0.01, and 0.93 seconds, respectively.

This confirms that the trained model is a usable correctness fixture for
MarginSynth development. It is not a paper-quality accuracy or compression
result.

Synthesis-boundary verification:

- Verilog export now refuses unsupported floating `GroupSum` scores instead of
  silently discarding `tau` or `beta`;
- finite non-negative integer `beta` is included exactly in nonempty
  reductions;
- `normalized_for_hardware_argmax()` creates a non-mutating integer-score copy
  only when removing a shared positive temperature and common offset is
  guaranteed to preserve every argmax and tie;
- the development checkpoint's common `tau=10` and `beta=0` were converted to
  `tau=1` and `beta=0`;
- compiled-C predictions before and after conversion were exactly equal across
  all 6,000 validation examples, with score-transformation error 0.0;
- Yosys 0.9 completed synthesis and structural checks with 7,608 primitive
  cells before ABC mapping;
- Berkeley ABC 1.01 completed mapping with 8,885 AIG AND nodes, 53 levels,
  784 inputs, 80 score-output bits, and no latches; and
- 75 Circuit and experiment-protocol regression tests passed.

The Yosys/ABC counts above validate the pipeline only. Final paper PPA and
netlist comparisons will use a frozen, newer synthesis toolchain and identical
scripts across every method.

Calibration-trace results:

- used only the reserved 6,000-example calibration partition; validation and
  test examples were not accessed;
- all ten classes are represented, with exact class counts stored in
  `calibration_trace/metadata.json`;
- recorded 784 input nodes, 2,009 gates, 10 class outputs, and 94 packed
  64-bit words per node;
- packed scores and predictions exactly match the hardened Boolean backend
  across all calibration examples, with maximum score difference 0.0;
- raw trace arrays occupy 3,765,752 bytes and construction took approximately
  0.026 seconds;
- calibration accuracy of the untouched exact-simplified circuit is 0.868833;
- median winner margin is 1.8 score units; 105 examples (1.75%) have a zero
  winner margin and therefore depend on the declared lowest-class-index tie
  rule;
- the median affected downstream gate cone contains 2 gates, the 99th
  percentile contains 14, and the maximum contains 19; and
- exhaustive tiny-circuit tests verify packed gate values, score aggregation,
  tie handling, fan-out reachability, transitive cones, and memory-mapped
  save/load.

Rewrite and incremental-simulation mechanism check:

- safe rewrite objects now cover constant 0/1, copy A/B, invert A/B, and all
  supported alternative binary gate functions;
- apply/undo fails closed if the target changes, copy-based evaluation cannot
  mutate the accepted circuit, and JSON rewrites replay deterministically;
- exhaustive truth tables pass for every rewrite family;
- incremental simulation recomputes the target, its transitive downstream
  cone, and only affected class reductions;
- tiny-circuit tests, including an overlapping three-rewrite sequence, match
  complete resimulation exactly;
- the exact-simplified development circuit exposes 27,287 unique single-gate
  candidates over 2,009 targets: 15,948 alternative gates, 4,018 constants,
  3,779 copies, and 3,542 inversions;
- a deliberately bounded smoke study evaluated 224 candidates, 32 from each
  rewrite family, in 0.512 seconds without accepting or applying any rewrite;
- mean incremental evaluation time was 2.26 ms per candidate and p95 was
  3.13 ms;
- 14 selected real-circuit candidates matched full packed resimulation with
  exact scores and predictions; and
- the bounded sample contained 32 zero-flip candidates, providing an initial
  mechanism signal only. Because the selection is not a complete or unbiased
  search, it is not evidence of compression quality.

Overall development verification after the Fashion scalability changes: 102
Circuit, experiment-protocol, multi-threshold export, unsigned-score,
trace, rewrite, incremental-simulation, deterministic search, replay, and
starting-equivalence tests passed in 127.69 seconds.

First greedy development search:

- decision-preserving mode with a maximum 2% global disagreement budget and 5%
  per-original-class budget;
- deterministic seed 0 pool of 1,024 candidates per iteration, followed by
  exact cost evaluation of at most 64 feasible candidates;
- 25 accepted rewrites in 68.79 seconds;
- accepted composition: 13 constant-0, 5 constant-1, 4 copy-A, 1 copy-B,
  1 inversion, and 1 alternative-gate rewrite;
- every accepted rewrite was followed by exact simplification and full packed
  resimulation;
- live gates decreased from 2,009 to 1,939, an additional 70 gates or 3.48%
  beyond the exact-simplified baseline;
- connections decreased from 4,780 to 4,645 and logic depth decreased from 7
  to 6;
- the frozen proxy cost decreased from 2,249.75 to 2,172.75;
- calibration decision flips remained 0/6,000, calibration accuracy remained
  0.868833, and every per-class disagreement remained zero;
- all 25 logged rewrites replay from the baseline with matching before/after
  circuit hashes;
- the replayed final circuit matches the final saved snapshot and its fully
  rebuilt calibration trace matches the recorded summary;
- all files in all 26 snapshots pass their recorded SHA-256 checks; and
- the deterministic rewrite-log SHA-256 is
  `7e637792506b9b8dd807d8176930f3bc83a36ec40cca75f77a81a753a5400569`.

This is a bounded mechanism run, not a paper frontier. The pool evaluates only
1,024 of roughly 27,000 candidates per iteration, it stopped at the configured
25 accepted rewrites, and zero calibration disagreement does not imply
functional equivalence or zero held-out loss.

Longer budget-aware development frontier:

The first long run started directly from the exact-simplified baseline. It
accepted 100 rewrites and reached 1,493 gates at 2% calibration disagreement,
but several of its early lossy points were dominated by the verified 1,939-gate
zero-disagreement result above. This diagnostic run is retained as
`search_dev_frontier_seed0/`.

The corrected protocol first takes the free zero-disagreement reductions, then
spends the permitted behavior budget. Its frozen configuration is
`configs/dev_frontier_after_zero_mnist_seed0.json`, and its artifacts are in
`search_dev_frontier_after_zero_seed0/`.

Commands:

```bash
venv/bin/python experiments/marginsynth/search.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --config experiments/marginsynth/configs/dev_frontier_after_zero_mnist_seed0.json

venv/bin/python experiments/marginsynth/verify_search.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --search search_dev_frontier_after_zero_seed0

venv/bin/python experiments/marginsynth/synthesize_frontier.py \
  experiments/marginsynth/results/dev_mnist_tiny_raw_seed0 \
  --search search_dev_frontier_after_zero_seed0
```

Results:

- the starting circuit has different score magnitudes from the exact baseline
  but exactly the same calibration predictions;
- the chained stage accepted 100 additional rewrites in 491.56 seconds;
- accepted composition was 74 constant-0, 23 constant-1, 2 copy-A, and 1
  inversion rewrite;
- live gates decreased from the 1,939-gate zero-loss point to 1,429, a 26.30%
  reduction in this stage and a 28.87% reduction relative to the 2,009-gate
  exact-simplified baseline;
- connections decreased from 4,645 to 3,612, maximum fan-out from 12 to 11,
  and logic depth remained 6;
- calibration disagreement reached exactly 120/6,000 or 2.0%;
- worst original-class disagreement was 3.44%, below the frozen 5% cap;
- calibration accuracy changed from 0.868833 to 0.862167, a decrease of 0.667
  percentage points;
- replay verification passed all 100 rewrites and all 101 snapshots;
- a reporting-only bug in the starting-equivalence fields was corrected, after
  which a complete rerun produced the identical deterministic rewrite-log hash
  `7feb0a6929b91451768f5bbc29a3d33b95f23d6ccabc8903763dd36322451d2e`;
  and
- no validation or test examples were used.

Selected calibration frontier and same-flow ABC check:

| Budget | Actual disagreement | Live gates | ABC AND nodes | ABC levels |
|---:|---:|---:|---:|---:|
| 0.00% | 0.000% | 1,939 | 8,787 | 53 |
| 0.10% | 0.050% | 1,930 | 8,776 | 53 |
| 0.25% | 0.217% | 1,915 | 8,767 | 53 |
| 0.50% | 0.450% | 1,885 | 8,723 | 53 |
| 1.00% | 0.783% | 1,859 | 8,649 | 53 |
| 2.00% | 2.000% | 1,429 | 7,871 | 50 |

For reference, the exact-simplified starting circuit mapped to 8,885 ABC AND
nodes at 53 levels. The 2% point therefore reduces this development-flow AIG
count by 11.41% and depth by 3 levels. These are Yosys 0.9/ABC 1.01 mechanism
measurements, not final paper PPA results.

## `pilot_fashion_mnist_paper_small_raw_seed0`

Frozen architecture and training protocol:

- Fashion-MNIST with fixed thresholds 0.25, 0.5, and 0.75, giving 2,352
  Boolean inputs per image;
- six dense layers with 8,000 rank-2 gates per layer, or 48,000 nominal DLGN
  units;
- raw 16-function gate parametrization and fixed random connectivity;
- 768,000 trainable parameters;
- 80% training, 10% validation, and 10% calibration from the official training
  partition; official test data remains sealed;
- Adam with learning rate 0.01, batch size 100, and 108,000 GPU optimizer
  steps; and
- validation checkpoint selection every 2,000 steps.

The seed-0 run completed on GPU 0 in 850.85 seconds. The best hardened
validation accuracy was 0.873000 at step 36,000; final hard validation accuracy
was 0.869833.

The first export attempt exposed and corrected a verifier assumption that the
Boolean backend always received MNIST's 784 inputs. Fashion-MNIST instead
passes a `[1, 28, 84]` Boolean sample with 2,352 flattened inputs. The
generalized verifier derives this shape from the actual encoded tensor and has
regression coverage. A first GCC `-O1` packed-C build of the 33,843-gate exact
circuit ran for more than ten minutes without completing and was stopped. This
failed attempt is retained here as scalability evidence; compiled semantic
verification is rerun at `-O0`, with the optimization level recorded explicitly.

Export and calibration results:

- the selected checkpoint is `best_checkpoint.pt`, step 36,000, with SHA-256
  `edcc3334a9f2c1a34c8875767f08c8094d29aabd8c15ff9c1c1c8decc12a91f6`;
- hardening produced 65,208 primitive Boolean gates and 8,000 class-sum inputs;
- exact function-preserving simplification produced 33,843 live primitive
  gates and 7,988 class-sum inputs;
- full-model, Boolean-backend, hardened-Circuit, exact-Circuit, and packed-C
  scores were exactly equal over all 6,000 validation examples;
- the `-O0` compiled-C build took 1.81 seconds; the complete verification
  command took 134.06 seconds;
- the exact hardware-normalized baseline mapped through Yosys/ABC to 99,365
  AND nodes at 78 levels;
- the calibration trace contains 6,000 examples, 36,195 nodes, 33,843 gates,
  and 185,677,680 bytes of arrays; trace construction itself took 6.92 seconds;
- the untouched calibration accuracy is 0.869500; and
- the trace exactly matches the Boolean backend, with validation and test
  unused.

Candidate smoke:

- 432,495 unique single-gate candidates over 33,843 targets: 248,229
  alternative gates, 67,686 constants, 61,421 copies, and 55,159 inversions;
- 56 sampled candidates were evaluated incrementally and four selected
  candidates matched full packed resimulation exactly;
- mean incremental candidate time was 27.49 ms and p95 was 32.46 ms; and
- 16 of the 56 candidates changed no calibration decisions.

Search configuration and commands:

```bash
venv/bin/python experiments/marginsynth/search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --config experiments/marginsynth/configs/pilot_fashion_mnist_decision_zero_seed0.json

venv/bin/python experiments/marginsynth/verify_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --search search_pilot_decision_zero_seed0

venv/bin/python experiments/marginsynth/search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --config experiments/marginsynth/configs/pilot_fashion_mnist_frontier_after_zero_seed0.json

venv/bin/python experiments/marginsynth/verify_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --search search_pilot_frontier_after_zero_seed0

venv/bin/python experiments/marginsynth/synthesize_frontier.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --search search_pilot_frontier_after_zero_seed0
```

The zero-disagreement stage evaluated at most 512 candidates and exactly costed
at most 32 candidates per iteration. It accepted 25 rewrites in 723.26 seconds:
16 constant-0, 8 constant-1, and 1 copy-A. Live gates decreased from 33,843 to
33,685, a 0.47% reduction, while all 6,000 calibration predictions and the
0.869500 calibration accuracy remained unchanged. Replay verification passed
all 25 rewrites and all 26 snapshot hash checks. Its rewrite-log SHA-256 is
`546e8dda19e4fa93dd19e91a40eecbba3da27019416b2f046ef85c774442b2d9`.

The chained budgeted stage used the same candidate and shortlist limits,
accepted 50 additional rewrites in 1,418.94 seconds, and stopped at its
configured rewrite limit before exhausting the 2% disagreement budget. All 50
accepted rewrites were constants: 29 constant-0 and 21 constant-1. Relative to
the exact baseline, the final circuit has:

- 33,030 live gates, a reduction of 813 gates or 2.40%;
- 67,878 connections instead of 69,412;
- unchanged editable-circuit depth of 11;
- 80/6,000 changed calibration predictions, or 1.333% disagreement;
- 3.943% worst-original-class disagreement, below the frozen 5% cap; and
- calibration accuracy 0.869000, a decrease of 0.05 percentage points.

Replay verification passed all 50 rewrites, the complete final trace, and all
51 snapshot hash checks. The budgeted rewrite-log SHA-256 is
`10e61f12a273d964772abdb36a784c2f10362b9749ad305c2de56a7bc2aaa9c9`.

Selected calibration frontier and same-flow ABC results:

| Budget | Actual disagreement | Live gates | Calibration accuracy | ABC AND nodes | ABC levels |
|---:|---:|---:|---:|---:|---:|
| 0.00% | 0.000% | 33,685 | 86.950% | 99,173 | 78 |
| 0.10% | 0.100% | 33,633 | 86.917% | 99,102 | 78 |
| 0.25% | 0.233% | 33,609 | 86.933% | 99,059 | 78 |
| 0.50% | 0.500% | 33,572 | 86.917% | 99,019 | 78 |
| 1.00% | 0.950% | 33,318 | 86.967% | 98,605 | 79 |
| 2.00% | 1.333% | 33,030 | 86.900% | 98,202 | 78 |

The exact baseline has 33,843 live gates and maps to 99,365 ABC AND nodes at
78 levels. The final bounded point therefore provides a 2.40% live-gate
reduction but only a 1.17% same-flow AIG-node reduction, with no depth
improvement. This validates correctness and scalability of the pipeline, but it
is not yet a compelling compression result: the bounded pool did not exhaust
the behavior budget, and 74 of 75 accepted rewrites were constants. A stronger
search and direct Two-Stage Unit Tying comparison are required before making a
novelty or superiority claim. No validation or test examples were used by
either rewrite stage.

The final recursive artifact audit checked 456 files against recorded byte
counts and SHA-256 hashes with zero failures. The run directory occupies
approximately 976 MiB, including checkpoints, the 185.7 MB trace database, 77
fully replayable search snapshots, and synthesis logs for all selected points.

### Two-Stage Unit Tying baseline

This is a local Fashion-MNIST adaptation of the paper's main accuracy
protocol, run from exactly the same seed-0 checkpoint as MarginSynth. It is a
development baseline, not yet a multi-seed paper result.

Frozen protocol:

- process eligible logic layers sequentially from early to late;
- exclude the first and final logic layers, leaving four 8,000-unit layers;
- use 16 fixed calibration examples for Gauss--Newton constant-direction
  screening (Stage A);
- select the target plus 40 candidates, then use 80 fixed calibration examples
  and batch-tiling factor 2 for Binary Split removal of the 40-unit overshoot
  (Stage B);
- independently evaluate 10%, 20%, 30%, 40%, and 50% tying per eligible
  layer;
- start every ratio from the same frozen checkpoint;
- perform no recovery fine-tuning;
- use validation only for final evaluation, never selection; and
- leave the official test set sealed.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/unit_tying.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --config experiments/marginsynth/configs/two_stage_unit_tying_fashion_seed0.json

DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/synthesize_unit_tying.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --examples 6000 \
  --equivalence-examples 32 \
  --pack-bits 16 \
  --compile-opt-level 0
```

The untouched checkpoint has 87.300% validation accuracy, 33,843 exact-
simplified live gates, and 99,365 ABC AND nodes at 78 levels. The no-fine-
tuning results are:

| Tied ratio per eligible layer | Tied units | Validation accuracy | Accuracy loss | Validation disagreement | Live gates | Live-gate reduction | ABC AND nodes | ABC reduction | ABC levels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 3,200 | 86.967% | 0.333 pp | 2.483% | 30,405 | 10.16% | 94,084 | 5.31% | 79 |
| 20% | 6,400 | 86.233% | 1.067 pp | 5.383% | 25,237 | 25.43% | 85,231 | 14.22% | 78 |
| 30% | 9,600 | 85.467% | 1.833 pp | 7.300% | 20,273 | 40.10% | 75,375 | 24.14% | 78 |
| 40% | 12,800 | 81.600% | 5.700 pp | 13.433% | 15,778 | 53.38% | 64,803 | 34.78% | 73 |
| 50% | 16,000 | 80.183% | 7.117 pp | 15.050% | 11,942 | 64.71% | 54,632 | 45.02% | 72 |

Interpretation:

- the method has no intrinsic fixed removal limit; the evaluated ratios are
  operating points, and the accuracy loss becomes steep after 30% here;
- 10% tying is the only point within one percentage point of the baseline;
- at that point it substantially out-compresses the current bounded
  MarginSynth pilot in both live gates and ABC nodes; their accuracy losses
  are not yet directly comparable because the current MarginSynth point has
  only been evaluated on its calibration partition;
- tied-unit count is not equivalent to synthesized cost: 10% tying of the
  eligible units yields a 10.16% live-gate reduction but only a 5.31% ABC-node
  reduction;
- all five checkpoints passed Boolean-backend, hardened-Circuit,
  exact-simplified-Circuit, packed compiled-C, hardware-normalization, Yosys,
  and ABC checks; full-validation score and prediction differences were zero
  relative to each tied model; and
- this single-seed result does not establish a paper claim. The paired five-
  seed protocol, matched recovery fine-tuning experiment, and held-out test
  evaluation remain pending.

Reproducibility artifacts are under
`baselines/two_stage_unit_tying/`. They include the resolved configuration,
hashed Stage-A/Stage-B sample indices, per-layer saliency arrays, complete
Binary Split paths, selected units and directions, tied checkpoints, full
calibration and validation metrics, exact circuits, Verilog, BLIF, Yosys/ABC
logs, aggregate tables, and recursive SHA-256 manifests. The directory occupies
approximately 187 MiB. The aggregate selection and synthesis files have SHA-256
hashes `3feb4a5a953f1ca1536197982c24a9a18da89a341a26f526d6102420fc0072b9`
and `e88b69fa5fe0c9b6205a2c8232226ed006e83f4f80403eb8ecf77468528b9abd`,
respectively.

After implementation, 3,328 repository tests passed and 3,038 optional/device-
specific cases were skipped; there were no failures.

### MarginSynth v2 convergence pilot

This development run evaluates the complete v2 search on the same frozen
Fashion-MNIST seed-0 checkpoint. It uses structural pre-ranking, exact
affected-cone simulation, cache invalidation after interacting mutations,
an operation-aware ABC-node cost model, targeted single-gate rewrites, and
atomic two-gate coordinated proposals. Accuracy is the primary quality
constraint; global and per-class disagreement are retained as additional hard
constraints and reported diagnostics. The official test split remained sealed.

The cost estimator was fit once from 12 pre-existing circuits synthesized by
the frozen Yosys/ABC flow. Its held training-set diagnostics are 217.34 ABC
nodes RMSE and 0.2003% mean absolute percentage error. At the selected v2
endpoint it predicted 93,508.37 nodes and ABC measured 93,508 nodes.

The first invocation failed closed at proposal 207 when exact revalidation of a
cached shortlist entry detected a stale per-class constraint result. No invalid
proposal was accepted. The implementation was hardened to exactly revalidate
every cached shortlisted proposal, and the deterministic search resumed from
the durable step-200 rewrite log. It then converged at step 294 because no
feasible shortlisted proposal reduced estimated cost.

Final calibration result:

- 294 atomic proposals containing 577 gate edits;
- 283 coordinated two-gate groups and 11 individual proposals;
- 33,843 to 30,456 live gates, a 10.01% reduction;
- 99,365 to 93,508 measured ABC AND nodes, a 5.89% reduction;
- unchanged 78-level ABC depth;
- calibration accuracy 86.867%, a 0.083 percentage-point loss;
- calibration disagreement 3.000%, exactly the frozen global cap; and
- maximum per-class calibration disagreement 6.989%, below the 7.5% cap.

All 294 before/after circuit hashes, the reconstructed final behavior, and both
saved Pareto snapshots replayed exactly. The rewrite-log SHA-256 is
`6d359a4f114f60cf048b3a1c12b17f4434551f5b8b3906132f343b3a3e8ab80a`;
the search-summary SHA-256 is
`5a9726fc66865e6d579421c82951a3e9bb61eecc590908033acf29b6ef5d7384`.

Held-out validation revealed an important negative result. The baseline
accuracy is 87.300%. Both step 290 and step 294 achieve 86.333%, a 0.967
percentage-point loss. Step 294 has 4.667% validation disagreement and 9.797%
maximum per-class disagreement. The calibration-selected endpoint therefore
overfits the rewrite calibration set and is not a frozen paper operating point.
It must not be presented as satisfying the intended 0.333 percentage-point
validation regime. The predeclared ablations and less aggressive validation-
selected search configuration are required before the five-seed protocol is
frozen.

The focused MarginSynth, Circuit, and experiment-protocol test suites pass:
111 tests passed in 135.15 seconds. No test examples were accessed.

### Seed-0 controlled ablations

The predeclared ablation matrix limits every variant to 75 accepted proposals
from the identical exact-simplified seed-0 circuit and packed calibration
trace. The full v2 variant completed first:

- calibration accuracy loss: 0.000 percentage points;
- calibration disagreement: 1.300%;
- validation accuracy loss: 0.150 percentage points;
- validation disagreement: 2.433%;
- maximum per-class validation disagreement: 7.154%;
- live gates: 33,843 to 32,837, a 2.97% reduction;
- ABC AND nodes: 99,365 to 97,749, a 1.63% reduction; and
- ABC levels: 78 to 77.

All 75 proposal hashes, final behavior, and the selected snapshot replayed
exactly. The 24-node estimator error at this point is 0.025%. This conservative
point generalizes much better than the 294-proposal convergence endpoint, but
does not match the 10% Two-Stage Unit Tying point's 5.31% ABC reduction. The
remaining random-candidate, gate-count-cost, individual-only, constants-only,
and disagreement-only variants were launched from their materialized,
hash-recorded configurations. Their validation results remain pending at this
log position. The held-out test split remains sealed.

All six 75-proposal ablations subsequently completed, replayed exactly, and
passed the frozen Yosys/ABC flow:

| Variant | Validation loss | Validation disagreement | ABC nodes | ABC reduction | Rewrite composition |
|---|---:|---:|---:|---:|---|
| Full v2 | 0.150 pp | 2.433% | 97,749 | 1.626% | 75 coordinated proposals, 150 constant edits |
| Random candidates | 0.167 pp | 0.683% | 98,672 | 0.697% | 7 coordinated, 80 constant, 2 alternative-gate edits |
| Gate-count cost | 0.433 pp | 2.000% | 97,764 | 1.611% | 75 coordinated proposals, 150 constant edits |
| Individual only | 0.250 pp | 1.383% | 98,834 | 0.534% | 32 constant and 43 routing edits |
| Constants only | 0.117 pp | 1.767% | 98,197 | 1.175% | 75 constant edits |
| Disagreement only | 0.150 pp | 2.433% | 97,749 | 1.626% | identical to full v2 at this horizon |

The results support three narrow conclusions. Structural candidate ranking
substantially improves synthesized reduction over random ranking. The
operation-aware objective has nearly the same node count as gate-count ranking
here but much lower validation loss. Coordinated rewrites add compression over
individual constants. Conversely, the accuracy constraint is inactive at 75
proposals, and nonconstant individual rewrites do not yet provide a compelling
cost/accuracy improvement. These null/negative results remain part of the
paper record.

The final regenerated machine-readable JSON and CSV tables have SHA-256 hashes
`c394471e0071828633447f43958e7423853e7e5d9aa51d57a7f31145a505df42`
and
`c3c973c57cdf0ea07f0cfdf999255fce89344a3a846ffa56e5a5f9166dbd16e2`,
respectively. No test examples were accessed.

### Convergence replay and rewrite-family stratification

The 294-proposal deterministic log was replayed at steps 100, 125, 150, 175,
200, 225, 250, and 275. Every before/after hash was checked while the snapshots
were regenerated. Validation loss rises to 0.383 pp at step 100, 0.533 pp at
step 125, 0.600 pp at step 150, and at least 0.75 pp for every sampled point
from step 175 onward. Calibration accuracy loss remains zero through step 275.
This documents adaptive calibration overfitting and rules out selecting a
longer point merely because it has lower estimated cost.

Inspection of the original search showed that coordinated constant pairs
dominated the structural and exact-cost shortlists. A v2.1 development variant
therefore round-robin stratified all three shortlist stages across coordinated,
individual-constant, routing, and alternative-gate proposals. Tests confirm an
equal 64/64/64/64 behavioral pool and 4/4/4/4 exact-cost shortlist when every
family is available.

At 75 proposals, v2.1 accepts 32 coordinated proposals, 36 individual constant
proposals, and 7 routing proposals. It passes replay and synthesis, but is
strictly worse than full v2:

| Method | Validation loss | Validation disagreement | Live gates | ABC nodes | ABC levels |
|---|---:|---:|---:|---:|---:|
| Full v2 | 0.150 pp | 2.433% | 32,837 | 97,749 | 77 |
| Stratified v2.1 | 0.483 pp | 2.100% | 33,021 | 97,962 | 79 |

The stratified variant is rejected for the paired protocol. This negative
experiment also shows that increasing nonconstant exposure alone is not a
methodological improvement: the routing edits hurt the measured tradeoff, and
no alternative-gate edit was selected. The paired pilot therefore freezes the
validated 75-proposal full-v2 configuration while retaining the negative
nonconstant result in the paper record.

### Frozen five-seed Fashion-MNIST study

Seeds 0--4 use the identical six-layer, 8,000-gate-per-layer rank-2 topology
(48,000 trained gates) and 108,000 GPU training steps. All five runs have the
same training-implementation SHA-256
`82b23afd16c5a86fbd38f3b8422a01d9b272ef41e91d380b60dc400311e81a38`.
Their best hard validation accuracies before circuit rewriting are 87.300%,
86.850%, 87.350%, 86.967%, and 87.417%.

The method was frozen at 75 proposals and the zero calibration-accuracy-loss
Pareto budget. Every seed passed export equivalence, exact simplification,
rewrite replay, packed compiled-C validation, and the identical Yosys/ABC
flow. The paired validation aggregate is:

| Method | Accuracy loss | Disagreement | Live-gate reduction | ABC-node reduction |
|---|---:|---:|---:|---:|
| MarginSynth | 0.390 +/- 0.201 pp | 2.460% | 2.895% | 1.587% |
| Two-Stage Unit Tying, 10% | 0.360 +/- 0.180 pp | 2.700% | 11.094% | 5.894% |

Here `+/-` is the across-seed sample standard deviation, not a confidence
interval. MarginSynth reduces validation disagreement by 0.240 percentage
points on average; the exact paired-bootstrap 95% interval is
[-0.417, -0.063] percentage points. Its accuracy-loss difference from Unit
Tying is inconclusive, while its hardware reduction is decisively worse:
4.306 percentage points less ABC-node reduction, with paired-bootstrap 95%
interval [-4.814, -3.912] percentage points. This is a negative result against
the primary baseline and must not be presented as a hardware advantage.

After these validation results were fixed, a format-v2 protocol manifest
hashed every input and authorized the first held-out test access. Its SHA-256
is `a7c170a0de42828c139d1acef32db734fd0d94b3b64000f4ac8eddafc1fb0015`.
The five-seed test aggregate is:

| Method | Test accuracy | Accuracy loss | Disagreement | Maximum per-class disagreement |
|---|---:|---:|---:|---:|
| Exact baseline | 86.204% | -- | -- | -- |
| MarginSynth | 85.952% | 0.252% | 2.484% | 6.324% |
| Two-Stage Unit Tying, 10% | 86.054% | 0.154% | 2.784% | 7.811% |

MarginSynth again has lower disagreement (paired mean -0.300 percentage
points; exact-bootstrap 95% interval [-0.604, -0.040]) and lower worst-class
disagreement on average, but removes far less synthesized hardware and has no
supported accuracy advantage. Consequently, the current method does not pass
the paper go/no-go criterion; further methodological work is required before
claiming DATE-level superiority.

The validation summary, raw frozen test result, test aggregate, and consolidated
idempotent execution manifest have SHA-256 hashes
`7dfe30d13346fbd79816e4ac1197c9680d0ec566fd8738295ccc97bd4c1fe76a`,
`8ab2131079acae657a3b5a661084f38999c90453e61452f5b80a06c9d4d9ed8e`,
`9e65a9c0c48c8709fe994c65ba558676a669574d43fd8dc651ee157180f09d28`,
and
`b1280f6fe0af5871126fbd545d2fa53b5dc78f3abfc303e9fb7d79e342c99893`,
respectively.

Final verification passed: all 115 focused MarginSynth/Circuit/protocol tests
and the complete repository suite of 3,337 tests passed; 3,038 optional or
device-specific cases were skipped, with no failures.

### Margin-aware Unit-Tying redesign: seed-0 development

The original implementation was preserved on branch `mmarginsynth` at commit
`7086fe3` before development. New outputs live under `hybrid/`; no prior result
was deleted or reused as a destination.

The first unconstrained global selector was rejected: at 3,200 ties it lost
1.583 percentage points of validation accuracy. Restricting selection to the
Gauss--Newton shortlist improved the loss to 0.467 points but remained worse
than Unit Tying. These negative diagnostics are preserved.

The final development variant starts from the 10% Unit-Tying set and evaluates
deterministic margin/class/stability/synthesis-aware swaps. It selected 3,200
ties in 3.19 seconds. Validation accuracy was 87.033% versus 86.967% for Unit
Tying, disagreement was 2.450% versus 2.483%, live gates were 30,384 versus
30,405, and ABC nodes were 94,070 versus 94,084.

An optional 16-proposal residual cleanup took 210.67 seconds, replayed exactly,
and preserved zero calibration accuracy loss relative to the hybrid. Held-out
validation accuracy became 87.017%; live gates became 30,207 and measured ABC
nodes became 93,763 at 78 levels. Thus the combined method improves over Unit
Tying by 0.050 percentage points of validation accuracy and 321 ABC nodes
(0.34%) on seed 0. This is a promising pilot but far below a publishable effect
size. Test data remained sealed throughout redesign development.

The final redesign-focused MarginSynth/Circuit/protocol suite passed all 123
tests. Python compilation and `git diff --check` also passed.

### Four-case Bayesian exploration: seed-0 development

This run implements the prespecified multi-objective exploration of guarded
two-pass MarginSynth and aggressive MarginSynth plus short recovery, each with
and without disagreement constraints. It starts from the original hardened
checkpoint, never a Unit-Tying checkpoint. The frozen source revision used by
all 160 trials is `7623e0eb7d7fe367eee077cc40205203a962c666`; the canonical
protocol SHA-256 is
`965bcab6101ea706f980a87a659046b2ac916779ddaf23fb6abc56232aee1323`.

Frozen search protocol:

- Optuna 4.6.0 constrained multi-objective TPE, sampler seed 314159;
- 40 trials per case: one enqueued reference, 11 additional startup trials,
  and 28 guided suggestions;
- minimized guard accuracy loss and an operation-aware ABC-node estimate;
- 0.333 percentage-point global accuracy-loss budget and 1.5-point
  worst-class accuracy-loss budget in every case;
- additional 3% global and 7.5% worst-class disagreement budgets only in the
  constrained cases;
- exact Yosys/ABC promotion of at most ten deterministic feasible
  proxy-Pareto/diversity candidates per case; and
- a fixed stratified 60/20/20 calibration partition. The 1,200-example guard
  is unseen by both resynthesis gradients and exact repair.

Launch commands, from `repos/torchlogix`:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/bayesian_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study all

DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/promote_bayesian_trials.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study all

DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/evaluate_calibration_guard.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0/baselines/two_stage_unit_tying/ratio_10 \
  --checkpoint tied_checkpoint.pt \
  --partition-config experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0/bayesian_search/marginsynth_bayesian_fashion_seed0_v2/guarded_constrained/trials/trial_00028/input_configs/first_resynthesis.json \
  --output bayesian_guard_reference.json --report-only

venv/bin/python experiments/marginsynth/summarize_bayesian_results.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json
```

All 160 acquisition trials completed without an execution failure. Twenty-nine
were behaviorally feasible. The frozen promotion rule selected and exactly
synthesized 27 candidates; all 27 passed export, compiled-C equivalence,
hardware normalization, Yosys, and ABC.

| Case | Feasible / 40 | Exact | Best trial | Guard accuracy loss | Disagreement | Worst-class loss | Worst-class disagreement | Live gates | ABC nodes | Levels | Method time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Guarded two-pass, constrained | 4 | 4 | 28 | -0.250 pp | 2.083% | 0.893 pp | 4.464% | 27,493 | 91,919 | 79 | 28.73 s |
| Guarded two-pass, unconstrained | 3 | 3 | 18 | -0.083 pp | 1.083% | 0.877 pp | 2.500% | 31,818 | 96,994 | 78 | 27.34 s |
| Aggressive + recovery, constrained | 11 | 10 | 16 | -0.083 pp | 1.167% | 0.833 pp | 3.150% | 33,230 | 98,273 | 78 | 46.25 s |
| Aggressive + recovery, unconstrained | 11 | 10 | 16 | -0.083 pp | 1.167% | 0.833 pp | 3.150% | 33,230 | 98,273 | 78 | 46.22 s |

Negative accuracy loss means an improvement relative to the original teacher
on the reserved guard. The aggressive constrained and unconstrained studies
are identical in suggested parameters, guard outputs, and synthesized costs:
every sampled disagreement violation also violated worst-class accuracy, so
disagreement was redundant under the accuracy guard in this pipeline. The
hardware-best aggressive point selected recovery step zero; short recovery did
not produce its best hardware point.

The seed-0 hardware winner is guarded-constrained trial 28. Relative to the
10% Two-Stage Unit Tying point, it has 2,165 fewer ABC nodes (2.301%), 2,912
fewer live gates (9.577%), and the same 79-level depth. Relative to the original
exact circuit, it reduces ABC nodes by 14.390% and live gates by 20.861%.

The Unit-Tying checkpoint was evaluated after search, in report-only mode, on
the exact same 1,200-example guard. It loses 0.083 percentage points of global
accuracy, has 2.000% disagreement, loses 2.632 points in its worst class, and
has 5.042% worst-class disagreement. Trial 28 improves global guard accuracy
by 0.250 points, has 2.083% disagreement, loses 0.893 points in its worst
class, and has 4.464% worst-class disagreement. Thus trial 28 is one sample
worse in global disagreement but materially better in worst-class accuracy
and worst-class disagreement. Unit Tying violates the frozen 1.5-point
worst-class accuracy guard; trial 28 passes every guard.

Trial 28 uses the gate-count training proxy, 400 first-pass updates, and 600
second-pass updates. The two GPU optimizations take 9.03 and 10.12 seconds;
complete first/second resynthesis takes 10.01 and 11.21 seconds. Peak allocated
GPU memory is 234,798,080 bytes. Exact repair retains 4,211 first-pass and
2,017 second-pass LUT changes. Across the 6,228 retained changes, 1,626 are
constants and 4,602 are routing/inversion functions; no alternative binary
function survives repair. Retained changes by internal layer are 1,764,
1,652, 1,519, and 1,293. These outcome counts are an important action-space
ablation: the method offered all 16 rank-2 Boolean functions, but this winning
point obtained its gain from constants and routing.

Runtime is a current weakness. The frozen trial takes 28.73 seconds versus
2.016 seconds for the existing Unit-Tying 10% implementation, approximately
14.3 times slower before exact synthesis. The serial four-case acquisition
takes 6,511.53 seconds (1.809 hours); summed completed-trial time is 6,474.42
seconds. Exact promotion adds 5,259.92 seconds (1.461 hours), dominated by the
CPU export/simplification/Yosys flow. Exploration, frozen-method application,
and common exact synthesis must remain separate columns in paper tables.

Development failures were preserved instead of removed. The first v1 smoke
launch recorded four CUDA-not-visible failures inside the restricted sandbox
before successful GPU reruns. The first v2 smoke recorded one cost-proxy shape
failure per case; the proxy was corrected to take `[1, 28, 84]` from the
source export-verification artifact and each case then completed. The v1 exact
integration also remains archived: its guarded point was infeasible at 95,371
nodes and its aggressive point was feasible at 98,304 nodes. None of these
diagnostic runs enters the v2 Pareto tables.

The run used PyTorch 2.9.0+cu130, CUDA 13.0, an NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition, Yosys 0.9, and Berkeley ABC 1.01. The original
teacher checkpoint SHA-256 is
`edcc3334a9f2c1a34c8875767f08c8094d29aabd8c15ff9c1c1c8decc12a91f6`;
trial 28's promoted checkpoint SHA-256 is
`ef269006bda3f0490ce087760013ff51756bc64688cdf824b4a889c58e490b3f`.

Machine-readable meta-analysis artifacts include the Optuna SQLite database,
append-only lifecycle events, all failed/infeasible trials, resolved configs,
checkpoint and split hashes, optimization/recovery traces, per-class/fold
metrics, subprocess commands/resources, exact circuits, synthesis logs,
portable long-form CSV tables, per-case exact Pareto tables, and the generated
cross-case JSON/CSV summary. The cross-case JSON/CSV SHA-256 hashes are
`faa35f2326c542da96b730516a6e1ad2970558447185c62e5c7879f1f6cdaec2`
and
`937c7712688e572ed462dcc162ea55b6c5a7b8556e062060d29cd533a2b6dc72`.
The report-only Unit-Tying same-guard result has SHA-256
`a87bfa569e054063dc489930be4d32aeb9a5c3042806dab2ca06a73e51afbbd3`.
The root exact-promotion summary SHA-256 is
`1b761fe2923eb25d423cf05f86a35543765da69212f02d5e91e51d79a7107fb0`;
the frozen protocol file SHA-256 is
`dd520ad4d42e4f87480b6250f9ab43bee45f26ae3d3c0f49ca8c4c078e54813d`.

Post-run verification used:

```bash
venv/bin/python -m py_compile \
  experiments/marginsynth/evaluate_calibration_guard.py \
  experiments/marginsynth/summarize_bayesian_results.py \
  tests/test_bayesian_summary.py
PYTHONPATH=. venv/bin/pytest -q \
  tests/test_marginsynth.py tests/test_bayesian_summary.py
PYTHONPATH=. venv/bin/pytest -q
```

Compilation passed, all 67 focused tests passed, and the complete repository
suite reported 3,364 passed, 3,038 skipped, and one pre-existing tensor-copy
warning in 146.30 seconds. An initial test collection without `PYTHONPATH=.`
failed because this repository exposes `experiments` as a namespace from its
root; the logged commands make that environment requirement explicit. The new
checks cover exact-cost winner selection and baseline deltas, acquisition-only
event timing, and the distinction between a selection measurement and a
post-search report-only audit. `venv/bin/black` was not available, so no
formatter was installed or applied after the frozen runs; compilation and
`git diff --check` were used as the source-format sanity checks.

This is a strong seed-0 development result, not yet a paper claim. Validation
and test were not loaded anywhere in v2 acquisition or promotion. The next
protocol step is to repeat the guarded-constrained configuration on seeds 1
and 2, freeze one transferable configuration without inspecting validation,
and only then perform the predeclared multi-seed validation/test evaluation.

## 2026-08-11: component isolation and dense-CIFAR preparation

No new accuracy or hardware result is claimed in this entry. A source audit
found that the archived 48K/512K CIFAR standard-random result directories keep
configs and metrics but not their git-ignored checkpoints. Their training
configs also set no calibration holdout. Reusing trained examples for
post-training selection would violate the paper data boundary, so the new
dense-source protocol regenerates standard-random checkpoints with split seed
2027 and a 10% calibration partition excluded from source training. Coverage
connectivity remains excluded from the primary study.

The implemented component mechanisms are:

- safe topological reachability masks that follow both fixed inputs;
- descriptive current-LUT algebraic liveness;
- constants-plus-routing/inversion action masks;
- source-to-candidate activity mismatch measured on the calibration
  optimization subset, with worst global/class/fold risk used only to order
  exact repair; and
- a post-training Silicon-Aware control using the paper's 16-entry SkyWater
  cell-area table and `CE + 0.01 * expected area`.

Tracked, resumable runners now freeze configs and record commands, source and
checkpoint hashes, partition hashes, optimization/repair traces, wall time,
memory, export equivalence, compiled C, Yosys, and ABC artifacts. Dense seed
transfer is predeclared to require guard feasibility plus at least 10% measured
two-pass method-time acceleration from liveness; optimizable-gate reduction by
itself is not accepted as acceleration. Test data remains sealed.

Development verification at this point reports 69/69 focused MarginSynth tests
and the complete 3,369-test repository suite passing, with 3,038 skips and one
pre-existing tensor-copy warning. Python compilation and `git diff --check`
also pass. The venv
contains PyTorch 2.9.0+cu130; escalated hardware preflight sees two NVIDIA RTX
PRO 6000 Blackwell GPUs, with GPU 0 initially providing about 97 GB free.
Yosys 0.9 and Berkeley ABC are installed. No component or CIFAR experiment was
started before committing the implementation revision.

### Fashion-MNIST fixed trial-28 component outcomes

Commit `467282f` was used for every run. All four component/control checkpoints
passed model-to-circuit, exact simplification, compiled-C, hardware
normalization, Yosys, and ABC verification on all 6,000 calibration examples.
The source checkpoint hash remained
`edcc3334a9f2c1a34c8875767f08c8094d29aabd8c15ff9c1c1c8decc12a91f6`.

| MarginSynth variant | Guard feasible | Guard accuracy loss | Disagreement | Worst-class loss | Live gates | ABC nodes | Levels | Two-pass method time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current trial 28 | yes | -0.250 pp | 2.083% | 0.893 pp | 27,493 | 91,919 | 79 | 21.227 s |
| Safe topological liveness | yes | -0.250 pp | 2.083% | 0.893 pp | 27,493 | 91,919 | 79 | 21.493 s |
| Constants + routing/inversion only | no | +0.167 pp | 1.583% | 1.786 pp | 27,577 | 92,181 | 78 | 22.466 s |
| Class/fold activity repair ordering | no | +0.667 pp | 3.333% | 3.125 pp | 25,320 | 89,817 | 78 | 22.984 s |

The safe structural analysis found all 32,000 eligible internal gates
topologically reachable. It found 1,465 algebraically dead gates under the
checkpoint's current LUT functions, but these were not masked because changing
a LUT can reactivate such paths. Consequently the liveness result is exactly
identical to current trial 28 and is 1.25% slower in total two-pass method time
(2.31% slower for optimization alone). It provides no Fashion-MNIST runtime
claim.

The class-aware ordering exposes a useful but currently unsafe direction: it
reduces ABC cost by 2,102 nodes (2.287%) relative to current MarginSynth, but
fails all global accuracy, disagreement, and worst-class guard limits. The
restricted action space also fails the worst-class accuracy guard and has 262
more ABC nodes. Under the predeclared feasible-first exact-ABC selection rule,
the scaling candidate is therefore **current trial-28 MarginSynth**. The dense
small smoke will additionally retain a liveness arm to determine whether the
CIFAR topology contains safe dead gates.

The Silicon-Aware-style post-training control finished 1,000 updates in 13.46
seconds and synthesized to 29,025 live gates, 91,736 ABC nodes, and 80 levels.
It is infeasible: its unseen guard loses 1.333 percentage points globally,
11.607 points in the worst class, and has 10.667% disagreement. Its apparent
6.900-point accuracy improvement on the full calibration set is therefore
optimization-subset overfitting, not a valid gain.

One-time characterization took 2.60 seconds. The four export/synthesis stages
took 171.14, 171.80, 163.61, and 176.14 seconds, confirming that common exact
hardware verification dominates component wall time. The machine-readable
comparison JSON and CSV hashes are
`15d7341861dbbb59c578ea4139dea8830bbd9f1505b31efb382a2aea7033f459`
and `8ec15cc8d14cf0c79105a591efe71c35e17d16c9f9873b2b7df004cdca0b8151`.
Validation and test were not loaded by component selection.

## 2026-08-12: dense CIFAR-10 scaling and frozen transfer

This entry answers whether a much denser CIFAR-10 DLGN exposes useful
MarginSynth headroom after the Fashion-MNIST pilot. It does, but the outcome is
a fidelity/hardware trade-off rather than a complete win over Unit Tying.
MarginSynth preserves the source decisions substantially better, while 10%
Unit Tying remains smaller and about 17 times faster. The frozen MarginSynth
second pass also fails the worst-class guard on seed 2. These qualifications
are part of the result and were not filtered after inspection.

### Frozen data, model, and training protocol

The primary model is `DlgnCifar10Medium`: four rank-2 logic layers of 128,000
gates each, or 512,000 trained gates in total. The first and last layers remain
fixed during MarginSynth, leaving 256,000 eligible gates. Connectivity is the
repository's fixed standard-random topology, independently generated with
topology/training seeds 0, 1, and 2. Coverage connectivity is not used.

All sources use raw LUT parameters, no data augmentation, batch size 100,
learning rate 0.01, 108,000 GPU iterations, and validation every 2,000
iterations. Split seed 2027 gives 40,000 training, 5,000 validation, 5,000
calibration, and 10,000 official-test examples. Source checkpoint selection
uses validation only; post-training resynthesis uses calibration only. The
test set stayed sealed throughout this study. The identical split artifact in
all three runs has SHA-256
`35c2858e05b27e9d6ff6545bedb23fee84e3bcf4215c485190fa5cba4314bf1c`.
Its train/validation/calibration/test index hashes are, respectively,
`f3eac954878110839561746cd18b55abafe36e33b5b19a2d01fb4c7cbf76a213`,
`96eec96edd6886d0ea983446d0125e76caef15044fddfb0672e101f88c0dcee5`,
`56f3aab0c3081505753c8a1b3c9d1d74128275e942ca59302ffe999f359f9b16`,
and `9e1c19b3fdc185411bd1a987deb6a9fda2531116aac060a84c4d878854d1c099`.

| Seed | Best hard validation accuracy | Training time | Peak GPU allocation | Source checkpoint SHA-256 |
|---:|---:|---:|---:|---|
| 0 | 54.64% | 4,444.77 s | 1,206,192,640 B | `74a1f154ab075f083286b44b9ab201a85668173c8e5a9b8fe52e36cfe783b39e` |
| 1 | 55.16% | 4,429.25 s | 1,206,192,640 B | `54381f37a8ceaab5dd47083be9f760d254be5cbb19444a411551ea29fac59b59` |
| 2 | 53.98% | 4,250.22 s | 1,206,192,640 B | `8d6f0541636663c10bf8ccf979227eb84618a3366dc09133fedf16b60bf0bdc7` |

Training and MarginSynth optimization ran on GPU, using PyTorch 2.9.0+cu130,
CUDA 13.0, and an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
Exact Boolean simulation, circuit export, compiled-C verification, Yosys, and
ABC are CPU/compiler stages by design. Yosys 0.9 and Berkeley ABC 1.01 were
used. The machine was shared and other GPU jobs were visible during part of
the transfer runs, so wall-time comparisons should be repeated on a dedicated
machine before publication.

### Small-model go/no-go smoke

The first scaling gate used `DlgnCifar10Small`, four layers of 12,000 gates
(48,000 total), seed 0, and the same split/training policy. The source reached
49.34% hard validation accuracy in 996.36 seconds. Exact simplification had
36,200 live gates, 131,461 ABC AND nodes, and 80 levels. Current MarginSynth
was guard-feasible, reached 49.36% validation accuracy with 1.80%
disagreement, and produced 35,025 live gates, 130,191 nodes, and 79 levels.
Its 1,009 retained actions were 130 constants and 879 routing/inversion
functions; no alternative binary gate survived repair.

The liveness arm produced the identical checkpoint and circuit because the
safe topological mask removed zero eligible gates. It took 33.56 seconds
versus 30.74 seconds for current MarginSynth, so the 48K smoke provided no
liveness acceleration claim. It did confirm an end-to-end feasible CIFAR-10
path and authorized the 512K run.

### Seed-0 component comparison on the 512K model

The predeclared Fashion-selected current trial-28 configuration was compared
with exact simplification, 10% Unit Tying, safe liveness, class-aware activity
ordering, and the Silicon-Aware-style post-training control. All hardware
counts below follow identical exact simplification, compiled-C verification,
Yosys, and ABC commands.

| Method | Guard feasible | Validation accuracy | Disagreement | Live gates | ABC nodes | Levels | Method time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact simplification | yes | 54.64% | 0.00% | 389,970 | 1,484,154 | 105 | 0.00 s |
| 10% Unit Tying | not a constrained method | 54.16% | 5.60% | 353,549 | 1,419,811 | 106 | 10.86 s |
| Current MarginSynth | yes | 54.40% | 2.28% | 380,330 | 1,472,970 | 105 | 210.41 s |
| Safe liveness | yes | 54.40% | 2.28% | 380,330 | 1,472,970 | 105 | 186.30 s |
| Class-aware activity | yes | 54.60% | 2.14% | 369,523 | 1,460,978 | 106 | 230.00 s |
| Silicon-Aware control | no | 47.78% | 35.20% | 370,634 | 1,443,196 | 105 | 117.51 s |

The liveness arm appears 11.46% faster than current MarginSynth in this single
run, crossing the mechanical transfer threshold. However, it removed no
eligible gates and produced identical normalized behavior and hardware cost.
It was slower on Fashion-MNIST and on the 48K CIFAR smoke. The timing delta is
therefore treated as noise, not as a causal liveness speedup. The class-aware
variant is seed 0's strongest feasible size/fidelity point, but it was not
transferred because the protocol had already selected current trial 28 from
Fashion-MNIST. The Silicon-Aware control again overfits calibration and is not
feasible.

### Frozen current-versus-Unit-Tying transfer

Seeds 1 and 2 reuse the exact trial-28 hyperparameters; there is no new search
or per-seed tuning. The table reports validation only after the calibration
procedure was frozen.

| Seed | Method | Source loss | Disagreement | Live reduction | ABC reduction | ABC nodes | Method time | Guard feasible |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10% Unit Tying | 0.48 pp | 5.60% | 9.339% | 4.335% | 1,419,811 | 10.86 s | n/a |
| 0 | MarginSynth | 0.24 pp | 2.28% | 2.472% | 0.754% | 1,472,970 | 210.41 s | yes |
| 1 | 10% Unit Tying | 0.44 pp | 5.68% | 9.596% | 4.443% | 1,422,831 | 10.88 s | n/a |
| 1 | MarginSynth | 0.48 pp | 2.76% | 5.290% | 1.647% | 1,464,460 | 192.28 s | yes |
| 2 | 10% Unit Tying | 0.78 pp | 5.88% | 9.590% | 4.430% | 1,425,407 | 13.52 s | n/a |
| 2 | MarginSynth | 0.24 pp | 1.24% | 3.055% | 0.928% | 1,477,627 | 194.53 s | **no** |

MarginSynth's seed-2 second pass is calibration-feasible and has no global
guard accuracy loss, but its worst-class guard loss is 1.961 percentage
points, above the frozen 1.5-point limit. The first-pass snapshot on that seed
was guard-feasible after 7,809 retained actions, with a 0.200-point global
accuracy gain, 0.400% disagreement, and no worst-class accuracy loss. It was
not synthesized or substituted after seeing the failure because the frozen
protocol declared the second-pass output as the result. A future protocol may
predeclare fallback to the most aggressive guard-feasible snapshot.

Across three seeds, exact circuits average 391,623 live gates, 1,488,203 ABC
nodes, and 107 levels. Unit Tying averages 0.567 percentage points of source
accuracy loss, 5.720% disagreement, 9.508% live-gate reduction, and 4.403% ABC
reduction. Current MarginSynth averages 0.320 points of source loss, 2.093%
disagreement, 3.606% live-gate reduction, and 1.110% ABC reduction. Its mean
method time is 199.07 seconds versus 11.75 seconds for Unit Tying.

Paired by seed, MarginSynth improves validation accuracy by 0.247 percentage
points on average and reduces disagreement by 3.627 points relative to Unit
Tying. The price is 49,003 more ABC nodes on average, 3.293 points less ABC
reduction, 5.903 points less live-gate reduction, and a 17.15-times larger
method runtime. Thus dense CIFAR-10 is more informative than Fashion-MNIST and
supports MarginSynth's fidelity motivation, but the current method is not yet
better overall. It needs a predeclared feasible-snapshot fallback and more
hardware-aware candidate ranking before the paper can claim a Pareto advantage.

Across all three selected MarginSynth results, retained actions are exclusively
constants and routing/inversion; the accepted alternative-binary count is
zero. This independently reinforces the Fashion action-space ablation and
motivates restricting or strongly regularizing the binary action family.

### Reproduction and artifact audit

From `repos/torchlogix`, the source, frozen component/transfer, and aggregation
entry points are:

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/train_dense_sources.py \
  --protocol experiments/marginsynth/configs/dense_cifar_standard_random_sources.json \
  --sources small_seed0 medium_seed0 medium_seed1 medium_seed2 --resume

CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/run_component_protocol.py \
  --protocol <source-run>/marginsynth_comparison_protocol.json --resume

CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/run_component_protocol.py \
  --protocol <source-run>/marginsynth_transfer_protocol.json --resume

PYTHONPATH=. venv/bin/python experiments/marginsynth/summarize_dense_transfer.py \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed0/marginsynth_comparison/trial28_transfer_v1/dense_comparison.json \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed1/marginsynth_transfer/trial28_transfer_v1/dense_comparison.json \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed2/marginsynth_transfer/trial28_transfer_v1/dense_comparison.json \
  --output-dir experiments/marginsynth/results/dense_cifar_sources/transfer_summary/trial28_transfer_v1
```

The seed-0/1/2 comparison-input hashes are
`7c33b29073e4ca68eb7c3490e26864e2de4a1bddc5cac8a953f35833d7126feb`,
`4eb7a40ffedbd8a4c614b8e7f36b0945924468c0e8401cc0e6b10cd777072d1d`,
and `f67136236012841a664e5cb4f432c771cc663a96e7213c281a33eece22236758`.
The aggregate JSON and CSV hashes are
`79ff3b98518c56d2e66a5f1a67d7b4dee30a6f41af7480872b163ee0d75efccd`
and `f46bedee70f390cdc443a13d72f7e60db4f69529fc238c81437195a23613fbbd`.
Every exact, Unit-Tying, and MarginSynth output passed model/circuit export,
compiled-C calibration equivalence, Yosys, and ABC. All execution-stage
records report completion. No test metrics were read or generated.

Final verification used Python compilation, the focused dense-transfer tests,
`git diff --check`, explicit completion/test-sealing assertions over all three
comparison artifacts, and `PYTHONPATH=. venv/bin/pytest -q`. The focused tests
reported 2 passed. The complete repository suite reported 3,376 passed, 3,038
skipped, and one pre-existing tensor-copy warning in 120.45 seconds.

## 2026-08-12: operation-aware ranking and feasible-snapshot fallback

This run implemented the next hardware-aware MarginSynth iteration without
altering or deleting any earlier result. Raw artifacts live under the
versioned `marginsynth_hardware_ablation/structural_rank_v2` root. The official
CIFAR-10 test set remained sealed.

### Method and calibration

The implementation adds four static rewrite features: local two-input AIG
operation reduction, exact-Boolean-cofactor constant-propagation gain, direct
fan-out, and fixed-topology downstream influence. Nonnegative ridge
coefficients were fitted only to earlier seed-0 development circuits. Unit
Tying was removed from the fit and retained solely as an estimator check. No
images, validation predictions, or test predictions were loaded during
calibration.

The frozen model SHA-256 is
`2a1e154bcd4d67ae90fd02289ba04cfb167a9b3d33c3c71ea9bebe7176aa2046`.
Its coefficients for operation gain, constant propagation, log fan-out, and
log downstream influence are 0.596458, 1.184420, 0.027643, and effectively
zero. In-sample Pearson/Spearman were 0.988/1.000; leave-one-fit-method-out
Pearson/Spearman were 0.914/0.800. The held-out Unit-Tying reduction was
predicted as 62,525 nodes versus 64,343 observed, an absolute error of 1,818
nodes (2.82%).

All 16 LUTs remained legal. Alternative binary LUTs alone received a fixed
two-AIG-unit prior penalty; the original LUT was exempt. Source, pass-one, and
pass-two checkpoints were selected by repair, complete-calibration, and
untouched-guard feasibility followed by the frozen structural estimate. The
source's byte-identical exact export was reused only when the source fallback
won and after checkpoint, split, tool-flow, and test-sealing checks.

An initial version-1 diagnostic had included Unit Tying as a coefficient-fit
point. This violated its intended held-out role. The run was interrupted before
completion, its diagnostic artifacts were preserved under `structural_rank_v1`,
and no value from it was used for the final result. Version 2 corrected the
calibration boundary before the reported ablation.

### Seed-0 ablation

The source is `DlgnCifar10Medium`, with four 128,000-gate layers (512,000
trained gates), of which the two middle layers (256,000 gates) are eligible.
The source checkpoint and split are identical to the earlier dense-CIFAR study.
All optimization ran with the repository virtual environment on GPU 0;
export, exact Boolean simplification, compiled-C verification, Yosys, and ABC
were CPU/compiler stages.

| Method | Snapshot | Guard feasible | Retained edits | Validation accuracy | Disagreement | Worst-class validation loss | Live gates | ABC nodes | Levels | Yosys cells | Area proxy (um2) | Method time | Optimization time | Peak GPU allocation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact simplification | source | yes | 0 | 54.64% | 0.00% | 0.000 pp | 389,970 | 1,484,154 | 105 | 1,180,212 | 4,354,161 | 0.00 s | 0.00 s | 0 B |
| 10% Unit Tying | tied | n/a | 20,797 | 54.16% | 5.60% | 3.353 pp | 353,549 | 1,419,811 | 106 | 1,115,763 | 3,953,329 | 10.86 s | 10.86 s | n/a |
| Prior current MarginSynth | second | yes | 13,034 | 54.40% | 2.28% | not recorded in old summary | 380,330 | 1,472,970 | 105 | 1,166,026 | 4,253,383 | 210.41 s | 161.25 s | 2,044,764,672 B |
| Focused control | source fallback | yes | 0 selected | 54.64% | 0.00% | n/a | 389,970 | 1,484,154 | 105 | 1,180,212 | 4,354,161 | 177.34 s | 126.24 s | 2,061,640,192 B |
| Hardware-aware | second | yes | 314 | 54.52% | 2.30% | 1.386 pp | 388,326 | 1,480,257 | 104 | 1,177,324 | 4,333,693 | 176.51 s | 123.11 s | 2,061,640,192 B |
| Class + hardware-aware | second | yes | 2,075 | 54.68% | 1.14% | 1.183 pp | 384,998 | 1,475,742 | 105 | 1,172,283 | 4,297,589 | 232.48 s | 175.95 s | 2,061,640,192 B |

The focused control learned 13,401 and then 6,674 retained edits, but both
candidate checkpoints failed the untouched guard; the source fallback worked
as designed. Hardware-aware retained 314 constants and no routing or binary
edits. Class + hardware-aware retained 1,643 constants and 432 routing or
inversion edits, with no alternative binary LUTs. Its validation accuracy was
0.04 percentage points above the source while disagreement stayed at 1.14%,
but its hardware reduction was smaller than the prior current method.

Across source plus the three new predeclared observations, predicted versus
observed ABC reductions were `[0, 9777.6, 0, 3432.5]` and
`[0, 8412, 0, 3897]`. Pearson was 0.993 and Spearman was 1.000, above the
frozen 0.5 transfer threshold. The remaining bottleneck is therefore not the
relative ranking signal. It is the optimizer/repair interaction: hardware-only
ranking is too conservative, while the 75/25 class/hardware combination does
not retain enough high-impact guard-feasible edits.

### Frozen no-go decision and artifacts

The predeclared seed-0 selector minimizes exact ABC nodes among guard-feasible
MarginSynth variants. It selected the prior current reference at 1,472,970
nodes; class + hardware-aware had 1,475,742 and hardware-aware had 1,480,257.
The freeze record is consequently `not-frozen`. An explicit seed-1 transfer
preparation attempt failed as intended with `transfer freeze record is not
frozen`. Seeds 1 and 2 were not run for this variant, and the official test set
was not opened.

The v2 ablation protocol SHA-256 is
`adcf0ca71203b87480c9490dc7c6e38e08b0f4f69a42f922e145d88f86a7a366`.
The machine-readable ablation JSON/CSV SHA-256 values are
`7dd1134e6010ea6b8eb45730f69f4adb4c26185e339344265cd4c98d1c169436`
and `ef291b00cfe46d3a2fd2a00c4cb8cf52373485152bfe9ed8aed8d41cde5780d1`.
The no-go freeze-record SHA-256 is
`0b8e91bb88c7791249db790a74360146177baf9de0ab01ed6798a76d54b1bc40`.
Every executed stage returned zero; every exact export passed compiled-C,
Yosys, and ABC verification; every selection and synthesis record states
`test_used: false`.

Final verification comprised `git diff --check`, an explicit 12-stage
completion/test-sealing assertion, the expected rejection of an attempted
seed-1 protocol generated from the `not-frozen` record, 82 focused MarginSynth
tests, and the complete repository suite. The complete suite reported 3,384
passed, 3,038 skipped, and the same pre-existing tensor-copy warning in 142.89
seconds. No seed-1/2 hardware-aware protocol, checkpoint, or test result was
created.
