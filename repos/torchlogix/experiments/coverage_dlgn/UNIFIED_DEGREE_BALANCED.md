# Unified Semantic Degree-Balanced Candidate

## Scope and frozen-method policy

This document defines the post-V3/V4 candidate without changing either frozen
method:

- dense V3 remains `semantic_balanced_hybrid`;
- convolutional V4 remains `semantic_channel_hybrid`;
- the failed channel-spatial adapter remains
  `semantic_channel_spatial_hybrid`;
- the new candidate is `semantic_degree_balanced`, also called **U1** below.

U1 introduces no trainable parameters, no routing optimization during training,
and no circuit gates. Its fixed indices remain exportable through the existing
TorchLogix circuit path.

## Why the V4 no-swap ablation won

`analyze_cifar10_conv_no_swap.py` reloads the exact historical best checkpoints
for CIFAR-10 LogicTreeNet-S, reconstructs channel pairs from their stored
indices, and compares V4 against its no-swap base. It does not rewrite the
historical result directories.

The base is an **affine-ordered balanced butterfly**, not a round-robin
schedule. This corrects the earlier informal “round-robin” description.

Three-seed mean topology diagnostics are:

| Layer | Channels -> kernels | V4 changed output pairs | Pair multiset Jaccard | Duplicate pairs, no-swap -> V4 | Mean pair span, no-swap -> V4 | Mean predecessor Jaccard, no-swap -> V4 | Mean raw ancestry, no-swap -> V4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 -> 32 | 8.33% | 0.869 | 5.00 -> 4.67 | 2.854 -> 2.938 | 0.000 -> 0.000 | 2.000 -> 2.000 |
| 1 | 32 -> 128 | 20.31% | 0.666 | 48.00 -> 31.67 | 5.625 -> 6.984 | 0.125 -> 0.125 | 3.625 -> 3.625 |
| 2 | 128 -> 512 | 10.68% | 0.807 | 64.00 -> 51.33 | 16.375 -> 19.591 | 0.264 -> 0.266 | 5.828 -> 5.825 |
| 3 | 512 -> 1,024 | 2.08% | 0.959 | 0.00 -> 0.00 | 30.000 -> 32.770 | 0.485 -> 0.486 | 7.894 -> 7.885 |

For every layer and seed:

- the complete predecessor degree vector is identical;
- spatial coordinates are identical;
- gate count and trainable parameter count are identical.

V4 swaps therefore do what their implementation promises: they preserve exact
fan-out while reducing duplicate pairs and increasing pair span. However, the
largest mean change in raw predecessor Jaccard is only 0.00246, and raw
ancestry size is effectively unchanged. The swaps disrupt part of the useful
butterfly pair set without producing the intended ancestry benefit.

The three-seed mean hardened validation learning curves also show a late
separation:

| Step | Random | Frozen V4 | No-swap | No-swap - random | No-swap - V4 |
|---:|---:|---:|---:|---:|---:|
| 2K | 47.553% | 49.293% | 48.427% | +0.873 pp | -0.867 pp |
| 8K | 54.720% | 55.987% | 55.487% | +0.767 pp | -0.500 pp |
| 12K | 56.000% | 56.213% | 56.627% | +0.627 pp | +0.413 pp |
| 16K | 56.027% | 56.387% | 56.867% | +0.840 pp | +0.480 pp |
| 18K | 56.053% | 56.833% | 57.887% | +1.833 pp | +1.053 pp |
| 20K | 56.673% | 56.587% | 57.760% | +1.087 pp | +1.173 pp |

The no-swap curve is not uniformly better early, but it overtakes V4 at 12K
and finishes higher. This argues against paying V4's topology-construction
cost for ancestry swaps that do not improve the measured ancestry.

Machine-readable evidence is stored in:

- `summary/cifar10_conv_small_no_swap_diagnostics.json`;
- `summary/cifar10_conv_small_no_swap_diagnostics.csv`.

## U1 method definition

U1 applies one rule to both dense and convolutional DLGNs:

1. Preserve the input representation's semantic order. Dense image inputs use
   explicit channel, coordinate, and threshold metadata. Convolutional inputs
   retain the architecture's RGB/thermometer channel ordering.
2. Construct an affine-ordered butterfly pair schedule. It is deterministic for
   an architecture and topology seed and balances predecessor use over complete
   stages.
3. Preserve the schedule's complete predecessor degree vector exactly. U1 has
   no ancestry-swap or task-adaptive refinement phase.
4. For convolution, use the selected two-channel group with the unchanged
   random receptive-field sampler. Do not force the two leaves of each
   bottom-level logic gate to use different channels.
5. Store only ordinary fixed connection indices. Training effort, gate budget,
   parameter count, and inference operators are unchanged.

For non-power-of-two widths, a truncated cyclic butterfly stage is
near-balanced rather than perfectly uniform; its observed degree range can be
two. The exact base degree vector is nevertheless deterministic. Paper
convolution layers with power-of-two channel widths are uniform, while the
9-to-32 input layer has the minimum integer imbalance (fan-out 7 or 8).

The `coverage_swap_fraction`, candidate-pool, and novelty controls do not alter
U1. They remain accepted in legacy experiment configurations, but the
`semantic_degree_balanced` implementation has no swap call. This prevents U1
from becoming a hidden per-dataset hyperparameter sweep.

## Evidence already available

The U1 behavior has already been trained under legacy ablation names:

| Domain | Architecture | Random | U1-equivalent no-swap | Paired mean gain |
|---|---|---:|---:|---:|
| Dense CIFAR-10 | `DlgnCifar10Medium`, 512K gates | 54.820% | 59.253% | +4.433 pp |
| Convolutional CIFAR-10 | `ClgnCifar10PaperSmall` | 56.673% | 58.013% | +1.340 pp |

The dense value is best hardened validation accuracy from three paired seeds.
The convolutional value is also three-seed best hardened validation accuracy.
Neither row uses the held-out test set for selection.

Regression tests prove that U1's convolutional full model state is bitwise
identical to the historical V4/no-swap arm for a matched seed. This permits
seeds 0-2 to be reused rather than rerun.

## Predeclared promotion experiment

Only seeds 3 and 4 are new. For both seeds, random, frozen V4, and U1 use the
same dataset split, initialization seed, topology seed, architecture,
augmentation, batch size, and 20K-step schedule. Historical seeds 0-2 are
reused without retraining.

U1 advances to convolutional CIFAR-10 M only if both conditions hold across
five paired seeds:

- mean U1-minus-random gain is at least +1.0 percentage point;
- U1 beats random on at least four of five seeds.

Failure stops the convolutional M escalation. Success triggers a matched M
validation; CIFAR-100 transfer, full schedules, hardened held-out test
accuracy, circuit cost, runtime, and memory are deferred until M also succeeds.

The frozen protocol is
`protocols/cifar10_conv_small_unified_five_seed.json`.

## Five-seed outcome

All six missing seed-3/4 runs completed in 51.5 minutes on two GPUs with zero
failures. The held-out test set was not accessed.

| Seed | Random | Frozen V4 | U1 | U1 - random |
|---:|---:|---:|---:|---:|
| 0 | 56.10% | 57.60% | 57.92% | +1.82 pp |
| 1 | 57.40% | 57.80% | 57.66% | +0.26 pp |
| 2 | 56.52% | 56.16% | 58.46% | +1.94 pp |
| 3 | 56.88% | 57.12% | 57.56% | +0.68 pp |
| 4 | 57.42% | 58.56% | 56.52% | -0.90 pp |
| **Mean** | **56.864%** | **57.448%** | **57.624%** | **+0.760 pp** |

The paired U1-minus-random 95% Student-t interval is
[-0.700, +2.220] pp. U1 wins four of five seeds, so the consistency gate
passes, but +0.760 pp is below the predeclared +1.0 pp mean-gain threshold.
U1 is therefore **not promoted** to convolutional CIFAR-10 M.

The newly trained seeds also verify equal resource budgets:

| Method | Mean topology construction | Mean training wall time | Mean peak GPU allocation |
|---|---:|---:|---:|
| Random | 0.176 s | 1,021.68 s | 1,874.59 MiB |
| Frozen V4 | 0.414 s | 1,022.16 s | 1,874.62 MiB |
| U1 | 0.165 s | 1,021.51 s | 1,874.62 MiB |

All methods have 1,336,832 trainable parameters, zero training routing
parameters, 1,945,600 deployed routing bits, and identical recorded gate/cost
fields. U1 removes V4's swap-search overhead, although that overhead is small
at S.

Per the locked plan, no convolutional M, CIFAR-100 transfer, full-schedule,
held-out test, or extended circuit/runtime/memory experiment was launched.
The machine-readable decision is
`summary/cifar10_conv_small_unified_five_seed.json`.

## Failure history retained

- Frozen V4 improved pair diversity/span but not the ancestry diagnostics it
  was meant to improve.
- Forced channel-spatial leaf pairing reduced the three-seed mean gain to
  +0.360 pp over random and was not promoted.
- Explicit classifier overrides changed the historical RNG protocol; those
  invalid attempts remain under `results/failed/` and are excluded.
- U1 does not erase any of these methods or results; it is a separately named
  candidate motivated by their diagnostics.
- U1's five-seed signal remained positive (+0.760 pp, four wins) but failed
  the +1.0 pp promotion threshold. This is an architecture-limited negative
  promotion result, not evidence for an M-scale claim.
