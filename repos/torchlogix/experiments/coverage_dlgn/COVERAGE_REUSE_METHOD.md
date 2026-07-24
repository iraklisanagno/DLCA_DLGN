# Coverage--Reuse Balanced Topology Refinement

**Status:** implemented; CIFAR-10 S/M validation screening complete; not promoted  
**Strategy name:** `coverage_reuse_hybrid`

## Motivation

Dense semantic-balanced v3 and convolutional semantic-channel v4 showed that
fixed DLGN routing benefits from structured source coverage. The ancestry-only
v5 experiment then showed that maximizing distinct pairs and ancestry is not
monotonically related to hardened accuracy: v5 improved every topology
diagnostic but tied random routing over three paper-S seeds.

Coverage--reuse refinement tests the resulting hypothesis: a useful topology
must expose gates to complementary ancestry while retaining repeated
predecessor motifs that the base construction treats as useful.

## Generic algorithm

The refinement receives:

1. any fixed rank-2 predecessor-index matrix;
2. packed ancestry for its input features;
3. a maximum output-change fraction;
4. a bounded candidate-pool size;
5. ancestry-novelty and base-reuse weights; and
6. an independent topology seed.

For each candidate pair of output gates, it considers the two non-trivial
rewirings of their four predecessor occurrences. A proposal is valid only when
both resulting gates have distinct inputs. Its per-gate score is:

```text
union efficiency
- within-gate ancestry overlap
+ novelty_weight * (1 - group predecessor Jaccard)
+ reuse_weight * normalized base-pair frequency
```

Only strictly improving proposals are accepted. Construction stops when the
bounded gate set is exhausted or no improving candidate exists. Thus the
change fraction is an upper bound rather than a forced replacement rate.

## Invariants

Every accepted two-edge swap uses exactly the same four predecessor
occurrences. The method therefore preserves:

- exact predecessor degree and fan-out;
- layer widths and gate count;
- LUT parameter count;
- routing-index tensor shape and storage;
- convolutional receptive-field coordinates;
- training and inference operations; and
- the absence of learned routing parameters.

Only the offline integer topology changes. Packed ancestry and scoring state
are released after construction.

## Dense and convolutional use

The refinement itself is architecture-independent.

- Dense DLGNs use the frozen semantic-balanced v3 topology as their base.
- Convolutional DLGNs use the frozen semantic-channel v4 channel topology as
  their base and leave all spatial coordinates unchanged.
- If semantic source metadata is unavailable, singleton input identity defines
  ancestry.

The same `coverage_reuse_change_fraction` and `coverage_reuse_weight`
parameters apply in every layer. No CIFAR-specific layer schedule is used.

## Initial frozen screening configuration

The first S/M screen uses:

```text
candidate_pool_size = 8
base_swap_fraction = 0.25
change_fraction = 0.25
novelty_weight = 1.0
reuse_weight = 1.0
```

These values are shared by paper-S and paper-M. Candidate promotion is based
only on hardened validation accuracy. Held-out CIFAR-10 test data remain
unexamined until a method and configuration are frozen by multi-seed
validation.

## CIFAR-10 S/M screening result

The implementation passed CUDA smoke tests on the paper-faithful
`ClgnCifar10PaperSmall` and `ClgnCifar10PaperMedium` architectures. Paired
seed-0 screens then used identical initialization, data split, augmentation,
gate budget, training effort, and validation schedule within each comparison.

| Architecture / protocol | Frozen v4 | Coverage--reuse | Difference |
|---|---:|---:|---:|
| Paper S, best hard validation in 5K steps | 54.92% | 55.40% | +0.48 pp |
| Paper M, hard validation at 1K, evaluation every 500 steps | 35.78% | 38.92% | +3.14 pp |
| Paper M, best hard validation in confirmatory 5K run | **61.56%** | 58.44% | **-3.12 pp** |

The M 1K signal did not reproduce after the confirmation protocol changed the
evaluation interval from 500 to 1,000 steps. In the confirmatory M run,
coverage--reuse led by +1.00 pp at 2K but trailed by 2.82, 1.42, and 3.12 pp
at 3K, 4K, and 5K. This is evidence of an unstable early-training observation,
not a successful accuracy improvement.

Both M variants used 668,416 learned gates and the same
15,692,077,568-byte peak GPU allocation. Convolutional topology construction
increased from 4.884 seconds for v4 to 15.498 seconds for coverage--reuse; the
approximately 10.6-second difference is offline construction overhead.
End-to-end wall time increased from 2,314.5 to 2,334.2 seconds.

The predefined +3 pp promotion criterion was therefore not met robustly.
Multi-seed 20K training and held-out test evaluation were deliberately not
run. Frozen v4 remains the retained convolutional method. Coverage--reuse is
kept as a generic, tested experimental mechanism and negative result; v3 and
v4 are unchanged.

Exact records are in
`summary/paper_conv_coverage_reuse_screen.json` and the paired result
directories named by that file. Reproduce the decisive pair with:

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/confirm_conv_cifar10_paper_medium_v4_control_seed0.json

CUDA_VISIBLE_DEVICES=1 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/train.py \
  --config experiments/coverage_dlgn/configs/confirm_conv_cifar10_paper_medium_coverage_reuse_seed0.json
```

## Evidence required for a paper claim

An initial positive screen is not sufficient. A generic-method claim requires:

- multiple seeds under paired initialization and data splits;
- dense and convolutional architectures;
- at least two model scales or gate budgets;
- random, frozen-base, and named published baselines;
- hardened accuracy and learning curves;
- circuit cost, construction time, training memory, and inference runtime; and
- no dataset-specific topology parameters.
