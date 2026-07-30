# CIFAR-100 dense baseline reproduction audit

Date: 2026-07-29

This audit explains why the local 6-by-64K random result must be labeled
`[REPRODUCED]` rather than an exact numerical reproduction of the reported
22.54 +/- 0.26%. It does not change frozen V3 and it does not authorize a
second held-out-test query.

## Result

| Source | Hardened test accuracy |
|---|---:|
| Scalability-boundaries paper | 22.54 +/- 0.26% (three runs) |
| Local TorchLogix random | 20.677 +/- 0.522% (three seeds) |
| Local TorchLogix CoverageDLGN V3 | 21.010 +/- 0.131% (three paired seeds) |

The reproduced random mean is 1.863 percentage points below the reported
mean. V3 improves the paired local control by +0.333 pp, but its 95% paired
interval [-0.816, +1.483] crosses zero.

## What matches

The local run artifacts and the paper agree on:

- CIFAR-100 with no augmentation;
- 20% of the 50,000 training examples reserved for validation;
- three fixed thresholds at 0.25, 0.50, and 0.75, producing 9,216 input bits;
- six rank-2 logic layers, each with 64,000 gates (384,000 gates total);
- a 64,000-gate GroupSum input split across 100 classes;
- GroupSum temperature 10;
- Adam, learning rate 0.01, batch size 100;
- 100 epochs, implemented locally as 40,000 optimizer steps over 40,000
  training examples; and
- three independent seeds.

The exact local thresholds are recorded in every `thresholds.csv`. All six
random/V3 runs share the same source revision and training implementation
hash.

## Material mismatch: random-routing generator

The paper builds on the original `difflogic` implementation. Its random
`LogicLayer` routing performs two Torch permutations:

```python
c = torch.randperm(2 * out_dim) % in_dim
c = torch.randperm(in_dim)[c]
```

The paired local study deliberately uses TorchLogix's independently seeded
NumPy topology generator:

```python
values = rng.permutation(n_slots) % in_dim
```

These produce different fixed graphs. They also have different random-number
stream semantics: the original Torch generator advances the same framework
RNG used around model initialization, whereas `topology_seed` is deliberately
independent so random and V3 can share weight initialization. This independence
is desirable for the causal random-versus-V3 comparison, but it means the
local random arm is not a bit-faithful reproduction of the published baseline.

## Other unresolved reproduction details

- The paper states that 20% is sampled for validation but does not report its
  split seed; the local split seed is fixed to 2027.
- The paper does not state whether Table 12 uses the final epoch or a
  validation-selected checkpoint. The local result uses the best hardened
  validation checkpoint and queries the test set once.
- The exact source revision and dependency stack used for the reported table
  are unavailable in this workspace.

These differences are sufficient to prevent attributing the entire 1.863 pp
gap to CoverageDLGN, optimizer behavior, or a TorchLogix defect.

## Decision

Keep the completed paired local result as the scientific control for the V3
claim. In the paper table:

- label 20.677 +/- 0.522% as `[REPRODUCED, topology-adapted]`;
- retain 22.54 +/- 0.26% separately as `[REPORTED]`;
- do not query the completed checkpoints on test again; and
- if exact paper reproduction becomes essential, create a new, explicitly
  labeled canonical-difflogic-routing cohort. Do not replace or mix it with
  the independently seeded paired V3 control.
