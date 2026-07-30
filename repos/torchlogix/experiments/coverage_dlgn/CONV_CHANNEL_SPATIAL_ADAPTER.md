# CoverageDLGN convolutional channel-spatial adapter

## Status and method-freeze policy

This experiment is a separate candidate extension. It does **not** modify the
frozen dense V3 (`semantic_balanced_hybrid`) or frozen convolutional V4
(`semantic_channel_hybrid`) behavior. Existing random, V3, and V4 results are
reused and are never retrained merely to support this experiment.

The paper-facing method should be described as one CoverageDLGN construction
principle with domain-specific source representations:

- dense inputs are image-source/threshold tokens;
- convolutional inputs are channel/receptive-field-offset tokens;
- both use a degree-balanced base, coverage-aware fixed routing, and no
  trainable connectivity or additional deployed gates.

The descriptive internal strategy name for this candidate is
`semantic_channel_spatial_hybrid`. Version labels are not intended as paper
method names.

## Motivation from the frozen V4 audit

Frozen V4 changes channel pairs but deliberately leaves receptive-field
coordinates equal to the random control. Thus its coverage objective sees only
the channel axis. The CIFAR-10 S pilot produced a small, inconclusive validation
gain of +0.513 percentage points over random.

The missing component is channel-spatial source pairing inside each logic tree. A
depth-3 rank-2 convolutional tree consumes eight leaves from a 3x3 receptive
field and two selected channels. Frozen V4 samples the four offsets for each
channel independently with replacement and randomly permutes all eight leaves.
It may therefore:

- give a bottom-level LUT two leaves from the same channel.

The project specification explicitly requires the convolutional extension to
leave spatial receptive-field indexing unchanged. The adapter therefore does
not resample, balance, or otherwise change spatial coordinates.

## Proposed adapter

The adapter preserves the exact frozen V4 channel-pair generator, its
degree-preserving coverage swaps, and its sampled spatial-coordinate tensor. It
changes only the assignment of the two selected channels to those existing
local convolutional leaves:

1. Generate the complete frozen V4 kernel tensor.
2. Preserve every spatial coordinate bit-for-bit.
3. Assign one selected channel to each input of every bottom-level rank-2 LUT.
4. Alternate rank orientation across leaves and kernels, preserving four
   occurrences of each selected channel per kernel.

For the paper S/M/L architecture (depth 3, 3x3 receptive field), every kernel
therefore has:

- the same two selected channels as frozen V4;
- cross-channel mixing in 100% of bottom-level LUTs;
- exactly the same spatial-offset multiset and spatial fan-out as frozen V4.

The construction is deterministic for a fixed topology seed and uses an
independent generator, so it does not change weight initialization or training
randomness.

## Cost

The adapter adds:

- zero trainable parameters;
- zero logic gates;
- zero deployed routing entries;
- zero training-time objectives;
- only offline integer topology construction.

## Controlled experiment

Dataset and architecture: CIFAR-10, `ClgnCifar10PaperSmall`.

Selection uses hardened validation accuracy only, with seeds 0, 1, and 2,
20,000 iterations, split seed 2027, and the same training configuration as the
completed random and V4 controls. The held-out test set is not used.

Only the new adapter is trained. Controls are reused from:

- `pilot_conv_cifar10_paper_small_random_seed{0,1,2}`;
- `pilot_conv_cifar10_paper_small_semantic_channel_v4_seed{0,1,2}`.

All V4-derived arms use the historical component-wide method selector rather
than explicit convolution/classifier overrides. This preserves the frozen V4
dense-classifier routing and the global RNG state used for later dense weight
initialization.

An initial explicit-override attempt was stopped after a protocol audit found
that it gave dense classifier routing an independent topology RNG. That changed
later dense parameter initialization and therefore did not isolate the
convolutional mechanism. The two completed no-swap runs and two interrupted
runs are retained, but excluded, under
`results/failed/cifar10_conv_small_explicit_classifier_rng_attempt1`.

The candidate advances to CIFAR-10 M only if its three-seed mean gain is both:

- at least +2.0 percentage points over random;
- at least +1.0 percentage point over frozen V4.

If either condition fails, the M experiment is not run. The result remains
documented as a reproducible negative or diagnostic result.

## Verification

Before the pilot, the focused suite passed:

- strategy alias and CLI acceptance;
- deterministic topology construction;
- exact equality of V4 and adapter channel pairs;
- exact equality of their spatial coordinates;
- unchanged model-weight initialization and subsequent Torch RNG state;
- exact equality of V4 and adapter spatial-coordinate hashes and fan-out;
- cross-channel bottom-level pairing;
- hardened/export-mode repeatability.

Command:

```text
venv/bin/python -m pytest tests/test_clgn.py tests/test_coverage_topology.py tests/test_experiment_protocol.py tests/test_circuit.py -q
```

Final corrected result: 1,972 passed and 1,660 skipped. This includes the
full-model common-RNG regression for trainable parameters and dense routing.

## Result and decision

All three corrected 20K runs completed. The adapter reached 57.033% mean
best hardened validation accuracy, versus 56.673% random and 57.187% frozen
V4:

- adapter minus random: +0.360 pp, 95% paired Student-t CI
  [-0.501, +1.221];
- adapter minus frozen V4: -0.153 pp, 95% CI [-2.714, +2.407].

It failed both predeclared promotion gates (+2 pp over random and +1 pp over
V4). No CIFAR-10 M run, held-out test evaluation, or further scaling is
authorized for this adapter.

The accompanying V4 component ablation was more informative. Balanced channel
routing with swaps disabled reached 58.013%, versus 57.187% V4 and 56.673%
random:

- no-swaps minus random: +1.340 pp, 95% CI [-0.988, +3.668];
- no-swaps minus V4: +0.827 pp, 95% CI [-2.394, +4.047].

The direction is positive but inconclusive with three seeds and remains below
the +2 pp random threshold. It is retained as the best convolutional
diagnostic candidate, not promoted or silently substituted for frozen V4.

Machine-readable results:

- `summary/cifar10_conv_small_v4_components.json`;
- `summary/cifar10_conv_small_channel_spatial.json`.
