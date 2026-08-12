# Frozen U2 Published-Protocol Round

## Purpose

This round tests the already frozen `semantic_multiscale_balanced` U2 rule at
dense CIFAR-10 M/L scale and places it directly on the LILogicNet and BitLogic
training protocols. It does not alter U2, V3, or V4.

All training is CUDA-only through `repos/torchlogix/venv`. Existing complete
random, V3, and Mommen runs are reused and never rerun.

## Experiment matrix

| Phase | Local methods | Coordinates | Seeds | Full runs |
|---|---|---|---:|---:|
| Existing Deep-DLGN M/L | U2 only | 4 x 128K and 5 x 256K | 0, 1, 2 | 6 |
| LILogicNet protocol | fixed random, U2 | 1 x 64K and 2 x 128K | 0, 1, 2 | 12 |
| LILogicNet protocol | Top-32 | same two coordinates | 0 | 2 |
| BitLogic common protocol | rank-2 random, rank-2 U2, rank-4 best-of-space | 2 x {4K, 16K, 64K} | 0, 1 | 18 |

Total: 38 full runs. The six CUDA smokes are not accuracy experiments.

## Existing Deep-DLGN M/L

These runs inherit the frozen local protocol exactly: raw rank-2 gates, fixed
uniform input thresholds, no augmentation, batch 100, Adam at 0.01, 108,000
updates, split seed 2027, and paired training/topology seeds. Only the fixed
connection strategy changes to U2.

## LILogicNet protocol

The M coordinate has one 64K layer and GroupSum temperature 90. The L
coordinate has two 128K layers and temperature 100. Both use raw rank-2 gates,
seven uniform thresholds per RGB channel, reflect-padded crop and horizontal
flip, Adam at 0.075, batch 256, and 200 base-dataset epochs. Validation is
unaugmented.

The paper describes augmentation as an eight-fold expansion while separately
specifying 200 epochs. Its reported fixed-model times are consistent with 200
passes over the base dataset using online augmentation, not 1,600 effective
epochs. The local protocol therefore uses 35,000 updates:
`floor(45,000 / 256) * 200`.

The local Top-32 M row reproduces the paper's reported `1Top32-64K` coordinate,
not its more expensive fully learnable `1L-64K` headline. The latter remains a
reported-only reference. The L row is the exact reported `2Top32-128K`
coordinate. LILogic's BasisProj and TorchLogix's non-materialized raw rank-2
basis are algebraically equivalent, but runtime numbers remain labeled by
implementation and GPU.

## BitLogic common protocol

All cells use a two-layer width ladder of 4K, 16K, and 64K per layer, a 90/10
split, reflect-padded crop/flip, AdamW at 0.01 with zero weight decay, batch
128, and 100 epochs (`35,100` updates). The rank-2 random and U2 arms use three
uniform thresholds and raw gates. The best-of-space arm uses four global
quantile thresholds, Light rank-4 gates, and learnable-16 routing.

U2 currently supports rank-2 LUTs only. Therefore random versus U2 is the
controlled topology comparison. U2 versus rank-4 BitLogic is an explicitly
labeled accuracy--cost Pareto comparison. U2 is not extended or modified.

The BitLogic appendix contains an internal inconsistency: the axis-constraint
prose says the cross-method rows are pinned to rank 4, while Table 8 lists
DiffLogic, WARP-LUT, and LILogicNet at their native rank 2. The local rank-2
coordinate follows Table 8 and the native DiffLogic definition. This
discrepancy must remain visible when reported values are transcribed.

## Evaluation and accounting

Each full run records hardened and relaxed learning curves, the best hardened
validation checkpoint, final checkpoint, wall time, peak allocated GPU memory,
topology-construction time, trainable gate and routing parameters, deployed
routing bits, and environment fingerprint. After a whole phase completes,
both the predeclared best-validation and final checkpoints are evaluated once
on held-out test. The final checkpoint is the paper-protocol coordinate; the
best-validation checkpoint supports the repository's existing selection
convention.

Hardened inference benchmarking reports CUDA latency/throughput and peak
allocated memory for every run; CPU/circuit timing is added only where a
separate export study supports it. Gate count is accompanied by rank,
truth-table bits, routing bits, and simplified circuit cost where export
supports it. Training runtime from a different GPU is never presented as a
direct speed ratio.

## Promotion and failure handling

The planned Top-32 comparator budget is one complete local seed per M/L cell,
plus the paper's five-run reported distribution. It is promoted to three local
seeds only if the local accuracy differs materially from the reported
coordinate and the discrepancy cannot be explained by a documented protocol
difference. Failed or interrupted outputs are moved under `results/failed/`
before retry; incomplete directories are never silently overwritten.

## Completion status

Completed August 12, 2026: all 38 full CUDA runs finished without a training
failure. The manifest `summary/third_round_validation_freeze.json` froze 76
checkpoint hashes before held-out access. Both predeclared checkpoints were
then evaluated once for every run on CUDA; all 38 evaluations completed with
zero failures. Aggregated records are in `summary/third_round_results.json`,
`summary/third_round_runs.csv`, and `summary/third_round_groups.csv`.

The local BitLogic rank-4 arms are reproduced-negative protocol transfers,
not faithful numerical reproductions of the paper. Their relaxed models learn
substantial signal, but hardened accuracy collapses. The paper values remain
separate `[REPORTED]` references. No failed local value is substituted for a
published value.
