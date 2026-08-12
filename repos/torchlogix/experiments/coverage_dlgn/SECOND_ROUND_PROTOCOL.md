# CoverageDLGN DATE second-round protocol

This protocol was frozen before launching the second round. It preserves the
existing V3, V4, and U1 implementations and records every new result as local
screening, adapted reproduction, reproduced matched control, or reported-only
evidence.

## Objectives and order

1. Correct the CIFAR-10 table: BitLogic and WARP-LUT use a separate two-layer
   8K/32K/128K ladder and are not S/M/L results.
2. Run architecture-matched 6 x 8K Mommen, LILogicNet, and BitLogic controls on
   MNIST and Fashion-MNIST with three seeds and the existing 200-epoch effort.
3. Measure fixed-random versus frozen V3 below 48K gates. MNIST uses
   4K/8K/16K/32K; Fashion-MNIST uses 8K/16K/32K/64K. The completed 48K results
   are reused, never rerun.
4. Train one full 350K-update LogicTreeNet-S run for fixed random, frozen V4,
   and frozen U1. This is validation selection evidence; held-out test remains
   locked until checkpoints are frozen.
5. Strengthen dense CIFAR-100 using a baseline-only training screen, complete
   the missing seeds of the 3 x 128K pair, and test two same-384K class-scaled
   output allocations.
6. Evaluate a separate unified candidate, U2
   (`semantic_multiscale_balanced`). U2 uses semantic input ordering followed
   by deterministic regular matching stages. It minimizes fan-out spread,
   then selects scale using normalized ancestry novelty; a final partial stage
   remains a prefix of one matching. It has no swaps, learned routing, or
   deployed routing overhead.
7. Validate U2 at five coordinates: compressed MNIST, compressed
   Fashion-MNIST, dense CIFAR-10 S, convolutional CIFAR-10 S, and dense
   CIFAR-100 3 x 128K. All pilots use seeds 0, 1, and 2.
8. Freeze one topology rule only after the pilot results are available, then
   promote only coordinates with a positive paired result to final multi-seed
   accuracy, circuit, time, and memory measurements.
9. A calibrated fixed-routing fallback is permitted only if both CIFAR-100 and
   convolutional pilots fail. It must reconstruct a fixed topology before
   final training, reset weights, and count calibration effort. It is not part
   of U2 and cannot be introduced post hoc into successful coordinates.

## Reproducibility and compute rules

- All training uses `repos/torchlogix/venv/bin/python` and an explicit CUDA
  device. The queue launcher allocates and synchronizes a probe tensor on
  every requested GPU and rejects any non-CUDA config before launching work.
- The data split seed is fixed at 2027. Training and topology seeds are paired.
- Existing completed result directories are skipped. A nonempty incomplete
  directory is classified before any retry; it is never silently overwritten.
- Training curves, best hardened-validation checkpoints, topology reports,
  environment manifests, elapsed time, and peak GPU memory are retained by
  the existing harness.
- No partial run is described as an exact resume because optimizer,
  data-loader, and RNG states are not currently checkpointed.

Generated configs and queues are produced by `prepare_second_round.py`.
`summarize_second_round.py` regenerates a live per-run and grouped status ledger;
pending, incomplete, and complete outputs remain explicitly distinguished.

## Frozen protocol outcome

All nine steps completed on August 10, 2026. The live ledger closed at
110/110 complete CUDA runs. U2 promoted on MNIST-8K, Fashion-16K, dense
CIFAR-10 S, and convolutional CIFAR-10 S, and was rejected on dense
CIFAR-100. The fallback in step 9 was not eligible because the convolutional
pilot succeeded. Full convolutional validation was frozen before one-time
test access; U2 reached 61.000% hardened validation and 60.630% hard test,
+2.320/+3.260 pp over the matched random control. Complete interpretation and
limitations are in `SECOND_ROUND_CONCLUSIONS.md`.
