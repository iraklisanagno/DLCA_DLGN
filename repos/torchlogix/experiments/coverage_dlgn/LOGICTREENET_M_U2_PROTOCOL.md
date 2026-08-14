# LogicTreeNet-M U2 protocol

Status: 200K training complete and validation selection SHA-256 frozen; held-out
test not yet run.

## Purpose

Measure U2 on the paper-faithful, nine-channel CIFAR-10 LogicTreeNet-M
architecture using one seed and the same 200,000-update budget as the completed
fixed-random and Legacy V4 runs. This is an additional U2 transfer point, not a
replacement for frozen V3 or Legacy V4.

## Frozen comparison

The control is
`configs/full_conv_cifar10_paper_medium_random_seed0_200k.json`. The U2 run is
`configs/logic_tree_m_u2/full_conv_cifar10_paper_medium_u2_seed0_200k.json`.
The output directory is
`results/full_conv_cifar10_paper_medium_u2_seed0_200k`.

Both configurations use:

- `ClgnCifar10PaperMedium` (LogicTreeNet-M, `k=256`);
- three uniform thermometer thresholds for each RGB channel, producing nine
  Boolean input channels;
- seed 0, split seed 2027, and topology seed 0;
- 45,000 training and 5,000 validation images;
- batch size 128 and 200,000 optimizer updates;
- AdamW, learning rate 0.02, and weight decay 0.002;
- standard CIFAR crop/flip augmentation on training only;
- raw rank-2 gates, soft training, residual initialization probability 0.951;
- fixed global uniform input binarization estimated with 100 batches;
- hardened and relaxed validation evaluation every 2,000 updates.

The only intended method difference is fixed-routing construction. The random
control uses `random` for the generic, convolutional, and classifier routing
initializers. U2 uses `semantic_multiscale_balanced` for all three. The coverage
candidate-pool, swap-fraction, and novelty fields are retained for exact JSON
shape parity but are not used by U2 or random routing.

## Execution and selection policy

1. Run the ten-update CUDA smoke configuration first and require CUDA device
   metadata, finite loss, validation output, topology diagnostics, and
   checkpoints.
2. Start the full run only on a visible CUDA GPU and refuse to overwrite an
   existing output.
3. Keep the complete learning curve, topology diagnostics, thresholds,
   environment, configuration, wall time, and peak CUDA memory.
4. Select the checkpoint with maximum hardened validation accuracy after the
   full 200,000 updates and freeze its SHA-256 hash.
5. Do not query the held-out CIFAR-10 test set before selection is frozen. Then
   evaluate that selected checkpoint once and report both hardened and relaxed
   test accuracy.

The paper reports 71.01% test accuracy for LogicTreeNet-M. That value is an
external reference, not a matched claim: this run uses the repository's frozen
TorchLogix reproduction protocol. The matched primary comparison is U2 versus
the already completed fixed-random seed-0 run at the same 200,000-update budget.

## CUDA smoke outcome

The ten-update preflight completed on physical GPU 0 with PyTorch 2.9.0+cu130
and CUDA build 13.0. Metrics were finite, hardened and relaxed validation both
ran at step 10, and the expected topology, threshold, environment,
configuration, best-checkpoint, and final-checkpoint artifacts were created.
Peak allocated CUDA memory was 15,692,077,568 bytes. The smoke output is
`results/smoke_conv_cifar10_paper_medium_u2_seed0`; model/checkpoint binaries
remain excluded from Git.

Immediately before the full launch, physical GPU 0 acquired two unrelated
`/opt/conda/bin/python3` workloads and sustained approximately 48% utilization,
while physical GPU 1 remained idle. Because both devices are the same RTX PRO
6000 Blackwell model, the full job is assigned to physical GPU 1 to avoid
resource contention. This allocation does not change any training or method
field in the frozen JSON configuration.

## Full-training outcome and validation freeze

The GPU-1 run completed all 200,000 updates in 128,468.16 seconds. Peak PyTorch
CUDA allocation was 15,692,077,568 bytes. The maximum hardened validation
accuracy was 72.38% at step 136,000; final hardened and relaxed validation were
71.64% and 73.36%, respectively. Across the 100 matched evaluations, U2 beat
fixed-random 97 times with a mean hardened gain of 1.7876 percentage points,
and beat Legacy V4 87 times with a mean gain of 0.7328 points.

Before any held-out test access, `freeze_logic_tree_m_u2_200k.py` verified the
complete validation history and froze `best_checkpoint.pt` at step 136,000 in
`summary/cifar10_paper_medium_u2_200k_freeze.json`. Its SHA-256 is
`f2bf39448cd8810ffaa349d2a96c078eef2b3c1fd3b6f8c8e144645ce3d2f69b`.
