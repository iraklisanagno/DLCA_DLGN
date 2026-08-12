# LogicTreeNet-M U2 protocol

Status: preregistered; training and held-out test not yet run.

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
2. Start the full run only on GPU 0 and refuse to overwrite an existing output.
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
