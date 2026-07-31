# WARP Figure 4 reproduction and Legacy V4 integration

This experiment reconstructs Figure 4 of
`pdfs/warp_logic_neural_networks.pdf` using the public TorchLogix
implementation and then evaluates the frozen CoverageDLGN Legacy V4 topology
in the same CIFAR-10 Medium, two-threshold setting.

The WARP paper reports validation learning curves rather than exact numerical
endpoints. Reading the 50K-step plot gives approximately 64.0% for fixed
uniform thresholds, 65.0% for fixed distributive thresholds, and 66.6% for
learnable thresholds. Each curve averages three seeds on an 80/20
train/validation split.

## Protocol reconstruction

The numerical reproduction uses `ClgnCifar10Medium`, raw DLGN gate
parameterization, two global thresholds per RGB channel, batch size 128,
30,000 updates, and three seeds. The public WARP repository's Medium example
uses `random-unique` convolutional routing; that is the reproduction baseline.
The public data loader at the imported revision applies `ToTensor` without
crop/flip augmentation, so `augmentation=none` is retained.

Learnable thresholds use distributive initialization, relative learning-rate
argument 0.02, sampling temperature 0.0002, and Softplus temperature 0.01,
following WARP Table 4 and the public command-line semantics.

This is a reconstructed reproduction because the paper/repository does not
provide the exact Figure 4 command, split indices, checkpoints, or exact
numeric endpoints. Every ambiguity is frozen before observing results in
`protocols/warp_fig4_cifar10_medium.json`.

The public README recommends `torch.compile`. A paired CUDA preflight on
2026-07-30 reached model construction but remained inside PyTorch 2.9
`max-autotune` for more than two minutes before the first update on both GPUs,
with repeated symbolic-shape warnings. It was interrupted cleanly; no training
update or result was produced. The locked runs therefore use
`compile_model=false`. Compilation is an execution optimization and does not
alter the numerical model, optimizer, topology, or number of training updates.

## Legacy V4 attribution

Legacy V4 is not called a WARP method. It is our frozen
`semantic_channel_hybrid` topology evaluated in the WARP experimental setting.
Its matched control uses the same `random` spatial sampler as the original V4
pilot, a fixed split seed, paired topology/training seeds, and identical
training settings. Only channel topology differs. The V4 parameters remain:

- candidate pool size 8;
- ancestry-swap fraction 0.25;
- novelty weight 1.0;
- raw rank-2 DLGN gate parameterization.

No V3 or V4 source code is changed by this experiment.

## Staged execution

```bash
bash experiments/coverage_dlgn/run_warp_fig4_cifar10_medium_two_gpus.sh screen
venv/bin/python experiments/coverage_dlgn/summarize_warp_fig4_cifar10_medium.py
```

The initial fixed-uniform and fixed-distributive seed-0 jobs were launched with
the plotted 50K budget. Once both had reached or exceeded the approximate WARP
endpoints, they were interrupted immediately after their common 30K validation
boundary. Neither reached a later evaluation. The frozen budget was then
changed to 30K before any Medium Legacy V4 accuracy was observed. Every
remaining arm and seed uses the same 30K maximum training effort.

This result must be described as reproducing the plotted WARP accuracy within
30K updates, not as an exact 50K reproduction. The screen uses full 30K runs
for each WARP thresholding arm and the fixed-uniform random/V4 pair; it is not
a reduced-training proxy relative to the frozen protocol. Continue
the remaining fixed-uniform seeds only if the predeclared promotion rule passes:

```bash
bash experiments/coverage_dlgn/run_warp_fig4_cifar10_medium_two_gpus.sh fixed-full
```

The learnable-threshold V4 composition is run only after the three-seed
fixed-uniform V4 mean is positive:

```bash
bash experiments/coverage_dlgn/run_warp_fig4_cifar10_medium_two_gpus.sh learnable-full
```

All selection uses hardened validation accuracy. The CIFAR-10 test set is not
loaded into the training decision and must not be queried until the final
protocol is frozen.

## Seed-0 result

All five planned seed-0 screen arms completed at the frozen 30K boundary:

| Method | Best hardened validation accuracy |
|---|---:|
| WARP fixed uniform | 65.35% |
| WARP fixed distributive | 66.12% |
| WARP learnable | 65.88% |
| Matched random, fixed uniform | 64.58% |
| Frozen Legacy V4, fixed uniform | **66.23%** |

Legacy V4 is +1.65 percentage points over its matched random control. This is
one paired seed and is insufficient for a mean or confidence interval. The
three WARP arms are compared descriptively with the approximate Figure 4
endpoints; the V4 attribution uses only the matched-random pair because its
sampler deliberately follows the historical V4 experiment rather than the
public WARP `random-unique` setting. No held-out test evaluation was run.

Machine-readable results are in
`summary/warp_fig4_cifar10_medium.{json,csv}`. Seeds 1 and 2 are pending.
