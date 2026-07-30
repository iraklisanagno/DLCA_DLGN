# Task-aware rewiring extension

Date: 2026-07-29

## Status

`[TRIED, STOPPED]` on dense CIFAR-10 M. This method is complementary to
CoverageDLGN V3 and does not replace or modify V3. It failed its predeclared
promotion gate and must not be presented as the paper's primary method.

## Mechanism

The optional training event starts from the exact frozen V3 topology. On one
ordinary minibatch at step 10,000, every dense layer records
class-conditional absolute activation-gradient signatures for its input
features and output gates. A bounded search then proposes two-edge swaps that
improve output-gate/task-signature affinity and input-signature diversity.

Every accepted swap:

- preserves the exact predecessor degree vector;
- preserves gate count, LUT rank, gate weights, output groups, and training
  effort;
- introduces no trainable routing parameters;
- leaves only fixed integer indices in the deployed circuit; and
- discards all activations, gradients, labels, and signatures immediately
  after the event.

The event is disabled by default (`task_aware_rewire_step=None`). Frozen V3's
constructor in `topology.py` was not edited.

## Frozen pilot

Protocol: `protocols/cifar10_medium_task_aware.json`.

- Dataset/architecture: CIFAR-10, `DlgnCifar10Medium`, four layers by 128K
  rank-2 gates (512K gates).
- Training: the existing 20K protocol, seeds 0--2, split seed 2027.
- V3 base: candidate pool 8, swap fraction 0.25, novelty weight 1.0.
- Intervention: step 10K, maximum eligible fraction 0.125, candidate pool 8,
  diversity weight 0.25, one ordinary 100-example batch.
- Promotion required both +2 pp over random and +1 pp over frozen V3.
- Existing random and V3 controls were reused and were not rerun.
- Held-out test access was prohibited.

The pre-event index hashes for every layer and seed match the corresponding
old frozen V3 checkpoint.

## Result

| Method | Best hardened validation accuracy |
|---|---:|
| Fixed random | 54.820 +/- 0.530% |
| Frozen CoverageDLGN V3 | 59.293 +/- 0.214% |
| V3 + task-aware event | 59.093 +/- 0.234% |

V3 plus task-aware rewiring remains +4.273 pp over random (paired 95% CI
[+3.421, +5.126]), but is -0.200 pp below V3 (paired 95% CI
[-0.679, +0.279]). Its three per-seed changes relative to V3 are -0.120,
-0.420, and -0.060 pp.

The event changed 57,818, 57,582, and 57,722 gates (about 11.3% of the 512K
gates). Swap computation took 3.23, 3.22, and 3.09 seconds. Training wall time
increased by only a few seconds, while peak allocated GPU memory rose from
about 1.123 GiB for the old V3 pilot to 1.336 GiB because the one-batch
activation/gradient signatures are transiently retained. Deployment cost is
unchanged: 512K gates, 8.192M LUT parameters, zero training routing
parameters, and 16.64M deployed routing bits.

## Decision

The +1 pp-over-V3 threshold failed and all paired effects were negative.
Therefore:

- do not run the 108K schedule;
- do not evaluate these checkpoints on held-out test;
- do not transfer this method to other datasets or convolutional models; and
- retain the implementation and artifacts as a documented negative result.

Source of truth:
`summary/cifar10_medium_task_aware.json`.
