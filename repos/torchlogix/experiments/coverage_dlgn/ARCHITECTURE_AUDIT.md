# Convolutional DLGN architecture audit

This audit compares `src/torchlogix/models/conv.py` with pages 6, 14, and 15
of `pdfs/convolutional_differentiable_logic_gate_networks.pdf`. It prevents a
TorchLogix class name from being mistaken for a protocol-identical
reproduction of the published LogicTreeNet.

## Paper scale definitions

The CIFAR-10 models use `S: k=32`, `M: k=256`, `B: k=512`, and
`L: k=1024`, with the appendix listing `G: k=2560`. Page 6 instead prints
`G: k=2048`; this internal paper inconsistency does not affect S, M, or B.

MNIST has a separate scale table: `S: k=16`, `M: k=64`, and `L: k=1024`.
The CIFAR scale mapping must not be applied to MNIST.

## CIFAR-10 S and M

`ClgnCifar10PaperSmall` and `ClgnCifar10PaperMedium` implement the appendix
body:

1. Four `3x3`, depth-3 logic-tree convolution blocks with output channels
   `k`, `4k`, `16k`, and `32k`.
2. Padding one in every convolution, followed by `2x2`, stride-2 OR pooling.
3. Flattening from `32k x 2 x 2` to `128k`.
4. Dense logic layers `128k -> 1280k -> 640k -> 320k`.
5. Ten-class GroupSum with paper temperatures 20 (S) and 40 (M).
6. Exactly two input channels observed by each convolutional tree.
7. Two-bit RGB precision encoded with three thermometer thresholds, producing
   nine Boolean channels.

The new classes are separate from legacy `ClgnCifar10Small` and
`ClgnCifar10Medium`, which produce six Boolean channels from two thresholds.
This preserves every pre-audit checkpoint and result.

S has 83,552 learned gate functions: 11,872 shared convolution-tree functions
and 71,680 dense functions. Changing from six to nine input channels does not
change that gate budget or the number of spatial gate applications.

## Fair CoverageDLGN comparison

Random and semantic-channel v4 use the same:

- architecture, threshold values, gate count, and tensor shapes;
- data split, augmentation, minibatch order, optimizer, and training effort;
- gate-function initialization and spatial receptive-field coordinates.

Only convolutional channel IDs differ. The smoke topology reports verify
bit-identical spatial-coordinate hashes and identical state tensor shapes.
This uses the paper's two-channel restriction and leaves the learned logic
trees and dense head unchanged.

## CIFAR-10 B is intentionally not claimed yet

The local legacy class named `ClgnCifar10Large` has `k=512` and is therefore
the paper's B scale, not its L scale. It is not a paper-exact B reproduction:

- the paper doubles the final output layer for B (`output gate factor = 2`);
- B uses five-bit input preprocessing with fixed edge and curvature detectors;
- B is trained with teacher supervision of class scores.

The PDF does not specify the complete detector bank and thresholds needed to
reconstruct that preprocessing. No `PaperBig` class is exposed until those
details are obtained from an authoritative implementation or reproduced and
explicitly documented. This guard prevents an approximate raw-RGB model from
being compared with the published 80.17% / 16.0M-gate result as if identical.

## Smoke validation

On July 23, 2026, paired CUDA smoke runs completed on two RTX PRO 6000
Blackwell GPUs:

- S: two updates plus hardened validation evaluation;
- M: one update, used only to validate construction, forward/backward, and
  checkpointing.

Both scales used threshold tensors of shape `[1, 3]`. Random and v4 had
identical state shapes and convolutional spatial hashes. Peak allocated
memory was 508.25/508.27 MiB for S and 1178.63/1178.83 MiB for M
(random/v4). These runs are execution checks, not accuracy evidence.

A separate 20-step M timing run at the pilot batch size of 128 used 14.61 GiB
peak allocated GPU memory and took 9.49 seconds, or 0.475 seconds per step
including setup. This projects to about 2.6 hours per 20K run before evaluation
overhead and approximately eight hours for three paired seeds on two GPUs.

The managed command environment cannot retain detached GPU children after its
session exits. A `nohup` M launch was cleaned up before the first training
step; its startup logs are retained under
`results/failed/paper_medium_async_launch_attempt/`. This is not an
architecture or CUDA failure. The M queue must run from a persistent terminal
using `run_conv_cifar10_paper_sm_pilot_two_gpus.sh medium`.
