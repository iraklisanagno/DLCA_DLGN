# Experiments brief

- Main dense table: same paired seeds 0/1/2 for random, V3, U2 on MNIST 8K, Fashion-MNIST 16K, and CIFAR-10 S/M/L. Give sample SD and paired U2-minus-random intervals. Preserve paper-reported baselines separately where exact architecture references exist.
- Convolutional table: S/M existing seed-0 held-out accuracy and resources, including the longer M U2 wall time. Separate full-S three-seed validation table; no new seed-1/2 test numbers exist.
- Attribution: dense M SS/SR/RS/RR at 20K updates, three seeds; retain all five exploratory contrasts in supporting notes. Main plot/table should show first-layer effects and uncertain deeper effects.
- Include the convolutional body/classifier ablation: +2.140 pp conditional body effect with the reference-U2 classifier, CI [0.674,3.606]. The classifier was built using reference U2-body ancestry; no head-independent claim.
- Training trade-off: LILogic protocol 1x64K and 2x128K, U2 versus local Top-32, including lost accuracy. Top-32 has n=1. Keep published Top-32 values in a separate reported column.
- Constraints: dense 108K and conv 20K adapted WARP have different outcomes; ancestry novelty is not separately established; CIFAR-100 is inconclusive; CIFAR-10.1 is positive dense but mixed convolutional. Use supporting evidence notes for full raw rows and mention the principal boundaries in the paper.
- Complete protocol table: architecture, encoding, optimization, updates/epochs, batch size, augmentation, split, selection cadence, and seed counts. Inspect saved configs rather than infer them from defaults.

Authoritative sources: second/third/fourth round JSON summaries, full-M run/test receipts, and UR1 summary JSON. Training times are measured wall time including evaluation; GPU memory is peak PyTorch allocation. Existing runs were not re-executed for the manuscript.
