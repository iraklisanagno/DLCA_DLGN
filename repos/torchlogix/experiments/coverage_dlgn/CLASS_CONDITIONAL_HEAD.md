# Class-conditional coverage head

## Status

The class-conditional coverage head is a reproducible negative-result
extension. It improves its intended CIFAR-100 topology diagnostic but does
not improve accuracy meaningfully over frozen V3. It is not promoted as the
paper method, and no full, transfer, convolutional, or held-out-test
experiment is authorized.

## Motivation

Offline analysis of frozen topologies showed that aggregate ancestry coverage
was already saturated. Every CIFAR-100 class under V3 covered all 3,072 raw
RGB spatial sources, but its mean per-class source-usage coefficient of
variation was 0.2465 for seed 0. CIFAR-10 L V3 was much more uniform at
0.0529. A new mechanism therefore needed to redistribute existing ancestry,
not maximize already complete coverage.

The complete offline comparison is
`summary/classwise_head_diagnostics.json`.

## Mechanism

The implementation is a separate `class_conditional_coverage` fixed-topology
strategy:

1. Construct the backbone and final-layer base using frozen V3.
2. Restrict the new strategy to the final `GroupSum` classifier layer.
3. Select gates from different class groups.
4. Exchange their predecessor occurrences using two-edge swaps.
5. Accept only swaps that improve within-gate ancestry efficiency plus use
   of sources currently underrepresented in each affected class.

Each swap uses the same four predecessor occurrences before and after.
Consequently, predecessor degrees, gate count, LUT parameters, deployed
routing bits, circuit export, inference operations, and inference storage are
invariants. The refinement uses no data, labels, gradients, or learned routing
parameters. V3 and V4 code paths remain separately selectable and unchanged
when the classifier override is absent.

The single frozen head parameter is a maximum changed-gate fraction of 0.25.
V3 retains candidate pool 8, swap fraction 0.125, and novelty weight 1.0.

## Correctness evidence

The seed-0 CUDA smoke preserved layers 1--5 bit-for-bit relative to the frozen
V3 checkpoint. It changed 15,088 of 64,000 final-layer gates while retaining
the exact predecessor-degree vector. Both variants have 384,000 gates,
6,144,000 trainable LUT parameters, 12,032,000 deployed routing bits, and
0.795 GiB peak allocated training memory. Circuit functional-equivalence,
determinism, bounds, degree, zero-change V3 equivalence, initialization-RNG,
and budget tests cover the implementation.

## Frozen experiment

The protocol is `protocols/table4_cifar100_class_head.json`. Completed random
and V3 controls were reused without retraining. Only the three missing head
runs were executed for 20K steps on seeds 0--2. Held-out CIFAR-100 test data
was never accessed.

| Method | Best hardened validation | Source-usage CV | Within-class Jaccard |
|---|---:|---:|---:|
| Random | 21.040 +/- 0.420% | 0.27146 | 0.01049 |
| Frozen V3 | 21.580 +/- 0.100% | 0.25655 | 0.01051 |
| V3 + class head | 21.593 +/- 0.133% | **0.23475** | **0.01042** |

The head improves random by +0.553 pp, with paired 95% CI
[-0.565, +1.671], and improves V3 by only +0.013 pp, with paired 95% CI
[-0.385, +0.412]. The predeclared promotion requirements were +2 pp over
random and +1 pp over V3. Both failed.

Mean offline topology construction was 2.07 seconds for random, 37.47 seconds
for V3, and 49.50 seconds for V3 plus the head. Mean training wall time and
peak GPU allocation were unchanged within normal run variance.

Machine-readable results are in
`summary/table4_cifar100_class_head.{json,csv}`.

## Interpretation

Uniform per-class ancestry is not the missing CIFAR-100 ingredient. The
result strengthens the broader boundary diagnosis: structural coverage
statistics can be improved without improving classification. Future work
must target task-relevant representation or optimization rather than another
label-free ancestry-balancing objective.
