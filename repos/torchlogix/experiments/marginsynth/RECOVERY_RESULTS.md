# Short-recovery development results

These are seed-0 Fashion-MNIST development results. They are not a five-seed
paper claim, and the test split remains sealed. Snapshot selection used only a
held-out subset of the original training partition. Calibration and validation
curves were evaluated after snapshot selection was complete.

## Recovery-length result

The aggressive first pass contained 6,177 locked heterogeneous LUT changes. It
started at 85.950% validation accuracy and recovered monotonically to 86.600%
at 5,000 updates. The selected checkpoint used 1,280,000 training examples
(29.63 equivalent passes over the 43,200-example recovery subset) and 76.58 s
of GPU recovery time. All snapshots had zero locked-function violations.

| Updates | Validation accuracy | Validation disagreement | New unlocked LUT changes | AIG proxy |
|---:|---:|---:|---:|---:|
| 0 | 85.950% | 5.167% | 0 | 0.552375 |
| 1,000 | 86.133% | 5.033% | 7 | 0.552375 |
| 2,000 | 86.233% | 4.700% | 53 | 0.552125 |
| 3,000 | 86.283% | 4.817% | 153 | 0.551469 |
| 5,000 | 86.600% | 4.783% | 364 | 0.551250 |

Exact synthesis of the 5,000-update checkpoint produced 27,189 live gates,
87,568 ABC AND nodes, and 78 levels. Compared with unrecovered 10% Unit Tying
(86.967%, 30,405 live gates, 94,084 AND nodes, 79 levels), this point saves
10.58% live gates and 6.93% AND nodes but loses 0.367 accuracy points. Neither
point dominates the other.

The conservative first pass was already feasible at update 0: 87.233%
validation accuracy and 0.767% disagreement. Recovery did not improve it and
the training-holdout policy correctly selected update 0. By 5,000 updates it
had drifted to 86.917%, demonstrating that recovery is optional rather than an
automatic stage.

## Matched Unit-Tying recovery

The identical recovery implementation was applied to the 10% Unit-Tying
checkpoint. Its training holdout selected 3,000 updates. This checkpoint had
86.917% validation accuracy, 2.367% disagreement, 30,358 live gates, 93,990 ABC
AND nodes, and 79 levels. Thus the matched recovery slightly reduced hardware
but did not improve Unit Tying's original 86.967% validation accuracy.

The 30,000-update reference degraded to 86.267% validation accuracy and 7.433%
disagreement under this fixed optimizer/objective. This is a matched-method
ablation, not a reproduction of the Unit-Tying paper's own fine-tuning recipe,
and must not be described as evidence that Unit Tying inherently fails at
30,000 updates.

## Objective and stability ablations at most 3,000 updates

| Recovery variant | Selected updates | Validation accuracy | Disagreement | AIG proxy |
|---|---:|---:|---:|---:|
| Full locked hard-forward | 3,000 | 86.283% | 4.817% | 0.551469 |
| Label only | 1,000 | 86.117% | 5.017% | 0.552344 |
| No hardware ceiling | 1,000 | 86.067% | 4.983% | 0.552344 |
| Soft forward | 3,000 | 86.183% | 4.367% | 0.552344 |
| Unlocked first pass | 3,000 | 86.250% | 4.567% | 0.551844 |
| Fast crossing (gap 2, LR 0.005) | 500 | 86.083% | 4.750% | 0.552250 |

Forcing early hard changes was harmful: the fast-crossing run changed 2,389
unlocked rows by 3,000 updates and degraded its training holdout. The slower
run changed only 153 rows by the same point and recovered more accuracy. The
bottleneck is therefore stable task recovery, not simply rewrite count.

## Repeated resynthesis audit

The second pass locked the 2,694 accepted first-pass transformations, retained
1,390 new changes after exact repair, and kept the original trained model as
the cumulative teacher. It obtained 87.067% validation accuracy, 1.483%
validation disagreement, 29,812 live gates, 94,489 ABC AND nodes, and 78
levels. This is better than Unit Tying in accuracy, disagreement, live gates,
and levels, but has 405 more ABC AND nodes.

It is not a feasible final result: the untouched 1,200-example calibration
guard violated the worst-class accuracy-loss budget (2.632% measured versus
1.5% allowed), although the repair set, full calibration set, and validation
set passed. This negative result validates the three-way guard protocol and
prevents promotion of an overfit second-pass point.

## Current conclusion

The implementation demonstrates fast locked recovery, cumulative-teacher
repeat resynthesis, exact synthesis, and fair recovery curves. Seed 0 does not
yet establish superiority over Unit Tying. The strongest aggressive point is
a better-hardware/lower-accuracy Pareto point; the second pass nearly dominates
Unit Tying but fails the untouched guard. The next technical target is a
guard-robust pass-two objective or repair rule, followed by five seeds and a
harder dataset—not additional unconstrained rewrites.
