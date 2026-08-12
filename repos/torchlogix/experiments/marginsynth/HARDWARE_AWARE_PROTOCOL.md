# Hardware-Aware MarginSynth Protocol

## Purpose

This protocol addresses the main dense-CIFAR weakness of MarginSynth: it
preserves predictions better than 10% Unit Tying, but removes fewer live gates
and ABC nodes and takes longer to apply.

The study is a development/validation protocol. The official CIFAR-10 test set
stays sealed until the method, all hyperparameters, and the seed-transfer rule
have been frozen.

## Method changes

### 1. Structural hardware-gain estimator

Each learned LUT replacement receives four static features:

- local two-input AIG operation reduction;
- downstream operation reduction predicted from exact Boolean cofactors when
  the replacement is constant;
- `log(1 + direct fan-out)`; and
- `log(1 + fixed-topology path multiplicity to class reductions)`.

The coefficients are fitted with deterministic nonnegative ridge regression
to earlier seed-0 MarginSynth/control development checkpoints synthesized by
the identical Yosys/ABC flow. Unit Tying is excluded from coefficient fitting
and used only as a held-out proxy-validation point, so the proposed method is
not tuned to its main baseline. Calibration loads no image data and cannot
access validation or test predictions. In-sample, leave-one-fit-method-out,
and held-out correlations/errors are recorded. Constant propagation is
additive across reconvergent paths and is therefore explicitly treated as an
estimator, not an exact node count.

### 2. Focused action prior

All 16 rank-2 Boolean functions remain legal. The original function is never
penalized. A candidate binary function different from the original receives a
fixed two-AIG-unit penalty. Constants and routing/inversion functions do not.
This focuses probability and repair priority on the action families that
survived all previous selected runs without imposing the hard restricted mask
that failed the Fashion-MNIST guard.

### 3. Hardware and class-aware repair

Three new seed-0 components retain the frozen trial-28 optimization settings:

- `focused_control`: focused action prior with the original repair order;
- `hardware_aware`: focused prior plus structural hardware-gain repair order;
- `class_hardware_aware`: focused prior plus a combined repair order consisting
  of 75% normalized hardware rank and 25% normalized worst class/fold activity
  risk rank.

The existing current MarginSynth result is the unchanged reference. No
Bayesian retuning is permitted in this component study.

### 4. Guard-feasible snapshot selection

Source, first-pass, and second-pass snapshots are candidates. A snapshot must
pass the repair, complete-calibration, and untouched-guard budgets. Among
feasible candidates, selection maximizes the frozen estimated hardware gain;
ties favor more retained edits and then the later pass. Validation and test
are never consulted. Consequently, a failing second pass deterministically
falls back to a feasible first pass or, if necessary, the source checkpoint.

When the source is selected, its previously passed exact export and synthesis
record are reused only after byte-identical checkpoint, calibration-split,
tool-flow, and test-sealing checks. This avoids recompiling and resynthesizing
an unchanged 512K-gate source while retaining the original artifact hashes and
an explicit reuse-provenance record.

## Seed-0 selection and transfer

Every new seed-0 component is exported, exactly simplified, checked with
compiled C, and synthesized with identical Yosys/ABC commands. Among
guard-feasible MarginSynth variants, the selected method minimizes exact ABC
AND nodes. Ties use, in order:

1. worst-class guard accuracy loss;
2. global guard accuracy loss;
3. complete method time; and
4. component name.

Validation accuracy is reported but excluded from selection. Transfer is
authorized only when a new component beats the current reference under this
rule and the structural estimator has Spearman correlation at least 0.5 over
the predeclared seed-0 validation points. The complete component overlay,
hardware-model hash, protocol hash, and selection result are frozen before
unchanged execution on seeds 1 and 2.

## Reported metrics

- global and worst-class validation accuracy loss;
- global and worst-class teacher disagreement;
- repair/calibration/guard feasibility and snapshot fallback rate;
- live gates and reduction from exact simplification;
- ABC AND nodes, node reduction, and levels;
- Yosys generic cell count;
- explicitly labeled SkyWater operation-area proxy;
- optimization and complete method time;
- peak GPU allocation; and
- structural-estimator Pearson/Spearman correlation.

A true mapped standard-cell area is reported only if a common characterized
Liberty library is available. The current machine has no such library, so
`mapped_area` remains null rather than being conflated with the operation-area
proxy.

## Data boundary

- Training: CIFAR-10 training partition only.
- Source checkpoint selection: validation partition only.
- MarginSynth optimization, repair, guard, and snapshot selection: calibration
  partition only.
- Seed-0 method selection: calibration guards plus synthesis metrics only.
- Test: sealed until the full multi-seed method is frozen.
- Connectivity: fixed standard-random only; Coverage connectivity is excluded
  from the primary experiment.

## Executed seed-0 outcome (2026-08-12)

The version-2 protocol completed on the 512K-gate CIFAR-10 seed-0 source. The
structural estimator was fitted without loading any dataset and without using
Unit Tying as a fit point. It obtained in-sample Pearson/Spearman correlations
of 0.988/1.000, leave-one-fit-method-out correlations of 0.914/0.800, and a
1,817.6-node absolute error when Unit Tying was used only as a held-out point.
On the three new predeclared seed-0 observations, estimated gain versus exact
ABC reduction had Pearson 0.993 and Spearman 1.000.

| Method | Selected snapshot | Guard feasible | Validation accuracy | Disagreement | Live gates | ABC nodes | Levels | Area-proxy reduction | Method time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact simplification | source | yes | 54.64% | 0.00% | 389,970 | 1,484,154 | 105 | 0.000% | 0.00 s |
| 10% Unit Tying | tied | unconstrained baseline | 54.16% | 5.60% | 353,549 | 1,419,811 | 106 | 9.206% | 10.86 s |
| Prior current MarginSynth | second | yes | 54.40% | 2.28% | 380,330 | 1,472,970 | 105 | 2.315% | 210.41 s |
| Focused control | source fallback | yes | 54.64% | 0.00% | 389,970 | 1,484,154 | 105 | 0.000% | 177.34 s |
| Hardware-aware | second | yes | 54.52% | 2.30% | 388,326 | 1,480,257 | 104 | 0.470% | 176.51 s |
| Class + hardware-aware | second | yes | 54.68% | 1.14% | 384,998 | 1,475,742 | 105 | 1.299% | 232.48 s |

The focused control's first and second passes both failed the untouched guard,
so the predeclared fallback correctly returned the source snapshot. The pure
hardware method retained 314 cumulative edits, all constants. The combined
method retained 2,075 edits (1,643 constants and 432 routing/inversion edits).
No selected new result used an alternative binary LUT.

The estimator validation threshold passed, but neither new component beat the
prior current result's 1,472,970 exact ABC nodes. The frozen selection therefore
returned `current_reference`, and the freeze record is intentionally marked
`not-frozen`. Seed-1/2 transfer and official-test evaluation were not run. The
transfer protocol generator independently rejects this record with `transfer
freeze record is not frozen`.

This is a useful negative result: operation-aware ranking predicts the relative
exact gains well, but it does not by itself generate enough guard-compatible
edits. The next method change should improve joint candidate generation and
class-aware repair under the hardware rank; another fit of the estimator to
these same seed-0 observations would be post-hoc and is not authorized by this
protocol.
