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
to earlier seed-0 development checkpoints synthesized by the identical
Yosys/ABC flow. Calibration loads no image data and cannot access validation
or test predictions. Both in-sample and leave-one-method-out correlations are
recorded. Constant propagation is additive across reconvergent paths and is
therefore explicitly treated as an estimator, not an exact node count.

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
