# MarginSynth paper evaluation reference

This document freezes the intended implementation sequence, comparison methods,
metrics, and fairness rules for the MarginSynth DATE study. It is a compact
reference for development and experiment decisions. The complete technical
description remains in [README.md](README.md), and the original project brief is
[margin_synth.md](../../../../ideas/date_ideas/margin_synth.md).

## Method in one paragraph

The current primary MarginSynth hypothesis is post-training whole-circuit LUT
resynthesis for hardened rank-2 DLGNs. It jointly optimizes every eligible LUT
over all 16 Boolean functions using winner--runner margins, group robustness,
and a hardware proxy; a disjoint calibration subset then exactly repairs hard
budget violations before identical Yosys/ABC measurement. It does not use a
Unit-Tying warm start, Gauss--Newton shortlist, Binary Split, or fixed tie
quota. The earlier greedy circuit-rewrite and Unit-Tying + Margin Refinement
pipelines remain controlled baselines.

MarginSynth is approximate. A rewritten circuit must not be described as
functionally equivalent to the original unless that equivalence has separately
been proved.

## Development sequence

1. Freeze rank-2 architectures, checkpoints, seeds, dataset partitions,
   tie-breaking rules, calibration sizes, loss budgets, and the circuit-cost
   proxy.
2. Harden each checkpoint, export it to TorchLogix `Circuit`, and prove model,
   exported-circuit, serialized-circuit, and compiled-C agreement.
3. Apply exact `Circuit.simplify()` and record the untouched exact baseline.
4. Build fan-in/fan-out indices and bit-packed calibration traces containing
   gate values, class scores, winners, challengers, and decision margins.
5. Implement constant, bypass, inversion, and alternative rank-2 gate rewrites,
   including apply, undo, serialization, and deterministic replay.
6. Implement incremental affected-cone simulation and require exact agreement
   with full simulation throughout correctness development.
7. Implement deterministic greedy margin-guided search with global accuracy,
   disagreement, and per-class budgets. Recompute interacting candidates after
   every accepted mutation.
8. Save circuits at predeclared calibration-loss budgets such as
   \(0, 0.1, 0.25, 0.5, 1.0,\) and \(2.0\) percentage points.
9. Run controlled ablations and reproduce the paper baselines below.
10. Compile and benchmark every selected circuit, then synthesize all methods
    through one frozen Yosys/ABC flow.
11. Freeze the chosen operating points before evaluating the held-out test set.
12. Run paired multi-seed experiments and generate tables and plots directly
    from saved raw results.

The search does not enumerate combinations of gate replacements. It evaluates a
bounded candidate pool, accepts one rewrite at a time, and refreshes only
candidates affected by the changed circuit region.

## Default DLGN references

These papers define the unmodified starting architectures. They are reference
points rather than competing simplification algorithms and therefore do not
count toward the five comparison methods below.

| Reference | Role |
|---|---|
| Petersen et al., [*Deep Differentiable Logic Gate Networks*](../../../../pdfs/deep_differentiable_logic_gate_networks.pdf), NeurIPS 2022 | Default dense DLGN reference for MNIST and Fashion-MNIST |
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Default convolutional DLGN reference for CIFAR-10 |

For every starting model, report both the untouched hardened circuit and its
function-preserving exact-synthesis result.

## External paper comparison methods

The central study uses the following five methods from recent papers.
MarginSynth is evaluated alongside them but is not counted in this list.

| Paper method | Year | Required comparison |
|---|---:|---|
| Lee et al., [*Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*](../../../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf) | 2026 | Faithfully reproduce constant-only unit tying, Gauss--Newton screening, Binary Split refinement, tied-ratio points, and recovery fine-tuning. This is the mandatory direct post-training competitor. |
| Yousefi et al., [*Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks*](../../../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf) | 2025 | Train the hard-forward/Gumbel formulation, harden it, and apply the same exact synthesis flow. This tests MarginSynth against a recent model designed to avoid unused gates and the discretization gap. |
| Mommen et al., [*A Method for Optimizing Connections in Differentiable Logic Gate Networks*](../../../../pdfs/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.pdf) | 2025 | Reproduce partial learned connectivity at matched final gate budgets, especially on MNIST and Fashion-MNIST. |
| Fojcik et al., [*LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment*](../../../../pdfs/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.pdf) | 2025/2026 | Reproduce the relevant Top-\(K\) compact configurations at matched datasets and hardware budgets. |
| Fieldhouse and Tang, [*Silicon-Aware Neural Networks*](../../../../pdfs/silicon_aware_neural_networks.pdf) | 2026 | Reproduce the expected-area training penalty with one declared cell-cost library and compare final accuracy versus same-flow mapped cost. |

### How the comparisons must be presented

Use two separate groups rather than mixing incompatible claims:

1. **Same-checkpoint post-training comparison:** MarginSynth versus Two-Stage
   Unit Tying. Both methods start from the identical trained checkpoint.
2. **End-to-end accuracy--hardware-cost comparison:** MarginSynth versus Mind
   the Gap, optimized connections, LILogic Net, and Silicon-Aware Neural
   Networks. These are training-time alternatives and need not share the same
   starting checkpoint, but they must use matched data, final cost reporting,
   and synthesis conditions.

Numbers copied from papers may be shown only in a clearly labelled related-work
table. Only locally reproduced, same-pipeline results belong in statistical
comparisons or superiority claims.

Random candidate ordering, exact synthesis, constant-only rewriting, and
objective ablations remain required controls. They are not presented as
external paper-method competitors.

## Dataset coverage and selection

The selected papers do not all evaluate the same datasets. Their reported
coverage is:

| Paper | Reported datasets relevant to this study | Main emphasis |
|---|---|---|
| Two-Stage Unit Tying | MNIST and CIFAR-10 | CIFAR-10 is the main hardware and ablation result; MNIST demonstrates aggressive tying with little recovered accuracy loss. |
| Mind the Gap | CIFAR-10 and CIFAR-100 | Both datasets are central to its discretization-gap study. |
| A Method for Optimizing Connections | Yin-Yang, MNIST, and Fashion-MNIST | MNIST supplies the headline gate-reduction result; Fashion-MNIST provides the more difficult dense-image comparison. |
| LILogic Net | MNIST, Fashion-MNIST, and CIFAR-10 | All three are relevant; the paper also reports an FPGA throughput result for Fashion-MNIST. |
| Silicon-Aware Neural Networks | MNIST and CIFAR-10 | MNIST is the physical-design demonstration; CIFAR-10 evaluates the area-aware training trade-off. |
| Deep Differentiable Logic Gate Networks | MONK, Adult, Breast Cancer, MNIST, and CIFAR-10 | MNIST and CIFAR-10 are the relevant default dense references. |
| Convolutional Differentiable Logic Gate Networks | CIFAR-10 and MNIST | CIFAR-10 is the primary architectural and hardware result. |

This overlap determines the MarginSynth dataset order:

1. **MNIST -- correctness and debugging.** Use a small dense rank-2 model for
   exhaustive rewrite tests, exported-circuit equivalence, incremental/full
   simulation checks, and the first end-to-end smoke frontier. MNIST is not the
   central DATE claim.
2. **CIFAR-10 -- central paper benchmark.** It has the strongest comparison
   overlap: Two-Stage Unit Tying, Mind the Gap, LILogic Net, Silicon-Aware
   Neural Networks, and Convolutional DLGN all report it. The main
   same-checkpoint post-training comparison and the central five-seed result
   must use CIFAR-10.
3. **Fashion-MNIST -- secondary dense-circuit benchmark.** It provides a
   simpler export and synthesis path than the convolutional model and enables
   direct comparison with the optimized-connections and LILogic Net papers.
4. **CIFAR-100 -- optional scalability extension.** Add it only after the
   CIFAR-10 mechanism and synthesis flow succeed. Among the five selected
   comparison papers, only Mind the Gap reports CIFAR-100, so it cannot replace
   CIFAR-10 as the central benchmark without changing the comparison set.

Each dataset must use disjoint training, validation, calibration, and held-out
test partitions. The exact split sizes and hashes are frozen before baseline
training. Test labels remain inaccessible during checkpoint selection, rewrite
search, budget selection, and Pareto-point selection.

The archived dense CIFAR standard-random result directories predate this rule:
they used no independent calibration split, and their git-ignored checkpoints
are not present in the repository. They must not be used by selecting rewrites
on formerly trained examples. The MarginSynth dense study regenerates the same
fixed-random 48K- and 512K-gate architectures with split seed 2027 and a 10%
calibration partition excluded from training and validation. The original
configs and run summaries remain hashed provenance references; the regenerated
checkpoint is the common source for exact simplification, Unit Tying, and all
post-training controls.

### Conditional CIFAR-100 comparison set

If CIFAR-100 is promoted from an optional extension to a central result, revise
the external comparison set rather than presenting unsupported cross-paper
numbers. Add:

- Rüttgers et al.,
  [*Light Differentiable Logic Gate Networks*](../../../../pdfs/light_differentiable_logic_gate_networks.pdf),
  2025; and
- Bührer et al.,
  [*BitLogic: A Framework for Gradient-Based LUT-Native Neural Networks*](../../../../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf),
  2026.

Both report CIFAR-100. They replace non-CIFAR-100 comparison papers in that
dataset's central table; they are not added on top of an unlimited baseline
list. Two-Stage Unit Tying may also be adapted and run on CIFAR-100, but such a
result must be labelled as a local extension of the published method because
the paper itself reports only MNIST and CIFAR-10.

## Metrics for the paper

### Primary classifier metrics

- hardened top-1 accuracy;
- accuracy change from the untouched hardened circuit, in percentage points;
- macro-F1;
- per-class accuracy and worst-class accuracy loss; and
- soft-to-hard accuracy gap for training-time baseline methods.

### Primary circuit and hardware metrics

- live gate count and percentage reduction;
- mapped FPGA LUT count or mapped standard-cell area;
- logic depth, mapped logic levels, or critical-path estimate;
- wire/connection count;
- batch-1 mean, median, and p95 compiled-C latency;
- packed-inference throughput; and
- compiled binary or serialized circuit size.

FPGA LUT and ASIC cell-area results are different targets and must never be
merged into one cost metric. A hardware claim must use measured same-flow
results rather than unsynthesized gate count alone.

### Trade-off metrics

The main result is a hardened-accuracy versus synthesized-cost Pareto frontier.
Report:

- cost reduction at matched accuracy-loss budgets;
- accuracy at matched synthesized costs;
- which method dominates at each declared operating point;
- Pareto hypervolume using one predeclared reference point; and
- compiled latency versus hardened accuracy as a second frontier.

The central operating point is no more than \(0.5\) percentage-point held-out
accuracy loss. Also report the full frontier so that one selected point cannot
hide unfavorable trade-offs.

### Behavioral and calibration metrics

- prediction disagreement with the original hardened circuit;
- calibration-predicted accuracy loss versus held-out test loss;
- global and per-class decision-flip rates;
- original and remaining decision-margin distributions;
- accepted rewrite counts by type;
- exact-simplification contribution after approximate rewrites; and
- nonconstant-rewrite contribution beyond constant-only tying.

Decision-preserving and label-aware accuracy-budgeted modes must be reported
separately.

### Method overhead

- rewrite-search wall time and peak RAM;
- candidates generated, evaluated, accepted, and rejected;
- affected-cone sizes;
- training or recovery-fine-tuning time;
- compilation and synthesis time; and
- peak GPU memory for methods that require training.

## Fairness and reproducibility rules

1. Use disjoint train, validation, calibration, and test partitions.
2. Test labels must be unavailable to rewrite selection and hyperparameter
   tuning.
3. Use identical starting checkpoints for MarginSynth and Two-Stage Unit Tying.
4. Report both methods without recovery fine-tuning.
5. In a separate experiment, give both methods the same recovery optimizer,
   data, iteration count, and compute budget.
6. Match dataset encoding, data augmentation, class aggregation, and
   tie-breaking rules.
7. Use one Yosys/ABC version, script, target LUT size or cell library, timing
   constraint, and top module for every method in a hardware table.
8. Verify each synthesized circuit against its own pre-synthesis circuit.
   Approximate circuits are not expected to be equivalent to the original.
9. Use three paired seeds for pilots and at least five paired seeds for the
   central result.
10. Report mean, standard deviation, and a paired confidence interval or paired
    significance test.
11. Save configurations, source revision, package environment, checkpoints,
    rewrite logs, synthesis scripts, tool versions, and raw per-seed results.
12. Report failed or diverged runs rather than silently replacing them.

## Required plots and tables

1. Hardened accuracy versus same-flow mapped LUT or cell cost.
2. Hardened accuracy versus live gate count.
3. Compiled batch-1 latency versus hardened accuracy.
4. Calibration-predicted versus held-out accuracy loss.
5. Central operating-point table with MarginSynth and every external paper
   method applicable to that dataset. Clearly label locally adapted methods
   whose original papers did not report that dataset.
6. MarginSynth ablation table separating the scoring objective and rewrite
   space:

| Objective | Rewrite space |
|---|---|
| logit MSE | constants only |
| decision margin | constants only |
| logit MSE | full rewrite set |
| decision margin | full rewrite set |

## Success and stop criteria

The preferred result is that, at no more than \(0.5\) percentage-point held-out
accuracy loss, MarginSynth obtains at least 10--15% lower same-flow synthesized
cost than Two-Stage Unit Tying. An alternative success is at least
\(0.5\) percentage-point higher accuracy at a matched synthesized cost.

MarginSynth does not need to dominate unit tying at every point. A defensible
paper may instead establish a distinct central regime, such as a better
no-fine-tuning frontier, lower search cost, or better stringent-loss behavior.

If Two-Stage Unit Tying Pareto-dominates MarginSynth under every matched
protocol, improvements over the four training-time papers may still support an
end-to-end hardware-efficiency result, but the paper must not claim to be the
best post-training DLGN simplification method. If MarginSynth also fails to beat
those recent alternatives at matched synthesized cost, it should not proceed as
a standalone DATE paper.
