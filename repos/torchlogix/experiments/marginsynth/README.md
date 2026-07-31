# MarginSynth experiments

This directory is reserved for the implementation and evaluation of
**MarginSynth**, an accuracy-budgeted simplification method for hardened
differentiable logic gate networks (DLGNs).

MarginSynth starts after training. It edits the Boolean circuit that will
actually be deployed, rather than pruning the soft training model. Its central
question is:

> Can we use the classifier's decision margins to replace expensive pieces of a
> hardened DLGN with cheaper Boolean logic while controlling the resulting
> accuracy loss?

The original project brief is
[`ideas/date_ideas/margin_synth.md`](../../../../ideas/date_ideas/margin_synth.md).
The most important direct comparison is Lee et al.,
[*Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*](../../../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf),
ICML 2026.

This document defines the intended contribution and the implementation plan. It
does not claim that the method has already been implemented or validated.

## The idea in simple words

A trained DLGN becomes a circuit made from Boolean gates such as AND, OR, and
XOR. The circuit produces several votes for every class and selects the class
with the most evidence.

Some predictions are close:

```text
cat:  40 votes
dog:  39 votes
```

Other predictions have much more room:

```text
cat:  40 votes
dog:   8 votes
```

Changing one internal gate in the first example could easily change the final
answer. The same change may be harmless in the second example because the
winner has a large lead.

MarginSynth uses this available "room" to edit the deployed circuit. Possible
edits include:

- replacing a gate with constant 0 or constant 1;
- replacing a gate with one of its input wires;
- using an inverted input;
- changing a gate to a cheaper or more useful Boolean function; and
- deleting logic that becomes disconnected or unnecessary after an edit.

Every proposed edit is tested on a separate calibration set. An edit is kept
only when it stays inside a predeclared decision-flip or accuracy-loss budget.
The result is not one circuit, but a menu of operating points:

```text
original circuit
    ↓
slightly smaller, almost no measured accuracy loss
    ↓
smaller, small measured accuracy loss
    ↓
much smaller, larger measured accuracy loss
```

The user can select the circuit that fits the deployment budget.

### What MarginSynth does not promise

MarginSynth is approximate. A rewritten circuit will generally not compute the
same Boolean function as the original circuit for every possible input.

The method instead controls behavior on representative calibration data and
measures generalization on previously untouched test data. A rewrite must never
be described as functionally equivalent unless equivalence has separately been
proved for that rewrite.

## Why this is not just ordinary circuit simplification

Exact synthesis may simplify a circuit only when its input/output function is
preserved for every possible input. That is the correct rule for an arithmetic
unit or protocol controller, but it can be unnecessarily strict for a learned
classifier.

A classifier mainly needs to preserve its winning class on realistic inputs.
Its internal vote values may change substantially without changing that
decision. MarginSynth uses this classification-specific slack.

Exact simplification remains part of the flow:

```text
hard DLGN
    ↓ exact simplification
exactly smaller circuit
    ↓ MarginSynth
approximately smaller classifier
    ↓ exact clean-up
remove logic made redundant by the accepted approximations
```

## Technical formulation

### Starting point

Train and harden a rank-2 TorchLogix model, then export it to the editable
[`Circuit`](../../src/torchlogix/circuit.py) representation. Exact
`Circuit.simplify()` is applied before MarginSynth so that the approximate
method does not receive credit for ordinary constant propagation, duplicate
elimination, or dead-code removal.

For every calibration input \(x\), record:

- every live gate output;
- the original class-score vector \(s(x)\);
- the original winning class \(k(x)\);
- all winner-versus-challenger margins; and
- the ground-truth label, when running the label-aware accuracy-budget mode.

For challenger class \(c\), define the original pairwise margin

\[
m_{k,c}(x)=s_k(x)-s_c(x).
\]

The implementation must define and test the same tie-breaking rule used by
TorchLogix inference.

### Exact decision condition on a calibration sample

Let a rewrite change the score vector by \(\Delta s(x)\). The original winner
\(k\) remains the winner when

\[
\Delta s_c(x)-\Delta s_k(x) < m_{k,c}(x)
\qquad \text{for every } c\ne k,
\]

with strict or non-strict inequalities adjusted for the declared class-index
tie rule.

This condition is more targeted than minimizing the mean-squared difference
between all original and rewritten logits. Large score changes are acceptable
when they do not consume the complete winner margin.

Useful per-sample quantities are:

\[
m_r'(x)=\min_{c\ne k}
\left[
m_{k,c}(x)-\left(\Delta s_c(x)-\Delta s_k(x)\right)
\right],
\]

the remaining decision margin after rewrite \(r\), and

\[
\ell_{\text{margin}}(r,x)
=
\max\left(0,\rho-m_r'(x)\right),
\]

where \(\rho\ge 0\) is an optional safety reserve. Setting \(\rho=0\) penalizes
only boundary crossings; a positive reserve discourages rewrites that leave a
decision extremely close to changing.

The evaluation must always compute the rewritten scores and decisions exactly.
The margin expression is a ranking and budgeting tool, not a substitute for
hard-circuit simulation.

### Two operating modes

MarginSynth should support two related but separately reported modes.

**Decision-preserving mode**

- Does not need calibration labels.
- Measures disagreements with the original hardened circuit.
- At budget zero, accepts only rewrite sequences with no calibration decision
  changes.

**Accuracy-budgeted mode**

- Uses calibration labels.
- Constrains the cumulative calibration-accuracy and per-class-accuracy loss.
- Can retain a rewrite that changes an original prediction when the new
  prediction is still correct, or becomes correct.

The decision-preserving result is easier to interpret. The label-aware result
more directly optimizes classifier quality but has greater calibration-overfit
risk. They must not be mixed into one metric.

### Rewrite primitives

The minimum novelty-bearing implementation supports:

1. `constant-0` and `constant-1`;
2. `copy-a`, `copy-b`, `not-a`, and `not-b`;
3. replacement by another rank-2 Boolean function;
4. exact dead-cone removal after replacement; and
5. exact simplification of logic exposed by the accepted rewrite.

Constant replacement intentionally overlaps unit tying and is needed for a fair
controlled comparison. Nonconstant replacements are central to MarginSynth's
novelty. If constant replacement accounts for essentially all useful savings,
the method is too close to unit tying to support the intended claim.

After the single-gate MVP works, an optional extension may replace a small
two-to-four-gate window with a cheaper local implementation. Arbitrary global
resynthesis, reinforcement-learning search, and general Verilog rewriting are
outside the initial scope.

### Candidate evaluation

For every candidate rewrite:

1. identify its transitive fan-out cone;
2. resimulate only that cone on bit-packed calibration traces;
3. update affected class sums;
4. measure exact decision changes and accuracy changes;
5. estimate the exact clean-up enabled by the rewrite; and
6. reject the candidate without mutating the accepted circuit if any hard
   budget would be exceeded.

Candidate evaluation should produce:

- calibration decision-flip count;
- calibration accuracy change;
- per-class accuracy and disagreement changes;
- remaining-margin loss;
- live-gate reduction after exact local clean-up;
- logic-depth change;
- removed connections and fan-out changes; and
- affected-cone size and evaluation time.

A first ranking score may be

\[
Q(r)=
\frac{\Delta C(r)}
{\epsilon
 +\alpha L_{\text{margin}}(r)
 +\beta N_{\text{flip}}(r)
 +\gamma N_{\text{class-violations}}(r)},
\]

where \(\Delta C(r)\) is a circuit-cost reduction. Hard global and per-class
budgets are enforced separately; they must not be replaced by a soft score.

The initial cost model should combine live gates, logic depth, and connection
count. Measured compiled-C and later post-synthesis results determine whether
that proxy is useful.

### Sequential search

Candidate effects interact. Two individually safe rewrites may be unsafe when
combined. MarginSynth therefore uses a sequential search:

1. score an initial candidate pool;
2. evaluate the best candidates exactly;
3. accept one rewrite;
4. update traces and the circuit;
5. invalidate or recompute candidates whose cones overlap the changed region;
6. periodically perform a complete hard-circuit resimulation; and
7. save a Pareto snapshot whenever a requested budget is reached.

The initial search is deterministic greedy selection. Beam search or other
global search is considered only after the greedy baseline is correct and
scalable.

## Novelty boundary

### Honest overlap with two-stage unit tying

Both MarginSynth and two-stage unit tying:

- operate after initial DLGN training;
- use calibration examples;
- make approximate, potentially accuracy-changing modifications;
- can introduce constants that enable downstream simplification;
- target an accuracy-versus-hardware-cost trade-off; and
- may compare variants before and after fine-tuning.

Therefore, "post-training DLGN simplification" and "data-aware constant
replacement" are no longer novel claims.

### Intended differences from two-stage unit tying

| Dimension | Two-stage unit tying | MarginSynth |
|---|---|---|
| Representation being edited | Trained DLGN and its soft gate distributions | Hardened Boolean `Circuit` |
| Main rewrite | Tie a selected unit to constant 0 or 1 | Constants, bypasses, inversions, arbitrary gate substitutions, and optional small-cone replacements |
| Convolutional granularity | Structured root/output-channel tying | Internal gates and live Boolean cones, subject to export support |
| Preservation objective | Mean-squared distortion of all original logits | Winner-versus-challenger margin slack and explicit decision/accuracy budgets |
| Compression target | Predetermined tied-unit ratio | Best circuit cost inside a predetermined behavior-loss budget |
| Candidate scoring | Gauss--Newton screening of soft parameters | Exact hard-circuit affected-cone simulation |
| Interaction handling | Binary-split finite-difference refinement of an overshoot set | Recompute effects after every accepted circuit mutation |
| Fine-tuning | A 30,000-step recovery run is central to the strongest results | No-retraining frontier is central; matched fine-tuning is a secondary comparison |
| Hardware objective | Tying first, Vivado measurement afterward | Circuit-cost reduction enters candidate selection; saved snapshots are synthesized later |

The central hypothesis is:

> At the same post-synthesis hardware cost or the same allowed accuracy loss,
> decision-margin-aware hard-netlist rewriting is better than logit-preserving
> constant tying because it spends only classification-relevant slack and can
> choose nonconstant Boolean replacements.

### When MarginSynth would not be sufficiently different

The intended paper claim must be abandoned or substantially revised if:

- the implementation supports only constant replacement;
- margin is used only as another one-shot saliency score;
- the search selects a fixed percentage of units rather than enforcing a
  behavior-loss budget;
- all reported savings come from subsequent ordinary exact synthesis;
- fine-tuning, rather than rewrite selection, explains the gain; or
- MarginSynth does not beat a faithful unit-tying baseline under a matched
  protocol.

### Differences from other nearby work

**Exact logic synthesis**

Exact synthesis preserves every output for every input. MarginSynth permits
controlled classifier-level changes on representative data and must report the
resulting decision and accuracy errors.

**Conventional pruning and sensitivity pruning**

These methods generally remove weights, channels, or units from a trainable
model. MarginSynth edits a discrete Boolean netlist, considers alternative
Boolean functions rather than deletion alone, and evaluates every accepted
change using hardened execution.

**Activation/path saliency methods**

Activation frequency and path importance are useful baselines. They do not by
themselves optimize multi-class decision slack and circuit cost under a
cumulative accuracy budget.

**Logic Shrinkage and trainable fan-in methods**

Those methods change LUT fan-in during training. MarginSynth is a post-training
procedure intended to operate on any supported hardened rank-2 circuit without
retraining the original architecture.

**Generic approximate synthesis such as BLASYS**

Generic approximate synthesis controls Boolean output error without knowing
that outputs are grouped into class evidence and compared by `argmax`.
MarginSynth uses the classifier's score aggregation, class margins, and
per-class error constraints.

## Claims the project may make

If supported by experiments, acceptable claims are:

- MarginSynth produces a better hardened-accuracy versus synthesized-cost
  frontier than exact simplification, random rewriting, sensitivity ranking,
  and two-stage unit tying.
- Decision-margin ranking is more effective than logit-MSE ranking for
  classifier circuits.
- Nonconstant Boolean rewrites provide savings unavailable to
  constant-only tying.
- The estimated cost model correlates with compiled or synthesized cost.

The project must not claim:

- exact equivalence of approximate snapshots;
- formal accuracy guarantees outside the evaluated calibration distribution;
- lower FPGA area based only on unsynthesized gate counts; or
- superiority to published Vivado numbers when MarginSynth was measured using
  a different synthesis flow.

## Implementation plan

The implementation is deliberately staged. A later phase starts only after the
previous phase has tests and a positive mechanism check.

### Phase 0: freeze scope and protocols

1. Support rank-2 hardened circuits only.
2. Select one small dense MNIST checkpoint for correctness work.
3. Select one Fashion-MNIST checkpoint for the first meaningful frontier.
4. Use CIFAR-10 only after the search is correct and scalable.
5. Freeze train, validation, calibration, and test partitions.
6. Declare the class tie-breaking rule, calibration budgets, and circuit-cost
   proxy in configuration files.
7. Record all checkpoint, dataset, split, and source hashes.

Exit condition: a written configuration reproduces the untouched hardened
model and contains no test-set access during rewrite search.

### Phase 1: establish a trustworthy hard-circuit baseline

1. Load a trusted TorchLogix checkpoint.
2. Harden it and export it with `Circuit.from_model`.
3. Compare model and circuit class scores and predictions.
4. Run `Circuit.simplify()` and prove exact prediction equivalence.
5. Record live gates, depth, fan-out, connection count, serialized bytes, and
   compiled-C performance.
6. Add immutable circuit cloning and stable node identifiers.

Tests:

- model versus exported-circuit equivalence;
- original versus exact-simplified equivalence;
- serialization round trip;
- clone isolation; and
- stable IDs across non-destructive analysis.

Exit condition: every supported baseline vector produces identical class
scores or a documented exactly equivalent score representation.

### Phase 2: calibration traces and graph indices

1. Build fan-in and fan-out adjacency indices.
2. Record gate outputs over calibration examples using packed bit vectors.
3. Record class sums, winners, runner-ups, and pairwise class margins.
4. Add affected-cone discovery.
5. Add full resimulation from an arbitrary changed node set.
6. Measure trace memory and construction time.

Tests:

- packed traces match ordinary Boolean simulation;
- discovered fan-out cones match exhaustive reachability;
- recalculated class sums match full evaluation; and
- all classes are represented in the calibration partition.

Exit condition: changing a gate in a tiny test circuit and resimulating its
cone produces exactly the same outputs as complete simulation.

### Phase 3: safe circuit mutations

Implement rewrite objects without adding search logic:

1. constant 0 and 1;
2. copy A and B;
3. invert A and B;
4. alternative rank-2 gate functions;
5. mutation apply, undo, and replay;
6. exact dead-cone and local clean-up after a mutation; and
7. JSON rewrite-log serialization.

Each rewrite log entry contains:

- stable target ID;
- original and replacement functions;
- original input IDs;
- affected outputs/cone;
- predicted behavior and cost changes;
- measured behavior and cost changes; and
- accepted or rejected status and reason.

Tests:

- exhaustive truth tables for every rewrite;
- apply/undo restores the exact serialized circuit;
- rejected candidates cannot mutate accepted state;
- rewrite logs replay to an identical final circuit; and
- exact simplification after a rewrite preserves that rewritten function.

Exit condition: random rewrite sequences on tiny circuits agree with exhaustive
truth-table simulation.

### Phase 4: exact incremental candidate evaluation

1. Evaluate a rewrite using only its affected fan-out cone.
2. Update class sums and margins incrementally.
3. Compute decision, accuracy, and per-class changes.
4. Estimate live-gate, depth, and connection reductions.
5. Compare every incremental result with a full-circuit evaluation during
   correctness development.
6. Add fail-closed behavior: any mismatch rejects the candidate and stops the
   run with a diagnostic artifact.

Tests:

- incremental versus full outputs;
- incremental versus full class scores;
- tie-case handling;
- overlapping rewrite sequences;
- constant cascades; and
- candidates that affect more than one class group.

Exit condition: zero incremental/full mismatches across deterministic property
tests and randomized small circuits.

### Phase 5: greedy MarginSynth and Pareto snapshots

1. Generate a bounded candidate pool.
2. Implement the margin-aware score.
3. Enforce global disagreement, global accuracy, and per-class budgets.
4. Recompute candidates after every accepted mutation.
5. Run periodic full resimulation.
6. Save snapshots at calibration-loss budgets such as
   \(0,0.1,0.25,0.5,1.0,\) and \(2.0\) percentage points.
7. Export every snapshot to TorchLogix JSON, C, and Verilog.
8. Record search runtime, peak RAM, candidate counts, and rewrite composition.

The first scalability optimization should be lazy invalidation and local
candidate refresh. A beam search is not part of this phase.

Exit condition: the same checkpoint, configuration, and calibration seed
produce byte-identical rewrite logs and Pareto metadata.

### Phase 6: controlled baselines and novelty ablations

Use identical checkpoints, calibration data, candidate budgets, and evaluation
code for:

1. random candidate order;
2. activation-frequency ranking;
3. first-order sensitivity ranking;
4. exact `Circuit.simplify()` only;
5. uniform smaller/static DLGNs;
6. constant-only margin rewriting;
7. logit-MSE scoring with the MarginSynth candidate set;
8. full MarginSynth without the safety reserve;
9. full MarginSynth without per-class budgets; and
10. complete MarginSynth.

Implement or faithfully reproduce two-stage unit tying on the same starting
checkpoints:

- its constant-only rewrite space;
- Gauss--Newton Stage-A screening;
- Binary Split refinement;
- tied-ratio operating points;
- the no-fine-tuning results; and
- its fixed-tie 30,000-step fine-tuning protocol.

For fairness, report two comparisons:

1. all methods without recovery fine-tuning; and
2. unit tying and MarginSynth with an identical recovery budget.

The most important factorial ablation is:

| Objective | Rewrites |
|---|---|
| logit MSE | constants only |
| decision margin | constants only |
| logit MSE | full rewrite set |
| decision margin | full rewrite set |

This separates gains from the objective and gains from the richer synthesis
space.

Exit condition: MarginSynth's advantage, if any, can be attributed separately
to decision margins, sequential exact evaluation, and nonconstant rewrites.

### Phase 7: compiled measurements

For every saved snapshot:

1. compile generated C with a frozen compiler and flags;
2. verify compiled predictions against the snapshot;
3. measure batch-1 mean, median, p95, and worst-case latency;
4. measure packed throughput separately;
5. record binary size and compile time; and
6. correlate these results with the internal cost proxy.

Do not infer speedup from removed-gate count alone.

Exit condition: at least one useful accuracy-budget point shows a repeatable
compiled benefit or the paper explicitly narrows its claim to circuit cost.

### Phase 8: Yosys/ABC synthesis after installation

Yosys and ABC are not required for Phases 0--7. Once Yosys is installed, use it
as the common Verilog front end and synthesis driver; allow Yosys to invoke ABC
for exact logic optimization and LUT mapping.

Synthesize only:

- the original circuit;
- the exact-simplified circuit;
- saved MarginSynth Pareto points;
- matched unit-tying points; and
- the strongest controlled baselines.

Use one frozen script, tool version, top module, target LUT size or cell
library, and timing constraints for every method. Never compare an absolute
Yosys LUT count directly with the unit-tying paper's Vivado count. Direct
comparison requires running both methods through the same local flow.

Record:

- mapped LUT or cell count;
- logic levels or critical-path estimate;
- wires/connections;
- synthesis runtime and peak memory;
- tool and script hashes; and
- equivalence of each synthesized snapshot to its own pre-synthesis rewritten
  circuit.

Combinational equivalence must compare a synthesized snapshot with that same
snapshot. An approximate MarginSynth snapshot is not expected to be equivalent
to the original model.

Exit condition: the main internal cost proxy has a measured relationship with
post-synthesis cost, and all methods have same-flow results.

### Phase 9: final experimental protocol

1. Use MNIST for exhaustive correctness and debugging only.
2. Use Fashion-MNIST for the first complete accuracy/cost frontier.
3. Use CIFAR-10 for the main scale result after export and search scalability
   are established.
4. Evaluate two starting gate budgets where feasible.
5. Use three paired seeds for pilots and five for the central comparison.
6. Freeze all search settings before held-out test evaluation.
7. Run the test set only on preselected Pareto snapshots.
8. Report failed runs and calibration sizes.

Main plots:

- hardened test accuracy versus post-synthesis LUT/cell count;
- hardened test accuracy versus live gates;
- accuracy versus compiled batch-1 latency;
- calibration-predicted loss versus held-out test loss; and
- accepted rewrite type versus cumulative cost reduction.

## Suggested implementation layout

The exact file boundaries may change as the existing `Circuit` API is audited,
but the initial organization should be:

```text
experiments/marginsynth/
├── README.md
├── configs/
├── trace.py
├── rewrites.py
├── incremental.py
├── search.py
├── evaluate.py
├── benchmark.py
├── synthesize.py
├── summarize.py
└── baselines/
    └── unit_tying.py
```

Generic, well-tested graph operations should eventually move into
`src/torchlogix/circuit.py` or a reusable circuit submodule. Experiment
orchestration, scoring policies, configurations, and paper-specific baselines
should remain under `experiments/marginsynth`.

Tests should begin in `tests/test_marginsynth.py` and may be split by component
when they become large.

## Implemented reproducible workflow

The current implementation keeps development, validation selection, protocol
freeze, and test evaluation as separate commands:

```bash
# Materialize and execute the frozen objective/rewrite-space ablations.
venv/bin/python experiments/marginsynth/run_ablation_matrix.py RUN_DIR \
  --matrix experiments/marginsynth/configs/v2_ablation_matrix_fashion_seed0.json \
  --run

# Run or resume paired training/export/search/baseline/synthesis stages.
venv/bin/python experiments/marginsynth/run_paired_study.py \
  --manifest experiments/marginsynth/configs/paired_fashion/paired_study_manifest.json \
  --cost-model RUN_DIR/synth_cost_model.json

# Aggregate validation results only; this command cannot load test data.
venv/bin/python experiments/marginsynth/summarize_paired_study.py \
  --paired-manifest experiments/marginsynth/configs/paired_fashion/paired_study_manifest.json \
  --accuracy-budget 0.0 --unit-tying-ratio 0.1 \
  --output experiments/marginsynth/results/paired_fashion_validation_summary.json

# Freeze hashes only after all five replay, validation, and synthesis stages pass.
venv/bin/python experiments/marginsynth/freeze_protocol.py \
  --paired-manifest experiments/marginsynth/configs/paired_fashion/paired_study_manifest.json \
  --method-config experiments/marginsynth/configs/v2_fashion_frozen75.json \
  --operating-point 0.0 --unit-tying-ratio 0.1 \
  --output experiments/marginsynth/configs/paired_fashion/protocol_freeze.json

# This is the only MarginSynth command authorized to open the held-out test set.
venv/bin/python experiments/marginsynth/evaluate_frozen_test.py \
  --freeze experiments/marginsynth/configs/paired_fashion/protocol_freeze.json \
  --dataset-path /tmp/torchlogix-datasets \
  --output experiments/marginsynth/results/paired_fashion_test_summary.json

# Aggregate the immutable raw test result without reopening the dataset.
venv/bin/python experiments/marginsynth/summarize_frozen_test.py \
  --input experiments/marginsynth/results/paired_fashion_test_summary.json \
  --output experiments/marginsynth/results/paired_fashion_test_aggregate.json
```

`freeze_protocol.py` records and hashes every selected checkpoint, resolved
configuration, exact baseline, replay verification, validation result,
synthesis result, MarginSynth circuit, and optional Two-Stage circuit.
`evaluate_frozen_test.py` verifies all hashes before it requests even the first
test example. Failed or incomplete seeds prevent protocol freeze.

## Required measurements

Every Pareto snapshot should report:

- original, calibration, validation, and held-out test accuracy;
- macro-F1 and per-class accuracy;
- disagreement with the original hardened circuit;
- remaining-margin distribution;
- live gates and removed gates by rewrite type;
- logic depth, fan-out, connection count, and serialized bytes;
- exact-simplification contribution after approximate edits;
- compiled-C latency, throughput, binary size, and compile time;
- later Yosys/ABC LUT or cell count and logic levels;
- search time, candidate evaluations, and peak RAM; and
- whether any recovery fine-tuning was used.

Learned gate functions and spatially replicated convolutional gate
applications must be reported separately.

## Go/no-go criteria

Correctness gates:

1. Exported and hardened TorchLogix predictions agree.
2. Exact simplification preserves the hardened circuit.
3. Incremental and full rewrite simulation always agree.
4. Replayed logs reproduce byte-identical circuits and predictions.
5. Test examples and labels are unavailable to search and hyperparameter
   selection.

Research gates:

1. At no more than 0.5 percentage-point held-out accuracy loss, MarginSynth
   should obtain at least 10--15% lower same-flow post-synthesis cost than
   two-stage unit tying; or it should obtain at least 0.5 percentage-point
   higher accuracy at a matched synthesized cost.
2. The central advantage must hold on two datasets or model budgets.
3. Decision-margin scoring must beat logit-MSE scoring with the same candidate
   set.
4. Nonconstant rewrites must provide a material incremental benefit over
   constant-only rewriting.
5. Calibration-predicted loss must track held-out loss well enough to select a
   deployment point.
6. A claimed runtime benefit must be measured, not inferred.

If MarginSynth only marginally improves the selection of constant ties, it is
not sufficiently different from the ICML 2026 work and should not proceed as a
standalone DATE paper.

## Definition of done

MarginSynth is ready for a paper only when:

- hard-circuit mutation and incremental simulation are thoroughly tested;
- every result can be reproduced from a checkpoint, configuration, and rewrite
  log;
- exact simplification and two-stage unit tying are same-pipeline baselines;
- objective and rewrite-space ablations isolate the source of improvement;
- several predeclared accuracy budgets produce held-out Pareto points;
- compiled measurements support any software-runtime claim;
- same-flow synthesis supports any FPGA/ASIC resource claim; and
- five paired central seeds support the claimed advantage.
