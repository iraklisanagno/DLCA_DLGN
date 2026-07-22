# MarginSynth: Accuracy-Budgeted Circuit Simplification

**Source concept:** MarginSynth in [july_20.md](../july_20.md).

## Narrative

A trained DLGN becomes a real Boolean circuit. Conventional circuit simplification is conservative: it removes only logic that can be proven redundant for every possible input. A classifier, however, does not need to preserve every internal bit on every artificial input. It needs to preserve useful decisions on realistic data.

MarginSynth treats simplification as a controlled engineering trade. When the winning class is far ahead of the alternatives, a small internal change may not alter the decision. The method uses those decision margins to identify gates that can be replaced, bypassed, or removed with low risk. It then produces a menu of circuits: the original accurate circuit and progressively smaller circuits with a measured accuracy cost.

The story for a nontechnical reader is that MarginSynth edits a decision process where it has "room to spare." It does not merely compress weights before deployment; it operates on the hardened circuit that will actually execute.

## Research question and claim

**Question:** Can class-margin-aware rewrites reduce a hardened DLGN more effectively than exact simplification or unguided pruning at the same accuracy loss?

**Target claim:** On a held-out calibration set, margin-guided rewrites construct a better accuracy--circuit-cost Pareto frontier than exact simplification, random rewrites, and activation-only sensitivity pruning.

The proposed method is approximate. The paper must never describe its rewrites as functionally equivalent unless an individual rewrite has been proven exact.

## Scope

The minimum viable method supports exported rank-2 circuits and four rewrite types:

1. replace a gate by a constant;
2. replace a gate by either input, including inversion when available;
3. change a gate's Boolean function;
4. delete the now-dead fan-in cone after a replacement.

Do not begin with arbitrary Verilog parsing, SAT-guided global synthesis, rank-4/6 export, or a large reinforcement-learning search. These can become future work. The first paper should determine whether classifier-aware local circuit editing is useful.

## Technical design

Export a trained hard DLGN into TorchLogix's editable `Circuit` graph. For each calibration sample, record the gate values, output class scores, winner, runner-up, and winner margin:

$$
m(x) = y_{(1)}(x) - y_{(2)}(x).
$$

For a candidate rewrite $r$, incrementally simulate only the affected fan-out cone and compute:

- decision-flip count;
- reduction in winner margin;
- class-conditional flip count;
- circuit-cost reduction.

Rank candidates with a simple score such as:

$$
Q(r) = \frac{\Delta C(r)}{\epsilon + \Delta L_{\mathrm{margin}}(r) + \lambda N_{\mathrm{flip}}(r)},
$$

where $\Delta C$ is the estimated cost reduction. Accept the best candidate that keeps the cumulative calibration loss within a predeclared budget. Recompute candidate effects locally after every accepted rewrite and periodically run a full-circuit simulation to prevent accumulated approximation errors.

Produce a Pareto trace by saving circuit snapshots at several calibration budgets, for example 0, 0.1, 0.25, 0.5, 1.0, and 2.0 percentage points of allowed hard-accuracy loss. The test set is evaluated once after these snapshots have been selected using validation/calibration data.

## Implementation plan

### Milestone 1: Reliable circuit baseline

1. Reproduce and harden one dense model and one small convolutional-front-end model.
2. Export each supported model into `Circuit` and prove output equivalence on validation vectors.
3. Run existing exact simplification and record gate count, depth, fan-out, compiled-C latency, and accuracy.
4. Add immutable circuit copies and stable gate identifiers so a failed candidate cannot corrupt the accepted circuit.

### Milestone 2: Trace and candidate engine

1. Add bit-packed calibration traces for gate outputs and class sums.
2. Build fan-in/fan-out indices and affected-cone discovery.
3. Generate constant, bypass, inversion, and gate-function candidates.
4. Implement an exact incremental simulator and compare it with full simulation after every rewrite in tiny tests.

### Milestone 3: Search and Pareto export

1. Implement greedy margin-guided ranking with one cost model.
2. Enforce global and per-class flip budgets.
3. Run dead-code and duplicate-gate elimination after accepted batches.
4. Save every Pareto point as JSON, C, and Verilog plus a rewrite log.

### Milestone 4: Baselines and optional EDA check

1. Add random-candidate and activation-sensitivity rankings using the same candidate set.
2. Compare against exact `Circuit.simplify()` and uniform model narrowing.
3. If available, run open-source Yosys/ABC only as a final exact-synthesis reference. The main contribution and all primary metrics must work without an external EDA tool.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md), specifically:

- `repos/torchlogix/src/torchlogix/circuit.py` for graph representation, simplification, simulation, C, JSON, and Verilog;
- `repos/torchlogix/src/torchlogix/layers/groupsum.py` for class-score semantics;
- `repos/torchlogix/tests/test_circuit.py` as the base for equivalence tests.

TorchLogix is preferable to `repos/difflogic` because the latter has mature compiled execution but not the same editable circuit IR. The original repository should still be used as an execution and gate-count sanity check. `repos/difflogic-light-master` is not the foundation because it has no convolutional LGN and lacks the required editable circuit workflow.

## Datasets and protocol

| Dataset | Role | Protocol |
|---|---|---|
| MNIST | Search correctness and exhaustive small-model checks | Separate train, validation, calibration, and test partitions |
| Fashion-MNIST | Primary low-cost compression result | Tune rewrite score only on validation/calibration data |
| CIFAR-10 | Primary DATE scale result | Use a fixed calibration subset disjoint from validation and test |

Use a calibration set large enough to contain all classes and report class counts. Build the rewrite frontier without test labels. If a convolutional rank-2 export is not yet supported for the selected TorchLogix path, use a hardened convolutional feature extractor followed by an exportable rank-2 logic back end and state this limitation explicitly.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Deep Differentiable Logic Gate Networks*](../../pdfs/deep_differentiable_logic_gate_networks.pdf), NeurIPS 2022 | Unmodified hardened rank-2 circuit at each selected gate budget | Defines the unsimplified classifier and accuracy reference |
| Lee et al., [*Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*](../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf), ICML 2026 | Reproduce two-stage unit tying and its fine-tuning at the same allowed accuracy loss | Most direct published DLGN simplification competitor |
| Mishchenko et al., *DAG-Aware AIG Rewriting: A Fresh Look at Combinational Logic Synthesis*, DAC 2006 | ABC-style exact AIG rewriting after export | Establishes what ordinary function-preserving logic synthesis removes |
| Hashemi et al., [*BLASYS: Approximate Logic Synthesis Using Boolean Matrix Factorization*](https://arxiv.org/abs/1805.06050), DAC 2018 | BLASYS or a documented Boolean-matrix-factorization approximation on exported output cones | Published approximate-synthesis comparison at controlled error budgets |
| Han et al., [*Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*](https://arxiv.org/abs/1510.00149), ICLR 2016 | Magnitude/sensitivity-style pruning adapted to gate or cone importance, followed by retraining | Provides a recognized data-independent neural compression baseline |
| Yousefi et al., [*Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks*](../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf), NeurIPS 2025 | Hardened hard-forward model followed by identical exact simplification | Tests whether MarginSynth remains useful when the starting circuit has little discretization gap |

The minimum paper set is the unmodified circuit, TorchLogix exact simplification, Two-Stage Unit Tying, a Deep-Compression-style sensitivity baseline, and MarginSynth. ABC and BLASYS are strongly preferred DATE comparisons, but remain optional supporting experiments if integrating an external EDA flow threatens the core schedule. Their omission must be stated explicitly rather than replaced by an unnamed "EDA baseline."

### Controlled search ablations

Use the identical candidate set and calibration budget for:

1. random candidate order;
2. activation-frequency ranking;
3. first-order sensitivity ranking;
4. MarginSynth without the margin term;
5. complete MarginSynth;
6. uniform last-stage narrowing followed by the same retraining budget.

All approximate methods receive the same calibration examples and permitted accuracy loss. Report published-method results obtained from code separately from values copied from papers; only same-pipeline runs belong in statistical comparisons.

## Metrics

Report at every Pareto point:

- hardened accuracy, macro-F1, and per-class accuracy;
- calibration-predicted loss versus actual test loss;
- number and rate of decision flips relative to the original circuit;
- winner-margin distribution before and after rewriting;
- internal gate count, live gate count, logic depth, fan-out, and estimated connection bits;
- compiled-C mean and p95 latency, binary size, and throughput at batch 1;
- rewrite-search runtime, peak RAM, number of candidates evaluated, and accepted rewrite types;
- optional post-Yosys LUT/cell count and critical path, but only as supporting evidence.

The main plot is hard accuracy versus live gate count, with a second plot of latency versus accuracy. Report mean and paired confidence intervals over at least five trained seeds for the central CIFAR-10 budget; the rewrite search itself should be deterministic given a checkpoint and calibration seed.

## Minimum DATE experiment matrix

- Fashion-MNIST and CIFAR-10, with MNIST as a correctness appendix.
- Two starting gate budgets per primary dataset.
- Exact simplify, random, sensitivity, and MarginSynth at identical loss budgets.
- At least six saved Pareto points per circuit.
- Ablations for margin term, class-conditional budget, and periodic full resimulation.
- Five trained seeds for the central claimed operating point and three for surrounding points.

## Agent deliverables and tests

The assigned agent must provide:

- circuit copy, mutation, serialization, and stable-ID tests;
- exhaustive truth-table tests on tiny circuits for every rewrite type;
- incremental-versus-full simulation equivalence tests after rewrite sequences;
- a replayable JSON rewrite log containing old node, new node, estimated effect, and measured effect;
- scripts that compile and benchmark every saved Pareto circuit;
- plots generated directly from raw circuit and accuracy records;
- validation that test examples and labels are inaccessible to the search process.

## Risks, controls, and kill criterion

- **Calibration overfitting:** use a separate validation set for hyperparameters, class-aware budgets, and multiple calibration sizes.
- **Incorrect incremental simulation:** compare with full simulation frequently and fail closed on any mismatch.
- **Proxy cost does not predict latency:** present both gate/depth proxies and measured compiled-C latency.
- **Convolutional export limitations:** establish the method on rank-2 exported circuits rather than silently approximating unsupported ranks.
- **External EDA dependence:** keep Yosys/ABC optional so the core method remains self-contained.

Continue to a DATE manuscript only if MarginSynth removes at least 15% more live gates than exact simplification at no more than 0.5 percentage-point test-accuracy loss on two datasets, and it beats random rewriting at matched loss. It should also show either a measurable compiled-C speedup or a convincing depth/gate Pareto benefit.

## Definition of done

The project is complete when exported baselines are proven equivalent, rewrite logs are replayable, the test set never guides search, at least two datasets have Pareto frontiers against all required baselines, and the central reduction survives five-seed analysis.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Two-stage unit tying](../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf)
- [Mind the Gap](../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf)
