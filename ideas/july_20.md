# DLGN Research Ideas for DATE 2027

**Prepared:** July 20, 2026  
**Target:** Design, Automation and Test in Europe (DATE) 2027

## Executive recommendation

The default recommendation is **Idea 3, CoverageNet**, because its central experiments can be completed with the code and GPU already available. **Idea 1, PowerDLGN, becomes the preferred hardware-oriented direction only if a first-week ASIC characterization shows that dynamic switching is a material part of total energy.** DLGNs already have very low energy per inference, so a generic claim that they "consume too much power" would not be credible. PowerDLGN is worthwhile specifically for high-throughput or continuously active ASICs, not automatically for low-duty sensor nodes or FPGA deployments dominated by device static power.

The strongest fallback is **Idea 2, MarginSynth**. It treats a hardened DLGN as a classifier circuit and performs deliberately approximate, label-aware netlist optimization. This is different from ordinary exact synthesis, exact equivalence pruning, and generic neural-network pruning. It also makes good use of expertise in logic synthesis and approximate computing.

PowerDLGN still has a focused EDA opportunity: existing DLGN power results use a fixed or vectorless activity assumption rather than workload-derived internal transitions. The project should therefore begin as a measurement question, then proceed to training only if the controllable dynamic component is large enough. If access to an ASIC flow is uncertain, do not wait for it; begin CoverageNet immediately.

The [DATE 2027 call](https://www.date-conference.com/date-2027-call-papers) sets abstract registration for **September 13, 2026 AoE** and the firm paper deadline for **September 20, 2026 AoE**. From July 20, this is an eight-week research window. DATE expects novel, complete work supported by experiments, so the preferred project must produce its first accuracy/hardware Pareto curve by approximately August 10.

## Ranking

| Rank | Idea | Main direction | Novelty confidence | Feasibility by Sep. 20 | DATE fit | Main dependency |
|---:|---|---|---|---|---|---|
| 1 | CoverageNet | Accuracy / new architecture / no routing parameters | Medium-high | High | Good | Modify connection generator |
| 2 | PowerDLGN | Workload-aware ASIC energy | High if dynamic power is material | Conditional | Excellent | Trace-driven gate-level power flow |
| 3 | MarginSynth | Approximate computing / circuit compression | High | Medium | Excellent | RTL/netlist export and synthesis |
| 4 | TreeHead-DLGN | Popcount removal / new output architecture | Medium-high | Medium-high | Excellent | New hierarchical loss and head |
| 5 | FaultDLGN | Dependability / edge reliability | High, pending a deeper fault-literature search | Medium-high | Excellent | Fast gate-level fault injection |
| 6 | LogicSketch Distillation | Accuracy at small gate budgets | Medium | High | Moderate | Teacher inference and auxiliary losses |

The rankings reflect the deadline and the probability of a meaningful result. PowerDLGN has an excellent EDA claim only after its Phase-0 power decomposition passes. CoverageNet is the safest route to completed results. MarginSynth has a strong DATE story but more tool-engineering risk. Idea 6 is feasible but needs a more distinctive result than simply applying knowledge distillation.

## State-of-the-art boundaries that constrain the ideas

Several attractive first ideas are no longer novel enough as standalone projects:

- The original [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) already mentions pruning unused neurons and simplifying logical expressions.
- [Mind the Gap](https://openreview.net/forum?id=chYXaetMmz) largely resolves the soft-to-hard discretization gap for two-input DLGNs using a hard Gumbel/straight-through formulation.
- [Light DLGNs](https://arxiv.org/abs/2510.03250) reduce a two-input neuron's training parameters from 16 gate logits to four truth-table parameters.
- [WARP](https://arxiv.org/abs/2602.03527) already provides a fully expressive Walsh parameterization for 2-, 4-, and 6-input LUTs, Gumbel smoothing, and learned thresholds. Therefore, "IWP for larger LUTs," "a Walsh basis," or "IWP plus Gumbel" is insufficient.
- [LILogicNet](https://arxiv.org/abs/2511.12340), [Scalable Interconnect Learning](https://arxiv.org/abs/2507.02585), and the July 2026 [Fully Trainable Deep DLGN/LUTN](https://arxiv.org/abs/2607.09399) already learn DLGN connections at scale. A generic learned-connectivity proposal is obsolete.
- [Logic Shrinkage](https://arxiv.org/abs/2112.02346) learns per-LUT input removal and heterogeneous fan-in. Generic LUT-input pruning or adaptive arity is not a clean novelty claim.
- [Scalable Interconnect Learning](https://arxiv.org/abs/2507.02585) also performs exact SAT-equivalence pruning and data-driven similarity pruning. A proposal based only on deleting equivalent gates is insufficient.
- [Resource Utilization of DLGNs on FPGAs](https://arxiv.org/abs/2605.04109) shows that the last layer and its summation logic are critical and reports about a 28% resource/timing reduction from narrowing it. Merely making the final layer narrower is not new.
- [BitLogic](https://arxiv.org/abs/2602.07400) already unifies encoder, connectivity, fan-in, parameterization, and output-head choices, with bit-exact RTL export. It identifies depth as an open sixth axis, but a benchmark-only depth sweep would be a weak DATE paper.
- The July 2026 [FPGN](https://arxiv.org/abs/2607.08427) already combines FPGA-aligned LUT6 training, structured local topology, streaming hardware, and compiler-driven design-space exploration. A generic "physically aware FPGA DLGN" is no longer sufficiently differentiated.
- [Logic Gate Neural Networks are Good for Verification](https://proceedings.mlr.press/v288/kresse25a.html) already encodes LGNs in SAT for global robustness and fairness. Generic formal verification is not an open standalone idea.
- Recurrent ternary DLGNs for online Signal Temporal Logic monitoring appeared in [May 2026](https://arxiv.org/abs/2605.24649), directly occupying a future-work direction suggested by the ternary PST paper.

These collisions are why the proposals below target switching activity, controlled semantic approximation, deterministic coverage, the arithmetic output bottleneck, internal hardware faults, and DLGN-specific distillation.

---

## Idea 1: PowerDLGN - workload-aware switching-energy training

### Focus

Workload-aware ASIC energy and silicon-aware learning. The paper should make one primary claim: **when switching energy is a material part of the system budget, train a DLGN to minimize measured dynamic energy under its deployment workload rather than using gate area as an indirect power proxy.** This is not a general low-power argument. FPGA and low-duty edge cases must be treated as possible negative results.

### When this is worth pursuing

PowerDLGN is potentially valuable when:

- The target is an ASIC, where internal gate and interconnect activity is a controllable fraction of energy.
- The accelerator operates continuously or at high inference rates, such as packet classification, event triggers, or high-rate monitoring.
- Accuracy requires a large DLGN, so the number of switching gates and wires is substantial.
- Workload traces are representative and stable enough to train against.

It is probably not valuable when:

- A sensor performs only hundreds or thousands of inferences per second.
- FPGA static/device power dominates total power.
- I/O, memory, preprocessing, or the output head dominates the system energy.
- Clock gating or duty cycling can remove more energy than changing gate activity.

### Published scale and opportunity

| Implementation | Power | Throughput | Energy/inference | Caveat |
|---|---:|---:|---:|---|
| Silicon-aware DLGN, SkyWater 130 nm | 83.88 mW | 41.8 M/s | 2.0 nJ from power/throughput | Approximately 97% MNIST; assumed input activity factor 0.7 |
| BitLogic, ZU7EV max-throughput mode | 0.72 W | 127.2 M/s | 6 nJ | 88.79% MNIST; Vivado vectorless estimate |
| BitLogic, U55C max-throughput mode | 3.40 W | 126.6 M/s | 27 nJ | Same checkpoint; Vivado vectorless estimate |
| BitLogic, ZU7EV fewest-resource mode | 0.68 W | 10.5 M/s | 65 nJ | Similar device power but much lower throughput |

The ASIC figures come from [Silicon Aware Neural Networks](https://arxiv.org/abs/2604.19334). Dividing 83.88 mW by 41.8 million inferences/s gives approximately 2.0 nJ/inference. A hypothetical 20% reduction would save 0.4 nJ/inference and 16.8 mW at full throughput, but only 0.4 uW at 1,000 inferences/s. This quantifies the operating-regime boundary: the same relative improvement can be meaningful in a continuously active accelerator and irrelevant in a low-duty sensor.

The FPGA figures come from [BitLogic](https://arxiv.org/abs/2602.07400). On the ZU7EV, reducing throughput from 127.2 M/s to 10.5 M/s changes reported power only from 0.72 W to 0.68 W, while energy/inference worsens from 6 nJ to 65 nJ. This indicates that static/device overhead is large and that reducing network toggles alone may have little effect on total FPGA power. BitLogic also uses vectorless activity estimation and explicitly leaves an activity-trace-driven flow to future work.

There is evidence that hardware-aware training can materially reshape these circuits, although it does not yet prove equal energy savings. The silicon-aware area loss reduces MNIST area from 3.00 to 1.95 mm2, about 34.9%, while accuracy changes from 98.04% to 97.66%. On CIFAR-10 it reduces area from 12.79 to 9.62 mm2, about 24.8%, while accuracy changes from 60.07% to 58.82%. In the adjacent LUT-native literature, [Logic Shrinkage](https://arxiv.org/abs/2112.02346) reports a 1.31x energy-efficiency improvement at comparable CIFAR-10 accuracy, equivalent to approximately 24% lower energy.

One reporting issue strengthens the need for a reproducible flow. The silicon-aware paper's prose reports 83.88 mW at 41.8 M/s, which implies 2.0 nJ/inference, while its comparison table lists 352 pJ for "This Work." PowerDLGN must not build its claim on such headline estimates. It must report workload-derived VCD/SAIF activity, dynamic and leakage power separately, and arithmetic that can be independently checked.

### Hypothesis

Conditional on dynamic power being material, two hardened DLGNs with the same gate count and cell area can have different energy because their internal transition rates, fanout loads, and selected gate functions differ. The soft DLGN may expose enough information during training to estimate those effects and move the final circuit to a better accuracy-energy Pareto frontier. This is a hypothesis to test, not an established property.

### Proposed method

For gate $n$, estimate a differentiable output-one probability over a representative workload:

$$
p_n = \mathbb{E}_{x}[q_n(x)],
$$

where $q_n(x)$ is the relaxed gate output. For independently ordered samples, a first activity estimate is

$$
\alpha_n = 2p_n(1-p_n).
$$

For streams, sensor traces, or video, use the more faithful sequential estimate

$$
\alpha_n = \mathbb{E}_{t}\left[\left|q_n(x_t)-q_n(x_{t-1})\right|\right].
$$

Combine it with target-library energy data:

$$
L = L_{task} + \lambda_E \sum_n \sum_g \pi_{n,g}
\left(E^{int}_g(\text{input transitions}) + \alpha_n C^{eff}_{n,g}V^2\right).
$$

Here, $\pi_{n,g}$ is the soft probability of selecting gate $g$, $E^{int}_g$ is a characterized internal switching cost, and $C^{eff}_{n,g}$ contains load/fanout capacitance. Start with a simple fanout-based capacitance proxy. Add placement-derived wire capacitance only after the loss shows a useful signal.

Use mini-batch exponential moving averages for $p_n$ and activity so that the loss is stable. After hardening, emit stimulus traces, collect VCD/SAIF activity, and use gate-level power analysis to validate whether the differentiable proxy ranks models correctly.

### Clear novelty boundary

*Silicon Aware Neural Networks* maps DLGNs to standard cells and optimizes expected **cell area**. Its reported power analysis assumes a fixed input switching activity. BitLogic reports vectorless FPGA power and identifies an activity-driven flow as future work. PowerDLGN instead learns **workload-specific internal switching behavior**, includes fanout/load, separates dynamic power from leakage, and validates the learned proxy against activity-annotated netlists. This is also distinct from FPGN, whose primary hardware objective is FPGA latency/routability rather than workload-dependent dynamic energy.

The contribution is not "add an energy term." It is the derivation and validation of a differentiable DLGN switching model that remains predictive after hardening and synthesis.

### Minimum experiment set

- Phase 0: harden an unmodified baseline and use representative VCD/SAIF traces to decompose cell-internal, net-switching, leakage, clock, input-encoding, and output-head power. Do this before implementing the training loss.
- Primary target: ASIC standard cells. Treat FPGA as a secondary characterization or negative-control platform because device static power may dominate.
- Tasks: MNIST and Fashion-MNIST for iteration; CIFAR-10 for the main result; one high-rate or naturally sequential dataset, such as network-flow classification, if time permits.
- Models: original DLGN and Light/IWP or hard-Gumbel DLGN at two gate budgets.
- Baselines: task loss only; the area-aware loss from [Silicon Aware Neural Networks](https://arxiv.org/abs/2604.19334); constant gate-cost regularization; post-training exact synthesis.
- Ablations: marginal $2p_n(1-p_n)$ versus sequential activity; cell-only versus fanout-aware; soft estimated activity versus hardened measured activity.
- Metrics: hard accuracy, area, worst path delay, dynamic power, leakage power, clock power, total power, energy/inference, total toggles, controllable dynamic fraction, correlation between predicted and post-synthesis energy, and Pareto hypervolume.
- Report at least three seeds for accuracy and repeat power estimation with identical activity traces.

### Feasibility and risk

Training changes are small and compatible with the existing DLGN code. The main external dependency is a trace-driven gate-level power flow. No synthesis or place-and-route executable was found on the current `PATH`, so access to Yosys/OpenROAD plus a characterized open library, or to an existing commercial flow, must be resolved immediately.

The first risk is that there is no important problem to solve: baseline DLGN energy may already be negligible at the intended inference rate, or leakage/platform overhead may dominate. The second risk is that synthesis absorbs the trained activity differences or that lowering toggles simply creates constant gates and harms accuracy. Preserve functional utilization through the task loss and evaluate actual post-layout netlists rather than raw neuron statistics.

**Phase-0 kill criterion, end of week 1:** stop PowerDLGN if workload-dependent gate/net switching is less than 30% of total core power at the intended duty cycle, or if baseline inference energy is immaterial to the system budget. Do not use FPGA vectorless power to pass this test.

**Week-3 kill criterion:** the differentiable proxy should rank post-layout dynamic energy reliably, with a target Spearman correlation of at least 0.8 across baseline variants. At equal accuracy within 0.5 percentage point, training should reduce dynamic energy by at least 15-20% and total energy by at least 10% on two datasets. Otherwise switch to CoverageNet.

### DATE paper statement

> We show when switching energy remains material in already-efficient DLGNs, introduce a workload-aware differentiable activity objective, and reduce trace-annotated post-layout ASIC energy at matched accuracy.

---

## Idea 2: MarginSynth - accuracy-budgeted approximate synthesis for DLGNs

### Focus

Approximate computing and post-training circuit compression. This idea begins after the DLGN is hardened.

### Hypothesis

Exact logic synthesis must preserve every output bit for every input, although a classifier only needs the winning class to remain correct on the target distribution. DLGNs contain many Boolean evidence bits and often have a nonzero winner-runner-up margin. Spending a controlled portion of that margin can enable circuit reductions unavailable to exact synthesis.

### Proposed method

1. Export the hard DLGN, including its output head, to a bit-exact Boolean netlist.
2. Generate local approximate rewrites: replace a gate by a cheaper gate, bypass or invert one input, substitute a correlated signal, remove a cone, merge similar signals, or approximate/truncate parts of the output aggregation.
3. Evaluate many candidates rapidly with bit-parallel simulation over a calibration set.
4. Score each rewrite by

$$
\text{score}(r) =
\frac{\Delta \widehat{PPA}(r)}
{\epsilon + \Delta L_{margin}(r)},
$$

where $L_{margin}$ penalizes changes to the winner-runner-up class margin more strongly near the decision boundary.
5. Apply rewrites iteratively with periodic exact resimulation because approximation errors interact.
6. Optionally generate a SAT miter for the high-confidence region to certify that the optimized netlist preserves the class label whenever the original score margin exceeds a chosen bound. Treat this certificate as a stretch goal; the paper remains viable with measured risk and statistical confidence intervals.

The key output is an accuracy-area-delay-energy Pareto curve parameterized by an explicit permitted accuracy loss.

### Clear novelty boundary

- Ordinary ABC/Yosys optimization is exact and cannot exploit classification tolerance.
- Scalable Interconnect Learning performs exact equivalence pruning plus a label-agnostic similarity heuristic; its authors explicitly identify label information as unused.
- eXpLogic traces class-specific activation paths and can reduce class-specific inference, but it does not formulate global multi-class, PPA-driven approximate synthesis.
- Logic Shrinkage removes LUT inputs during training; MarginSynth operates on any already-hardened DLGN and can change gates, signals, cones, and aggregation logic jointly.
- Narrowing the last layer manually, as studied in the FPGA resource paper, is one uniform design point rather than a semantic, rewrite-level optimizer.

### Minimum experiment set

- Models: one 2-input DLGN and one LUT4/LUT6 model if a compatible exporter is available.
- Tasks: MNIST, Fashion-MNIST, and CIFAR-10 at small and medium gate budgets.
- Baselines: exact Yosys/ABC; dead-cone/trivial pruning; SAT-equivalence pruning; similarity pruning; random rewrite; uniform last-layer narrowing.
- Targets: one FPGA family and one open ASIC library if possible. If only one can be completed, choose ASIC because gate replacement and Boolean resynthesis have a clearer cost gradient there.
- Metrics: test accuracy, calibration/test disagreement, area or LUTs, critical path, power/energy, optimization time, and accuracy loss predicted versus observed.
- Essential ablations: confidence margin versus output-bit Hamming error; individual rewrite classes; calibration-set size; rewrite recomputation frequency.

### Feasibility and risk

This is algorithmically feasible because hardened DLGNs are simple DAGs and can be simulated bit-parallel. A narrow first implementation should support only gate replacement, bypass, constants, and cone deletion. Add popcount approximation after that pipeline works.

The risk is tool engineering: parsing and rewriting netlists can consume the schedule. Avoid editing arbitrary Verilog text. Keep an internal graph IR, emit clean Verilog, and use synthesis only for periodic cost feedback.

**Week-3 kill criterion:** the margin-aware search must dominate random and exact-only baselines by at least 15% area at no more than 0.5 percentage-point accuracy loss on two datasets. If it only reproduces synthesis cleanup, stop.

### DATE paper statement

> We formulate hardened DLGN compression as classifier-aware approximate logic synthesis, using decision margins to trade bounded semantic error for circuit reductions that exact synthesis cannot realize.

---

## Idea 3: CoverageNet - parameter-free structured global wiring

### Focus

Accuracy, parameter reduction, and a new feed-forward architecture. Its single central question is whether DLGNs need learned connectivity, or instead need a better deterministic coverage schedule.

### Hypothesis

The main weakness on natural images is not necessarily GroupSum. A depth-$d$, two-input output can depend on at most $2^d$ source signals, and random wiring can duplicate paths long before reaching this bound. A deterministic multi-scale wiring schedule can guarantee rapid input coverage, bounded fanout, and regular hardware without storing or training connection weights.

### Proposed architecture

Assign each signal a dependency signature describing which input regions can influence it. Construct each layer offline by pairing signals that maximize new coverage while respecting fanout and locality budgets:

$$
(i,j)^* = \arg\max_{i,j}
\left|S_i \cup S_j\right|
- \lambda_f \operatorname{fanoutCost}(i,j)
- \lambda_w \operatorname{wireCost}(i,j).
$$

Use a deterministic schedule rather than an expensive global search:

- Early layers mix within small spatial or feature blocks.
- At layer $l$, a fixed shuffle pairs blocks separated by a power-of-two stride.
- Every few layers, retain a local pair alongside the long-range pair.
- Use balanced fanout so no attractive signal becomes a routing hot spot.

This resembles a butterfly/expander communication pattern. The design goal is a theorem or constructive guarantee: after $O(\log D)$ stages, every output cone covers all $D$ input groups, subject to layer width.

The gate functions remain trainable with Light/IWP or hard Gumbel. The connections are generated from a seed and a schedule, so connection-training parameters and index storage can be zero or negligible.

### Clear novelty boundary

- The original DLGN uses fixed pseudo-random connectivity without coverage guarantees.
- LILogicNet, Scalable Interconnect Learning, and Fully Trainable Deep DLGN/LUTN optimize connection choices and pay training memory/compute for them.
- OSLGN uses a proximity-biased small-world prior while still learning operands.
- FPGN uses structured CNN-like, in-order local topology to improve FPGA locality and streaming. CoverageNet's defining objective is **provable global dependency coverage with no learned router**, while balancing local and long-range edges.

Do not sell this as merely a butterfly network. The publishable result is the link among coverage guarantees, duplicate-path reduction, accuracy, trainable-parameter count, and post-route wiring cost.

### Minimum experiment set

- Backbones: existing Light/IWP and Mind-the-Gap implementations, keeping gate parameterization fixed.
- Connectivity baselines: fixed random, random-unique, bounded learned candidates, full learned connectivity where feasible, OSLGN-style locality prior, and an FPGN-like local schedule.
- Tasks: MNIST/Fashion-MNIST for debugging; CIFAR-10 and CIFAR-100 for the main claim; a tabular or flow task to show that the method is not image-specific.
- Sweep depth at fixed total gate count and fixed training-parameter budget.
- Metrics: hard accuracy, coverage fraction per layer, duplicate ancestors, input influence distribution, fanout, connection parameters, GPU training memory/time, wirelength proxy, and post-route timing where available.
- Ablations: only local, only long-range, mixed schedule, random permutation, coverage-aware pairing, and fanout constraint.

### Feasibility and risk

This is the most feasible idea with the present repository. It primarily changes index generation and instrumentation, and the codebases for original DLGN, Light DLGN, and Mind the Gap are available locally. Use 8K-64K gate models first; the RTX 2060 has only 6 GB, so avoid reproducing the largest published networks until the trend is established.

The main novelty risk is that a reviewer may view the schedule as a standard interconnect pattern. Counter this with a formal coverage property, matched learned-router comparisons, and physical results showing that accuracy is not purchased with pathological long wires.

**Week-2 kill criterion:** at a matched gate count, the structured schedule must improve hard CIFAR-10 accuracy by at least 2 percentage points over fixed random, or match bounded learned connectivity with materially less training memory and no worse fanout/wire proxy.

### DATE paper statement

> We replace random or learned DLGN interconnects with a deterministic, hardware-regular schedule that guarantees global input coverage, improving accuracy without connection parameters or router-training overhead.

---

## Idea 4: TreeHead-DLGN - a popcount-free hierarchical class decoder

### Focus

New output architecture and edge circuit reduction. This idea targets only the DLGN readout bottleneck.

### Hypothesis

The standard GroupSum head converts cheap Boolean features into wide popcount/adder trees. Recent FPGA and ASIC results show that this final aggregation can dominate resources. A balanced hierarchy of learned Boolean decisions can decode the class using LUTs and multiplexers, avoiding wide integer sums.

### Proposed architecture

1. Train a normal DLGN backbone to produce Boolean features.
2. Build a balanced class tree. Derive the tree from class-confusion statistics so classes that need similar evidence are separated late rather than early.
3. At each internal node, train a small LUT-reduction network that decides between two class subsets using a bounded feature subset.
4. The root-to-leaf decisions form the class code. During training, sum binary cross-entropies for every internal decision on the target path, optionally adding a consistency term between soft and hard routing.
5. For hardware, instantiate the Boolean decision tree and a compact encoder/multiplexer. Evaluate both fully combinational execution and sequential path evaluation for resource-constrained edge devices.

The final circuit contains no class-wise accumulation and no Hamming-distance popcount. For $C$ classes, inference makes $\lceil \log_2 C \rceil$ hierarchical decisions on the active path, although the fully combinational implementation may instantiate multiple tree nodes.

### Clear novelty boundary

- *From MNIST to ImageNet* evaluates GroupSum, dense heads, individual binary losses, and codebook/Hamming decoding. A hierarchical learned class tree is different because it eliminates both class sums and Hamming-distance sums.
- The FPGA resource study manually narrows the final layer but retains the summation scheme.
- BitLogic compares GroupSum with a quantized DSP-backed head; TreeHead remains logic-only.
- FPGN still uses a group-sum style output, so its streaming topology does not solve this head problem.

### Minimum experiment set

- Tasks: MNIST, CIFAR-10, CIFAR-100, and a controlled synthetic dataset with 10-1000 classes.
- Baselines: GroupSum with tuned temperature; uniformly narrowed GroupSum; codebook/Hamming head; quantized dense/DSP head; random versus confusion-derived class tree.
- Metrics: hard accuracy, logic cells/LUTs, adder count, critical path, throughput, energy, output parameters, and scaling with number of classes.
- Ablations: balanced random tree, semantic/confusion tree, feature sharing among nodes, tree depth, and hard versus soft path training.

### Feasibility and risk

The head is much smaller than the backbone, so training is affordable. The primary risk is cascading classification error: an early wrong branch cannot be recovered. Soft routing during early training, teacher initialization from a GroupSum model, and restructuring the class tree from the confusion matrix should be treated as core mechanisms rather than optional refinements.

**Week-3 kill criterion:** on CIFAR-10 or CIFAR-100, the head must retain accuracy within 1 point while reducing synthesized output-head area by at least 30%. An improvement only on MNIST is not enough for DATE.

### DATE paper statement

> We replace the arithmetic GroupSum bottleneck of DLGNs with a learned hierarchical Boolean decoder, preserving logic-only inference while reducing output-head area and delay as class count grows.

---

## Idea 5: FaultDLGN - fault-aware training of logic-native classifiers

### Focus

Dependability and test. This is deliberately separate from accuracy or compression work: the goal is graceful inference under internal circuit faults with less overhead than blanket redundancy.

### Hypothesis

DLGNs map directly to a Boolean circuit, which makes permanent and transient hardware faults easy to inject exactly during training. A hard-forward/straight-through model can learn where redundancy is useful and which gate functions/topologies are naturally insensitive to target fault modes.

### Proposed method

- Define FPGA fault models: LUT truth-table bit upset, stuck-at LUT output, and selected routing stuck-at or disconnection.
- Define ASIC fault models: gate-output stuck-at-0/1 and transient output flips.
- During hard forward passes, sample one or more persistent fault maps per mini-batch and reuse each map across several steps so training sees permanent, correlated faults rather than independent activation noise.
- Optimize

$$
L = L_{clean} + \lambda_f\mathbb{E}_{F\sim\mathcal{D}}L(x,y;F)
+ \lambda_w \max_{F\in\mathcal{B}}L(x,y;F),
$$

where the last term approximates the worst fault in a small sampled budget.
- Add a small fanout-aware duplication budget only for gates with high measured fault sensitivity. This produces selective redundancy rather than full triple modular redundancy.

### Clear novelty boundary

Existing LGN verification work proves robustness/fairness with respect to **input perturbations**. Recurrent ternary logic work handles unknown or missing sensor information. Neither addresses faults inside the deployed learned circuit. The proposed work trains the Boolean implementation itself against internal gate, LUT-memory, and routing faults and evaluates the accuracy-reliability-area frontier.

The literature search performed for this memo found no direct DLGN fault-aware training paper, but this is a negative search result rather than proof of absence. A systematic IEEE/ACM literature search must be the first task before committing.

### Minimum experiment set

- Tasks: MNIST, Fashion-MNIST, CIFAR-10, plus one safety- or health-related dataset such as ECG if it can be reproduced quickly.
- Baselines: unprotected DLGN; activation bit-flip noise training; random duplication; selective duplication without fault training; full TMR at matched reliability where feasible.
- Report clean accuracy and accuracy versus fault rate for each fault type, gate criticality coverage, silent data corruption rate, area overhead, delay, and energy.
- Test unseen fault distributions and multi-fault cases to show that the method did not memorize one injector.
- Use formal or exhaustive analysis on small networks to validate the Monte Carlo fault-sensitivity estimator.

### Feasibility and risk

Fault injection at the DLGN graph level is straightforward and needs no hardware tool for the first result. Bit-parallel simulation should support millions of fault/sample pairs. Hardware synthesis is still needed to measure the cost of selective redundancy.

The risk is a shallow contribution if it is only noise injection. The paper needs a DLGN-specific mechanism: persistent structural fault maps, exact Boolean fault semantics, sensitivity-directed selective redundancy, and a hardware reliability Pareto frontier.

**Week-3 kill criterion:** at less than 20% synthesized area overhead, the method should reduce accuracy degradation under the target fault rate by at least 2x relative to a standard DLGN and outperform random duplication.

### DATE paper statement

> We exploit the direct circuit semantics of DLGNs to train against persistent internal logic faults and allocate redundancy by learned fault sensitivity, improving reliability at substantially lower cost than uniform replication.

---

## Idea 6: LogicSketch Distillation - teacher-guided compact DLGNs

### Focus

Accuracy at a fixed, small gate budget. This is the least hardware-dependent idea and the fastest to prototype.

### Hypothesis

Logit-only distillation gives a sparse DLGN a target at the output but does not address its weak intermediate gradient flow or limited information coverage. A conventional teacher can provide layer-wise **binary sketches** that are directly representable by Boolean gates, giving compact DLGNs a sequence of attainable subproblems.

### Proposed method

1. Train or obtain a compact teacher such as ResNet-18. Precompute its logits and selected intermediate activations so the 6 GB GPU is sufficient for student training.
2. Convert each teacher feature vector to a small binary sketch using fixed random hyperplanes, learned binary hashing, or thresholded teacher features.
3. Attach temporary probes to selected DLGN layers. Each probe predicts the corresponding teacher sketch with a binary loss.
4. Train progressively from shallow to deep, then remove the probes and fine-tune with hard-forward DLGN training.
5. Match teacher pairwise class margins in addition to ordinary softened logits so the student preserves ranking information near decision boundaries.

The research question is whether Boolean intermediate supervision reduces the number of gates needed for a target hard accuracy more effectively than ordinary KD.

### Clear novelty boundary

No DLGN-specific knowledge-distillation method was found in the reviewed papers or the primary-source search. However, applying standard KD alone would be incremental. The differentiator must be the conversion of teacher representations into hardware-free Boolean training targets, progressive layer-wise supervision, and evaluation in **gates/LUTs at matched hard accuracy**.

This does not overlap with learned connectivity: the method can use fixed random or CoverageNet wiring. It also does not change the deployed circuit because all teacher projections and auxiliary probes are removed after training.

### Minimum experiment set

- Tasks: CIFAR-10 and CIFAR-100, with MNIST only as a debugging benchmark.
- Students: original DLGN, Light/IWP, and hard-Gumbel DLGN at 8K, 16K, and 32K gates per layer as memory permits.
- Baselines: label-only training; standard temperature-based logit KD; margin-only KD; binary sketches only; a same-sized BNN student.
- Metrics: hard accuracy, soft-hard gap, gates required to reach accuracy thresholds, training time/memory, and synthesized cost for selected points.
- Ablations: sketch dimension, teacher layer, random versus learned sketches, number of supervised student layers, and probe removal/fine-tuning.

### Feasibility and risk

This uses existing training code and requires no netlist manipulation. Precomputed teacher targets keep GPU memory manageable. The main risk is novelty: reviewers may interpret the method as conventional intermediate-feature distillation with binarization.

**Week-2 kill criterion:** the Boolean-sketch loss must beat standard logit KD by at least 1.5 hard-accuracy points at one compact CIFAR-10 budget and show the same direction at a second budget. Otherwise it should remain an auxiliary technique, not the paper's central contribution.

### DATE paper statement

> We distill continuous teachers into compact DLGNs through disposable layer-wise Boolean sketches, improving hard-circuit accuracy without adding inference parameters or arithmetic.

---

## Suggested eight-week execution

### July 20-26: de-risking

- Reproduce one small hard-accuracy baseline from the local Light or Mind-the-Gap code.
- Secure the synthesis/power tool flow and a standard-cell library.
- Before implementing PowerDLGN, generate representative VCD/SAIF traces and measure dynamic versus leakage/clock/head power for one hardened ASIC baseline.
- Implement the PowerDLGN proxy only if its Phase-0 power decomposition passes; implement the CoverageNet connection generator regardless.
- Run a systematic fault-literature search before retaining FaultDLGN.

### July 27-August 9: decisive pilot

- Produce two-seed MNIST/Fashion and one-seed CIFAR-10 results for Ideas 1 and 3.
- Validate estimated switching against activity-annotated post-layout results, not vectorless estimates.
- Apply the kill criteria and choose exactly one primary paper.
- Freeze datasets, baselines, hardware target, and model budgets.

### August 10-30: complete evidence

- Run three-seed accuracy experiments and matched hardware synthesis.
- Complete ablations before scaling to the largest model.
- Generate the main Pareto figures and test the central claim on at least two nontrivial datasets.

### August 31-September 12: manuscript

- Write the six-page DATE paper around one claim and one hardware story.
- Run missing reviewer-facing baselines and sensitivity checks.
- Register title, abstract, and all authors by September 13 AoE.

### September 13-20: finalization

- Freeze new features.
- Check bit-exact hard inference, seed aggregation, figure legibility, and double-blind compliance.
- Submit by September 20 AoE.

## Ideas not recommended as standalone DATE 2027 projects

- **Generic learned connectivity:** superseded by LILogicNet, Scalable Interconnect Learning, and Fully Trainable Deep DLGN/LUTN.
- **IWP plus learned connections:** the pieces and their combination are already too close to July 2026 work.
- **IWP/WARP with 4- or 6-input gates:** already covered directly.
- **Per-LUT fan-in pruning or adaptive arity alone:** Logic Shrinkage already establishes this method and hardware benefit.
- **Simple pruning of unused/equivalent gates:** original DLGN, eXpLogic, and Scalable Interconnect Learning cover variants of this.
- **Only narrowing the final layer:** the 2026 FPGA resource study already quantifies this.
- **A codebook output head:** evaluated in *From MNIST to ImageNet* and did not consistently dominate GroupSum.
- **A generic physically aware FPGA compiler/topology:** FPGN now occupies this space.
- **Recurrent ternary DLGN for STL:** published in May 2026.
- **A benchmark that only adds depth as BitLogic's sixth axis:** useful infrastructure, but insufficient without a new depth-aware method or hardware result.
- **Natural-image GIC-DLC compression:** scientifically interesting, but dataset/model/hardware expansion is too broad for the current eight-week deadline.

## Evidence base reviewed

All 21 Markdown notes in `notes/` were read. The corresponding PDFs were used to check precise limitations, conclusions, future-work statements, and hardware details where the summaries alone were insufficient. The convolutional DLGN PDF was not used as the basis for a proposal.

### Foundation, training, and parameterization

- [Deep Differentiable Logic Gate Networks](../notes/deep_differentiable_logic_gate_networks.md)
- [Light Differentiable Logic Gate Networks](../notes/light_differentiable_logic_gate_networks.md)
- [Mind the Gap](../notes/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.md)
- [WARP Logic Neural Networks](../notes/warp_logic_neural_networks.md)
- [Fitting Multilinear Polynomials for Logic Gate Networks](../notes/fitting_multilinear_polynomials_for_logic_gate_networks.md)
- [Polynomial Surrogate Training for Differentiable Ternary Logic Gate Networks](../notes/polynomial_surrogate_training_for_differentiable_ternary_logic_gate_networks.md)
- [Learning Interpretable Differentiable Logic Networks](../notes/learning_interpretable_differentiable_logic_networks.md)

### Connectivity, architecture, and scale

- [A Method for Optimizing Connections in DLGNs](../notes/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.md)
- [LILogicNet](../notes/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.md)
- [Truth Table Net](../notes/a_scalable_interpretable_verifiable_differentiable_logic_gate_convolutional_neural_network_architecture_from_truth_tables.md)
- [From MNIST to ImageNet](../notes/from_mnist_to_imagenet_understanding_the_scalability_boundaries_of_differentiable_logic_gate_networks.md)
- [BitLogic](../notes/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.md)
- [Recurrent DLGN](../notes/recurrent_deep_differentiable_logic_gate_networks.md)
- [Differentiable Logic Cellular Automata](../notes/celluar_automata.md)

### Hardware, edge applications, and adjacent logic learning

- [Silicon Aware Neural Networks](../notes/silicon_aware_neural_networks.md)
- [Logic Neural Networks for Efficient FPGA Implementation](../notes/logic_neural_networks_for_efficient_fpga_implementation.md)
- [ET: An Energy Efficient Edge Transformer Architecture](../notes/et_an_energy_eficient_edge_transformer_architecture.md)
- [Lightweight In-Network Flow Classification](../notes/lightweight_in_network_flow_classification_with_deep_differentiable_logic_gate_networks.md)
- [Inter-Patient ECG Classification with LGNs and LUTNs](../notes/inter_patient_ecg_arrhythmia_classification_with_lgns_and_lutns.md)
- [GIC-DLC Image Compression](../notes/gic_dlc_differentiable_logic_circuits_for_hardware_friendly_grayscale_image_compression.md)
- [DeepGate](../notes/deepgate_learning_neural_representations_of_logic_gates.md)

## Final choice rule

Choose **PowerDLGN** only if an ASIC trace-driven power flow is available this week, switching contributes at least 30% of total core power at the target duty cycle, and the baseline energy is material to the intended system. Otherwise choose **CoverageNet**. Choose **MarginSynth** only if a clean Boolean graph/export pipeline can be operational by July 27; otherwise its engineering risk is too high for the deadline.
