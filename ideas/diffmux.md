# Diffmux-Derived Research Ideas for DATE 2027

**Prepared:** July 21, 2026  
**Feasibility audit:** July 22, 2026  
**Target:** Design, Automation and Test in Europe (DATE) 2027  
**Starting point:** [Optimizing Differentiable Logic Gate Networks Thesis](../pdfs/Optimizing_Differentiable_Logic_Gate_Networks_Thesis.pdf)

## Executive recommendation

The thesis contains a useful primitive, but **the present static Diffmux is not yet a publishable pooling contribution**. It learns two address variables for a 4-to-1 multiplexer. After hardening, the address is constant for every sample and the multiplexer forwards one fixed source. It is therefore a learned static connection, not input-dependent pooling.

The thesis results support further work on routing and width reduction, but not the pooling claim:

- DLGN max pooling achieved 70.89% hard CIFAR-10 accuracy, versus 62.64% for the best reported DLGN Diffmux result: an 8.25 percentage-point gap.
- Light DLGN max pooling achieved 71.23%, versus 65.53% for the best reported Light DLGN Diffmux result: a 5.70-point gap.
- In the reduced-parameter comparison, Diffmux used 1,424,000 training parameters versus 1,824,000 for 2:1 fully connected reduction, a 21.9% reduction, but hard accuracy changed from 47.64% to 47.37%. This is promising evidence of a trade-off, not yet evidence of superiority.
- At a matched 512,000-gate budget, Diffmux achieved 48.60% hard accuracy, versus 46.85% for 2:1 reduction and 47.91% for the all-fully-connected architecture. At smaller budgets, the reported confidence intervals mostly overlap.

There is also a direct prior-art issue. The binary-address selector

```math
r_i(\mathbf{a}) =
\prod_{j:b_j(i)=1} a_j
\prod_{j:b_j(i)=0}(1-a_j)
```

with `log2(K)` relaxed address variables for `K` choices is already the central single-selector construction of [DSelect-k](https://papers.nips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html). Recent DLGN work also learns operands or connections, including [connection optimization](https://arxiv.org/abs/2507.06173), [LILogicNet](https://arxiv.org/abs/2511.12340), [Operand-Selective LGNs](https://openreview.net/forum?id=sfqIc7BnwS), and the July 2026 [fully trainable DLGN/LUTN](https://arxiv.org/abs/2607.09399). Therefore, neither "use a multiplexer" nor "learn a route with logarithmic address bits" is a defensible novelty claim by itself.

After a second feasibility and prior-art audit, **Rank 1, RepairMux-DLGN** has the clearest research merit and DATE fit, but only if it is narrowed to selective producer-level repair and one hardware flow. **Rank 2, ElasticMux-DLGN** is easier to prototype, but its novelty is weaker because binary early-exit networks and FPGA early-exit toolflows already exist. **Rank 3, ConfigMux-DLGN** remains a plausible follow-up project rather than the recommended September submission.

No Diffmux-derived idea is currently ready to execute as a full paper without a short feasibility phase. The first decision is whether the student's implementation can be recovered. If it cannot, the project must budget time to reimplement and validate Diffmux before testing any extension.

### Ranked shortlist

| Rank | Working title | Research merit | Novelty confidence | Software feasibility | DATE readiness | Verdict |
|---:|---|---|---|---|---|---|
| 1 | RepairMux-DLGN | High if spare signals are demonstrably substitutable | Medium-high | Medium | Medium-low | Primary, conditional on an early repair result and a hardware flow |
| 2 | ElasticMux-DLGN | Medium; useful system capability | Medium-low | Medium-high | Medium-low | Fast backup; needs more than ordinary early exit |
| 3 | ConfigMux-DLGN | Medium for multi-domain edge devices | Medium-low | Medium-low | Low | Follow-up paper unless a strong pilot already exists |
| 4 | HypercubeMux-DLGN | Low as a standalone contribution | Low | Medium | Very low | Do not select without an unexpectedly strong pilot |
| 5 | VectorMux Pooling | Medium conceptually, low under present evidence | Medium-low | Low | Very low | Exploratory future work |

These are **separate papers**. Combining runtime elasticity, repair, personalization, code assignment, and pooling into one submission would obscure the main claim and is not feasible within the DATE schedule.

The detailed sections retain ElasticMux first because it motivates the runtime-configurability discussion that led to this audit. The ranked table and final recommendation are authoritative.

### Repository readiness discovered during the audit

The schedule must reflect the repository that actually exists:

- The repository contains the thesis PDF but no identifiable Diffmux implementation.
- `difflogic-light-master` currently supports CIFAR-10 and CIFAR-100, not MNIST, Fashion-MNIST, HAR, PAMAP2, or MLPerf Tiny.
- The original `difflogic` code supports MNIST and CIFAR-10, but it does not contain Diffmux or multi-exit execution.
- The compiled C backend assumes one sequential logic network followed by one `GroupSum`; it has no runtime route configuration, intermediate exits, or fault-repair table.
- No Verilog/SystemVerilog exporter or synthesis scripts were found in the repository.
- In the current environment, Yosys, ABC, OpenROAD, Icarus Verilog, Verilator, Vivado, and Quartus are not available on `PATH`.
- The Light DLGN default CIFAR experiment uses 250,000 training iterations and a width of 128,000 gates, so dozens of full-scale five-seed runs cannot be treated as a one-week task.

These are fixable engineering gaps, but they change every idea's feasibility rating. By July 26, the project needs (1) the student's source or a validated reimplementation, (2) a measured training-time estimate for one representative run, and (3) a decision on exactly one deployment target. Without those three items, a hardware-backed DATE paper is not a realistic plan.

---

## ElasticMux-DLGN (Rank 2): Runtime-Selectable Depth and Accuracy

### Paper question

Can one hardened DLGN provide several runtime-selectable accuracy, latency, and energy operating points by using multiplexers to bypass optional logic stages, while remaining close to independently trained static DLGNs at every operating point?

### Target use cases

- a battery-powered edge device that changes mode as its battery state changes;
- a thermally constrained accelerator that temporarily disables refinement stages;
- a real-time controller that chooses a depth from the remaining deadline budget;
- an always-on sensor that normally uses a shallow classifier and enables deeper inference during important events;
- one accelerator shared by tasks with different service-level objectives;
- graceful service under voltage or frequency reduction, without loading another model.

The recommended first paper uses an **external runtime constraint** such as a latency or energy budget. Per-sample confidence-based early exit can be added later, but it requires a confidence circuit and introduces another learned policy that can distract from the hardware contribution.

### Central reinterpretation of Diffmux

In the thesis, the Diffmux address is learned and then permanently hardened. ElasticMux changes the ownership of that address:

| Diffmux in the thesis | ElasticMux-DLGN |
|---|---|
| Address is a model parameter | Address is a runtime control input or a small mode-table entry |
| One route is retained after hardening | Several legal routes remain in the deployed circuit |
| Same depth for every inference | Active depth is selected for each inference or batch |
| MUX normally collapses to a wire | MUX remains physical and its cost is reported |
| Primary purpose is learned width reduction | Primary purpose is a runtime accuracy-cost trade-off |

The differentiable multiplexer remains useful during supernetwork training, but logarithmic address learning is not the novelty. The contribution is the construction and training of **nested hardened DLGN modes**, together with hardware mechanisms that prevent inactive logic from consuming dynamic energy.

### High-overhead variant: fixed-width refinement stages

Start with a feature bus of fixed width `W`. Each optional stage contains a DLGN refinement block `F_l` and a bypass path `B_l`. The runtime enable `q_l` selects the next representation:

```math
r_l = F_l(h_l),
\qquad
h_{l+1} =
\operatorname{MUX}\left(q_l, B_l(h_l), r_l\right).
```

Use `B_l(h_l)=h_l` when the width is unchanged. If a stage changes width or spatial resolution, use a small fixed Boolean projection and count it as part of every mode that traverses the stage. Avoid arbitrary width changes in the first implementation because they make the bypass expensive and complicate fair latency comparisons.

Define ordered modes by prefix activation:

```math
q_l(d)=\mathbb{1}[l\le d],
\qquad d\in\{d_1,d_2,\ldots,d_M\}.
```

This gives a monotonic hardware structure: mode `d_1` uses the mandatory stem and first refinement stage, while mode `d_M` uses the complete network. Restricting the legal mode vectors to prefixes avoids an exponential set of arbitrary bypass combinations and makes timing, training, and verification manageable.

The hidden cost is severe. A width-`W` bypass boundary needs `W` physical 2-to-1 multiplexers, and operand isolation can add another `W` simple gates. For `S` bypassable equal-width stages,

```math
N_{\mathrm{mux}}=\sum_{l=1}^{S}W_l,
\qquad
A_{\mathrm{overhead}}\approx
\sum_{l=1}^{S}W_l(A_{\mathrm{mux}}+A_{\mathrm{iso}}).
```

An equal-width DLGN layer itself contains approximately `W` learned two-input gates. Consequently, a per-feature bypass boundary can cost the same order of hardware as an entire layer; a 4-to-1 MUX is worse. This architecture is not the recommended first implementation. Before training it, synthesize a microbenchmark containing one DLGN layer, one vector MUX, and one isolation bank in the selected library/device. Abandon pervasive bypasses if that boundary already exceeds the final overhead budget.

### Recommended first architecture: prefix exits

The lowest-risk prototype attaches a hardened classifier at several depths:

```text
input -> stem -> stage 1 -> exit A
                         -> stage 2 -> exit B
                                    -> stage 3 -> exit C
```

The external mode selects `exit A`, `B`, or `C`. In software, evaluation stops at the selected exit. In a pipelined accelerator, later stages receive no new valid token and their registers use clock enable. The output selector is narrow compared with a feature-wide bypass. This architecture should be implemented first because it establishes whether useful depth-accuracy operating points exist before adding pervasive bypass multiplexers.

An intermediate `GroupSum` is not automatically cheap. Its vote gates, adders/popcounts, and class comparison must be included in the exit cost. A small class-aligned exit slice should be tested against applying `GroupSum` directly to a large intermediate layer.

Multiple exits and refinement bypasses answer slightly different questions:

- **Multiple exits** minimize prototype risk and give straightforward variable latency.
- **Refinement bypasses** share a final head and use Diffmux more directly, but require all bypassed representations to remain compatible.

If both work, use prefix exits as the low-overhead architecture and feature-wide refinement bypass as an ablation. Do not assume that the more Diffmux-like design is preferable when its MUX bank costs as much as the logic it bypasses.

Prefix exits also weaken the connection to the student's Diffmux: the MUX only selects a small output bus, while the real mechanism is early termination. The paper must be presented as elastic DLGN co-design, not as a novel multiplexer formulation.

### Optional extension: nested width modes

After depth modes work, divide each layer into a mandatory core and progressively enabled gate groups:

```text
mandatory gates | refinement group 1 | refinement group 2
```

Mode `m` enables the first `m` groups. Downstream routes must be trained so that a disabled group is never required by a lower mode. A source MUX can choose a mandatory substitute when an optional source is unavailable.

This extension can create finer operating points, but it should not be part of the minimum DATE contribution. Supporting both arbitrary width and arbitrary depth would make the configuration space too large for the available schedule.

### Training method

Train one supernetwork by sampling an operating mode for every mini-batch. All enabled gates use hard-forward logic during the final training phase. A suitable objective is

```math
\mathcal{L} =
\sum_{m=1}^{M}\lambda_m\mathcal{L}_{\mathrm{task}}^{(m)}
+ \lambda_d\sum_{m<M}
  D_{\mathrm{KL}}\left(p^{(M)}_{T}\;\|\;p^{(m)}_{T}\right)
+ \lambda_r\mathcal{L}_{\mathrm{rank}}
+ \lambda_g\mathcal{L}_{\mathrm{gap}},
```

where:

- `M` is the deepest teacher mode;
- `lambda_m` controls how often each operating point matters;
- the distillation term transfers the deep mode's class distribution to shallower modes;
- `L_rank` discourages a deeper mode from having greater per-sample loss than the preceding mode;
- `L_gap` penalizes soft-to-hard disagreement at every exit.

For two adjacent modes, use

```math
\mathcal{L}_{\mathrm{rank}}^{(m)}=
\max\left(0,
\ell^{(m+1)}-\ell^{(m)}+\delta
\right)
```

where mode `m+1` is deeper than mode `m`. This penalizes a deeper mode whose loss exceeds the shallower mode's loss; a zero margin is the safe starting point. Report whether this term actually improves monotonic test accuracy rather than assuming that it does.

Use a staged training schedule:

1. train the deepest network to a stable hard accuracy;
2. attach the intermediate exits or bypasses;
3. sample all modes while initially freezing the deepest path;
4. jointly fine-tune every mode with distillation;
5. harden all gates while continuing to sample modes;
6. calibrate exit thresholds or the external mode table only on validation data.

Compare this schedule with training all modes from initialization. Progressive training is likely more stable, but it is not automatically better and requires an ablation.

### Runtime controller

The simplest controller is a three-entry table:

| Runtime condition | Mode | Enabled computation |
|---|---|---|
| tight deadline, low battery, or thermal warning | Economy | mandatory stem and shallow exit |
| normal operation | Balanced | stem plus intermediate stages |
| accuracy-critical request | Accurate | complete network |

Only a few mode bits are required. The system controller can choose them from battery state, temperature, task priority, or an externally supplied latency budget. Mode changes can occur per sample, per batch, or over a longer operating interval.

An optional adaptive controller can estimate confidence at an intermediate exit and continue only when the vote margin is small. This is attractive but has hidden cost: GroupSum or vote-margin computation, comparison logic, calibration storage, and control latency. Evaluate it only after the external-mode system is complete.

### Hardware mechanisms required for real savings

A MUX that selects a shallow output changes the result but does **not** by itself stop the deeper gates from toggling. The implementation must match the deployment substrate.

#### Bit-packed CPU implementation

- Dispatch only the selected prefix of layer kernels.
- Do not compute unused exits or later layers.
- Report actual instructions, cycles, wall-clock latency, and energy when measurable.
- This is the easiest platform on which runtime depth produces real latency savings.

#### Pipelined FPGA implementation

- Put stage boundaries at pipeline registers.
- Use clock enables or valid-token propagation so disabled stages do not accept new work.
- Hold their inputs stable to suppress combinational switching.
- Stop after the selected exit and assert completion immediately.
- Do not use fabric-generated clocks; use the target FPGA's supported clock-enable structure.

#### Combinational FPGA or ASIC implementation

Add operand isolation at each optional block:

```math
\widetilde{h}_l = e_l \land h_l,
```

where `e_l=0` forces a stable input when the block is disabled. The bypass must not pass through the isolated block. Operand isolation reduces switching activity but does not remove leakage or physical area.

Variable latency also needs a completion protocol. A circuit synthesized to one conventional clock period is normally constrained by the deepest active path even when a shallow output is selected. Use one of:

- a pipelined multi-cycle interface where shallow modes return after fewer stages;
- separate characterized timing constraints for static operating modes;
- an asynchronous or valid/ready completion signal;
- DVFS selected together with the depth mode.

Power gating is an optional ASIC extension. It requires isolation cells, state handling, wake-up latency, and a break-even analysis. It is too large for the core contribution unless an existing flow already supports it.

### What can and cannot improve at runtime

| Quantity | Can improve? | Condition |
|---|---|---|
| Executed Boolean operations | Yes | Software skips or hardware stages are disabled |
| Dynamic switching energy | Yes | Inputs/registers of inactive stages are isolated |
| Latency in cycles | Yes | Pipelined or sequential execution stops at the chosen exit |
| Wall-clock combinational delay | Sometimes | Variable completion or per-mode timing is implemented |
| Leakage power | Usually little | Requires power gating for a substantial reduction |
| Physical area/LUT count | No | The largest mode remains instantiated |
| Model storage versus several separate models | Yes | Modes share gates and routing |
| Accuracy | Normally increases with depth | Must be demonstrated; it is not guaranteed by construction |

Never describe inactive gates as removed resources. Runtime configuration trades active computation against accuracy; it does not reduce the fabricated area unless partial reconfiguration loads a different circuit.

### Novelty boundary

Runtime-configurable DNNs, slimmable networks, and early-exit networks are established research areas. [Runtime Configurable Deep Neural Networks for Energy-Accuracy Trade-off](https://arxiv.org/abs/1607.05418) incrementally trains channel groups that can be selected at runtime. More directly, [Binary Early-Exit Network](https://arxiv.org/abs/2206.09029) combines binary networks with intermediate exits, and [ATHEENA](https://arxiv.org/abs/2304.08400) automates FPGA early-exit architectures and reports throughput/resource improvements. Therefore, neither "choose a smaller binary network at runtime," "add intermediate exits," nor "put a MUX around a block" is new.

The targeted search for this memo did not identify a DLGN that exposes jointly trained, runtime-selectable hardened depths with measured isolation and hardware behavior. Current DLGN work, including the [original DLGN](https://arxiv.org/abs/2210.08277), connection optimization, LILogicNet, and the fully trainable DLGN/LUTN, produces a static circuit after training.

The targeted search did not reveal this exact mechanism in a DLGN, but applying established early-exit ideas to a DLGN is only a domain transfer. A full DATE paper needs at least one additional DLGN-specific result. Plausible options are a prefix compiler that exploits the static Boolean graph to omit complete unused cones, an exact vote-bound continuation rule tied to `GroupSum`, or a demonstrably lower exit/control overhead than early-exit BNN hardware. Select one; do not attempt all three.

The strongest defensible combination is:

1. nested, hard Boolean operating modes trained in one DLGN supernetwork;
2. multiplexer-controlled refinement or exit selection with only a few runtime bits;
3. DLGN-specific deep-to-shallow training that controls the hard accuracy gap at every mode;
4. operand isolation or stage disabling that makes the unused Boolean network physically inactive;
5. a measured accuracy-latency-energy Pareto frontier and comparison with storing separate static DLGNs;
6. one DLGN-specific compiler, certification, or hardware result beyond ordinary early exit.

A formal IEEE/ACM/Scopus search for elastic Boolean networks, early-exit BNNs, and runtime-reconfigurable logic classifiers is still required before submission.

### Required comparison methods

| Baseline | Purpose |
|---|---|
| Independently trained static DLGN at every target depth | Accuracy upper reference for each operating point |
| Deepest static DLGN always executed | Energy and latency reference |
| Shallowest static DLGN | Minimum-cost reference |
| Naive truncation of the deepest DLGN | Tests whether joint multi-mode training is necessary |
| Multi-exit DLGN without bypass refinement | Separates ordinary early exit from the proposed architecture |
| ElasticMux without distillation/ranking | Tests the training contribution |
| ElasticMux without operand isolation or clock enable | Measures whether MUX selection alone saves energy |
| Several complete DLGNs stored together | Compares total area/storage when every mode must be available |
| Binary Early-Exit Network or another early-exit BNN | Direct binary-model precedent at matched accuracy/latency |
| ATHEENA-style hardware early exit | Required systems reference for FPGA scheduling/resource claims |
| Larger static DLGN at matched area | Ensures the MUX and exit overhead is not better spent on gates |

When possible, include the original DLGN, Light DLGN/IWP, and a hard-forward `Mind the Gap` training baseline. Elastic behavior should not depend on only one gate parameterization.

### Experimental program

#### Phase 0: recover a trustworthy baseline

- Recover the student's Diffmux source, configuration files, and exact train/validation/test protocol. If unavailable, reimplement only the 4-to-1 module and reproduce one thesis result before extending it.
- Use the original DLGN MNIST path only to debug control flow. The Light DLGN repository does not currently support MNIST or Fashion-MNIST.
- Measure wall-clock time and GPU memory for one reduced-width CIFAR-10 run. Use that measurement to budget the final seed count.
- Decide whether the deployment evidence will be compiled CPU execution or one FPGA/ASIC flow. Do not plan all three.

#### Phase A: prove that useful exits exist

- Start from the existing four-layer CIFAR-10 Light DLGN configuration, using a reduced width for the pilot.
- Add two exits, for example after layers 2 and 4. Add a third exit only after the two-mode system works.
- Compare independent two- and four-layer models, naive truncation, separately trained heads, and joint mode training.
- Use three seeds for the pilot and hard validation accuracy for decisions.
- Reject the idea if the deeper mode does not consistently improve hard validation accuracy or if the intermediate head is comparable in cost to the omitted suffix.

#### Phase B: primary result

- Use CIFAR-10 only. Adding a new application data pipeline before establishing the core result is not realistic.
- Use one gate budget until the training and hardware mechanisms pass. Add a second budget only for the final scaling check.
- Evaluate Economy, Balanced, and Accurate modes, with five paired seeds for the proposed method and its two principal baselines.
- Limit secondary ablations to three seeds.
- Use the same augmentation, optimizer-step budget, validation split, and hardening point across comparisons.

#### Phase C: deployment evidence

Choose one path by July 26:

1. **Compiled CPU:** extend `CompiledLogicNet` to generate prefix functions and intermediate `GroupSum` heads, then measure real prefix latency. This is closest to the current repository, but a CPU-only result gives a weaker DATE hardware story.
2. **Pipelined FPGA:** build one RTL/export path, insert stage valid/enable signals, and measure latency, resources, and power on one device. This is the stronger DATE route but requires tools not currently present.
3. **Standard-cell flow:** use one installed synthesis/place-and-route stack and one library. This is viable only if the flow already exists outside this repository.

For FPGA or ASIC results, record switching activity separately for every mode. Synthesis-only operation counts do not demonstrate energy savings.

### Metrics

**Accuracy and training**

- hard accuracy, macro-F1, and soft-to-hard gap for every mode;
- accuracy difference from the independently trained static model of the same depth;
- monotonicity violations, meaning samples or seeds where deeper modes perform worse;
- convergence time, total training time, and peak GPU memory;
- accuracy under mode frequencies different from the training distribution.

**Runtime behavior**

- enabled stages and executed gate operations per inference;
- latency in cycles, nanoseconds/microseconds, and throughput;
- mode-switch latency and control/configuration bits;
- deadline-miss rate and average accuracy under runtime workload traces;
- controller overhead for the selected external-mode controller. Confidence-based continuation is deferred unless it is the chosen DLGN-specific contribution.

**Hardware cost**

- total gate/cell/LUT count and area of the supernetwork;
- number, fan-in, and area of retained multiplexers;
- exit-head and operand-isolation overhead;
- critical path and maximum frequency in every characterized mode;
- dynamic power, leakage/static power, and energy per inference;
- toggle rate inside disabled stages;
- total storage/area compared with deploying all independent static modes;
- total cost compared with making all independently trained modes available. Do not combine FPGA area and software model-storage claims in one number.

**Pareto and system metrics**

- accuracy versus latency;
- accuracy versus energy;
- accuracy versus active gate operations;
- Pareto hypervolume over accuracy, latency, and energy;
- expected energy and expected accuracy under a declared distribution of runtime modes;
- energy-delay product only as a secondary summary, because it can hide accuracy loss.

### Essential ablations

Required:

- independent static models versus naive truncation versus jointly trained exits;
- intermediate-head size and cost;
- deep-to-shallow distillation on/off;
- progressive versus joint-from-scratch training;
- two versus three runtime modes;
- MUX-only selection versus actual prefix stopping or stage disabling.

Deferred unless the core result is already complete:

- feature-wide refinement bypasses;
- nested width modes;
- confidence-based continuation;
- four or more runtime modes;
- power gating;
- multiple hardware substrates.

### Feasibility and main risks

Software feasibility is **medium-high for a two-exit training prototype**, not high for the complete paper. The model wrapper, training loop, hard evaluation, and compiled C backend all assume one final output. Diffmux itself is also missing from the repository. A reliable three-mode compiled implementation is therefore a real code change rather than a configuration edit.

Hardware feasibility is **low until a tool and target are selected**. The current repository has no HDL exporter or synthesis scripts, and the audited environment exposes no synthesis or HDL simulation tools on `PATH`. A purely combinational implementation also makes credible variable latency harder. The major risks are:

- deeper DLGN stages may not improve accuracy enough to create useful modes;
- shared training may reduce the deepest mode's accuracy;
- exit heads may dominate small models;
- feature-wide MUX and isolation banks may cost as much as the bypassed layer;
- MUXes can extend the critical path;
- inactive combinational logic may continue toggling without careful isolation;
- the result may look like a direct import of ordinary early exit unless the hardened Boolean and hardware aspects are substantial.

### Go/no-go criterion

Use staged gates rather than waiting for the final experiment:

1. **By July 26:** Diffmux or the required selector code is recovered/reimplemented, a representative run time is known, and one deployment path is available.
2. **By July 31:** two hardened CIFAR-10 modes differ by at least 1 percentage point in accuracy, the shallow mode is within 1.5 points of an independently trained shallow DLGN, and prefix execution reduces measured CPU time or enabled stages by at least 20%.
3. **By August 7:** a third mode works, the deepest shared mode loses no more than 0.75 points, and exit/control overhead is below 15% in the selected implementation estimate.
4. **For submission:** at least three modes form a monotonic hard-accuracy/energy or hard-accuracy/latency curve; Economy reduces measured energy or latency by at least 25%; final MUX/exit/isolation overhead is at most 10%; and the paper contains one DLGN-specific contribution beyond standard binary early exit.

If only software prefix latency improves, the work may merit an embedded-systems paper, but the DATE claim is weak. If prefix exits work and feature-wide bypasses fail, do not present Diffmux as the method's main novelty.

### Candidate paper statement

> ElasticMux-DLGN jointly trains nested hardened DLGN exits and compiles them into a runtime-selectable implementation that stops or isolates unused Boolean stages, providing measured accuracy-cost operating points with minimal control state.

---

## RepairMux-DLGN (Rank 1): Budgeted Reconfigurable Routing for Fault Repair

### Paper question

Can a DLGN be trained with a small number of retained multiplexers so that permanent gate or route faults can be repaired after manufacturing or deployment by programming only a short address-bit patch?

### Target use case

The same trained DLGN is fabricated as an ASIC or mapped to an FPGA across many devices. Manufacturing test or an in-field self-test identifies a faulty gate output, LUT, or connection. A normal hardened DLGN cannot change its wiring without resynthesis or reconfiguration of the full design. RepairMux retains a limited number of configurable routes, allowing each device to select a healthy, functionally useful source.

This fits DATE topics in test, dependability, reconfigurable systems, and design methods for machine-learning hardware. It is stronger than claiming another small power reduction for an already efficient model.

### Proposed architecture

1. Train and harden the Boolean functions as in Light DLGN or a hard-forward DLGN.
2. Restrict the first candidate set to one late layer or a bounded high-fan-out subset. Scoring every net in a 128,000-wide, multi-layer DLGN with separate fault simulations is not computationally realistic.
3. Rank the candidate producers with a cheap soft-gradient/outage proxy, then validate exact hard stuck-at sensitivity only for the shortlist. Select the top nets under a fixed protection budget; differentiable placement can be explored later only if this ranking is inadequate.
4. For each protected producer, train one candidate spare signal or small spare subcone using different predecessors. A spare that simply duplicates the same vulnerable input cone does not protect against common-cause faults.
5. Place one 2-to-1 MUX before the protected net's fan-out so all consumers see the same repaired signal. Per-consumer MUXes are an extension for individual broken branches, but their area can grow with fan-out and should not be the core design.
6. Encourage the spare to be functionally substitutable for the primary while using a structurally or physically distinct cone. Low common ancestry alone is insufficient: a disjoint signal that computes a different feature is not a valid repair.
7. During training, hold a sampled persistent fault/route choice for several steps. Train both primary and spare modes. Use local hard/soft activation agreement where optimization permits, plus final-logit consistency so that local imitation does not damage the classifier.
8. Precompute the fallback address for each protected net. After manufacturing test identifies a covered fault, configuration should use the fault map and fallback table without labeled per-device optimization.
9. Store the resulting per-device patch as `(mux_id, address_bit)` entries. Calibration-based address search is an optional upper bound, not the default repair path.

A suitable objective is

```math
\mathcal{L} =
\mathcal{L}_{\mathrm{clean}}
+ \lambda_f\,\mathbb{E}_{F\sim\mathcal{D}_F}
  \mathcal{L}_{\mathrm{repair}}(F)
+ \lambda_a\sum_i z_i D\left(p_i(x),s_i(x)\right)
+ \lambda_s\,D\left(f_{\mathrm{primary}}(x),f_{\mathrm{spare}}(x)\right)
+ \lambda_o\mathcal{L}_{\mathrm{overlap}},
\qquad
\sum_i z_i C_i\le B_A,
```

where `p_i` and `s_i` are the primary and spare signals, `z_i` indicates that net `i` is protected, `C_i` includes the spare cone, MUX, configuration bit, and added routing cost, and `B_A` is the hard protection budget. The local and output consistency terms make the spare useful, while the overlap term discourages common vulnerable ancestry. In the minimum method, sensitivity ranking fixes `z` before spare training. Jointly learning `z` is a later extension.

### What is novel, and what is not

**Not novel:** multiplexers, spare wires, fault injection, generic reconfigurable repair, or training DLGNs with random bit flips.

**Potentially novel combination:** a DLGN-specific co-design that (1) allocates a scarce repair-MUX budget from exact Boolean fault sensitivity, (2) trains functionally useful and structurally distinct spare signals, and (3) applies a tiny per-device address patch after external circuit diagnosis.

The 2026 paper [From Arithmetic to Logic](https://arxiv.org/abs/2603.22770) evaluates inherent logic/LUT-network resilience under parameter bit flips. It does not establish trained, structural post-deployment route repair. However, fault-aware mapping and lightweight hardware repair exist for other neural accelerators; [RescueSNN](https://arxiv.org/abs/2304.04041) is a relevant example. The claim must therefore be DLGN-specific substitutable Boolean cones and budgeted repair, not generic fault-aware remapping.

### Required comparison methods

| Baseline | Why it is required |
|---|---|
| Unprotected hardened DLGN/Light DLGN | Establish clean accuracy and natural degradation |
| Fault-aware training without multiplexers | Separates training robustness from actual repair capability |
| Random MUX locations and random spare sources | Tests whether the learned placement and route diversity matter |
| Sensitivity-ranked MUX locations with random spares | Separates placement quality from spare-route training |
| Uniform MUX on every route | Accuracy/reliability upper bound and hardware-cost upper bound |
| Selective gate duplication | Standard low-cost redundancy without re-routing |
| Triple modular redundancy | Conventional reliability reference; compare at matched area |
| Calibration-based address search | Upper bound on precomputed fallback addresses |
| Larger unprotected DLGN at matched post-route area | Prevents extra hardware alone from explaining the gain |

### Fault models

The minimum paper should cover only:

- producer-gate output stuck-at-0 and stuck-at-1;
- a broken primary branch between the producer and its repair MUX;
- one and several simultaneous persistent faults;
- faults on protected and unprotected nets.

Spatial clusters, LUT truth-table upsets, consumer-branch opens, MUX faults, and configuration-memory protection are valuable extensions, but doing all of them would require different repair semantics and verification. For the final selected design, inject MUX/address faults as a limitation check so the repair fabric is not silently assumed perfect; do not make their correction a second contribution.

Assume an external manufacturing or in-field test supplies the fault map. Fault detection and diagnosis are out of scope and must be stated as such.

A break after the repair MUX on the shared fan-out net is not covered by this architecture. Report it as an uncovered fault rather than counting it as a failed repair.

### Experimental plan

Use the original DLGN MNIST code only for graph-level debugging, then use CIFAR-10 with Light DLGN for the primary result. Fashion-MNIST and MLPerf Tiny are not supported by the current local data pipelines and should not be added before the core result.

Start with 2-to-1 producer repair in one selected late layer and protection budgets of 1%, 5%, and 10% of that candidate set. Use one and ten persistent faults plus two normalized rates selected from a pilot. Add whole-network placement, 4-to-1 spares, spatial clusters, or a second gate budget only after the main reliability-area curve exists. Choose primary fault operating points before viewing final test results.

### Metrics

**Predictive/reliability metrics**

- clean hard accuracy;
- hard accuracy before and after repair at each fault rate;
- area under the accuracy-versus-fault-rate curve;
- recovered-loss fraction;
- fraction of faulty devices satisfying an accuracy target, as a yield proxy;
- repair success probability and number of unrepairable fault maps;
- hard primary/spare activation-agreement rate on held-out samples;
- primary/spare ancestor overlap and common-cause-fault coverage;
- worst-case, mean, and 5th-percentile accuracy across fault maps;
- generalization to fault types and clusters not used in training.

Define recovered-loss fraction as

```math
R = \frac{A_{\mathrm{repaired}}-A_{\mathrm{faulty}}}
         {A_{\mathrm{clean}}-A_{\mathrm{faulty}}}.
```

Compute `R` only when `A_faulty < A_clean`; otherwise report the accuracies directly. Values above one indicate that the repaired configuration happens to exceed the nominal clean accuracy and should not be silently clipped.

**Cost metrics**

- number and type of retained multiplexers;
- configuration bits and patch bytes per device;
- fallback-table lookup time and patch bytes; report calibration-search cost only for its upper-bound baseline;
- synthesized cell/LUT count, total area, critical-path delay, maximum frequency, and power;
- post-route wire length and congestion when available;
- energy per inference and reliability improvement per 1% area overhead.

Power, timing, and area claims require the same synthesis/place-and-route flow, library/device, clock constraint, and optimization effort for every method. [FPGN](https://arxiv.org/abs/2607.08427) makes physically aware LUT topology a current baseline expectation; pre-synthesis gate counts alone are insufficient for a DATE hardware claim.

### Essential ablations

- learned versus random MUX placement;
- trained versus random spare sources;
- ancestry-disjoint versus unconstrained candidates;
- clean-only versus persistent-fault training;
- precomputed fallback versus calibration-based search;
- producer-level versus consumer-level placement on one small configuration;
- matched MUX count and matched post-route area.

### Feasibility and dependency gate

Graph-level feasibility is **medium** because Light DLGN already contains persistent/random output-outage hooks that can seed a fault injector. The missing work is spare-signal construction, joint primary/spare training, net-level fan-out semantics, hardened evaluation, and the repair table.

Hardware feasibility is **medium-low**. There is no Diffmux implementation, HDL exporter, or synthesis flow in the repository. The minimum credible paper needs one selected library/device and at least synthesis plus timing; a stronger paper needs placed results or board measurements. Do not promise both FPGA and ASIC evaluation.

Run two early tests before building the full method:

1. Measure how rapidly hard accuracy degrades under persistent producer faults. If the baseline is already insensitive at meaningful fault rates, there is little repair problem to solve.
2. For a small hand-selected protected set, duplicate or train one spare per producer and test whether switching to the spare recovers accuracy. If functional substitutes cannot be trained, sophisticated placement optimization will not rescue the idea.

### Go/no-go criterion

Continue after the first week only if a simple 5% producer-protection pilot recovers at least half of the accuracy lost under a predeclared persistent-fault setting and beats random spares. Continue to submission only if, at no more than 15% synthesized/placed area overhead, RepairMux recovers at least 80% of the loss at one predeclared operating point and gives a clear reliability-area advantage over fault-aware training and selective duplication. A result limited to single faults on MNIST is not sufficient for DATE.

### Candidate paper statement

> RepairMux-DLGN co-trains a hardened logic network with a budgeted set of structurally disjoint spare routes, enabling post-test repair through a small per-device address patch and improving the reliability-area frontier over fault-aware training and conventional redundancy.

---

## ConfigMux-DLGN (Rank 3): Address-Only Adaptation of a Shared Edge Circuit

### Paper question

Can one hardened Boolean core serve multiple users, sensors, or operating domains by updating only retained multiplexer addresses, rather than changing gate truth tables or distributing a new model?

### Target use cases

- cross-subject wearable activity recognition;
- electrode or patient adaptation in ECG/EEG;
- cross-device sensor calibration;
- network-flow classification under site-specific or chronological drift;
- one edge accelerator that switches among several operating profiles.

### Proposed method

Train a shared DLGN core with candidate route banks at selected layers. The gate truth tables/functions are shared and frozen after base training. Each domain `d` receives only an address profile `a_d`:

```math
f_d(x)=f_{\theta^*}(x;\mathbf{a}_d),
```

where `theta*` contains the common hardened gates and `a_d` contains discrete multiplexer addresses. The first experiment should use ordinary multi-domain pretraining followed by address-only fine-tuning. Episodic/meta-learning is an extension only if simple fine-tuning demonstrates that route addresses contain a repeatable adaptation signal. At deployment, harden the address logits and transmit only changed addresses relative to a base profile.

Two deployment modes must be separated:

1. **Single-profile hardwired:** the chosen MUX route is compiled to a wire. This minimizes hardware but cannot change in the field.
2. **Multi-profile configurable:** the multiplexer remains in hardware and its address comes from a small register/SRAM bank. This enables rapid profile switching but has real area, delay, power, and configuration-memory cost.

The paper must not combine the zero-cost claim from mode 1 with the adaptability claim from mode 2.

### Novelty boundary

Connection-optimization DLGNs learn routes once and then harden them. Generic adapters, meta-learning, configurable networks, and task-specific binary masks already adapt small discrete parameter sets. [Piggyback](https://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html), for example, adapts a frozen network using learned binary masks. The possible new contribution is narrower: **a measured edge system in which per-domain state changes physical DLGN routes while every Boolean function remains shared**.

This is also distinct from `PatchLogic` in [july_21.md](july_21.md), which changes truth-table entries, and `PersonalDLGN` in [july_20_no_DATE.md](july_20_no_DATE.md), which changes input thresholds. Neither method is implemented in the current repository, so requiring both as full baselines would create two additional research projects. Implement one simple threshold or gate-patch baseline only after address-only adaptation passes its pilot.

### Required comparison methods

- no adaptation;
- full-model fine-tuning and a complete per-domain checkpoint;
- output-head-only adaptation;
- address fine-tuning without episodic preparation;
- random address changes with the same patch size;
- candidate-softmax connection adaptation using `K` logits per route;
- Piggyback-style binary masking or another discrete adapter at matched profile bits;
- one threshold-only or sparse gate-patch baseline if the pilot succeeds.

Connection-learning references should include the local notes on [connection optimization](../notes/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.md) and [LILogicNet](../notes/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.md).

### Experimental plan

Use one application family for the pilot. CIFAR-10-C is the lowest-integration option because it preserves the CIFAR dimensions already supported by the Light DLGN code. Generate the selected corruptions on CIFAR-10 training/validation images for supervised address adaptation, and reserve the official CIFAR-10-C test images for evaluation. Do not fine-tune on the official corrupted test samples. If this works, a later paper can add leave-subject-out HAR or PAMAP2 with 0, 10, 25, 50, and 100 labeled examples.

Start with 2-to-1 MUXes at one selected layer and profile budgets of 0.5%, 1%, and 2% of the full model state. Early/middle/late placement, 4-to-1 MUXes, and five budgets are second-stage ablations. Use per-domain validation sets; never choose the patch budget on the target test set.

### Metrics

- hard accuracy and macro-F1 before and after adaptation;
- per-class recall/sensitivity for imbalanced sensing tasks;
- adaptation gain versus number of labels;
- gap to full fine-tuning;
- forgetting or source-domain regression;
- number of changed MUXes, address bits, compressed patch bytes, and total profile-bank bytes;
- adaptation steps, wall-clock time, and peak training memory;
- profile-switch latency and configuration energy;
- configurable-MUX area, critical path, power, and energy per inference;
- accuracy per adapter byte and Pareto hypervolume over accuracy, bytes, and area.

Report both the absolute address-profile size and the delta-patch size. A selected connection already requires an index, so a binary address is not automatically smaller than every other hardened routing representation; the storage gain must come from sharing the core and changing few routes.

### Essential ablations

- ordinary versus episodic base training;
- address-only versus truth-table-only versus threshold-only updates;
- sparse delta versus complete address profile;
- one shared candidate bank versus domain-specific candidate banks;
- MUX location and candidate count;
- hard-forward versus soft-forward address training;
- field-configurable versus separately synthesized profiles.

### Feasibility and merit

Software feasibility is **medium-low** because the repository has no Diffmux code, no domain-adaptation loop, and no CIFAR-10-C loader. The method also assumes that changing a small number of routes can alter domain behavior without invalidating downstream Boolean features; this is plausible but unproven.

The system merit is real only when many profiles share one physical core and the device must switch profiles repeatedly. For one target domain, separately hardening the adapted route to a wire is simpler and cheaper. The evaluation must therefore include at least several domains and total storage/hardware across all profiles.

### Go/no-go criterion

Do not schedule this as the September paper unless a one-layer, 1%-profile pilot improves hard validation accuracy by at least 2 points over no adaptation on multiple corruptions and clearly beats random address changes. For a later full paper, target a gap within 1.5 points of full fine-tuning, no more than 2% model bytes per profile, and less than 15% configurable hardware overhead. It must also outperform a binary-mask adapter at matched profile bits.

### Candidate paper statement

> ConfigMux-DLGN prepares a shared hardened Boolean core for few-shot domain adaptation through sparse route-address profiles, providing near-full-fine-tuning accuracy with measured profile-storage and reconfiguration costs.

---

## HypercubeMux-DLGN (Rank 4): Code-Aware Connection Learning

### Audit verdict

This is **not recommended as a standalone DATE 2027 paper**. DSelect-k already establishes binary-coded differentiable selection, the thesis already instantiates the 4-way case, and the proposed code assignment currently lacks either a theorem or evidence that it improves DLGN optimization. It is retained as a bounded exploratory experiment because it is relatively inexpensive after Diffmux is available.

### Why the plain idea is insufficient

For `K=2^m`, Diffmux can select candidate `i` using `m` relaxed address variables:

```math
y=\sum_{i=0}^{K-1}x_i
\prod_{j=0}^{m-1}
a_j^{b_j(i)}(1-a_j)^{1-b_j(i)}.
```

This provides `m` trainable address values instead of `K` categorical logits, but the construction is already present in DSelect-k. At hardening, both forms store a selected source index using approximately `log2(K)` bits. Furthermore, exact soft evaluation still touches all `K` candidates, so lower parameter count does not automatically imply lower training time or activation memory.

The only bounded question worth piloting is: **does candidate-to-code assignment materially change optimization stability for larger Diffmux selectors?** Physical locality and fault tolerance should not be added unless that basic effect exists.

### Proposed method

Treat the `m`-bit codewords as vertices of a hypercube. For a narrow pilot, assign candidate signals using activation distance on a fixed calibration subset. A similarity assignment solves

```math
\min_{\pi}
\sum_{d_H(u,v)=1}
D_{\mathrm{act}}(x_{\pi(u)},x_{\pi(v)}).
```

Negating or maximizing the same objective produces the deliberately dissimilar control. Use recursive balanced partitioning or a small local-swap optimizer to find `pi`; do not introduce a dense learned permutation that removes the claimed parameter advantage. Compare only random assignment, ordinary index order, similarity-clustered assignment, and deliberately dissimilar assignment.

The effect of similarity on training is ambiguous. Hamming-neighbor similarity can make address-bit flips less damaging, but it can also shrink the gradient signal used to distinguish alternatives. Gradient cancellation depends on signed contrasts across the hypercube partitions, not similarity alone. The experiment must test both similar and dissimilar assignments rather than writing the preferred result into the hypothesis. The thesis's failed 16-to-1 experiment is the key stress case rather than a result to omit.

### Required comparison methods

- fixed random DLGN connectivity;
- `K`-candidate softmax connection selection;
- DSelect-1 with random code assignment;
- original Diffmux at `K=4`;
- similar, dissimilar, random, and ordinary index assignments;
- direct narrow DLGN without a separate MUX layer.

LILogicNet and fully trainable DLGN connectivity remain required related work, but implementing them for a one-week stop/go pilot is not realistic unless compatible code is obtained.

The direct-narrow baseline is essential. A normal DLGN layer can use arbitrary predecessors, so a Diffmux contraction layer must demonstrate value beyond simply reducing the next layer's width and learning its connections directly.

### Experimental plan and metrics

Use the original DLGN MNIST path for debugging and one reduced-width CIFAR-10 Light DLGN for the actual pilot. Evaluate `K` in `{4,8,16}` with three seeds. Test `K=32`, more gate budgets, or five seeds only if `K=16` is stable.

Report:

- hard accuracy and soft-to-hard accuracy gap;
- success rate of reaching each candidate source;
- address entropy and bit saturation;
- gradient norm/variance and convergence failures;
- routing-logit parameters and optimizer-state bytes;
- measured peak GPU memory, step time, and time to target accuracy;
- final hardwired model size, reported separately from training state.

### Essential ablations

- similar, dissimilar, random, and ordinary index assignments;
- the thesis address relaxation versus one smooth-step alternative;
- progressive `K=4 -> 8 -> 16` training versus training at final `K`;
- fixed versus one-time recomputed source fingerprints.

### Go/no-go criterion

Spend no more than one week on this pilot. Do not submit a paper whose only positive result is `K/log2(K)` fewer routing logits. Promote it to a paper candidate only if assignment makes `K=16` materially more stable than random/DSelect-1 encoding across seeds and stays within 0.5 points of categorical routing with at least a 4x router-state reduction. Otherwise record the negative result and stop.

### Candidate paper statement

> HypercubeMux-DLGN studies whether candidate-to-code assignment can overcome the optimization instability of binary-address DLGN routing at larger candidate counts.

This is not the recommended first choice because its novelty depends on the code-assignment result, not the multiplexer equation.

---

## VectorMux (Rank 5): Input-Dependent, Vector-Coherent Boolean Pooling

### Audit verdict

This idea has an interpretable representational hypothesis, but it is not feasible as the DATE 2027 primary project from the current repository. The local Light DLGN `cnn` path is a conventional PyTorch CNN rather than the Convolutional DLGN architecture, the student's Diffmux code is absent, and the thesis starts 5.70--8.25 hard-accuracy points behind OR pooling. It would require acquiring or rebuilding the convolutional logic model before testing the new selector.

### Why the thesis version did not pool

The thesis address variables are model parameters. Every input example therefore selects the same local source after hardening. A true adaptive selector must compute its address from the current local feature block.

For scalar binary activations, maximum pooling is exactly OR. A multiplexer that selects one scalar cannot reproduce OR in general and is unlikely to beat it. The more defensible target is a **multi-channel feature vector**: select one complete spatial vector from a local window so that all channels preserve the same spatial origin.

### Proposed method

For a 2-by-2 window containing four `C`-bit feature vectors, use a small shared Boolean selector `g_phi` to compute two hard address bits from the current window:

```math
\mathbf{a}(X)=g_{\phi}(X),
\qquad
Y_c=X_{q(\mathbf{a}(X)),c},\quad c=1,\ldots,C.
```

The same two address bits control all `C` channel multiplexers. This preserves a vector that actually occurred at one spatial position. In contrast, channelwise OR pooling can combine channel bits from different positions into a synthetic vector. Train the selector with a hard-forward straight-through estimator; retain it as Boolean logic at inference.

Possible selector inputs, in increasing order of cost, are:

1. per-position popcount or small Boolean summaries;
2. a shallow shared DLGN over the 2-by-2-by-`C` block;
3. local summaries plus a small context signal from the preceding layer.

Begin with option 1, but count its arithmetic honestly: a popcount-based selector needs add/compare logic and may be more expensive than a small Boolean selector.

VectorMux and OR pooling both reduce four `C`-bit positions to one `C`-bit output. VectorMux therefore does **not** inherently remove more downstream gates than OR. A four-way VectorMux requires a selector plus a 4-to-1 MUX for every channel, whereas four-input OR requires an OR tree. The default expectation is higher pooling area and delay, not a 15% area saving. Any hardware benefit must come indirectly from improved accuracy permitting fewer channels or fewer downstream gates.

### Novelty boundary

[Convolutional DLGNs](https://openreview.net/forum?id=4bKEFyUHT4) already use logical OR pooling. Input-dependent pooling is well established in arithmetic CNNs through methods such as [Local Importance Pooling](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.html) and [SoftPool](https://openaccess.thecvf.com/content/ICCV2021/html/Stergiou_Refining_Activation_Downsampling_With_SoftPool_ICCV_2021_paper.html). The possible novelty is narrower: an all-Boolean, input-dependent selector that forwards a **coherent channel vector** and is evaluated as an actual synthesized circuit.

### Required comparison methods

- channelwise OR/max pooling;
- fixed-stride subsampling;
- the thesis's static Diffmux;
- random input-dependent position selection;
- a learned categorical/Gumbel position selector;
- independent per-channel multiplexers;
- strided logic convolution or a direct narrow convolutional DLGN;
- matched-area OR pooling with additional downstream gates;
- LIP and SoftPool only as contextual arithmetic-CNN references, not as hardware-matched Boolean baselines.

### Experimental plan

Use CIFAR-10 because the thesis already exposes the gap there. First recover the convolutional DLGN baseline and reproduce OR pooling. Then test one 2-by-2 pooling location, one shared-address selector, and three seeds. A second pooling location, per-channel addresses, 3-by-3 windows, and five final seeds are justified only if the first selector matches OR pooling closely.

### Metrics

- hard CIFAR-10 accuracy and soft-to-hard gap;
- total gates, selector gates, live gates after dead-cone elimination, and connection count;
- output-vector realizability: an explanatory diagnostic, noting that it is 100% by construction for VectorMux and therefore not an independent success metric;
- selected-position entropy and class-conditional position distribution;
- translation, occlusion, and corruption robustness;
- synthesized LUT/cell count, area, delay, power, and energy;
- activity factor and switching caused by the retained selector;
- accuracy per area and accuracy per nanosecond.

### Essential ablations

- static versus input-dependent addresses;
- one address per vector versus one per channel;
- popcount rule versus learned Boolean selector;
- selector context on/off;
- selector overhead versus downstream gate savings;
- OR pooling with the same total post-synthesis area.

### Go/no-go criterion

Do not allocate DATE-critical time to this idea. For a later project, continue only if the first input-dependent prototype reaches within 1 point of OR pooling and provides a credible path to either (1) higher accuracy than OR at no more than 20% total-network area overhead, or (2) an accuracy gain that allows at least 20% channel/downstream-gate reduction at matched accuracy. If it only improves over static Diffmux but remains several points behind OR, it has no publication merit as a pooling replacement.

### Candidate paper statement

> VectorMux introduces input-dependent, shared-address Boolean pooling that preserves the spatial coherence of multi-channel logic features and tests whether that coherence can offset the selector's additional circuit cost.

This is the most conceptually direct continuation of the student's pooling goal, but it is also the highest-risk option.

---

## Shared Experimental Rules

These rules apply to whichever single idea is selected.

### Evaluation discipline

1. Split train, validation, and test data before tuning. Do not use test accuracy for configuration selection or early stopping.
2. Report the hardened network as the primary result. Soft accuracy is diagnostic only.
3. Use three paired seeds for feasibility pilots and at least five for the principal final comparison. Do not spend five seeds on an idea that has not passed its effect-size gate.
4. Report mean, standard deviation, and 95% confidence intervals. Use a paired test or bootstrap interval for the principal difference.
5. Predeclare an accuracy non-inferiority margin, normally 0.5 or 1.0 percentage point depending on the task.
6. Match baselines in several meaningful ways: gate count, training parameters, post-synthesis area, and latency. One matched-gate table is not enough.
7. Run dead-cone and constant propagation before hardware comparison, and apply the same optimization to all models.

### Accounting discipline

Report these quantities separately:

- gate-function training parameters;
- route/address training parameters and optimizer state;
- hardened gate bits and connection indices;
- retained physical multiplexer count;
- configurable address bits and error-protection bits;
- per-profile or per-device patch bytes;
- complete synthesized and placed hardware cost.

This prevents three invalid shortcuts: counting a training parameter reduction as an inference memory reduction, treating a configurable MUX as a free wire, or attributing savings from a narrower downstream layer to the selector itself.

### Hardware flow

The current repository has no HDL exporter or synthesis flow, and the audited environment has no common synthesis tool on `PATH`. Select exactly one deployable flow by July 26. Generate hard Boolean netlists for every principal baseline and apply the same optimization. Record tool versions, target device/library, clock constraints, switching-activity method, and whether figures are pre- or post-route. Do not claim power savings from operation counts alone. If only compiled CPU results are available, describe them as software execution measurements rather than ASIC/FPGA PPA.

---

## Eight-Week Decision Schedule

The official [DATE 2027 Call for Papers](https://www.date-conference.com/date-2027-call-papers) lists abstract registration on **September 13, 2026 AoE** and final submission on **September 20, 2026 AoE**.

| Dates | Required outcome |
|---|---|
| July 22--26 | Recover or reimplement Diffmux; reproduce one thesis result; measure one representative run; choose and verify one deployment flow |
| July 27--31 | Run two cheap pilots: RepairMux fault sensitivity plus one hand-built spare, and ElasticMux two-exit hard accuracy plus prefix timing |
| August 1 | Select exactly one project. Choose RepairMux if substitutes recover faults; otherwise choose ElasticMux only if useful depth modes exist |
| August 2--9 | Implement the selected minimum method and its two strongest baselines; obtain three-seed reduced-width CIFAR-10 results |
| August 10 | Submission viability gate: require a clear hard-accuracy effect, a credible novelty statement, and a working netlist/measurement path |
| August 11--23 | Run the primary CIFAR-10 experiment with five paired seeds and the mandatory ablations; keep one main gate budget |
| August 24--September 2 | Complete the selected synthesis/measurement flow and the matched-cost comparison |
| September 3--8 | Freeze results; add a second budget only if the main story is complete; generate statistics and final figures |
| August 24--September 12 | Write concurrently with hardware runs; complete limitations, related work, and artifact documentation |
| September 13 | Register title, abstract, and authors |
| September 14--20 | Final verification, related-work audit, and submission |

## Final Recommendation

Use the tentative title:

> **RepairMux-DLGN: Budgeted Reconfigurable Repair for Differentiable Logic Gate Networks**

Build the paper around three claims only:

1. selected DLGN producer nets receive functionally trained, structurally distinct spare signals under a hard MUX budget;
2. diagnosed covered faults are repaired using precomputed address patches without retraining gate functions;
3. measured results improve the reliability-area frontier over fault-aware training, random spares, and selective duplication.

Use ElasticMux as the backup only if the repair pilot fails and the two-exit pilot shows a real depth-accuracy trade-off with measured prefix savings. Even then, add a DLGN-specific compiler, certificate, or hardware result beyond standard binary early exit. Do not switch to ConfigMux, HypercubeMux, or VectorMux for the September deadline without pre-existing positive data. If neither primary pilot passes by August 1, stop forcing a Diffmux-derived DATE paper and return to the lower-dependency ideas in [july_21.md](july_21.md).

### Overall merit verdict

- **RepairMux has genuine paper merit**, because a learned Boolean circuit with budgeted post-test structural repair is a specific DLGN/hardware problem. Its feasibility is conditional rather than high.
- **ElasticMux has useful systems merit**, but its current form is close to established binary early exit. It needs unusually good hardware results or a DLGN-specific mechanism to support a full DATE contribution.
- **ConfigMux is a reasonable longer project**, particularly for a device serving many domains, but binary-mask adaptation substantially narrows its novelty and the required data/training infrastructure is absent.
- **HypercubeMux does not presently justify a paper.** The encoding is prior art and the code-assignment hypothesis is unresolved.
- **VectorMux is scientifically testable but presently unfavorable.** It begins well behind OR pooling, has higher expected pooling cost, and lacks a local convolutional DLGN implementation.

## Primary Sources for the Novelty Audit

- [DSelect-k: Differentiable Selection in the Mixture of Experts](https://papers.nips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)
- [Runtime Configurable Deep Neural Networks for Energy-Accuracy Trade-off](https://arxiv.org/abs/1607.05418)
- [Binary Early-Exit Network for Adaptive Inference](https://arxiv.org/abs/2206.09029)
- [ATHEENA: A Toolflow for Hardware Early-Exit Network Automation](https://arxiv.org/abs/2304.08400)
- [When Do Early-Exit Networks Generalize?](https://arxiv.org/abs/2604.15764)
- [Piggyback: Adapting a Single Network with Binary Masks](https://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://openreview.net/forum?id=4bKEFyUHT4)
- [A Method for Optimizing Connections in DLGNs](https://arxiv.org/abs/2507.06173)
- [LILogicNet](https://arxiv.org/abs/2511.12340)
- [Operand-Selective Logic Gate Network](https://openreview.net/forum?id=sfqIc7BnwS)
- [Fully Trainable Deep DLGNs and LUTNs](https://arxiv.org/abs/2607.09399)
- [From Arithmetic to Logic: Resilience Under Parameter Bit-Flips](https://arxiv.org/abs/2603.22770)
- [RescueSNN: Fault-Aware Mapping Under Permanent Faults](https://arxiv.org/abs/2304.04041)
- [FPGN: Physically Aware Differentiable LUT Networks](https://arxiv.org/abs/2607.08427)
- [Local Importance-Based Pooling](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.html)
- [SoftPool](https://openaccess.thecvf.com/content/ICCV2021/html/Stergiou_Refining_Activation_Downsampling_With_SoftPool_ICCV_2021_paper.html)
- [DATE 2027 Call for Papers](https://www.date-conference.com/date-2027-call-papers)

The negative novelty findings in this memo are based on the repository notes, a repository/tool audit, and a targeted primary-source search updated July 22, 2026. Before submission, run a formal search in IEEE Xplore, ACM Digital Library, Scopus, and patent databases, especially for runtime-elastic Boolean networks, early-exit BNN hardware, programmable fault bypass, reconfigurable logic repair, and task-specific routing masks.
