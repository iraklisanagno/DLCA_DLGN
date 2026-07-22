# Diffmux-Derived Research Ideas for DATE 2027

**Prepared:** July 21, 2026  
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

The best balance of novelty, feasibility, and direct continuity with the thesis is **Idea 1, ElasticMux-DLGN**. It retains multiplexers in the deployed design and drives their addresses with runtime quality-of-service controls, allowing one hardened Boolean network to expose several depth, latency, and accuracy modes. The strongest higher-risk alternative is **Idea 2, RepairMux-DLGN**, which uses retained multiplexers for per-device fault repair. **Idea 3, ConfigMux-DLGN** is the recommended application-oriented backup.

### Ranked shortlist

| Rank | Working title | Publishable contribution | Novelty confidence | Feasibility by September 20 | Main risk |
|---:|---|---|---|---|---|
| 1 | ElasticMux-DLGN | Runtime-selectable depth and accuracy in one hardened Boolean supernetwork | Medium-high | High | Generic early-exit networks exist; physical savings require real isolation |
| 2 | RepairMux-DLGN | Budgeted, trained spare routes for per-device fault repair | High | Medium-high | Requires credible synthesis and fault evaluation |
| 3 | ConfigMux-DLGN | Address-only adaptation profiles for a shared edge circuit | Medium-high | High | Generic adapter literature is adjacent |
| 4 | HypercubeMux-DLGN | Function- and locality-aware address-code assignment | Medium | Medium-high | Binary encoding itself is prior art |
| 5 | VectorMux Pooling | Input-dependent, vector-coherent Boolean downsampling | Medium | Medium | Current thesis result is 5.70--8.25 points behind |

These are **separate papers**. Combining runtime elasticity, repair, personalization, code assignment, and pooling into one submission would obscure the main claim and is not feasible within the DATE schedule.

---

## Idea 1 -- ElasticMux-DLGN: Runtime-Selectable Depth and Accuracy

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

### Recommended architecture: fixed-width refinement stages

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

### Alternative architecture: multiple exits

The lowest-risk prototype attaches a small hardened classifier at several depths:

```text
input -> stem -> stage 1 -> exit A
                         -> stage 2 -> exit B
                                    -> stage 3 -> exit C
```

The external mode selects `exit A`, `B`, or `C`. In software, evaluation stops at the selected exit. In a pipelined accelerator, later stages receive no new valid token and their registers use clock enable. This architecture should be implemented first because it establishes whether useful depth-accuracy operating points exist before adding pervasive bypass multiplexers.

Multiple exits and refinement bypasses answer slightly different questions:

- **Multiple exits** minimize prototype risk and give straightforward variable latency.
- **Refinement bypasses** share a final head and use Diffmux more directly, but require all bypassed representations to remain compatible.

If both work, use multiple exits as a baseline and refinement bypass as the proposed method. Do not claim a new method from adding exit heads alone.

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

Runtime-configurable DNNs, slimmable networks, and early-exit networks are established research areas. For example, [Runtime Configurable Deep Neural Networks for Energy-Accuracy Trade-off](https://arxiv.org/abs/1607.05418) incrementally trains channel groups that can be selected at runtime. General early-exit work also studies adaptive depth. Therefore, neither "choose a smaller network at runtime" nor "put a MUX around a block" is new.

The targeted search for this memo did not identify a DLGN that exposes jointly trained, runtime-selectable hardened depths with measured isolation and hardware behavior. Current DLGN work, including the [original DLGN](https://arxiv.org/abs/2210.08277), connection optimization, LILogicNet, and the fully trainable DLGN/LUTN, produces a static circuit after training.

The defensible contribution is the complete combination of:

1. nested, hard Boolean operating modes trained in one DLGN supernetwork;
2. multiplexer-controlled refinement or exit selection with only a few runtime bits;
3. DLGN-specific deep-to-shallow training that controls the hard accuracy gap at every mode;
4. operand isolation or stage disabling that makes the unused Boolean network physically inactive;
5. a measured accuracy-latency-energy Pareto frontier and comparison with storing separate static DLGNs.

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
| Conventional early-exit BNN or small MLP | Contextual edge baseline at matched accuracy/latency |
| Larger static DLGN at matched area | Ensures the MUX and exit overhead is not better spent on gates |

When possible, include the original DLGN, Light DLGN/IWP, and a hard-forward `Mind the Gap` training baseline. Elastic behavior should not depend on only one gate parameterization.

### Experimental program

#### Phase A: prove that depth modes exist

- Use MNIST and Fashion-MNIST for rapid development.
- Train 3-, 5-, and 7-stage networks or another set supported by the local implementation.
- Compare independent training, naive truncation, multiple exits, and joint mode training.
- Reject configurations in which additional depth does not improve hard validation accuracy.

#### Phase B: primary accuracy experiment

- Use CIFAR-10 as the primary benchmark because the thesis already uses it.
- Evaluate at least three modes: Economy, Balanced, and Accurate.
- Use two maximum gate budgets so conclusions are not tied to one oversized model.
- Run at least five paired seeds for every principal comparison.
- Use the same augmentation, optimizer-step budget, and validation protocol across baselines.

#### Phase C: edge use case

Add one workload with a natural runtime constraint, such as keyword spotting, human-activity recognition, or network-flow classification. Define a trace of changing deadlines or energy budgets and let the external controller choose modes. This makes the system claim more credible than reporting three disconnected CIFAR-10 points.

#### Phase D: hardware evaluation

- Generate a hardened implementation for the complete ElasticMux supernetwork.
- Generate every independently trained static baseline with the same flow.
- Record switching activity separately for each runtime mode using representative input traces.
- Evaluate a pipelined version if variable-cycle latency is claimed.
- Report both synthesis and post-route numbers; label estimates clearly.

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
- controller overhead for externally selected and confidence-selected modes.

**Hardware cost**

- total gate/cell/LUT count and area of the supernetwork;
- number, fan-in, and area of retained multiplexers;
- exit-head and operand-isolation overhead;
- critical path and maximum frequency in every characterized mode;
- dynamic power, leakage/static power, and energy per inference;
- toggle rate inside disabled stages;
- total storage/area compared with deploying all independent static modes;
- break-even frequency at which retaining one ElasticMux design is cheaper than storing or loading separate models.

**Pareto and system metrics**

- accuracy versus latency;
- accuracy versus energy;
- accuracy versus active gate operations;
- Pareto hypervolume over accuracy, latency, and energy;
- expected energy and expected accuracy under a declared distribution of runtime modes;
- energy-delay product only as a secondary summary, because it can hide accuracy loss.

### Essential ablations

- two, three, and four runtime modes;
- multiple exits versus bypassable refinement blocks;
- identity versus learned Boolean bypass;
- progressive versus joint-from-scratch training;
- deep-to-shallow distillation on/off;
- monotonic ranking loss on/off;
- uniform versus workload-weighted mode sampling;
- hard-forward versus soft-forward final training;
- external mode control versus confidence-based continuation;
- MUX only versus MUX plus operand isolation/clock enable;
- depth elasticity alone versus the optional width extension;
- CPU, combinational hardware, and pipelined hardware execution semantics.

### Feasibility and main risks

Software feasibility is high: multiple exits, prefix execution, and mode-sampled training are localized changes to the current DLGN implementations. A first three-mode result should be possible within one to two weeks.

Hardware feasibility is medium-high if the design is pipelined at stage boundaries. A purely combinational implementation makes credible variable latency harder. The major risks are:

- deeper DLGN stages may not improve accuracy enough to create useful modes;
- shared training may reduce the deepest mode's accuracy;
- exit heads may dominate small models;
- MUXes can extend the critical path;
- inactive combinational logic may continue toggling without careful isolation;
- the result may look like a direct import of ordinary early exit unless the hardened Boolean and hardware aspects are substantial.

### Go/no-go criterion

Continue with ElasticMux as the DATE paper only if all of the following hold:

1. at least three modes form a monotonic hard-accuracy/energy or hard-accuracy/latency curve on CIFAR-10;
2. every elastic mode is within 1 percentage point of its independently trained static counterpart;
3. Economy mode reduces measured energy or latency by at least 25% relative to Accurate mode;
4. retained MUX, exit, and isolation logic adds no more than 10% area over the deepest static network;
5. MUX plus isolation measurably suppresses toggling in disabled stages;
6. the deepest elastic mode loses no more than 0.5 percentage points relative to the deepest static model.

If only software latency improves, the result may still support an embedded-systems paper, but the DATE hardware claim becomes weaker. If the additional exits work but bypass refinement does not, frame the paper around hardware-aware elastic DLGN execution rather than claiming Diffmux as the algorithmic novelty.

### Candidate paper statement

> ElasticMux-DLGN converts a static hardened logic network into a runtime-selectable Boolean supernetwork whose nested depth modes are jointly trained and physically isolated, providing a measured accuracy-latency-energy frontier with only a few control bits.

---

## Idea 2 -- RepairMux-DLGN: Budgeted Reconfigurable Routing for Fault Repair

### Paper question

Can a DLGN be trained with a small number of retained multiplexers so that permanent gate or route faults can be repaired after manufacturing or deployment by programming only a short address-bit patch?

### Target use case

The same trained DLGN is fabricated as an ASIC or mapped to an FPGA across many devices. Manufacturing test or an in-field self-test identifies a faulty gate output, LUT, or connection. A normal hardened DLGN cannot change its wiring without resynthesis or reconfiguration of the full design. RepairMux retains a limited number of configurable routes, allowing each device to select a healthy, functionally useful source.

This fits DATE topics in test, dependability, reconfigurable systems, and design methods for machine-learning hardware. It is stronger than claiming another small power reduction for an already efficient model.

### Proposed architecture

1. Train and harden the Boolean functions as in Light DLGN or a hard-forward DLGN.
2. At only a budgeted subset of consumer inputs, replace the single incoming wire with a 2-to-1 or 4-to-1 multiplexer.
3. Give each protected input one primary source and one or three spare sources. Candidate sources should have low common-ancestor overlap and should not share the same physical fault region.
4. During training, sample persistent fault maps for several consecutive steps. Optimize the gate functions, primary route, spare routes, and optional placement score jointly.
5. After a fault map is known, freeze all gate functions and solve only for the multiplexer addresses. Use greedy search first; use beam search or SAT only if greedy repair is inadequate.
6. Store the resulting per-device patch as `(mux_id, address_bits)` entries.

A suitable objective is

```math
\mathcal{L} =
\mathcal{L}_{\mathrm{clean}}
+ \lambda_f\,\mathbb{E}_{F\sim\mathcal{D}_F}
  \mathcal{L}_{\mathrm{repair}}(F)
+ \lambda_b\sum_i z_i
+ \lambda_c\sum_i z_i C_i
+ \lambda_o\mathcal{L}_{\mathrm{overlap}},
```

where `z_i` indicates that route `i` retains a physical multiplexer, `C_i` is its estimated hardware cost, and the overlap term discourages primary and spare paths from sharing the same vulnerable ancestry. Use a hard budget on `sum(z_i)` in the final experiments; a penalty alone makes area comparisons difficult.

### What is novel, and what is not

**Not novel:** multiplexers, spare wires, fault injection, generic reconfigurable repair, or training DLGNs with random bit flips.

**Potentially novel combination:** a DLGN-specific co-design that learns (1) where a scarce repair multiplexer should remain, (2) which spare Boolean sources are useful and structurally disjoint, and (3) a tiny per-device address patch after exact circuit-level fault diagnosis.

The 2026 paper [From Arithmetic to Logic](https://arxiv.org/abs/2603.22770) evaluates inherent logic/LUT-network resilience under parameter bit flips. It does not establish trained, structural post-deployment route repair. Generic fault-tolerant circuit literature will still need a systematic IEEE Xplore and ACM search before the title and novelty statement are fixed.

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
| Full retraining/resynthesis after each fault map | Accuracy upper bound, but report its time and update cost |
| Larger unprotected DLGN at matched post-route area | Prevents extra hardware alone from explaining the gain |

### Fault models

Evaluate both random and spatially correlated faults:

- gate-output stuck-at-0 and stuck-at-1;
- broken or stuck route;
- FPGA LUT truth-table/configuration-bit upset;
- multiplexer address-bit upset;
- clustered defects affecting a small physical neighborhood;
- unseen multi-fault combinations at evaluation time.

Report multiplexer and configuration-memory faults explicitly. Treating the repair network as fault-free would overstate the result. For address storage, compare unprotected bits with parity or SEC coding so that the reliability cost is visible.

### Experimental plan

**Development datasets:** MNIST and Fashion-MNIST for rapid debugging.  
**Primary dataset:** CIFAR-10 with at least two gate budgets.  
**Optional application:** one MLPerf Tiny task only if the CIFAR-10 pipeline is complete by the go/no-go date.

Use 2-to-1 and 4-to-1 repair choices, and protect 1%, 2%, 5%, 10%, and 100% of eligible routes. Evaluate fixed fault counts such as 1, 5, 10, and 50 faults and normalized fault rates such as 0.01%, 0.05%, 0.1%, and 0.5%. Choose the primary operating points before viewing final test results.

### Metrics

**Predictive/reliability metrics**

- clean hard accuracy;
- hard accuracy before and after repair at each fault rate;
- area under the accuracy-versus-fault-rate curve;
- recovered-loss fraction;
- fraction of faulty devices satisfying an accuracy target, as a yield proxy;
- repair success probability and number of unrepairable fault maps;
- worst-case, mean, and 5th-percentile accuracy across fault maps;
- generalization to fault types and clusters not used in training.

Define recovered-loss fraction as

```math
R = \frac{A_{\mathrm{repaired}}-A_{\mathrm{faulty}}}
         {A_{\mathrm{clean}}-A_{\mathrm{faulty}}}.
```

**Cost metrics**

- number and type of retained multiplexers;
- configuration bits and patch bytes per device;
- repair-search time and calibration examples required;
- synthesized cell/LUT count, total area, critical-path delay, maximum frequency, and power;
- post-route wire length and congestion when available;
- energy per inference and reliability improvement per 1% area overhead.

Power, timing, and area claims require the same synthesis/place-and-route flow, library/device, clock constraint, and optimization effort for every method. [FPGN](https://arxiv.org/abs/2607.08427) makes physically aware LUT topology a current baseline expectation; pre-synthesis gate counts alone are insufficient for a DATE hardware claim.

### Essential ablations

- learned versus random MUX placement;
- trained versus random spare sources;
- ancestry-disjoint versus unconstrained candidates;
- clean-only versus persistent-fault training;
- greedy versus beam/SAT address repair;
- 2-to-1 versus 4-to-1 repair;
- configuration-memory protection on/off;
- matched MUX count and matched post-route area.

### Go/no-go criterion

Continue with this paper only if, at no more than 15% post-route area overhead, RepairMux recovers at least 80% of the accuracy lost at one predeclared fault operating point and gives a clear reliability-area advantage over random repair and selective duplication. It must also beat fault-aware training without MUXes. If the result appears only for single faults on MNIST, it is not sufficient for DATE.

### Candidate paper statement

> RepairMux-DLGN co-trains a hardened logic network with a budgeted set of structurally disjoint spare routes, enabling post-test repair through a small per-device address patch and improving the reliability-area frontier over fault-aware training and conventional redundancy.

---

## Idea 3 -- ConfigMux-DLGN: Address-Only Adaptation of a Shared Edge Circuit

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

where `theta*` contains the common hardened gates and `a_d` contains discrete multiplexer addresses. Use episodic training: sample a training domain, adapt address logits on a small support set, and update the shared core using a disjoint query set. At deployment, harden the address logits and transmit only changed addresses relative to a base profile.

Two deployment modes must be separated:

1. **Single-profile hardwired:** the chosen MUX route is compiled to a wire. This minimizes hardware but cannot change in the field.
2. **Multi-profile configurable:** the multiplexer remains in hardware and its address comes from a small register/SRAM bank. This enables rapid profile switching but has real area, delay, power, and configuration-memory cost.

The paper must not combine the zero-cost claim from mode 1 with the adaptability claim from mode 2.

### Novelty boundary

Connection-optimization DLGNs learn routes once and then harden them. Generic adapters, meta-learning, and configurable networks already adapt small parameter sets. The possible new contribution is **a measured edge system in which the complete per-domain state is a sparse set of physical route addresses while every Boolean function remains shared**.

This is also distinct from `PatchLogic` in [july_21.md](july_21.md), which changes truth-table entries, and `PersonalDLGN` in [july_20_no_DATE.md](july_20_no_DATE.md), which changes input thresholds. Those two methods are required internal baselines, not ideas to merge into the method.

### Required comparison methods

- no adaptation;
- full-model fine-tuning and a complete per-domain checkpoint;
- output-head-only adaptation;
- input-threshold-only adaptation (`PersonalDLGN`);
- sparse truth-table/gate-function patches (`PatchLogic`);
- address fine-tuning without episodic preparation;
- random address changes with the same patch size;
- candidate-softmax connection adaptation using `K` logits per route;
- a small MLP or BNN adapter with matched stored bytes;
- replay, distillation, or EWC for the chronological-drift experiment.

Connection-learning references should include the local notes on [connection optimization](../notes/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.md) and [LILogicNet](../notes/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.md).

### Experimental plan

Use two application families so that the result is not specific to one split:

1. **UCI HAR or PAMAP2:** leave-subjects-out base training, then adapt to each held-out subject with 0, 10, 25, 50, and 100 labeled examples.
2. **CIFAR-10-C or a chronological network-traffic dataset:** train the shared core on source domains/time windows, then adapt addresses on later domains/windows.

Evaluate base profiles with 2-to-1 and 4-to-1 MUXes, candidate-route locations in early/middle/late layers, and address budgets of 0.1%, 0.5%, 1%, 2%, and 5% of the full model state. Use per-domain validation sets; never choose the patch budget on the held-out test subjects.

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

### Go/no-go criterion

Proceed only if address-only adaptation reaches within 1.5 percentage points of full fine-tuning on at least two domain-shift settings while storing no more than 2% of the full model bytes per profile, and it clearly outperforms threshold-only, truth-table-only, and random address patches at a matched byte budget. The configurable hardware overhead should remain below 15% area for the chosen design.

### Candidate paper statement

> ConfigMux-DLGN prepares a shared hardened Boolean core for few-shot domain adaptation through sparse route-address profiles, providing near-full-fine-tuning accuracy with measured profile-storage and reconfiguration costs.

---

## Idea 4 -- HypercubeMux-DLGN: Code-Aware, Reliable Connection Learning

### Why the plain idea is insufficient

For `K=2^m`, Diffmux can select candidate `i` using `m` relaxed address variables:

```math
y=\sum_{i=0}^{K-1}x_i
\prod_{j=0}^{m-1}
a_j^{b_j(i)}(1-a_j)^{1-b_j(i)}.
```

This provides `m` trainable address values instead of `K` categorical logits, but the construction is already present in DSelect-k. At hardening, both forms store a selected source index using approximately `log2(K)` bits. Furthermore, exact soft evaluation still touches all `K` candidates, so lower parameter count does not automatically imply lower training time or activation memory.

The publishable question must therefore be: **can the assignment of DLGN sources to address codewords be co-designed to improve optimization, physical locality, and resilience to address uncertainty?**

### Proposed method

Treat the `m`-bit codewords as vertices of a hypercube. Assign candidate signals to codewords so that Hamming-neighbor addresses correspond to sources that are:

- functionally similar on a calibration set;
- drawn from distinct fault regions when redundancy is desired;
- physically close to the destination pin; and
- complementary in ancestry when accuracy requires diversity.

One assignment objective is

```math
\min_{\pi}
\sum_{d_H(u,v)=1}
\left[
\alpha D_{\mathrm{act}}(x_{\pi(u)},x_{\pi(v)})
+\beta D_{\mathrm{phys}}(\pi(u),\pi(v))
+\gamma D_{\mathrm{anc}}(\pi(u),\pi(v))
\right].
```

Use recursive balanced partitioning or a small local-swap optimizer to find `pi`; do not introduce a dense learned permutation that removes the claimed parameter advantage. Compare ordinary binary order, random assignment, Gray-code order, activation-clustered order, and the combined objective.

The hypothesis is that neighboring-code similarity reduces destructive gradient cancellation during soft training and reduces damage when an address bit is uncertain or corrupted. The thesis's failed 16-to-1 experiment is the key stress case rather than a result to omit.

### Required comparison methods

- fixed random DLGN connectivity;
- `K`-candidate softmax connection selection;
- Gumbel-Softmax/straight-through categorical selection;
- DSelect-1 with random code assignment;
- original Diffmux at `K=4`;
- random, binary, and Gray candidate-to-code assignments;
- LILogicNet-style candidate connectivity;
- the fully trainable DLGN connection method on configurations that fit memory;
- direct narrow DLGN without a separate MUX layer.

The direct-narrow baseline is essential. A normal DLGN layer can use arbitrary predecessors, so a Diffmux contraction layer must demonstrate value beyond simply reducing the next layer's width and learning its connections directly.

### Experimental plan and metrics

Use MNIST and Fashion-MNIST for `K` and optimizer sweeps, followed by CIFAR-10 at two or three gate budgets. Evaluate `K` in `{2,4,8,16,32}` and at least five seeds.

Report:

- hard accuracy and soft-to-hard accuracy gap;
- success rate of reaching each candidate source;
- address entropy and bit saturation;
- gradient norm/variance and convergence failures;
- routing-logit parameters and optimizer-state bytes;
- measured peak GPU memory, step time, and time to target accuracy;
- accuracy under single and multiple address-bit flips;
- post-route area, delay, wire length, and congestion for configurable mode;
- final hardwired model size, reported separately from training state.

### Essential ablations

- code assignment objective terms individually and together;
- sigmoid, smooth-step, and Gumbel-Sigmoid address relaxations;
- direct product versus tree evaluation;
- progressive `K=2 -> 4 -> 8 -> 16` training versus training at final `K`;
- fixed versus periodically recomputed source fingerprints;
- configurable versus hardwired deployment.

### Go/no-go criterion

Do not submit a paper whose only positive result is `K/log2(K)` fewer routing logits. Continue only if code-aware assignment makes `K=16` or `K=32` materially more stable than random/DSelect-1 encoding and either (1) stays within 0.5 points of categorical routing with at least a 4x router-state reduction, or (2) reduces address-fault degradation by at least 2x at matched hardware cost.

### Candidate paper statement

> HypercubeMux-DLGN co-designs candidate-source semantics and physical locality with binary route codes, improving the optimization and configuration-fault resilience of multiplexer-parameterized DLGN connectivity.

This is not the recommended first choice because its novelty depends on the code-assignment result, not the multiplexer equation.

---

## Idea 5 -- VectorMux: Input-Dependent, Vector-Coherent Boolean Pooling

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

Begin with option 1. The selector must remain much smaller than the downstream gates removed by downsampling.

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

Use CIFAR-10 as the primary task because the thesis already exposes the gap there. Start with one pooling location and then test two. Evaluate `C`, selector depth, shared versus per-channel addresses, and 2-by-2 versus 3-by-3 windows. Use at least five seeds and the same data augmentation and training budget for all Boolean models.

### Metrics

- hard CIFAR-10 accuracy and soft-to-hard gap;
- total gates, selector gates, live gates after dead-cone elimination, and connection count;
- output-vector realizability: fraction of pooled vectors equal to one of the input spatial vectors;
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

Continue only if the first VectorMux prototype closes most of the thesis gap and reaches within 1 percentage point of OR pooling on CIFAR-10, while reducing total post-synthesis area or latency by at least 15% at matched accuracy. If it only improves over static Diffmux but remains several points behind OR, it is not a DATE paper.

### Candidate paper statement

> VectorMux introduces input-dependent, shared-address Boolean pooling that preserves the spatial coherence of multi-channel logic features while reducing downstream circuit cost.

This is the most conceptually direct continuation of the student's pooling goal, but it is also the highest-risk option.

---

## Shared Experimental Rules

These rules apply to whichever single idea is selected.

### Evaluation discipline

1. Split train, validation, and test data before tuning. Do not use test accuracy for configuration selection or early stopping.
2. Report the hardened network as the primary result. Soft accuracy is diagnostic only.
3. Use at least five paired seeds; use ten for the final key comparison if runtime permits.
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

At minimum, generate hard Boolean netlists and run a reproducible Yosys/ABC flow for every principal baseline. For the DATE submission, use either a consistent FPGA implementation flow or OpenROAD/another standard-cell flow through placement and routing. Record tool versions, target device/library, clock constraints, switching-activity method, and whether figures are pre- or post-route. Do not claim power savings from operation counts alone.

---

## Eight-Week Decision Schedule

The official [DATE 2027 Call for Papers](https://www.date-conference.com/date-2027-call-papers) lists abstract registration on **September 13, 2026 AoE** and final submission on **September 20, 2026 AoE**.

| Dates | Required outcome |
|---|---|
| July 22--28 | Reproduce the thesis's key Diffmux tables with a clean validation protocol; implement three hardened exits, prefix execution, and one bypassable fixed-width stage |
| July 29--August 4 | Run the ElasticMux pilot on MNIST/Fashion-MNIST and a small RepairMux fallback pilot; synthesize MUX-only and MUX-plus-isolation variants |
| August 5 | Select exactly one project using its go/no-go criterion; default to ElasticMux if it produces three useful modes and measurable isolation |
| August 5--18 | Complete CIFAR-10/application experiments, required baselines, and main ablations |
| August 19--30 | Run the full synthesis/place-and-route matrix, switching-activity simulations, and runtime-constraint traces |
| August 31--September 6 | Repeat key results with final seeds; statistical analysis and figures |
| September 7--12 | Draft the complete paper, limitations, artifact, and reproducibility material |
| September 13 | Register title, abstract, and authors |
| September 14--20 | Final verification, related-work audit, and submission |

## Final Recommendation

Use the tentative title:

> **ElasticMux-DLGN: Runtime-Selectable Depth and Accuracy in Hardened Logic Networks**

Build the paper around three claims only:

1. one jointly trained and hardened Boolean supernetwork provides nested depth modes controlled by only a few runtime bits;
2. DLGN-specific multi-mode training keeps every elastic mode close to its independently trained static counterpart;
3. stage disabling and operand isolation convert the selected depth into a measured accuracy-latency-energy Pareto frontier rather than merely selecting a different output.

Choose RepairMux instead if the first two weeks show that DLGN depth does not produce useful monotonic accuracy modes, or if inactive stages cannot be isolated with measurable energy or latency savings. Choose ConfigMux if application adaptation produces a substantially stronger accuracy-per-byte result and no credible variable-depth hardware flow is available. Do not lead with logarithmic address parameterization: DSelect-k makes that claim too close to established prior art. Do not lead with pooling unless the input-dependent VectorMux pilot closes the existing CIFAR-10 gap.

## Primary Sources for the Novelty Audit

- [DSelect-k: Differentiable Selection in the Mixture of Experts](https://papers.nips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)
- [Runtime Configurable Deep Neural Networks for Energy-Accuracy Trade-off](https://arxiv.org/abs/1607.05418)
- [When Do Early-Exit Networks Generalize?](https://arxiv.org/abs/2604.15764)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://openreview.net/forum?id=4bKEFyUHT4)
- [A Method for Optimizing Connections in DLGNs](https://arxiv.org/abs/2507.06173)
- [LILogicNet](https://arxiv.org/abs/2511.12340)
- [Operand-Selective Logic Gate Network](https://openreview.net/forum?id=sfqIc7BnwS)
- [Fully Trainable Deep DLGNs and LUTNs](https://arxiv.org/abs/2607.09399)
- [From Arithmetic to Logic: Resilience Under Parameter Bit-Flips](https://arxiv.org/abs/2603.22770)
- [FPGN: Physically Aware Differentiable LUT Networks](https://arxiv.org/abs/2607.08427)
- [Local Importance-Based Pooling](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.html)
- [SoftPool](https://openaccess.thecvf.com/content/ICCV2021/html/Stergiou_Refining_Activation_Downsampling_With_SoftPool_ICCV_2021_paper.html)
- [DATE 2027 Call for Papers](https://www.date-conference.com/date-2027-call-papers)

The negative novelty findings in this memo are based on the repository notes and a targeted primary-source search as of July 21, 2026. Before submission, run a formal search in IEEE Xplore, ACM Digital Library, Scopus, and patent databases, especially for runtime-elastic Boolean networks, early-exit BNN hardware, programmable fault bypass, and reconfigurable logic repair.
