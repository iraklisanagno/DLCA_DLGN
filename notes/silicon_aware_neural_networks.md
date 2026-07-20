### Silicon Aware Neural Networks - Summary notes

In this paper, the authors propose **Silicon-Aware Neural Networks**, a DLGN training and implementation flow that explicitly considers the physical cost of logic gates in a target CMOS standard-cell library.

The paper’s two main contributions are:

1. **an area-aware training loss** that encourages each DLGN neuron to select physically smaller gate implementations;
2. **a complete DLGN-to-ASIC physical-design flow**, including synthesis, placement, routing, and post-layout power analysis in SkyWater 130 nm.

The authors do not improve the fundamental DLGN parameterization or connectivity. They use the original 16-way Soft-Mix DLGN and make it aware of the physical silicon cost of each gate. 

# 1. What problem do the authors address?

In an ordinary DLGN, every gate type is treated equally during training.

For example, the loss does not distinguish between choosing:

* NAND,
* OR,
* XOR,
* XNOR.

From the learning perspective, each is simply one of 16 Boolean functions.

However, in a CMOS standard-cell library, they can have very different hardware costs.

For the SkyWater 130 nm library used in the paper, the authors report examples such as:

| Logic function | Mapped cell |             Area |
| -------------- | ----------- | ---------------: |
| Inverter       | INVX1       |  (5.713,\mu m^2) |
| NAND           | NAND2X1     |  (7.618,\mu m^2) |
| AND            | AND2X1      |  (9.522,\mu m^2) |
| XOR            | XOR2X1      | (15.235,\mu m^2) |

Some Boolean functions do not exist as a single library cell and must be implemented using two cells, making them even more expensive.

Therefore, two DLGNs with the same number of neurons can have substantially different:

* total area,
* power,
* routing complexity,

depending on which gate functions they learn. The authors exploit this during training. 

# 2. Mapping the 16 Boolean functions to standard cells

The authors first create a hardware-cost table for all 16 two-input Boolean functions.

Some gates map directly:

```text
NAND → NAND2X1
NOR  → NOR2X1
XOR  → XOR2X1
```

Others require combinations. For example, a function not directly supported may be implemented using:

```text
INVX1 + NOR2X1
```

or:

```text
INVX1 + NAND2X1.
```

They assign each Boolean function an area:

$
A_i,
\qquad i=0,\ldots,15.
$

This mapping is technology- and library-specific. A gate that is cheap in one standard-cell library may be more expensive or unavailable in another.

That dependence is important: this is not a generic “NAND is always best” loss. The model is trained for the exact cell library being targeted.

# 3. Their main algorithmic idea: expected-area loss

An original DLGN neuron learns a softmax distribution over the 16 gates:

$
p_{n,0},p_{n,1},\ldots,p_{n,15}.
$

During training, the final gate has not yet been selected, but the probabilities can be used to compute its **expected physical area**:

$
\mathbb{E}[\operatorname{area}_n]
=================================

\sum_{i=0}^{15}p_{n,i}A_i.
$

For example, suppose a neuron currently has:

$
p(\text{NAND})=0.6,
$

$
p(\text{XOR})=0.3,
$

$
p(\text{{OR}})=0.1.
$

Its expected area is:

$
0.6A_{\text{NAND}}
+
0.3A_{\text{XOR}}
+
0.1A_{\text{OR}}.
$

Because (p_{n,i}) comes from a differentiable softmax, this expected area is differentiable with respect to the gate logits.

The network-wide area loss is:

$
L_{\text{area}}
===============

\frac{1}{N}
\sum_{n=1}^{N}
\mathbb{E}[\operatorname{area}_n].
$

They train with:

$
L_{\text{total}}
================

L_{\text{CE}}
+
\delta L_{\text{area}},
$

where:

* (L_{\text{CE}}) is classification cross-entropy;
* (\delta) controls the accuracy–area trade-off.

This is the key technical contribution.

The area term pushes probability mass away from expensive gates and toward cheaper ones, as long as the classification loss permits it. 

# 4. What does the loss actually change?

Without the area loss, the network chooses gates based only on accuracy.

With the area loss, two gates that provide similar task performance are no longer equivalent:

```text
Functionally useful but expensive XOR
versus
functionally useful and cheaper NAND
```

The optimizer is encouraged to select the cheaper alternative.

Importantly, this does **not** simply remove expensive gates from the candidate set. XOR remains available if it is necessary for accuracy. It just receives a larger hardware penalty.

This is better than globally restricting the gate library to NAND/NOR because it allows the optimizer to retain expensive gates where their logical function is genuinely valuable.

# 5. Their selected area coefficient

The authors test several values of:

$
\delta\in
{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}}.
$

Figure 2 on page 3 shows the trade-off.

Their selected value is:

$
\delta=0.01.
$

They state that this substantially reduces average gate area without causing a large accuracy reduction.

A larger value, such as (0.1), applies a stronger area bias but harms training accuracy.

This is a scalarized multi-objective optimization:

$
\text{accuracy}
\quad\text{versus}\quad
\text{expected cell area}.
$

It does not generate the complete Pareto frontier; it chooses one trade-off through (\delta).

# 6. Area results

For MNIST:

| Model      | Accuracy | Average area/neuron |  Total area |
| ---------- | -------: | ------------------: | ----------: |
| Baseline   |   98.04% |     (9.380,\mu m^2) | (3.00,mm^2) |
| Area-aware |   97.66% |     (6.107,\mu m^2) | (1.95,mm^2) |

The average gate area falls by approximately:

$
\frac{9.380-6.107}{9.380}
\approx34.9%.
$

The reported accuracy loss is:

$
0.38\text{ percentage points}.
$

For CIFAR-10:

| Model      | Accuracy | Average area/neuron |   Total area |
| ---------- | -------: | ------------------: | -----------: |
| Baseline   |   60.07% |     (9.994,\mu m^2) | (12.79,mm^2) |
| Area-aware |   58.82% |     (7.514,\mu m^2) |  (9.62,mm^2) |

The area reduction is approximately 25%, but accuracy falls by 1.25 percentage points.

So the loss works, but the trade-off is more favorable on MNIST than CIFAR-10. 

# 7. Direct mapping from DLGN to a gate-level netlist

After training, each neuron selects one hard Boolean gate through argmax.

The authors then generate a netlist where each DLGN neuron is replaced by the corresponding standard cell.

Conceptually:

```text
Trained neuron:
Gate 17 = NAND

        ↓

Gate-level netlist:
NAND2X1 gate_17 (...)
```

The random fixed wiring of the DLGN becomes physical netlist connectivity.

The GroupSum output is implemented with population-count adder trees using half-adder and full-adder cells.

Therefore, the complete classifier becomes:

```text
Input threshold bits
        ↓
standard-cell logic-gate network
        ↓
population-count adder tree
        ↓
class scores
```

There are no conventional stored weights or multiply-accumulate units in the hidden network.

# 8. Why do they use 18 layers of 4,000 neurons?

Previous DLGN architectures often use six layers of 64,000 neurons.

The authors attempted to map such a wide architecture to SkyWater 130 nm, but found it difficult to route because the process has only five metal layers and limited routing resources.

They therefore change the shape to:

$
18\text{ layers}\times4{,}000\text{ neurons}.
$

This is a much narrower but deeper architecture.

Their motivation is physical design:

```text
Very wide DLGN
→ enormous inter-layer wiring demand
→ routing congestion

Narrower, deeper DLGN
→ fewer wires crossing each layer boundary
→ more tractable placement and routing
```

This is an important practical finding.

The original DLGN literature tends to argue that width is preferable to depth. This paper shows that physical implementation may force the opposite choice because routing—not gate count—can become the binding constraint.

However, the authors do not introduce routing awareness into training. They manually select a narrower architecture after encountering physical-design difficulty.

# 9. Their silicon implementation

The authors convert the trained MNIST DLGN into a fully combinational hard macro using:

* SkyWater 130 nm PDK;
* Cadence standard cells;
* Xcelium for gate-level simulation;
* Innovus for placement and routing.

Figure 3 on page 4 shows the final placed-and-routed macro.

They describe this as the first DLGN implementation “in silicon simulation.” That wording means:

* the complete physical layout was generated;
* timing and power were analyzed after layout;
* the chip was **not fabricated or measured**.

It is therefore a post-layout ASIC implementation, not a taped-out silicon measurement. 

# 10. Reported hardware performance

The text reports:

* worst-case critical path: (23.9) ns;
* maximum throughput: (41.8) million classifications/s;
* power: (83.88) mW;
* energy: approximately (2.0) nJ/inference.

The reported MNIST accuracy is approximately 97.5%.

They also scale the post-layout values analytically to a hypothetical 16 nm process, estimating:

* latency: (4.2) ns;
* energy: (69) pJ/inference.

These 16 nm values are not obtained from synthesis in a 16 nm library. They are extrapolated using FO4 delay and inverter-energy ratios.

Thus, the most trustworthy values are the 130 nm post-layout estimates; the 16 nm values should be treated as rough projections.

# 11. An important inconsistency in the paper

Table III on page 4 lists for “This Work”:

* latency: (15) ns;
* energy: (352) pJ.

But the surrounding text states:

* critical path: (23.9) ns;
* energy: (2.0) nJ.

These figures are not consistent.

The authors do not clearly explain whether Table III refers to:

* an earlier implementation;
* pre-layout rather than post-layout values;
* a different switching activity;
* a different model;
* or an editing error.

This is a significant issue because latency and energy are central claims of the paper.

The more detailed post-layout subsection gives (23.9) ns and (2.0) nJ, so I would treat those as the primary reported results unless the authors provide clarification. 

# 12. Does minimizing area automatically minimize power?

Not necessarily.

The paper argues that area reduction should indirectly reduce power because smaller gates generally contain fewer transistors and lower capacitance.

That is directionally reasonable, but actual dynamic power is:

$
P_{\text{dynamic}}
\propto
\alpha C V^2 f,
$

where:

* (\alpha) is switching activity;
* (C) is capacitance;
* (V) is voltage;
* (f) is frequency.

A smaller gate may switch more frequently than a larger gate. Wiring capacitance can also dominate cell capacitance in a large random network.

Their loss minimizes:

$
\text{cell area},
$

not:

* switching activity;
* interconnect capacitance;
* buffer insertion;
* leakage;
* critical-path delay;
* actual post-layout energy.

Therefore, it is more precise to call the method **cell-area-aware**, rather than fully power-aware or PPA-aware.

# 13. Relationship to IWP, WARP, and CovJac

This paper retains the original 16-gate Soft-Mix parameterization.

That choice is important because the area loss relies directly on the gate probabilities:

$
L_{\text{area}}
===============

\sum_i p_iA_i.
$

## With IWP

IWP learns four truth-table values rather than a probability over named gates. There is no immediate probability (p_i) for assigning the area of gate (i).

An IWP-compatible area objective would need to:

1. reconstruct how close the learned truth table is to every Boolean gate;
2. assign soft probabilities or distances;
3. calculate expected hardware cost from those distances.

## With CovJac

CovJac already computes proximity weights over the 16 gate prototypes:

$
\omega_i
========

\operatorname{softmax}
\left(-|c-G_i|^2/\tau\right).
$

An area loss could naturally use:

$
L_{\text{area}}
===============

\sum_i\omega_iA_i.
$

That would combine:

* four trainable parameters;
* better deep-network optimization;
* silicon-area awareness.

## With WARP

WARP reconstructs a truth table after applying the Walsh transform. A differentiable hardware objective could evaluate the relaxed truth-table function’s distance to hardware-mapped Boolean gates, but this would require an additional soft projection.

This paper therefore provides a hardware-cost concept that is orthogonal to newer gate parameterizations.

# 14. Is this the hardware-aware idea we discussed earlier?

Yes. This paper directly implements the research direction we previously described:

$
L=
L_{\text{task}}
+
\lambda_{\text{area}}L_{\text{area}}.
$

Therefore, simply proposing “hardware-aware DLGN training using gate area” would no longer be sufficiently novel.

However, the paper covers only a narrow version of hardware awareness:

```text
gate standard-cell area
```

It does not optimize:

* critical-path delay;
* dynamic energy;
* leakage;
* routing congestion;
* fan-out;
* buffer cost;
* wirelength;
* GroupSum cost;
* reliability;
* carbon.

A stronger successor would need to optimize actual post-layout PPA or differentiable estimates of multiple physical effects.

# 15. Critical assessment

The strongest parts are:

* direct use of standard-cell-library costs during training;
* a full DLGN-to-netlist implementation flow;
* post-layout timing and power estimation;
* evidence that physical routing changes the preferred network topology;
* substantial cell-area reduction with modest MNIST accuracy loss.

The primary limitations are:

1. **Original DLGN optimization remains unchanged.**
   The method inherits Soft-Mix gradient cancellation and discretization issues.

2. **The loss is area-only.**
   It assumes area is a proxy for power and ignores switching and interconnect.

3. **No tapeout or measured silicon.**
   “Silicon implementation” means placed-and-routed simulation.

4. **Only one physically implemented task.**
   The final hard macro is for MNIST.

5. **Routing is handled manually.**
   The authors switch from a wide to a narrow architecture rather than learning a routing-aware topology.

6. **Fixed random connectivity remains.**
   Random connections may cause excessive wirelength and congestion.

7. **GroupSum hardware is not optimized jointly.**
   The population-count tree may consume substantial area and delay, as seen in other DLGN hardware papers.

8. **Inconsistent hardware numbers.**
   Table III and the post-layout text report different latency and energy values.

9. **The technology-scaled 16 nm results are extrapolations.**
   They do not account for actual placement, wire scaling, congestion, or library differences.

# Bottom line

The paper proposes:

$
\boxed{
\text{Original DLGN}
+
\text{standard-cell cost model}
+
\text{expected-area regularization}
+
\text{gate-level ASIC generation}
+
\text{post-layout Sky130 evaluation}
}
$

Its key contribution is the transition from:

> “Train a DLGN and map it to hardware afterward”

to:

> “Expose the target standard-cell library to the training process so that the network learns physically cheaper gate choices.”

This is one of the more relevant papers for your research because it directly enters the hardware-aware DLGN space. However, it should be viewed as a first step: it optimizes cell area, while the larger and more difficult problem is **jointly learning gate functions, connectivity, topology, and output aggregation using actual post-layout area, delay, energy, and routing feedback**.
