### Lightweight in-Network Flow Classification with Deep Differentiable Logic Gate Networks - Summary notes

In this paper, the authors propose **SwitchLGN**, a hardware-adapted DLGN for performing flow classification directly inside a programmable network switch.

The paper is primarily a **systems and hardware-mapping contribution**, not a new DLGN training method or gate parameterization. The authors take the original DLGN and restructure it so it can execute at line rate on an Intel Tofino programmable switch using P4. 

## 1. What problem are the authors solving?

Programmable switches are highly constrained:

* They have a fixed pipeline with a limited number of stages.
* They support simple Boolean and integer operations, but not floating-point arithmetic or multiplication.
* Intermediate packet values must fit inside the Packet Header Vector, or PHV.
* PHV storage is divided into clusters, and operations generally need their operands to be in the same cluster.
* Complex models often require SRAM/TCAM lookup tables or packet recirculation.

Conventional neural networks therefore map poorly to switches. Multiplications, normalization, and activations must be approximated using tables, which consumes scarce SRAM and TCAM.

DLGNs are attractive because, after training, inference consists only of operations such as:

$
\text{AND},\quad \text{OR},\quad \text{XOR},\quad \text{NOT}.
$

These operations are natively supported by the switch ALUs. However, the **original DLGN topology still cannot be mapped directly** because its random inter-layer connections create cross-cluster communication and too many intermediate values. 

## 2. Their main idea: hardware-aligned sub-networks

The authors divide every DLGN layer into multiple independent **sub-layers** or sub-models.

Instead of arbitrary random connectivity across the entire preceding layer, each neuron connects only within its own sub-layer.

For neuron (j) in sub-layer (s), the two inputs are fixed as:

$
h_l^{(s)}[j]
\quad\text{and}\quad
h_l^{(s)}[(j+1)\bmod n_l].
$

Thus, each neuron receives:

1. the neuron at the same position, and
2. the adjacent neuron, with wraparound.

Conceptually:

```text
Previous sub-layer:
x0  x1  x2  x3

Next sub-layer:
g0(x0,x1)
g1(x1,x2)
g2(x2,x3)
g3(x3,x0)
```

This ring-like fixed topology is much more restrictive than the original random DLGN, but it ensures that all required values can remain inside the same PHV cluster. 

The important architectural trade-off is:

```text
Original DLGN:
more flexible random connectivity
but difficult to map to a switch

SwitchLGN:
highly constrained local connectivity
but predictable hardware mapping
```

## 3. What is learned?

The connections are **not learned**.

They are fixed according to the local ring pattern.

Only the Boolean gate implemented by each neuron is learned.

During training, every neuron maintains logits over candidate Boolean functions and computes a soft mixture:

$
\tilde h
========

\sum_r \pi_r,g_r(a,b),
$

where:

$
\pi_r=
\frac{\exp(\alpha_r)}
{\sum_{r'}\exp(\alpha_{r'})}.
$

After training, the selected gate is:

$
r^\star=\arg\max_r\alpha_r.
$

The final hardware performs only that one discrete Boolean function.

Therefore, this paper uses the **original 16-way DLGN parameterization**, not IWP, Gumbel gate selection, or learnable connectivity. 

## 4. Reduced gate set in the first layer

The first layer uses only ten two-variable Boolean functions, called (O_{\text{thin}}), rather than all sixteen.

The authors exclude gates such as:

* constant FALSE,
* constant TRUE,
* direct pass-through (A) or (B),
* unary NOT (A) or NOT (B).

Why?

The switch compiler behaves more predictably when every first-layer expression explicitly references both input variables. That encourages the compiler to place both operands and the output within the same PHV cluster.

Later layers use the full sixteen-function set. This is therefore not an ML-motivated restriction; it is a **compiler- and placement-motivated restriction**. The authors report that it causes negligible accuracy loss in their evaluated models. 

## 5. Feature processing

Network features are normally multi-bit integers, such as:

* packet length,
* source and destination addresses,
* protocol fields,
* flags.

SwitchLGN requires binary inputs. Therefore, each field is **bit-sliced**.

For a 16-bit feature:

```text
Original feature:
101101...

↓ masks and shifts

16 separate binary neurons:
x0, x1, ..., x15
```

Each bit is stored in a separate PHV container.

The switch implements this in stages:

1. Assign the original field to PHV containers.
2. Apply bit masks to isolate each bit.
3. Right-shift each isolated bit to produce a 0 or 1.

Features are preferentially assigned to a single sub-model to preserve PHV-cluster locality. When a feature is too wide, it may be split across sub-models or, in some cases, truncated. 

That truncation is a potentially important limitation because information can be discarded for hardware convenience.

## 6. Mapping layers to the switch pipeline

Each DLGN layer maps to one programmable-switch pipeline stage.

The design uses:

```text
Parser
  ↓
Feature bit slicing
  ↓
Logic layer 1
  ↓
Logic layer 2
  ↓
...
  ↓
Local vote counting
  ↓
Global aggregation
  ↓
Final class decision
```

Figure 2 on page 4 is the central systems diagram. It shows:

* offline training on the control plane,
* automatic generation of P4 code,
* stage-by-stage execution on the data plane,
* local and global population counting. 

The compiler converts trained gates directly into P4 expressions such as:

```text
var_0_tmp = ~(var_0 | var_1);
var_1 = var_1 | (~var_2);
var_2 = ~(var_2 & var_3);
```

No probabilities or floating-point parameters remain at inference.

## 7. Local and global voting

The model uses a Group-Sum-like classification scheme.

For binary classification, every sub-layer reserves some final neurons for class 1 and some for class 2.

Within each sub-model:

$
\tilde c_k^{(s)}
================

\sum_j \tilde h_L^{(s,k)}[j].
$

The global score is:

$
\tilde c_k
==========

\sum_s\tilde c_k^{(s)}.
$

The class with the larger total vote is selected.

Because sub-models reside in separate PHV clusters, their votes cannot be added directly in one operation. The authors first calculate local counts and then move them to a common cluster for global aggregation. 

## 8. An important inconsistency: “without recirculation”

The paper repeatedly emphasizes inference “without packet recirculation.” That is technically true in the narrow sense that they do not send packets back through an external physical port.

However, on Tofino 1, their five-layer model requires a **resubmit**, meaning that the packet traverses the pipeline a second time internally:

```text
First pass:
feature processing + logic layers + local counting

Resubmit

Second pass:
global aggregation + final decision
```

The authors distinguish resubmit from recirculation because resubmit does not consume an external port and does not reduce line-rate throughput in the same way. Nevertheless, it is still a second pipeline traversal and adds latency.

On Tofino 2, which has more stages, the complete model can execute in one pass. 

A more precise claim would therefore be:

> SwitchLGN avoids external-port recirculation, although its Tofino 1 implementation uses an internal resubmit for final aggregation.

## 9. Automatic compilation toolchain

The authors also create a compiler that takes the trained model and produces deployable P4 code.

The toolchain performs:

* gate discretization,
* sub-layer partitioning,
* PHV field generation,
* PHV-cluster-aware placement,
* layer-to-stage mapping,
* local and global vote-count generation.

This is one of the paper’s stronger practical contributions because it goes beyond simulation and demonstrates a deployable end-to-end workflow. 

## 10. Experimental evaluation

The authors evaluate two binary network tasks:

### Intrusion detection

Dataset: UNSW-NB15

Classes:

* normal traffic,
* attack traffic.

Accuracy:

$
99.42%.
$

### Flow-size classification

Dataset: UNIV1

Classes:

* elephant flows,
* mice flows.

Accuracy:

$
95.59%.
$

Compared with the original DLGN, the restricted SwitchLGN topology loses approximately 0.4 percentage points or less. 

## 11. Hardware results

The final tested configuration has:

* five layers,
* 112 neurons per layer,
* temperature (\tau=6).

Reported results include:

* throughput: (99.8) Gbit/s,
* latency: (283) ns per packet,
* SRAM: (2.6%),
* TCAM: (0%),
* PHV usage: (56.7%).

The key resource shift is important:

```text
Traditional in-switch ML:
large SRAM/TCAM use

SwitchLGN:
very low SRAM/TCAM use
but high PHV use
```

Thus, SwitchLGN does not eliminate resource pressure; it moves the dominant pressure to the PHV. Figure 3(c) on page 7 shows PHV consumption increasing approximately linearly with network width and reaching the hardware limit at 16 neurons per sub-layer. 

## 12. Relation to the papers we discussed

### IWP

IWP improves the internal gate parameterization:

$
16\text{ gate logits}\rightarrow4\text{ truth-table values}.
$

SwitchLGN does not use this.

### Learnable-connectivity papers

Those papers improve accuracy and compactness by learning which inputs feed each gate.

SwitchLGN does the opposite: it imposes extremely constrained fixed connections to guarantee hardware feasibility.

### LILogicNet

LILogicNet asks:

> How can connectivity be learned efficiently?

SwitchLGN asks:

> How must connectivity be constrained so that compilation onto a programmable switch becomes possible?

### Group-Sum scalability paper

Both use output voting. SwitchLGN distributes the vote across independent hardware-local sub-models and then aggregates the local counts.

## 13. Critical assessment

The paper’s strongest contribution is the **real hardware deployment**, rather than innovation in logic-network learning.

The authors demonstrate convincingly that a DLGN can be:

* compiled to P4,
* executed on a commercial Tofino switch,
* sustained at almost 100 Gbit/s,
* implemented with almost no TCAM and little SRAM.

However, the ML evaluation is relatively narrow:

* only two binary-classification tasks,
* small networks,
* heavily engineered features,
* no complex multiclass task,
* no comparison with IWP or newer DLGN training approaches,
* fixed and highly restricted connectivity,
* possible feature truncation,
* substantial PHV consumption.

The reported high accuracy should therefore not be interpreted as showing that SwitchLGN is broadly more accurate than existing DLGNs. The tasks are comparatively structured binary flow-classification problems.

## Bottom line

The paper introduces:

$
\boxed{\text{SwitchLGN}
=======================

\text{original DLGN training}
+
\text{hardware-local fixed topology}
+
\text{P4 compiler}
+
\text{Tofino deployment}}
$

It does not improve the fundamental DLGN neuron. Instead, it demonstrates how to **co-design the topology, feature representation, voting scheme, and compiler around programmable-switch constraints**.

Among the papers we reviewed, this is one of the strongest in terms of actual systems deployment, but much weaker as a new DLGN learning methodology.
