### Inter-patient ECG Arrhythmia Classification with LGNs and LUTNs - Summary notes

In this paper, the authors apply both **2-input Logic Gate Networks (LGNs)** and **higher-input Lookup Table Networks (LUTNs)** to **ECG arrhythmia classification**, targeting extremely low-power wearable or implantable hardware.

However, the paper is more than an application study. It introduces two technically relevant ideas:

1. a **MUX-based differentiable formulation for directly training (n)-input LUTs**;
2. **rate coding for LGNs and LUTNs**.

It also provides an FPGA resource and power study for a medically relevant, inter-patient classification task. 

# 1. What problem do the authors solve?

They classify heartbeats from the MIT-BIH arrhythmia dataset into four classes:

* (N): normal beats,
* (S): supraventricular ectopic beats,
* (V): ventricular ectopic beats,
* (F): fusion beats.

Crucially, they use the **inter-patient paradigm**:

```text
Training patients ≠ Testing patients
```

This is much more realistic than randomly mixing heartbeats from all patients and dividing them into training and test sets.

In a mixed-patient split, the model may see different beats from the same person during both training and testing. It can therefore exploit patient-specific characteristics. In the inter-patient split, the test patients are entirely unseen, so the model must generalize to new individuals. 

This is one reason the paper matters: it tests LGNs and LUTNs under a considerably more credible medical evaluation protocol.

# 2. They compare LGNs with LUTNs

## LGN

The LGN is essentially the original DLGN:

* two inputs per neuron,
* soft mixture over 16 Boolean functions during training,
* one Boolean gate selected after training,
* fixed random connectivity.

## LUTN

A LUTN replaces each two-input gate with an (n)-input lookup table:

$
f:{0,1}^n\rightarrow{0,1}.
$

They evaluate:

* 2-input LUTs,
* 4-input LUTs,
* 6-input LUTs.

A 2-LUT is equivalent in expressiveness to a conventional two-input logic gate. A 6-LUT can represent any of:

$
2^{2^6}=2^{64}
$

Boolean functions.

This greater local expressiveness lets a LUTN use fewer nodes and fewer interconnections than an LGN.

For example, their experiments use:

| Model        | Nodes per layer | Connections per layer |
| ------------ | --------------: | --------------------: |
| LGN / 2-LUTN |           8,000 |                16,000 |
| 4-LUTN       |           3,000 |                12,000 |
| 6-LUTN       |           2,000 |                12,000 |

Despite using fewer nodes and connections, the 4- and 6-LUT networks often match the LGN’s accuracy. 

# 3. Their main methodological contribution: train a LUT as a MUX

The authors observe that an FPGA LUT is physically equivalent to a multiplexer.

For a three-input LUT, there are:

$
2^3=8
$

truth-table entries:

$
W_0,\ldots,W_7.
$

The three inputs (L_0,L_1,L_2) act as selector bits and choose one of those eight values.

For example:

```text
L2 L1 L0 = 000 → W0
L2 L1 L0 = 001 → W1
...
L2 L1 L0 = 111 → W7
```

Figure 3 on page 5 illustrates the equivalence between an 8-to-1 MUX and a 3-input LUT. 

They express this as a differentiable sum of products:

$
L_{\mathrm{out}}
================

\sum_{i=0}^{2^N-1}
W_i
\prod_{j=0}^{N-1}
\left(
s_{ij}L_j+\bar{s}_{ij}(1-L_j)
\right),
$

where (s_{ij}) is the (j)-th bit of the binary address (i).

Each product acts as a soft address-selection term.

For binary inputs, exactly one product equals 1 and all others equal 0, so the equation selects exactly one truth-table entry.

For continuous inputs in ([0,1]), several terms can contribute softly, giving a differentiable interpolation among LUT entries.

# 4. Is this actually different from IWP?

Conceptually, it is extremely close to IWP.

IWP for (N) inputs computes:

$
f(x)
====

\sum_{i=0}^{2^N-1}
W_i \phi_i(x),
$

where (\phi_i(x)) is the minterm corresponding to input pattern (i).

That is effectively what the MUX equation does.

For two inputs:

$
f(A,B)
======

W_{00}(1-A)(1-B)
+
W_{01}(1-A)B
+
W_{10}A(1-B)
+
W_{11}AB.
$

Therefore:

```text
IWP viewpoint:
learn the truth-table entries

This paper’s viewpoint:
learn the data inputs of a MUX
```

Mathematically, these are essentially the same minterm interpolation.

The difference is mostly interpretation and implementation emphasis:

* IWP is presented as an improved DLGN parameterization.
* This paper presents the same type of equation as the native Boolean structure of an FPGA LUT/MUX.

The authors claim novelty over Differentiable WNNs because their formulation follows the actual MUX Boolean equation rather than using finite-difference gradients for LUT addressing. That is a meaningful difference from DWN, but the connection to IWP/Light DLGN should have been discussed more directly. 

# 5. How do they discretize the LUT?

The (W_i) values are continuous during training.

After or during layer-wise training, the values are binarized:

$
W_i\rightarrow{0,1}.
$

The resulting vector:

$
[W_0,W_1,\ldots,W_{2^N-1}]
$

is the final LUT truth table.

For a 6-LUT, every neuron learns:

$
2^6=64
$

truth-table entries.

At inference, the six input bits form the address, and one stored bit is returned.

Thus, the deployed computation is exactly FPGA-native:

```text
Six binary inputs
       ↓
Physical 6-LUT
       ↓
One binary output
```

# 6. What is rate coding?

This is their second notable idea.

A normal LGN receives a feature as a bit:

$
x\in{0,1}.
$

Rate coding instead represents a real value:

$
p\in[0,1]
$

as the fraction of ones in a binary sequence.

For example, (p=0.75) could be represented approximately as:

```text
1 1 0 1 1 1 0 1
```

where six of the eight bits are 1.

The logic gate is repeatedly evaluated over the bitstream. The output rate represents the probability that the gate outputs 1.

For an AND gate with statistically independent inputs:

$
P(A\land B=1)=P(A=1)P(B=1).
$

For OR:

$
P(A\lor B=1)
============

P(A=1)+P(B=1)-P(A=1)P(B=1).
$

These are exactly the continuous differentiable gate expressions used by the original DLGN.

This gives an interesting interpretation:

> The “soft” DLGN values during training can be viewed as the ideal rates of infinitely long binary streams.

Therefore, during training they can propagate real-valued probabilities directly, without explicitly generating long bitstreams.

At hardware inference, finite bitstreams are needed. The longer the stream, the closer the measured rate approaches the training-time real value.

Figure 6 on page 9 shows that performance increases as the number of timesteps grows and approaches an asymptotic maximum. 

# 7. Does rate coding help?

On MNIST and Fashion-MNIST, yes.

Figure 5 on page 7 shows that rate-coded LGNs consistently outperform single-threshold binary LGNs across several depths. 

For ECG, the outcome is mixed:

* rate-coded LGNs achieve the best reported (j_\kappa) score of 0.683;
* rate-coded LUTNs generally perform worse than ordinary binary LUTNs;
* implementing rate coding requires repeated inference and counters.

For example, a rate-coded model with 128 timesteps must execute the network 128 times and count output spikes. That sharply increases:

* latency,
* FPGA LUT count,
* power,
* energy.

The authors report:

* one-layer rate-coded LGN: 133 mW and 33,526 FPGA LUTs;
* two-layer rate-coded LGN: 271 mW and 53,476 LUTs.

By comparison, the ordinary binary models consume only about 5–7 mW and require roughly 2,000–3,000 LUTs.

The authors therefore conclude that the modest accuracy or (j_\kappa) benefit does not justify the hardware cost of rate coding. 

# 8. Their preprocessing is essential

A major part of the accuracy gain comes from hand-designed ECG preprocessing.

They construct a 138-bit vector from:

* four RR intervals,
* changes in RR intervals,
* local RR statistics,
* RR ratios,
* a tachycardia indicator,
* heartbeat morphology,
* crest factors,
* delta-encoded waveform samples.

The diagram on page 4 shows how the current heartbeat and preceding/following R peaks are used to calculate the RR intervals. 

This preprocessing is not a minor detail. The paper shows that earlier LGN work using simpler dynamic thresholds achieves only about 90–91% inter-patient accuracy, whereas their engineered features push most models above 94%.

Therefore, much of the advance comes from:

```text
better ECG representation
+
logic/LUT classifier
```

not from the logic network alone.

# 9. Main results

Under the inter-patient protocol:

| Model          | Best accuracy | Best (j_\kappa) |
| -------------- | ------------: | --------------: |
| LGN            |        94.28% |           0.650 |
| Rate-coded LGN |        93.63% |       **0.683** |
| 4-LUTN         |        94.26% |           0.646 |
| 6-LUTN         |        94.24% |           0.651 |

The highest raw accuracy comes from a four-layer LGN:

$
94.28%.
$

The highest imbalance-aware metric comes from a two-layer rate-coded LGN:

$
j_\kappa=0.683.
$

An ensemble of six SVMs remains better:

* 94.50% accuracy,
* (j_\kappa=0.773).

A deep residual CNN also has a better (j_\kappa) of 0.701, though lower raw accuracy.

So the logic networks are competitive but do not establish the best medical-classification performance. 

# 10. Computational-efficiency claim

The authors estimate total computation, including preprocessing and output readout:

| Model        | Estimated FLOPs/inference |
| ------------ | ------------------------: |
| LGN          |                     2.89K |
| 4-LUTN       |                     3.81K |
| 6-LUTN       |                     6.17K |
| SVM ensemble |                  (>4.93)M |
| SNN          |                       31M |
| CNN          |                  (>1.35)B |

Their logic models are therefore claimed to require three to six orders of magnitude fewer equivalent operations.

However, their conversion assumes:

$
100\text{ Boolean operations}\approx1\text{ FLOP}.
$

That conversion is approximate and architecture-dependent. FLOP equivalence is not an ideal metric for comparing Boolean LUT hardware with CPUs or accelerators.

The FPGA results are more informative.

# 11. FPGA results

They target an Artix-7 FPGA and estimate:

* 2,000–2,990 FPGA LUTs,
* 5–7 mW network power,
* one 10 ns clock cycle per inference,
* 50–70 pJ per inference.

The 6-LUTN maps particularly naturally:

$
2{,}000\text{ trained 6-LUT neurons}
\rightarrow
2{,}000\text{ physical FPGA 6-LUTs}.
$

This one-to-one software-to-hardware correspondence is one of the strongest hardware points in the paper.

Figure 7 on page 9 shows that their one-layer models use substantially fewer FPGA LUTs and less estimated power than the prior ECG LGN baseline. 

# 12. Why is the 6-LUTN not clearly better?

One might expect 6-LUTs to dominate because they are much more expressive.

They often perform well in shallow networks, but their optimization deteriorates with depth.

The MUX/minterm equation multiplies many terms:

$
\prod_{j=0}^{N-1}
\left(s_{ij}L_j+\bar{s}_{ij}(1-L_j)\right).
$

For (N=6), each minterm multiplies six values. If these values are soft and smaller than 1, the product may become very small.

Across deep layers, this can cause:

* weak activations,
* weak gradients,
* numerical instability,
* vanishing gradients.

The paper observes that a four-layer 6-LUTN collapses badly, reaching 88.67% accuracy and a reported (j_\kappa) of zero.

This limitation is directly relevant to IWP because generalized (n)-input IWP uses the same minterm-product structure.

# 13. Relationship to WARP

WARP was uploaded after this paper’s development but directly addresses the weakness seen here.

### This paper / IWP-like MUX formulation

$
f(x)=\sum_i W_i\prod_j\text{minterm}_{ij}(x_j).
$

Weakness:

* products of many soft inputs become tiny,
* deeper high-arity networks become difficult to train.

### WARP

$
f(x)=
\sigma\left(
\sum_i\theta_i\phi_i^{W}(x)
\right),
$

using Walsh interaction terms.

WARP claims better gradient behavior and shows stronger training for deep 4- and 6-input LUT networks.

Therefore, this ECG paper provides empirical evidence for exactly the optimization limitation WARP later targets.

# 14. Is this paper important?

It is important, but for different reasons than WARP or IWP.

## Important contributions

* Demonstrates LGNs and LUTNs on a realistic inter-patient biomedical task.
* Shows direct training of FPGA-native 4- and 6-input LUTs.
* Presents the LUT-as-MUX differentiable formulation.
* Provides a one-to-one mapping from learned 6-LUTs to physical FPGA LUTs.
* Demonstrates extremely low estimated energy per inference.
* Carefully shows that rate coding improves model behavior but is generally hardware-inefficient.

## Less fundamental aspects

It does not solve the general optimization problem for high-arity LUTs. In fact, its deep 6-LUT results expose that problem.

The MUX parameterization is also mathematically very close to generalized IWP, even though the paper frames it through hardware structure.

The model depends heavily on manually engineered ECG features, so it is not an end-to-end raw-signal learning system.

# Critical assessment

The main scientific value is the combination of:

$
\boxed{
\text{realistic ECG evaluation}
+
\text{direct LUT training}
+
\text{FPGA-native mapping}
}
$

The paper demonstrates that LUTNs can deliver useful accuracy at exceptionally low energy, but it does not establish that higher-arity LUTs are universally superior to 2-input LGNs. The best raw accuracy is still achieved by the ordinary LGN, while the 6-LUTN is mainly attractive because it achieves nearly the same accuracy with fewer nodes and cleaner FPGA mapping.

The rate-coding contribution is intellectually interesting because it explains the probabilistic meaning of soft DLGN activations. Practically, however, their own hardware results show that rate coding largely defeats the original low-power motivation.

## Bottom line

The paper introduces:

$
\boxed{
\text{ECG-specific preprocessing}
+
\text{LGNs and directly trained 4/6-LUTNs}
+
\text{MUX-based LUT relaxation}
+
\text{rate coding}
+
\text{Artix-7 implementation estimates}
}
$

For your DLGN research, the most relevant takeaway is not the ECG application itself. It is that **the MUX/minterm formulation enables direct training of physical 6-LUT networks, but becomes unstable in deep, high-arity models**. That limitation creates a clear connection to WARP and remains a valuable target for further improvement.
