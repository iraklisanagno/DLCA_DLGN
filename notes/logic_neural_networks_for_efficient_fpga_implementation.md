### Logic Neural Networks for Efficient FPGA Implementation - Summary notes

In this paper, the authors take the original **Deep Differentiable Logic Gate Network** and study how to design and implement it efficiently on an FPGA.

The paper does **not introduce a fundamentally new DLGN neuron or training method**. Instead, the authors make three main contributions:

1. They implement trained DLGNs as real FPGA circuits.
2. They study which of the 16 Boolean functions are actually important.
3. They analyze width, depth, accuracy, LUT usage, latency, and power to derive architectural guidelines. 

# 1. Their basic LNN is essentially the original DLGN

Each neuron has two inputs and learns a softmax distribution over the 16 possible two-input Boolean functions:

$
p=\operatorname{softmax}(\omega).
$

During training, the neuron computes the expected output across all candidate gates:

$
y=\sum_{i=0}^{15}p_i f_i(a,b).
$

After training, they keep only:

$
i^*=\arg\max_i p_i,
$

and replace the differentiable mixture with the corresponding hard Boolean gate.

Thus, this is the original DLGN parameterization:

```text
Training:
mixture over 16 gates

Inference:
one selected Boolean gate
```

The connections are randomly fixed rather than learned. The final classification uses majority voting over groups of output gates. 

# 2. Their first study: which gates are actually necessary?

The authors begin with all 16 Boolean functions and perform a greedy gate-removal experiment.

At every iteration, they:

1. Remove one candidate gate type.
2. Retrain the entire network.
3. Measure the accuracy loss.
4. Permanently remove the gate whose removal causes the smallest loss.

They repeat this until only two gate types remain.

This is not pruning individual neurons. It is pruning the **global library of gate functions available to every neuron**.

For example, after removing XOR, no neuron anywhere in the network is allowed to select XOR.

## Main finding

The authors identify:

* NOR,
* NAND,
* NOT,

as the most important functions.

The accuracy falls only gradually while many other gate types are removed. A more noticeable decline occurs when NAND is removed from the remaining set of NAND, NOR, and NOT gates.

They connect this result to classical logic theory:

* NAND is functionally complete.
* NOR is functionally complete.
* Other functions can be synthesized by composing these gates.

Therefore, a DLGN does not necessarily need all 16 functions to remain expressive. 

## Important nuance

Functionally complete does not mean equally easy to train.

For instance, a NAND function can be synthesized from multiple NOR gates, but this may require several logic levels. Therefore, removing NAND may force the network to become deeper before it can reconstruct the same functionality.

This produces a trade-off:

```text
More gate types
→ richer computation per layer
→ potentially shallower network

Fewer gate types
→ simpler and more homogeneous hardware
→ additional gates and depth required
```

# 3. They compare three gate libraries

The authors train networks using:

### Full library

All 16 two-input Boolean functions.

### Reduced library

Eight selected Boolean functions.

### Minimal libraries

Two-function combinations:

* NOR + NOT,
* NAND + NOR.

They then vary:

* number of layers: 1 to 6,
* neurons per layer: 2,000 to 8,000.

Their goal is to determine how the richness of the gate library affects the required width and depth.

# 4. Their width-versus-depth conclusion

For networks with 16 or 8 candidate functions, they find that increasing **width** generally improves accuracy more than increasing depth.

For example, in the 16-function network, increasing from 2,000 to 8,000 neurons produces a larger gain than adding several layers at a fixed width.

Their explanation is that every logic neuron has a fan-in of only two. Unlike an MLP neuron, it cannot combine hundreds of preceding activations in one step. Adding more neurons creates more independent logic paths and allows the network to cover more input combinations.

Thus:

```text
DLGN with rich gate library:
width is often more valuable than depth
```

However, the result changes when only two function types are available.

With NAND–NOR or NOR–NOT networks, depth becomes more important initially because the network must compose simple gates across several levels to synthesize missing Boolean operations.

Once sufficient depth is reached—typically around four or more layers in their experiments—the benefits of further depth diminish, and width becomes important again. 

## Their practical observation

Accuracy often saturates at approximately five or six layers.

This suggests:

* excessive depth provides limited additional accuracy,
* wider layers are generally preferable once sufficient compositional depth exists.

This is an empirical result for their MNIST-like architecture, not a universal theorem about all DLGNs.

# 5. Why use only NAND and NOR?

The authors propose that a restricted gate library could simplify hardware.

An ASIC could potentially be constructed primarily from homogeneous NAND or NOR cells. On an FPGA, synthesis tools may also optimize a smaller, more regular set of logic expressions effectively.

The potential benefits are:

* fewer differentiable candidate gates evaluated during training,
* simpler gate-selection search,
* more homogeneous final circuits,
* potentially easier synthesis and physical implementation.

However, the reduced library lowers model capacity per neuron. Their results show that achieving similar accuracy then requires more layers, more neurons, or both.

For approximately 96% MNIST accuracy, they report that:

* the 16-function architecture can use roughly 4 layers and 4,000 neurons per layer;
* the 8-function architecture needs approximately 6,000 neurons per layer;
* the 2-function architecture needs approximately 6 layers and 8,000 neurons per layer.

So reducing gate diversity does not automatically reduce the final circuit size.

# 6. Their FPGA implementation flow

The hardware flow is one of the paper’s central contributions.

They:

1. Train the differentiable LNN in PyTorch.
2. Discretize every neuron to one Boolean gate.
3. Generate VHDL for the complete network.
4. Synthesize and place-and-route the design in Xilinx Vivado.
5. Generate a bitstream.
6. Deploy it on a Zynq UltraScale+ ZCU102 FPGA.

Figure 6 on page 6 illustrates this complete PyTorch-to-VHDL-to-FPGA workflow. 

The final network is treated as a large combinational Boolean circuit.

# 7. What hardware resources are used?

The authors report that their implementation uses primarily FPGA lookup tables.

They claim that the logic network requires:

* no stored weights,
* no DSP units,
* no block RAM,
* no conventional multiply-accumulate operations.

Conceptually:

```text
Input bits
   ↓
Combinational LUT network
   ↓
Majority vote
   ↓
Class
```

This gives inference delays on the order of tens of nanoseconds for the evaluated designs.

Their reported implementations commonly fall around:

$
50\text{–}90\ \mathrm{ns},
$

depending on width, depth, and gate set.

The selected MNIST design is reported at:

* 97.34% accuracy,
* approximately 29.1 ns inference time in their comparison table.

They estimate power consumption at around 2–3 mW. 

# 8. Why is there no deployment accuracy loss?

After training, the model is already discretized to exact Boolean gates.

The generated VHDL implements precisely those selected operations. Therefore, there is no additional:

* weight quantization,
* fixed-point conversion,
* activation approximation,
* rounding of learned numerical parameters.

The software Boolean model and hardware circuit should compute the same function.

This differs from deploying an FP32 neural network as a BNN or QNN, where hardware conversion can change its behavior.

However, the DLGN itself can still have a **training-to-discretization gap**. The paper means that there is no further loss between the already-discretized DLGN and its FPGA implementation—not that the original soft training model necessarily equals the hard model.

# 9. Relationship to IWP

This paper uses the original 16-way parameterization:

$
\omega_0,\ldots,\omega_{15}.
$

It predates or does not incorporate IWP.

Its gate-removal study tries to reduce training complexity by reducing the number of available gate categories:

```text
16 candidates → 8 candidates → 2 candidates
```

IWP takes a different approach:

```text
16 gate-selection parameters
→
4 truth-table parameters
```

This is an important distinction.

### This paper

Reduces the number of allowable Boolean functions.

Consequences:

* lower training computation,
* but reduced function choice,
* possibly lower accuracy or more required depth.

### IWP

Retains all 16 possible Boolean functions implicitly.

Consequences:

* full expressiveness,
* only four parameters,
* no need to remove gates from the function space.

Therefore, IWP is theoretically a cleaner way to reduce gate-parameter overhead than restricting the network to NAND and NOR.

The hardware after discretization is largely unaffected by this distinction: both approaches end with one Boolean gate per neuron. The main difference concerns training.

# 10. Relationship to weightless neural networks

The authors explicitly characterize LNNs as a form of weightless neural network because the deployed model contains no conventional numerical weights.

That classification is reasonable in a broad sense, but these networks differ from classical WiSARD-style WNNs:

| This LNN/DLGN                               | Classical WNN                                             |
| ------------------------------------------- | --------------------------------------------------------- |
| Deep graph of Boolean gates                 | RAM or LUT-based associative nodes                        |
| Gate type selected through gradient descent | Memory contents often learned through counting or writing |
| Two inputs per neuron                       | Address formed from multiple input bits                   |
| Logic synthesis implementation              | Memory-lookup implementation                              |

So they belong to the broader weightless-computation family, but are not identical to classical RAM-based WNNs.

# 11. Main findings

The authors’ central conclusions are:

* NOR, NAND, and NOT are particularly important gate types.
* A DLGN can work with a much smaller gate library.
* Reducing the gate library speeds and simplifies training, but usually requires more width or depth.
* With a rich gate library, width generally contributes more than depth.
* With a restricted gate library, sufficient depth is needed to reconstruct missing operations.
* DLGNs map naturally to FPGA LUTs and can achieve nanosecond-scale inference with very low estimated power. 

# Critical assessment

The paper is useful as an early **architecture and FPGA implementation study**, but its ML contribution is limited.

The most valuable parts are:

* the gate-library ablation,
* the width/depth analysis,
* the automated FPGA implementation flow,
* the demonstration that the hard network can be implemented without quantization loss.

Several conclusions should be treated cautiously.

First, identifying NAND and NOR as important is not surprising because they are universal gates. The more interesting question is whether restricting the gate library actually improves the full accuracy–area–latency Pareto frontier. Their results show that fewer gate types can require substantially more neurons and depth, so the hardware advantage is not consistently clear.

Second, their claim that wider networks are preferable is based mainly on a small set of datasets and random fixed connectivity. Learnable connectivity or convolutional logic architectures could change this conclusion.

Third, FPGA LUTs can implement arbitrary small truth tables. A six-input FPGA LUT does not obtain an obvious hardware advantage merely because the logical function is described as NAND rather than XOR. Restricting the gate library might matter more for ASIC standard-cell implementation than for LUT-based FPGA implementation.

Fourth, the reported hardware measurements rely heavily on Vivado reports and a relatively simple fully combinational implementation. Scalability to millions of gates, routing congestion, timing closure, I/O bandwidth, and pipelining receive limited treatment.

## Bottom line

The paper can be summarized as:

$
\boxed{
\text{Original DLGN}
+
\text{gate-library ablation}
+
\text{width/depth study}
+
\text{VHDL/FPGA deployment}
}
$

It does not improve the fundamental DLGN optimization problem. Instead, it provides practical evidence about how gate diversity, width, and depth affect accuracy and FPGA cost, and shows that discretized DLGNs can be translated directly into extremely low-latency combinational FPGA circuits.
