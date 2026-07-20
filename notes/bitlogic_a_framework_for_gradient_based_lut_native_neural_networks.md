### BitLogic: A Framework for Gradient-Based LUT-Native Neural Networks - Summary notes

This paper introduces **BitLogic**, a unified framework for training, comparing, and deploying gradient-based LUT and logic-gate neural networks.

Unlike IWP, WARP, or CovJac, the authors do **not primarily propose a new neuron parameterization**. Instead, they argue that the existing LUT/DLGN literature is fragmented: every paper uses different input encodings, connectivity, LUT fan-in, training rules, output heads, and hardware-reporting conventions. Consequently, published accuracy and hardware numbers are not directly comparable.

Their main contribution is to place these methods inside one common five-axis design space, retrain them under a shared protocol, and determine which architectural choices actually matter. 

# 1. What problem do the authors identify?

The paper considers nine LUT-native methods, including:

* DiffLogic/DLGN,
* LightLUT or Light DLGN,
* WARP-LUT,
* DWN,
* PolyLUT,
* NeuraLUT,
* LILogicNet,
* LUTNet,
* LogicNets.

All ultimately deploy a Boolean LUT network, but their papers differ in many other choices.

For example, one method may report higher accuracy because it uses:

* a larger LUT fan-in,
* learned rather than random connectivity,
* a stronger input encoding,
* a DSP-based output head,
* calibration or pruning,
* a different network width.

It is therefore difficult to determine whether the improvement came from the claimed new LUT parameterization or from these surrounding design decisions.

BitLogic addresses this by factorizing a LUT network into **five independently selectable components**.

# 2. The five design axes

The full pipeline is:

$
\text{Encoder}
\rightarrow
\text{Connectivity}
\rightarrow
\text{LUT nodes}
\rightarrow
\text{Output head}.
$

The authors define five axes.

## Axis 1: Input encoder

The encoder converts continuous input values into Boolean wires.

They evaluate:

* uniform thermometer encoding;
* quantile/distributive thermometer encoding;
* binary fixed-point encoding.

For example, an 8-bit thermometer encoder generates eight threshold comparisons per feature, while an 8-bit fixed-point encoder generates the ordinary binary representation.

## Axis 2: Connectivity

This determines which preceding signals feed each LUT.

They evaluate:

* fixed random connections;
* fixed random-unique connections;
* learned connectivity with 4 candidates;
* learned connectivity with 8 candidates;
* learned connectivity with 16 candidates;
* full-layer learned connectivity.

This includes the LILogicNet-style candidate-pool approach.

## Axis 3: Fan-in

This is the number of inputs per LUT:

$
n\in{2,4,6}.
$

A LUT with (n) inputs contains:

$
2^n
$

truth-table entries.

This axis determines both local expressive power and hardware cost.

## Axis 4: Node parameterization

This determines how the truth table is relaxed and trained.

They implement:

* DiffLogic,
* LightLUT soft,
* LightLUT hard/STE,
* WARP-LUT,
* DWN,
* PolyLUT,
* NeuraLUT,
* a linear LUT baseline.

This is the axis corresponding most closely to our discussions of IWP, WARP, and alternative gate formulations.

## Axis 5: Output head

They compare:

* GroupSum/popcount;
* a quantized DSP-backed output transformation.

The DSP head increases accuracy, particularly for narrow networks, but makes the model no longer purely LUT-only. 

# 3. Their most important conclusion: fan-in matters more than parameterization

The strongest empirical conclusion is that **LUT fan-in is the largest accuracy lever**.

On MNIST with 8,000 nodes per layer:

| Fan-in | Accuracy | ASIC NAND2-equivalent cost |
| -----: | -------: | -------------------------: |
|      2 |   84.69% |                     11,358 |
|      4 |   91.13% |                     60,673 |
|      6 |   93.83% |                    248,588 |

Moving from:

$
n=2\rightarrow n=4
$

improves accuracy by approximately 6.4 percentage points.

By contrast, when fan-in and the remaining architecture are fixed, the best node parameterizations differ by much less. At 32K width, LightLUT, WARP-LUT, and DWN are within about 0.3 percentage points of one another.

The paper therefore argues:

> Much of the apparent superiority of later LUT methods over original 2-input DLGNs comes from using larger LUTs, not necessarily from a fundamentally better training parameterization.

This is particularly important for interpreting the original DiffLogic result. Its deficit is partly caused by being restricted to two-input gates. When all methods are compared at (n=2), the gap becomes considerably smaller. 

# 4. FPGA and ASIC react very differently to fan-in

The fan-in results have an important hardware nuance.

## On FPGA

Modern Xilinx devices use physical 6-input LUTs. Therefore:

* a 2-input function,
* a 4-input function,
* a 6-input function

can each fit inside one physical LUT in many cases.

In their MNIST experiment at 8K width:

* (n=4): about 15,250 FPGA LUTs;
* (n=6): about 16,134 FPGA LUTs.

The difference is relatively small.

## On ASIC

An arbitrary (n)-input Boolean function becomes much more expensive as (n) grows.

Their NAND2-equivalent counts are approximately:

* (n=4): 60.7K;
* (n=6): 248.6K.

Thus, the 6-input network costs about four times more standard-cell logic than the 4-input network.

The conclusion is:

```text
FPGA:
LUT-6 can be an attractive choice

ASIC:
LUT-4 is a more favorable accuracy–cost compromise
```

This means there is no universally optimal fan-in. The target hardware architecture must be considered.

# 5. Connectivity results

The authors find that **bounded candidate-pool learning** works best.

At width 4K:

| Connectivity            | Accuracy |
| ----------------------- | -------: |
| Learned, 16 candidates  |   88.84% |
| Learned, 8 candidates   |   88.51% |
| Learned, 4 candidates   |   88.55% |
| Fixed random            |   88.19% |
| Fixed random-unique     |   87.97% |
| Learned from full layer |   81.47% |

The surprising result is that fully learned connectivity performs much worse.

Why?

When every preceding signal is available, many connection slots select the same strong signals. This creates **mode collapse**:

```text
LUT input 1 → wire 17
LUT input 2 → wire 17
LUT input 3 → wire 17
LUT input 4 → wire 17
```

A nominal 4-input LUT may then functionally depend on fewer than four distinct inputs.

This reduces accuracy and causes synthesis tools to eliminate redundant logic. Therefore, the lower hardware cost of full-layer learned connectivity is not a positive efficiency result; it is a symptom of the collapsed model.

Their conclusion is:

> Randomly bounded candidate sets act as a useful regularizer.

Interestingly, fixed random connectivity is only about 0.9 percentage points behind the best learned approach. Thus, trainable connectivity helps, but much less dramatically than some earlier papers’ cross-paper comparisons imply. 

# 6. Node-parameterization comparison

Under the same (n=4) architecture, the paper reports the following MNIST results at width 32K:

| Parameterization |   Accuracy |
| ---------------- | ---------: |
| LightLUT soft    | **97.11%** |
| LightLUT hard    |     96.94% |
| DWN              |     96.94% |
| WARP-LUT         |     96.88% |
| PolyLUT          |     96.53% |
| NeuraLUT         |     96.22% |
| LinearLUT        |     95.84% |

The main message is not that LightLUT wins by a large margin. It is that:

$
\text{LightLUT}\approx\text{DWN}\approx\text{WARP}
$

once architecture, fan-in, width, encoder, connectivity, and training protocol are controlled.

The parameterization differences matter more at small widths and strict hardware budgets. As the networks become wider, the accuracy gap shrinks.

The hardware curves on page 6 show that most parameterizations cluster close to the same accuracy–cost Pareto frontier after discretization. This makes sense because all ultimately become ordinary truth tables; the training method changes how the LUT is found, but not necessarily the cost of the resulting LUT. 

# 7. Their “best-of-space” model

The authors select the best-performing choice from each axis and combine them into a configuration that no earlier paper had evaluated as a complete system.

The selected combination is approximately:

```text
Quantile/distributive thermometer encoder
+
bounded learned connectivity
+
4-input LUTs
+
LightLUT soft parameterization
+
GroupSum/popcount head
```

They deliberately keep:

* fan-in at four;
* the GroupSum head;

for the cross-method comparison, because most baselines can be represented fairly at (n=4), and using the DSP head would turn the comparison into LUT-only versus LUT-plus-arithmetic.

This combined model is called the **BitLogic best-of-space configuration**.

It is not a wholly new type of neuron. It is a new combination of existing components found through controlled ablation.

# 8. Cross-method benchmark

They retrain six methods under the same protocol:

* DiffLogic,
* PolyLUT,
* NeuraLUT,
* DWN,
* WARP-LUT,
* LILogicNet.

They test:

* MNIST,
* Fashion-MNIST,
* CIFAR-10,
* CIFAR-100;

at widths:

$
4K,;16K,;64K.
$

The BitLogic best-of-space combination wins every comparable cell in which all relevant methods fit within the training-memory budget.

At width 64K:

| Dataset       | BitLogic accuracy |
| ------------- | ----------------: |
| MNIST         |            97.84% |
| Fashion-MNIST |            89.16% |
| CIFAR-10      |            58.06% |
| CIFAR-100     |            18.82% |

For example, on CIFAR-10 at width 64K:

* BitLogic: 58.06%;
* PolyLUT: 53.02%;
* WARP-LUT: 52.12%;
* DiffLogic: 51.73%;
* LILogicNet: 51.67%;
* NeuraLUT: 47.30%.

However, these numbers need careful interpretation.

The paper intentionally removes method-specific procedures such as:

* calibration;
* structured pruning;
* specialized thresholding;
* residual blocks.

Consequently, the retrained versions sometimes perform much worse than the original papers’ published models. The authors clearly state that this is an **axis-level comparison**, not a full reproduction of each complete method.

Thus, BitLogic shows which design coordinates work best under one shared protocol—not necessarily that the complete BitLogic system dominates every prior paper’s fully optimized pipeline. 

# 9. Unified deployment

A major engineering contribution is that one trained checkpoint can deploy to three backends:

## GPU

The network is evaluated using bit-packed operations.

One 64-bit machine word processes 64 samples simultaneously for each Boolean operation.

## FPGA

The framework emits synthesizable SystemVerilog and evaluates it post-route on:

* Alveo U55C;
* Zynq UltraScale+ XCZU7EV.

## ASIC proxy

The same network is synthesized using:

* Yosys;
* Nangate 45 nm standard-cell library.

This provides a target-independent NAND2-equivalent cost estimate.

The authors emphasize that the Python discrete inference, GPU bit-packed path, and generated HDL are bit-exact. Therefore, there is no ambiguity between software accuracy and deployed accuracy.

# 10. Hardware results

They deploy a relatively small MNIST model:

* two layers;
* width 4,000;
* 88.79% accuracy.

The low accuracy is important: this is not their largest or most accurate MNIST network. It is the largest configuration their FPGA build host could compile for the full hardware study.

The maximum-throughput FPGA versions report approximately:

* U55C: 126.6 million samples/s;
* ZU7EV: 127.2 million samples/s.

Estimated energy per sample:

* U55C: 27 nJ;
* ZU7EV: 6 nJ.

The RTX 3090 bit-packed implementation reaches approximately:

* 8.66 million samples/s at batch 1024;
* roughly 319 μJ/sample.

Thus, the FPGA offers approximately:

* 15× higher throughput;
* four to five orders of magnitude lower estimated energy.

However, the FPGA numbers are post-route estimates, not direct board-level power measurements. Also, the network accuracy is only 88.79% on MNIST, so the result demonstrates hardware throughput rather than a competitive accuracy point. 

# 11. What does this paper say about IWP?

The paper refers to IWP as **LightLUT** and evaluates both:

* soft training;
* straight-through hard training.

LightLUT soft is the best node parameterization in their fixed-architecture sweep, although the margin over WARP and DWN is small.

The important conclusion is:

> IWP-style parameterization remains strong, but its advantage is smaller than one might conclude from comparisons against 2-input original DLGNs.

Once the architecture is controlled:

* fan-in matters more;
* connectivity matters moderately;
* width reduces differences among parameterizations.

This changes how we should interpret the IWP paper. IWP is a strong optimization method, but it is only one component of the full LUT-network design.

# 12. What does it say about WARP?

WARP-LUT performs close to LightLUT in the isolated parameterization sweep:

* LightLUT soft at 32K: 97.11%;
* WARP-LUT: 96.88%.

That is only a 0.23-point gap.

But WARP performs much worse in the cross-method table because its mapped configuration uses other less favorable axis choices under the shared protocol.

This supports the paper’s central argument:

> A method’s headline performance is often determined more by its complete architecture than by its node parameterization alone.

It would therefore be incorrect to conclude from Table 6 that WARP’s spectral representation is poor. The node-only comparison shows it is competitive.

# 13. What does it say about learned connectivity?

The findings are more nuanced than the earlier connectivity papers suggest:

* bounded candidate-pool learning improves accuracy;
* the gain over random fixed routing is modest;
* full connectivity can be harmful;
* random candidate selection functions as a regularizer;
* connection learning has training-memory cost.

Thus, a strong practical configuration is not:

```text
learn from every possible input
```

but:

```text
randomly expose 8–16 candidates
and learn one connection among them.
```

This confirms the central design in LILogicNet while challenging the assumption that more candidate connections are always better.

# 14. What is genuinely new?

The authors’ strongest contribution is not a new logical neuron. It is the **experimental methodology and software framework**.

Specifically:

1. A common abstraction that separates LUT networks into five axes.
2. Implementations of several prior methods in one codebase.
3. Controlled per-axis ablations.
4. A shared training protocol.
5. Bit-exact GPU, FPGA, and ASIC-oriented deployment.
6. A new best-of-space combination derived from existing components.

This is analogous to building a benchmark suite and modular compiler for the field.

For your work, this paper is particularly important because any proposed new DLGN idea can now be evaluated as an isolated axis rather than compared against poorly matched published baselines.

# 15. Critical assessment

The paper is valuable, but its conclusions have boundaries.

## Strong aspects

* It exposes confounding factors in existing comparisons.
* It shows that fan-in may matter more than the node formulation.
* It provides a useful modular framework.
* It reports both accuracy and hardware cost.
* It finds a real connectivity failure mode: full-router collapse.
* It provides bit-exact multi-backend deployment.

## Important limitations

### Only two-layer networks

The entire study is restricted to:

$
D=2.
$

This excludes the central optimization problem we have discussed:

* gradient cancellation at depth;
* residual initialization;
* CovJac depth stability;
* layer-wise learning;
* recurrent and convolutional architectures.

The authors explicitly call depth the “open sixth axis.”

### MNIST is used to select the axis winners

The best configuration is derived primarily from MNIST sweeps and then transferred to the other datasets.

An axis that is optimal for MNIST may not be optimal for CIFAR-100.

They themselves acknowledge that the quantile encoder’s apparent success is partly caused by an MNIST-specific threshold-collapse phenomenon.

### No full pipelines for prior methods

They remove method-specific advantages such as:

* PolyLUT pruning;
* DWN calibration;
* WARP residual blocks;
* specialized thresholding.

This improves experimental control but weakens claims about outperforming the actual published systems.

### Hardware deployment uses a weak model

The deployed model has only 88.79% MNIST accuracy.

That makes the spectacular hardware throughput less meaningful than it would be at 97–99% accuracy.

### Post-route power, not measured power

The FPGA energy values are estimates, not measurements from physical boards.

### Limited architectures and tasks

They consider feed-forward image classification only:

* no convolutional networks;
* no recurrent networks;
* no attention;
* no tabular or temporal tasks;
* no learned thresholds;
* no ternary logic;
* no multi-objective training.

# Bottom line

BitLogic can be summarized as:

$
\boxed{
\text{Unified LUT-network framework}
+
\text{five-axis design space}
+
\text{controlled cross-method benchmark}
+
\text{best-of-space component combination}
+
\text{GPU/FPGA/ASIC deployment}
}
$

Its most important scientific conclusion is:

> **Fan-in and surrounding architectural choices can matter more than the particular differentiable LUT parameterization.**

This is likely one of the most practically important papers in the collection because it changes the standard of evidence for future work. A new method should no longer compare only against the original 2-input DLGN. It should be inserted into BitLogic and tested at matched:

* fan-in,
* encoding,
* connectivity,
* width,
* output head,
* training protocol,
* and hardware budget.

For your research, the clearest open opportunity is the one the authors explicitly leave unresolved: extend this framework to **deep networks**, then compare IWP, WARP, CovJac, Gumbel, and RI under matched depth and hardware constraints.
