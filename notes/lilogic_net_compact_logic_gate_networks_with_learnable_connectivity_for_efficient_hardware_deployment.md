### LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment - Summary notes

This paper proposes **LILogicNet**, a DLGN-style architecture that learns both:

1. the Boolean function of every gate, and
2. the connections feeding each gate,

while trying to make connectivity learning practical for larger networks and real hardware. 

Its two main technical ideas are **Top-(K) sparse connectivity** and **Basis Projection**.

## 1. The problem they address

The original DLGN uses:

```text
Random fixed connections
+
Learnable gate functions
```

The connection-optimization papers we discussed instead make the wiring trainable. However, fully trainable connectivity can become enormous.

Suppose the preceding layer contains (N) neurons. Each gate requires two inputs. A fully learnable gate therefore needs:

$
2N
$

connection logits, in addition to its 16 gate-selection logits.

For (N=32{,}000), that is approximately:

$
64{,}000
$

connection parameters **per gate**. With thousands of gates, the training memory becomes prohibitive.

LILogicNet therefore asks:

> Can we obtain most of the benefit of learned connectivity without connecting every gate to every possible preceding neuron?

## 2. Top-(K) sparse connectivity

For each of the two gate inputs, the authors randomly select only (K) candidate sources at initialization.

For example, with (K=4):

```text
Gate input A may choose from:
x17, x82, x193, x440

Gate input B may choose from:
x6, x29, x201, x719
```

The network then learns a softmax distribution over those candidates.

For input (A):

$
A=\sum_{i=1}^{K}p_i x_i,
\qquad
p_i=\frac{e^{u_i}}{\sum_j e^{u_j}}.
$

During training, (A) is a weighted mixture. After training, they select:

$
i^*=\arg\max_i u_i
$

and permanently connect the gate to (x_{i^*}).

They test:

$
K\in{2,4,8,16,32,64,128}.
$

Thus, the method does **not** test every pair of connections combinatorially. Gradient descent simultaneously optimizes the (K) candidates for each input. 

### Comparison of the three connectivity strategies

The paper evaluates:

| Strategy        | Training-time choices                 |
| --------------- | ------------------------------------- |
| Fixed           | One randomly chosen source per input  |
| Dense learnable | Every preceding neuron is a candidate |
| Top-(K)         | Only (K) randomly selected candidates |

Top-(K) is the compromise:

```text
Fixed:
cheap, but poor connectivity

Dense:
flexible, but expensive

Top-K:
partially flexible and substantially cheaper
```

Figure 2 on page 3 visually compares these three designs. 

## 3. Why Top-(K) can outperform dense connectivity

This might initially seem counterintuitive: shouldn't searching all inputs always be better?

Empirically, no. Dense connectivity gives the optimizer a much larger search space and can:

* consume too much memory,
* overfit,
* produce unstable optimization,
* become harder to train in deeper networks.

Top-(K) imposes structured sparsity as an inductive bias. It restricts each gate to a manageable local search space.

Their MNIST results show that dense connectivity is often strongest for a single shallow layer, but Top-(K) frequently becomes better as depth increases. For instance, the best reported MNIST architecture uses two layers with Top-32 connectivity and 16,000 gates per layer, achieving 98.95% accuracy. 

The authors also show that performance tends to approach dense connectivity as (K) increases. Figure 5 on page 6 shows a saturation-like curve: small values of (K) provide large initial gains, while increasing (K) further yields diminishing returns. 

## 4. Their second contribution: Basis Projection

This part relates closely to our earlier discussion about representing two-input functions with four basis terms.

The original DLGN evaluates all 16 differentiable Boolean operators:

$
y=\sum_{i=1}^{16}p_i f_i(A,B).
$

That means computing all 16 functions and weighting their outputs.

The authors observe that every real-valued two-input Boolean expression used by DLGN can be written using the basis:

$
{1,\ A,\ B,\ AB}.
$

Thus, they first project the 16 gate probabilities into four coefficients:

$
[c_1,c_2,c_3,c_4]^T
===================

W_{16\rightarrow4}p,
$

and then compute only:

$
y=c_1+c_2A+c_3B+c_4AB.
$

For example:

* AND: (AB)
* OR: (A+B-AB)
* XOR: (A+B-2AB)
* NAND: (1-AB)

All 16 gates are representable in this four-term basis.

### Important distinction from IWP

This is **not IWP**.

The paper still learns:

$
p_1,\ldots,p_{16},
$

a softmax distribution over the 16 gates.

Basis Projection merely computes the resulting mixture more efficiently.

```text
Original DLGN:
16 probabilities
→ evaluate 16 gate functions
→ weighted sum

BasisProj:
16 probabilities
→ project to 4 coefficients
→ evaluate 1, A, B, AB
```

Therefore, BasisProj reduces computational cost, but **not the number of trainable gate logits**.

IWP instead replaces the 16 logits themselves with four directly learned truth-table parameters. That is a more fundamental reparameterization.

The paper reports that BasisProj speeds training by approximately (1.9\times) to (4\times), with the largest gains for deeper and wider models. 

## 5. Their complete training and inference process

During training, each gate learns two things.

### Gate function

$
p^{gate}=\operatorname{softmax}(w_1,\ldots,w_{16})
$

determines the mixture of Boolean functions.

### Input connections

For each of its two inputs:

$
p^{connection}
==============

\operatorname{softmax}(u_1,\ldots,u_K)
$

determines the mixture over candidate sources.

After training, both are discretized:

```text
Gate function:
argmax over 16 functions

Input A:
argmax over K candidates

Input B:
argmax over K candidates
```

The final gate therefore has:

* two fixed binary inputs,
* one fixed Boolean operation,
* one binary output.

There are no floating-point calculations during inference. Figure 3 on page 3 shows this final fully binarized inference pipeline. 

## 6. Main experimental results

### MNIST

They report:

* 4,000 gates: 97.96%
* 8,000 gates: 98.45%
* 32,000 gates: 98.95%

The 8,000-gate model approximately matches the original 384,000-gate DiffLogicNet result of 98.47%, requiring about **48 times fewer gates** according to their comparison. 

### Fashion-MNIST

They report:

* 8,000 gates: 89.95%
* 64,000 gates: 90.26%
* 128,000 gates: 90.61%

The 64,000-gate model slightly surpasses their reproduced 384,000-gate DiffLogicNet result.

### CIFAR-10

They report:

* 8,000 gates: 55.11%
* 64,000 gates: 57.66%
* 256,000 gates: 60.98%

The 256,000-gate LILogicNet slightly exceeds the reported 1.28-million-gate DiffLogicNet-L result of 60.78%, although it remains far below the much larger convolutional LogicTreeNet-G at approximately 86%. 

## 7. Hardware contribution

Unlike many algorithm-only DLGN papers, these authors also examine deployment.

They report:

* FPGA resource utilization,
* FPGA latency and throughput,
* GPU and CPU inference throughput,
* ASIC tape-outs using open 130-nm and 180-nm processes.

For Fashion-MNIST, their compact model reaches roughly 32.9 million frames per second on FPGA. They also note that the final population-count classifier consumes more than two-thirds of the total FPGA LUTs in some implementations. 

That observation is important: after reducing the logic network itself, the output popcount becomes the dominant hardware bottleneck.

## 8. How is this different from the earlier connection-optimization paper?

They are conceptually very similar.

Both use:

```text
Random candidate subset
→ softmax connection weights
→ argmax after training
```

The main difference is emphasis and scale.

### Earlier connection paper

* (N_c=8) or (16) candidates.
* Demonstrates that learned connections reduce gate count.
* Includes fully learnable connections for shallow models.
* Limited hardware analysis.
* Uses conventional evaluation of the 16 gates.

### LILogicNet

* Systematically evaluates (K=2) through (128).
* Compares fixed, sparse learnable, and fully dense learnable connectivity.
* Introduces Basis Projection for faster gate evaluation.
* Uses data augmentation.
* Evaluates GPU, CPU, FPGA, and ASIC implementations.
* Scales experiments to 256,000 gates.

So LILogicNet is best understood as a more scalable and hardware-oriented realization of learnable DLGN connectivity.

## 9. Critical assessment

The strongest contribution is **not entirely new connectivity learning**; that concept already existed. The stronger aspects are:

* a broad analysis of sparse versus dense connectivity,
* demonstrating that moderate (K) can outperform full connectivity,
* substantial gate-count reductions,
* BasisProj training acceleration,
* actual hardware implementation evidence.

There are also important limitations.

First, the (K) candidates are still chosen randomly. The model only learns the best connection **within that random subset**. It does not globally discover the candidate set.

Second, it retains the original 16-way gate parameterization, so it may still inherit:

* gate competition,
* gradient cancellation,
* discretization mismatch.

Third, some of the performance gains may be partly influenced by data augmentation, which the original DLGN baselines did not use. The authors attempt controlled fixed-versus-learnable comparisons under the same augmentation setup, but cross-paper headline comparisons must still be interpreted carefully.

## Bottom line

LILogicNet combines:

```text
Original DLGN gate learning
+
Sparse trainable connectivity
+
Efficient four-basis gate evaluation
+
Hardware deployment
```

It is one of the more practically compelling papers we have examined because it shows that **learning a small set of candidate connections can reduce the required number of gates dramatically**, without paying the full cost of dense connectivity.

It does not replace IWP. A natural stronger architecture would combine:

$
\text{IWP}
+
\text{Top-}K\text{ connectivity}
+
\text{hardware-aware output reduction}.
]

That would eliminate the 16-way gate mixture, retain sparse learnable wiring, and address the population-count bottleneck identified by this paper.
