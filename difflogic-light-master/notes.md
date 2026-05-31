### Light Differentiable Logic Gate Networks (IWP) – Summary Notes

#### Motivation

The original Differentiable Logic Gate Network (DLGN) represents a binary gate as a weighted combination of all 16 possible Boolean functions of two inputs. Each neuron learns a probability distribution over these 16 gates using a softmax.

Problems with the original parameterization (OP):

* Over-parameterized (16 parameters per binary gate).
* Significant gradient cancellation between competing gates (e.g., AND vs NAND, XOR vs XNOR).
* Training becomes difficult for deep networks.
* Large discretization gap between training and inference.

The goal of the paper is to develop a more efficient gate parameterization that improves optimization while preserving the full expressiveness of DLGNs.

---

## Key Observation

A Boolean function of two inputs is completely determined by its truth table.

For two inputs there are four possible input combinations:

| A | B |
| - | - |
| 0 | 0 |
| 0 | 1 |
| 1 | 0 |
| 1 | 1 |

Thus, every Boolean function can be described using only four truth-table entries:

$[
\omega_{00},\omega_{01},\omega_{10},\omega_{11}
]

Examples:

AND:

$[
[0,0,0,1]
]$

OR:

$[
[0,1,1,1]
]$

XOR:

$[
[0,1,1,0]
]$

Therefore, instead of learning a distribution over 16 gates, the network can directly learn these four values.

---

## Input-Wise Parameterization (IWP)

The main contribution of the paper is Input-Wise Parameterization (IWP).

Instead of learning:

$[
p_0,p_1,\ldots,p_{15}
]$

the neuron learns:

$[
\omega_{00},\omega_{01},\omega_{10},\omega_{11}
]$

and computes

$[
g_\omega(p,q)
]$
=============
$[
(1-p)(1-q)\omega_{00}
+
(1-p)q\omega_{01}
+
p(1-q)\omega_{10}
+
pq\omega_{11}
]$

Interpretation:

* Every neuron learns its truth table directly.
* The parameters correspond to outputs for specific input combinations.
* The neuron behaves like a trainable 2-input LUT.

---

## Parameter Reduction

For a gate with (n) inputs:

Original Parameterization:

$[
2^{2^n}
]$

parameters.

IWP:

$[
2^n
]$

parameters.

Examples:

| Inputs | OP     | IWP |
| ------ | ------ | --- |
| 2      | 16     | 4   |
| 3      | 256    | 8   |
| 4      | 65,536 | 16  |

The reduction becomes dramatic for larger-input gates.

---

## Gradient Cancellation Analysis

One of the major contributions of the paper is identifying gradient cancellation in the original DLGN.

Original DLGN:

[
f = \sum_i p_i G_i
]

where (G_i) are logic gates.

Many gates produce opposing gradients:

* AND vs NAND
* OR vs NOR
* XOR vs XNOR

As a result:

* useful gradient signals cancel,
* optimization becomes harder,
* deep networks become difficult to train.

IWP eliminates this gate competition by directly learning truth-table entries.

Result:

* cleaner gradients,
* more stable optimization,
* better scaling to deeper networks.

---

## Discretization

During training:

$[
\omega_{ij} \in [0,1]
]$

After training:

$[
\omega_{ij}
\rightarrow
{0,1}
]$

via rounding.

Example:

$[
[0.02,0.95,0.91,0.07]
]$

becomes

$[
[0,1,1,0]
]$

which is XOR.

Because each parameter directly corresponds to a truth-table entry, discretization is straightforward and introduces less error than the original DLGN.

---

## Relationship to Sum-of-Products

IWP can be interpreted as learning a Sum-of-Products (SOP) representation.

The four basis terms are:

$[
\bar A\bar B
]$

$[
\bar AB
]$

$[
A\bar B
]$

$[
AB
]$

Thus:

$[
g_\omega
]$
========
$[
\omega_{00}\bar A\bar B
+
\omega_{01}\bar AB
+
\omega_{10}A\bar B
+
\omega_{11}AB
]$

Therefore:

* IWP directly learns truth-table entries.
* Equivalently, it learns coefficients of the minterm basis.
* Each neuron is effectively a trainable LUT.

---

## Residual Initialization (RI)

RI is **not introduced by the IWP paper**.

RI originates from the Convolutional Logic Gate Network paper and is adopted by IWP.

Idea:

Initialize gates close to the pass-through function:

$[
g(A,B)=A
]$

so that the network initially behaves like an identity mapping.

Benefits:

* improves gradient flow,
* stabilizes training,
* enables deeper logic networks.

The IWP paper shows that:

* IWP alone is insufficient,
* RI remains necessary for training deep architectures.

Thus:

```text
IWP → fixes neuron-level optimization
RI  → fixes network-level optimization
```

These techniques solve different problems and are complementary.

---

## Connection to Weightless Neural Networks

Classical DLGNs:

```text
inputs
  ↓
logic gate
  ↓
output
```

IWP:

```text
inputs
  ↓
truth table
  ↓
output
```

Therefore, IWP moves DLGNs closer to LUT-based computation.

However:

* DLGNs still use gradient descent and deep architectures.
* Weightless Neural Networks (e.g., WiSARD) are memory-centric and use RAM-like lookup structures.

Current IWP is not a WNN, but it can be viewed as a step toward trainable LUT-based computation.

---

## Main Contributions of the Paper

1. Identifies gradient cancellation in the original DLGN parameterization.
2. Introduces Input-Wise Parameterization (IWP).
3. Reduces gate parameters from (2^{2^n}) to (2^n).
4. Directly learns truth-table entries instead of gate probabilities.
5. Improves optimization and discretization behavior.
6. Demonstrates significantly better scalability for deep logic networks.
7. Shows that RI remains important even when using IWP.

---

## Critical Assessment

The strongest contribution is **not** the parameter reduction itself.

The most important insight is that:

> Learning truth-table entries directly is a better optimization problem than learning a distribution over all possible Boolean functions.

IWP largely solves the neuron-level issues of DLGNs.

However, several limitations remain:

* Fixed random connectivity.
* Fixed two-input gate structure.
* No hardware-aware optimization.
* No learned residual/skip connections.

These limitations likely represent the next opportunities for advancing DLGN research beyond the current state of the art.
