### Fitting Multilinear Polynomials for Logic Gate Networks - Summary notes


This paper proposes a new way to train **2-input DLGN neurons directly in a four-dimensional polynomial space**, instead of learning a softmax over the 16 possible gates.

The paper introduces two methods:

* **Multilinear-STE**
* **Multilinear-CovJac**

The second, CovJac, is the main contribution. Its goal is to preserve the four-parameter efficiency of IWP while fixing what the authors identify as IWP’s poor gradient utilization under hard discrete training. 

# 1. Their starting observation

Every two-input Boolean gate can be written uniquely as:

$
g(a,b)=c_0+c_a a+c_b b+c_{ab}ab.
$

Therefore, each gate corresponds to one vector:

$
[c_0,c_a,c_b,c_{ab}]
$

in a four-dimensional space.

Examples:

$
\mathrm{AND}(a,b)=ab
\quad\Rightarrow\quad
[0,0,0,1],
$

$
\mathrm{OR}(a,b)=a+b-ab
\quad\Rightarrow\quad
[0,1,1,-1],
$

$
\mathrm{XOR}(a,b)=a+b-2ab
\quad\Rightarrow\quad
[0,1,1,-2].
$

The 16 Boolean gates are consequently just **16 prototype vectors in (\mathbb{R}^4)**.

The authors interpret gate learning as a vector-quantization problem:

```text
Learn a continuous 4D point
        ↓
Snap it to the nearest of 16 gate prototypes
        ↓
Deploy the corresponding Boolean gate
```

This is shown clearly in Figure 1 on page 2. 

# 2. Their criticism of the original DLGN

The original DLGN learns:

$
\pi_0,\ldots,\pi_{15}
$

and computes a soft mixture:

$
z=\sum_{j=0}^{15}\pi_jg_j(a,b).
$

However, because every (g_j) can be represented with four polynomial coefficients, the effective coefficient vector is:

$
c_{\mathrm{eff}}=\pi^\top G,
$

where (G) is the (16\times4) matrix containing the polynomial representations of the gates.

The rank of (G) is only four.

Thus, the 16-way softmax uses 15 independent simplex directions, but only four dimensions affect the neuron’s function. The authors state that **11 of the 15 directions are nullspace directions** that cannot change the effective Boolean polynomial.

So the original DLGN is not merely overparameterized. Much of its gradient movement may occur in directions that do not change the neuron’s output. 

# 3. Exact gradient cancellation in Soft-Mix

The paper strengthens the gradient-cancellation argument discussed in the Light DLGN paper.

At uniform initialization:

$
\pi_j=\frac1{16},
$

the complementary gates cancel one another.

For example:

* AND cancels with NAND,
* OR cancels with NOR,
* XOR cancels with XNOR.

They prove:

$
\frac{\partial z}{\partial a}
=============================

# \frac{\partial z}{\partial b}

0
$

for every input (a,b) when the gate probabilities are uniform.

This means that, initially, a layer sends no gradient signal to the preceding layer. In a deep network, near-cancellation compounds across layers.

Figure 4 on page 9 measures this directly. In a 12-layer MNIST model, Soft-Mix preserves only around 29% of the gradient signal per layer, while their hard single-gate methods preserve almost all of it. 

# 4. Their first method: Multilinear-STE

Each neuron directly learns:

$
c=[c_0,c_a,c_b,c_{ab}].
$

During the forward pass, they snap (c) to its nearest valid Boolean-gate vector:

$
\hat c
======

\underset{G_j}{\arg\min},
|c-G_j|_2^2.
$

They then evaluate:

$
z=\hat c_0+\hat c_a a+\hat c_b b+\hat c_{ab}ab.
$

Because (\hat c) is already one of the 16 gates, the training forward pass and deployed gate are identical.

Thus, unlike Soft-Mix:

```text
Training forward:
one discrete gate

Inference:
the same discrete gate
```

There is **zero forward discretization gap** for Boolean inputs.

The nearest-gate operation is nondifferentiable, so the authors use a straight-through estimator:

$
\frac{\partial \hat c}{\partial c}\approx I.
$

The backward gradient becomes:

$
\frac{\partial \mathcal L}{\partial c}
\approx
\delta[1,a,b,ab],
$

where (\delta=\partial\mathcal L/\partial z).

This removes the 16-gate mixture and complementary-gate cancellation. 

# 5. Why this is not simply IWP

IWP uses the corner or truth-table basis:

$
[(1-a)(1-b),;(1-a)b,;a(1-b),;ab].
$

This paper uses the canonical multilinear basis:

$
[1,;a,;b,;ab].
$

Both have four parameters and span exactly the same function space. They are related by an invertible linear transformation.

However, the authors argue that they behave very differently under STE.

## IWP basis

For a binary input, exactly one truth-table basis term is active.

For example, if:

$
(a,b)=(0,1),
$

then only:

$
(1-a)b
$

is nonzero.

Consequently, only one of the four IWP parameters receives a gradient on each sample.

Expected active parameters per sample:

$
1.
$

## Multilinear basis

For:

$
[1,a,b,ab],
$

the constant term is always active, while the other terms are active depending on the inputs:

| Coefficient | Gradient coverage |
| ----------- | ----------------: |
| (c_0)       |              100% |
| (c_a)       |               50% |
| (c_b)       |               50% |
| (c_{ab})    |               25% |

The expected number of active coefficients is:

$
1+\frac12+\frac12+\frac14=2.25.
$

So the multilinear basis uses approximately 2.25 coefficient updates per sample versus one for the IWP basis.

The authors report a very large MNIST difference under the same hard-STE setup:

$
\text{IWP-STE}: 80.84%,
$

$
\text{Multilinear-STE}: 98.09%.
$

They attribute the 17.25 percentage-point difference entirely to the basis and resulting gradient-coverage pattern. Table 2 on page 7 shows this ablation. 

# 6. The remaining problem: interaction-coefficient starvation

Although the multilinear basis updates more coefficients than IWP, it has its own weakness.

The interaction coefficient (c_{ab}) distinguishes important gate structures:

* AND from simpler separable functions,
* OR through its negative interaction term,
* XOR through its strong interaction term,
* XNOR through the corresponding complementary interaction.

Under STE:

$
\frac{\partial\mathcal L}{\partial c_{ab}}
==========================================

\delta ab.
$

This is nonzero only when:

$
a=1,\qquad b=1.
$

Assuming balanced independent inputs, that happens only:

$
25%
$

of the time.

The authors call this **interaction-coefficient starvation**.

Thus, Multilinear-STE avoids Soft-Mix cancellation, but may still struggle to learn interactive gates such as AND and XOR, especially on tasks requiring complex feature interactions.

# 7. Their main method: Multilinear-CovJac

CovJac replaces the hard nearest-gate selection during training with soft vector quantization.

Given a learned point (c), they calculate a proximity score for every gate prototype:

$
\omega_j
========

\operatorname{softmax}
\left(
-\frac{|c-G_j|^2}{\tau}
\right).
$

Then:

$
c_{\mathrm{soft}}
=================

\sum_j\omega_jG_j.
$

Nearby gate prototypes receive greater weight.

The training output is:

$
z=c_{\mathrm{soft}}^\top[1,a,b,ab].
$

At inference, they still select the nearest single gate.

This superficially resembles Soft-Mix, but the learned object is different:

### Soft-Mix

Learns 16 independent gate logits:

$
\pi_0,\ldots,\pi_{15}.
$

### CovJac

Learns one four-dimensional point:

$
c\in\mathbb R^4.
$

The 16 proximity weights are derived from distances between (c) and the fixed gate codebook. They are not independently learned parameters.

# 8. What is the “Covariance Jacobian”?

The Jacobian of the soft quantizer is:

$
J
=

# \frac{\partial c_{\mathrm{soft}}}{\partial c}

\frac{2}{\tau}\operatorname{Cov}_{\omega}(G).
$

It is the weighted covariance matrix of the 16 gate prototypes.

This matrix contains off-diagonal elements, so the four polynomial coefficients are coupled during backpropagation.

For the interaction coefficient:

$
\frac{\partial\mathcal L}{\partial c_{ab}}
==========================================

\delta
\left(
J_{03}
+aJ_{13}
+bJ_{23}
+abJ_{33}
\right).
$

The key term is:

$
J_{03}.
$

It is multiplied by the constant basis value 1, which is active for **every input**.

Therefore, even when:

$
ab=0,
$

the interaction coefficient can still receive gradient through its covariance coupling with the constant coefficient.

In plain language:

> CovJac allows evidence reaching the always-active constant channel to also update the interaction coefficient.

That is the central technical novelty of the paper. 

# 9. Why not just use the Walsh basis?

This paper is especially interesting in light of the WARP paper.

WARP uses the Walsh basis:

$
[1,u,v,uv],
\qquad u,v\in{-1,+1}.
$

Because every Walsh term is nonzero, all coefficients receive gradient on every sample.

However, these authors argue that under a basic STE, the Walsh interaction term changes sign across inputs. Consequently, gradients may have full coverage but poor directional coherence and cancel across samples.

They formulate a theoretical result: no affine product basis can simultaneously provide:

1. full gradient coverage,
2. fully coherent gradient direction,
3. unbiased STE updates.

Their conclusion is that changing the basis alone cannot solve every problem. A different **gradient mechanism**, such as CovJac, is required.

This is effectively a critique of the idea that the Walsh basis by itself necessarily solves optimization. Their ablation reports poor performance for Walsh with a naïve STE.

However, this does **not directly refute WARP**, because WARP uses:

* continuous sigmoid relaxation,
* Walsh-domain parameterization,
* stochastic smoothing,
* residual initialization,
* a particular LUT projection.

This paper tests Walsh primarily within its own STE-based analysis. The complete WARP training method is not equivalent to Walsh+STE.

# 10. Experimental results

They evaluate:

* Adult,
* Splice,
* MONK’s-2,
* MNIST,
* SVHN,
* CIFAR-10,
* CIFAR-100.

The main comparison is:

| Method             | Parameters/neuron |
| ------------------ | ----------------: |
| Soft-Mix           |                16 |
| Gumbel-ST          |                16 |
| Multilinear-STE    |                 4 |
| Multilinear-CovJac |                 4 |

CovJac is best or tied for best across all seven datasets.

Selected results:

| Dataset   | Soft-Mix |  M-STE |   M-CovJac |
| --------- | -------: | -----: | ---------: |
| MNIST     |   98.29% | 98.09% | **98.30%** |
| SVHN      |   68.21% | 66.69% | **68.91%** |
| CIFAR-10  |   58.13% | 56.02% | **58.97%** |
| CIFAR-100 |   27.92% | 24.02% | **28.37%** |
| MONK’s-2  |   81.32% | 78.53% | **86.06%** |

The advantage over STE grows with the amount of interaction needed by the task. This supports their claim that (c_{ab}) starvation becomes important for interaction-heavy problems. 

# 11. Deep-network stability

This is one of the strongest results.

At 12 layers:

### MNIST

Soft-Mix loses:

$
15.4\text{ percentage points}.
$

CovJac remains approximately stable.

### CIFAR-10

Soft-Mix loses:

$
37.3\text{ percentage points}.
$

CovJac loses only:

$
0.5\text{ percentage points}.
$

Figure 2 on page 7 shows this clearly. The authors argue that Soft-Mix’s complementary-gate cancellation compounds with depth, whereas CovJac and other single-gate-forward methods avoid it. 

# 12. Commitment behavior

The diagnostic plots on page 9 are particularly useful.

They find:

### Soft-Mix

* improves somewhat,
* but remains weakly committed to specific gates.

### Multilinear-STE

* commits to gates very quickly,
* often commits to the wrong gates,
* then stalls.

### CovJac

* explores softly at the beginning,
* learns the useful coefficient direction,
* commits later,
* reaches higher accuracy.

Their interpretation is:

```text
M-STE:
commit first, learn later
→ premature gate locking

CovJac:
learn first, commit later
→ better gate selection
```

CovJac also chooses more strong-interaction gates and almost no constant gates, suggesting better neuron utilization. 

# 13. Training and inference cost

At deployment, all methods end with one two-input Boolean gate per neuron, so inference hardware is essentially unchanged.

During training:

* Soft-Mix evaluates 16 gate functions.
* Multilinear-STE evaluates one polynomial.
* CovJac computes distances to the 16 four-dimensional prototypes and a weighted coefficient vector.

The authors claim Multilinear-STE is approximately 25 times cheaper than Soft-Mix for training-gate evaluation.

Both proposed methods reduce:

$
16\rightarrow4
$

learned parameters per neuron, resulting in approximately four times less Adam optimizer state.

CovJac is more computationally expensive than Multilinear-STE because it still compares against all 16 codebook entries during training, but it has only four trainable parameters.

# 14. Comparison with IWP and WARP

## Compared with IWP

Both use four parameters, but:

| IWP                                         | This paper                                       |
| ------------------------------------------- | ------------------------------------------------ |
| Truth-table/corner basis                    | Canonical polynomial basis                       |
| Direct truth-table values                   | Polynomial coefficients                          |
| Often continuous during training            | Hard gate or soft codebook quantization          |
| One coefficient active per Boolean sample   | Average 2.25 coefficients active                 |
| RI addresses depth                          | Single-gate forward avoids Soft-Mix cancellation |
| No explicit interaction-starvation solution | CovJac explicitly targets (c_{ab})               |

This paper’s strongest result against IWP is specifically under **STE-based hard gate training**. It does not necessarily prove that every continuous IWP configuration is inferior.

## Compared with WARP

| WARP                                         | This paper                                   |
| -------------------------------------------- | -------------------------------------------- |
| Walsh basis                                  | Canonical multilinear basis                  |
| Supports general (n)-input LUTs              | Only evaluates 2-input gates                 |
| Sigmoid/Gumbel relaxation                    | Hard STE or soft vector quantization         |
| Optimal Walsh-to-LUT projection              | Nearest prototype in coefficient space       |
| Learnable thresholds                         | No threshold contribution                    |
| Emphasizes higher arity                      | Emphasizes deep 2-input networks             |
| Avoids constraints through real coefficients | Avoids interaction starvation through CovJac |

The papers are therefore complementary competitors.

WARP has the broader architectural scope because it supports 4- and 6-input LUTs. This paper provides a more detailed analysis of the optimization pathology for two-input gate selection.

# 15. Critical assessment

This is an important paper, particularly for understanding **why four parameters alone are not enough**.

The main insight is:

> The basis and parameter count do not fully determine trainability; the gradient mechanism determines which logical interactions can actually be learned.

The strongest contributions are:

* representing the 16 gates as a rank-four codebook;
* proving exact Soft-Mix backward cancellation at uniform probabilities;
* identifying interaction-coefficient starvation;
* introducing the covariance-Jacobian gradient coupling;
* demonstrating strong depth stability.

Important limitations remain:

* It addresses only two-input gates.
* The nearest-gate metric is Euclidean distance in coefficient space; this metric may not correspond perfectly to functional importance or downstream loss.
* CovJac reintroduces a soft training/hard deployment mismatch, although the reported gap is small.
* Some theoretical claims depend on balanced Bernoulli input assumptions; actual layer activations may be correlated and highly imbalanced.
* The comparison with WARP is incomplete because WARP’s full stochastic continuous method is not evaluated directly.
* No FPGA or ASIC results are reported.
* The absolute CIFAR accuracies remain modest, as in standard dense DLGNs.

# Bottom line

The paper proposes:

$
\boxed{
\text{4D multilinear gate codebook}
+
\text{nearest-gate quantization}
+
\text{STE or CovJac gradients}
}
$

Its main method, **Multilinear-CovJac**, uses the covariance of the Boolean-gate codebook to route gradient into the undertrained interaction coefficient (c_{ab}). This allows deep two-input logic networks to avoid the gradient cancellation of Soft-Mix and the interaction starvation of basic STE.

Among the papers we have reviewed, this is one of the strongest **optimization-focused** follow-ups to IWP. WARP is broader because it handles higher arity, but this paper offers a sharper diagnosis of the two-input case and presents a concrete mechanism that may outperform both Soft-Mix and basic IWP-style hard training.
