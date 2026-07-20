### WARP Logic Neural Networks - Summary notes

**This is one of the most important papers we have discussed**, because it directly proposes a successor to both the original DLGN parameterization and IWP/Light DLGN.

The paper introduces **WARP Logic Neural Networks**, where WARP stands for:

> **WAlsh Relaxation for Probabilistic logic neural networks**

The central idea is to represent each Boolean neuron using a **Walsh–Hadamard/Fourier basis**, rather than:

* a softmax over every possible gate, as in original DLGN, or
* direct truth-table values, as in IWP.

The authors claim that WARP is simultaneously:

1. fully expressive,
2. exactly differentiable,
3. minimally parameterized.

That combination is why the paper may be important. 

# 1. What problem are they trying to solve?

The paper organizes previous logic-network approaches into three families.

### Original DLGN

It learns a probability over every possible Boolean function.

For (n) inputs, there are:

$
2^{2^n}
$


Boolean functions, so DLGN needs that many gate-selection parameters.

For (n=2):

$
2^{2^2}=16.
$


For (n=6):

$
2^{64}\approx1.84\times10^{19}.
$


Thus, the original DLGN is essentially restricted to two-input gates.

### Differentiable Weightless Neural Networks

They directly learn the (2^n) truth-table values, but accessing the truth table requires discretizing the input into an address. The authors argue that the gradients through this address-selection process are approximate and become unstable in deep networks.

### IWP / Light DLGN

IWP also uses (2^n) parameters, corresponding to truth-table entries.

However, WARP argues that IWP has two remaining weaknesses:

* the parameters are constrained to ($0,1$
) through a sigmoid;
* rounding the parameters individually can produce a large discretization gap.

WARP attempts to retain the minimal (2^n) parameter count while improving the optimization geometry and discretization procedure. 

# 2. The core WARP representation

For binary inputs, they first map:

$
x_i\in{0,1}
$


to:

$
s_i=1-2x_i\in{-1,+1}.
$


For two inputs (u,v\in{-1,+1}), every Boolean function can be represented as:

$
g(u,v)
======

\operatorname{sgn}
\left(
\theta_0+
\theta_1u+
\theta_2v+
\theta_3uv
\right).
$


The four basis functions are:

$
1,\quad u,\quad v,\quad uv.
$


For (n) inputs, the basis contains all products of subsets of the inputs:

$
1,\quad s_1,\quad s_2,\quad s_1s_2,\quad \ldots,\quad
s_1s_2\cdots s_n.
$


There are exactly:

$
2^n
$


such basis terms.

Therefore, WARP learns:

$
\theta_0,\ldots,\theta_{2^n-1}.
$


During training, the hard sign is relaxed using a sigmoid:

$
f_{\text{WARP}}(x)
==================

\sigma\left(
\frac{1}{\tau}
\sum_i \theta_i\phi_i(x)
\right).
$


Here, (\tau) is a temperature controlling how closely the output approximates a binary step. 

# 3. Is this the ANF/Reed–Muller idea we discussed?

It is closely related, but not exactly the same basis.

Earlier, we discussed:

$
c_0\oplus c_1A\oplus c_2B\oplus c_3AB,
$


which is Algebraic Normal Form over arithmetic modulo 2.

WARP instead uses:

$
\theta_0+\theta_1u+\theta_2v+\theta_3uv
$


over real numbers, followed by a sign or sigmoid.

The conceptual similarity is strong:

* both use constant terms,
* individual variables,
* interaction products,
* higher-order interactions.

However, WARP uses the **Walsh–Hadamard/Fourier basis over ({-1,+1})**, not XOR arithmetic over (GF(2)).

This is effectively the structured-basis direction we discussed, but developed into a complete differentiable training framework.

# 4. Does WARP use fewer parameters than IWP?

No.

For an (n)-input function:

| Method        | Parameters |
| ------------- | ---------: |
| Original DLGN |  (2^{2^n}) |
| IWP/Light     |      (2^n) |
| DWN           |      (2^n) |
| WARP          |      (2^n) |

Thus, WARP does **not** improve the parameter count over IWP.

For two inputs:

$
\text{IWP}=4,\qquad \text{WARP}=4.
$


For six inputs:

$
\text{IWP}=64,\qquad \text{WARP}=64.
$


When the authors call WARP “most parameter-efficient,” they mean that (2^n) is the minimum number of continuous degrees of freedom needed to represent arbitrary (n)-input Boolean functions. IWP already reaches that count.

WARP’s claimed contribution is therefore not fewer parameters than IWP. It is a **better coordinate system and optimization procedure**.

# 5. The precise difference from IWP

This is the most important part.

## IWP

IWP directly learns truth-table outputs:

$
\gamma_{00},\gamma_{01},\gamma_{10},\gamma_{11}\in[0,1]
.
$

For two inputs:

$
f_{\mathrm{IWP}}(x_1,x_2)
=========================

\gamma_{00}(1-x_1)(1-x_2)
+\gamma_{01}(1-x_1)x_2
+\gamma_{10}x_1(1-x_2)
+\gamma_{11}x_1x_2.
$

Each parameter corresponds to one truth-table row.

At inference, each value is thresholded:

$
\gamma_i\rightarrow 0\text{ or }1.
$

## WARP

WARP learns spectral coefficients:

$
\theta_0,\theta_1,\theta_2,\theta_3\in\mathbb R.
$

The parameters are unconstrained real numbers.

After training, WARP does not independently round the (\theta_i). Instead, it transforms the coefficients back into the truth-table domain using the Walsh–Hadamard transform:

$
z=H\theta,
$

and then applies a sign:

$
t_i=
\phi^{-1}\big(\operatorname{sgn}(z_i)\big).
$

So:

```text
IWP:
learn truth-table entries directly
→ round each entry

WARP:
learn spectral coefficients
→ inverse Walsh transform
→ threshold the resulting truth table
```

The authors prove that their discretization gives the closest Boolean truth table under any (L_p) norm. 

# 6. A two-input example

For two inputs, WARP computes:

$
z(u,v)
======

\theta_0+\theta_1u+\theta_2v+\theta_3uv.
$

The four possible input combinations give:

$
\begin{aligned}
z(+1,+1)&=\theta_0+\theta_1+\theta_2+\theta_3,\
z(+1,-1)&=\theta_0+\theta_1-\theta_2-\theta_3,\
z(-1,+1)&=\theta_0-\theta_1+\theta_2-\theta_3,\
z(-1,-1)&=\theta_0-\theta_1-\theta_2+\theta_3.
\end{aligned}
$

The truth-table output at each row is determined by whether the corresponding (z) value is positive or negative.

For XOR, the paper gives a representation equivalent to:

$
\theta=[0,0,0,1].
$

Then:

$
g(u,v)=\operatorname{sgn}(uv).
$

Depending on their (0/1\leftrightarrow\pm1) convention, this corresponds to XOR or its complement.

The important point is that one interaction coefficient captures the parity structure cleanly.

# 7. Why could this train better than IWP?

The authors offer several arguments.

## Unconstrained parameters

IWP typically uses:

$
\gamma_i=\sigma(w_i),
$

so the truth-table values are constrained to ([0,1]).

When (\gamma_i) approaches 0 or 1, the sigmoid derivative becomes small:

$
\sigma'(w)=\sigma(w)(1-\sigma(w)).
$

This can lead to saturation and vanishing gradients.

WARP coefficients are unconstrained:

$
\theta_i\in\mathbb R.
$

The sigmoid is applied only to the final weighted sum, not independently to every coefficient.

The authors argue that this improves the optimization landscape.

## Structured interactions

The Walsh coefficients separate different kinds of behavior:

* constant tendency,
* individual input influence,
* pairwise interactions,
* higher-order interactions.

For a six-input function, the coefficients explicitly represent:

* first-order effects,
* second-order effects,
* parity-like higher-order effects.

IWP instead treats the 64 truth-table rows independently.

This spectral structure may provide better gradients when the target Boolean functions have low-order or concentrated interactions.

## Global optimal discretization

IWP rounds each truth-table value independently.

WARP first reconstructs the complete truth table from all coefficients, then thresholds it. This respects the joint function represented by the coefficient vector.

The authors claim this yields the nearest discrete Boolean function to the relaxed WARP neuron.

# 8. Stochastic smoothing

WARP still has a training–inference mismatch:

```text
Training:
continuous sigmoid output

Inference:
hard truth-table output
```

To reduce this gap, they inject Gumbel noise into the neuron:

$
z(x)
====

\sigma\left(
\frac{
\sum_i\theta_i\phi_i(x)+G_1-G_2
}{\tau}
\right),
$

where:

$
G_1,G_2\sim\operatorname{Gumbel}(0,1).
$

This is a Gumbel–Sigmoid or binary Concrete relaxation.

The idea is that stochastic training smooths the boundaries between different discrete truth tables and encourages parameters that remain robust after discretization.

Their experiments compare:

* soft WARP,
* Gumbel-soft WARP,
* straight-through WARP,
* Gumbel straight-through WARP.

Figure 3 on page 6 shows that the plain soft version can develop a growing discretization gap, especially for 4- and 6-input LUTs, whereas Gumbel smoothing keeps the gap considerably smaller. 

# 9. Residual initialization

They also incorporate residual initialization from Convolutional DLGNs.

The initial neuron is biased toward copying one of its inputs:

$
f(x_1,\ldots,x_n)\approx x_n.
$

This allows information and gradients to pass through deep networks at initialization.

In WARP, initializing a pass-through function is particularly simple because an individual-input term is directly one Walsh basis function.

Thus:

```text
WARP parameterization
+
Residual initialization
→ deeper trainable logic networks
```

This is not a new residual connection. It is the same initialization principle we discussed previously, adapted to WARP.

# 10. Learnable thresholding

This is the paper’s other major contribution.

Logic networks require continuous inputs to be converted into bits.

Traditional thermometer encoding might use fixed thresholds such as:

$
x>0.2,\quad x>0.4,\quad x>0.6,\quad x>0.8.
$

WARP makes those thresholds trainable:

$
x>\Omega_j,
$

where (\Omega_j) is optimized jointly with the network.

During training, they relax the hard threshold:

$
1[x>\Omega_j]
\approx
\sigma\left(\frac{x-\Omega_j}{\rho}\right).
$

They enforce ordered thresholds by learning positive differences between consecutive threshold values.

This idea is not specific to WARP. They show it also improves original DLGNs and DWNs. 

## Why this matters

On the JSC dataset, the authors report that fixed distribution-based thresholding needs approximately:

$
20\text{ bits per feature}
$

to reach peak performance.

Learnable thresholding reaches comparable accuracy with roughly:

$
5\text{ bits per feature}.
$

That could reduce:

* input dimensionality,
* first-layer size,
* connectivity,
* FPGA resources.

This may be one of the most practically important results in the paper.

# 11. Higher-input LUTs

This is where WARP directly advances beyond conventional DLGNs.

They train neurons with:

$
n=2,\quad4,\quad6
$

inputs.

A six-input WARP neuron has:

$
2^6=64
$

parameters.

The corresponding original DLGN formulation would require:

$
2^{64}
$

gate-selection parameters, which is impossible.

IWP would also need 64 parameters, so the key comparison is whether WARP trains more reliably.

Their results show that:

* DWN degrades strongly as arity increases,
* their implementation of LLNN fails for (n=4) and (n=6),
* WARP remains trainable in deeper architectures,
* WARP supports higher-input LUTs with reasonable convergence.

Figure 7 on page 8 is the key comparison. 

# 12. Higher-input convolutional logic trees

Convolutional DLGN builds a larger receptive field through a binary tree.

For 16 inputs, a binary tree requires depth four:

```text
16 inputs
→ 8 two-input gates
→ 4 gates
→ 2 gates
→ 1 output
```

WARP can use four-input LUTs and reduce the tree depth:

```text
16 inputs
→ 4 four-input LUTs
→ 1 four-input LUT
```

Both cover 16 inputs, but the WARP tree is shallower.

The paper compares:

### DLGN binary tree

$
16(8+4+2+1)=240
$

training parameters.

### WARP four-input tree

$
16(4+1)=80
$

training parameters.

The WARP version has:

* fewer parameters,
* fewer sequential levels,
* more parallelism,
* greater local expressiveness.

They report slightly higher final discrete accuracy, although WARP converges more slowly in this convolution experiment. 

# 13. Their generality claim

The authors make a strong theoretical claim:

> Previous fully expressive Boolean-neuron parameterizations can be understood as restricted or transformed versions of WARP.

Their taxonomy is roughly:

* **DLGN:** redundant representation using all Boolean functions as basis elements.
* **DWN:** direct LUT values, but approximate gradients through addressing.
* **IWP/Light:** indicator/minterm basis with constrained coefficients.
* **WARP:** orthogonal Walsh basis with unconstrained coefficients and exact gradients.

This is theoretically appealing because it gives a unified framework for comparing methods.

However, the phrase “previous methods are special cases” should be interpreted carefully. They may be related through basis transformations or restrictions, but the complete training procedures—including nonlinearities, gradient estimators, stochasticity, and discretization—are not necessarily identical.

# 14. Why your student considers it important

It directly tackles the main issue we identified earlier:

> How can we move beyond two-input IWP neurons without losing full expressiveness or training stability?

WARP’s answer is:

$
\boxed{
\text{Walsh basis}
+
\text{unconstrained coefficients}
+
\text{optimal truth-table projection}
+
\text{Gumbel smoothing}
+
\text{RI}
}
$

It also connects several threads we discussed:

* IWP as a truth-table/minterm basis;
* ANF/Reed–Muller-style structured representations;
* higher-input LUT neurons;
* Gumbel methods for reducing discretization mismatch;
* residual initialization;
* trainable thresholds;
* hardware-native LUTs.

It is therefore unusually comprehensive.

# 15. Is it clearly better than IWP?

The paper presents evidence that WARP is better in some important settings:

* deeper architectures,
* higher-input LUTs,
* discretization behavior,
* unconstrained optimization,
* learnable threshold integration.

But the comparison is not yet completely decisive.

## Concerns

### It is a preprint

The paper is dated February 2026 and described as under review. It has not necessarily undergone peer-review validation.

### Limited benchmark scale

Most results center on:

* CIFAR-10,
* JSC,
* relatively controlled architectures.

There is no ImageNet-scale demonstration.

### No hardware implementation

They claim hardware-native benefits, but do not report:

* FPGA synthesis,
* LUT utilization,
* delay,
* power,
* routing complexity.

### WARP still has (2^n) scaling

It improves dramatically over original DLGN but remains exponential in input arity:

$
n=6\rightarrow64,
\quad
n=10\rightarrow1024,
\quad
n=16\rightarrow65536.
$

It makes LUT-4 and LUT-6 feasible; it does not solve arbitrary high-arity scaling.

### Spectral structure does not guarantee sparsity

Many Boolean functions have dense Walsh spectra. For such functions, WARP does not offer parameter compression relative to IWP.

### Discretization is improved, not eliminated

Their own results show that WARP-soft can still develop a significant discretization gap. Gumbel smoothing reduces it, but adds stochasticity and another temperature schedule.

### Comparison with Light may be incomplete

The authors state that public Light code was unavailable and rely partly on reported or approximate numbers. Thus, the WARP-versus-IWP comparison is less controlled than ideal.

# My assessment

This paper is important because it may represent the first genuinely comprehensive effort to build a **general, higher-arity successor to IWP**.

I would rank its contributions as follows:

1. **Walsh parameterization for fully expressive (n)-input logic neurons** — strongest conceptual contribution.
2. **Learnable thresholding** — perhaps strongest practical contribution.
3. **Higher-arity deep-network experiments** — important empirical contribution.
4. **Optimal discrete projection** — strong theoretical contribution.
5. **Stochastic smoothing and RI integration** — useful but built from existing techniques.

The key research insight is not merely “use the Walsh transform.” It is:

> A Boolean neuron can be trained in a structured orthogonal basis with the same minimal parameter count as a truth table, while preserving exact gradients and producing the nearest deployable LUT through an efficient transform.

That is meaningfully more ambitious than IWP.

## Bottom line

WARP is:

$
\boxed{
\text{IWP-scale parameter count}
+
\text{Walsh/Fourier basis}
+
\text{unconstrained training}
+
\text{optimal LUT projection}
+
\text{Gumbel smoothing}
+
\text{learnable input thresholds}
}
$

For your research planning, this paper changes the landscape. Simply proposing “IWP with larger LUTs,” “ANF instead of IWP,” or “IWP plus Gumbel” would now be insufficient because WARP already combines much of that territory in a more unified framework.

A new paper would likely need to advance beyond WARP through something such as learned sparse spectra, adaptive arity, structured connectivity, hardware-aware coefficient pruning, or actual FPGA/ASIC co-optimization.
