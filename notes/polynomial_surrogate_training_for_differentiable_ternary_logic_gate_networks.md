### Polynomial Surrogate Training for Differentiable Ternary Logic Gate Networks - Summary notes

This paper introduces **Differentiable Ternary Logic Gate Networks (DTLGNs)** and a training method called **Polynomial Surrogate Training (PST)**.

The central change is fundamental:

> Instead of restricting every signal to FALSE/TRUE, the authors use three logic values: FALSE, UNKNOWN, and TRUE.

They represent these as:

$
{-1,0,+1}.
$

The value (0) is not merely a numerical intermediate. It is intended to represent **UNKNOWN**, allowing the final logic circuit to abstain when the input is ambiguous or incomplete. 

# 1. Why move from binary to ternary logic?

A binary DLGN neuron outputs only:

$
-1\quad\text{or}\quad+1
$

corresponding to FALSE or TRUE.

Therefore, the model must always commit to a decision, even when:

* sensor data are missing,
* an input lies near the decision boundary,
* medical information is incomplete,
* a temporal event cannot yet be determined.

The authors use **Kleene’s three-valued logic (K_3)**:

| Value | Meaning |
| ----: | ------- |
|  (-1) | FALSE   |
|   (0) | UNKNOWN |
|  (+1) | TRUE    |

This gives the circuit an internal uncertainty state.

Conceptually:

```text
Binary DLGN:
FALSE or TRUE

Ternary DLGN:
FALSE, UNKNOWN, or TRUE
```

The authors’ main argument is that UNKNOWN enables **principled abstention**, not merely lower confidence in a softmax score. 

# 2. Why can they not use the original DLGN formulation?

For a two-input binary gate, each input has two values and the truth table has:

$
2^2=4
$

rows.

Each row can output either 0 or 1, producing:

$
2^4=16
$

possible gates.

That is why the original DLGN can maintain a softmax over 16 gate choices.

For ternary inputs, each input has three values. The truth table therefore has:

$
3^2=9
$

rows.

Each row can output one of three values, producing:

$
3^9=19{,}683
$

possible two-input ternary gates.

A conventional DLGN would consequently require **19,683 gate logits per neuron**.

That is clearly impractical.

The authors therefore abandon the softmax-over-all-gates formulation entirely. 

# 3. Their solution: Polynomial Surrogate Training

Every two-input ternary function can be represented uniquely using a polynomial that has degree at most two in each input:

$
p_w(a,b)
========

w_0
+w_1a
+w_2b
+w_3ab
+w_4a^2
+w_5b^2
+w_6a^2b
+w_7ab^2
+w_8a^2b^2.
$

Thus, every neuron learns only nine parameters:

$
w=[w_0,\ldots,w_8].
$

This polynomial is evaluated over the nine possible input pairs:

$
(a,b)\in{-1,0,+1}^2.
$

Because the corresponding (9\times9) Vandermonde matrix is invertible, these nine coefficients can represent **any assignment of nine output values**. Therefore, the nine-parameter polynomial spans the complete 19,683-gate ternary vocabulary.

The parameter reduction is:

$
19{,}683\rightarrow9,
$

or:

$
2{,}187\times
$

fewer parameters per neuron than a categorical softmax over all ternary gates. 

# 4. How is this related to IWP and the multilinear paper?

The idea is conceptually close to both.

## Binary IWP

For binary two-input logic, IWP learns four truth-table values:

$
2^2=4.
$

## Multilinear polynomial method

It learns four polynomial coefficients:

$
1,\ a,\ b,\ ab.
$

## This paper

For ternary two-input logic, it learns nine polynomial coefficients:

$
1,\ a,\ b,\ ab,\ a^2,\ b^2,\ a^2b,\ ab^2,\ a^2b^2.
$

So the general pattern is:

$
q\text{-valued two-input logic}
\rightarrow q^2\text{ coefficients}.
$

For binary logic:

$
q=2\rightarrow4.
$

For ternary logic:

$
q=3\rightarrow9.
$

The paper is therefore extending direct function-space parameterization from Boolean logic into **many-valued logic**.

The major novelty is not simply the polynomial representation. It is using that representation to make the enormous ternary gate vocabulary trainable.

# 5. What happens inside each layer during training?

The inputs and outputs of the training network are continuous values in:

$
[-1,1].
$

Each neuron evaluates its polynomial and then clips the result:

$
h_j^{(l)}
=========

\operatorname{clip}
\left(
p_{w_j^{(l)}}(h_s,h_t)
\right),
$

where:

$
\operatorname{clip}(x)
======================

\max(-1,\min(1,x)).
$

The clipping serves several purposes:

* prevents polynomial outputs from exploding across layers;
* maintains the ternary-compatible range;
* leaves the exact values (-1,0,+1) unchanged;
* saturates large outputs and thereby regularizes training.

Thus, during training the network is still continuous. It does not execute a discrete ternary gate at every forward pass.

# 6. How do they convert the polynomial into a real ternary gate?

After training, they evaluate the polynomial at all nine ternary input combinations.

This produces a soft truth table:

$
t=Vw,
$

where (V) is the fixed Vandermonde evaluation matrix.

For example, the nine values might be:

$
[-0.92,;0.08,;0.84,\ldots].
$

Each entry is then rounded to the nearest member of:

$
{-1,0,+1}.
$

So:

$
-0.92\rightarrow-1,
$

$
0.08\rightarrow0,
$

$
0.84\rightarrow+1.
$

The resulting nine ternary outputs form one valid truth table among the 19,683 possibilities.

The process is:

```text
Learn nine continuous polynomial coefficients
                 ↓
Evaluate polynomial on the 3×3 input grid
                 ↓
Round each output to −1, 0, or +1
                 ↓
Obtain one discrete ternary gate
```

Figure 1 on page 3 contrasts this process with the binary DLGN’s argmax over 16 gates. 

# 7. Commitment loss

The major problem is that a learned polynomial output might remain between valid ternary values.

For example:

$
p_w(a,b)=0.43.
$

At deployment, this must become either:

$
0
\quad\text{or}\quad
1.
$

The authors introduce a commitment regularizer:

$
R_A(W)
======

\frac{1}{N}
\sum_j
\frac{1}{9}
\sum_{(a,b)\in{-1,0,1}^2}
\operatorname{dist}
\left(
p_{w_j}(a,b),{-1,0,1}
\right)^2.
$

This directly measures how far every neuron’s nine soft truth-table entries are from valid ternary outputs.

The training loss is:

$
L
=

L_{\text{task}}
+
\lambda(t)R_A.
$

The commitment weight is gradually increased:

```text
Early training:
small commitment penalty
→ explore freely

Late training:
large commitment penalty
→ move outputs toward −1, 0, +1
```

The authors prove that this commitment loss equals the average squared difference between each soft truth table and its hardened truth table.

That is useful, but the guarantee is **per neuron**. It does not guarantee that the complete hardened network will produce exactly the same classifications as the continuous training network.

# 8. The hardening gap remains substantial

The paper distinguishes:

* **soft accuracy**: continuous polynomial network;
* **circuit accuracy**: discretized ternary circuit;
* **hardening gap**: their difference.

On CIFAR-10, the gap varies significantly:

| Network size | Soft accuracy | Circuit accuracy |     Gap |
| -----------: | ------------: | ---------------: | ------: |
|  48K neurons |         45.4% |            42.6% |  2.8 pp |
|          96K |         50.3% |            36.1% | 14.1 pp |
|         144K |         51.5% |            38.5% | 13.0 pp |
|         192K |         51.2% |            39.2% | 11.9 pp |
|         512K |         52.1% |            48.4% |  3.7 pp |

The authors emphasize that the gap contracts at large scale. However, it is non-monotonic: it first becomes much worse and only later improves.

Their strongest circuit still loses:

$
3.7
$

percentage points after hardening.

Thus, PST does **not solve the discretization problem**. It provides a measurable and regularized formulation of it, and overparameterization appears empirically to reduce it. 

# 9. Why does overparameterization reduce the gap?

At large width, many neurons harden to gates that output UNKNOWN or zero.

These neurons contribute little or nothing to the GroupSum output.

The authors interpret this as implicit pruning:

```text
Poorly committed neuron
        ↓
hardens to zero/UNKNOWN output
        ↓
has little effect on class score
```

Meanwhile, enough other neurons remain to preserve the classification.

This is an interesting interpretation, but it also means their solution to hardening partly depends on a very large amount of redundancy:

$
512{,}000\text{ neurons}.
$

That is not necessarily consistent with the original DLGN goal of producing extremely compact circuits.

# 10. The UNKNOWN signal and abstention

The most distinctive contribution is their use of UNKNOWN as a confidence signal.

The output layer contains ternary neurons. For each sample, some output neurons vote:

* (+1),
* (-1),
* or (0).

A high fraction of zero outputs indicates that the circuit is unwilling to strongly support either class.

The authors use the classification margin to rank examples by confidence, retain only the most confident fraction, and measure accuracy on that subset.

For example, on the Moons dataset:

* binary full-coverage accuracy: 91.8%;
* ternary raw accuracy: 85.8%;
* ternary accuracy at 50% coverage: 98.1%.

So the ternary model is worse when forced to classify every sample, but becomes more accurate when allowed to reject half of them.

This is **selective prediction**:

$
\text{higher accuracy}
\quad\text{in exchange for}\quad
\text{lower coverage}.
$

Table 2 on page 9 reports similar behavior for several synthetic datasets. 

# 11. Does UNKNOWN really represent uncertainty?

The authors attempt to show that UNKNOWN is not simply a failed activation.

They use Gaussian classification problems with adjustable class separation.

As the classes become easier to distinguish:

* Bayes error decreases;
* the fraction of UNKNOWN outputs decreases;
* the overlap between UNKNOWN density and regions of high Bayes entropy changes consistently.

Figure 3 on page 10 shows that UNKNOWN outputs concentrate near ambiguous decision regions. They report a strong relationship between mean Bayes uncertainty and the UNKNOWN fraction.

Their interpretation is:

> The ternary circuit naturally sends UNKNOWN where the posterior class probability is ambiguous.

This is an interesting result because the model is not explicitly trained with a calibration or abstention loss.

However, “Bayes-optimal uncertainty proxy” is stronger wording than the experiments fully justify. The evidence is based mainly on controlled two-dimensional synthetic datasets.

# 12. Fourier analysis for ternary gates

The paper also develops a ternary Fourier basis.

For a single ternary variable:

$
\phi_0(x)=1,
$

$
\phi_1(x)=x,
$

$
\phi_2(x)=x^2-\frac{2}{3}.
$

The third term is important.

For:

$
x=\pm1,
$

it has one value, while for:

$
x=0,
$

it has another.

Therefore, it explicitly measures whether the input is:

* decided: (\pm1);
* unknown: (0).

There is no equivalent extra dimension in binary logic because for binary values (x^2=1), so the quadratic term contains no new information.

For two inputs, the basis contains nine products:

$
\phi_i(a)\phi_j(b),
\qquad i,j\in{0,1,2}.
$

The authors use this representation to:

* measure spectral complexity;
* distinguish genuinely ternary gates from gates behaving essentially like binary gates;
* regularize toward spectrally sparse, interpretable gates.

This is mainly an analysis and regularization contribution, not the polynomial used directly for the primary training implementation.

# 13. Training-speed result

PST evaluates one nine-term polynomial per neuron.

The binary DLGN baseline evaluates a softmax mixture over 16 gates.

The authors report that ternary PST models train approximately:

$
1.5\times\text{ to }3.1\times
$

faster than binary Soft-Mix DLGNs on an RTX 4090.

At the largest scale:

* binary DLGN soft accuracy: 52.5%;
* ternary PST soft accuracy: 52.1%;
* reported relative training speed: 3.1×.

The comparison is somewhat surprising because ternary neurons are richer, but the nine-term direct polynomial remains cheaper than evaluating and weighting 16 candidate gates.

# 14. Is this hardware efficient?

The final model is a discrete ternary circuit, but hardware efficiency is not demonstrated.

Binary gates map naturally to ordinary digital CMOS.

Ternary logic requires one of the following:

* two-bit encoding per signal;
* multi-level voltage logic;
* special ternary standard cells;
* binary emulation of ternary gates.

That can increase:

* wire count,
* gate complexity,
* switching circuitry,
* encoding and decoding overhead.

The paper states that the circuits could potentially be taped out, but it does not provide:

* FPGA synthesis;
* ASIC synthesis;
* energy;
* area;
* delay;
* routing cost.

Thus, the claim that the final ternary circuit is ultra-efficient remains unverified.

A ternary neuron may require fewer logical nodes while still costing substantially more than one binary gate physically.

# 15. Relation to WARP and the multilinear-CovJac paper

## Compared with WARP

WARP generalizes across the **number of binary inputs**:

```text
2-input binary
4-input binary
6-input binary
```

PST generalizes across the **number of logic values**:

```text
binary values
ternary values
potentially quaternary values
```

Both directly parameterize finite function spaces using polynomial or spectral coefficients rather than enumerating every possible gate.

WARP:

$
2^n
$

coefficients for an (n)-input binary LUT.

PST:

$
q^2
$

coefficients for a two-input (q)-valued gate.

These are different axes of generalization.

## Compared with Multilinear-CovJac

The CovJac paper focuses on:

* two-input binary logic;
* four polynomial coefficients;
* hard or soft vector quantization;
* gradient cancellation and interaction starvation.

PST focuses on:

* two-input ternary logic;
* nine polynomial coefficients;
* continuous polynomial training;
* commitment regularization;
* abstention.

PST does not use:

* STE;
* CovJac;
* hard gate selection during training.

Consequently, it retains a substantial network-level hardening gap—the exact problem the CovJac and Gumbel approaches try to reduce.

# 16. Is the paper important?

Yes, but primarily because it opens a new direction:

> Differentiable logic networks do not need to be Boolean.

The strongest contributions are:

1. making all 19,683 ternary gates trainable with only nine parameters;
2. introducing a general polynomial-surrogate framework for many-valued logic;
3. demonstrating UNKNOWN-based selective prediction;
4. developing a ternary Fourier basis that explicitly captures sensitivity to UNKNOWN.

The paper is less convincing as evidence of a practical replacement for binary DLGNs.

Key weaknesses include:

* relatively low CIFAR-10 circuit accuracy;
* very large networks, up to 512K neurons;
* a remaining 3.7–14.1 percentage-point hardening gap at many scales;
* no hardware implementation;
* abstention results concentrated on simple synthetic tasks;
* raw ternary accuracy is usually worse than binary accuracy;
* no explicit comparison with IWP, WARP, or CovJac;
* fixed random connectivity remains unchanged.

# Bottom line

The paper proposes:

$
\boxed{
\text{Ternary DLGNs}
+
\text{9-coefficient polynomial neurons}
+
\text{commitment-based hardening}
+
\text{UNKNOWN-driven abstention}
+
\text{ternary Fourier analysis}
}
$

The fundamental research contribution is the transition from:

$
\text{Boolean circuit learning}
$

to:

$
\text{many-valued circuit learning}.
$

It is an important conceptual paper because it shows that the direct polynomial representation used in IWP-like methods can scale not only to more inputs, but also to more logic values. However, the current implementation is still a proof of concept: the hardening gap, hardware cost, and full-scale predictive performance remain unresolved.
