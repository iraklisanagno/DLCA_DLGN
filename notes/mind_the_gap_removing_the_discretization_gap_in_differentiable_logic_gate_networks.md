### Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks – Summary Notes


This paper is actually very interesting because it attacks a **different problem than IWP**.

If I had to summarize the evolution of the ETH Zürich DLGN work:

```text
2022 DLGN
↓
2024 Convolutional DLGN + RI
↓
2025 Mind the Gap (GLGN)
↓
2025 Light DLGN (IWP)
```

Each paper fixes a different weakness.

---

# What problem are they solving?

The paper identifies what they call the **discretization gap**. 

Recall how a DLGN neuron works:

During training:

$
f=\sum_{i=1}^{16} p_i h_i
$

where:

* ($h_i$) are the 16 logic gates
* ($p_i$) are softmax probabilities

During inference:

$
f=h_k
$

where:

$
k=\arg\max_i p_i
$

Thus training and inference are different.

---

## Example

Suppose a neuron learns:

$
[0.45,0.40,0.15]
$

During training:

```text
45% AND
40% OR
15% XOR
```

During inference:

```text
100% AND
```

because AND has the largest probability.

These are different functions.

The paper argues this mismatch causes a significant accuracy drop. 

---

# Their key observation

Many neurons never fully commit to a single gate.

Figure 9 is perhaps the most important figure in the paper. 

The authors measure neuron entropy.

They find:

* many DLGN neurons still have high entropy
* almost 50% of neurons remain "undecided"

They report:

> 49.81% unused gates in DLGNs. 

This is a surprisingly strong result.

---

# Their solution: Gumbel Logic Gate Networks (GLGN)

Instead of using:

$
f=\sum_i p_i h_i
$

they sample a gate during training using Gumbel noise.

---

### DLGN

Forward pass:

$
f=\sum_i p_i h_i
$

Soft mixture.

---

### GLGN

Forward pass:

$
f=h_k
$

where

$
k=\arg\max_i(z_i+g_i)
$

and

$
g_i\sim\text{Gumbel}(0,1)
$

So every forward pass uses exactly one gate. 

---

# Why is this clever?

Training now resembles inference.

The neuron is forced to make discrete choices.

Instead of:

```text
40% AND
35% OR
25% XOR
```

it learns:

```text
AND
```

or

```text
OR
```

much earlier.

---

# Straight-Through Estimator

The problem:

$
\arg\max
$

is not differentiable.

The solution:

Forward pass:

$
\text{hard gate}
$

Backward pass:

$
\text{soft Gumbel-Softmax}
$

This is the Straight-Through (ST) estimator. 

This lets them:

* train discrete gates,
* still use backpropagation.

---

# Their theoretical contribution

This is the part I found most interesting.

They prove that injecting Gumbel noise approximately adds:

$
\lambda \operatorname{tr}(H)
$

to the objective.

Where:

$
H
$

is the Hessian.

Thus GLGN implicitly minimizes curvature. 

The authors connect this to:

* SAM (Sharpness-Aware Minimization)
* flat minima
* smoother loss landscapes

---

# Main empirical results

The headline numbers are:

| Metric             | Improvement    |
| ------------------ | -------------- |
| Training speed     | 4.5× faster    |
| Discretization gap | 98% reduction  |
| Unused neurons     | 100% reduction |

on CIFAR-10/CIFAR-100. 

---

# How is this different from IWP?

This is the key point.

### GLGN paper

Focus:

> How do we make neurons collapse to a discrete gate faster?

Fixes:

* discretization gap
* slow convergence

---

### IWP paper

Focus:

> How do we parameterize a gate better?

Fixes:

* gradient cancellation
* over-parameterization

---

# From a high-level view

GLGN changes:

```text
training procedure
```

IWP changes:

```text
neuron representation
```

These are orthogonal ideas.

---

# What I find most interesting

After reading both papers, I think they are solving two different symptoms of the same underlying problem.

The original DLGN has:

### Problem 1

Bad gate parameterization

```text
16 gate probabilities
```

causing:

* gradient cancellation
* poor optimization

IWP fixes this.

---

### Problem 2

Training/inference mismatch

```text
soft gates during training
hard gates during inference
```

causing:

* discretization gap
* unused neurons

GLGN fixes this.

---

# If I were extending the work

The obvious next step would be:

```text
IWP
+
Gumbel training
+
Straight Through estimator
```

In fact, I was surprised the Light DLGN paper did not go further in this direction.

A neuron could:

1. Use IWP truth-table parameters

$
\omega_{00},\omega_{01},\omega_{10},\omega_{11}
$

2. Use Gumbel sampling on those entries

3. Train with an ST estimator

This would combine:

* IWP's gradient advantages
* GLGN's discretization advantages

and might be stronger than either paper individually.

From a research perspective, if you are looking for the next paper after IWP, I would seriously investigate whether **IWP + Gumbel + ST** yields another measurable gain, or whether IWP already removes most of the issues that GLGN was addressing. That is a question neither paper appears to answer directly. 


------------------

I have the following questions?
1. Why does the mind the gap paper reduce 100% Unused neurons? What does this mean?
2. You said that in ST inference we use only the gate with the highest value. But isn't it what we do in classic DLGN?

# 1. What does "100% reduction of unused neurons" mean?

The key is Figure 9. The authors measure the **entropy** of each neuron's gate distribution. 

Recall that in the original DLGN a neuron has:

$
p_1,p_2,\ldots,p_{16}
$

representing probabilities of the 16 gates.

---

### Example of a useful neuron

Suppose:

$
[0.99,0.01,0,\ldots]
$

Entropy is very low.

The neuron has essentially chosen:

```text id="08vd5a"
AND
```

and is contributing meaningful logic.

---

### Example of an unused neuron

Suppose:

$
[0.06,0.07,0.05,0.08,\ldots]
$

almost uniform.

Entropy is very high.

The neuron has not decided anything.

It is basically saying:

```text id="w5ytjg"
I don't know whether I am AND
OR
XOR
NAND
...
```

Such a neuron contributes little useful computation.

---

The authors define "unused neurons" as neurons whose entropy remains above a threshold derived from the initialization distribution. 

After training they find:

* DLGN: about 49.8% of neurons remain high entropy
* GLGN: essentially 0%

Thus:

```text id="lmxuln"
49.8% → 0%
```

which they report as a **100% reduction in unused neurons**. 

---

## Why does GLGN help?

Because every forward pass forces a hard choice.

Suppose the logits are:

$
[0.4,0.3,0.3]
$

In DLGN:

```text id="w88gzh"
40% gate 1
30% gate 2
30% gate 3
```

forever.

Nothing forces collapse.

---

In GLGN:

```text id="xwxho8"
sample gate 1
sample gate 2
sample gate 1
sample gate 1
sample gate 3
...
```

Eventually one gate starts winning repeatedly.

The distribution sharpens.

Entropy approaches zero.

So GLGN acts like a gate-selection regularizer.

---

# 2. Isn't inference already selecting the maximum gate in DLGN?

Yes!

This is exactly correct.

Both DLGN and GLGN use:

$
\arg\max_i z_i
$

during inference. 

So the difference is **not inference**.

The difference is **training**.

---

## Original DLGN

### Training

$
f=\sum_i p_i h_i
$

All 16 gates participate.

Example:

```text id="fzzocj"
40% AND
35% OR
25% XOR
```

---

### Inference

$
f=h_k
$

where

$
k=\arg\max_i p_i
$

Only one gate remains.

Example:

```text id="dffw5n"
100% AND
```

---

Notice the mismatch.

Training used:

```text id="e1e4m2"
AND + OR + XOR
```

Inference uses:

```text id="by8yma"
AND only
```

This is the discretization gap.

---

## GLGN

### Training

Forward pass:

$
f=h_k
$

where

$
k=\arg\max_i(z_i+g_i)
$

Only one gate is active.

---

### Inference

$
f=h_k
$

Only one gate is active.

---

Now training and inference look much more similar.

---

# Another way to think about it

The original DLGN trains this neuron:

```text id="5l7f4f"
0.4 AND
0.3 OR
0.3 XOR
```

but deploys:

```text id="5u1gsv"
AND
```

---

GLGN trains:

```text id="jlwmf2"
AND
```

(or OR, or XOR)

and deploys:

```text id="qepv8z"
AND
```

The training objective is much closer to deployment.

This is why the discretization gap becomes tiny.

---

# The interesting question for IWP

This is where I think the paper leaves an opening.

IWP already pushes neurons toward a truth table:

$
[\omega_{00},\omega_{01},\omega_{10},\omega_{11}]
$

instead of a 16-way distribution.

So I would ask:

> Do high-entropy unused neurons still exist under IWP?

If IWP naturally causes truth tables to collapse toward binary values:

$
[0,1,1,0]
$

then much of the motivation for GLGN may disappear.

I have not yet seen a paper that carefully measures:

```text id="pk9gmx"
DLGN
vs
GLGN
vs
IWP
vs
IWP + GLGN
```

using entropy and discretization-gap metrics.

That comparison would be extremely valuable, and could be a publishable study by itself.

---------

But after traing, DLGN will select the highest wo 1 gate only. Smae with GLGN. They select the max during training and ifnerence eventually. I struggle to see the argument of neuron reduction

I think you're noticing a real weakness in the paper's presentation.

Let's separate two things:

### Claim A

> GLGN reduces the discretization gap.

I largely agree.

### Claim B

> GLGN eliminates unused neurons.

This requires more careful interpretation.

---

## What you're correctly observing

At deployment both methods do:

$
\text{gate}=\arg\max_i p_i
$

So after training:

DLGN:

```text
[0.40,0.35,0.25]
→ AND
```

GLGN:

```text
[0.40,0.35,0.25]
→ AND
```

Both end up selecting AND.

So why should we care that the DLGN neuron still has entropy?

At inference we don't.

This is exactly your point.

---

# The paper's argument

The authors are not saying:

> High entropy neurons hurt inference directly.

They are saying:

> High entropy neurons indicate poor training.

Consider two neurons:

### Neuron 1

$
[0.99,0.01,0,\ldots]
$

Clearly committed.

### Neuron 2

$
[0.34,0.33,0.33,\ldots]
$

Almost undecided.

Both may eventually deploy as a single gate.

But during training:

Neuron 1 behaves almost identically to deployment.

Neuron 2 behaves very differently.

---

## Example

Suppose the top gates are:

```text
40% AND
35% OR
25% XOR
```

Training behavior:

$
0.4AND + 0.35OR + 0.25XOR
$

Deployment behavior:

$
AND
$

Very different function.

The neuron never really learned to be AND.

It learned to be a mixture.

---

GLGN tries to force neurons to commit earlier.

The claim is:

```text
training gate
≈
deployment gate
```

which improves optimization.

---

# Why I'm somewhat skeptical

Your observation exposes something important.

The paper measures:

> entropy reduction

and interprets that as:

> neuron utilization improvement.

Those are not exactly the same thing.

---

A high-entropy neuron could still be extremely useful.

Imagine:

$
[0.45,0.45,0.10]
$

Maybe that mixture produces excellent gradients.

Maybe it helps the network learn.

Entropy alone does not prove the neuron is useless.

---

# What I would have liked to see

The paper would be much stronger if it measured:

### Actual gate switching

How many neurons change gate after discretization?

For example:

```text
epoch 200:
40% AND
35% OR
25% XOR

deployment:
AND
```

Did this change the output significantly?

---

### Functional importance

Ablate those "unused" neurons.

Remove them entirely.

Does accuracy drop?

If not:

```text
high entropy
→ truly unused
```

If accuracy drops:

```text
high entropy
→ still useful
```

---

# My interpretation

I think "100% reduction of unused neurons" is slightly marketing-oriented wording.

A more precise statement would be:

> GLGN forces nearly all neurons to converge to low-entropy gate distributions.

That is unquestionably true according to their measurements. 

But:

> GLGN removes 100% of functionally unnecessary neurons

is not something the paper proves.

---

# The deeper question

Interestingly, this is exactly where IWP may change the story.

The GLGN paper assumes the original DLGN parameterization:

$
p_1,\ldots,p_{16}
$

and measures entropy over gate probabilities.

IWP has no gate probabilities.

It has:

$
[\omega_{00},\omega_{01},\omega_{10},\omega_{11}]
$

So the notion of:

> "This neuron is 40% AND and 35% OR"

doesn't even exist anymore.

That's one reason I suspect that some of the GLGN motivation may disappear under IWP.

If I were reviewing a future paper, one of the first questions I'd ask is:

> Does the 50% unused-neuron phenomenon still exist under IWP, or was it largely a consequence of the original softmax-over-16-gates formulation?

That experiment would be very revealing.

-------------------------


The core purpose of Gumbel is:

> **Make a discrete decision trainable with gradient descent.**

Classification is just one application.

---

## What Gumbel is solving

Suppose your model must choose one option:

$
k = \arg\max_i z_i
$

The problem is:

$
\arg\max
$

is not differentiable.

So gradients cannot flow through the decision.

Gumbel-Softmax provides a differentiable approximation.

This can be used anywhere you need:

* gate selection,
* routing,
* architecture search,
* expert selection,
* connectivity learning,
* operator selection.

---

# Example 1: DLGNs (Mind the Gap paper)

Choose:

```text
AND
OR
XOR
...
```

Gumbel makes gate selection trainable.

---

# Example 2: Mixture of Experts (MoE)

Choose:

```text
Expert 1
Expert 2
Expert 3
```

instead of evaluating all experts.

Modern LLMs often use similar ideas.

---

# Example 3: Connectivity Learning

Remember one of the ideas we discussed:

Instead of random inputs:

```text
Neuron
  ↓
Input 17
Input 54
```

learn which inputs to connect to.

You could use Gumbel to select:

$
x_{17}
$

instead of

$
x_{54}
$

This is actually one of the ideas I find more promising than IWP itself.

---

# Example 4: Neural Architecture Search

Choose:

```text
3x3 Conv
5x5 Conv
Skip Connection
Attention
```

Gumbel is widely used here.

---

# Example 5: Hardware Mapping

This is where it becomes relevant to your work.

Suppose a runtime system must choose:

```text
CPU
GPU
NPU
FPGA
```

for each layer.

You can make this decision differentiable with Gumbel.

People do this in accelerator co-design papers.

---

# For your DLGN research

The interesting question is not:

> Can Gumbel be used?

Absolutely yes.

The interesting question is:

> Where should Gumbel be used?

I see three possibilities.

---

## Option 1: Gate Selection (Mind the Gap)

Current paper:

```text
16 candidate gates
↓
Gumbel
↓
select one
```

---

## Option 2: Input Selection

Much more interesting in my opinion:

```text
100 candidate inputs
↓
Gumbel
↓
select two
```

This would learn connectivity.

Current DLGNs do not do this.

---

## Option 3: Variable-Arity Gates

Suppose a neuron chooses:

```text
2-input gate
3-input gate
4-input gate
```

using Gumbel.

This directly addresses one of the limitations of IWP.

---

# What happens with IWP?

This is where things become tricky.

The Mind the Gap paper works because OP has:

$
p_1,\ldots,p_{16}
$

a categorical distribution.

Gumbel naturally operates on categories.

---

IWP does not have categories.

It has:

$
\omega_{00},\omega_{01},\omega_{10},\omega_{11}
$

continuous truth-table entries.

So there is nothing obvious to "sample."

You would need to define a new discrete decision.

For example:

### Gate discretization

Sample:

$
\omega_{ij}\in{0,1}
$

using Straight-Through Bernoulli estimators.

---

### Connectivity selection

Use Gumbel on input choices.

This is the direction I would explore.

---

# My research opinion

If I were trying to outperform IWP, I would **not** use Gumbel for gate selection.

IWP already removed much of the gate-selection machinery.

Instead, I would use Gumbel for:

```text
Which inputs should a gate connect to?
```

because:

1. Current DLGNs use fixed random connectivity.
2. Connectivity probably affects accuracy more than the exact gate parameterization.
3. This problem remains completely unsolved by IWP.

That, in my opinion, is where Gumbel could have the largest impact on future DLGN architectures.
