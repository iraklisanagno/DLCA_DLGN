### Learning Interpretable Differentiable Logic Networks – Summary Notes 

This paper is actually quite different from the ETH/Max Planck DLGN line.

The original DDLGN papers assume:

```text
Fixed Thresholds
Fixed Connectivity
Learn Gate Type
```

Later papers add:

```text
Learn Connectivity
(IWP) Better Gate Parameterization
(GLGN) Better Training
```

This Princeton paper asks:

> Can we learn **everything** from scratch and still end up with an interpretable logic network? 

---

# Main idea

The authors introduce what they call a **Differentiable Logic Network (DLN)**.

The architecture consists of three layers:

```text
Continuous Inputs
        ↓
ThresholdLayer
        ↓
LogicLayers
        ↓
SumLayer
```

Page 3 contains the overview figure. 

The key difference from DDLGN is:

They make **all three components trainable**:

1. Thresholds
2. Logic functions
3. Connections

---

# Contribution 1: Trainable ThresholdLayer

This is probably their most novel idea.

Remember that DDLGNs assume binary inputs.

The ETH papers therefore require:

```text
continuous feature
      ↓
manual threshold
      ↓
binary input
```

or decision-tree preprocessing.

---

These authors instead learn the thresholds.

Each threshold neuron learns:

$
y=\sigma(s(x-b))
$

where:

* (b) = threshold
* (s) = slope

Both are trainable. 

---

Example:

Initially:

```text
x > 0.5
```

After training:

```text
x > 0.73
```

or

```text
x > 0.12
```

depending on what improves accuracy.

---

The authors show that some thresholds even move outside the valid range:

```text
b < 0
```

or

```text
b > 1
```

which effectively creates:

```text
TRUE
```

or

```text
FALSE
```

constants. 

This acts as feature selection.

---

# Contribution 2: Learn Logic Gates

This part is basically DDLGN.

Each neuron learns:

$
p_0,\ldots,p_{15}
$

over the 16 Boolean functions.

Output:

$
y=\sum_k p_k,Logic_k(a,b)
$

using softmax weights. 

After training:

```text
argmax
```

selects a single gate.

This is identical in spirit to the original DDLGN.

---

# Contribution 3: Learn Connectivity

This is where the paper becomes interesting.

Instead of fixed random wiring:

```text
Input A ← random neuron
Input B ← random neuron
```

they learn the connection.

For input (a):

$
a=\sum_j Softmax(u_j)x_j
$

For input (b):

$
b=\sum_j Softmax(v_j)x_j
$

where (u) and (v) are trainable connection weights. 

---

This is very similar to the connectivity-learning paper we discussed earlier.

The neuron can decide:

```text
I want input #17
```

instead of accepting random connectivity.

---

# Contribution 4: Two-Phase Training

This is the most distinctive part of the paper.

The authors argue that simultaneously learning:

```text
gate type
+
connections
```

is difficult.

So they alternate.

---

## Phase I

Freeze connections.

Learn:

```text
thresholds
gate types
```



---

## Phase II

Freeze gate types.

Learn:

```text
connections
```



---

Then repeat:

```text
Phase I
↓
Phase II
↓
Phase I
↓
Phase II
...
```

This is shown in Figure 2 and Algorithm 1. 

---

# Contribution 5: Straight-Through Estimators

The authors explicitly use STE.

During forward pass:

```text
hard threshold
hard argmax
```

During backward pass:

```text
soft sigmoid
softmax
```

This allows them to train discrete logic while maintaining gradients. 

This is conceptually similar to what GLGN later does with Gumbel/ST.

---

# Contribution 6: Search Subspaces

This section caught my attention.

The authors observed that searching all:

```text
16 gates
all connections
```

is unnecessary.

They rank gates by:

```text
logic completeness
gradient quality
```

and often search only:

```text
8 gates
8 candidate connections
```

instead of everything. 

This is very similar to the candidate-pool idea from the connectivity paper.

---

# How does this compare to DDLGN?

### Original DDLGN

Learns:

```text
gate type
```

Assumes:

```text
binary inputs
fixed connectivity
```

---

### This DLN paper

Learns:

```text
thresholds
gate type
connectivity
```

from scratch. 

---

# How does this compare to IWP?

This paper predates IWP and still uses:

$
p_0,\ldots,p_{15}
$

the original 16-gate parameterization. 

Therefore it still suffers from:

* gate competition,
* gradient cancellation,
* discretization gap,

which IWP later tries to address.

---

# My assessment

Among the ideas in this paper, the most valuable are:

### Very interesting

* Trainable ThresholdLayer
* Trainable Connectivity

### Moderately interesting

* Alternating Phase I / Phase II optimization

### Least interesting

* Logic gate learning itself (borrowed from DDLGN)

---

From a research perspective, the paper is important because it demonstrates that **learning thresholds and connectivity matters a lot**. In some sense, it attacks exactly the limitation that bothered us in the original DLGN:

> Why should thresholds and wiring be random or fixed if the gate itself is trainable?

The paper's answer is:

> They shouldn't be. Learn them jointly (or alternately) with the logic functions. 

If I were building a next-generation DLGN today, I would probably combine:

```text
IWP
+
Trainable Thresholds (this paper)
+
Trainable Connectivity (this paper)
```

because these ideas are largely orthogonal and address different weaknesses of the original formulation.


----------------------

I did not understnad the learning threshold? Which threshold did the original DLGN have?


This is actually the key insight of the Princeton paper, and it's easy to miss because they use the word **ThresholdLayer**.

The original DDLGN papers assume the inputs are already binary.

---

# Original DDLGN

Suppose you have a feature:

```text
Age = 42
```

A logic gate cannot process 42.

It expects:

```text
0
or
1
```

So before training, somebody must convert:

```text
Age = 42
```

into something like:

```text
Age > 30 ? 1 : 0
```

or

```text
Age > 50 ? 1 : 0
```

or

```text
Age > 70 ? 1 : 0
```

---

The question becomes:

> Where do these thresholds come from?

In the original DDLGN paper they are fixed before training.

For image datasets they often use uniform thresholds.

For tabular datasets they use manually selected or decision-tree-based thresholds. 

So the pipeline is:

```text
Raw Feature
      ↓
Fixed Thresholds
      ↓
Binary Inputs
      ↓
DLGN
```

---

# Example

Suppose income ranges from:

```text
0 - 100k
```

A designer might choose:

```text
Income > 20k
Income > 40k
Income > 60k
Income > 80k
```

producing:

| Income | >20 | >40 | >60 | >80 |
| ------ | --- | --- | --- | --- |
| 15k    | 0   | 0   | 0   | 0   |
| 35k    | 1   | 0   | 0   | 0   |
| 90k    | 1   | 1   | 1   | 1   |

These thresholds are fixed.

The network cannot change them.

---

# What this paper does

Instead of:

```text
Income > 20k
```

they learn:

```text
Income > b
```

where

$
b
$

is trainable. 

---

Initially:

$
b=0.2
$

After training:

$
b=0.37
$

or

$
b=0.64
$

or whatever improves accuracy.

---

# Why use the sigmoid?

A hard threshold:

$
y=
\begin{cases}
1 & x>b \
0 & x\le b
\end{cases}
$

has zero gradient almost everywhere.

You cannot train (b).

---

So they temporarily replace it with:

$
y=\sigma(s(x-b))
$

during training. 

This gives gradients with respect to:

* threshold (b)
* slope (s)

Both become trainable.

---

# Visual intuition

Suppose initially:

```text
Threshold = 0.5
```

```text
0--------0.5--------1
          ^
```

Training may discover:

```text
Threshold = 0.8
```

```text
0----------------0.8----1
                  ^
```

because the logic network performs better.

---

# Why I think this is important

Think about the logic rule:

```text
Age > 35
AND
Income > 70k
```

The DDLGN can learn:

```text
AND
```

but not:

```text
35
70k
```

Those numbers are fixed.

---

The Princeton paper can learn both:

```text
Age > 41.7
AND
Income > 62.4k
```

because the thresholds themselves are trainable. 

---

# Relation to decision trees

This is the easiest way to think about it.

A decision tree learns:

```text
Age > 37.2 ?
```

automatically.

The threshold is part of training.

---

Original DDLGN learns:

```text
AND
OR
XOR
```

but assumes:

```text
Age > 30
```

was already given.

---

This paper essentially says:

> Why not learn the decision-tree-style thresholds too?

That is what the ThresholdLayer is doing.

---

From a research perspective, I think this is actually one of the strongest ideas in the paper. The gate-learning part is inherited from DDLGN, but **learning the threshold locations together with the logic rules** is genuinely new and removes a major manual preprocessing step that the original DDLGN relied upon.
