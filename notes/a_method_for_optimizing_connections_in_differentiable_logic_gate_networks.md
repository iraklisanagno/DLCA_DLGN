### A Method for Optimizing Connections in Differentiable Logic Gate Networks – Summary Notes 

This paper is actually much more interesting from a research perspective than the recurrent DLGN paper.

The key observation of the authors is:

> The original DLGN learns the gate type but keeps the connections random and fixed.

They argue that this is probably wasteful. 

---

# Original DLGN

Each neuron has:

```text
Input A  ← random wire
Input B  ← random wire

Gate:
AND/OR/XOR/etc.
```

Training learns:

```text
Which gate?
```

but not:

```text
Which inputs?
```

---

# What these authors propose

They introduce trainable connectivity. 

Instead of saying:

```text
Input A ← wire #17
Input B ← wire #93
```

forever,

they say:

```text
Input A can choose among Nc candidates
Input B can choose among Nc candidates
```

and learn the choice.

---

# This is very similar to gate selection

Remember how DLGN chooses gates?

$
p_i=\text{softmax}(w_i)
$

over the 16 gates.

The output is:

$
\sum_i p_i f_i(x_1,x_2)
$

---

They use exactly the same idea for connectivity.

For input A:

$
a=\sum_i p_i g_i
$

where:

* (g_i) are candidate inputs from the previous layer
* (p_i) is a learned softmax distribution



Thus they learn:

```text
Which previous neuron should feed me?
```

---

# Example

Suppose a neuron can choose among:

```text
Input 3
Input 7
Input 10
Input 21
```

Initially:

$
[0.25,0.25,0.25,0.25]
$

After training:

$
[0.01,0.95,0.02,0.02]
$

The neuron learns:

```text
I want Input 7
```

Then they collapse to argmax exactly like DLGN. 

---

# Why this is important

This addresses something we've discussed several times.

In standard DLGN:

```text
gate = trainable
connectivity = random
```

which is strange.

In most neural architectures:

```text
weights
=
both computation + connectivity
```

but DLGNs only learn half of the problem.

---

# What do they find?

They consistently show that trainable connectivity requires dramatically fewer gates. 

Examples:

### Yin-Yang

200 trainable-connectivity gates

outperform

2000 fixed-connectivity gates.

Approximately:

$
10\times
$

smaller network. 

---

### MNIST

3 layers × 4000 gates

beats

6 layers × 8000 gates. 

Approximately:

$
4\times
$

fewer gates.

---

### Fashion-MNIST

2 layers × 4000 gates

matches

6 layers × 8000 gates. 

Approximately:

$
6\times
$

fewer gates.

---

# The most interesting experiment

They attempt to learn **all connections**.

Not just a subset.

For MNIST they report:

```text
2 × 8000 gates
98.68%
```

and claim this is:

```text
24× fewer gates
```

than previous fully-connected DLGNs. 

This is the strongest result in the paper.

---

# Relation to IWP

This paper is orthogonal to IWP.

IWP changes:

```text
gate representation
```

from:

```text
16 gate probabilities
```

to:

```text
truth-table weights
```

---

This paper changes:

```text
connectivity
```

from:

```text
fixed random
```

to:

```text
trainable
```

So they solve different problems.

---

# My assessment

Among all the post-DLGN papers we discussed:

| Paper                   | Importance |
| ----------------------- | ---------- |
| Original DLGN           | Very High  |
| IWP                     | Very High  |
| Mind the Gap (GLGN)     | High       |
| Connection Optimization | High       |
| Recurrent DLGN          | Moderate   |

---

The reason I like this paper is that it attacks what I consider one of the biggest remaining weaknesses of DLGNs:

> Why are we trusting random connectivity?

The results strongly suggest that connectivity matters a lot. 

---

# The most important limitation

The authors still use:

```text
16 gate probabilities
```

for every node. 

So they still inherit:

* gate competition,
* discretization gap,
* gradient cancellation,

that IWP later tries to solve.

---

If I were building the next generation system, I would combine:

```text
IWP
+
Trainable Connectivity
```

because these two papers are attacking completely different bottlenecks.

In fact, among all the DLGN papers you've shown me so far, **this connectivity paper is the one that aligns most closely with the research direction I suggested earlier**: learning *which signals should be connected*, rather than only improving the gate parameterization itself.

--------------

