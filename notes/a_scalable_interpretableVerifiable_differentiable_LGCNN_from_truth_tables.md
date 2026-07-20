### A Scalable, Interpretable, Verifiable & Differentiable Logic Gate Convolutional Neural Network Architecture From Truth Tables – Summary Notes 

This paper is **not really a DLGN paper**, even though it is often discussed in the same logic-learning space.

The authors are not learning logic gates like:

```text
AND
OR
XOR
NAND
```

as DLGNs do.

Instead, they are trying to answer:

> Can we build a CNN whose filters are equivalent to small truth tables that can later be converted into Boolean formulas, decision trees, and logic circuits? 

---

# Core idea

The key building block is called a:

### Learning Truth Table (LTT)

Instead of a DLGN neuron:

```text
Input A
Input B
     ↓
16-gate mixture
     ↓
Output
```

they build a small CNN block:

```text
Binary inputs
      ↓
Small CNN
      ↓
Binary output
```

and then enumerate all possible inputs to create a truth table. 

---

# Example

Suppose the block receives:

$
n=6
$

binary inputs.

Then there are:

$
2^6 = 64
$

possible input combinations.

The authors evaluate all 64 combinations and store:

```text
Input Pattern → Output
```

as a truth table. 

---

# Compare with DLGN

DLGN:

Learns:

$
f(x_1,x_2)
$

by selecting among 16 Boolean functions.

---

TTNet:

Learns an arbitrary function:

$
f(x_1,\ldots,x_n)
$

using a tiny CNN.

After training:

```text
CNN
↓
Truth Table
↓
Boolean Formula
↓
Logic Circuit
```



---

# Why can they do this?

The trick is keeping the truth table small.

They enforce:

$
n \le 16
$

inputs per block. 

Otherwise:

$
2^n
$

becomes enormous.

---

Example:

| Inputs | Truth Table Size |
| ------ | ---------------- |
| 6      | 64               |
| 10     | 1024             |
| 16     | 65536            |
| 20     | 1,048,576        |

You can see why they stop at 16.

---

# What is an LTT block?

Figure 1 is the key figure. 

An LTT block is:

```text
Conv
↓
BatchNorm
↓
SeLU
↓
Conv
↓
BatchNorm
↓
Binary Step Function
```

The weights remain real-valued.

Only the inputs and outputs are binary.

---

This is very different from DLGNs.

DLGN computes logic directly.

TTNet learns a CNN and then extracts logic afterward.

---

# How do they get Boolean formulas?

After training they:

1. Enumerate all possible inputs.
2. Construct a truth table.
3. Run Quine-McCluskey minimization.
4. Obtain DNF/CNF formulas. 

Example from the paper:

A trained filter becomes:

$
(x_2 \wedge x_3) \vee x_0
$



---

# The most interesting idea

In my opinion, it is **not** the truth table.

It is the concept of:

### Learn first, extract logic later

DLGN philosophy:

```text
Constrain network to be logic
during training
```

---

TTNet philosophy:

```text
Train a small CNN
↓
Convert to logic afterward
```

---

# Human knowledge injection

This is another unusual contribution.

After extracting rules, they allow insertion of:

### Don't Care Terms (DCTs)

Example:

The rule:

```text
Born in USA
AND
Born in France
```

is impossible.

So they mark such combinations as:

```text
Don't Care
```

and simplify the logic. 

---

This is something DLGNs cannot do naturally.

---

# Relation to DLGNs

The connection is actually weaker than it first appears.

DLGN learns:

```text
Logic Gates
```

directly.

---

TTNet learns:

```text
CNN Filters
```

and later converts them into:

```text
Truth Tables
```

---

So I would classify TTNet as:

```text
Logic-Extractable CNN
```

rather than:

```text
Differentiable Logic Gate Network
```

---

# What I like

1. Very interpretable.
2. Can be converted to SAT formulas.
3. Formal verification becomes easy.
4. Human knowledge can be injected after training. 

---

# What I don't like

The biggest limitation is scalability.

Everything relies on:

$
2^n
$

truth table enumeration.

The authors explicitly restrict themselves to:

$
n \le 16
$

because larger truth tables become impractical. 

---

# Compared to IWP

This is where things become interesting.

IWP learns:

$
2^n
$

truth-table coefficients directly.

TTNet extracts:

$
2^n
$

truth-table entries after training.

So both papers are actually moving toward the same destination:

> Represent computation as truth tables instead of mixtures of logic gates.

The difference is:

| IWP                       | TTNet                              |
| ------------------------- | ---------------------------------- |
| Truth table is the neuron | Truth table is extracted afterward |
| Logic-first               | CNN-first                          |
| End-to-end differentiable | CNN training + logic extraction    |
| Tiny logic neurons        | Small CNN filters                  |

From a research perspective, I think IWP is more elegant mathematically, while TTNet is more focused on interpretability and formal verification. The two papers are much closer philosophically than they first appear, because both ultimately treat computation as a truth table rather than as a weighted mixture of AND/OR/XOR gates.
