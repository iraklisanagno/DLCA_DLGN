### Recurrent Deep Differentiable Logic Gate Networks – Summary Notes

The authors are essentially asking:

> "Can we build an RNN using DLGN layers instead of dense layers?"

rather than introducing a new training method, new parameterization, or new theory. 

---

# What do they actually do?

They take a standard encoder-decoder RNN architecture and replace the dense transformations with DLGN layers. 

In a normal RNN:

$
h_t = f(W_x x_t + W_h h_{t-1})
$

In RDDLGN:

$
h_t = \text{LogicLayer}(x_t,h_{t-1})
$

where LogicLayer is the same DLGN layer from the original work. 

So the main contribution is:

```text
Feedforward DLGN
      ↓
Recurrent DLGN
```

---

# What is actually recurrent?

The encoder contains K-layers that receive:

$
[h_t,k_{t-1}]
$

where ($k_{t-1}$) is the previous hidden state. 

Similarly, the decoder contains P-layers that receive:

$
[p_{t-1},c,l_t]
$

where:

* ($p_{t-1}$) = previous decoder state
* ($c$) = encoder context
* ($l_t$) = current decoder input

This is essentially a DLGN version of a classic RNN encoder-decoder. 

---

# New DLGN ideas introduced?

Very few.

Most of the paper is engineering.

The main additions are:

### 1. Binary embeddings

They use a large embedding layer followed by a sigmoid:

$
x_t=\sigma(E_t)
$

and add a regularizer

$
x(1-x)
$

to push embeddings toward 0 and 1. 

This is probably the most interesting idea in the paper.

They want embeddings that eventually resemble binary vectors.

---

### 2. Recurrent hidden state

They add recurrence by feeding previous hidden states into future DLGN layers. 

Nothing particularly novel mathematically.

---

### 3. Residual hidden-state initialization

They test several hidden-state initialization methods.

Residual initialization performs best. 

This is inherited from the earlier DLGN work rather than being a new contribution.

---

# Experimental results

This is where I became skeptical.

They evaluate on WMT14 English-German translation. 

Results:

| Model       | BLEU |
| ----------- | ---- |
| Transformer | 5.98 |
| GRU         | 5.41 |
| RNN         | 4.59 |
| RDDLGN      | 5.00 |



---

The problem is:

A BLEU score of 5–6 on WMT14 is extremely low.

Modern translation systems are vastly higher.

Even classic attention models are much higher.

This means the paper is really demonstrating:

> "The architecture works."

not

> "The architecture is competitive."

---

# Something I found odd

Look at the parameter counts.

| Model  | Parameters |
| ------ | ---------- |
| GRU    | 9.0M       |
| RNN    | 8.5M       |
| RDDLGN | 40.8M      |



RDDLGN uses:

$
4.5\times
$

more trainable parameters than the GRU.

Yet still performs slightly worse.

This weakens the efficiency argument considerably.

---

# The memorization result

The strongest result in the paper is Figure 4. 

They create a shifted-copy task.

Example:

```text
Input:
the cat sat on mat

Target:
PAD PAD the cat sat
```

The shift increases from 0 to 12.

RDDLGN retains much higher accuracy than RNNs and GRUs. 

This suggests the logic-based recurrence may act like a stronger memory mechanism.

This is probably the most interesting experimental observation.

---

# The gradient analysis section

I was not convinced.

The authors claim robust gradient flow and derive:

$$
\frac{\partial y}{\partial w_i} =
p_i
(
f_i
-
\sum_j p_jf_j
)
$

and argue gradients remain non-zero if experts are diverse. 

However:

1. This is essentially inherited from the original DLGN formulation.
2. It does not address the gradient cancellation issues later identified by IWP.
3. It does not analyze recurrence-specific gradients.

After reading IWP, this section feels outdated.

---

# Relation to IWP

This paper uses the **old DLGN formulation**, not IWP. 

Therefore it still suffers from:

* 16 gate probabilities per node
* gate competition
* gradient cancellation
* discretization gap

all of which later papers try to fix.

---

# If I were extending this work

I would not continue with this architecture as written.

Instead I would build:

```text
RDDLGN
   +
IWP
   +
RI
```

immediately.

The current paper is effectively:

```text
Recurrent + Original DLGN
```

while the field has already moved to:

```text
IWP
```

for better optimization.

---

# My assessment

### Technical novelty

**Low to moderate**

Mostly adapts DLGNs to an RNN setting.

---

### Experimental novelty

**Moderate**

The memorization experiments are interesting.

---

### DLGN innovation

**Low**

No new gate parameterization, training algorithm, or logic representation.

---

### Most interesting takeaway

The paper provides evidence that:

> Logic-based computation can be recurrent and can maintain state across time.

That is valuable because it suggests DLGNs are not limited to image classification and feed-forward tasks. However, from a DLGN research perspective, IWP and Mind-the-Gap are substantially more important papers because they address fundamental limitations of the underlying logic neurons themselves. 
