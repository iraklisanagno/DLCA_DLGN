### Deepgate: Learning neural representations of logic gates - Summary notes

This paper, **DeepGate: Learning Neural Representations of Logic Gates**, is **not a Differentiable Logic Gate Network paper in the Petersen sense**.

It does not train a classifier made from learnable Boolean gates. Instead, the authors use a **graph neural network to understand an already existing digital circuit**.

Their goal is:

> Learn a reusable vector representation, or embedding, for every gate that captures both its logic function and its structural role in the circuit. 

## 1. The fundamental difference from DLGNs

### DLGN

A DLGN is itself the predictive model:

```text
Input data
   ↓
Learnable logic gates
   ↓
Classification output
```

The network learns which Boolean function each artificial neuron should implement.

### DeepGate

DeepGate receives an existing hardware circuit:

```text
Existing circuit/netlist
   ↓
Graph neural network
   ↓
Embedding for every physical logic gate
```

The circuit already contains fixed gates and fixed wiring. DeepGate does not alter or select those gates. It learns a numerical description of each gate so that later EDA tools can analyze the circuit more effectively.

So despite the name “DeepGate,” it belongs mainly to:

* graph representation learning,
* machine learning for EDA,
* logic-circuit analysis,

rather than the DLGN architecture family.

---

# 2. What problem do the authors address?

Earlier machine-learning-based EDA methods generally train a separate model for each task:

* congestion prediction,
* power estimation,
* testability,
* satisfiability,
* net-length prediction.

Such models often require manually prepared circuit features and cannot easily transfer knowledge to another EDA task.

The authors ask:

> Can we pretrain one model that learns a general representation of circuit gates, similar to how pretrained image or language models learn reusable representations?

Each gate receives an embedding:

$
h_v\in\mathbb{R}^d,
$

which should encode:

* what the gate computes,
* where it lies in the graph,
* what logic influences it,
* what downstream logic it influences,
* whether it participates in reconvergent paths.

The authors envision these embeddings later supporting tasks such as power estimation, logic optimization, verification, and equivalence checking. 

---

# 3. Step 1: Convert every circuit to an AIG

Different circuits can use different libraries:

```text
AND, OR, XOR, NAND, NOR, MUX, AOI, ...
```

This creates heterogeneous graphs that are difficult for one GNN to learn.

The authors use the ABC synthesis tool to convert all circuits into an **And-Inverter Graph**, or AIG.

An AIG contains primarily:

* two-input AND nodes,
* inversion operations,
* primary inputs.

Conceptually:

```text
Original circuit:
XOR, NAND, OR, MUX, ...

        ↓ logic synthesis

Unified AIG:
AND + inversion only
```

The Boolean function of the overall circuit remains unchanged, but the representation becomes standardized. 

## Why is this useful?

It reduces the diversity of node types.

Instead of asking the GNN to learn dozens of gate semantics, it primarily learns:

```text
Primary input
AND
NOT
```

Their controlled evaluation shows that training on the unified AIG representation gives noticeably lower prediction error than training directly on heterogeneous original gate-level circuits.

This is one of the paper’s strongest design choices.

---

# 4. Step 2: Generate supervision using signal probability

The authors need a training target that reflects both circuit logic and structure.

For each gate (v), they calculate its **signal probability**:

$
y_v=P(v=1).
$

For example:

* (P(v=1)=0.1): the gate rarely outputs 1.
* (P(v=1)=0.5): the gate is equally likely to be 0 or 1.
* (P(v=1)=0.95): the gate is almost always 1.

They obtain the ground-truth probabilities by simulating each circuit with up to 100,000 random input patterns.

DeepGate then predicts:

$
\hat y_v\approx y_v.
$

The loss is the average absolute error:

$
\mathcal L=
\frac{1}{N}\sum_{v\in V}|y_v-\hat y_v|.
$

The prediction task is not necessarily the final application. It is a **pretraining objective** used to force the gate embeddings to capture meaningful circuit behavior. 

---

# 5. Why signal probability contains meaningful information

Consider a two-input AND gate:

$
z=a\land b.
$

If (a) and (b) were independent:

$
P(z=1)=P(a=1)P(b=1).
$

But practical circuits frequently contain correlated signals. Two inputs might originate from the same earlier signal and later reconverge.

Then:

$
P(a=1,b=1)
\neq
P(a=1)P(b=1).
$

Therefore, accurately predicting signal probability requires understanding more than just the local gate type. The model must understand:

* upstream structure,
* logical dependencies,
* reconvergent fan-out,
* circuit depth,
* signal correlations.

That makes probability prediction a useful representation-learning task.

---

# 6. Their dedicated circuit GNN

The authors argue that an ordinary GNN is insufficient because circuits are directed acyclic graphs with explicit computation order.

DeepGate therefore propagates information in **topological order**.

For each gate (v), it aggregates information from its predecessors:

$
m_v^t
=====

\sum_{u\in P(v)}
\alpha_{uv}^t h_u^t.
$

It then updates the gate representation using a GRU:

$
h_v^t
=====

\operatorname{GRU}
\left(
[m_v^t,x_v],
h_v^{t-1}
\right),
$

where:

* (x_v) represents the fixed gate type,
* (h_v^t) is the learned hidden representation,
* (P(v)) is the predecessor set.

They repeat this process recurrently, typically for ten iterations. 

---

# 7. Why do they use attention?

The aggregation weights (\alpha_{uv}) are learned through attention.

The motivation comes from controlling values in Boolean logic.

For an AND gate:

```text
0 AND anything = 0
```

Thus, an input equal to 0 dominates the output.

For an OR gate:

```text
1 OR anything = 1
```

Thus, an input equal to 1 dominates the output.

The authors argue that not all gate inputs contribute equally. Attention allows the model to place more importance on the logically controlling predecessor.

This is intended to mimic real logic evaluation more closely than simply summing or averaging predecessor embeddings. 

A caveat is that after converting everything to AIG, the primary multi-input gate is AND, so this inductive bias is particularly tailored to AND/inverter logic.

---

# 8. Forward and reverse propagation

Most circuit GNNs propagate information only from inputs toward outputs:

```text
Primary inputs → internal gates → outputs
```

DeepGate also performs **reverse propagation**:

```text
Outputs → internal gates → primary inputs
```

Why?

Forward propagation captures:

> What upstream signals influence this gate?

Reverse propagation captures:

> How is this gate used by downstream logic?

The authors relate this to logic implication and backtracking. A node’s role is not determined solely by its fan-in; its fan-out and downstream consequences also matter.

The resulting architecture alternates:

```text
Forward topological propagation
          ↓
Reverse topological propagation
          ↓
Repeat
```

This produces embeddings containing both upstream and downstream context.

---

# 9. Special treatment for reconvergent fan-out

This is arguably the most technically interesting contribution.

## What is reconvergence?

Suppose signal (x) splits into two paths:

```text
       → path 1 →
x                    → gate z
       → path 2 →
```

The paths later reunite at (z).

The two inputs to (z) are therefore correlated because both originate from (x).

A normal local message-passing GNN may not recognize this dependency, especially when the divergence and reconvergence are many levels apart.

## What DeepGate does

The authors detect:

* the fan-out source node,
* the corresponding reconvergence node,
* their logic-level distance.

They then insert a direct **skip edge**:

```text
Fan-out node ─────────→ Reconvergence node
```

Figure 3 on page 4 illustrates this explicit shortcut. 

They also encode the distance between the source and reconvergence node using positional encoding. This tells the GNN how far apart the two nodes are.

This improves average signal-probability error from:

$
0.0234
$

without reconvergence skip connections to:

$
0.0204
$

with them.

So the authors show that explicitly modeling reconvergence produces a measurable improvement.

---

# 10. Training dataset

The authors extract 10,824 subcircuits from:

* EPFL,
* ITC’99,
* IWLS’05,
* OpenCores.

Training circuits contain approximately:

* 36 to 3,214 nodes,
* 3 to 24 logic levels.

They generate probability labels through random logic simulation and use a 90/10 training/test split. 

---

# 11. Generalization to much larger circuits

The authors train on relatively small subcircuits, then evaluate on circuits containing tens of thousands of gates, including:

* Arbiter: 23.7K nodes,
* Squarer: 36.0K,
* Multiplier: 47.3K,
* 80386 processor: 13.2K,
* Viper processor: 40.5K.

DeepGate continues to predict signal probabilities accurately and consistently outperforms the DAG-GNN baseline.

The strongest improvement is on the Arbiter circuit, where prediction error falls from:

$
0.0277
$

to:

$
0.0073,
$

a reported relative reduction of approximately 73.6%.

The authors attribute this particularly large gain to repetitive and reconvergent structures, which DeepGate models explicitly. 

---

# 12. What do they actually output?

DeepGate produces two related outputs.

### Gate-level embedding

For every node:

$
h_v^T\in\mathbb{R}^{64}.
$

This is intended as a general neural representation of the gate.

### Signal-probability prediction

An MLP maps the embedding to:

$
\hat y_v\in[0,1].
$

The probability prediction is how they train and evaluate the embeddings. The long-term objective is to reuse (h_v^T) for other EDA applications.

However, this paper itself mainly evaluates **probability prediction**, not a broad suite of downstream transfer tasks.

---

# 13. Comparison with DLGNs

| Aspect           | DLGN                                     | DeepGate                                    |
| ---------------- | ---------------------------------------- | ------------------------------------------- |
| Starting point   | Dataset examples                         | Existing circuits                           |
| Nodes            | Trainable artificial logic gates         | Physical gates in a netlist                 |
| Gate function    | Learned                                  | Already fixed                               |
| Connectivity     | Fixed or learned architecture            | Given by the circuit                        |
| Objective        | Classification or prediction             | Circuit representation learning             |
| Inference result | A logic circuit implementing an ML model | Embeddings and signal-probability estimates |
| Main domain      | Efficient neural inference               | EDA and circuit analysis                    |

The only broad similarity is that both concern Boolean gates.

DeepGate does **not**:

* learn AND versus OR for each node,
* discretize a soft gate network,
* create an inference classifier from logic gates,
* use IWP,
* use the DLGN Group-Sum layer.

---

# 14. Critical assessment

The authors’ strongest contributions are:

1. **Unified AIG preprocessing**, which reduces heterogeneous circuit-library effects.
2. **Circuit-specific GNN design** with topological and reverse propagation.
3. **Explicit reconvergence modeling**, rather than expecting generic message passing to discover it.
4. **Generalization from small training subcircuits to much larger circuits.**

The main limitation is that the “general representation” claim is only partially demonstrated. The authors train and evaluate primarily on one task:

$
\text{signal-probability prediction}.
$

They suggest transfer to:

* power estimation,
* logic reduction,
* equivalence checking,
* Boolean satisfiability,

but do not establish this comprehensively in this paper.

Therefore, a more conservative conclusion is:

> The authors demonstrate a strong pretrained representation for signal-probability analysis, with promising—but not yet fully proven—general applicability to broader EDA tasks.

## Bottom line

DeepGate can be summarized as:

$
\boxed{
\text{Circuit}
\rightarrow
\text{AIG}
\rightarrow
\text{circuit-aware GNN}
\rightarrow
\text{embedding per gate}
}
$

It is best viewed as an early **foundation-model-style representation learner for logic circuits**, rather than as a member of the Differentiable Logic Gate Network architecture family.
