### ET: An Energy Eficient Edge Transformer Architecture - Summary notes

In this paper, the authors propose **ET: an Energy-Efficient Edge Transformer**. Their key idea is to retain conventional self-attention but replace the expensive feed-forward MLP blocks with differentiable LUT-based networks.

This is therefore not a conventional DLGN classifier. It is a **hybrid transformer architecture**:

$
\boxed{
\text{standard self-attention}
+
\text{differentiable LUT feed-forward block}
}
$

The LUT training method is primarily based on **Differentiable Weightless Neural Networks (DWN)** rather than the original 16-gate DLGN formulation. 

# 1. What problem are the authors addressing?

A transformer block normally consists of:

```text
Self-attention
      ↓
Feed-forward MLP
```

The MLP typically contains two large dense transformations:

$
\operatorname{FFN}(x)
=====================

W_2,\phi(W_1x).
$

These layers require many:

* multiply-accumulate operations;
* parameters;
* memory accesses;
* weight transfers.

The authors characterize GPT-2, BERT, and Llama-like models and find that the feed-forward blocks account for roughly **50–65% of the parameters and computations**, depending on the model and context length.

For GPT-2, Figure 1 on page 3 attributes about:

$
64.87%
$

of the multiplications to the feed-forward block, versus approximately:

$
35.13%
$

to attention. 

Their conclusion is:

> If the feed-forward block can be replaced with a cheaper structure, the complete transformer can become substantially more efficient without redesigning attention.

# 2. Their interpretation: attention versus recall

The authors give the two transformer components different conceptual roles.

### Attention

```text
Attend to tokens
→ derive context
→ computation/reasoning
```

### Feed-forward block

```text
Retrieve stored information
→ recall learned knowledge
```

They argue that information retrieval should behave more like a lookup than a large matrix multiplication.

Therefore, ET performs:

```text
Attention:
standard arithmetic self-attention

Recall:
LUT-based weightless network
```

Figure 1(b) on page 3 illustrates this “compute versus recall” interpretation. This framing is intuitive, although the claim that transformer MLPs are simply knowledge-retrieval blocks is an abstraction rather than a strict functional separation.

# 3. What does the new transformer block look like?

The architecture on page 8 is:

```text
Input token representation
        ↓
LayerNorm
        ↓
Conventional multi-head self-attention
        ↓
Residual connection
        ↓
Learned binarization
        ↓
Two layers of 6-input LUTs
        ↓
Learned conditional summation
        ↓
Residual connection
```

The attention part remains mostly conventional.

The original dense MLP is replaced by what the authors call the **recall block**. 

# 4. Step 1: Learned binarization

The attention output is real-valued, but LUTs need binary inputs.

For each feature (h_d), the model learns several thresholds:

$
b_{d,r}
=======

\mathbf{1}[h_d>\tau_{d,r}],
$

where (\tau_{d,r}) is trainable.

The default configuration uses:

$
8
$

thresholds per input feature, producing eight Boolean wires from each continuous activation.

Because the hard threshold is nondifferentiable, they use a straight-through estimator:

$
\frac{\partial b_{d,r}}{\partial h_d}\approx1,
\qquad
\frac{\partial b_{d,r}}{\partial \tau_{d,r}}\approx-1.
$

This allows the thresholds to be trained end-to-end with the transformer.

This is directly related to the **learnable thresholding** ideas we previously discussed in the Princeton DLN and WARP papers.

# 5. Step 2: Learn which signals connect to the LUTs

The Boolean wires are not connected randomly.

For each LUT input position, the authors learn which binarized feature should be used.

They maintain a score matrix:

$
W\in\mathbb{R}^{B\times Q},
$

where:

* (B) is the number of available binary wires;
* (Q) is the total number of LUT input slots.

During the forward pass, each slot selects:

$
b_{\arg\max_p W_{p,q}}.
$

Thus, connectivity is hard during inference:

```text
LUT input slot q
→ one selected binary wire
```

During training, they use the DWN learnable-mapping gradient rule to update both:

* the connection scores;
* the preceding binary features.

So ET learns:

1. the thresholds converting activations to bits;
2. which bits feed each LUT;
3. the contents of each LUT;
4. the values used to reconstruct the real-valued output.

# 6. Step 3: Learn the LUT truth tables

Their default neuron is a **6-input LUT**.

Each LUT contains:

$
2^6=64
$

trainable entries:

$
u_0,\ldots,u_{63}.
$

During the hard forward pass, these entries are binarized:

$
\hat u_i=H(u_i),
$

and the six input bits form an address:

$
a\in{0,1}^6.
$

The LUT output is:

$
y=\hat u_{\delta(a)},
$

where (\delta(a)) converts the six-bit input into an integer address between 0 and 63.

Therefore, at inference, the computation is literally:

```text
6 input bits
      ↓
64-entry truth table
      ↓
1 output bit
```

They use the **Extended Finite Difference gradient estimator from DWN** to propagate gradients through the LUT address. Unlike an ordinary finite difference that only examines neighboring addresses, EFD uses contributions from all LUT entries, weighted according to Hamming distance. 

# 7. This is not original DLGN or IWP

The distinction is important.

## Original DLGN

A two-input neuron learns a distribution over 16 named Boolean gates.

## IWP

An (n)-input node directly learns (2^n) truth-table entries through minterm interpolation.

## ET

A six-input node learns 64 truth-table entries using:

* hard LUT indexing;
* STE for the stored bits;
* DWN’s Extended Finite Difference estimator for input gradients.

So ET is better classified as:

$
\text{Differentiable Weightless/LUT Transformer}
$

rather than a DLGN transformer.

It belongs to the broader LUT-native network family, but its neuron training is not based on the original Soft-Mix DLGN.

# 8. Step 4: Convert binary LUT outputs back to real activations

A transformer block expects a real-valued output vector of hidden dimension (D). But the final LUT layer produces binary values:

$
z_1,\ldots,z_{N_2}.
$

The authors therefore learn an encoding vector for each LUT:

$
E_i\in\mathbb{R}^D.
$

The output is:

$
o
=

# \sum_{i=1}^{N_2}z_iE_i

z^\top E.
$

This resembles a dense matrix multiplication during training. However, during inference, each (z_i) is binary:

* if (z_i=1), add (E_i);
* if (z_i=0), add nothing.

Thus, inference becomes a **conditional summation** rather than a conventional multiplication:

```text
LUT output = 1 → add learned encoding
LUT output = 0 → skip it
```

The paper claims that the recall block therefore contains no multiplications during inference.

However, it is not completely weightless: the encoding vectors (E_i) are learned numerical values and must still be stored, fetched, and accumulated.

# 9. Default recall-block configuration

The default design reported on pages 10–11 uses:

| Component              | Default |
| ---------------------- | ------: |
| Total LUTs             |     768 |
| First LUT layer        |     256 |
| Second LUT layer       |     512 |
| Inputs per LUT         |       6 |
| Thresholds per feature |       8 |
| LUT layers             |       2 |
| Summation blocks       |     768 |

The authors choose two LUT layers because this mirrors the two dense layers of a conventional transformer FFN.

They report that adding more LUT layers did not provide meaningful accuracy improvements once learnable connectivity was enabled.

# 10. Shared recall block across transformer layers

A major area problem arises if every transformer block contains its own physically instantiated LUT network.

LUTs are stationary logic:

```text
traditional FFN weights:
can be stored off-chip and streamed/reused

physical LUT network:
occupies permanent chip area
```

To reduce this cost, the authors **share one LUT recall block across multiple transformer layers**, inspired by parameter-sharing architectures such as MobiLlama.

Conceptually:

```text
Attention block 1 ─┐
Attention block 2 ─┼→ shared LUT recall block
Attention block 3 ─┘
```

This greatly reduces LUT parameters and physical area, but also constrains every transformer layer to use the same recall function.

The authors find that sharing provides a reasonable accuracy–efficiency trade-off, but this is a potentially substantial reduction in model capacity.

# 11. Hardware architecture

The proposed accelerator is hybrid.

## Attention engine

The attention matrices and projections execute on:

* systolic arrays;
* dedicated softmax hardware;
* layer-normalization units;
* residual adders.

## Recall engine

The feed-forward operation executes using:

* comparator blocks for learned thresholds;
* physical 6-input LUTs;
* routing and fanout trees;
* conditional accumulation units.

Figure 7 on page 11 shows these components connected to shared on-chip SRAM. 

The authors also design separate compute structures for:

* **prefill**, which is matrix-operation dominated;
* **decode**, which processes one token at a time and is vector-operation dominated.

The decode systolic arrays therefore have only one row to avoid low utilization.

# 12. Why fanout is a serious issue

Although the LUT computations themselves are inexpensive, the outputs have large fanout.

In the default architecture:

* each first-layer LUT output drives approximately 12 second-layer inputs;
* each second-layer LUT may feed all 768 summation blocks.

The authors insert buffer trees:

* depth 4 after the first LUT layer;
* depth 10 after the second layer.

This is a very important hardware detail. It shows that the cost of LUT networks is not only the LUT truth tables. Routing, fanout, buffers, and accumulation can dominate.

This issue connects directly to what we saw in the silicon-aware DLGN paper:

> Logic operations may be cheap, while interconnect and aggregation become the actual physical bottlenecks.

# 13. GPU implementation

The authors also implement optimized GPU kernels.

The potential GPU gain comes less from LUTs being naturally efficient on GPUs and more from:

* the reduced parameter count;
* fewer memory transfers;
* replacing parts of the dense FFN;
* fitting larger models into constrained GPU memory.

They report approximately:

* (1.3\text{–}1.6\times) throughput gains in memory-constrained settings;
* around (1.12\times) latency and energy improvement on an NVIDIA T4.

They also claim that a GPT-2 Medium variant that did not fit on the target edge GPU can fit after their ET modifications. 

# 14. Main reported hardware gains

The headline claims include:

* approximately (1.34\times) more prompts/Joule than a BinaryBERT FPGA accelerator;
* approximately (2.1\times) more prompts/Joule than their baseline ASIC accelerator;
* approximately (1.2\times) better prompts/Joule than a BitNet-1.58 comparison;
* approximately (1.12\times) GPU efficiency improvement.

These are meaningful but not orders-of-magnitude improvements at the complete-transformer level.

That is expected: ET only replaces the FFN. Attention, normalization, softmax, residual operations, embeddings, output projection, and KV-cache activity remain.

# 15. What is genuinely new?

The main novelty is not a new LUT gradient method. They inherit much of that from DWN.

Their actual contribution is the integration of differentiable LUTs into a complete transformer:

$
\boxed{
\text{learned thresholds}
+
\text{learned connectivity}
+
\text{DWN-trained LUTs}
+
\text{learned real-valued readout}
+
\text{shared recall block}
+
\text{hybrid accelerator}
}
$

Specifically, they demonstrate that LUT networks can serve as an internal transformer component rather than only as standalone classifiers.

That is important because it moves LUT-native networks toward modern architectures.

# 16. Relation to the recurrent DLGN paper

The recurrent DLGN paper showed that logic layers can process sequential information by maintaining a recurrent state.

ET takes a different route:

```text
Recurrent DLGN:
logic network itself models sequence and memory

ET:
standard attention models sequence/context
LUT network only replaces the token-wise FFN
```

ET is more conservative and probably more scalable because it does not ask the LUT network to learn long-range token relationships.

# 17. Relation to BitLogic

BitLogic explicitly mentioned LUT-native attention, recurrent, and transformer architectures as future directions.

ET is an example of such an extension, but it would not fit cleanly into BitLogic’s current five-axis two-layer classification framework because ET adds:

* transformer topology;
* residual paths;
* attention;
* continuous-to-binary transitions inside every block;
* numerical LUT readout;
* cross-layer parameter sharing.

A useful future study would integrate ET’s recall block into BitLogic and compare:

* DWN;
* LightLUT/IWP;
* WARP;
* CovJac-like LUT training;

under the same transformer architecture.

# 18. Critical assessment

The paper is important because it is among the first serious attempts to incorporate **learnable LUT networks inside transformers**. However, several claims need careful interpretation.

## Strong aspects

* Targets the dominant FFN cost rather than replacing individual multiplications with lookups.
* Trains the LUT transformer end-to-end rather than performing post-training substitution.
* Preserves standard self-attention for token interactions.
* Supports learned thresholds and learned connectivity.
* Proposes GPU, FPGA, and ASIC implementations.
* Separately designs prefill and decode hardware.
* Recognizes the physical-area issue and introduces cross-layer LUT sharing.
* Evaluates both language and vision transformer settings.

## Limitations

### The recall block is not fully weightless

The learned encoding matrix (E) contains real-valued parameters and the output requires many accumulations.

Therefore, the FFN becomes multiplication-free, but not arithmetic-free or memory-free.

### LUT sharing reduces capacity

One recall block shared across multiple transformer layers is not equivalent to the separate FFNs in a conventional transformer. Some of the efficiency comes from parameter sharing, not solely from replacing MACs with LUTs.

A fair ablation must separate:

```text
benefit from LUT computation
versus
benefit from weight sharing.
```

### The method uses DWN’s approximate gradients

Extended Finite Difference is an estimator, not an exact derivative. Its scalability and stability in much deeper transformer models remain uncertain.

### Large routing and fanout costs

The reported buffer-tree depths show that interconnect is a major issue. LUT arithmetic may be cheap while routing and conditional summation become expensive.

### Attention remains conventional

The design does not solve the cost of:

* Q/K/V projections;
* attention matrix computation;
* softmax;
* KV-cache memory;
* output projection.

For long contexts, these can become dominant.

### Edge scale remains limited

The experiments focus on relatively small transformer models such as BERT and GPT-2 variants. It is unclear whether the method scales to modern billion-parameter LLMs without severe LUT area and routing growth.

### Hardware comparisons are complex

Comparisons against BinaryBERT, BitNet, and baseline accelerators involve different models, precisions, hardware assumptions, and design points. The reported ratios are informative, but they are not perfectly controlled iso-model comparisons.

### Training cost is secondary in the paper

The LUT gradient rules and learned connectivity can make training expensive. The authors emphasize inference efficiency, but a complete sustainability comparison should include training cost.

# Bottom line

The paper proposes:

$
\boxed{
\text{ET}
=========

\text{traditional transformer attention}
+
\text{shared differentiable 6-LUT recall network}
}
$

The LUT recall block performs:

```text
continuous activation
→ learned threshold bits
→ learned LUT connectivity
→ two layers of DWN-trained LUT6 nodes
→ conditional addition of learned vectors
→ continuous transformer activation
```

This is an important paper because it moves differentiable LUT networks beyond small standalone classifiers and into a modern transformer architecture.

Its core lesson is:

> Do not try to replace the entire transformer with logic. Preserve attention for context and replace the computation-heavy feed-forward knowledge block with hardware-native LUT operations.

For your research, the major open question is whether the **DWN recall block is actually the best LUT formulation**. The same ET architecture could be evaluated using IWP/LightLUT, WARP, or another higher-arity training method, potentially improving optimization, reducing training cost, and controlling the substantial routing and accumulation overheads.
