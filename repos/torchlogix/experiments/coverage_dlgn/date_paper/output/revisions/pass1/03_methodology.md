# Unified Connectivity Construction

## Graph and construction metadata

We represent layer $l$ by $m_l$ two-input gates and a fixed predecessor matrix $I_l\in\{0,\ldots,n_l-1\}^{2\times m_l}$. Here, $n_l$ is the preceding layer's width. Gate $j$ receives predecessors $a=I_l[0,j]$ and $b=I_l[1,j]$, with $a\ne b$. During training, the gate uses the usual mixture of the 16 relaxed Boolean functions [@petersen2022]:

\begin{equation}
h_{l,j}=\sum_{q=0}^{15}\mathrm{softmax}(\theta_{l,j})_q\,\widetilde f_q(h_{l-1,a},h_{l-1,b}).
\label{eq:gate}
\end{equation}

Training optimizes $\theta$ while keeping $I_l$ fixed. Hardening selects the highest-weight Boolean function, and class-group summation produces the output scores. Our method changes only the construction of $I_l$; the loss, gate parameterization, and inference operators remain those of the corresponding baseline.

To construct these indices, we associate each predecessor $i$ with a set $A_i$ of structural ancestry labels. In a dense image network, an input label identifies one raw channel-pixel value, so multiple thermometer thresholds of that value share a label. Propagation takes the union $A_j=A_a\cup A_b$. These sets describe potential structural access, not functional dependence after gate learning. Figure \ref{fig:overview} shows how the metadata supports construction and is then discarded.

{{figure:overview}}

## Semantic input pairing

At the image input, ancestry alone does not distinguish useful pairings between different sources. We therefore retain each bit's channel, spatial coordinates, and threshold index. Candidate stages pair positions along the horizontal and vertical axes at increasing power-of-two strides, capped at half the axis length. The finest scale also includes aligned channel pairs and threshold-changing pairs with a spatial displacement. Pairs from the same raw source are excluded.

The constructor cycles through these semantic stages to represent several axes rather than exhausting one axis first. Within the selected stage, it examines up to 64 unused candidates and chooses the pair minimizing endpoint fan-out sum, then maximum endpoint fan-out, then endpoint indices. A seeded affine permutation determines candidate order without consuming the training generator. Once a stage's distinct pairs are exhausted, repeated pairs become eligible using the same degree ordering over the stage. This procedure provides a reproducible semantic input graph without claiming globally optimal balance.

## Deeper multiscale matchings

For deeper layers, we replace semantic-coordinate stages with regular matchings over predecessor indices. With $n$ inputs, the candidate family contains $K=\lceil\log_2 n\rceil$ scales. At power-of-two widths, scale $s$ pairs $i$ with $i\mathbin{\oplus}2^s$. Otherwise, an affine traversal with a coprime step visits each predecessor once, and adjacent elements form pairs. Odd widths leave one predecessor unmatched, with the omitted index rotating across the schedule.

These matchings control endpoint reuse independently of scale selection. Let $d_i$ denote the current fan-out and $P_s$ the pairs retained from candidate stage $s$. If only part of a stage fits the remaining width, we select its disjoint pairs by endpoint degree sum, maximum degree, and index. We then evaluate the prospective spread

\begin{equation}
D_s=\max_i(d_i+\delta_i(P_s))-\min_i(d_i+\delta_i(P_s)),
\label{eq:degree}
\end{equation}

where $\delta_i(P_s)$ counts occurrences of predecessor $i$ in the retained pairs. Among stages attaining the minimum spread, the constructor favors lower ancestry overlap. For the complete candidate matching $M_s$, its score is

\begin{equation}
N_s=\frac{1}{|Q_s|}\sum_{(a,b)\in Q_s}\left(1-\frac{|A_a\cap A_b|}{\max(1,\min(|A_a|,|A_b|))}\right).
\label{eq:novelty}
\end{equation}

Here, $Q_s$ contains all pairs in $M_s$, or 2,048 uniformly indexed pairs for a larger stage. The score approaches zero for identical ancestry and one for disjoint ancestry. We score the complete candidate stage even when only a prefix fits. Equal scores follow the cyclic nominal scale order, and each scale is used once before a new cycle begins. Algorithm \ref{alg:construction} summarizes this selection.

{{algorithm:construction}}

For even widths, every complete matching increments each predecessor's degree once. A final subset increments a predecessor at most once, so an initially uniform degree vector ends with spread at most one. This property follows from disjoint endpoints; it does not establish an accuracy advantage. Native dense random connectivity already balances fan-out, making a degree-preserving ablation necessary to examine the benefit of structure.

## Convolutional adaptation

The convolutional adaptation applies the pair constructor to channels while retaining each logic tree's spatial receptive-field sampler, depth, and weight sharing. The evaluated architecture uses two-channel groups, depth-three trees, and OR pooling [@petersen2024]. First-layer pairing uses channel and threshold semantics; later layers use the multiscale matching rule. Thus, the construction changes channel access without replacing convolution with a dense spatial graph.

For this channel representation, ancestry starts with encoded-channel identities and gains a distinct identity label for each subsequent output channel. At flattening, each feature retains its upstream labels and receives a spatial-feature identity. The classifier first uses semantic pairing over the resulting channel-position grid, then the deeper matching rule. These labels prevent channel-support saturation from erasing all distinctions, but remain construction metadata rather than exact pixel-dependence sets.

## Cost and deployment

Across both families, construction uses packed ancestry bitsets on the CPU before training. Candidate-stage scoring adds offline work and temporary storage, which we measure separately from GPU allocation. For a raw dense layer with $m$ gates, the model retains $16m$ trainable gate logits and $2m$ fixed predecessor indices. U2 adds no trainable routing state. Export discards the ancestry metadata and retains the selected Boolean functions and indices; physical implementation costs still require synthesis and routing measurements.
