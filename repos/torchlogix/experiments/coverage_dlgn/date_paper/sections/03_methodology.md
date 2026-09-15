# LogicConnect Methodology

## Designing the graph at a fixed budget

LogicConnect constructs the predecessor graph before optimizing gate functions. The objective is to organize signal access while retaining the baseline architecture and training state. For layer $l$ with $m_l$ two-input gates and $n_l$ predecessors, let $I_l\in\{0,\ldots,n_l-1\}^{2\times m_l}$ contain the connection indices. Gate $j$ receives distinct inputs $a=I_l[0,j]$ and $b=I_l[1,j]$. Its relaxed output follows the baseline parameterization [@petersen2022]:

\begin{equation}
h_{l,j}=\sum_{q=0}^{15}\mathrm{softmax}(\theta_{l,j})_q\,\widetilde f_q(h_{l-1,a},h_{l-1,b}).
\label{eq:gate}
\end{equation}

Here, $\widetilde f_q$ is a continuous relaxation of Boolean function $q$. Training updates $\theta$ while holding $I_l$ fixed; hardening selects the highest-weight function. LogicConnect changes the admissible signal combinations through $I_l$, without modifying the loss, gate relaxation, or output aggregation.

The constructor represents available signals through graph metadata. We associate predecessor $i$ with structural ancestry $A_i$ and propagate $A_j=A_a\cup A_b$. In dense image inputs, an ancestry label identifies a raw channel-pixel value, so thermometer thresholds of the same value share a label. Ancestry records potential graph access, rather than functional dependence after gates have been learned. Figure \ref{fig:overview} separates this construction metadata from the fixed indices retained during training and inference.

{{figure:overview}}

## Semantic pairing at the input

Image encodings expose a distinction that index-only pairing misses: two bits can represent either different image sources or different thresholds of one source. LogicConnect therefore uses channel, position, and threshold metadata to form semantic candidate stages, excluding pairs drawn from the same raw source.

Spatial stages pair positions along horizontal and vertical axes at power-of-two strides up to half the axis length, with cyclic boundaries. At the finest scale, additional stages pair aligned channels and mix thresholds with a spatial displacement. Cycling across stages distributes pairs across these axes before repeatedly using one kind of relation. Within a stage, selection favors the smallest endpoint fan-out sum, then maximum endpoint fan-out, then endpoint indices. The initial search examines up to 64 unused candidates in a seeded affine order; after distinct pairs are exhausted, repeated pairs follow the same degree priority. This bounded selection balances semantic variety against construction cost without claiming a global optimum.

## Multiscale structure in deeper layers

Hidden features no longer have the raw image coordinates needed by the input rule. LogicConnect instead constructs regular matchings over predecessor indices and uses ancestry to distinguish their structural support. A matching changes pair identities while using each predecessor at most once.

For $n$ predecessors, the candidate schedule has $K=\lceil\log_2 n\rceil$ scales. At power-of-two widths, scale $s$ pairs $i$ with $i\mathbin{\oplus}2^s$. Other widths use an affine traversal with a coprime step and pair adjacent elements; odd widths rotate the unmatched predecessor. A complete stage contains $\lfloor n/2\rfloor$ pairs. A final partial stage selects disjoint pairs by endpoint degree sum, maximum degree, and index, retaining the regular matching structure.

Stage selection gives priority to predecessor balance. Let $d_i$ denote current fan-out and $P_s$ the pairs retained from candidate stage $s$. The prospective degree spread is

\begin{equation}
D_s=\max_i(d_i+\delta_i(P_s))-\min_i(d_i+\delta_i(P_s)),
\label{eq:degree}
\end{equation}

where $\delta_i(P_s)$ counts occurrences of $i$. Minimizing $D_s$ prevents ancestry preferences from overriding balance. Among equally balanced candidates, the constructor favors smaller relative ancestry overlap:

\begin{equation}
N_s=\frac{1}{|Q_s|}\sum_{(a,b)\in Q_s}\left(1-\frac{|A_a\cap A_b|}{\max(1,\min(|A_a|,|A_b|))}\right).
\label{eq:novelty}
\end{equation}

Here, $Q_s$ is the complete candidate matching, or 2,048 evenly indexed pairs for a larger matching, including when only a partial stage will be retained. The score is zero for identical nonempty ancestry and one for disjoint ancestry. LogicConnect minimizes $D_s$, then maximizes $N_s$, using cyclic nominal scale order to resolve ties. Every scale is used once per cycle. This criterion is a structural heuristic; its incremental accuracy value is tested separately from that of the complete construction.

{{algorithm:construction}}

For even widths, a complete matching increments every predecessor degree once. A final partial matching increments each degree at most once, giving a final spread of at most one from an initially uniform vector. For example, eight predecessors and ten gates use two complete matchings and two pairs from a third, producing degrees of two or three. Degree-preserving randomization tests the value of pair identities beyond this balance property.

## Preserving convolutional semantics

Convolutional DLGNs share gate functions across spatial positions, so unrestricted dense rewiring would change the architecture being evaluated. LogicConnect instead assigns channel pairs while retaining the existing spatial receptive-field sampler, depth-three logic trees, weight sharing, and OR pooling [@petersen2024]. The first channel assignment uses semantic metadata; subsequent channel groups use the multiscale rule.

Channel ancestry begins with encoded-channel identities. Each output channel receives an additional identity label, preserving distinctions after upstream channel support saturates. Flattened classifier features retain upstream ancestry and receive a spatial-feature identity; semantic pairing then precedes deeper classifier matchings. These labels are construction metadata, not exact pixel-dependence sets. The construction operates at each architecture's signal granularity.

## Construction and retained cost

Construction uses packed ancestry bitsets on the CPU. Each stage decision considers $K$ candidate matchings, with overlap work bounded by 2,048 sampled pairs per candidate; ancestry storage still grows with the label universe and layer width. This cost is paid before training and measured separately from GPU allocation. A dense layer retains $16m_l$ gate logits and $2m_l$ fixed indices, with no trainable routing parameters. Deployment retains the hardened functions and connections, while discarding ancestry metadata. Equal logical budgets leave physical area and wire cost dependent on synthesis and placement.
