# Continual Learning with Differentiable Logic Gate Networks

**Prepared:** July 22, 2026

**Terminology:** The established research term is *continual learning* (CL). "Continuous learning" is used here only when referring to a continuously operating deployment.

## Executive assessment

A targeted search found no paper that directly develops continual learning for differentiable logic gate networks (DLGNs). This is a real gap, but the adjacent binary-network literature is already strong:

- fully binarized networks have been studied for class-incremental learning since at least ISCAS 2024;
- binary native and latent experience replay have been evaluated on CIFAR-100 and CORe50;
- on-device continual learning with quantized gradients and binary latent replay has been demonstrated;
- Generative Binary Memory (2026) generates pseudo-replay samples in a binary embedding space.

Consequently, **applying experience replay, EWC, or Learning without Forgetting to an ordinary DLGN is not sufficient novelty**.

The strongest research direction is:

> **RePatch-DLGN: boundary-selected continual learning in which an immutable hard-logic prefix feeds a small plastic suffix, binary latent replay protects previous knowledge, and every deployed update is represented as an exact sparse truth-table patch.**

This is stronger than the earlier DriftPatch/PatchLogic concept because it answers the architectural question "which part of the DLGN should change?" and creates a direct hardware partition:

~~~text
immutable hard prefix | programmable plasticity island | expandable head
~~~

The same method can first be validated in software. An FPGA implementation can then compare a memory-programmable LUT island against partial reconfiguration. FPGA support should strengthen the paper, but it should not be required to establish the learning contribution.

### Recommended ranking

| Rank | Working title | Core contribution | Feasibility | Novelty confidence |
|---:|---|---|---|---|
| 1 | RePatch-DLGN | Joint selection of plastic boundary, binary replay representation, and sparse truth-table update budget | High in software; medium with FPGA | High |
| 2 | EvidenceHead-DLGN | Class-independent logic evidence bus with an expandable quantized or LUT head | High | Medium |
| 3 | Plasticity-Island FPGA | Immutable spatial prefix plus a dual-bank programmable LUT suffix | Medium | Medium-high as a systems contribution |
| 4 | Grow-and-Freeze DLGN | Reserve or append logic capacity for new tasks without modifying old gates | Medium | Medium |
| 5 | Whole-DLGN continual fine-tuning | Conventional replay/regularization applied to all gates | High | Low; baseline only |

Ideas 1--3 are compatible parts of one paper. Idea 4 is better treated as a separate alternative because its resource growth and evaluation assumptions are different.

## 1. Problem definition

### Continual-learning scenarios must be separated

| Scenario | What changes | Does the number of outputs change? | Appropriate DLGN update |
|---|---|---:|---|
| Domain-incremental learning | Sensor, corruption, patient, device, or environment distribution | No | Threshold, route, suffix, or sparse gate patch |
| Class-incremental learning | New classes arrive and task identity is unavailable at inference | Yes | Expandable head plus representation retention |
| Task-incremental learning | New tasks arrive and task identity is known | Usually | Separate small heads or task-specific patches |
| Online/general continual learning | Drift is not divided into clean task boundaries | Maybe | Streaming replay and periodic budgeted patches |

A single paper should evaluate **class-incremental learning plus one domain-incremental stream**. Class incremental learning tests expandable outputs and forgetting; domain incremental learning tests whether updating only the head is insufficient under feature drift.

### What constitutes a DLGN model state?

A deployed DLGN contains several distinct kinds of state:

1. input thresholds or encoder parameters, $\Omega$;
2. connection indices, $M$;
3. hardened gate/LUT truth tables, $T$;
4. GroupSum assignments or evidence groups, $G$;
5. optional quantized output-head parameters, $W$.

Training may additionally retain:

- real-valued gate or truth-table proxies;
- connection logits;
- optimizer moments;
- a hardening temperature or schedule.

This distinction is important. A hard two-input gate needs only four truth-table bits, but continuing gradient training from it requires choosing a new soft state. Many soft states harden to the same function. A continual-learning method should therefore report whether it starts from:

- the complete training checkpoint;
- only the deployed hard circuit;
- the hard circuit plus a compact update-state sidecar.

The most practically interesting setting is the second or third. Requiring a full optimizer checkpoint at every device would undermine the deployment argument.

## 2. Status of the closest state of the art

### Generic continual learning

The established families are:

- **rehearsal:** store old examples, latent features, logits, or generated pseudo-examples;
- **regularization:** penalize changes important to old tasks, as in EWC;
- **distillation:** preserve old outputs without retaining all old labels, as in Learning without Forgetting;
- **parameter isolation:** allocate masks, adapters, or separate capacity;
- **dynamic expansion:** append new modules while freezing previous computation.

Replay remains the most relevant family for this project because DLGNs naturally produce one-bit latent states. DER/DER++ is a particularly important baseline because it combines replay with stored logits rather than relying only on labels.

### Binary and edge continual learning

The following work directly constrains the novelty claim:

1. **On Class-Incremental Learning for Fully Binarized CNNs (ISCAS 2024)** already studies CIL in fully binary models.
2. **Enabling On-device Continual Learning with BNNs (2024/2025)** combines binary latent replay with reduced-precision gradient computation.
3. **Towards Experience Replay for Class-Incremental Learning in Fully-Binary Networks (2025)** compares native and latent replay at equal memory, studies loss balancing and semi-supervised pretraining, and reports that latent replay can update about 90% fewer parameters.
4. **Generative Binary Memory (2026)** fits Bernoulli mixtures in binary embedding space and generates pseudo-replay exemplars, reporting strong CIFAR-100, TinyImageNet, and CORe50 results.

Therefore:

- binary replay is not new;
- freezing a feature extractor and retraining a classifier is not new;
- an expandable binary head is not new by itself;
- an FPGA or edge implementation of generic EWC/replay is not automatically new.

### Current DLGN work remains static

The original DLGN, DLGN-Light, Mind the Gap, WARP, LILogicNet, fully trainable DLGN/LUTN, BitLogic, and FPGN all produce a fixed inference graph after training. Connection-learning papers optimize the graph during initial training; they do not study sequential tasks, forgetting, replay, or patch size. Two-stage unit tying compresses a trained circuit but does not update it from new data.

This leaves a defensible DLGN-specific gap:

> How should the static Boolean circuit be partitioned and edited so that it retains old behavior, learns new behavior, and minimizes the exact number of deployed logic bits that change?

## 3. Why whole-network updating is not the default answer

Updating every gate is an important upper baseline, but it has four disadvantages:

1. **Catastrophic forgetting:** all old Boolean features can change.
2. **Training memory:** differentiable proxies and optimizer state are needed for the entire graph.
3. **Deployment bandwidth:** the complete hard model or bitstream may have to be redistributed.
4. **FPGA disruption:** a hardwired implementation normally needs complete or large-region reconfiguration.

Updating only the final component is much cheaper, but it assumes the old representation remains useful. That assumption is plausible for new classes in a similar visual domain and weak for sensor/domain drift.

The core research question should therefore be empirical and budgeted:

> At which logic-layer boundary should plasticity begin, and how many hard truth-table edits are required to obtain the best stability--plasticity trade-off?

## 4. What the BitLogic output head actually provides

The BitLogic paper factors a LUT-native model into encoder, connectivity, fan-in, node parameterization, and output head. Its alternative head is not a replacement for the logic feature extractor. It first computes a length-$C$ GroupSum vector and then applies a quantized $C\times C$ transform:

$$
u = \operatorname{GroupSum}(h)\in\mathbb{Z}^{C},
\qquad
s = W_q u,\quad W_q\in\mathbb{Z}^{C\times C}.
$$

On the paper's MNIST sweep, this DSP-backed head improves accuracy most for narrow models, but costs DSPs and additional LUT/ASIC logic. It becomes less attractive at the widest tested network. Two limitations matter for continual learning:

1. the input and output dimensions both depend on the current class count $C$;
2. GroupSum has already compressed the features into class-specific evidence before the quantized transform.

For a CIFAR-10 model, a $10\times10$ head cannot simply become a CIFAR-100 classifier by appending 90 output rows: its input evidence vector also has only ten class-specific components. The following options are possible.

### Option A: head-only update

Freeze the DLGN and retrain a new classifier on the final binary gate vector or GroupSum scores.

- **Advantage:** cheapest and fastest update.
- **Problem:** a dense head over all final gates is large; GroupSum features may be too class-specific.
- **Research role:** mandatory baseline, not the principal contribution.

### Option B: class-independent evidence bus

Before the task-specific head, form $m$ reusable evidence groups that do not depend on the number of classes:

$$
e_j = \sum_{i\in\mathcal{G}_j} h_i,\quad j=1,\dots,m,
\qquad
s_t=W_t e,\quad W_t\in\mathbb{Z}^{C_t\times m}.
$$

When classes arrive, append rows to $W_t$. The evidence bus is trained for bit balance, diversity, and augmentation consistency so it is not merely a source-class logit vector.

- **Advantage:** the Boolean core and evidence dimension stay fixed as $C_t$ grows.
- **Cost:** the head is arithmetic unless implemented as a bit-serial or LUT classifier.
- **Novelty:** not enough alone because binary feature extractors with retrained software heads already exist.
- **Best role:** the expandable head inside RePatch-DLGN.

### Option C: pure-logic class slots

Reserve class-specific output-vote groups and activate new groups when classes arrive.

- **Advantage:** remains logic-only.
- **Problem:** requires a known maximum class count or physical expansion/reconfiguration.
- **Best role:** FPGA ablation where maximum capacity is known.

### Option D: update the final logic block and head

Freeze a prefix and retrain a small DLGN suffix whose output feeds GroupSum or the evidence head.

- **Advantage:** much more expressive than head-only adaptation.
- **Hypothesis:** this is the best default operating point.

## 5. Main proposal: RePatch-DLGN

### 5.1 Architecture

Partition a DLGN at layer $l$:

$$
f_t(x)=H_t\left(G_t\left(F_l(x)\right)\right),
$$

where:

- $F_l$ is an immutable, hardened prefix;
- $G_t$ is a plastic logic suffix;
- $H_t$ is an expandable task/class head;
- $z=F_l(x)$ is a binary latent vector.

At deployment, only $G_t$ and $H_t$ can change. The prefix is shared across every update and can be permanently hardwired.

### 5.2 Binary latent replay

Store replay entries at the immutable/plastic boundary:

~~~text
(binary latent z, label y, optional old logits q)
~~~

If the boundary has $d_l$ binary wires, one replay sample costs approximately

$$
B_{\mathrm{latent}}=d_l+B_y+B_q
$$

bits, before compression. This must be compared with raw-image replay at the **same total memory**, not the same number of examples.

Native replay retains more adaptability because the prefix can change. Latent replay stores more examples and avoids recomputing or updating the prefix. This trade-off is already known for BNNs; the new contribution is how it interacts with the DLGN patch and plastic boundary.

### 5.3 Warm-starting from a hard circuit

For each old truth table $T_g^{t-1}$, reconstruct a soft proxy centered on its hard entries. For an IWP/LightLUT-style table:

$$
\theta_{g,r}^{(0)}=
\begin{cases}
+\rho,&T_{g,r}^{t-1}=1,\\
-\rho,&T_{g,r}^{t-1}=0,
\end{cases}
$$

where $\rho$ controls the initial commitment. Compare this hard-only reconstruction with retaining the previous real-valued proxy.

This experiment is important because it determines whether deployed devices need a large training sidecar. A successful hard-only warm start is more valuable than a method that silently retains the original optimizer state.

### 5.4 Patch-aware continual objective

For task/update $t$, optimize

$$
\begin{aligned}
\mathcal{L}_t={}&
\mathcal{L}_{\mathrm{new}}
+\alpha\mathcal{L}_{\mathrm{replay}}
+\beta\mathcal{L}_{\mathrm{DER/KD}}\\
&+\lambda_T\sum_{g\in G_t}
\mathbb{E}\!\left[d_H(T_g,T_g^{t-1})\right]
+\lambda_M\sum_g m_g,
\end{aligned}
$$

where $m_g$ is a hard or straight-through edit mask. A top-$K$ projection enforces at most $K$ changed LUTs/gates.

The expected Hamming term should be computed in truth-table space, not as an $L_2$ distance between arbitrary logits. This makes the training regularizer correspond to the final update representation.

### 5.5 Exact deployed patch

For a fixed graph with $N_u$ updateable $k$-input LUTs, an uncompressed sparse patch costs approximately

$$
B_{\mathrm{patch}}
=|\mathcal{P}|
\left(
\left\lceil\log_2N_u\right\rceil+2^k
\right)
+B_{\mathrm{head}},
$$

where $\mathcal{P}$ is the changed-LUT set and $B_{\mathrm{head}}$ is the separately serialized head update. For two-input gates, the new truth table is four bits. Delta IDs, XOR masks, sorted indices, run-length encoding, and entropy coding should be measured rather than assumed.

If connections also change, add the source indices or retained MUX-address bits explicitly. Gate-only and gate-plus-route updates must be reported separately.

### 5.6 Selecting the plastic boundary

Enumerate the small set of layer boundaries rather than learning a complicated differentiable cut. For each boundary $l$ and patch budget $K$, measure

$$
J(l,K)=
A_{\mathrm{CL}}(l,K)
-\lambda_R B_{\mathrm{replay}}(l)
-\lambda_P B_{\mathrm{patch}}(l,K)
-\lambda_U C_{\mathrm{update}}(l,K).
$$

The paper's central result should be a Pareto surface, not one hand-selected suffix:

~~~text
continual accuracy / forgetting
versus replay bits / patch bits / update computation
~~~

A useful hypothesis is that middle boundaries dominate:

- very early cuts permit adaptation but require large update regions and high-dimensional replay;
- very late cuts are cheap but too rigid;
- an intermediate binary representation stores many replay samples and leaves enough plastic logic.

## 6. Alternative update strategies to evaluate

### 6.1 Head-only continual learning

Update only $H_t$ and append class outputs.

- Use it when new classes are visually related and the prefix is transfer-ready.
- Expect it to fail under large domain shifts.
- Compare the original GroupSum, BitLogic's quantized head, and the class-independent evidence bus.

### 6.2 Last-block continual learning

Freeze all but the final logic layer/block and head.

- This is the simplest RePatch instance.
- It supports latent replay naturally.
- It should be the first prototype.

### 6.3 Sparse edits anywhere

Allow top-$K$ gates across the complete network to change based on a plasticity score such as

$$
P_g=\frac{\|\nabla_g\mathcal{L}_{\mathrm{new}}\|}
{\epsilon+I_g^{\mathrm{old}}},
$$

where $I_g^{\mathrm{old}}$ measures old-task sensitivity.

- More flexible than a suffix.
- Harder to map to one FPGA reconfiguration region.
- Requires scattered patch addressing.
- Useful as a software upper baseline for boundary-restricted patches.

### 6.4 Threshold-only update

Update input thresholds while all gates remain fixed.

- Best for calibration or monotonic sensor offsets.
- Very small adapter.
- Insufficient for new classes or new feature interactions.
- Corresponds to PersonalDLGN in the existing idea notes and must be included for domain drift.

### 6.5 Route-only update

Retain small candidate MUXes and change their addresses while truth tables stay fixed.

- Best when old functions remain useful but need recomposition.
- Incurs permanent MUX cost if routes remain field-programmable.
- Closely related to ConfigMux-DLGN; binary masks and Piggyback are important adjacent work.

### 6.6 Whole-network replay

Update all gate proxies and optionally all connections.

- Highest plasticity.
- Highest training and deployment cost.
- Required as an accuracy upper reference.
- Not the proposed method.

### 6.7 Progressive growth

Freeze old gates and append new gates or a new column for every task.

- Eliminates parameter overwriting.
- Closely related to progressive networks and dynamic expansion.
- Circuit area grows with tasks.
- Class-incremental calibration can still reduce old-class accuracy even if old logits are unchanged.

A DLGN-specific version can reserve disconnected constant/copy gates and activate them later, but preallocating unused FPGA capacity must be counted. This is a follow-up if sparse overwriting cannot preserve old tasks.

## 7. FPGA realizations

### 7.1 Server-trained patch, software-applied LUT array

Train the update on a GPU/server, send a semantic patch, and apply it to a packed software model.

- Lowest engineering risk.
- Directly validates patch bytes and update atomicity.
- No FPGA claim should be made from this experiment.

### 7.2 Memory-programmable plasticity island

Implement the immutable prefix as ordinary hardwired logic. Implement only the suffix as a small programmable LUT overlay:

~~~text
hardwired prefix
      |
binary boundary registers
      |
dual-bank truth-table SRAM/BRAM + fixed candidate routing
      |
expandable head
~~~

The inactive bank receives a patch while the active bank continues inference. A version check and bank swap make the update atomic and permit rollback.

**Advantages**

- table words and route addresses can be changed without resynthesis;
- update time is proportional to patch words;
- the mutable region is much smaller than a fully programmable accelerator.

**Costs**

- memory and MUX overhead compared with hardwired FPGA LUTs;
- lower maximum frequency and higher latency;
- fixed candidate routing limits future plasticity;
- on-device gradient training still needs a processor or dedicated learning engine.

The research comparison is not "FPGA is reconfigurable." It is:

> How small can the programmable island be while retaining continual-learning accuracy, and what performance is lost relative to a fully hardwired DLGN?

### 7.3 Dynamic partial reconfiguration

Place the prefix in the static region and the suffix/head in one reconfigurable partition. AMD Dynamic Function eXchange and Intel/Altera partial reconfiguration can replace a region while the rest of a design remains active.

**Advantages**

- the suffix can remain spatial, hardwired, and fast;
- arbitrary logic inside the region may change.

**Costs**

- every update needs synthesis, placement, routing, and a partial bitstream;
- region boundaries and capacity are fixed;
- partial bitstream bytes can greatly exceed semantic truth-table patch bytes;
- update downtime, tool versions, and security/version management become part of the system.

Report both:

1. semantic model delta in truth-table bits;
2. physical partial bitstream bytes and measured reconfiguration time.

Do not equate them.

### 7.4 Full reconfiguration

Generate and load a complete new bitstream.

- Simplest hardware baseline.
- Usually provides the best optimized implementation.
- Has the largest payload and interruption.

### 7.5 True on-device learning

On-device learning is a separate, harder claim. A hard DLGN forward path is cheap, but backpropagation still requires:

- real-valued or low-precision gate proxies;
- gradient/error propagation;
- replay storage;
- optimizer state.

The feasible first hardware system is **server-trained, edge-applied continual updates**. A second paper can update only a small suffix on the SoC processor using quantized gradients. It should compare against on-device BNN continual-learning methods rather than imply that Boolean inference makes training free.

## 8. Main novelty claim

The strongest defensible claim is the combination of:

1. a DLGN-specific search over immutable/plastic circuit boundaries;
2. replay directly at a one-bit hard-logic boundary;
3. a truth-table-space retention/edit objective;
4. an exact sparse patch representation and update-side accounting;
5. optional co-design of the selected plastic boundary with one FPGA update mechanism.

### Differentiation table

| Prior method/family | What it already provides | RePatch-DLGN differentiator |
|---|---|---|
| EWC / LwF / DER++ / iCaRL | Generic retention, distillation, and replay | Exact Boolean edit cost and circuit partitioning |
| Fully-binary CIL | CIL training for binarized CNNs | DLGN truth tables and connection graph are the learned model, not binary MAC weights |
| Binary latent replay | Memory-efficient replay and frozen feature extractors | Joint cut selection plus sparse deployed LUT patches |
| Generative Binary Memory | Pseudo-replay in binary embedding space | Optimizes the update region and changed circuit bits; GBM can be a replay backend |
| Piggyback/masks | Task-specific binary parameter isolation | Changes DLGN gates/routes and measures physical patch mechanisms |
| DLGN connection learning | Learns routes during initial training | Sequential updates, forgetting, replay, and field-deployment cost |
| PatchLogic/DriftPatch idea | Sparse truth-table updates | Adds the plastic-boundary architecture, latent replay, class expansion, and FPGA mapping |
| BitLogic head | Swappable GroupSum or quantized output head | Generalizes the interface for growing classes and studies how far head-only updates can go |
| FPGA partial reconfiguration | Replaces hardware regions | Learning method determines the smallest useful reconfigurable region and patch content |

## 9. Experimental plan

### 9.1 Benchmarks

**Class incremental**

- CIFAR-100 50 + 5 x 10, matching the fully-binary replay literature.
- CIFAR-100 10 x 10 as a harder longer sequence.
- CORe50 only after CIFAR-100 works; its object/session streams are relevant but add pipeline cost.

**Domain incremental**

- CIFAR-10-C with an explicit sequence of corruptions and severities.
- One subject/session dataset such as UCI HAR, PAMAP2, or ECG if application evidence is needed.

Do not use only Split MNIST. It is useful for debugging but too easy to support a modern CL claim.

### 9.2 DLGN backbones

- Original two-input DLGN for a simple reproducible baseline.
- Light/IWP or hard-Gumbel DLGN for the principal method.
- At least three depths so the boundary study is meaningful.
- One fixed-connection and one bounded learned-connection model if runtime permits.

The complete BitLogic framework is not present locally. Reproduce only the relevant evidence/quantized head rather than making framework integration a prerequisite.

### 9.3 Required baselines

**Training baselines**

- offline joint training with all data, as an upper bound;
- naive sequential fine-tuning;
- no update;
- EWC;
- Learning without Forgetting;
- experience replay;
- DER++;
- iCaRL or nearest-class-mean replay for CIL;
- fully-binary native/latent replay at matched memory where code is available;
- Generative Binary Memory where reproducible.

**DLGN update baselines**

- head only;
- final logic layer plus head;
- complete DLGN;
- random top-$K$ gate patch;
- sensitivity-selected top-$K$ without Hamming/edit regularization;
- threshold-only and route-only adaptation for domain drift;
- progressive expansion at matched total model bits.

**Deployment baselines**

- full hardened-model replacement;
- compressed full checkpoint;
- semantic sparse patch;
- full FPGA reconfiguration;
- partial reconfiguration or programmable overlay, if hardware is included.

### 9.4 Metrics

**Continual-learning quality**

- final average accuracy;
- average incremental accuracy over the stream;
- average forgetting;
- backward transfer;
- old-class and new-class accuracy;
- worst-task accuracy;
- class-order sensitivity.

**DLGN behavior**

- hardened accuracy after every update;
- soft-to-hard gap after every update;
- changed gates and changed truth-table bits;
- changed routes and thresholds;
- layer distribution and stability of changed gates;
- source-task regression before and after head calibration.

**Resource and update cost**

- replay memory in bits;
- training-side model and optimizer memory;
- semantic patch bytes including indices and metadata;
- compressed full-model bytes;
- server/GPU update time and energy proxy;
- edge application time, downtime, and rollback storage;
- number of examples and labels needed per update.

**FPGA metrics**

- LUT, FF, BRAM/URAM, and DSP utilization;
- maximum frequency, latency, throughput, and energy;
- programmable-island overhead versus hardwired suffix;
- partial/full bitstream bytes;
- measured update time and unavailable-service time.

All comparisons should use at least two matched constraints:

1. equal replay-plus-model memory;
2. equal update payload or equal number of changed LUTs.

### 9.5 Essential ablations

- plastic boundary at every layer;
- patch budgets $K$;
- native versus latent replay at equal bits;
- label-only versus stored-logit replay;
- hard-circuit reconstruction versus retained soft checkpoint;
- head-only, suffix-only, sparse-anywhere, and whole-network updates;
- original GroupSum versus BitLogic-style head versus class-independent evidence bus;
- truth-table Hamming cost versus proxy-logit $L_2$;
- fixed versus learned connections;
- source class order and domain order;
- semantic patch versus compressed full checkpoint.

## 10. Feasibility plan

### Phase 1: software proof, approximately two weeks

1. Add checkpoint-to-hard-table and hard-table-to-soft-proxy conversion.
2. Implement one class-incremental CIFAR-100 stream.
3. Implement head-only, last-layer, and full-update replay.
4. Serialize exact gate patches.
5. Sweep three plastic boundaries with three replay budgets.

### Phase 2: proposed method, approximately two weeks

1. Add top-$K$ truth-table edit masks.
2. Add DER-style stored logits.
3. Add the class-independent evidence bus.
4. Produce the accuracy--forgetting--replay--patch Pareto surface.

### Phase 3: second stream and robustness, one to two weeks

1. Add CIFAR-10-C or a subject/session stream.
2. Run central comparisons over at least five class orders/seeds.
3. Test hard-only warm starts and patch compression.

### Phase 4: optional FPGA evidence

Choose exactly one:

- a small BRAM/distributed-memory programmable suffix;
- one partial-reconfiguration boundary;
- or, if tools are unavailable, measured compiled-CPU patch application.

Do not block the learning paper on a full custom on-device training accelerator.

## 11. Risks and kill criteria

### Risk: the frozen prefix is not transferable

**Kill criterion:** no intermediate cut retains a useful Pareto advantage over full replay; suffix methods lose more than 3 accuracy points while saving less than 4x update/replay memory.

**Response:** use transfer-aware source pretraining or progressive growth; do not force head-only adaptation.

### Risk: good adaptation changes most truth tables

**Kill criterion:** reaching within 1 point of full replay consistently requires changing more than 25% of updateable LUTs.

**Response:** the contribution may become boundary-selected latent replay rather than sparse patching, or the idea should stop.

### Risk: patch metadata dominates

For two-input gates, a four-bit truth table is smaller than a wide gate index. Sparse patches can therefore be inefficient when changes are scattered.

**Response:** compare sorted delta indices, block patches, contiguous suffix replacement, and compressed full-tail images. Report the winning representation honestly.

### Risk: the evidence head hides a large arithmetic cost

**Response:** report GroupSum and quantized-head cost separately; compare a pure-LUT head; do not describe a DSP-backed head as logic-only.

### Risk: the FPGA overlay loses the DLGN speed advantage

**Kill criterion:** programmable suffix overhead exceeds 2x the hardwired suffix area or lowers frequency by more than 30% without a substantial update benefit.

**Response:** use partial reconfiguration or keep FPGA as future work.

### Risk: novelty collapses to binary latent replay

**Response:** the paper must retain the exact patch objective, boundary optimization, hard-only warm start, and deployment accounting. If those do not improve a matched-memory FBNN-style replay baseline, the project is not ready.

## 12. Candidate paper

### Recommended title

> **RePatch-DLGN: Boundary-Selected Continual Learning with Binary Replay and Sparse Logic Updates**

### Three central claims

1. Selecting a small plastic DLGN suffix gives a better stability--plasticity--memory frontier than head-only or whole-network updates.
2. Truth-table-aware training converts continual adaptation into compact, exact, rollback-safe circuit patches.
3. The selected boundary maps naturally to a programmable software/FPGA region, reducing update payload and mutable hardware without sacrificing the hard-network inference model.

### Minimum publishable result

- two continual-learning streams;
- a matched-memory comparison with strong replay baselines;
- a consistent intermediate-boundary Pareto advantage;
- at least 4x smaller deployed updates than compressed full-model replacement at comparable final accuracy;
- hardened results after every task;
- one real patch-application measurement.

## 13. Primary references

### Continual learning

- [Overcoming Catastrophic Forgetting in Neural Networks (EWC)](https://doi.org/10.1073/PNAS.1611835114)
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)
- [Dark Experience for General Continual Learning](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Piggyback: Adapting a Single Network with Binary Masks](https://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html)

### Binary and edge continual learning

- [On Class-Incremental Learning for Fully Binarized Convolutional Neural Networks](https://ieeexplore.ieee.org/document/10558661/)
- [Enabling On-device Continual Learning with Binary Neural Networks](https://arxiv.org/abs/2401.09916)
- [Towards Experience Replay for Class-Incremental Learning in Fully-Binary Networks](https://arxiv.org/abs/2503.07107)
- [Generative Binary Memory: Pseudo-Replay Class-Incremental Learning on Binarized Embeddings](https://doi.org/10.1016/j.neunet.2026.108884)
- [Enabling Binary Neural Network Training on the Edge](https://arxiv.org/abs/2102.04270)

### DLGNs and LUT networks

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Mind the Gap](https://openreview.net/forum?id=chYXaetMmz)
- [Light Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250)
- [BitLogic](https://openreview.net/forum?id=ZbsSZAfDod)
- [Fully Trainable Deep DLGNs and LUTNs](https://arxiv.org/abs/2607.09399)
- [Two-Stage Unit Tying for Simplifying DLGNs](../notes/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.md)
- [Local BitLogic PDF](../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf)

### FPGA reconfiguration

- [AMD Dynamic Function eXchange User Guide](https://docs.amd.com/r/en-US/ug909-vivado-partial-reconfiguration/Introduction)
- [Intel/Altera Partial Reconfiguration](https://www.intel.com/content/www/us/en/docs/programmable/813762/25-1/partial-reconfiguration.html)

## Bottom line

Do not begin by implementing continual learning across the whole DLGN. First implement **head-only, last-block, and full-model replay** and measure the boundary trade-off. If the final block recovers most of full-update accuracy, develop RePatch-DLGN around that result. If it does not, test sparse-anywhere patches and progressive growth.

For FPGA deployment, the best first architecture is an immutable hardwired prefix plus a small dual-bank programmable suffix. Partial reconfiguration is a valuable comparison, not the learning novelty. The complete method should be judged by four quantities together:

$$
\text{hard accuracy},\quad
\text{forgetting},\quad
\text{replay bits},\quad
\text{deployed update bits}.
$$
