# Research Directions for Differentiable Logic Gate Networks

**Prepared:** July 21, 2026  
**Target:** Design, Automation and Test in Europe (DATE) 2027  
**Scope:** accuracy, parameter efficiency, edge deployment, new architectures, and application-driven DLGNs

## Executive recommendation

The safest DATE-sized project is **Idea 1: CoverageDLGN**, a parameter-free, structured connectivity scheme that gives every output an intentionally broad receptive field without learning millions of routing scores. It directly addresses the largest remaining architectural weakness shared by the original DLGN, DLGN-Light, and several recent variants: fixed random connectivity can waste gates and leave inputs poorly represented, while fully learned routing is expensive.

The strongest alternatives are:

1. **PatchLogic** if the desired contribution is edge adaptation and compressed model updates.
2. **SketchDistill-DLGN** if the priority is maximum hard-network accuracy at a fixed gate budget.
3. **PersonalDLGN** if access to subject-dependent sensing data makes an application paper preferable.

The seven proposals below are deliberately separated. Combining two complete ideas would probably be too much for the approximately eight weeks before the DATE paper deadline. A sensible strategy is to run the short pilot for Ideas 1--3, select one by a predetermined kill criterion, and develop only that idea into a full paper.

### Ranked shortlist

| Rank | Working title | Main contribution | Novelty confidence | Eight-week feasibility | EDA dependency |
|---:|---|---|---|---|---|
| 1 | CoverageDLGN | Deterministic routing with coverage guarantees and no learned router | High | High | None |
| 2 | PatchLogic | Continual adaptation represented as sparse truth-table patches | High | High | None |
| 3 | SketchDistill-DLGN | Logic-specific intermediate distillation into attainable Boolean sketches | Medium-high | High | None |
| 4 | PersonalDLGN | Few-shot personalization by updating only input thresholds | Medium-high | High | None |
| 5 | CertExit-DLGN | Exact, decision-preserving early exit from GroupSum evaluation | High | Medium-high | None |
| 6 | PaletteLUT | A shared dictionary of reusable truth tables | Medium | Medium | None for the core claim |
| 7 | TrajectoryGap-DLCA | Train against rollout-level soft-to-hard divergence in logic CAs | Medium-high | Medium | None |

## Novelty boundary established from the notes

Several tempting topics are already occupied and should not be presented as the main novelty:

| Tempting claim | Why it is no longer sufficient | Consequence for this project |
|---|---|---|
| "Learn DLGN connections" | Connection optimization, LILogicNet, and the July 2026 fully trainable DLGN/LUTN all optimize routing. | Idea 1 must contribute a **router-free coverage construction**, not merely another score function. |
| "Prune redundant gates" | eXpLogic prunes by saliency; exact/similarity pruning has been studied; two-stage unit tying reports substantial LUT reductions. | Do not build a paper around generic magnitude, saliency, or constant-gate pruning. |
| "Learn the LUT fan-in" | Logic Shrinkage already learns adaptive LUT input sets and arity. | Idea 6 shares complete functions instead of pruning LUT inputs. |
| "Use hardware-local connectivity" | FPGN co-designs LUT6 topology, streaming execution, compilation, and latency-oriented exploration. | Idea 1 needs global dependency coverage and topology analysis, not only locality. |
| "Use fewer gate parameters" | DLGN-Light/IWP, CovJac parameterization, WARP, and ternary polynomial surrogates already attack gate parameterization. | Parameter reduction must occur at a different level, such as topology or a shared function dictionary. |
| "Make DLGNs robust to bit flips" | A March 2026 study analyzes LUT-network resilience under parameter bit flips. | Fault-aware training is viable only with a clearly new selective-redundancy or certification contribution. |
| "Apply DLGNs to edge sensing" | ECG, EEG, acoustic sensing, packet classification, image compression, and video-copy detection now exist. | An application paper needs a new systems constraint such as few-shot personalization or tiny update payloads. |
| "Replace an MLP with LUTs" | Edge Transformer already inserts a differentiable LUT recall block into a transformer. | A new use case needs more than component substitution. |

This boundary is important because the DLGN literature is moving unusually quickly. The novelty statements below mean "not found in the repository notes or in a targeted primary-source search completed on July 21, 2026," not a substitute for a formal IEEE/ACM/Scopus review before submission.

---

## Idea 1 -- CoverageDLGN: Parameter-Free Routing with Global Coverage Guarantees

### Research question

Can fixed DLGN wiring be designed so that it retains the zero-routing-parameter advantage of random connectivity while providing measurably better input coverage, gate utilization, and hard-network accuracy?

### Motivation

The original DLGN and DLGN-Light use fixed random predecessor pairs. Random wiring is cheap, but it can repeatedly merge similar ancestry, leave some inputs underrepresented, and create dead or low-impact subgraphs. Fully trainable routing can correct this, but it introduces candidate scores or dense connection tensors, extra memory, and a difficult discrete optimization problem. This creates a useful unoccupied middle ground: **design the graph, but do not learn the graph**.

### Proposed method

Construct each layer from a deterministic mixture of three edge classes:

1. **Local edges** preserve nearby feature interactions and bounded index distance.
2. **Butterfly/expander edges** rapidly enlarge the receptive field in $O(\log d)$ layers.
3. **Coverage-balancing edges** preferentially merge nodes with complementary input ancestry.

Let $A_i^{(l)}$ be a compact bitset or sketch of input features that can influence node $i$ at layer $l$. For each new two-input gate, choose a pair from a small deterministic candidate schedule by

```math
(i,j)^* = \arg\max_{(i,j)\in\mathcal{C}_l}
\left[
|A_i^{(l-1)}\cup A_j^{(l-1)}|
- \lambda_o |A_i^{(l-1)}\cap A_j^{(l-1)}|
- \lambda_f F(i,j)
- \lambda_d D(i,j)
\right],
```

where $F$ penalizes excessive fan-out and $D$ is an optional locality/wire-length proxy. The construction runs once before training. The final model stores only a seed and topology parameters, or an explicit integer edge list; it has no differentiable router.

Useful variants are:

- `butterfly`: pure algebraic wiring, requiring no construction-time search;
- `coverage`: greedy ancestry balancing;
- `hybrid`: mostly butterfly edges plus a small fraction of local and coverage-balanced edges;
- `budgeted-hybrid`: adds maximum fan-out and connection-span constraints.

### Clear differentiator

- **Versus random DLGN/DLGN-Light:** explicit coverage, overlap, and fan-out objectives replace chance.
- **Versus LILogicNet and optimized-connections DLGNs:** no trainable routing logits, top-$K$ candidates, or router optimization.
- **Versus fully trainable DLGN/LUTN:** the graph is generated analytically and does not add routing parameters or training memory.
- **Versus FPGN:** the contribution is a global receptive-field guarantee and a zero-router architecture; hardware locality is only an optional constraint.
- **Versus OSLGN-like unique routing:** uniqueness alone does not maximize complementary ancestry or control global feature coverage.

The paper should not claim that expanders are new. The novelty is their DLGN-specific construction, coverage analysis, and demonstration that structured fixed wiring closes part of the accuracy gap to learned routing without its parameter cost.

### Core hypotheses

1. At equal gate count, hybrid coverage wiring improves hard accuracy over random wiring.
2. It reduces unused gates, repeated ancestry, and seed variance.
3. It approaches learned-routing accuracy with much lower peak training memory and wall-clock time.
4. The gain is largest in narrow or deep networks, where random coverage failures are most severe.

### Minimum experimental program

**Datasets**

- MNIST and Fashion-MNIST for fast architecture sweeps.
- CIFAR-10 for the primary accuracy result.
- One tabular dataset with heterogeneous features, such as Adult or Higgs, to test whether the effect is image-specific.

**Architectures**

- Original 16-gate mixture and DLGN-Light/IWP, to show that topology gains are not tied to one gate relaxation.
- Widths chosen to produce small, medium, and large gate budgets.
- At least three depths, including a narrow/deep setting designed to expose coverage failure.

**Baselines**

- Original random connections.
- Random connections with enforced uniqueness and matched fan-out.
- Pure local/cyclic wiring.
- Butterfly-only wiring.
- LILogicNet top-$K$ routing, if the released implementation can be integrated.
- Fully learned or candidate-based connection optimization on the smaller configurations.
- A parameter-matched MLP and BNN as context, not as the main architectural ablation.

**Metrics**

- Soft and hardened test accuracy; soft-to-hard gap.
- Gate count and total trainable scalar count.
- Peak accelerator memory, training time per epoch, and convergence epochs.
- Mean and minimum input coverage at each layer.
- Ancestry overlap/Jaccard similarity and number of unique influencing inputs.
- Gate utilization, constant/unused gate fraction, and output-reachable gate fraction.
- Maximum/mean fan-out and normalized connection-span proxy.
- Mean and standard deviation over at least five wiring/training seeds.

### Analysis that would make the paper stronger

Prove a simple receptive-field result for the butterfly variant: under the stated construction and no duplicate input positions, each node can depend on up to $2^l$ distinct inputs after $l$ layers, reaching all $d$ inputs after $\lceil\log_2 d\rceil$ stages. Then distinguish this structural reachability from functional influence, which is measured after training.

A useful visualization is an accuracy-versus-coverage Pareto plot. If accuracy correlates with early-layer ancestry diversity across seeds and architectures, that is a publishable architectural insight even before considering hardware.

### Feasibility and implementation

This is a localized change to connection-index generation in `difflogic` and `difflogic-light-master`. Bitset ancestry can be generated on CPU before training. No synthesis, place-and-route, or commercial EDA tool is required. An optional post-deadline analysis could use Yosys/ABC, but no core claim should depend on it.

**Estimated effort:** 2--3 days for topology generation and metrics, 1 week for pilots, 2--3 weeks for the full matrix, leaving time for analysis and writing.

### Risks and kill criteria

- **Risk:** gains disappear once random wiring is averaged over enough seeds.  
  **Kill criterion:** less than 0.3 percentage-point mean gain on both Fashion-MNIST and CIFAR-10 at small/medium budgets.
- **Risk:** greedy construction is too slow for very wide models.  
  **Mitigation:** fixed candidate schedules and ancestry sketches instead of all-pairs selection.
- **Risk:** reviewers see it as only a graph heuristic.  
  **Mitigation:** include the coverage bound, topology metrics, routing-memory comparison, and cross-parameterization results.

### DATE fit

This is an architecture/design-methodology paper with a clear accuracy--memory--connectivity trade-off. It can target embedded ML, design methodologies, or reconfigurable/logic architecture topics without relying on external EDA flows.

---

## Idea 2 -- PatchLogic: Continual DLGN Adaptation as Sparse Truth-Table Updates

### Research question

Can an already deployed hard DLGN adapt to concept or domain drift by transmitting and applying only a small set of gate truth-table edits rather than a complete new model?

### Motivation

DLGNs have an unusual deployment representation: a model is a graph plus a small truth table per gate. This makes model changes exactly countable in bits. Existing continual-learning methods minimize forgetting, but they are normally evaluated by parameter count rather than by the size of the actual binary update delivered to an edge device. Existing DLGN compression work mostly compresses a static network.

### Proposed method

Start from a hardened network with truth tables $T^0_g$. During adaptation, warm-start a soft representation at each old truth table and optimize

```math
\mathcal{L} =
\mathcal{L}_{\text{new}}
+ \lambda_r\mathcal{L}_{\text{replay/distill}}
+ \lambda_p\sum_g \pi_g d_H(T_g,T^0_g)
+ \lambda_s\sum_g \pi_g,
```

where $\pi_g$ is a learned or sensitivity-derived probability that gate $g$ is editable. A hard top-$K$ budget restricts adaptation to at most $K$ gates. The deployed patch contains:

```math
\{(\text{gate ID},\;\text{new truth-table bits})\}_{g\in\mathcal{P}},
```

plus optional input-threshold deltas. The base model remains unchanged and can be restored without retransmission.

Three variants form a clean progression:

1. **Sensitivity patching:** select editable gates using gradient/influence scores.
2. **Joint budgeted patching:** learn gate-selection scores with a top-$K$ straight-through estimator.
3. **Layer-aware patching:** allocate the patch budget by measured layer sensitivity.

### Clear differentiator

- **Versus static pruning/unit tying:** the objective is not a smaller base network but a smaller *update payload* under drift.
- **Versus ordinary fine-tuning:** the result is an exact list of changed truth-table bits, not a dense floating-point checkpoint.
- **Versus EWC/replay/distillation:** those methods control forgetting but do not exploit the hard LUT representation or optimize transmitted patch size.
- **Versus bit-flip resilience:** intentional adaptation under new data is different from robustness to random parameter faults.

### Primary use cases

- Sensor calibration drift or new user/device domains.
- Network-flow classifiers updated for new traffic classes or attacks.
- Image classifiers updated under corruption/domain shifts as a controlled benchmark.

The paper need not claim in-field partial reconfiguration. A software or firmware LUT array that can overwrite selected truth-table words is enough for the central result.

### Minimum experimental program

**Benchmarks**

- Split/permuted MNIST for rapid continual-learning validation.
- CIFAR-10-C or a sequence of selected corruptions for domain drift.
- One edge-relevant chronological or cross-domain dataset: UCI HAR/PAMAP2 users, ECG patients, or network traffic periods.

**Baselines**

- No adaptation.
- Full fine-tuning.
- Full retraining from scratch.
- Replay, learning without forgetting/distillation, and EWC.
- Random-$K$ gate editing.
- Sensitivity-$K$ editing without the Hamming regularizer.
- Sending a compressed full checkpoint using standard lossless compression.
- Threshold-only adaptation from Idea 4 as a particularly small baseline.

**Metrics**

- Current-domain accuracy/macro-F1 and average accuracy across encountered domains.
- Backward transfer and average forgetting.
- Number and fraction of changed gates.
- Number of changed truth-table bits and total patch bytes including gate IDs.
- Patch compression ratio relative to the full hardened model and a compressed checkpoint.
- Adaptation examples, optimization steps, time, and peak memory.
- Clean/base-domain regression after each patch.
- Accuracy--patch-size Pareto frontier for multiple $K$ values.

### Feasibility and implementation

The hard truth tables already exist in DLGN inference. The main additions are warm-start conversion, a gate edit mask, and patch serialization. The experiments can remain in PyTorch/Python and use the existing packed CPU evaluator. No hardware synthesis is necessary.

**Estimated effort:** one week for the patch mechanism, one week for continual-learning baselines, two weeks for experiments.

### Risks and kill criteria

- **Risk:** small patches cannot adapt without severe forgetting.  
  **Kill criterion:** no useful Pareto region below 10% of truth tables changed on two benchmark streams.
- **Risk:** truth-table changes are unstable across seeds.  
  **Mitigation:** report intersection/stability of selected gates and use layer-wise budgets.
- **Risk:** the work looks like generic continual learning.  
  **Mitigation:** make exact patch bits, update serialization, and hard-network behavior primary metrics.

### DATE fit

PatchLogic connects embedded AI, nonvolatile/on-device model storage, reconfiguration, and lifecycle maintenance. It has a clear edge-system metric that conventional ML papers rarely report: bits transmitted per recovered accuracy point.

---

## Idea 3 -- SketchDistill-DLGN: Distillation Through Attainable Boolean Features

### Research question

Can a full-precision teacher improve a hard DLGN more effectively when its intermediate features are translated into Boolean targets that a logic network can actually represent?

### Motivation

Standard knowledge distillation matches final logits or dense hidden vectors. A binary logic network cannot necessarily reproduce those hidden values, so the teacher may supply a target outside the student's representational geometry. DLGN training also suffers from imperfect credit assignment and a soft-to-hard gap. Intermediate **Boolean sketches** can provide local, discrete supervision while respecting the final model's representation.

### Proposed method

For selected teacher layers $z_l^T(x)$, define a fixed random or learned low-cost projection $R_l$ and target bits

```math
b_l^T(x)=\mathbb{1}[R_l z_l^T(x)>\tau_l].
```

Attach temporary probes to groups of student gates and optimize

```math
\mathcal{L}=
\mathcal{L}_{\text{CE}}
+\lambda_{KD}\,\tau^2\operatorname{KL}
\left(p_{\mathrm{teacher}}^{(\tau)}\|p_{\mathrm{student}}^{(\tau)}\right)
+\sum_l \lambda_l\operatorname{BCE}(q_l^S,b_l^T)
+\lambda_m\mathcal{L}_{\text{hard-margin}}.
```

The probes and teacher are discarded after training. The hardened inference graph is exactly the original DLGN architecture and size.

The key design choices are:

- one balanced sketch bit per gate group rather than one per gate;
- teacher layers matched to DLGN receptive-field depth;
- bit thresholds selected to avoid nearly constant targets;
- an optional error-correcting code constraint so nearby classes have deliberately separated sketches;
- hard-forward training or hard-gate consistency during the final epochs.

### Clear differentiator

- **Versus ordinary logit KD:** supervision is injected at logic layers through discrete, attainable targets.
- **Versus FitNets/feature regression:** no requirement to reproduce real-valued teacher features.
- **Versus GroupSum codebook heads:** the sketches supervise intermediate computation and disappear at inference.
- **Versus new gate relaxations:** the method is orthogonal to IWP, WARP, CovJac, and Gumbel hardening and can be tested with more than one.

The contribution must be framed as a DLGN-specific distillation objective, not "KD applied to DLGNs."

### Minimum experimental program

**Datasets and teachers**

- MNIST: small CNN teacher, primarily for ablations.
- CIFAR-10 and CIFAR-100: a standard ResNet teacher.
- Optional tabular dataset with an MLP teacher to test modality independence.

**Students**

- Original DLGN at three gate budgets.
- DLGN-Light/IWP at one or two budgets.
- Optional Mind-the-Gap hard-forward variant if integration is stable.

**Baselines**

- Labels only.
- Label smoothing.
- Logit KD.
- FitNet-style real-valued feature matching through temporary projections.
- Relational/contrastive distillation if implementation time allows.
- A parameter-matched BNN trained with the same teacher.

**Metrics**

- Hardened accuracy and soft-to-hard gap.
- Accuracy versus gate count and gates required to reach a fixed accuracy.
- Training-only parameter/memory overhead and inference overhead, which should be zero.
- Sketch balance, bit prediction accuracy, class separation, and mutual information proxies.
- Convergence speed and seed variance.
- Results with and without final hard-consistency training.

### Feasibility and implementation

Teacher models and KD losses are straightforward. Temporary probe heads can be implemented without modifying compiled inference. No external EDA tools are needed.

**Estimated effort:** 3--4 days for logit KD and sketch probes, 1 week for ablations, 2 weeks for full CIFAR experiments.

### Risks and kill criteria

- **Risk:** all gains come from conventional logit KD.  
  **Kill criterion:** Boolean sketches fail to improve hard accuracy by at least 0.3 points over tuned logit KD on two student budgets.
- **Risk:** fixed projections create poor targets.  
  **Mitigation:** compare random, PCA-like, and balanced learned projections, but keep the final method simple.
- **Risk:** probes increase training cost too much.  
  **Mitigation:** supervise gate groups at only two or three depths.

### DATE fit

The strongest framing is architecture-aware training for compact logic inference: better accuracy with exactly zero additional inference parameters, gates, or latency.

---

## Idea 4 -- PersonalDLGN: Few-Shot Personalization by Threshold Updates Only

### Research question

Can a shared hard logic network adapt to a new person, sensor, or device by updating only its input discretization thresholds?

### Motivation

Learnable thresholds already improve DLGNs, but they are normally trained globally. Edge sensing often suffers from user, electrode, sensor, and device shifts. Updating a full logic graph on a small support set is expensive and prone to overfitting, whereas a vector of threshold offsets may require only tens or hundreds of bytes.

### Proposed method

Train a global model with base thresholds $\Omega_0$. For user/domain $u$, define

```math
\Omega_u=\Omega_0+B\alpha_u,
```

where $B$ is an optional low-rank threshold basis learned across training domains and $\alpha_u$ is a very small adaptation vector. The gate graph and truth tables are frozen. Compare three levels of complexity:

1. independent per-feature threshold offsets;
2. group-wise offsets, such as one per sensor/channel;
3. low-rank offsets $B\alpha_u$.

Meta-training can optimize the shared model for post-adaptation performance, but a simpler train-then-calibrate method should be implemented first because it is more feasible and easier to interpret.

### Clear differentiator

- **Versus global learned thresholds:** thresholds are explicitly treated as the only user-specific state.
- **Versus the existing ECG/EEG DLGN applications:** those evaluate global models across subjects; this proposal studies support-set adaptation and storage per user.
- **Versus full fine-tuning/adapters:** the logic circuit remains bit-identical and only the input quantizer changes.
- **Versus WARP threshold learning:** WARP learns model thresholds during training, not tiny domain-specific calibration records for deployment.

### Minimum experimental program

**Primary datasets**

- MIT-BIH inter-patient ECG using the established DS1/DS2 split.
- UCI HAR or PAMAP2 with leave-one-subject-out evaluation.
- Optional EEG dataset only if preprocessing is readily available; the July 2026 EEG DLGN paper makes this a crowded application.

**Protocol**

- Hold out complete users/patients during global training.
- Adapt with 0, 5, 10, 25, 50, and 100 labeled examples per class or per user.
- Repeat support sampling and report confidence intervals.

**Baselines**

- No adaptation.
- Batch-statistic or feature normalization calibration only.
- Full DLGN fine-tuning.
- Output-head-only fine-tuning.
- Threshold-only adaptation without a learned basis.
- Small MLP/LoRA-style adapter at a matched byte budget.
- Nearest-centroid/prototype calibration for a non-neural low-data baseline.

**Metrics**

- Macro-F1, balanced accuracy, sensitivity, specificity, and AUROC where appropriate.
- Per-user worst case and standard deviation, not only the pooled mean.
- Adaptation bytes, labeled support examples, steps, time, and energy proxy.
- Base-domain regression and overfitting gap.
- Accuracy/F1 versus personalization bytes.

### Feasibility and implementation

This project changes only input binarization and the evaluation protocol. It can reuse the ECG pipeline described in the notes or use HAR for a simpler first pilot. No EDA tool is needed.

**Estimated effort:** 3 days for threshold adaptation, 1 week for a subject-wise pipeline, 2 weeks for experiments and statistical analysis.

### Risks and kill criteria

- **Risk:** threshold shifts are insufficient for label or temporal domain shifts.  
  **Kill criterion:** threshold adaptation does not recover at least 30% of the gap between no adaptation and full fine-tuning on both datasets.
- **Risk:** feature-engineered ECG dominates the contribution.  
  **Mitigation:** include raw/windowed HAR and make the personalization mechanism the paper's central result.
- **Risk:** low-rank meta-training expands scope.  
  **Mitigation:** keep it as an ablation; the independent/group-wise method is the minimum paper.

### DATE fit

The edge argument is concrete: a single shared immutable logic model plus a tiny per-user calibration vector, with direct storage and adaptation-cost measurements.

---

## Idea 5 -- CertExit-DLGN: Decision-Preserving Early Exit from GroupSum

### Research question

Can sequential software inference stop evaluating parts of a DLGN once the final class is mathematically guaranteed not to change?

### Motivation

DLGN classifiers often allocate many output gates to each class and sum their Boolean votes. On a CPU or microcontroller these votes need not all be computed before the winner becomes inevitable. Unlike confidence-based neural early exits, GroupSum permits a simple exact certificate based on remaining possible votes.

### Proposed method

Partition the output-reachable graph into ordered blocks. After evaluating block $k$, let $S_c^{(k)}$ be the accumulated score for class $c$ and $R_c^{(k)}$ an upper bound on its remaining score. Stop with winner $w$ when

```math
S_w^{(k)} > \max_{c\neq w}\left(S_c^{(k)}+R_c^{(k)}\right).
```

For binary unit votes, $R_c^{(k)}$ is simply the number of unevaluated votes assigned to class $c$. Tighter bounds can account for already computed shared ancestors or gates whose outputs have become logically forced.

Training can encourage early certificates by:

- ordering output groups by class discrimination measured on training data;
- adding auxiliary losses to earlier vote blocks;
- penalizing the expected certified stopping block;
- clustering blocks to maximize ancestor reuse in the packed evaluator.

### Clear differentiator

- **Versus ordinary early-exit networks:** the exit is exact relative to full-model prediction, not confidence based.
- **Versus a smaller DLGN:** hard examples can still use the complete network; easy examples spend fewer gate evaluations.
- **Versus GroupSum alternatives:** the output semantics stay unchanged.
- **Versus FPGA parallel execution:** this targets sequential packed CPU/MCU inference, where skipped blocks translate into real work reduction.

### Minimum experimental program

**Datasets**

- MNIST/Fashion-MNIST for method development.
- CIFAR-10 and one many-class task such as CIFAR-100 to test scaling.
- Optional tabular task for highly branchable inputs.

**Baselines**

- Full packed DLGN evaluation.
- Smaller DLGNs matched to the average number of evaluated gates.
- Static output-block truncation.
- Entropy/margin early exit with tuned allowable disagreement rates.
- An auxiliary-head early-exit MLP/BNN for context.

**Metrics**

- Exact agreement with full DLGN prediction; the certified method must achieve 100%.
- Average, median, and p95 evaluated gates/blocks.
- Measured single-thread CPU latency and throughput with warm caches.
- Energy proxy from retired instructions or CPU package energy if available.
- Accuracy remains the full model's accuracy; also report heuristic baseline disagreement.
- Storage and indexing overhead.
- Exit-depth distribution by class and correct/incorrect prediction.

### Feasibility and implementation

Start with output votes that have disjoint final gate groups and a conservative remaining-vote bound. Implement the scheduler in the packed CPU path. Avoid promising FPGA latency reductions, since a fully parallel design gains little from dynamic exit and may pay control overhead.

**Estimated effort:** one week for a Python simulator, one week for packed CPU execution, two weeks for ordering/training ablations.

### Risks and kill criteria

- **Risk:** most samples require nearly all blocks.  
  **Kill criterion:** less than 25% mean gate-evaluation reduction on both MNIST and CIFAR-10 after ordering.
- **Risk:** shared ancestors make blocks hard to skip.  
  **Mitigation:** construct output groups with explicit ancestor ownership or account for shared computed gates.
- **Risk:** GroupSum accumulation is not the main runtime bottleneck.  
  **Mitigation:** measure skipped *output-reachable subgraphs*, not only skipped integer additions.

### DATE fit

The paper would contribute an exact dynamic-computation mechanism for logic inference and an implementation-level latency study without requiring synthesis tools.

---

## Idea 6 -- PaletteLUT: Shared Truth-Table Dictionaries for Parameter-Efficient DLGNs

### Research question

Do trained multi-input DLGNs need a unique truth table at every node, or can many nodes select from a compact learned palette of reusable logic functions?

### Motivation

WARP and LUT-based networks make higher fan-in practical, but a $k$-input LUT still contains $2^k$ truth-table entries per node. Existing work reduces the parameterization of each gate or removes LUT inputs. A different compression axis is **function reuse across nodes**.

### Proposed method

Learn a dictionary $\mathcal{D}=\{D_1,\dots,D_M\}$ of $k$-input truth tables and a categorical assignment $a_g$ for each gate:

```math
T_g = \sum_{m=1}^{M} a_{gm}D_m,
\qquad
a_g\in\{e_1,\dots,e_M\}.
```

During training, use Gumbel-softmax or straight-through assignments and a WARP/IWP-style relaxation for dictionary entries. At deployment, storage is approximately

```math
M2^k + N\lceil\log_2 M\rceil
```

bits instead of $N2^k$ bits for $N$ independent LUTs, before connection indices. Optional layer-specific dictionaries trade compression for accuracy.

Important ablations are:

- global versus per-layer palettes;
- fixed Boolean basis versus learned dictionaries;
- shared dictionaries with residual per-gate corrections;
- assignment entropy and dictionary diversity regularization;
- post-training clustering versus jointly learned palettes.

### Clear differentiator

- **Versus DLGN-Light/CovJac:** those reduce coordinates within each two-input gate; PaletteLUT shares complete functions across gates.
- **Versus WARP:** WARP efficiently parameterizes each multi-input LUT but does not require a compact shared function vocabulary.
- **Versus Logic Shrinkage:** no input is removed; gates reuse truth tables.
- **Versus unit tying:** nodes retain separate connections and activations while sharing only the selected logic function.
- **Versus convolutional weight sharing:** sharing is global or layer-wise across arbitrary graph nodes, not tied to spatial kernel positions.

### Minimum experimental program

**Architectures and datasets**

- Begin with $k=4$ LUTs on MNIST/Fashion-MNIST.
- Continue with $k=6$ on CIFAR-10 if the implementation is stable.
- Evaluate several $M$ values from a very small palette to an effectively independent upper bound.

**Baselines**

- Independent LUTs with the same fan-in and gate count.
- Post-training $k$-medoids/Hamming clustering of independent truth tables.
- Fixed libraries of common Boolean functions.
- DLGN-Light at matched training-parameter count.
- A smaller independent-LUT network matched to deployed model bits.
- Unit tying or pruning numbers where a directly comparable setup is reproducible.

**Metrics**

- Hardened accuracy and gap.
- Trainable parameter count and peak training memory.
- Deployed model bits, including assignments and dictionary.
- Number of unique truth tables after hardening.
- Dictionary utilization, assignment entropy, and pairwise Hamming distance.
- Packed CPU latency and cache behavior.
- Accuracy versus stored bits and accuracy versus number of unique functions.

### Feasibility and implementation

The main cost is adding a multi-input LUT baseline because the repository primarily contains two-input DLGN variants. A $k=4$ prototype is still small enough to implement directly. No EDA tool is required for the memory/compression claim. Do not claim FPGA LUT savings merely because table parameters are shared: a spatial FPGA implementation may still instantiate each LUT separately.

**Estimated effort:** one week for a $k=4$ implementation, one week for palette training, two weeks for experiments.

### Risks and kill criteria

- **Risk:** trained truth tables are already highly diverse.  
  **Kill criterion:** fewer than 4x model-bit reduction at less than one accuracy-point loss on both MNIST and Fashion-MNIST.
- **Risk:** assignment logits erase training-memory savings.  
  **Mitigation:** separate training parameter cost from deployed bits and test factorized or top-$K$ assignments.
- **Risk:** reviewers classify it as ordinary vector quantization.  
  **Mitigation:** analyze the learned Boolean vocabulary, exact truth-table storage, and differences from node/unit tying.

### DATE fit

PaletteLUT is strongest as a model-representation and memory-hierarchy paper, not as a circuit-area paper unless optional synthesis evidence is later added.

---

## Idea 7 -- TrajectoryGap-DLCA: Rollout-Stable Hardening for Differentiable Logic Cellular Automata

### Research question

Can logic cellular automata be trained so that their hardened local rule remains close to the soft rule over an entire rollout rather than only for one-step outputs?

### Motivation

In an ordinary feed-forward classifier, a small soft-to-hard error can affect one prediction. In a cellular automaton, a small local-rule mismatch is repeatedly fed back and can grow over tens or hundreds of steps. The Google DiffLogic CA notebook demonstrates attractive learned local rules, but final success can be sensitive to hardware, JAX/CUDA versions, seeds, and hardening behavior. Existing hardening methods primarily optimize gates or layer-wise predictions rather than recurrent trajectory divergence.

### Proposed method

Run coupled soft and hard-forward rollouts from the same initial state and optimize

```math
\mathcal{L}=
\mathcal{L}_{\text{task}}(s_{1:T}^{H})
+\lambda\sum_{t=1}^{T} w_t d(s_t^S,s_t^H)
+\mu\sum_{t=1}^{T-1} d(\Delta s_t^S,\Delta s_t^H),
```

where $d$ can combine bitwise cross-entropy, Hamming distance, and a perceptual/image-space metric. Gradients use a hard-forward/soft-backward estimator or a temperature schedule. Curriculum training gradually increases $T$, and perturbation batches include asynchronous updates, random cell erasure, and state noise.

An additional **influence regularizer** can penalize local rules whose one-bit perturbations cause uncontrolled divergence while retaining enough sensitivity for pattern repair.

### Clear differentiator

- **Versus the original DiffLogic CA:** optimize the hardened rollout explicitly, not only the soft local rule and final task.
- **Versus Mind the Gap:** the target is recurrent trajectory stability under repeated rule application, not only feed-forward hardening.
- **Versus recurrent ternary DLGNs:** this concerns two-dimensional local dynamics, asynchronous updates, and self-repair rather than sequence monitoring.
- **Versus generic neural CA robustness:** the final rule is a discrete logic circuit whose soft/hard trajectory mismatch can be measured exactly.

### Minimum experimental program

**Tasks**

- Game of Life rule recovery as a sanity check.
- Pattern generation from a seed.
- Pattern repair after cell deletion.
- Asynchronous update robustness.

**Baselines**

- Original DiffLogic CA notebook training.
- Original training plus longer rollout curriculum.
- Gate-entropy regularization.
- Hard Gumbel/straight-through training adapted from Mind the Gap.
- IWP/RI if the CA implementation can support it cleanly.
- A small neural cellular automaton for context.

**Metrics**

- Final hardened task loss/accuracy and soft-to-hard gap.
- Trajectory Hamming distance at every step.
- Time to divergence or task failure.
- Success at 1x, 2x, and 4x the training horizon.
- Recovery probability and recovery time after perturbation.
- Robustness to asynchronous update rates.
- Gate count, local-rule bits, and variance over seeds and software environments.

### Reproducibility contribution

Record the exact JAX/JAXLIB/CUDA environment, deterministic flags, GPU type, and all random seeds. Save both the hardened rule and complete trajectory hashes. Cross-run agreement should be reported separately from task success. This turns the observed environment sensitivity into a measured phenomenon rather than an anecdote.

### Feasibility and implementation

The existing `DLCA/diffLogic_CA.ipynb` provides the starting point. Coupled rollouts and trajectory losses are conceptually simple, but long unrolling may be memory intensive. Begin with truncated horizons and checkpointing. No EDA tool is required.

**Estimated effort:** one week to refactor the notebook into reproducible scripts, one week for coupled training, two weeks for robustness experiments.

### Risks and kill criteria

- **Risk:** hard-forward rollouts do not train reliably.  
  **Kill criterion:** no improvement in hard rollout success over temperature/entropy baselines on both pattern generation and repair.
- **Risk:** results are highly task-specific.  
  **Mitigation:** require consistent trends over at least three CA tasks and multiple horizons.
- **Risk:** environment determinism becomes a distraction.  
  **Mitigation:** treat reproducibility as supporting analysis, not the only contribution.

### DATE fit

This is the highest-risk proposal, but it offers a distinctive combination of discrete dynamical systems, robust edge computation, and exact local-rule deployment.

---

## Ideas deliberately not recommended as the primary paper

### Generic gate pruning

The space is crowded by saliency pruning, similarity/equivalence methods, input pruning, and two-stage unit tying. A new pruning score alone is unlikely to be a defensible DATE contribution.

### Generic fault injection

Testing random truth-table bit flips would largely reproduce the emerging LUT-resilience literature. A publishable extension would need selective redundancy, a certified fault budget, or fault-map-aware training, and would benefit from hardware evidence.

### Replacing GroupSum with an arbitrary MLP or tree

GroupSum has already received dedicated scalability and alternative-output analysis. A replacement needs either an exact computational property such as Idea 5 or a demonstrated hardware bottleneck with a convincing implementation.

### Another hardware-aware local topology

FPGN and recent planar/hardware-aware logic networks make this difficult to claim without a full compiler or physical-design contribution. CoverageDLGN intentionally targets the different question of global information mixing with zero learned routing state.

### A new edge application with no architectural constraint

ECG, EEG, acoustic sensing, packet processing, compression, and video matching already show broad applicability. The scientific contribution should be a new constraint--personalization bytes, update bytes, certified work reduction, or similar--rather than merely applying DLGN to another dataset.

## Suggested eight-week decision and execution plan

### Week 1: three pilots

1. Implement `butterfly`, `coverage`, and `hybrid` connection generators; test small MNIST/Fashion-MNIST models.
2. Warm-start hard truth tables and allow top-$K$ gate edits on a two-domain MNIST pilot.
3. Add logit KD and one intermediate Boolean sketch probe to the same small model.

Use the kill criteria already stated. Select exactly one project at the end of the week.

### Weeks 2--3: establish the claim

- Reproduce the strongest baselines at matched gate and parameter budgets.
- Freeze datasets, architecture grids, metrics, and seeds.
- Generate one central Pareto plot that would support the abstract's main claim.

### Weeks 4--5: full experiments

- Run the complete benchmark matrix.
- Add the decisive ablations, not every possible variant.
- Measure actual CPU inference where deployment is part of the claim.

### Week 6: analysis and robustness

- Run five seeds on the central experiments.
- Inspect failure cases and calculate confidence intervals.
- Verify parameter, gate, byte, and runtime accounting from serialized artifacts.

### Weeks 7--8: manuscript

- Write the method and experimental protocol before chasing marginal extra results.
- Complete the related-work search using IEEE Xplore, ACM DL, Scopus, arXiv, OpenReview, and proceedings indexes.
- Keep optional EDA experiments out of the critical path.

The official DATE 2027 call lists paper registration on September 13, 2026 AoE and final submission on September 20, 2026 AoE. The schedule is therefore tight enough that a single clean architectural contribution is preferable to a broad combined system.

## Shared experimental rules

To make any selected idea convincing:

1. Report the **hardened** network as the primary result; soft accuracy is diagnostic only.
2. Match both gate count and trainable parameter count where possible; they are not interchangeable.
3. Include connection indices, thresholds, output state, dictionaries, and metadata in model-size accounting.
4. Report mean, standard deviation, and individual seeds for central comparisons.
5. Separate training cost, deployed storage, inference operations, and measured latency.
6. Use the same data augmentation, optimizer budget, and early-stopping rule across architectural baselines.
7. Publish serialized hard models and scripts that reproduce gate/bit counts.
8. Avoid FPGA area or energy claims based only on gate counts. Use proxy language unless real synthesis is performed.

## Repository material reviewed

The internal synthesis for this document used every Markdown note outside `ideas/`, plus the relevant implementation documentation. Corresponding PDFs were consulted when a note was absent or a precise claim needed checking; in particular, the convolutional DLGN PDF was read because it has no matching note file.

### Core architectures and training

- [Deep Differentiable Logic Gate Networks](../notes/deep_differentiable_logic_gate_networks.md)
- [Mind the Gap](../notes/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.md)
- [Light Differentiable Logic Gate Networks](../notes/light_differentiable_logic_gate_networks.md)
- [Fitting Multilinear Polynomials](../notes/fitting_multilinear_polynomials_for_logic_gate_networks.md)
- [WARP](../notes/warp_logic_neural_networks.md)
- [Connection optimization](../notes/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.md)
- [LILogicNet](../notes/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.md)
- [BitLogic](../notes/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.md)
- [From MNIST to ImageNet](../notes/from_mnist_to_imagenet_understanding_the_scalability_boundaries_of_differentiable_logic_gate_networks.md)
- [Learning interpretable logic](../notes/learning_interpretable_differentiable_logic_networks.md)
- [Convolutional DLGN paper](../pdfs/convolutional_differentiable_logic_gate_networks.pdf)

### Hardware, circuits, and scaling

- [Silicon-aware DLGNs](../notes/silicon_aware_neural_networks.md)
- [Logic neural networks on FPGAs](../notes/logic_neural_networks_for_efficient_fpga_implementation.md)
- [Scalable, interpretable, and verifiable logic CNNs](../notes/a_scalable_interpretable_verifiable_differentiable_logic_gate_convolutional_neural_network_architecture_from_truth_tables.md)
- [DeepGate](../notes/deepgate_learning_neural_representations_of_logic_gates.md)
- [Edge Transformer](../notes/et_an_energy_eficient_edge_transformer_architecture.md)

### Applications and dynamical models

- [Inter-patient ECG classification](../notes/inter_patient_ecg_arrhythmia_classification_with_lgns_and_lutns.md)
- [Lightweight in-network flow classification](../notes/lightweight_in_network_flow_classification_with_deep_differentiable_logic_gate_networks.md)
- [GIC-DLC image compression](../notes/gic_dlc_differentiable_logic_circuits_for_hardware_friendly_grayscale_image_compression.md)
- [Cellular automata](../notes/celluar_automata.md)
- [Recurrent DLGNs](../notes/recurrent_deep_differentiable_logic_gate_networks.md)
- [Polynomial surrogate ternary logic](../notes/polynomial_surrogate_training_for_differentiable_ternary_logic_gate_networks.md)
- [Original implementation notes](../difflogic/notes.md)
- [DLGN-Light implementation notes](../difflogic-light-master/notes.md)

## Selected current primary sources

- [DATE 2027 Call for Papers](https://www.date-conference.com/date-2027-call-papers)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
- [Mind the Gap: Differentiable Logic Networks versus Conventional Neural Networks](https://openreview.net/forum?id=chYXaetMmz)
- [Lightweight Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250)
- [WARP: Walsh-Arity-Reduced Parameterization](https://arxiv.org/abs/2602.03527)
- [Covariance-Jacobian Gate Parameterization](https://arxiv.org/abs/2605.08657)
- [LILogicNet](https://arxiv.org/abs/2511.12340)
- [Fully Trainable Deep Differentiable Logic Gate Networks and LUT Networks](https://arxiv.org/abs/2607.09399)
- [FPGN: A Differentiable Logic Gate Network Framework for Ultra-Low-Latency FPGA Inference](https://arxiv.org/abs/2607.08427)
- [Resource Utilization Optimization of Differentiable Logic Gate Networks on FPGAs](https://arxiv.org/abs/2605.04109)
- [From Arithmetic to Logic: Resilience of LUT Networks Under Parameter Bit-Flips](https://arxiv.org/abs/2603.22770)
- [DLGNs for EEG Classification at the Edge](https://arxiv.org/abs/2607.18149)
- [Recurrent Differentiable Ternary Logic Gate Networks](https://arxiv.org/abs/2605.24649)

## Final selection advice

Choose **CoverageDLGN** unless the first-week pilot fails. It is the best match to the available code, directly responds to the newest routing literature, requires no external EDA flow, and can yield both an empirical result and a small analytical result. If it fails, choose **PatchLogic** for a more application/system-oriented DATE paper. Choose **SketchDistill-DLGN** only if it beats a carefully tuned logit-distillation baseline, because otherwise its novelty will be difficult to defend.
