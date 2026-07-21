# Application-Driven DLGN Research Ideas

**Prepared:** July 20, 2026
**Scope:** Research directions that are not designed around a specific conference and can be evaluated mainly with the DLGN code already in this repository, PyTorch, public datasets, and software inference.

## Executive recommendation

The strongest starting point is **Idea 1, PersonalDLGN**. It addresses a real deployment problem in medical and wearable sensing: a model trained across subjects often loses accuracy on a new person or device. The proposed contribution is not another ECG classifier. It is a shared hardened DLGN whose small, threshold-only adapter can be calibrated from a few samples. This is technically distinct from learning one global set of input thresholds and can be evaluated without a hardware toolchain.

The second recommendation is **Idea 2, DriftPatch-DLGN**. It treats a hardened logic network as updateable edge software and minimizes the number of truth-table bits that must change when the data distribution drifts. Its main outputs are accuracy, forgetting, and update size in bytes. This fits networking, wearables, and deployed sensors without requiring synthesis or power tools.

**Idea 3, TriSense-DLGN**, is the best safety-oriented application. It uses ternary UNKNOWN values for missing or unreliable sensors and evaluates the model as a selective classifier. The important novelty must be a missingness-aware training objective on real multi-sensor data; simply applying the existing ternary DLGN to a sensor dataset would be too weak.

For a completely self-contained project, including synthetic data generation, **Idea 6, SyndromeDLGN**, is attractive. Its main risk is scientific rather than logistical: conventional decoders are strong, so the learned circuit must target short codes under bursty, asymmetric, or otherwise mismatched channels where a fixed textbook decoder is not optimal.

## Ranking

| Rank | Idea | Primary use case | Main contribution | External-tool dependence | Feasibility | Novelty confidence |
|---:|---|---|---|---|---|---|
| 1 | PersonalDLGN | Patient and device personalization | Few-shot threshold adapters around a shared Boolean core | Low | High | High, pending systematic search |
| 2 | DriftPatch-DLGN | Continually updated edge classifiers | Accuracy-preserving updates with few changed gate bits | Low | High | High, pending systematic search |
| 3 | TriSense-DLGN | Classification with failed or missing sensors | Ternary missingness encoding and safe abstention loss | Low | Medium-high | Medium-high |
| 4 | ExitLogic | Variable-latency CPU inference | Exact early-exit certificate for accumulated logic votes | Low | Medium-high | Medium-high |
| 5 | LogicFilter | Blacklists, cache directories, membership queries | Gate-budget-aware learned Bloom filter | Low | Medium-high | Medium |
| 6 | SyndromeDLGN | Short-packet error correction | Channel-specialized Boolean syndrome decoder | Very low | High | Medium |
| 7 | ClauseAD | Rare-event and equipment anomaly detection | One-class normality clauses learned by a DLGN | Low | Medium | Medium-low |
| 8 | EventLogic | Always-on acoustic event or keyword detection | Sparse spectral-change encoding plus recurrent DLGN | Low | Medium | Medium-low |

"External-tool dependence" refers to EDA, FPGA, ASIC, formal-verification, and cycle-accurate hardware tools. Public datasets and ordinary Python packages are still needed for most application studies. Compiled C inference is useful for Ideas 4 and 5, but it is an optional measurement stage rather than a prerequisite for validating the method.

## Literature boundaries

The ideas below were filtered against all notes in this repository and their corresponding papers. The following boundaries are especially important:

- The [original DLGN](https://arxiv.org/abs/2210.08277) already establishes very fast compiled Boolean inference. A new application still needs a DLGN-specific method, not just a different dataset.
- [WARP](https://arxiv.org/abs/2602.03527) and the work summarized in [learning_interpretable_differentiable_logic_networks.md](../notes/learning_interpretable_differentiable_logic_networks.md) already learn input thresholds. PersonalDLGN must therefore contribute subject-specific, few-shot adapters around a shared circuit rather than ordinary global threshold learning.
- [Inter-patient ECG classification](https://arxiv.org/abs/2601.11433) already applies LGNs and LUTNs to MIT-BIH and reports up to 94.28% accuracy. Its model is fixed for unseen patients; it does not study few-shot calibration after deployment.
- [Polynomial Surrogate Training](https://arxiv.org/abs/2603.00302) already introduces ternary UNKNOWN and selective prediction. Its abstention evidence is concentrated on simple synthetic data. A missing-sensor project must add a real use case, an explicit coverage/risk objective, and comparisons with imputation and mask-bit baselines.
- [Recurrent DLGNs](https://arxiv.org/abs/2508.06097) already establish stateful logic networks. "Use a recurrent DLGN on a sequence" is not a sufficient contribution by itself.
- [Lightweight in-network flow classification](https://ieeexplore.ieee.org/document/11323087) already covers static intrusion and flow-size classification. DriftPatch-DLGN is about small continual updates and forgetting, not another static flow classifier.
- Learned Bloom filters are an established field. [Partitioned Learned Bloom Filters](https://arxiv.org/abs/2006.03176) and [Adaptive Learned Bloom Filters](https://arxiv.org/abs/1910.09131) are required baselines for Idea 5.
- Syndrome-based neural decoding also exists. For example, [Improved Syndrome-based Neural Decoder for Linear Block Codes](https://arxiv.org/abs/2402.13948) evaluates BCH and polar codes. Idea 6 needs a circuit-native and channel-specialized claim, not merely a neural decoder with logic neurons.

The direct collision search performed for this memo did not reveal DLGN papers specifically on threshold-only personalization, continual gate-patch learning, certified early exit, learned Bloom filters, or error-correcting decoders. That is evidence of a gap, not proof of novelty. A systematic IEEE Xplore, ACM Digital Library, Scopus, and Google Scholar search is still required before committing to a manuscript claim.

---

## Idea 1: PersonalDLGN - threshold-only personalization

### Use case

A hospital, wearable, or industrial sensor deploys one shared classifier, but each patient, user, electrode placement, or sensor unit has a slightly different signal distribution. Sending or retraining a separate full network per user is undesirable. A small adapter containing only input thresholds can be stored per user while every deployment shares the same hardened logic circuit.

The first applications should be:

- inter-patient ECG arrhythmia classification;
- cross-subject human-activity recognition;
- cross-device sensor calibration if a suitable public dataset is available.

### Research question

Can a DLGN be trained so that a new subject can recover useful accuracy by changing only a few ordered input thresholds, using a small total labeling budget, while its gates and connectivity remain fixed?

### Proposed method

Split the model into a global Boolean core and a small subject adapter:

$$
f_u(x) = f_{\theta^*}\left(B(x;\Omega_0 + \Delta\Omega_u)\right),
$$

where $\theta^*$ contains the frozen gates and connections, $\Omega_0$ contains global thresholds, and $\Delta\Omega_u$ is the only subject-specific state.

1. Train the global thresholds and DLGN normally across training subjects.
2. Harden and freeze the DLGN gates and connectivity.
3. Adapt only $\Delta\Omega_u$ for each new subject using a small support set.
4. Regularize the adapter with $\|\Delta\Omega_u\|_1$ or $\|\Delta\Omega_u\|_2^2$ and preserve threshold ordering through positive increments.
5. As a stronger version, use episodic training: repeatedly hide one training subject and optimize the global circuit so that its thresholds adapt in a few gradient steps.

The first implementation should use ordinary fine-tuning of thresholds. Meta-learning is a second-stage addition only if the simple method demonstrates a clear adaptation signal.

### Clear novelty boundary

WARP learns one global set of thresholds jointly with the network. The interpretable DLGN work learns thresholds and connections for one trained model. The ECG paper evaluates generalization to unseen patients but does not personalize the deployed model. PersonalDLGN combines a shared hardened circuit with tiny, few-shot threshold adapters and measures accuracy gained per stored adapter byte.

The manuscript claim cannot be "learnable thresholds improve ECG." It must be:

> A logic network can be explicitly trained for parameter-efficient post-deployment personalization, with no change to its Boolean core.

### Minimum experiment

- **Datasets:** MIT-BIH with the established inter-patient split and one subject-partitioned activity-recognition dataset such as UCI HAR or PAMAP2.
- **Support sizes:** 0, 10, 25, 50, and 100 total labeled examples for each held-out subject. Do not assume that every ECG subject contains every arrhythmia class; stratify where possible and add a balanced per-class analysis only for subjects that support it.
- **Baselines:** global DLGN; full DLGN fine-tuning; output-head-only adaptation; global learned thresholds; fixed quantile thresholds; a small MLP with a parameter-matched adapter.
- **Metrics:** accuracy, macro-F1, per-class sensitivity, performance versus support size, adapter parameters and bytes, adaptation time, and variance across held-out subjects.
- **Ablations:** threshold count; per-feature versus shared threshold shifts; L1 versus L2 regularization; episodic versus ordinary global pretraining.

Use macro-F1 and class sensitivity as primary ECG metrics. Accuracy alone can hide failure on rare arrhythmia classes.

### Feasibility and failure condition

This requires only a differentiable threshold layer, a subject-aware data split, and freezing selected parameters. It can be built on the local DLGN or Light DLGN implementation. No recurrent model or hardware flow is required.

Stop or reformulate if threshold-only adaptation does not consistently outperform the unadapted global model on held-out subjects, or if full-model fine-tuning is equally stable with fewer than ten examples. In that case, test a slightly larger adapter containing thresholds plus the final class-vote assignment, but report the added storage honestly.

---

## Idea 2: DriftPatch-DLGN - compact continual updates

### Use case

An intrusion detector, wearable, or industrial monitor is deployed for months. Traffic types, sensor offsets, or operating regimes change. Replacing the complete model after every update wastes bandwidth and complicates version management. Since a hardened two-input DLGN stores a small truth table per gate, an update can be represented as a list of changed gate indices and their new truth tables.

### Research question

Can a hardened DLGN adapt to concept drift while limiting both forgetting and the number of changed truth-table bits?

### Proposed method

Warm-start a soft model from the deployed hard circuit. Penalize changes relative to the deployed truth table:

$$
L = L_{new} + \lambda_{old}L_{replay}
    + \lambda_{patch}\sum_n\sum_g \pi_{n,g}
      d_H(T_g,T_{g_n^{old}}),
$$

where $\pi_{n,g}$ is the relaxed probability of gate $g$ at node $n$, $T_g$ is its truth table, and $d_H$ is truth-table Hamming distance. After hardening, serialize only changed entries:

```text
(gate index, new gate id)
```

For larger LUTs, use changed truth-table words rather than the full LUT. Preserve old behavior with a small replay buffer or teacher distillation from the deployed model. Compare both because a replay buffer consumes device memory, while teacher outputs can be generated at the server.

A useful extension is a patch budget: select at most $K$ gate changes using a straight-through top-K mask. Begin with the regularizer because it is simpler and yields a complete Pareto curve over $\lambda_{patch}$.

### Clear novelty boundary

Existing DLGN connectivity and pruning papers optimize a static model. SwitchLGN trains a fixed in-network classifier. DriftPatch-DLGN instead treats the hardened truth tables as the unit of continual adaptation and jointly measures predictive quality, forgetting, and transmitted patch bytes.

The contribution is not generic continual learning with a DLGN backbone. The DLGN-specific contribution is an update cost defined directly on gate truth tables and an exact patch representation for the deployed Boolean circuit.

### Minimum experiment

- **Controlled drift:** rotate or offset selected continuous features before thresholding; change class priors; introduce new Boolean feature interactions.
- **Realistic drift:** chronological or domain splits from one network-security dataset and one subject/session sensor dataset.
- **Baselines:** full retraining; naive fine-tuning; replay; EWC or L2-to-old-weights; knowledge distillation; random K-gate updates.
- **Metrics:** current-task accuracy, average accuracy over all observed domains, backward transfer, forgetting, changed gates, changed truth-table bits, compressed patch bytes, and update time.
- **Ablations:** truth-table Hamming cost versus gate-ID change count; replay size; gate-patch budget; gate updates versus threshold updates.

Report both absolute patch bytes and patch size as a fraction of the full hardened model. A percentage alone can make a small model's update look more important than it is.

### Feasibility and failure condition

All drift can initially be generated from existing datasets, and patch size is computed exactly from the hardened model. No network switch, FPGA, or compiler is required. Bit-packed simulation is sufficient.

The idea fails if restricting gate changes produces the same accuracy-patch curve as simply retraining and compressing a full checkpoint, or if good adaptation consistently requires changing most gates. It remains useful if a small patch preserves old performance substantially better than an unconstrained update at the same new-domain accuracy.

---

## Idea 3: TriSense-DLGN - missing-sensor classification with UNKNOWN

### Use case

Wearables and medical or industrial systems commonly lose an electrode, axis, packet, or entire sensor channel. Zero imputation is dangerous because zero may be a valid measurement. A separate missingness bit tells the model what happened, but a binary DLGN still has to force every internal signal to TRUE or FALSE. Ternary logic provides an explicit UNKNOWN value that can propagate through the circuit and trigger abstention.

### Research question

Can a ternary DLGN reduce confidently wrong predictions under sensor loss while retaining useful coverage and accuracy when all sensors are present?

### Proposed method

Map a continuous feature to FALSE, UNKNOWN, or TRUE with a learned uncertainty interval:

$$
B_j(x)=
\begin{cases}
0 & \text{if feature }j\text{ is missing},\\
-1 & \text{if }x_j < \Omega_j^-,\\
0 & \text{if }\Omega_j^- \le x_j \le \Omega_j^+,\\
+1 & \text{if }x_j > \Omega_j^+.
\end{cases}
$$

Train on paired clean and sensor-masked samples. Optimize classification plus three DLGN-specific terms:

- a target-coverage selective loss, so abstention has an explicit cost;
- a clean/masked consistency loss, so removing evidence should produce the same answer or UNKNOWN rather than a different confident class;
- a wrong-confidence penalty, weighted more heavily than abstention under severe sensor loss.

Use block masks that remove complete sensors or channels, not only independent random values. Independent dropout is too easy and does not reflect electrode or communication failure.

### Clear novelty boundary

PST already supplies ternary gates and shows UNKNOWN-based selective prediction, mainly on synthetic datasets. Recurrent ternary DLGNs have also been applied to Signal Temporal Logic monitoring under sensor degradation. TriSense-DLGN is a feed-forward, multi-class sensor-fusion study with learned UNKNOWN intervals, channel-level missingness augmentation, and an explicit safety-oriented consistency objective.

Without those elements, this is only an application of PST and the novelty is insufficient.

### Minimum experiment

- **Datasets:** one multi-sensor activity dataset such as PAMAP2 or Opportunity and one medical or industrial dataset with meaningful channel groups.
- **Missingness tests:** one missing value, full channel loss, contiguous time-window loss, and a sensor with increasing noise before failure.
- **Baselines:** binary DLGN with zero/mean imputation; binary DLGN with mask bits; vanilla ternary PST; MLP with imputation and abstention; simple ensemble uncertainty.
- **Metrics:** full-coverage accuracy, selective accuracy, coverage, risk-coverage AUC, wrong-confident rate, class sensitivity, and robustness versus number of failed sensors.
- **Ablations:** fixed versus learned UNKNOWN bands; paired consistency loss; value dropout versus block dropout; binary versus ternary gates at equal node counts.

The principal plot should be selective risk versus coverage for each sensor-failure severity. A single accuracy number does not test the proposed behavior.

### Feasibility and failure condition

This requires implementing PST or a small ternary layer if no code accompanies the paper. That is more work than Ideas 1 and 2, but it remains entirely within PyTorch. Start with a shallow network and two or three sensor groups before scaling.

Stop if UNKNOWN density does not correlate with error under realistic block failures, or if a binary DLGN with explicit mask bits dominates the ternary model across the complete risk-coverage curve. That negative comparison is essential because mask bits are the simplest credible alternative.

---

## Idea 4: ExitLogic - certified early exit for compiled DLGNs

### Use case

An always-on CPU or microcontroller must process easy and difficult inputs under variable latency or battery constraints. A smaller fixed DLGN lowers every inference's cost but may reduce accuracy. An anytime DLGN can evaluate a few evidence blocks for easy inputs and continue only when the winner is not yet guaranteed.

This idea targets sequential software inference. It should not claim equivalent savings for a fully parallel combinational accelerator, where all gates switch together unless the architecture includes explicit gating.

### Proposed method

Partition the network into independently evaluable class-vote blocks. After block $k$, class $c$ has accumulated score $S_c^{(k)}$, and its unevaluated blocks contain at most $R_c^{(k)}$ additional positive votes. The current winner $c^*$ is final if

$$
S_{c^*}^{(k)} > \max_{c\ne c^*}\left(S_c^{(k)}+R_c^{(k)}\right).
$$

This certificate guarantees agreement with full-network inference for every unevaluated gate output. It does not certify that the model's class is correct; it certifies that additional computation cannot change the model's decision.

Train all blocks jointly, add auxiliary losses to make early blocks discriminative, and randomize or learn block order. Keep the exact certificate as the primary mode. A heuristic confidence threshold can be reported separately as an accuracy-latency tradeoff.

### Clear novelty boundary

The original DLGN provides fast full-network CPU inference, and class-specific path methods reduce some irrelevant logic. ExitLogic instead defines an accumulated-vote architecture with an exact per-input stopping rule. The exactness guarantee and block-order training are the methodological contribution; generic neural-network early exit is only adjacent prior art.

### Minimum experiment

- **Tasks:** UCI tabular tasks for rapid iteration, then ECG or human-activity recognition and MNIST for a larger model.
- **Baselines:** full compiled DLGN; a smaller DLGN; heuristic margin exit; entropy exit; equal-sized blocks in random order.
- **Metrics:** identical-to-full accuracy, average and p95 gates evaluated, exit-depth distribution, Python bit-simulation operations, compiled-C wall time, and energy proxy based on executed Boolean operations.
- **Ablations:** number of blocks; auxiliary losses; fixed versus learned ordering; exact versus heuristic exit.

Wall-clock CPU speed is required before claiming a deployment benefit. Gate counts alone ignore branch, memory, and function-call overhead.

### Feasibility and failure condition

The certificate and simulator are simple. Compiled C measurement can reuse the original repository's export path after the model behavior is established. No synthesis tool is needed.

The main risk is that the bound is too conservative, so almost all inputs execute every block. Stop if fewer than approximately 20% of test inputs exit early at useful model sizes, or if control-flow overhead removes the simulated gain. In that case, retain the block architecture but treat heuristic early exit as a different, non-exact project.

---

## Idea 5: LogicFilter - a DLGN learned membership filter

### Use case

Edge gateways, storage engines, and network services repeatedly ask whether a key belongs to a set: a malicious URL list, blocked device IDs, cached objects, or known signatures. Bloom filters are compact and have no false negatives for inserted keys, but they spend memory uniformly even when the key distribution has learnable structure.

### Proposed method

1. Encode each key into a fixed binary vector using reproducible hashes or byte-level features.
2. Train a gate-budgeted DLGN to distinguish members from representative non-members.
3. Store every member rejected by the DLGN in a conventional backup Bloom filter, preserving zero false negatives for the represented set.
4. Jointly select the DLGN score threshold, gate budget, and backup-filter size to minimize total bits at a target false-positive rate.

The accounting must include all state:

$$
M_{total}=M_{DLGN}+M_{encoder}+M_{backup}.
$$

Use an asymmetric loss because false negatives increase backup memory while false positives increase unnecessary downstream lookups. A more distinctive version directly fits a differentiable proxy for total memory plus query false-positive cost.

### Clear novelty boundary

Learned Bloom filters already combine a classifier with a backup filter, and partitioned/adaptive variants exploit score ranges. Replacing their classifier with a DLGN is not enough. LogicFilter must co-optimize the Boolean circuit's gate budget, operating threshold, and exact backup size, then demonstrate a useful memory-throughput frontier with bit-packed or compiled inference.

The reviewed literature did not reveal a DLGN learned Bloom filter, but the surrounding learned-filter literature is mature. Novelty confidence is therefore medium until a deeper systems-literature search is complete.

### Minimum experiment

- **Data:** one structured synthetic key distribution and one real membership workload, such as malicious URLs, cache keys, or genomic k-mers.
- **Baselines:** standard and blocked Bloom filters; small MLP learned Bloom filter; Adaptive Learned Bloom Filter; Partitioned Learned Bloom Filter; DLGN without memory-aware training.
- **Budgets:** compare methods at identical total memory, including model and encoder state.
- **Metrics:** false-positive rate with zero false negatives on inserted keys, bits per key, queries per second, p95 latency, build time, and sensitivity to a changed non-member query distribution.
- **Ablations:** raw bytes versus hashed bits; asymmetric versus cross-entropy loss; threshold selection; gate budget; backup-filter allocation.

### Feasibility and failure condition

Bloom-filter logic, hashes, and synthetic workloads are straightforward to implement in Python. C or C++ lookup measurement is optional until the memory result is promising.

The idea fails if the DLGN's circuit and input encoder consume more memory than they save in the backup filter, or if distribution shift causes unacceptable false positives. Classical filters are difficult to beat on unstructured keys, so the project should explicitly characterize which data distributions are learnable.

---

## Idea 6: SyndromeDLGN - channel-specialized short-block decoding

### Use case

Small sensors often transmit short binary packets over links with asymmetric, bursty, or device-specific errors. Algebraic decoders are excellent for the channel assumptions they target, but a tiny learned Boolean decoder may exploit a stable non-uniform error distribution while retaining a hard parity check as a safety mechanism.

### Research question

Can a hardened DLGN map a syndrome and a few quantized reliability bits to a correction pattern that improves block error rate under structured channel noise, without floating-point inference?

### Proposed method

For a received word $r$ and parity-check matrix $H$, compute the syndrome using exact XOR gates:

$$
s=Hr^T \pmod 2.
$$

Feed $s$, and optionally 2 to 4 threshold bits per channel reliability value, to a DLGN that predicts an error pattern or a small candidate list. Apply the correction and check the syndrome again. If parity still fails, abstain, request retransmission, or fall back to the conventional decoder.

Generate unlimited training data from:

- binary symmetric channels;
- Gilbert-Elliott burst channels;
- asymmetric 0-to-1 and 1-to-0 error rates;
- mixtures that shift between training and evaluation.

The most useful variant trains a small global decoder and adapts only thresholds or a gate patch when the channel changes, connecting this idea to PersonalDLGN or DriftPatch-DLGN without requiring both contributions in the first paper.

### Clear novelty boundary

Neural syndrome decoders already exist and can handle larger BCH and polar codes. The differentiator is a completely hardened Boolean correction network for short blocks, direct comparison against a full syndrome lookup table, and explicit specialization to structured channel errors. Do not claim that DLGNs invent neural decoding.

### Minimum experiment

- **Codes:** start with Hamming(15,11), then BCH(31,k) or BCH(63,k).
- **Baselines:** bounded-distance algebraic decoder; syndrome lookup table; small MLP; BNN; an available syndrome neural decoder if compatible.
- **Channels:** independent, burst, and asymmetric errors, plus mismatch between train and test channel parameters.
- **Metrics:** bit error rate, block error rate, undetected-error rate after parity checking, retransmission rate, truth-table bits, Boolean operations, and CPU throughput.
- **Ablations:** syndrome only versus syndrome plus reliability; DLGN size; training mixture; fallback decoder; channel-specific versus universal model.

### Feasibility and failure condition

This is the least dataset-dependent idea. Encoding, channel simulation, and evaluation can all be generated locally, and small codes allow exhaustive analysis of many error patterns.

The principal risk is that a conventional decoder or lookup table is both smaller and better. Continue only if structured channel statistics let the DLGN improve block error rate at a competitive memory budget, or if one shared DLGN covers several channel conditions more compactly than multiple lookup tables.

---

## Idea 7: ClauseAD - one-class anomaly detection with learned logic clauses

### Use case

Equipment monitoring and security systems often have abundant normal data but few representative anomalies. A DLGN could learn a bank of Boolean conditions that describe normal operation and report which conditions are violated, giving a compact anomaly score and a simple diagnostic trace.

### Proposed method

Threshold continuous sensor windows into Boolean events such as high, low, rising, persistent, or cross-channel disagreement. Train a DLGN clause bank on normal examples plus structured synthetic corruptions. Each output clause should vote for normality; the anomaly score is the number or weighted count of violated clauses.

To prevent all clauses from learning the same easy condition, add diversity and coverage losses. Constrain each clause's effective input cone so that a detected anomaly can be reported as a small set of implicated threshold events. Use the hard circuit for scoring after training.

### Clear novelty boundary

Anomaly detection with synthetic negative samples is established, and interpretability is already a general DLGN motivation. The potential contribution is a one-class DLGN objective that learns diverse normality clauses whose violations remain meaningful after hardening. Merely training a binary DLGN on normal versus anomaly labels is not novel enough.

### Minimum experiment

- **Datasets:** a small UCR anomaly subset for iteration and one multivariate equipment or server-sensor dataset.
- **Baselines:** Isolation Forest; one-class SVM; small autoencoder; binary DLGN trained with anomaly labels; DLGN without clause diversity.
- **Metrics:** area under the precision-recall curve, false alarms per hour, detection delay, circuit size, clause diversity, and accuracy of the implicated-sensor explanation where labels permit it.
- **Stress test:** train corruptions of one type and evaluate unseen anomaly types to measure whether the model learned normality or only the corruption generator.

### Feasibility and failure condition

The implementation is moderate and requires no special tools, but the evaluation protocol must be selected carefully to avoid leakage between neighboring time windows.

This is a lower-confidence idea because the technical novelty can collapse into an ordinary anomaly detector. Stop if clause diversity does not improve generalization to unseen anomaly types or if the hard clauses cannot be connected to stable input conditions.

---

## Idea 8: EventLogic - sparse always-on acoustic detection

### Use case

Battery-powered devices listen continuously for a small vocabulary of events: an alarm, glass break, a machine fault, or a wake word. DLGN inference is attractive after the audio has been converted to bits, but the preprocessing cost and the temporal nature of audio must be included rather than ignored.

### Proposed method

Compute a small log-mel or filter-bank representation, then encode only threshold crossings and changes:

```text
band became high
band became low
band remained high for K frames
```

Feed these sparse Boolean events to a recurrent DLGN. Learn the spectral thresholds jointly and penalize event rate so that silence and stationary background produce little downstream logic activity. Evaluate both frame-level and utterance-level decisions.

Do not call the complete system multiplication-free: the spectral front end still performs arithmetic unless it is separately approximated. Report front-end and DLGN costs separately.

### Clear novelty boundary

Recurrent DLGNs already model sequences, and small-footprint keyword spotting is a mature field. A generic recurrent DLGN backend is weak novelty. The defensible contribution would be joint threshold-crossing/event-rate learning for a hardened recurrent logic detector, together with a full pipeline cost analysis.

### Minimum experiment

- **Datasets:** Speech Commands for keyword spotting or a compact public acoustic-event dataset.
- **Baselines:** feed-forward DLGN over stacked frames; recurrent DLGN with ordinary thermometer encoding; small GRU; compact CNN or DS-CNN; BNN if an implementation is available.
- **Metrics:** false accepts, false rejects, accuracy, detection latency, model bits, Boolean events per second, DLGN operations, and measured CPU time including preprocessing.
- **Stress tests:** background noise, microphone gain changes, unseen speakers, and long silence streams.

### Feasibility and failure condition

This needs only audio preprocessing and the recurrent DLGN code already present in the repository, but experiments will be slower and the baseline field is stronger than for the top ideas.

Treat this as exploratory. Continue only if sparse event encoding either improves noise robustness or reduces downstream operations substantially at matched false-accept and false-reject rates. Otherwise it is an application demonstration rather than a strong methodological paper.

---

## Suggested project sequence

The following sequence minimizes sunk effort and does not require choosing a publication venue first:

1. **Build one reusable threshold layer.** It supports PersonalDLGN directly and supplies the input encoding for TriSense, LogicFilter, and EventLogic.
2. **Run a small cross-subject benchmark.** Use one DLGN size and compare global versus threshold-only adaptation. This gives a go/no-go result for Idea 1 in approximately one to two weeks.
3. **Add hard-model serialization.** Store every gate as a truth-table ID and compute exact changed-gate and changed-bit counts. This enables Idea 2 and improves reproducibility for all later work.
4. **Choose the second project from evidence.** If subject threshold adaptation works, deepen PersonalDLGN. If adaptation requires changing gates, transition naturally to DriftPatch-DLGN. If missing channels dominate the errors, prioritize TriSense-DLGN.

## Ideas to avoid as standalone projects

- Applying a standard DLGN to another classification dataset without a new training, architecture, or deployment method.
- Learning input thresholds globally; WARP and prior interpretable DLGN work already do this.
- Applying a recurrent DLGN to a sequence dataset without a temporal contribution.
- Generic network-flow classification; SwitchLGN already covers this application.
- Generic abstention with ternary outputs; PST already establishes UNKNOWN-based selective prediction.
- A DLGN Bloom filter without counting the classifier, encoder, and backup filter in total memory.
- A neural error-correcting decoder evaluated only on an independent symmetric channel where conventional decoding is already near optimal.

## Bottom line

For the user's research profile, the best balance of AI, edge deployment, and manageable implementation is:

1. **PersonalDLGN** for accuracy under patient/device shift;
2. **DriftPatch-DLGN** for compact continual updates;
3. **TriSense-DLGN** for robust sensor fusion and safe abstention;
4. **ExitLogic** for software-side adaptive latency.

These projects make the use case part of the algorithm and can be validated before any hardware implementation. The remaining four ideas broaden the application space, but their novelty depends more heavily on outperforming mature domain-specific baselines.
