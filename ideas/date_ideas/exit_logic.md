# ExitLogic: Certified Early Termination for Logic Classifiers

**Source concept:** ExitLogic in [july_20_no_DATE.md](../july_20_no_DATE.md).

## Narrative

A logic classifier may contain thousands of small votes for its output classes. Easy inputs often produce an overwhelming lead early, but a conventional implementation still evaluates every remaining gate. It is like counting every ballot after one candidate is already mathematically guaranteed to win.

ExitLogic organizes the output computation into independently executable blocks. After each block, it checks whether any unfinished block could still change the winner. If not, execution stops. The answer is exactly the answer that the full hardened model would have produced, not merely a confident guess.

This makes early exit particularly suitable for safety-conscious edge systems: difficult samples receive the full computation, easy samples save time, and the optimization does not introduce a new accuracy error relative to the deployed full model.

## Research question and claim

**Question:** Can a hardened DLGN be trained and organized so a large fraction of MNIST and CIFAR-10 samples receive an exact early-exit certificate?

**Target claim:** Certified vote-block execution reduces average active gates and measured latency while preserving 100% decision agreement with full hardened inference.

The certificate preserves the full model's prediction, which can still be wrong relative to the ground-truth label. The paper must make this distinction explicit.

## Exact certificate

Use nonnegative binary class votes compatible with TorchLogix `GroupSum`. After prefix $p$, let $s_c^{(p)}$ be the observed votes for class $c$, and let $R_c^{(p)}$ be the maximum number of uncomputed votes that class can still receive. If the current leader is $k$, a sufficient exact stopping condition is:

$$
s_k^{(p)} > \max_{c \ne k}\left(s_c^{(p)} + R_c^{(p)}\right).
$$

This strict inequality is valid for an argmax that favors no tied challenger. If the deployed implementation uses a deterministic class-index tie rule, implement and test the corresponding non-strict condition only where the tie rule proves it safe.

The certificate is invalid if future blocks can contribute negative or unbounded scores. Keep the MVP output as sums of binary positive votes and include every remaining vote in $R_c^{(p)}$.

## Technical design

Create $B$ ordered vote blocks. Each block contains a separately executable fan-in cone ending in a known number of votes per class. A shared input encoding or small convolutional stem is permitted, but any shared cost is always paid and counted. Once the shared stem is complete, execute blocks sequentially, update class sums, and test the exact certificate.

Develop two architectures:

1. **Dense vote blocks:** the simple MNIST implementation. Each block is an independent small DLGN branch over the encoded input.
2. **Shared stem plus vote branches:** the CIFAR-10 implementation. A convolutional logic stem produces shared features, followed by independent branch cones. The stem should remain shallow enough that branch skipping can produce real savings.

Train the full model with final classification loss plus optional prefix losses that encourage useful early votes. A prefix loss is a training aid only; stopping at inference always uses the certificate. Order branches using validation data and a predeclared utility such as certified exits per measured block cost. Compare learned/trained ordering against random and equal-cost ordering.

## Implementation plan

### Milestone 1: Certificate correctness

1. Implement a pure-Python vote-prefix evaluator around `GroupSum`.
2. Define the exact argmax/tie behavior once and share it with full inference.
3. Exhaustively enumerate small vote vectors and prove every early result equals the full result.
4. Add property tests over random block sizes, class counts, and ties.

### Milestone 2: Dense independent branches

1. Build a dense MNIST model with 4 or 8 independently callable vote blocks.
2. Train final-only and prefix-supervised variants at the same total gate budget.
3. Record per-prefix class sums, certified-exit rates, and active gates.
4. Export each block cone and verify that skipped blocks are never called.

### Milestone 3: CIFAR-10 architecture

1. Add a shallow shared `LogicConv2d` stem and independent rank-2 branch back ends.
2. Profile the shared-stem fraction before large training runs.
3. Train block orders jointly or reorder completed branches using validation data.
4. Compile prefix functions so runtime measurements include real control flow rather than simulated gate counts alone.

### Milestone 4: Runtime and robustness

1. Benchmark batch size 1 and a small batch relevant to edge execution.
2. Evaluate exit rates by class and by correct/incorrect full-model predictions.
3. Test distribution shifts such as common CIFAR-10 corruptions only after the clean-data result is stable.
4. Keep heuristic confidence exits as a separate baseline; never mix their errors into the certified result.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md):

- `repos/torchlogix/src/torchlogix/layers/groupsum.py` defines the positive class-vote output;
- `repos/torchlogix/src/torchlogix/circuit.py` supports executable fan-in cones and compiled C;
- `repos/torchlogix/src/torchlogix/models/dense.py` and `repos/torchlogix/src/torchlogix/models/conv.py` provide the dense and convolutional starting points.

The original `repos/difflogic` compiler is a useful full-model latency reference. Do not use `repos/difflogic-light-master` as the foundation: it has no convolutional LGN, which is needed for the CIFAR-focused story. ExitLogic and ElasticMux should share prefix-execution utilities but not stopping policies.

## Datasets and protocol

| Dataset | Role | Protocol |
|---|---|---|
| MNIST | Required correctness and dense result | Use the encoding and GroupSum output of the original DLGN experiments; test 4 and 8 blocks |
| CIFAR-10 | Primary DATE result | Use the convolutional DLGN protocol and standard augmentation; test two gate budgets |
| Fashion-MNIST | Optional generalization | Run only after MNIST/CIFAR-10; no method retuning beyond block count |

The selected emphasis on MNIST and CIFAR-10 aligns the study with the local DLGN papers. Report full-model hard accuracy before any savings result. Calibrate branch order and any prefix-loss weight on validation data only.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Deep Differentiable Logic Gate Networks*](../../pdfs/deep_differentiable_logic_gate_networks.pdf), NeurIPS 2022 | Full hardened DLGN and smaller static DLGNs at prefix-matched gate counts | Defines full-compute accuracy and the static accuracy--cost frontier |
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Full convolutional DLGN on CIFAR-10 | Ensures savings are not caused by replacing the published spatial model with a weaker dense model |
| Teerapittayanon et al., [*BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks*](https://arxiv.org/abs/1709.01686), ICPR 2016 | Entropy-threshold exits on the same DLGN branches | Canonical confidence-based early-exit comparison on MNIST/CIFAR-10 |
| Huang et al., [*Multi-Scale Dense Networks for Resource Efficient Image Classification*](https://ai.meta.com/research/publications/multi-scale-dense-networks-for-resource-efficient-image-classification/), ICLR 2018 | MSDNet-style jointly trained exits and budgeted evaluation | Strong published anytime/budgeted-classification baseline |
| Kaya et al., [*Shallow-Deep Networks: Understanding and Mitigating Network Overthinking*](https://proceedings.mlr.press/v97/kaya19a.html), ICML 2019 | SDN internal classifiers with confidence thresholds | Provides a modern multi-exit training and calibration baseline |
| Wang et al., [*SkipNet: Learning Dynamic Routing in Convolutional Networks*](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html), ECCV 2018 | Input-dependent block-skipping model at matched average cost, if reproducible | Contrasts certified termination with learned dynamic execution |

The minimum DATE set is full/static DLGNs, BranchyNet-style entropy exits, SDN-style internal classifiers, and certified ExitLogic. MSDNet and SkipNet may use their published CNN architectures as contextual baselines, but the main paired experiment must adapt their stopping policies to the same DLGN branch architecture so gate and latency measurements remain meaningful.

### ExitLogic-specific controls

Compare:

1. fixed-prefix execution with no per-sample decision;
2. entropy and top-two-margin exits calibrated to several error budgets;
3. random branch ordering;
4. equal-cost and validation-utility branch ordering;
5. multi-branch training without prefix losses;
6. proposed prefix-supervised certified ExitLogic.

For every heuristic paper baseline, report disagreement with complete DLGN inference and excess classification errors. For ExitLogic, this disagreement must remain zero.

## Metrics

Report:

- full hardened accuracy and early-execution accuracy;
- decision agreement with full inference, which must be exactly 100% for certified exits;
- certified exit rate at every prefix and fraction reaching the full network;
- mean, median, and p95 active gates and executed blocks;
- mean and p95 batch-1 latency, throughput at a stated batch size, and speedup;
- shared-stem cost, branch/control overhead, compiled code size, and total model storage;
- exit rate by class, full-model correctness, and input difficulty;
- energy measured with a reproducible tool if available, otherwise active-gate and latency proxies clearly labeled as proxies.

Report paired results over at least five training seeds for the central CIFAR-10 architecture. Each run must verify every test prediction against complete inference.

## Minimum DATE experiment matrix

- MNIST with 4 and 8 blocks for final-only and prefix-supervised training.
- CIFAR-10 with two gate budgets and at least two stem/branch cost splits.
- Full, fixed-prefix, heuristic, and certified execution.
- Random versus utility-ordered branches.
- Ablations for prefix loss and certificate-check frequency.
- Five seeds for the central CIFAR-10 result and three elsewhere.

## Agent deliverables and tests

The assigned agent must provide:

- exhaustive and property-based certificate tests, including all tie cases;
- an inference trace listing blocks executed, partial sums, remaining bounds, and stop reason;
- an automated full-versus-early prediction audit over every evaluated sample;
- compiled prefix functions and proof that skipped blocks are not executed;
- profiler output separating shared stem, branches, and control checks;
- per-sample exit and latency records, not only averages;
- scripts for accuracy--latency and exit-rate figures.

## Risks, controls, and kill criterion

- **Shared work dominates:** profile before final runs and branch earlier if the stem consumes most gates or latency.
- **Certificate rarely fires:** use independent, class-balanced vote blocks and prefix supervision, but do not weaken exactness.
- **Gate savings do not become latency savings:** compile real branch control flow and report overhead.
- **Tie-rule bug:** centralize the argmax policy and test exhaustive small cases.
- **Confusion with ElasticMux:** ExitLogic is automatic and exact per sample; ElasticMux is externally selected and permits a deliberate accuracy tradeoff.

Continue as a DATE paper only if certified execution preserves 100% full-model decisions, at least 20% of CIFAR-10 samples exit before the final block, and average measured latency or active gates improve by at least 15% without reducing full-model hard accuracy by more than 0.5 percentage points relative to the matched non-exit architecture.

## Definition of done

The project is ready when the certificate has exhaustive tests, all evaluation samples have a full-agreement audit, the compiled implementation physically skips block cones, and MNIST plus five-seed CIFAR-10 results show a meaningful cost reduction.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
