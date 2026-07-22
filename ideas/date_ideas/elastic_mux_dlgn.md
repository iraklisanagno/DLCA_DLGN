# ElasticMux-DLGN: One Circuit with Externally Selected Cost Modes

**Source concept:** ElasticMux-DLGN in [diffmux.md](../diffmux.md).

## Narrative

An edge device does not always have the same priorities. A battery-powered camera may need an economical mode most of the day, a balanced mode during normal activity, and its most accurate mode during an alarm. Shipping three unrelated models multiplies storage, update, and verification effort.

ElasticMux-DLGN trains one logic model with several useful prefixes. The system, not the classifier, selects `economy`, `balanced`, or `accurate`. A small runtime selector dispatches to the requested prefix, and all later stages are physically skipped. The modes share early logic and one checkpoint while exposing a controlled accuracy--latency tradeoff.

The first implementation deliberately avoids adding a multiplexer to every feature or gate. Such fine-grained feature-width switching can cost more area and routing than it saves. A single prefix selector is a feasible way to test the elastic-computation hypothesis before more complex nested-width circuits.

## Research question and claim

**Question:** Can one hardened DLGN provide three externally selected operating points that approach separately trained static models while using less total storage and update complexity?

**Target claim:** Prefix-trained ElasticMux offers a better joint storage--accuracy--latency tradeoff than storing independent DLGNs, and its selected mode creates real measured work reduction rather than only masking outputs.

ElasticMux is distinct from [ExitLogic](exit_logic.md):

- ElasticMux receives a runtime mode from the operating system/application and accepts mode-dependent accuracy.
- ExitLogic decides automatically for each sample and must exactly preserve the full model's prediction.

## Technical design

Build a shared stem followed by three cumulative logic stages:

~~~text
input -> shared stem -> stage 1 -> economy head
                             -> stage 2 -> balanced head
                                         -> stage 3 -> accurate head
~~~

Each head uses a positive-vote `GroupSum` output and is independently hard-evaluable. The economy path executes the stem and stage 1 only; balanced also executes stage 2; accurate executes all stages. A runtime enum selects one compiled prefix. The selector's code and storage overhead are included in measurements.

Train with a weighted multi-mode objective:

$$
\mathcal{L}=w_E\mathcal{L}_E+w_B\mathcal{L}_B+w_A\mathcal{L}_A+lambda_D(\mathcal{L}_{E\leftarrow A}+\mathcal{L}_{B\leftarrow A}).
$$

The accurate head is the internal teacher for optional self-distillation. Begin by training the accurate path, then enable all heads and jointly fine-tune. Keep only three modes in the main paper. Width nesting and gate-level learned muxes are future extensions unless prefix execution fails for a clearly diagnosed reason.

## Implementation plan

### Milestone 1: Prefix model and accounting

1. Add a three-stage model whose `forward(mode)` returns exactly one selected head.
2. Count gates and memory reachable from each mode, including the shared stem and selected head.
3. Add runtime hooks proving later stages are not executed.
4. Export independent prefix entry points from one checkpoint/Circuit description.

### Milestone 2: Training schedule

1. Reproduce the deepest static DLGN baseline.
2. Train the shared deepest path, attach shallow/middle heads, then jointly fine-tune.
3. Add optional accurate-to-balanced/economy logit distillation.
4. Select loss weights using validation Pareto quality, then freeze them.

### Milestone 3: Static and storage baselines

1. Train independent economy, balanced, and accurate DLGNs at matched executed-gate budgets.
2. Compare naive truncation of the accurate model, multi-head training without self-distillation, and proposed ElasticMux.
3. Serialize one elastic checkpoint and all three independent checkpoints to measure actual bytes.
4. Compile and benchmark each mode with batch-1 control flow.

### Milestone 4: Optional robustness

1. Test switching modes between consecutive samples and verify no hidden state.
2. Evaluate mode-specific calibration and class errors.
3. If the accurate mode degrades, test gradient balancing before adding architectural complexity.
4. Consider a fourth mode or nested width only after the three-prefix result meets the kill criterion.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md):

- `repos/torchlogix/src/torchlogix/models/dense.py` and `repos/torchlogix/src/torchlogix/models/conv.py` for staged models;
- `repos/torchlogix/src/torchlogix/layers/groupsum.py` for independent vote heads;
- `repos/torchlogix/src/torchlogix/circuit.py` for reachable-cone accounting and compiled prefix exports.

TorchLogix supports the convolutional LGNs required for CIFAR-10. `repos/difflogic-light-master` does not and therefore cannot be the foundation, although its Light parameterization is a useful compact baseline after porting through TorchLogix. Share only generic prefix execution code with ExitLogic.

## Datasets and protocol

| Dataset | Role | Required modes |
|---|---|---|
| MNIST | Debug and proof of physical prefix skipping | Economy, balanced, accurate |
| CIFAR-10 | Primary DATE result | Three modes at predeclared low/medium/high executed-gate budgets |
| Fashion-MNIST | Optional second compact dataset | Reuse mode ratios without retuning architecture |

Focus the manuscript on MNIST and CIFAR-10 because these are standard in the local DLGN papers. Define mode budgets before final training, for example approximately 35%, 65%, and 100% of accurate-path active gates. If latency is not proportional to gates, adjust budgets once using validation profiling and then freeze them.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Three independent static convolutional DLGNs at mode-matched costs | Defines the per-mode accuracy ceiling and total storage cost |
| Kim et al., [*NestedNet: Learning Nested Sparse Structures in Deep Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2018/html/Kim_NestedNet_Learning_Nested_CVPR_2018_paper.html), CVPR 2018 | Three nested gate subsets or prefixes trained with the paper's nested-loss principle | Closest published single-model nested-computation baseline |
| Yu et al., [*Slimmable Neural Networks*](https://iclr.cc/virtual/2019/poster/796), ICLR 2019 | Three switchable-width DLGN modes where legal, or a matched CNN baseline if gate nesting is unavailable | Canonical runtime width-switching comparison |
| Yu and Huang, [*Universally Slimmable Networks and Improved Training Techniques*](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Universally_Slimmable_Networks_and_Improved_Training_Techniques_ICCV_2019_paper.html), ICCV 2019 | Sandwich-rule and in-place-distillation training applied to the same three DLGN modes | Tests whether published elastic-training techniques explain the gain |
| Cai et al., [*Once-for-All: Train One Network and Specialize it for Efficient Deployment*](https://research.ibm.com/publications/once-for-all-train-one-network-and-specialize-it-for-efficient-deployment), ICLR 2020 | OFA progressive shrinking over the same three depth choices, without hardware search | Strong published one-network/many-subnet baseline |
| Huang et al., [*Multi-Scale Dense Networks for Resource Efficient Image Classification*](https://ai.meta.com/research/publications/multi-scale-dense-networks-for-resource-efficient-image-classification/), ICLR 2018 | MSDNet-style anytime prefixes with externally fixed exit points | Published prefix/anytime comparison distinct from width switching |

The minimum DATE set is independent static DLGNs, NestedNet-style nested training, US-Net sandwich/in-place distillation, naive prefixes, and ElasticMux. A complete OFA search is not required; use its progressive-shrinking training as a named baseline over the predeclared three modes. Conventional CNN implementations may be reported as context but cannot replace same-DLGN adaptations.

### ElasticMux-specific controls

Compare:

1. deepest static DLGN;
2. naive prefixes of that model without prefix-aware training;
3. shared multi-head model without self-distillation;
4. proposed ElasticMux with joint training and self-distillation;
5. one larger static model whose serialized bytes match the elastic model.

Compare both per-mode quality and total deployment. Independent models may win one operating point but pay the sum of all checkpoint bytes. ElasticMux counts every head and selector even when a mode does not execute them.

## Metrics

Report for each mode:

- hardened top-1 accuracy, soft accuracy, and soft-to-hard gap;
- executed/reachable gates, logic depth, connection bits, and head overhead;
- batch-1 mean and p95 latency, throughput, and optional measured energy;
- accuracy gap to an independently trained static model at matched cost;
- class-wise accuracy and calibration error if mode confidence is exposed.

Report for the complete elastic deployment:

- serialized bytes versus the sum of three independent model files;
- additional head/selector gates and bytes;
- training time, peak GPU memory, and compile time;
- monotonicity of accuracy and cost across economy, balanced, and accurate modes;
- Pareto hypervolume or area under the three-point accuracy--latency curve;
- mode-switch overhead and verification that mode selection has no state leakage.

Use at least five paired seeds for the central CIFAR-10 comparison. The main figure should overlay ElasticMux modes, independent static models, and naive prefixes in accuracy--latency and accuracy--storage space.

## Minimum DATE experiment matrix

- MNIST three-mode functional and compiled-prefix validation.
- CIFAR-10 two total model budgets, each with economy/balanced/accurate modes.
- Independent static, naive prefix, no-distillation multi-head, and proposed ElasticMux.
- Ablations for self-distillation, loss weighting, and shared-stem size.
- Five seeds for the central CIFAR-10 budget and three for secondary cells.
- One model update/storage case showing the cost of distributing one elastic checkpoint versus three independent checkpoints.

## Agent deliverables and tests

The assigned agent must provide:

- mode-dispatch tests that record exactly which stages execute;
- reachability-derived gate counts checked against runtime traces;
- compiled prefix entry points and a selector-overhead benchmark;
- serialization tests proving all modes reload from one checkpoint;
- per-mode and per-sample timing/accuracy records;
- independent static baselines trained with matched validation effort;
- scripts for accuracy--latency, accuracy--gate, and storage figures.

## Risks, controls, and kill criterion

- **Accurate mode is harmed by shallow heads:** use staged training or gradient balancing and report accurate-only versus joint training.
- **Shared stem dominates every mode:** reduce stem depth or move the first exit earlier; count the stem in all results.
- **Logical gate savings do not reduce runtime:** compile true prefix functions and measure control overhead.
- **Independent models are much more accurate:** report per-mode static gaps and optimize the shared representation, not just loss weights.
- **Feature-wide mux temptation:** do not add gate-level muxes until prefix execution is proven insufficient and a full mux-overhead model exists.

Continue as a DATE paper only if all three modes are monotonic in cost and accuracy, economy reduces measured latency or active gates by at least 25% relative to accurate, accurate loses no more than 0.75 percentage points versus the matched static deep model, and extra heads/selector add no more than 15% to the accurate model's serialized size. ElasticMux must also use fewer bytes than the three independent baselines combined.

## Definition of done

The project is ready when one checkpoint reliably exports three physically distinct prefixes, mode costs are measured rather than inferred only from masks, all required static baselines exist, and five-seed CIFAR-10 results demonstrate a useful joint accuracy--latency--storage tradeoff.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
- [Mind the Gap](../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf)
