# QBridge-DLGN: Transfer from Quantized CNNs to Convolutional Logic

**Source concept:** QBridge-DLGN in [transfer_learning.md](../transfer_learning.md).

## Narrative

Quantized CNNs such as low-bit ResNet and MobileNet models are widely studied and easier to train than DLGNs. They already express much of their computation as small integers. A DLGN goes one step further and turns computation into explicit logic gates, but training a competitive convolutional logic network from scratch remains difficult.

QBridge uses a trained quantized CNN as a bridge. It translates the teacher's low-bit intermediate codes into binary targets and uses the teacher's channel relationships to choose useful fixed connections for a convolutional DLGN. The final model is still a standalone logic circuit: the CNN and all transfer helpers disappear after training.

The intended contribution is a practical route from familiar quantized architectures to LUT/gate-native inference. It is functional transfer, not a claim that an arbitrary CNN can be converted gate-for-gate into an equivalent DLGN.

## Research question and claim

**Question:** Can native low-bit codes and connectivity statistics from established quantized CNNs make convolutional DLGNs more accurate or faster to train than scratch and ordinary knowledge distillation?

**Target claim:** QBridge improves the hard-accuracy--gate-count frontier by jointly transferring quantized intermediate codes and teacher-guided fixed routes, with no teacher cost at deployment.

Keep the claim precise:

- the teacher and student compute different functions and have different structures;
- no exact structural mapping from multiply-accumulate layers to two-input gates is promised;
- the final DLGN must be evaluated independently after every bridge component is removed.

## Teachers and transfer representation

Use recognizable quantized CNN families:

1. quantized ResNet-18 at W8A8 and W4A4;
2. quantized MobileNetV2 at W8A8 and W4A4;
3. an optional Brevitas/FINN-style CNV model as a LUT-oriented pilot if a reproducible checkpoint is available.

The primary paper comparison should use ResNet-18 and MobileNetV2 because they are familiar architectures. A TorchVision post-training/static INT8 ResNet can be an adapter smoke test, but CUDA training should use a reproducible fake-quantization or Brevitas model whose integer activation scales and codes are accessible. Pin package versions and archive teacher checkpoints in the experiment manifest.

At selected teacher stages, retain the native unsigned/signed activation integer $q_T$. Convert it to one of two Boolean representations:

- **bitplane:** one bit per integer bit position;
- **thermometer:** ordered threshold bits, used only at low precision because its width grows with the number of levels.

The bitplane representation is the default. It preserves the teacher's actual quantized code and avoids inventing a separate random projection.

## Technical design

QBridge has two independently testable mechanisms.

### Native-code distillation

Align one or two student stages with teacher stages after spatial pooling or resolution matching. Attach training-only probes that predict the teacher activation bitplanes. Combine label loss, logit KD, and code loss:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{KD}\mathcal{L}_{\mathrm{logit}} + \lambda_Q\sum_l w_l\mathcal{L}_{\mathrm{bit}}(l).
$$

Weight rare sign/high-order bits so the loss is not dominated by constant bitplanes. Remove probes after training.

### Teacher-guided fixed routes

Within the legal receptive field of each `LogicConv2d` stage, rank candidate teacher-code channels using training-set statistics. Assign student gate inputs to informative but nonredundant source channels using a simple score combining label or teacher-code mutual information and pairwise redundancy. The generated routes are fixed before DLGN training and add no trainable routing parameters.

Implement route generation after native-code distillation works. Keep the route score simple and deterministic; do not introduce a learned router in the MVP. Compare route-only, code-only, and combined QBridge to isolate the contribution.

Train progressively: first train an aligned early stage/probe, then add the next student stage, and finally fine-tune the complete student with hard accuracy monitored. This is a training schedule, not a requirement to retain the teacher online.

## Implementation plan

### Milestone 1: Quantized-teacher adapters

1. Define one adapter API that returns logits, integer activation codes, scales/zero points, layer names, and spatial shapes.
2. Implement ResNet-18 first, then MobileNetV2; use one quantization library for the main result.
3. Add a cache keyed by dataset/view, teacher hash, quantization configuration, and layer list.
4. Verify that dequantizing cached integer codes reproduces the adapter's quantized activations within the declared scale.

### Milestone 2: Code-distillation MVP

1. Start with one late teacher stage and bitplane targets.
2. Add one training-only student probe with explicit spatial/channel alignment.
3. Compare labels, logit KD, floating-feature KD, and native-code KD at an identical student budget.
4. Audit that all probes and teacher metadata are absent from the hardened circuit.

### Milestone 3: Guided connectivity

1. Compute per-channel informativeness and redundancy using training data only.
2. Generate fixed legal convolutional source indices from these statistics.
3. Add random-route and shuffled-teacher-route controls.
4. Evaluate code-only, route-only, and combined QBridge.

### Milestone 4: Progressive transfer and second teacher

1. Add a two-stage progressive schedule only after the single-stage result is positive.
2. Freeze the best bridge settings on ResNet-18/CIFAR-10.
3. Test MobileNetV2 without a broad retuning sweep.
4. Extend to CIFAR-100 at one compact and one medium DLGN budget.

Do not begin with exact CNN-to-Boolean synthesis, arbitrary ONNX conversion, or external HLS/EDA flows. These are outside the paper's feasible core.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md). It already provides `LogicConv2d`, which is essential for preserving spatial structure from quantized CNN teachers. Likely components are:

- `repos/torchlogix/src/torchlogix/layers/conv.py` for legal convolutional connections;
- `repos/torchlogix/src/torchlogix/connections.py` for teacher-guided route generation;
- `repos/torchlogix/src/torchlogix/models/conv.py` for stage-aligned students;
- a shared feature/code cache also used by LogicSketch.

`repos/difflogic-light-master` must not be used as the foundation because it does not implement convolutional LGNs. Its Light parameterization can be ported or compared within TorchLogix. `repos/difflogic` remains a dense baseline but cannot support the main famous-CNN transfer story as directly.

## Datasets and protocol

| Dataset | Role | Protocol |
|---|---|---|
| MNIST | Adapter and bitplane correctness only | Small quantized CNN teacher and compact DLGN student |
| CIFAR-10 | Primary DATE result | Quantized ResNet-18 primary teacher; MobileNetV2 cross-teacher validation |
| CIFAR-100 | Secondary scale result | Best bridge configuration from CIFAR-10 with limited retuning |

Use identical train/validation/test splits and augmentation views for teacher-target generation and student inputs. If the student receives random augmentation, either run the frozen teacher on that exact view or cache deterministic augmented views by seed. Never pair codes and images from different crops or flips.

Train the teacher only on training data. Select quantization settings and bridge hyperparameters with validation data. Test data may be passed through the teacher only for final reporting, never to select routes or losses.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Convolutional DLGN trained from scratch at each gate budget | Direct student baseline |
| Hubara et al., [*Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations*](https://www.jmlr.org/papers/v18/16-456.html), JMLR 2018 | Low-bit CNN teacher and parameter-matched QNN context model | Establishes the standard QNN family from which codes are transferred |
| Jacob et al., [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html), CVPR 2018 | W8A8 quantized ResNet/MobileNet teacher | Provides a published integer-only quantization reference |
| Mishra and Marr, [*Apprentice: Using Knowledge Distillation Techniques to Improve Low-Precision Network Accuracy*](https://openreview.net/pdf?id=B1ae1lZRb), ICLR 2018 | Apprentice logit-distillation schedule with the selected quantized teacher and DLGN student | Most direct published low-precision KD baseline |
| Zhuang et al., [*Towards Effective Low-Bitwidth Convolutional Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhuang_Towards_Effective_Low-Bitwidth_CVPR_2018_paper.html), CVPR 2018 | Progressive weight/activation quantization and hint transfer | Tests whether QBridge's progressive schedule, rather than native codes, explains the gain |
| Heo et al., [*Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons*](https://arxiv.org/abs/1811.03233), AAAI 2019 | Activation-boundary distillation at the same aligned stages | Particularly relevant baseline because it transfers binary boundary information |
| Umuroglu et al., [*FINN: A Framework for Fast, Scalable Binarized Neural Network Inference*](https://arxiv.org/abs/1612.07119), FPGA 2017 | FINN CNV/QNN accuracy and hardware results, rerun where practical | Published LUT-oriented QNN deployment baseline |
| Buehrer et al., [*BitLogic*](../../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf), TMLR 2026 | BitLogic best-of-space LUT-native model at matched dataset and approximate budget | Current unified LUT-native competitor |

The minimum DATE set is scratch convolutional DLGN, ordinary Hinton logit KD, Apprentice, activation-boundary distillation, and QBridge. Include BitLogic on every dataset/budget supported by its public pipeline. FINN and the quantized teachers are contextual hardware/model baselines unless they can be passed through the same measurement flow.

### QBridge-specific controls

At matched student architecture, gate budget, and teacher checkpoint, compare:

1. full-precision-teacher and quantized-teacher logit KD;
2. floating/dequantized feature distillation;
3. native-code distillation only;
4. teacher-guided routes only;
5. combined code plus routes;
6. random fixed routes and shuffled teacher statistics;
7. thermometer versus bitplane codes at one low precision;
8. the quantized CNN teacher itself, with its compute and storage reported separately.

Do not claim a direct hardware win over a QNN, FINN, or BitLogic result unless both sides are measured through a comparable implementation flow.

## Metrics

Report:

- student hardened top-1 accuracy, soft accuracy, and soft-to-hard gap;
- teacher quantized accuracy and quantization gap from its full-precision source;
- deployed DLGN gate count, connection bits, model bytes, logic depth, and compiled latency;
- steps/GPU-hours to fixed student hard-accuracy thresholds and final accuracy at fixed compute;
- teacher pretraining/QAT cost, code-cache time and bytes, and peak GPU memory;
- per-bitplane entropy, agreement, imbalance, and student code prediction accuracy;
- route-source diversity, redundancy, fan-out, and sensitivity to route seed;
- training-only probe parameters, which are excluded from deployment but reported;
- accuracy versus gate count and accuracy versus total transfer compute.

Use five paired student seeds for the central ResNet-18/CIFAR-10 result and three seeds for MobileNetV2 and CIFAR-100. Separate teacher seed, quantization seed, route seed, and student seed.

## Minimum DATE experiment matrix

- CIFAR-10 with W8A8 and W4A4 ResNet-18 teachers at two DLGN budgets.
- One MobileNetV2 cross-teacher experiment at the selected precision.
- Scratch, logit KD, floating feature, code-only, route-only, and combined QBridge.
- Bitplane/thermometer ablation and one/two aligned-stage ablation.
- CIFAR-100 at the strongest compact setting.
- Five seeds for the central combined method and matched baselines.

## Agent deliverables and tests

The assigned agent must provide:

- versioned teacher adapters with integer-code reconstruction tests;
- cache hash, shape, sample/view alignment, and corruption tests;
- deterministic legal-route generation tests;
- a deploy audit proving no teacher/probe dependency remains;
- frozen teacher checkpoint references and exact QAT commands;
- raw per-seed code agreement, route statistics, hard accuracy, memory, and time;
- scripts regenerating the main accuracy--gate and time-to-accuracy plots.

## Risks, controls, and kill criterion

- **Teacher codes are too wide:** begin with selected channels and bitplanes; avoid large thermometer encodings.
- **Quantization library dominates reproducibility:** use one pinned primary stack and export cached integer codes into a library-neutral format.
- **Gain is ordinary KD:** include tuned logit and dequantized-feature baselines.
- **Teacher-guided routes duplicate CoverageDLGN:** isolate code and route effects; QBridge routes are teacher/data-derived, while CoverageDLGN routes are task-agnostic structural schedules.
- **Exact conversion is infeasible:** state functional transfer explicitly and avoid structural-equivalence claims.
- **External tool burden:** no EDA tool is required for training or the primary evaluation.

Continue as a DATE submission only if combined QBridge exceeds a tuned scratch DLGN by at least 1.5 percentage points of hard CIFAR-10 accuracy at a compact gate budget, or cuts time to a fixed hard-accuracy target by at least 2x, and beats quantized-teacher logit KD. The effect must survive a second budget, teacher family, or CIFAR-100 result.

## Definition of done

The project is ready when two famous quantized CNN adapters are reproducible, native codes and augmentation views are aligned, deployed students contain no bridge artifacts, and code-only/route-only/combined ablations support the claimed mechanism across five central seeds.

## Primary references in this repository

- [BitLogic framework](../../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
