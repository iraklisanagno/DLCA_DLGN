# Transfer Learning for Differentiable Logic Gate Networks

**Prepared:** July 22, 2026

**Scope:** DLGN-to-DLGN transfer, architecture growth, and full-precision/quantized CNN-to-convolutional-DLGN transfer

## Executive assessment

A targeted search found no direct transfer-learning method for differentiable logic gate networks. However, all broad ingredients exist in adjacent fields:

- Net2Net and network morphism grow conventional neural networks through function-preserving transformations;
- binary networks have been used as fixed transferable feature extractors;
- conventional and binary students have been trained with knowledge distillation;
- LUTNet begins from BNN-style computation;
- LogicNets, TTNet, and NullaNet convert specially constrained quantized networks into truth-table or Boolean circuits;
- DLGN residual initialization already biases new gates toward copy/identity behavior.

Therefore, neither "fine-tune a DLGN," "use a CNN teacher," nor "initialize a new layer as identity" is sufficient novelty.

The best three research directions are:

1. **LogicMorph-DLGN:** exactly embed a trained hard DLGN into a deeper, wider, higher-fan-in child circuit, then adapt only new capacity before selectively unlocking inherited gates.
2. **QBridge-DLGN:** transfer established INT8 or few-bit CNNs into TorchLogix convolutional logic stages through quantization-code interfaces, teacher-guided logic connections, and progressive hard-stage training.
3. **Boolean Interface Distillation (BID-DLGN):** retain a full-precision CNN variant as a broader but less structurally aligned teacher baseline.

The direct DLGN-to-DLGN project remains the safer first paper. However, the local TorchLogix repository changes the feasibility assessment of CNN-to-DLGN transfer: it already supplies convolutional logic layers, CIFAR-10 models, hard export, and a circuit IR. QBridge-DLGN is therefore implementable without first building a convolutional DLGN framework. It has a more crowded distillation neighborhood, so the contribution cannot be merely using a quantized teacher.

### Ranked options

| Rank | Working title | Transfer direction | Core contribution | Feasibility | Novelty confidence |
|---:|---|---|---|---|---|
| 1 | LogicMorph-DLGN | DLGN to larger/deeper DLGN | Exact logic-specific depth, width, fan-in, and graph morphisms | High | Medium-high |
| 2 | QBridge-DLGN | Quantized CNN to convolutional DLGN | Quantization-code interfaces plus teacher-guided local logic topology | High | Medium-high |
| 3 | BID-DLGN | Full-precision CNN to DLGN | Intermediate teacher features translated into attainable Boolean interfaces | High | Medium |
| 4 | TransferBus-DLGN | DLGN source task to DLGN target task | Class-independent binary evidence bus and selective unfreezing | High | Medium |
| 5 | Compile-and-Refine | Quantized CNN to truth-table circuit to DLGN | Structural conversion followed by differentiable logic refinement | Low-medium | Low-medium |
| 6 | Logit KD only | CNN to DLGN | Ordinary teacher soft labels | Very high | Low; baseline only |

LogicMorph and TransferBus can form one paper. QBridge-DLGN should be the quantized-CNN paper, with BID-DLGN treated as its full-precision-teacher baseline/generalization rather than as a competing main idea.

## 1. Clarifying the two proposed transfer settings

### Setting A: trained DLGN to a deeper target DLGN

Example:

~~~text
source: shallow CIFAR-10 DLGN
              |
              v
target: deeper/wider CIFAR-100 DLGN
~~~

There are two distinct transfers:

1. **architecture transfer:** move learned gates and connectivity into a larger child model;
2. **task transfer:** reuse source features when the labels change.

These must be evaluated separately. A method can preserve the source function exactly while providing no useful CIFAR-100 features. Conversely, source features may help CIFAR-100 even if the child is not an exact source-function morphism.

### Setting B: conventional or quantized CNN to DLGN

A regular CNN has real-valued convolutions, normalization, nonlinearities, and high-fan-in accumulation. A DLGN has thresholded inputs and a graph of small Boolean functions. There is no natural one-to-one parameter map.

A quantized CNN narrows this representation gap because its intermediate activations have finite integer codes. It does not remove the structural gap: an INT8 convolution still accumulates hundreds or thousands of products before requantization, whereas one TorchLogix convolutional kernel evaluates a small logic tree over selected local bits.

Three levels of transfer are possible:

1. **output transfer:** match teacher logits;
2. **representation transfer:** match carefully designed Boolean sketches of teacher features;
3. **structural conversion:** first constrain/quantize the CNN so each neuron can be enumerated as a manageable truth table.

Level 2 is the best research opportunity. For a quantized teacher, the interfaces can preserve its native integer activation codes rather than thresholding arbitrary floating-point features. Level 3 is possible only after substantially changing the CNN and overlaps existing LUT-conversion frameworks.

## 2. Status of the closest state of the art

### 2.1 DLGN training is expensive, but optimization has improved

The original and convolutional DLGN papers report very high training costs for large models. Mind the Gap reduces wall-clock training time and the soft-to-hard gap substantially. DLGN-Light/IWP, WARP, residual initialization, and fully trainable DLGN/LUTN improve optimization and depth.

Transfer learning therefore remains motivated, but a paper must compare against these improved training recipes. Beating only the original soft 16-gate formulation would exaggerate the benefit.

### 2.2 BitLogic does not currently provide transfer learning

BitLogic exposes five independently configurable axes:

1. encoder;
2. connectivity;
3. fan-in;
4. node parameterization;
5. output head.

Its experiments are two-layer feed-forward networks. It chooses an effective design-space configuration on MNIST and retrains that configuration on Fashion-MNIST, CIFAR-10, and CIFAR-100. This is **hyperparameter/configuration transfer**, not checkpoint or feature transfer.

Its quantized head operates after a class-length GroupSum vector:

$$
u\in\mathbb{Z}^{C},
\qquad
s=W_q u,\quad W_q\in\mathbb{Z}^{C\times C}.
$$

It is useful for narrow models but is class-count dependent. A CIFAR-10 head cannot directly support 100 target classes. The source trunk or a class-independent evidence layer must be transferred; the source head normally must be replaced.

### 2.3 Existing DLGN identity initialization is not transfer

Convolutional DLGNs use residual initialization: a new gate is initialized with high probability on the copy-A function, which stabilizes deep training. This starts a new model near an identity mapping. It does not:

- transplant a trained parent graph;
- preserve a complete parent function exactly;
- widen or raise fan-in while preserving behavior;
- measure target-task training acceleration.

LogicMorph-DLGN must include residual initialization as a baseline and make exact parent-to-child transformation the differentiator.

### 2.4 Generic network growth is established

Net2Net and network morphism preserve the function of a conventional network while increasing depth, width, kernel size, or subnet capacity. Progressive networks and side-tuning freeze old computation and add new capacity.

The possible DLGN novelty is narrower:

> Define exact morphisms over hard truth tables, gate fan-in, discrete connections, and Boolean fanout, and show that these morphisms accelerate training of a larger hardened logic circuit.

### 2.5 Binary transfer and distillation are established

Transfer Learning with Binary Neural Networks (2017) freezes a binary ImageNet feature extractor and retrains later software layers. Training Binary Neural Networks with Knowledge Transfer improves binary students using full-precision teachers. The recent fully-binary continual-learning work also uses semi-supervised pretraining to produce transferable binary representations.

Therefore, a fixed DLGN feature extractor with a new dense head is an important baseline, not a sufficient paper.

### 2.6 CNN-to-logic conversion is established under constraints

| Method | What is converted or learned | Why it is not arbitrary CNN-to-DLGN transfer |
|---|---|---|
| LUTNet | BNN/pruned binary computation is expanded into trainable LUTs | Begins from a binarized/pruned network and targets FPGA LUTs |
| LogicNets | Sparse, low-fan-in quantized neurons are enumerated as truth tables | Network must be designed so truth-table fan-in is tractable |
| TTNet | Truth-table-compatible CNN blocks | Architecture is built for conversion from the beginning |
| NullaNet Tiny | DNN operations are transformed into fixed Boolean expressions | Conversion is a specialized logic-synthesis flow, not DLGN checkpoint transfer |
| NeuraLUT/PolyLUT | Small learned functions or polynomials are hidden in LUTs | The LUT-native architecture is trained with specific constraints |

An ordinary ResNet cannot be exactly mapped to a compact two-input DLGN merely by copying weights. At finite input precision it represents a Boolean function in principle, but enumerating or synthesizing the complete function is generally intractable and may produce an enormous circuit.

### 2.7 What TorchLogix already makes possible

The local [TorchLogix repository](../repos/torchlogix/README.md) does support convolutional logic gate networks. This conclusion is based on implementation and test evidence, not only package documentation.

| Capability | Local evidence | Consequence for transfer learning |
|---|---|---|
| Trainable 2D logic convolution | [LogicConv2d implementation](../repos/torchlogix/src/torchlogix/layers/conv.py) | A student can match spatial teacher stages rather than flattening all features |
| Trainable 3D logic convolution | [LogicConv3d implementation](../repos/torchlogix/src/torchlogix/layers/conv.py) | Later extension to video, volumetric, or temporal applications is possible |
| Local shared logic kernels | Receptive field, stride, padding, channels, number of kernels, and logic-tree depth are explicit layer arguments | ResNet/VGG/MobileNet stage resolutions can be approximated with logic blocks |
| LUT rank and parameterization choices | Raw rank-2 gates plus higher-rank WARP/Light paths | Teacher transfer can be evaluated across logic capacity and parameter reduction |
| Image binarization | Fixed, soft, and learnable threshold modules | Quantized image inputs and teacher activation codes can share threshold machinery |
| Convolutional benchmarks | [MNIST and CIFAR-10 CLGN models](../repos/torchlogix/src/torchlogix/models/conv.py) | A CIFAR-10 transfer pilot can reuse an existing architecture |
| Hardened execution and export | Export mode plus [Circuit IR](../repos/torchlogix/src/torchlogix/circuit.py), simplification, C, and Verilog generation | The final student remains a real logic circuit rather than a teacher-dependent hybrid |
| Verification tests | [2D CLGN tests](../repos/torchlogix/tests/test_clgn.py), [3D CLGN tests](../repos/torchlogix/tests/test_clgn_3d.py), and [circuit tests](../repos/torchlogix/tests/test_circuit.py) | Shape, hard behavior, export, and compilation have executable coverage |

The README provides a four-stage CIFAR-10 convolutional LGN and reports about 70% discrete accuracy for its medium configuration before the stronger augmentation/distillation recipe. This is enough for a transfer-learning starting point, although it is not yet a competitive final result.

### 2.8 TorchLogix limitations that define the research work

TorchLogix does **not** currently provide direct support for importing a quantized CNN checkpoint as a logic model.

1. Convolutional connections are fixed after initialization; learnable connections are implemented only for dense layers.
2. The provided experiment loader supports MNIST and CIFAR-10, not CIFAR-100 or ImageNet.
3. Each convolutional logic kernel emits a one-bit feature; INT8 activation tensors require an explicit code interface.
4. There is no TorchVision, Brevitas, QONNX, or ONNX teacher adapter.
5. Standard Conv--BN--ReLU, depthwise convolution, pointwise convolution, and arithmetic residual addition are not imported or reproduced directly.
6. A residual logic block exists, but it uses relaxed OR instead of ResNet addition and has no dedicated test or reference model in the repository.
7. The convolutional export implementation explicitly unpacks two LUT inputs; rank-4/rank-6 training and table extraction exist, but equivalent circuit export is not demonstrated by the current tests.
8. Circuit export supports TorchLogix's hardened Boolean path; it is not a compiler for arbitrary quantized PyTorch operators.

These are manageable limitations for functional transfer. They rule out claiming automatic structural conversion of ResNet or MobileNet. The practical contribution is to add training-time teacher adapters and transfer losses while leaving TorchLogix's deployable student path unchanged.

## 3. Direction 1: LogicMorph-DLGN

### 3.1 Research question

Can a trained hard DLGN be transformed into a larger child DLGN that initially computes exactly the same function, then reaches a better target hard accuracy in fewer updates and with less hyperparameter sensitivity than a child trained from scratch?

### 3.2 Parent representation

Let the parent contain layers

$$
h^{(l)}_i =
T^{(l)}_i
\left(
h^{(l-1)}_{M^{(l)}_{i,1}},
\dots,
h^{(l-1)}_{M^{(l)}_{i,k_l}}
\right),
$$

where $T_i^{(l)}$ is a hard truth table and $M_i^{(l)}$ contains source indices. The parent checkpoint therefore supplies:

- exact hard functions;
- graph topology;
- optional soft gate/table proxies;
- encoder thresholds;
- output grouping/head state.

LogicMorph defines transformations over this representation.

### 3.3 Depth morphism

Insert a layer of copy gates between two trained layers:

$$
\widetilde h_i=T_{\mathrm{copyA}}(h_i,r_i)=h_i.
$$

The second input $r_i$ is ignored by the hard copy-A truth table. The downstream source indices are remapped to $\widetilde h_i$. The parent hard function is preserved exactly.

To add useful capacity, divide the inserted layer into:

- **carrier channels:** exact copy gates that preserve inherited features;
- **growth channels:** trainable gates connected to carrier and earlier features.

The old head initially reads only carrier channels. A target head or later child block can read both.

### 3.4 Width morphism

There are two exact variants.

**Inactive growth**

- Copy every parent node unchanged.
- Add new nodes initialized as constants, copies, or diverse low-cost gates.
- Keep the old output cone disconnected from the new nodes.

**Duplicate-and-split fanout**

- Duplicate selected parent gates with the same inputs and truth table.
- Redirect part of the original fanout to the duplicate.
- Because both nodes initially compute the same bit, the complete function is unchanged.
- During target training, symmetry is broken and the two copies can specialize.

The second variant is the Boolean analogue of a function-preserving width morphism and may also reduce fanout/routing pressure.

### 3.5 Fan-in morphism

BitLogic finds fan-in to be a major accuracy lever. A trained $k$-input LUT can be lifted into a larger $k+r$ input LUT by initially ignoring the new pins:

$$
T'(x_1,\dots,x_k,z_1,\dots,z_r)
=T(x_1,\dots,x_k).
$$

Every old truth-table row is replicated across the new input combinations. The child initially computes exactly the parent function, but target training can use the new pins.

This transformation supports, for example:

- 2-input DLGN gate to LUT4;
- LUT4 to LUT6 on FPGA;
- fixed source routes plus newly trainable candidate pins.

This arity morphism is particularly interesting because it combines checkpoint transfer with BitLogic's strongest architectural axis.

### 3.6 Topology morphism

Embed the parent graph as a subgraph of the child through an injective node map $\phi$:

$$
\widetilde M_{\phi(i),j}=\phi(M_{i,j}),
\qquad
\widetilde T_{\phi(i)}=T_i.
$$

All inherited edges and truth tables remain intact. New edges are allowed only into growth nodes during the first training phase. This gives a precise test of where target improvement comes from:

- inherited gate functions;
- inherited connectivity;
- added depth/width/fan-in;
- later inherited-gate updates.

### 3.7 Soft initialization

Compare two source-state assumptions.

**Full checkpoint transfer**

Copy the real-valued source proxies and optimizer-independent parameters.

**Hard-only transfer**

Reconstruct each source truth-table proxy with commitment magnitude $\rho$:

$$
\theta_r^{(0)}=
\begin{cases}
+\rho,&T_r=1,\\
-\rho,&T_r=0.
\end{cases}
$$

Hard-only transfer is more deployable because a serialized logic circuit is enough. Full checkpoint transfer may optimize faster and serves as an upper reference.

Do not copy optimizer moments by default. If they are transferred, report their storage and ablate them separately.

### 3.8 Target adaptation schedule

Use a staged schedule:

1. **Morph:** create the exact child and verify bit-for-bit source agreement.
2. **Grow:** freeze inherited gates and train the new head plus growth channels.
3. **Unlock:** selectively unfreeze inherited gates with low target loss gradients or low source sensitivity.
4. **Refine:** optionally fine-tune the complete child with a small learning rate and hard-forward training.

A suitable target objective is

$$
\mathcal{L}=
\mathcal{L}_{\mathrm{target}}
+\lambda_a\mathcal{L}_{\mathrm{anchor}}
+\lambda_T\sum_{g\in\mathcal{I}}
\mathbb{E}\!\left[d_H(T_g,T_g^{\mathrm{source}})\right],
$$

where $\mathcal{I}$ contains inherited gates. The anchor term can preserve source outputs on a small source buffer or preserve inherited binary features on target images. If source data are unavailable, use only structural freezing and the truth-table trust region.

### 3.9 Transfer-aware binary evidence bus

Flat DLGNs and GroupSum tend to produce class-specific final evidence. Introduce a compact bus of $m$ binary or small-integer features before the task head:

$$
e(x)=B_{\mathrm{bus}}(h(x))\in\{0,1\}^{m}
\quad\text{or}\quad
e(x)\in\mathbb{Z}^{m}.
$$

During source training, regularize the bus for:

- balanced bit activation;
- low redundancy between bits;
- consistency under augmentation;
- source discrimination.

The target replaces only the task head initially:

$$
s_{\mathrm{target}}=W_{\mathrm{target}}e(x).
$$

This generalizes BitLogic's class-dependent $C\times C$ head. It should be treated as an optional transfer-aware architecture, because adding it makes the source training protocol different from an ordinary pretrained DLGN.

### 3.10 Why CIFAR-10 to CIFAR-100 is a difficult but useful test

CIFAR-10 and CIFAR-100 have the same input dimensions and broad natural-image domain, but their labels are different. A ten-class GroupSum head is not reusable, and a source network may have discarded fine-grained information that CIFAR-100 requires.

Use CIFAR-10 to CIFAR-100 as a stress test, not as the only transfer benchmark. Include:

1. **same-task growth:** shallow CIFAR-10 parent to deeper CIFAR-10 child, which isolates architecture-training acceleration;
2. **related-label transfer:** CIFAR-100 coarse superclasses to fine classes, or one subset of CIFAR-100 classes to another;
3. **cross-dataset transfer:** CIFAR-10 to CIFAR-100.

Keep input encoding identical across parent and child. If thresholds or bit depth change, the first logic layer no longer receives the same Boolean function inputs and exact graph transfer is lost.

## 4. LogicMorph novelty boundary

### Versus Net2Net/network morphism

Net2Net preserves arithmetic neural functions through weight transformations. LogicMorph operates on:

- exact discrete truth tables;
- ignored LUT pins;
- copy gates that synthesize to wires;
- duplicated Boolean fanout;
- explicit connection-index embeddings;
- hard/soft dual checkpoints.

The paper needs more than the observation that copy gates implement identity. It must provide a set of composable morphisms, prove hard-function preservation, and demonstrate DLGN training acceleration.

### Versus DLGN residual initialization

Residual initialization starts a random layer near copy behavior. LogicMorph:

- preserves a trained parent graph exactly;
- transfers learned truth tables and routes;
- grows width and fan-in as well as depth;
- initializes target training from a functioning hard circuit.

### Versus fully trainable DLGN connections

Learned-connectivity work optimizes routes from an initial candidate space. LogicMorph preserves a trained route subgraph and adds controlled capacity. Bounded learned routing remains a target-training component and baseline.

### Versus binary-network transfer

BNN transfer usually freezes a binary feature extractor and replaces later software layers. LogicMorph changes and grows the logic circuit itself while preserving an exact parent subfunction and tracking changed truth-table bits.

## 5. LogicMorph experimental plan

### 5.1 Parent and child models

- Parent depths: 2, 4, and optionally 6 layers.
- Child depths: parent depth plus 2, 4, or 6 layers.
- Width growth: 1x to 2x.
- Fan-in growth: 2 to 4, and optionally 4 to 6.
- Gate parameterizations: Light/IWP and hard Gumbel first; original soft DLGN only as context.

Do not sweep all combinations. Select one depth morph, one width morph, and one fan-in morph after a small pilot.

### 5.2 Required baselines

- child trained from scratch with the same final architecture and optimizer budget;
- parent fine-tuned without growth;
- frozen parent trunk plus new head;
- naive parameter copy without an exact graph morphism;
- DLGN residual/identity initialization from scratch;
- LogicMorph with only gate functions transferred;
- LogicMorph with only connections transferred;
- Net2Net-style conceptual baseline on a matched BNN or MLP;
- source teacher logit distillation into the child;
- fully-binary fixed-feature transfer where a compatible model is available.

### 5.3 Metrics

**Training acceleration**

- optimizer steps and wall time to fixed hard-accuracy thresholds;
- area under the target hard-accuracy learning curve;
- time to 90%, 95%, and 100% of the scratch child's final accuracy;
- total GPU-hours to the best validation checkpoint;
- peak memory and trainable parameter count during each stage.

**Final model**

- hard target accuracy and soft-to-hard gap;
- source accuracy before and after target adaptation;
- gate/LUT count, connection count, and model bits;
- changed inherited truth-table bits;
- fraction of inherited routes retained;
- carrier/growth gate utilization.

**Transfer quality**

- linear/head-only probe accuracy on frozen source features;
- negative-transfer rate across source/target pairs;
- gain over scratch at fixed update count;
- performance versus target-data fraction;
- class-order and seed variance.

**Hyperparameter robustness**

- success rate over a fixed learning-rate and temperature grid;
- variance of best hard accuracy;
- number of failed/diverged runs;
- tuning trials required to reach a declared target.

Transfer learning adds freezing schedules, anchor weights, and commitment $\rho$. It should not be claimed to simplify hyperparameters unless it succeeds over a broader fixed grid than scratch training.

### 5.4 Accounting source pretraining

Report two costs:

1. **marginal target cost:** source checkpoint already exists;
2. **amortized cost:** source training divided across the number of target models that reuse it.

It is misleading to claim a total training reduction while excluding a source model trained for only one target.

### 5.5 Kill criteria

- The exact child must achieve 100% prediction and internal carrier-bit agreement with the parent before target training.
- Continue only if LogicMorph reaches a fixed target hard accuracy at least 2x faster in steps or wall time than scratch on two transfer settings.
- Final hard accuracy must be within 0.5 points of, or better than, the fully trained scratch child.
- Stop if only the new head learns while growth gates remain unused; that result supports ordinary fixed-feature transfer rather than LogicMorph.
- Stop if CIFAR-10 to CIFAR-100 shows negative transfer and the related-label benchmark also fails.

## 6. Direction 2: QBridge-DLGN -- quantized CNN to convolutional DLGN

### 6.1 Feasibility verdict

TorchLogix makes spatial, block-aligned functional transfer feasible:

> Use an established quantized CNN as a frozen teacher, expose its native integer activation codes at stage boundaries, train TorchLogix convolutional logic blocks to reproduce selected code bits, and discard the teacher and all adapters after training.

This is **not** a one-to-one weight conversion. It is a transfer method from a high-fan-in arithmetic QNN into a small-fan-in Boolean circuit. The final model is a normal hardened TorchLogix circuit.

The quantized teacher is preferable to an arbitrary floating-point teacher for the main hypothesis because its intermediate representation is already finite and deployment-oriented. A full-precision teacher remains necessary as a baseline.

### 6.2 Which famous quantized CNNs should be supported?

Support should mean that a teacher adapter can extract logits, integer activation codes, scale, zero point, bit width, and named stage boundaries. It should **not** mean that TorchLogix can execute the teacher or import its operators.

| Priority | Teacher family | Initial models | Reason |
|---:|---|---|---|
| 1 | TorchVision INT8 | ResNet-18 and MobileNetV2 | Famous, pretrained, accessible through standard PyTorch, and structurally different enough to test generality |
| 2 | Brevitas QAT | ResNet-18 or MobileNetV2 at W4A4/W8A8 | Exposes quantization metadata during training and permits target-dataset QAT |
| 3 | FINN/Brevitas few-bit | CIFAR-10 CNV/VGG-like W1A1 or W2A2 | Closest representation match and easiest first transfer/debugging target |
| 4 | TorchVision INT8 | ShuffleNetV2 and MobileNetV3 | Tests channel shuffle, depthwise blocks, and mobile design |
| 5 | TorchVision INT8 | ResNet-50/ResNeXt | Scaling stress test only after smaller models work |

TorchVision currently provides pretrained INT8 variants of ResNet, ResNeXt, MobileNetV2/V3, ShuffleNetV2, GoogLeNet, and InceptionV3. Supporting all of them in software is possible through a common hook interface, but a paper should evaluate only two architecturally different primary teachers plus one few-bit pilot.

The official TorchVision quantized weights are CPU-inference models. Use them for one-time feature caching and compatibility tests, not in the online GPU student loop. Brevitas fake-quant/QAT teachers are better for repeated target-dataset experiments and GPU feature generation.

For CIFAR-100, use a teacher trained or quantization-aware fine-tuned on CIFAR-100. An ImageNet INT8 checkpoint used without target adaptation tests representation transfer, not teacher-to-student task distillation, and must be labeled separately.

### 6.3 Teacher adapter layer

Implement a small training-only abstraction:

~~~text
TeacherAdapter
  named_stages() -> ordered stage names and resolutions
  forward_codes(x) -> integer activations plus quantization metadata
  forward_logits(x) -> class logits
  architecture_metadata() -> channels, stride, bit width, operator family
~~~

Three adapters are enough for the first paper:

1. **TorchVisionINT8Adapter:** register hooks at quantized stage outputs; record `int_repr()`, per-tensor/per-channel scales, and zero points.
2. **BrevitasAdapter:** read integer values, scales, zero points, signedness, and bit width from `QuantTensor` outputs.
3. **GenericPyTorchQNNAdapter:** accept user-selected modules and a callback that converts their output to integer codes.

QONNX import is attractive for later interoperability with FINN, but it is unnecessary for the learning claim and would add toolchain risk. Cache the teacher outputs after verifying that cached and online codes are identical.

### 6.4 Stage correspondence without architecture imitation

The student should match stage resolutions, not individual arithmetic operators.

| Teacher | Suggested interfaces | TorchLogix student interpretation |
|---|---|---|
| ResNet-18 | stem, layer1, layer2, layer3, layer4 | One or more LogicConv2d blocks per resolution; optional logic residual ablation |
| MobileNetV2 | outputs after each resolution-changing inverted-residual group | Distill each depthwise-plus-pointwise group into a logic stage |
| ShuffleNetV2 | stage2, stage3, stage4 | Ignore explicit shuffle semantics; match the stage code and test teacher-guided channel groups |
| VGG/CNV | each pooling boundary | Direct correspondence with LogicConv2d plus OrPooling2d |

This avoids an invalid claim that a depthwise convolution or residual addition has been copied. The teacher supplies behavior and local saliency; the student chooses its own Boolean implementation.

### 6.5 Quantization-code interfaces

For a teacher activation $a_l$, let its quantized integer code be

$$
q_l=
\operatorname{clamp}
\left(
\operatorname{round}(a_l/s_l)+z_l,
q_{\min},q_{\max}
\right).
$$

Convert each selected teacher channel into Boolean targets. The recommended default is an ordinal threshold or thermometer code:

$$
b_{l,c,r}(u,v)=
\mathbb{1}
\left[q_{l,c}(u,v)\ge \tau_{l,c,r}\right],
\qquad r=1,\dots,R_{l,c}.
$$

Compare three code families:

1. **native bit planes:** compact and exactly reconstruct the integer code, but adjacent values can differ in many bits;
2. **thermometer thresholds:** preserve order and are naturally compatible with logic outputs, but use more channels;
3. **learned balanced thresholds:** select a small subset of code boundaries with high label relevance and approximately balanced activation.

The teacher's activation bit width does not have to equal the number of student logic kernels. Allocate more interface bits to sensitive stages/channels and merge redundant bits. This creates a gate-budget allocation problem rather than blindly reproducing every INT8 bit.

If student and teacher channel counts differ, use a training-only sparse assignment or $1\times1$ projection before thresholding the teacher. Remove this bridge at inference; it defines targets and is not part of the student.

### 6.6 Teacher-guided convolutional connections

TorchLogix currently samples convolutional input positions randomly and then fixes them. Quantized CNN weights provide a principled initialization.

For teacher output channel $o$ and local input coordinate $(c,\Delta u,\Delta v)$, rank candidate leaves with, for example,

$$
S_{o,c,\Delta u,\Delta v}
=
|w^{q}_{o,c,\Delta u,\Delta v}|
\sqrt{\operatorname{Var}(q^{\mathrm{in}}_{c})}
$$

or a calibration-set gradient sensitivity. Assign TorchLogix kernels to teacher channel/code groups and choose diverse high-score leaves for their first tree level. Preserve channel locality for depthwise teachers and use cross-channel scores for pointwise teachers.

Implement this as a new fixed connection initializer, conceptually `teacher-saliency`, alongside the current `random` and `random-unique` modes. The inherited indices remain fixed during the first experiment, so no learnable-convolutional-routing implementation is required.

Possible logic-function initializations are:

- residual/copy initialization, which TorchLogix already supports;
- random truth tables;
- calibration-fitted truth tables, selected by the teacher-code error observed for each local input pattern.

Teacher-guided routing is a stronger DLGN-specific differentiator than ordinary feature distillation. It transfers information from quantized convolutional weights into the discrete logic graph while still allowing the LUT functions to be learned.

### 6.7 Progressive hard-stage curriculum

Use TorchLogix's convolutional blocks in a stagewise curriculum:

1. Freeze the target-trained QNN and cache selected integer stage codes and logits.
2. Train the first logic stage from the image bits to the first teacher-code interface.
3. Switch that stage to hard-forward operation and train the next logic stage from its outputs.
4. Repeat at each resolution boundary.
5. Attach GroupSum or a compact target head and add logit distillation.
6. Jointly fine-tune the complete student with hard-forward or hard-Gumbel training.
7. Export the student through `Circuit.from_model`, simplify it, and verify predictions against TorchLogix evaluation mode.

Compare progressive training against end-to-end QBridge training. Stagewise pretraining is useful only if it improves final hardened accuracy, time to accuracy, or run-to-run success rate after all pretraining cost is counted.

### 6.8 Transfer objective

Let $s_l(x)$ be the student Boolean map aligned with teacher code $b_l^T(x)$. Optimize

$$
\begin{aligned}
\mathcal{L}={}&
\mathcal{L}_{\mathrm{CE}}
+\lambda_{\mathrm{KD}}T^2
D_{\mathrm{KL}}\left(p_T^{(T)}\|p_S^{(T)}\right)\\
&+\sum_l\lambda_l
\operatorname{WBCE}\left(s_l,b_l^T\right)
+\lambda_R\mathcal{L}_{\mathrm{Hamming-relation}}
+\lambda_H\mathcal{L}_{\mathrm{soft/hard}}.
\end{aligned}
$$

Weighted BCE is needed because native quantization bits may be highly imbalanced. The relation term preserves selected pairwise similarities in Hamming space without requiring one-to-one channel equality.

The teacher, adapters, projections, cached features, and auxiliary stage losses disappear after training. Their memory and preprocessing costs must still be reported.

### 6.9 Residual and mobile teachers

Do not require the student to reproduce the teacher macroarchitecture.

- For ResNet, first distill outputs after complete residual stages into a plain convolutional LGN.
- Treat TorchLogix's `ResidualLogicBlock` as an optional student ablation only after adding dedicated train/eval/export tests.
- Do not describe relaxed OR as equivalent to arithmetic residual addition.
- For MobileNet, distill a depthwise-plus-pointwise group as one behavior rather than converting each operator.
- For ShuffleNet, test whether teacher-guided channel grouping helps, but do not explicitly implement shuffle unless the ablation justifies it.

This keeps the first contribution focused on transfer rather than rebuilding every famous CNN primitive in Boolean form.

### 6.10 Full-precision BID remains a baseline

For a floating-point teacher feature $z_l^T$, retain the original Boolean Interface Distillation target

$$
b_l^T(x)=
\mathbb{1}
\left[P_l\operatorname{Pool}(z_l^T(x))>\tau_l\right].
$$

This baseline determines whether native QNN codes are actually more attainable than learned thresholds over real-valued features. If QBridge does not outperform this baseline at equal teacher accuracy and transfer cost, the quantized-teacher argument is unsupported.

## 7. QBridge-DLGN novelty boundary

### Versus ordinary knowledge distillation

Logit KD supervises only the class distribution. QBridge transfers spatial integer activation codes, initializes local logic connections from quantized kernel saliency, and measures the resulting hardened circuit.

### Versus quantized feature distillation

Quantized Feature Distillation and related methods already show that quantized representations can guide quantized neural students. QBridge's student is not a lower-bit convolutional MAC network. It learns arbitrary LUT functions and a sparse Boolean graph, uses the teacher to initialize discrete receptive-field leaves, and exports a pure circuit. Quantized features alone are therefore a baseline, not the novelty.

### Versus BNN knowledge transfer

BNN students retain XNOR/popcount convolution and usually mirror teacher channels or attention maps. QBridge distills arithmetic stages into logic trees with separately learned Boolean functions and connections. A matched BNN student remains necessary to show whether the method benefits DLGNs specifically.

### Versus LogicNets, LUTNet, TTNet, and NullaNet

Those methods constrain or transform a source architecture so its units can be enumerated or synthesized. QBridge accepts an off-the-shelf quantized teacher and transfers sampled behavior; it does not claim structural or formal equivalence. The deployed graph is learned by TorchLogix.

### Versus TorchLogix/WARP and convolutional DLGNs

TorchLogix provides the student architecture, parameterizations, hardening, and circuit export. It does not provide pretrained-QNN adapters, quantization-code losses, teacher-guided convolutional connections, progressive stage transfer, or a transfer evaluation. Those additions form the proposed contribution.

### Versus the earlier SketchDistill idea

QBridge makes the earlier concept concrete by adding:

1. native integer teacher codes rather than only arbitrary binary sketches;
2. a working convolutional logic student and circuit exporter;
3. architecture adapters for standard quantized models;
4. teacher-guided graph initialization;
5. stagewise hard training and deployment verification.

Without the routing initialization or another DLGN-specific mechanism, this can still be judged as ordinary intermediate-feature distillation with thresholding.

## 8. QBridge-DLGN experiments

### 8.1 Main hypotheses

- **H1:** quantized teacher codes improve hardened CLGN accuracy or time-to-accuracy over logits and floating-point hints.
- **H2:** teacher-saliency connections outperform random fixed convolutional connections at the same gate budget.
- **H3:** progressive hard-stage training reduces failed runs or soft-to-hard degradation compared with end-to-end distillation.
- **H4:** the method transfers across at least one residual teacher and one mobile/depthwise teacher.

### 8.2 Datasets and teachers

**Pilot**

- CIFAR-10;
- FINN/Brevitas CNV W1A1 or W2A2 teacher;
- target-trained quantized ResNet-18 if a reproducible checkpoint is available;
- existing `ClgnCifar10Small` and `ClgnCifar10Medium` students.

**Primary evidence**

- CIFAR-100;
- target-trained ResNet-18 W8A8 and W4A4;
- target-trained MobileNetV2 W8A8 or W4A4;
- a new TorchLogix CIFAR-100 CLGN differing only in input/output dimensions and declared stage widths.

**Optional off-the-shelf demonstration**

- an ImageNet subset or transfer dataset;
- pretrained TorchVision INT8 ResNet-18 and MobileNetV2/ShuffleNetV2;
- no full ImageNet CLGN claim until the repository scales reproducibly.

Use a common QAT implementation, preferably Brevitas, for target-trained teachers so activation codes and bit widths are observable. Use official TorchVision INT8 weights for the off-the-shelf compatibility demonstration.

### 8.3 Students

- rank-2 TorchLogix convolutional LGN with raw parameterization;
- the same rank-2 architecture with WARP and Light parameterizations;
- two gate budgets per dataset;
- optional plain versus residual logic macroarchitecture;
- a same-budget BNN/QNN student as an adjacent baseline.

The current TorchLogix CIFAR-10 model should be the starting point. Keep the main circuit results at LUT rank 2 until rank-4/rank-6 convolutional export equivalence is implemented and tested. Do not first create a one-to-one ResNet clone; stage resolution alignment is sufficient.

### 8.4 Required baselines

- TorchLogix CLGN trained from scratch;
- label smoothing;
- logit KD from the same QNN teacher;
- dequantized real-valued feature matching;
- full-precision teacher BID at matched teacher accuracy where possible;
- quantized-code loss with random connections;
- teacher-saliency connections without code distillation;
- random versus random-unique versus teacher-saliency connections;
- native bit-plane versus thermometer versus learned threshold codes;
- end-to-end versus progressive stage training;
- same-sized BNN/QNN student with standard KD or quantized feature distillation;
- LogicMorph from a DLGN checkpoint;
- LogicNets/TTNet/LUTNet result or implementation only where dataset and budget are genuinely comparable.

### 8.5 Metrics

**Learning**

- hardened top-1 accuracy and soft-to-hard gap;
- steps, wall time, and GPU-hours to a fixed hard accuracy;
- area under the hard-accuracy learning curve;
- failed-run rate and seed variance;
- stage-code accuracy, balanced accuracy, and Hamming distance;
- teacher/student representational similarity in Hamming space.

**Model and deployment**

- total gates/LUTs, logic depth, connection count, and serialized model bits;
- C/Verilog circuit size after `Circuit.simplify()`;
- inference latency and throughput of the compiled TorchLogix circuit;
- teacher-free inference verification;
- optional FPGA LUT/FF/BRAM/DSP and frequency.

**Transfer overhead**

- teacher checkpoint and feature-cache bytes;
- cache generation time;
- training-only adapter/projection parameters;
- target labels and calibration samples;
- extra pretraining steps for progressive stages.

### 8.6 Essential ablations

- full-precision versus INT8 versus 4-bit versus 1/2-bit teacher at controlled teacher accuracy;
- teacher architecture: ResNet-18 versus MobileNetV2;
- logits only, codes only, routing only, and all components;
- stage interface locations and number of interfaces;
- interface code family and bit budget;
- fixed uniform versus learned per-channel thresholds;
- teacher-saliency score based on weights, activations, or gradients;
- channel grouping for depthwise/mobile teachers;
- progressive versus end-to-end student training;
- soft-forward, hard-forward, and hard-Gumbel training;
- raw, WARP, and Light logic parameterizations.

### 8.7 Kill criteria

- QBridge must beat tuned logit KD by at least 0.5 hard-accuracy points on two student budgets, or reach the same accuracy at least 1.5x faster.
- Teacher-saliency routing must improve over random-unique connections on at least two teacher/student pairs; otherwise remove it from the central claim.
- A second teacher family must retain at least half of the gain observed with the first.
- The gain must remain after teacher-cache generation and progressive pretraining are counted.
- Stop the quantized-code claim if full-precision BID performs equally well at matched teacher quality and cost.
- Stop the broad compatibility claim if the adapter requires architecture-specific student code beyond named hook locations and stage metadata.

## 9. Direction 3: direct CNN-to-DLGN structural conversion

### 9.1 What is possible

A bounded pilot can use:

1. quantize CNN inputs, activations, and weights;
2. prune each neuron to small fan-in, ideally no more than six binary inputs;
3. enumerate each neuron's truth table;
4. map LUTs directly or decompose them into two-input gates;
5. initialize a DLGN/LUTN from the converted circuit;
6. fine-tune truth tables and selected routes.

The conversion is exact only for the **quantized, pruned intermediate network**, not for the original floating-point CNN.

TorchLogix's `Circuit` IR does not perform this conversion: it traces models that have already been expressed through TorchLogix's Boolean export path. A separate QONNX/logic-synthesis importer would be required before Compile-and-Refine could consume an external QNN structurally.

### 9.2 Why it is not recommended as the main idea

- LogicNets, TTNet, LUTNet, NullaNet, and related methods already occupy most of this path.
- Fan-in reduction may cause a large accuracy loss before DLGN training begins.
- Decomposing LUTs into two-input gates can create large circuits.
- A logic-synthesis/export/import flow is needed.
- The converted graph may be less trainable than a DLGN initialized directly.

### 9.3 Narrow possible novelty

The only promising differentiator is **compile-and-refine**:

> Use an exactly converted quantized teacher circuit as a functioning initialization, then allow differentiable gate and route refinement to recover accuracy or reduce circuit size.

This would need comparisons against:

- the converted circuit with no refinement;
- DLGN from scratch;
- ordinary KD;
- LogicNets/LUTNet-style training;
- exact logic synthesis and post-training pruning.

Treat this as exploratory unless the required conversion toolchain already exists.

## 10. Combined comparison matrix

| Method | Reuses source gates? | Teacher support | Exact source function initially? | Changes deployed architecture? | Extra inference cost? |
|---|---:|---:|---:|---:|---:|
| Scratch DLGN | No | No | No | New model | No |
| Frozen DLGN + new head | Yes | No | Trunk only | Head | Head dependent |
| Logit KD | No | Any logit teacher | No | New model | No |
| Full-precision BID | No | Any hookable CNN | No | New model | No |
| QBridge-DLGN | No; transfers topology evidence | Hookable INT/few-bit CNN | No | New convolutional logic model | No |
| Residual initialization | No | No | Identity blocks only | New model | No |
| LogicMorph-DLGN | Yes | No | Yes, before task-head replacement | Grows model | No |
| LogicNets/TTNet conversion | Not DLGN gates | No; teacher must be constrained | Yes for constrained quantized model | Converted model | No |
| Compile-and-Refine | Indirectly | Only after quantization/pruning | Yes for intermediate model | Converted and refined | No |

## 11. Practical implementation sequences

### 11.1 LogicMorph sequence

**Week 1: prove exact logic morphisms**

1. Load and harden a small trained DLGN.
2. Implement depth insertion with copy gates.
3. Implement width duplication and split fanout.
4. Verify every intermediate/output bit on the complete MNIST test set.
5. Implement 2-input to LUT4 truth-table lifting on a small model.

**Week 2: same-task acceleration**

1. Grow a shallow CIFAR-10 DLGN.
2. Compare scratch, residual initialization, and LogicMorph.
3. Measure hard accuracy versus steps and wall time.
4. Select one growth transformation.

**Weeks 3--4: target-task transfer**

1. Replace the source head for CIFAR-100 or a related-class task.
2. Compare frozen, growth-only, selective unlock, and full fine-tuning.
3. Add the evidence bus only if frozen features are clearly limiting.
4. Run three pilot seeds before a larger matrix.

**Weeks 5--6: robustness and final evidence**

1. Run the central comparison over five seeds.
2. Evaluate target-data fractions and a fixed hyperparameter grid.
3. Report inherited gate/route retention and exact model-state accounting.

### 11.2 QBridge sequence

**Week 1: adapter and reproducibility**

1. Reproduce the TorchLogix CIFAR-10 medium CLGN hard accuracy.
2. Add one TorchVision INT8 adapter and one Brevitas/few-bit adapter.
3. Cache logits, integer codes, scales, and zero points at two stage boundaries.
4. Add cache-versus-online equivalence tests.

**Week 2: minimal transfer test**

1. Distill one final-stage code from a quantized ResNet-18 into the existing CLGN.
2. Compare scratch, logit KD, real-valued hints, and native quantized codes.
3. Stop or revise if quantized codes do not approach the QBridge kill criterion.

**Weeks 3--4: DLGN-specific contribution**

1. Implement the teacher-saliency fixed connection initializer.
2. Add progressive hard-stage training.
3. Compare bit-plane, thermometer, and learned threshold interfaces.
4. Export and simplify every hardened student used in the central table.

**Weeks 5--6: breadth and robustness**

1. Add MobileNetV2 as the second teacher family.
2. Run central comparisons over at least five seeds.
3. Report cache, training, circuit, and inference costs.
4. Move to CIFAR-100 only after the CIFAR-10 mechanism is positive.

Do not build direct CNN structural conversion or QONNX import until either LogicMorph or QBridge has produced a positive result.

## 12. Risks

### Source features are too class-specific

The standard GroupSum output is explicitly class-specific, and random DLGN wiring may learn brittle source evidence.

**Mitigation:** transfer an earlier logic layer, use the evidence bus, or pretrain with augmentation consistency. Do not transfer source GroupSum as the feature representation.

### Deeper child still suffers gradient failure

Exact identity guarantees the initial function but does not guarantee that new gates learn.

**Mitigation:** use hard-Gumbel or IWP, residual initialization for growth channels, auxiliary targets, and progressive unfreezing.

### New capacity is ignored

A child may keep using carrier channels because they already solve the source task.

**Mitigation:** target-task head reads growth channels; add temporary auxiliary losses; measure utilization. Avoid forcing growth through arbitrary regularization if target accuracy does not benefit.

### Transfer increases rather than reduces hyperparameters

Freezing stages, commitment magnitude, and distillation weights add choices.

**Mitigation:** publish one robust default schedule and evaluate a fixed hyperparameter grid. Do not claim easier tuning from a single hand-tuned run.

### QBridge is only generic feature distillation

**Mitigation:** require a gain from native quantization codes and teacher-saliency logic connections over logit KD, real-valued hints, and random binary projections. If routing transfer does not help, narrow the claim rather than presenting thresholding as new.

### Quantized activation bits are imbalanced or unattainable

High-order bits can be almost constant, while a compact logic stage may not reproduce all low-order detail.

**Mitigation:** compare weighted bit-plane loss with thermometer and learned balanced thresholds; allocate interface bits by sensitivity rather than teacher precision alone.

### Popular teacher support becomes architecture-specific engineering

Hooks and stage names differ across ResNet, MobileNet, and ShuffleNet implementations.

**Mitigation:** keep architecture knowledge inside `TeacherAdapter`; require the TorchLogix student and loss code to consume only stage tensors and metadata. Treat QONNX as future work.

### Teacher topology is misleading

Large quantized weights do not necessarily identify the inputs most useful to a logic tree after nonlinear accumulation.

**Mitigation:** compare weight, activation, and gradient saliency against random-unique routing. Remove teacher-guided routing from the paper if it is not robust.

### Final model has no deployment advantage

Transfer learning reduces training cost, not automatically gate count or inference latency.

**Mitigation:** keep the child architecture identical to the scratch baseline. Any inference benefit must be measured separately and should not be implied by faster training.

## 13. Candidate papers

### Recommended first paper

> **LogicMorph-DLGN: Function-Preserving Growth and Transfer of Differentiable Logic Gate Networks**

Central claims:

1. composable depth, width, fan-in, and topology morphisms exactly preserve the hardened parent circuit;
2. growth-first and selective-unlock training accelerates larger DLGN optimization;
3. inherited Boolean features transfer across related tasks while requiring fewer target updates and tuning trials than scratch training.

### Quantized-CNN transfer paper

> **QBridge-DLGN: Transferring Quantized CNNs into Hardened Convolutional Logic Circuits**

Central claims:

1. standard INT8/few-bit ResNet and MobileNet teachers are exposed through quantization-code interfaces rather than structurally misrepresented as small LUTs;
2. quantized stage codes plus teacher-guided local connections improve hardened TorchLogix accuracy or training speed over conventional KD and random routing;
3. progressive hard-stage training produces a simplified teacher-free C/Verilog logic circuit with no inference adapter.

## 14. Primary references

### Function-preserving growth and adaptation

- [Net2Net: Accelerating Learning via Knowledge Transfer](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/)
- [Network Morphism](https://proceedings.mlr.press/v48/wei16.html)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Side-Tuning](https://arxiv.org/abs/1912.13503)

### Binary transfer and distillation

- [Transfer Learning with Binary Neural Networks](https://arxiv.org/abs/1711.10761)
- [Training Binary Neural Networks with Knowledge Transfer](https://doi.org/10.1016/j.neucom.2018.09.103)
- [Towards Experience Replay for Class-Incremental Learning in Fully-Binary Networks](https://arxiv.org/abs/2503.07107)
- [Collaborative Multi-Teacher Knowledge Distillation for Low-Bit Networks](https://openaccess.thecvf.com/content/WACV2023/html/Pham_Collaborative_Multi-Teacher_Knowledge_Distillation_for_Learning_Low_Bit-Width_Deep_Neural_WACV_2023_paper.html)

### Quantized CNN teachers and tooling

- [TorchVision pretrained quantized models](https://docs.pytorch.org/vision/master/models.html#quantized-models)
- [Brevitas quantization library](https://github.com/Xilinx/brevitas)
- [Brevitas model-definition and PTQ/QAT workflows](https://xilinx.github.io/brevitas/dev/getting_started.html)
- [FINN quantized accelerator examples](https://github.com/Xilinx/finn-examples)
- [Quantized Feature Distillation](https://arxiv.org/abs/2307.10638)
- [Model Compression via Distillation and Quantization](https://arxiv.org/abs/1802.05668)

### CNN and neural-network conversion to LUT/logic

- [LUTNet](https://arxiv.org/abs/1910.12625)
- [LogicNets](https://arxiv.org/abs/2004.03021)
- [NullaNet Tiny](https://arxiv.org/abs/2104.05421)
- [Truth Table Net](https://arxiv.org/abs/2208.08609)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)

### DLGN foundations and current training

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Mind the Gap](https://openreview.net/forum?id=chYXaetMmz)
- [Light Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250)
- [BitLogic](https://openreview.net/forum?id=ZbsSZAfDod)
- [Fully Trainable Deep DLGNs and LUTNs](https://arxiv.org/abs/2607.09399)
- [WARP Logic Neural Networks](https://arxiv.org/abs/2602.03527)
- [Local TorchLogix README](../repos/torchlogix/README.md)
- [Local TorchLogix convolutional layer](../repos/torchlogix/src/torchlogix/layers/conv.py)
- [Local TorchLogix CIFAR-10 models](../repos/torchlogix/src/torchlogix/models/conv.py)
- [Local BitLogic PDF](../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf)
- [Local convolutional DLGN PDF](../pdfs/convolutional_differentiable_logic_gate_networks.pdf)

## Bottom line

For the first transfer-learning project, implement **LogicMorph-DLGN**. Begin with same-task shallow-to-deep growth, because it cleanly proves whether transferred gates accelerate training. Then replace the head and test related-task transfer. The CIFAR-10 to CIFAR-100 experiment should be included, but it should not be the only evidence because the class semantics differ and the source DLGN may discard fine-grained information.

TorchLogix **does support convolutional LGNs**, so the quantized-CNN direction is now concrete. Implement **QBridge-DLGN** with a few-bit CNV teacher for debugging, then use quantized ResNet-18 and MobileNetV2 as the primary famous architectures. Support them through teacher adapters, native activation-code targets, and teacher-saliency fixed connections; do not attempt to reproduce their operators one by one.

Describe QBridge accurately as functional and topological transfer, not exact CNN conversion. Direct structural conversion is possible only after quantization and severe fan-in restriction, where LogicNets, LUTNet, TTNet, and NullaNet already establish much of the approach. The final evidence must come from hardened, simplified, teacher-free TorchLogix circuits.
