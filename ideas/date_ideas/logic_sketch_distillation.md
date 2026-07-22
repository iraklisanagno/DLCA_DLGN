# LogicSketch Distillation: Boolean Guidance from a Full-Precision Teacher

**Source concept:** LogicSketch Distillation in [july_20.md](../july_20.md).

## Narrative

A conventional neural network can solve an image task accurately, but its internal knowledge is expressed as large arrays of real numbers. A DLGN must eventually make every intermediate decision with bits and logic gates. Asking the DLGN to imitate all of the teacher's real-valued features is therefore like asking a student with a yes/no answer sheet to reproduce an essay.

LogicSketch gives the student better flashcards. A pretrained teacher converts important intermediate information into short, balanced binary sketches. During training, temporary probes ask the DLGN to predict these sketches as well as the class label. The probes are removed after training, so the deployed circuit remains a pure DLGN and pays no teacher or auxiliary-head cost.

The research story is that distillation should respect the form of the final machine. Instead of transferring an unattainable floating-point representation, it transfers compact Boolean relationships that a logic circuit can plausibly learn.

## Research question and claim

**Question:** Do teacher-derived binary intermediate targets improve compact hardened DLGNs beyond label training and standard logit distillation?

**Target claim:** At matched deployed gate count, LogicSketch improves hard accuracy and reduces the soft-to-hard gap because its training targets are structurally compatible with Boolean inference.

This project is distinct from [QBridge-DLGN](qbridge_dlgn.md). LogicSketch can use any accurate full-precision teacher and transfers learned binary sketches through temporary probes. QBridge starts from a quantized CNN, transfers its native quantized codes, and studies teacher-guided DLGN connectivity.

## Technical design

Train or load a target-task teacher, initially ResNet-18. Select one teacher feature tensor near the end of the network, apply global pooling if needed, and form a compact sketch:

$$
z_T(x) = \mathbb{1}[R h_T(x) > \tau].
$$

Here, $R$ is a fixed seeded random projection or a small learned projection frozen after fitting, and $\tau$ is a per-bit median estimated only on the training set. Median thresholds keep bits balanced and prevent an easy all-zero solution.

Attach a training-only logic or linear probe to one student stage. The minimum objective is:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{KD}\mathcal{L}_{\mathrm{logit}} + \lambda_S\mathcal{L}_{\mathrm{BCE}}(p_S, z_T).
$$

Start with one student stage, one 64- or 128-bit sketch, and a fixed random projection. Only add a second stage or a learned projection if the minimum method is positive. Remove all sketch probes for hardened inference and confirm that deployed gate count and outputs do not depend on them.

## Implementation plan

### Milestone 1: Teachers and cache

1. Reproduce the TorchLogix student baseline and an accurate ResNet-18 teacher on the same splits and augmentations.
2. Add a teacher-cache command keyed by dataset fingerprint, teacher checkpoint hash, feature name, projection seed, and sketch threshold hash.
3. Cache logits and compact sketches, not full feature maps, for the primary method.
4. Verify sketch balance, duplicate-bit rate, pairwise correlation, and deterministic regeneration on a sample.

### Milestone 2: Student probes

1. Add an optional intermediate-output interface to the TorchLogix model factory.
2. Attach one training-only probe after the selected logic stage.
3. Add CE-only, logit-KD, sketch-only auxiliary, and combined objectives behind configuration flags.
4. Ensure exporting or hard-evaluating the student ignores and removes the probes.

### Milestone 3: Hard-aware training

1. Train with the existing TorchLogix raw/WARP or Light parameterization.
2. Evaluate soft and hard accuracy throughout training.
3. If needed, use a short final hard-aware fine-tuning phase, applied identically to all distillation baselines.
4. Freeze sketch size, loss weights, and probe location using validation data before final seeds.

### Milestone 4: Focused extensions

In this order, test only extensions justified by the previous result:

1. learned balanced projection versus fixed random projection;
2. one sketch versus two hierarchical sketches;
3. a logic-gate probe versus a floating-point training-only probe;
4. teacher ensembles only if a single teacher is clearly insufficient.

Do not begin with end-to-end optimization of a large codebook or many layer pairings. That would obscure the central hypothesis.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md) because `LogicConv2d` is needed for CIFAR and the repository supports intermediate model composition and multiple gate parameterizations. Likely touch points are:

- `repos/torchlogix/src/torchlogix/models/conv.py` for student-stage access;
- `repos/torchlogix/src/torchlogix/layers/conv.py` and `repos/torchlogix/src/torchlogix/layers/dense.py` for probe-compatible shapes;
- `repos/torchlogix/experiments/train.py` for the training objective;
- a new shared teacher-cache package used later by QBridge.

Do not use `repos/difflogic-light-master` as the foundation because it lacks convolutional LGNs. It can be a Light parameterization baseline on flattened inputs, but it is not appropriate for spatial teacher-student alignment.

## Datasets and protocol

| Dataset | Role | Teacher and student protocol |
|---|---|---|
| MNIST | Pipeline smoke test | Small CNN teacher; compact dense or convolutional DLGN student |
| CIFAR-10 | Primary DATE result | ResNet-18 teacher; two compact TorchLogix convolutional gate budgets |
| CIFAR-100 | Secondary transfer-complexity result | ResNet-18 teacher; one compact and one medium student budget |

Use standard train augmentation and evaluate on unaugmented validation/test images. Compute sketch thresholds from training examples only. A cached target must be associated with the original unaugmented example identifier; if strong augmentation is used, cache teacher features per deterministic view or run only the teacher online for that augmentation. Do not silently pair an augmented student image with a sketch from a different view.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Label-only convolutional DLGN student at each gate budget | Defines the task-specific logic student baseline |
| Hinton et al., [*Distilling the Knowledge in a Neural Network*](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/), 2015 | Temperature-scaled teacher-logit KD | Standard output-distillation baseline |
| Romero et al., [*FitNets: Hints for Thin Deep Nets*](https://arxiv.org/abs/1412.6550), ICLR 2015 | Real-valued teacher feature regression through a training-only regressor | Directly tests the claim that Boolean sketches are more attainable than floating features |
| Zagoruyko and Komodakis, [*Paying More Attention to Attention*](https://openreview.net/pdf?id=Sks9_ajex), ICLR 2017 | Attention-transfer loss on aligned teacher/student spatial maps | Strong published intermediate-representation baseline for convolutional students |
| Park et al., [*Relational Knowledge Distillation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), CVPR 2019 | Distance-wise RKD on pooled student and teacher representations | Tests whether relationships, rather than binary targets themselves, explain the gain |
| Mishra and Marr, [*Apprentice: Using Knowledge Distillation Techniques to Improve Low-Precision Network Accuracy*](https://openreview.net/pdf?id=B1ae1lZRb), ICLR 2018 | Published low-precision KD schedule with the same teacher and DLGN student | Connects the method to established distillation for discrete students |
| Yousefi et al., [*Mind the Gap*](../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf), NeurIPS 2025 | Hard-forward DLGN trained with logit KD | Checks whether the sketch gain remains after controlling the discretization gap |

The minimum DATE set is label-only DLGN, Hinton KD, FitNets, Attention Transfer, and LogicSketch. RKD and Apprentice should be added for the final central CIFAR-10 budget if their adapters are stable. Every method uses the same teacher checkpoint, student architecture, augmentation views, optimizer budget, and validation effort.

### LogicSketch-specific controls

Compare:

1. label smoothing without a teacher;
2. LogicSketch without logit KD;
3. proposed logit KD plus LogicSketch;
4. random balanced sketches unrelated to the teacher;
5. teacher sketches with shuffled sample assignments;
6. equal-length sketches produced from an untrained teacher.

A parameter-matched binary neural network can be reported as context, but it does not replace the published distillation baselines. QBridge remains a separate project unless both implementations are mature.

## Metrics

Report:

- hardened top-1 accuracy, soft accuracy, and soft-to-hard gap;
- deployed gate count, deployed parameters/bits, compiled model size, and latency;
- training-only probe parameters and the fact that they are removed;
- teacher training cost, cache-generation cost, cache size, student training time, and peak GPU memory;
- sketch bit balance, entropy, duplicate rate, pairwise correlation, and student sketch agreement;
- convergence speed measured as steps and GPU-hours to fixed hard-accuracy thresholds;
- accuracy versus deployed gate count for at least two student budgets.

Use at least five paired seeds for the central CIFAR-10 comparison. Projection seeds and model seeds must be logged separately; the final result should include sensitivity to at least three projection seeds.

## Minimum DATE experiment matrix

- CIFAR-10 at two gate budgets: label-only, logit KD, real-feature KD, random sketch, and LogicSketch.
- CIFAR-100 at one predeclared compact budget for label-only, logit KD, and LogicSketch.
- Ablations for sketch length, probe position, and fixed versus learned projection.
- A deploy-time audit proving identical student outputs before and after deleting probe objects.
- Five student seeds for the central result and three seeds for secondary cells.

## Agent deliverables and tests

The assigned agent must provide:

- a versioned teacher/checkpoint registry and reproducible cache command;
- cache integrity tests and sample-ID/augmentation alignment tests;
- model tests showing intermediate outputs do not alter normal inference;
- tests showing probes are absent from deployed parameter and gate counts;
- per-epoch soft/hard/student-sketch metrics;
- a figure showing accuracy versus gate budget and a figure relating sketch agreement to hard accuracy;
- a storage and time accounting table that includes the offline teacher phase.

## Risks, controls, and kill criterion

- **Unattainable or uninformative sketches:** check balance/correlation and begin with short sketches near the teacher output.
- **Teacher cache mismatch under augmentation:** key by sample and view, or explicitly run the teacher online.
- **Improvement is ordinary KD:** include logit KD and real-feature KD with identical tuning effort.
- **Temporary probe leaks into deployment:** test export and count only the probe-free model.
- **Too many hyperparameters:** tune one sketch length and two loss weights on validation, then freeze them.

Continue as a DATE submission only if LogicSketch adds at least 1.5 percentage points of hard CIFAR-10 accuracy over label-only training at the compact budget, or at least 0.75 points over tuned logit KD, and shows a positive result at a second budget or on CIFAR-100. The result must not increase the deployed circuit.

## Definition of done

The project is ready when target caches are reproducible, augmentation alignment is correct, probes disappear at deployment, all required distillation controls are implemented, and the central accuracy gain is supported by paired five-seed CIFAR-10 results.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
- [Mind the Gap](../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf)
