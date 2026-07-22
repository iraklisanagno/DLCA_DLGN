# PersonalDLGN: Few-Shot Personalization Through Input Thresholds

**Source concept:** PersonalDLGN in [july_20_no_DATE.md](../july_20_no_DATE.md).

## Narrative

Signals produced by people are personal. The same heartbeat shape, movement intensity, or sensor voltage can have a different normal range for each user. A global classifier is trained to serve everyone, but replacing or retraining its whole circuit for every person is expensive and difficult to maintain on an edge device.

PersonalDLGN keeps one shared Boolean decision circuit and gives each user a tiny set of adjustable input thresholds. The shared gates describe general patterns learned from many people. A short labeled calibration session changes only where that person's continuous sensor values become zeros and ones. Personalization is therefore closer to calibrating an instrument than rebuilding a neural network.

The practical promise is one certified shared circuit image plus a small threshold record per user. This suits wearables and other edge devices where storage, update bandwidth, and adaptation time matter as much as accuracy.

## Research question and claim

**Question:** Can subject-specific input binarization recover personalization gains while leaving all DLGN gates and connections unchanged?

**Target claim:** Few-shot threshold adaptation improves unseen-subject macro-F1 over a global hardened DLGN and fixed-threshold calibration at substantially lower per-user storage and adaptation cost than head-only or full-model fine-tuning.

The method adapts to a person, not continuously to an unconstrained stream. Continuous learning, drift detection, and replay are separate future work.

## Technical design

Train a global TorchLogix model with learnable ordered thresholds. Harden its gate functions and freeze all gates, connections, and shared parameters. For a new subject $u$, instantiate only a small threshold delta:

$$
\tau_u = \tau_0 + \Delta\tau_u.
$$

Maintain ordered thresholds with a positive-increment parameterization rather than sorting after each update. Minimize the support-set classification loss with regularization:

$$
\mathcal{L}_u = \mathcal{L}_{\mathrm{CE}}(S_u) + \lambda_1\|\Delta\tau_u\|_1 + \lambda_2\|\Delta\tau_u\|_2^2.
$$

The minimum method uses one delta per original input feature or per feature channel. Do not start with a separate threshold for every time index if this makes the adapter larger than the logic core. Evaluate scalar-per-channel, grouped, and full-feature thresholds as a storage--accuracy ablation.

After adaptation, quantize thresholds to 8 or 16 bits and confirm that every downstream gate and connection bit is identical to the shared checkpoint. Report floating and quantized adapter performance separately.

## Implementation plan

### Milestone 1: Validate threshold semantics

1. Add regression tests for `LearnableBinarization` training and evaluation.
2. Verify that both modes apply the same positive-increment/cumulative transform and preserve threshold ordering.
3. Save thresholds and their parameterization metadata in checkpoints.
4. Reproduce a global fixed-threshold and global learned-threshold baseline before personalization.

This verification is mandatory. The current implementation must not be used for a personalization claim until ordered thresholds are confirmed to behave identically in training and evaluation.

### Milestone 2: Subject protocol and adapter

1. Add subject-aware datasets that return subject ID without exposing query labels to adaptation.
2. Train the global model on training subjects only.
3. Harden and freeze the complete logic core.
4. For each unseen subject, split examples into chronological or predeclared support and query sets.
5. Optimize only `DeltaThreshold` for a fixed number of support steps, then evaluate once on that subject's query set.

### Milestone 3: Storage-aware personalization

1. Add per-channel, grouped-feature, and per-feature delta modes.
2. Quantize deltas or final thresholds to 8 and 16 bits.
3. Serialize each subject adapter independently of the shared circuit.
4. Add an exact check that logic truth tables, connections, and shared threshold base are unchanged.

### Milestone 4: Focused robustness tests

1. Evaluate support sizes of 1, 5, 10, 25, and all available labeled samples per class where feasible.
2. Repeat support sampling to measure adaptation variance.
3. Test label imbalance and a no-update fallback when a class is absent.
4. Add unlabeled threshold calibration only as an optional baseline, not as the main method.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md). Relevant components are:

- `repos/torchlogix/src/torchlogix/layers/binarization.py` for fixed, soft, and learnable thresholds;
- `repos/torchlogix/src/torchlogix/models/dense.py` for compact sensor models;
- `repos/torchlogix/src/torchlogix/circuit.py` for verifying the frozen downstream circuit.

TorchLogix is the best fit because threshold adaptation is already a first-class layer choice. `repos/difflogic` remains a useful global dense baseline. `repos/difflogic-light-master` is not needed as the foundation; the target workloads are compact tabular/time-series representations, and its lack of convolutional LGNs would limit later raw-signal extensions.

## Datasets and protocol

### MIT-BIH Arrhythmia Database

This is the primary DATE use case. Follow the inter-patient DS1/DS2 split described in [the local ECG LGN/LUTNet paper](../../pdfs/inter_patient_ecg_arrhythmia_classification_with_lgns_and_lutns.pdf), not a random beat split. Use the paper's four AAMI classes: N, S, V, and F. Start with its 138-bit engineered feature representation so the initial project tests personalization rather than signal-processing choices.

Protocol:

1. train the global model on DS1 patients;
2. reserve patients or recordings within DS1 for architecture and adaptation hyperparameters;
3. for each DS2 patient, use only a predeclared labeled support subset for adaptation;
4. evaluate on the remaining chronological query beats;
5. report both zero-shot global performance and personalized performance;
6. never tune hyperparameters on aggregate DS2 query results.

Class imbalance must be handled with a fixed training policy and macro metrics. Report every DS2 subject, including subjects where adaptation hurts.

### UCI Human Activity Recognition

Use UCI HAR as a second subject-dependent dataset. Use official subject IDs and a leave-subjects-out or official train/test-subject protocol. Create support/query splits within unseen test subjects. This establishes that the method is not specific to ECG morphology.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| de Chazal et al., *Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features*, IEEE TBME 2004 | Reproduce the inter-patient feature/classification protocol or its closest available classifier on the exact DS1/DS2 split | Establishes the canonical non-personalized MIT-BIH protocol |
| Kiranyaz et al., *Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks*, IEEE TBME 2016 | Patient-specific 1-D CNN using the same labeled support/query beats | Direct published patient-personalization baseline |
| Mommen et al., [*Inter-patient ECG Arrhythmia Classification with LGNs and LUTNs*](../../pdfs/inter_patient_ecg_arrhythmia_classification_with_lgns_and_lutns.pdf), 2026 | Published global LGN and LUTNet models with the paper's 138-bit features and DS1/DS2 split | Most direct logic-network task baseline |
| Cai et al., [*TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning*](https://proceedings.neurips.cc/paper/2020/hash/81f7acabd411274fcf65ce2070ed568a-Abstract.html), NeurIPS 2020 | Frozen feature extractor with bias/lite-residual adaptation, scaled to the compact ECG/HAR model | Published edge-oriented parameter-efficient adaptation baseline |
| Finn et al., [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*](https://proceedings.mlr.press/v70/finn17a.html), ICML 2017 | MAML across training subjects, adapting on the same support sets | Strong few-shot adaptation baseline; run only if subject count supports a valid meta-training split |
| Ordonez and Roggen, *Deep Convolutional and LSTM Recurrent Neural Network for Multimodal Wearable Activity Recognition*, Sensors 2016 | DeepConvLSTM or its reproducible implementation on UCI HAR | Published task baseline for the secondary wearable dataset |

The minimum DATE comparison is the Mommen global LGN/LUTNet protocol, Kiranyaz patient-specific CNN, TinyTL-style adapter, head-only tuning, full tuning, and PersonalDLGN. MAML is a strong optional comparison because a small number of independent patients can make meta-learning statistically fragile.

Raw-signal CNNs and 138-bit engineered-feature models must be placed in separate table blocks unless they consume identical inputs. Report the published paper's number only as context; the claimed statistical comparison must come from reruns on the same support/query manifest.

### PersonalDLGN-specific controls

At identical support sizes and adaptation steps, compare:

1. global DLGN with fixed thresholds;
2. global DLGN with globally learned thresholds, no personalization;
3. subject-specific quantile or normalization thresholds without gradient updates;
4. PersonalDLGN threshold-only adaptation;
5. output-head-only adaptation with the logic core frozen;
6. full DLGN fine-tuning;
7. a parameter-matched affine input adapter.

For full fine-tuning, count the complete per-user checkpoint even if only a few bits changed.

## Metrics

Primary classification metrics are:

- macro-F1 and accuracy;
- per-class sensitivity/recall and precision;
- Cohen's kappa, and the j-kappa metric if reproduced exactly from the ECG reference;
- per-subject macro-F1 distribution and worst-quartile subject result;
- zero-shot-to-personalized improvement and fraction of subjects helped.

Primary edge/personalization metrics are:

- per-user parameter count and serialized bytes at 8/16/32-bit thresholds;
- shared circuit bytes and total storage for 1, 10, 100, and 1000 users;
- adaptation steps, wall time, peak RAM/GPU memory, and labeled support count;
- support-size accuracy/F1 curves;
- number of changed gate, connection, and shared-core bits, which must be zero;
- quantization loss between floating and stored thresholds.

Use subject-level confidence intervals and paired subject-level tests. Beats from one subject are not independent statistical replicates.

## Minimum DATE experiment matrix

- MIT-BIH DS1/DS2 with five support sizes and at least three support samplings where possible.
- UCI HAR with zero-shot, threshold-only, head-only, and full fine-tuning.
- Per-channel, grouped, and per-feature adapters at matched support data.
- 8-bit, 16-bit, and floating thresholds.
- One ablation for regularization strength and one for ordered-threshold parameterization.
- At least three global training seeds; report all test subjects for each seed.

## Agent deliverables and tests

The assigned agent must provide:

- code that deterministically recreates patient/subject and support/query splits;
- a split manifest with no train/test subject overlap;
- ordered-threshold unit tests and train/eval parity tests;
- a frozen-core hash before and after every adaptation run;
- one serialized adapter per subject plus a loader test;
- per-subject result files rather than only aggregate scores;
- storage calculations derived from serialized artifacts, not parameter estimates alone;
- a script that recreates support-size plots and subject-level statistical tests.

## Risks, controls, and kill criterion

- **Support leakage:** enforce subject manifests and chronological support/query boundaries.
- **Rare ECG classes:** report per-class sensitivity and repeat support sampling; do not hide missing-class subjects.
- **Threshold overfitting:** regularize deltas, use few fixed adaptation steps, and tune on DS1 validation subjects.
- **Adapter is not actually small:** compare serialized bytes and use grouped/per-channel variants.
- **Global model is too weak:** first match the local ECG paper's global protocol before judging personalization.

Continue as a DATE paper only if threshold-only adaptation gives a statistically supported macro-F1 gain on unseen MIT-BIH subjects and UCI HAR, helps a majority of subjects, and approaches or exceeds head-only adaptation with substantially fewer per-user bytes. Stop if full or head-only tuning dominates at comparable storage and adaptation time, or if threshold quantization removes the gain.

## Definition of done

The project is ready for writing when the exact inter-patient protocol is reproducible, global gates remain bit-identical during adaptation, subject-level results exist for all support sizes and baselines, serialized storage is measured, and the improvement generalizes beyond ECG.
