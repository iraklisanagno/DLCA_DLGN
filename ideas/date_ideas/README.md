# DATE 2027 Implementation Portfolio

**Prepared:** July 22, 2026

This directory turns the selected brainstorming ideas into focused implementation briefs. Each brief is intended to be understandable by both a researcher and a coding agent.

## Selected projects

| Project | Brief | Primary contribution |
|---|---|---|
| CoverageDLGN | [coverage_dlgn.md](coverage_dlgn.md) | Better fixed routing without trainable connection parameters |
| MarginSynth | [margin_synth.md](margin_synth.md) | Accuracy-budgeted approximate simplification of hardened circuits |
| LogicSketch Distillation | [logic_sketch_distillation.md](logic_sketch_distillation.md) | Full-precision teachers provide attainable Boolean intermediate targets |
| PersonalDLGN | [personal_dlgn.md](personal_dlgn.md) | Few-shot subject adaptation by changing only input thresholds |
| ExitLogic | [exit_logic.md](exit_logic.md) | Certified per-sample early termination with the full model's decision preserved |
| LogicMorph-DLGN | [logic_morph_dlgn.md](logic_morph_dlgn.md) | Function-preserving growth and transfer of trained DLGNs |
| QBridge-DLGN | [qbridge_dlgn.md](qbridge_dlgn.md) | Transfer from standard quantized CNNs into convolutional logic circuits |
| ElasticMux-DLGN | [elastic_mux_dlgn.md](elastic_mux_dlgn.md) | Externally selected runtime accuracy/latency modes in one hardened model |

## Execution at a glance

| Project | Minimum viable dataset | First mechanism to prove |
|---|---|---|
| CoverageDLGN | MNIST, then CIFAR-10 | Fixed schedules increase ancestry coverage at matched gates |
| MarginSynth | Fashion-MNIST, then CIFAR-10 | Margin ranking beats random rewrites at matched accuracy loss |
| LogicSketch | CIFAR-10 | One balanced teacher sketch improves a compact hard student |
| PersonalDLGN | MIT-BIH | Threshold-only adaptation helps unseen patients without changing gates |
| ExitLogic | MNIST, then CIFAR-10 | Certified prefixes exactly match complete inference |
| LogicMorph | MNIST, then CIFAR-10 | Deeper/wider children are hard-equivalent before training |
| QBridge | CIFAR-10 | Native QNN bit codes beat logit-only distillation |
| ElasticMux | MNIST, then CIFAR-10 | Three prefixes physically skip work and remain near static baselines |

## Repository decision

Use [TorchLogix](../../repos/torchlogix/README.md) as the common development foundation for all eight projects.

It is the only local repository that currently combines:

- dense and convolutional logic layers;
- rank-2, rank-4, and rank-6 LUT training paths;
- raw, WARP, and Light parameterizations;
- fixed and learnable dense connections;
- fixed convolutional connections;
- fixed, soft, and learnable input thresholds;
- MNIST and CIFAR-10 reference models;
- an editable graph-based `Circuit` IR;
- circuit simplification, compiled C, JSON, and Verilog export;
- documented PyTorch 2.9 and CUDA 13 compatibility.

The other repositories remain valuable references and baselines:

- [Original difflogic](../../repos/difflogic/README.md) provides the original experimental protocol and mature `CompiledLogicNet` behavior.
- [difflogic-light](../../repos/difflogic-light-master/README.md) provides the IWP implementation and CIFAR-100 recipes, but it does not provide a convolutional LGN. Do not use it as the main foundation for projects requiring spatial stages.
- [Mind the Gap](../../repos/neuripssubmision_mind_gap/README.md) provides a hard-forward optimization baseline but is a smaller research snapshot rather than a general library.

Using one foundation avoids eight incompatible training and export pipelines. Where a baseline is unavailable in TorchLogix, reproduce only the minimal behavior needed for a fair comparison.

## Common experiment rules

Every project must follow the same reporting discipline.

1. Use a separate validation split for model selection, thresholds, rewrite budgets, and early-exit calibration.
2. Treat hardened accuracy as the primary DLGN accuracy. Always report the soft-to-hard gap.
3. Use three paired seeds for pilots and at least five paired seeds for the final central comparison.
4. Reuse identical data splits, augmentations, training steps, and gate budgets across compared methods.
5. Report mean, standard deviation, and a paired confidence interval or paired significance test for the central result.
6. Save the complete configuration, random seed, source revision, environment fingerprint, checkpoint, and per-epoch CSV.
7. Keep the test set untouched until the method and hyperparameters are frozen.
8. Report failed/diverged runs instead of silently replacing them.
9. Distinguish training-only parameters from deployed circuit state.
10. Count preprocessing, teacher caching, source pretraining, and compilation time when they are part of the claimed advantage.

## Common implementation prerequisites

Complete these once before assigning all projects in parallel:

1. Add a configuration-driven experiment entry point rather than extending one long command-line script per idea.
2. Add Fashion-MNIST and CIFAR-100 to the TorchLogix dataset registry.
3. Add shared metric utilities for hard accuracy, macro-F1, soft-to-hard gap, peak GPU memory, wall time, model bits, gate count, and seed aggregation.
4. Add checkpoint metadata with architecture, thresholds, connections, parameterization, and dataset encoding.
5. Add a small model factory that can return intermediate activations without changing normal inference.
6. Add `Circuit` regression tests covering JSON round-trip, simplification, compiled C, and Verilog equivalence.
7. Fix any baseline issue before implementing an idea. In particular, verify that learnable thresholds use the same ordered-threshold transformation in training and evaluation.

## Agent ownership and conflict avoidance

Suggested branches and owned modules:

| Agent | Branch | Primary owned area |
|---|---|---|
| Coverage | `idea/coverage-dlgn` | fixed connection generators and topology metrics |
| MarginSynth | `idea/margin-synth` | `Circuit` rewrite/search package |
| LogicSketch | `idea/logic-sketch` | teacher cache, sketch generation, auxiliary losses |
| PersonalDLGN | `idea/personal-dlgn` | subject splits and threshold adapters |
| ExitLogic | `idea/exit-logic` | certified vote blocks and prefix compiler |
| LogicMorph | `idea/logic-morph` | checkpoint morphisms and exact-equivalence tests |
| QBridge | `idea/qbridge` | QNN adapters, code interfaces, teacher-guided routes |
| ElasticMux | `idea/elastic-mux` | multi-mode model and runtime mode selection |

Do not let multiple agents independently redesign the dataset loader, checkpoint schema, or `Circuit` serialization. Merge the common prerequisites first. ExitLogic and ElasticMux should share a multi-exit/prefix execution utility but keep different policies: ExitLogic decides per sample with an exact certificate; ElasticMux receives an external mode and intentionally trades accuracy for cost. LogicSketch and QBridge should share feature-cache plumbing but not their research methods.

## Minimum result package from every agent

Each agent must return:

- source code and tests;
- one smoke-test configuration;
- one pilot configuration per primary dataset;
- a baseline reproduction table;
- raw per-seed CSV/JSON results;
- a script that regenerates every table and figure;
- exact launch command and environment fingerprint;
- a short `RESULTS.md` stating what worked, what failed, and whether the kill criterion passed;
- no untracked manual preprocessing steps.

The first milestone is not a large accuracy number. It is a reproducible baseline plus one controlled experiment that isolates the proposed mechanism.

## Portfolio stage gate

Eight complete DATE papers cannot be developed with equal depth at once. Treat these briefs as a portfolio:

1. Merge the common prerequisites and reproduce the untouched TorchLogix baselines.
2. Give each idea a short mechanism sprint limited to its minimum dataset and central ablation.
3. Apply the kill criterion in each brief using raw results, not expected novelty.
4. Promote only the strongest two or three ideas to full CIFAR/multi-seed experiments.
5. Select one coherent primary DATE manuscript; keep compatible mechanisms as ablations or follow-up papers rather than combining unrelated contributions.

Use separate result directories and immutable run manifests from the first experiment. This makes a negative pilot useful and prevents later agents from reconstructing undocumented settings.
