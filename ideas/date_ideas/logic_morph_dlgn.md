# LogicMorph-DLGN: Function-Preserving Growth and Transfer

**Source concept:** LogicMorph-DLGN in [transfer_learning.md](../transfer_learning.md).

## Narrative

Training a DLGN from scratch can be slow and sensitive to its width, depth, connections, and gate initialization. If a trained circuit is too small, the usual response is to discard it and start again with a larger architecture.

LogicMorph treats a trained circuit as a structure that can be expanded. New layers begin as wires that copy the old signals. New gates begin disconnected from the output or duplicate existing behavior. The expanded network therefore starts with the same hardened decision as its parent, then gradually unlocks its new capacity for refinement.

The analogy is renovating a working building without first demolishing it. The immediate objective is not only higher final accuracy; it is to reach useful accuracy faster and make architecture exploration less wasteful.

## Research question and claim

**Question:** Can exact hard-function morphisms initialize deeper or wider DLGNs so they train faster and more reliably than the same child architectures trained from scratch?

**Target claim:** A morphed child exactly reproduces its parent's hard predictions before training, then reaches a target hard accuracy in fewer optimization steps/GPU-hours than scratch, naive copy, or ordinary distillation.

The strongest initial claim should be same-task growth. Cross-dataset transfer is a stress test, not a prerequisite for the core method.

## Supported morphisms

Implement in this order:

1. **Depth insertion:** insert rank-2 copy gates whose selected truth table returns one designated input. The second input is present but ignored.
2. **Width expansion:** copy all parent gates and append new inactive gates. Parent outputs initially ignore the appended gates, preserving the old function.
3. **Duplicate-and-unlock:** initialize new gates as duplicates of parent signals, then gradually permit downstream use.
4. **Rank lifting:** map a rank-2 truth table into rank 4/6 by making the new truth table independent of additional inputs. Treat this as an optional extension because export support must first be verified.
5. **Topology expansion:** add connections only into an initially inactive subgraph, then unlock them during training.

For each morphism, return both the child checkpoint and an explicit node map from parent gates to child gates. Exactness refers to hardened outputs before unlocking. Soft activations may differ because gate parameterizations and temperatures differ; measure this separately.

## Technical design

Represent a morph operation as a deterministic transformation:

~~~text
morph(parent_checkpoint, child_spec, morph_seed)
  -> child_checkpoint, parent_to_child_map, invariants
~~~

The invariants include copied truth-table choices, copied connection indices, ignored/inactive new gates, threshold equality, and output-vote equality. Immediately harden the child and compare every parent and child output on a validation corpus. For tiny models, exhaustively compare all binary inputs.

Training uses three short phases:

1. **Verify:** all new paths are inactive; parent and child hard decisions must match.
2. **Warm growth:** freeze copied parent gates and train only new gates/routes plus an optional small output allocation.
3. **Refine:** unfreeze the full child with a lower learning rate and optional parent-output distillation.

The phase lengths are fixed from validation experiments. The paper should compare the complete compute to target accuracy, not only epochs after the expensive parent was trained. Report both one-off cost and amortized cost when one parent initializes several child models.

## Implementation plan

### Milestone 1: Checkpoint schema and exact tests

1. Extend checkpoint metadata to fully describe thresholds, connections, gate ranks, parameterization, and output grouping.
2. Implement tiny deterministic rank-2 networks and enumerate all binary inputs.
3. Add a parent/child equivalence utility comparing class scores, decisions, intermediate mapped gates, and serialized metadata.
4. Fail the morph command if any declared invariant is violated.

### Milestone 2: Depth and width MVP

1. Implement identity/copy gate-logit initialization for raw, WARP, and Light parameterizations where well-defined.
2. Insert one or more copy layers while preserving parent routes.
3. Expand width by appending gates ignored by the original output allocation.
4. Add warm-growth masks and a scheduled unlock operation.

### Milestone 3: Same-task acceleration

1. Train small parent models on MNIST and CIFAR-10.
2. Morph each into one deeper and one wider child.
3. Train scratch, naive-copy, distilled, and LogicMorph children with matched optimizers and step budgets.
4. Save hard accuracy versus steps and wall-clock time from child initialization.

### Milestone 4: Transfer extensions

1. Test a CIFAR-10 parent used to initialize a larger CIFAR-10 child at a second gate budget.
2. Test a CIFAR-10 parent transferred into a CIFAR-100 child with a new output group and retained early logic.
3. Optionally pretrain on CIFAR-100 coarse labels and morph to fine labels.
4. Attempt rank lifting only after TorchLogix rank-4/6 export and hard-equivalence behavior are covered by tests.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md), primarily:

- `repos/torchlogix/src/torchlogix/parametrization.py` for truth-table parameter mappings;
- `repos/torchlogix/src/torchlogix/connections.py` for route preservation and expansion;
- `repos/torchlogix/src/torchlogix/layers/dense.py` and `repos/torchlogix/src/torchlogix/layers/conv.py` for child construction;
- `repos/torchlogix/src/torchlogix/circuit.py` for hard-function verification.

TorchLogix supports dense and convolutional models, which is necessary for meaningful CIFAR transfer. `repos/difflogic` should be used to validate rank-2 behavior against the original implementation. `repos/difflogic-light-master` is not the foundation because its models are non-convolutional even though it is useful for testing Light parameterization ideas.

## Datasets and protocol

| Dataset | Role | Transfer scenario |
|---|---|---|
| MNIST | Exactness and rapid debugging | Small parent to deeper/wider child on the same task |
| CIFAR-10 | Primary DATE result | Compact convolutional parent to medium/deep child on the same task |
| CIFAR-100 | Secondary transfer result | CIFAR-10 parent to CIFAR-100 child, or coarse-to-fine CIFAR-100 |

Same-task CIFAR-10 growth is the central controlled experiment because the parent and child output semantics are identical. For CIFAR-10 to CIFAR-100, replace the class-vote output, preserve only compatible feature logic, and do not claim exact end-to-end equivalence after changing the classes. Instead, verify equality of every retained mapped feature gate before fine-tuning.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Deep Differentiable Logic Gate Networks*](../../pdfs/deep_differentiable_logic_gate_networks.pdf), NeurIPS 2022 | Parent and child rank-2 DLGNs trained from scratch | Establishes original DLGN convergence and final accuracy |
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Scratch convolutional child on CIFAR-10/CIFAR-100 | Direct architecture baseline for the primary transfer experiments |
| Chen et al., [*Net2Net: Accelerating Learning via Knowledge Transfer*](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/), ICLR 2016 | Adapt Net2DeeperNet and Net2WiderNet to copy gates and replicated logic channels | Closest published function-preserving growth method |
| Wei et al., [*Network Morphism*](https://proceedings.mlr.press/v48/wei16.html), ICML 2016 | Apply the published depth/width morphing schedule with the nearest valid Boolean parameter mapping | Tests whether LogicMorph adds value beyond general network morphism |
| Hinton et al., [*Distilling the Knowledge in a Neural Network*](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/), 2015 | Scratch child trained with parent-logit KD | Standard non-function-preserving transfer baseline |
| Furlanello et al., [*Born Again Neural Networks*](https://proceedings.mlr.press/v80/furlanello18a.html), ICML 2018 | Same-size or larger child distilled from the trained parent | Controls for gains from teacher supervision without structural morphing |
| Li and Hoiem, [*Learning without Forgetting*](https://arxiv.org/abs/1606.09282), ECCV 2016 | Cross-dataset child with source-output preservation loss | Published baseline for the CIFAR-10 to CIFAR-100 extension |

The minimum DATE set is scratch, Net2Net, Network Morphism, parent-logit KD, and LogicMorph. Implement Net2Net and Network Morphism inside TorchLogix rather than comparing against their reported CNN numbers. State which continuous-network operator cannot be translated exactly to Boolean gates and use the nearest predeclared mapping.

### LogicMorph-specific controls

At the same child architecture and maximum training steps, compare:

1. parent checkpoint continued without growth;
2. naive tensor/gate copying without an exactness constraint;
3. function-preserving morph with immediate full unfreezing;
4. proposed morph with warm growth and scheduled unfreezing;
5. frozen parent trunk plus newly trained output head for cross-dataset transfer;
6. gate-only and route-only transfer.

All methods use the same optimizer family, augmentations, target thresholds, and validation effort. Include parent pretraining in both one-off and amortized compute accounting.

## Metrics

Report:

- exact pre-training equality of hard class scores and predictions for same-task morphs;
- retained-feature equality for cross-task morphs;
- hard/soft accuracy versus optimization steps and wall-clock GPU-hours;
- steps and time to fixed accuracy thresholds;
- final hard accuracy and soft-to-hard gap at a fixed compute budget;
- fraction of copied gate functions and routes retained after training;
- number of parent gates, new gates, active gates by phase, and child storage;
- peak GPU memory, morph time, parent pretraining cost, and amortized cost over multiple children;
- variance and failed-run rate across seeds.

The primary statistic is paired time-to-target across at least five seeds. If a run never reaches the target, include it as right-censored or report failure rather than discarding it.

## Minimum DATE experiment matrix

- MNIST depth and width morphs with exhaustive tiny-model support tests.
- CIFAR-10 one deeper and one wider child at two parent budgets.
- Scratch, naive copy, KD, immediate unfreeze, and scheduled LogicMorph.
- One CIFAR-100 transfer scenario as a secondary result.
- Ablations for freeze duration, copy-gate initialization, and route retention.
- Five paired seeds for the central CIFAR-10 child and three for secondary cells.

## Agent deliverables and tests

The assigned agent must provide:

- morph commands that are deterministic from parent hash, child specification, and seed;
- exhaustive Boolean equivalence tests for tiny depth/width morphs;
- corpus-wide class-score equivalence tests before training;
- a human-readable parent-to-child node map and invariant report;
- checkpoint migration tests and backward-compatible loading where practical;
- per-step accuracy/time logs for all methods, including failed targets;
- figures for hard accuracy versus steps and GPU-hours.

## Risks, controls, and kill criterion

- **Hard equality but poor soft initialization:** measure both; use a short parent-logit warm-up without weakening the hard invariant.
- **New gates remain unused:** monitor gradients, activation entropy, and route adoption during warm growth.
- **Acceleration excludes parent cost:** report one-off and amortized compute accounting.
- **Rank-lifting export gap:** keep rank lifting optional until exact hard export is supported.
- **Cross-task transfer obscures the contribution:** make same-task growth the central claim.

Continue as a DATE submission only if same-task LogicMorph is exactly hard-equivalent before training and reaches a predeclared CIFAR-10 hard-accuracy target at least 2x faster than scratch, or reaches at least 0.5 percentage points higher hard accuracy at the same child-training compute. The advantage must appear for both a deeper and a wider child or generalize to the secondary dataset.

## Definition of done

The project is ready when depth and width transformations have exhaustive exactness tests, every child records its parent map, same-task acceleration is supported by five paired seeds, and compute accounting includes both parent and child phases.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
- [Light DLGN paper](../../pdfs/light_differentiable_logic_gate_networks.pdf)
