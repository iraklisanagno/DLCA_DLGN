# CoverageDLGN: Coverage-Aware Fixed Connectivity

## Decision and relationship to CoverageNet

This brief merges [CoverageDLGN](../july_21.md) and CoverageNet from [july_20.md](../july_20.md) into one project. They address the same central problem: a DLGN can waste capacity when its fixed random connections repeatedly observe the same narrow subset of earlier signals. Running them as separate projects would duplicate the implementation and weaken the experimental story.

The combined project keeps the strongest parts of both notes:

- the coverage and overlap objective, and formal coverage reporting, from CoverageNet;
- the practical local, butterfly, greedy-coverage, and hybrid connection families from CoverageDLGN;
- a deterministic, hardware-regular connection schedule with no deployed routing parameters;
- ancestry analysis that measures which original inputs can influence each gate.

The paper name should be **CoverageDLGN**. "CoverageNet" can remain the name of the pure greedy-coverage baseline within its ablation.

## Narrative

Imagine an organization in which every employee asks the same two colleagues for information. Adding more employees does not improve the final decision because the organization keeps recycling the same facts. Randomly assigning colleagues helps, but it can still create blind spots and unnecessary repetition.

A DLGN has the same problem. Its logic gates are small and inexpensive, but a gate normally sees only two outputs from the previous layer. If those connections are poorly arranged, many parts of the input never reach important portions of the network. CoverageDLGN lays out the communication pattern before training so that gates collectively see a broad and complementary set of information. The layout is fixed, so it does not add routing weights or routing decisions to the deployed circuit.

The intended contribution is not "a new random seed that happened to work." It is a measurable connection-design rule that improves accuracy or reduces the number of gates required for the same accuracy, while retaining regular wiring suitable for efficient software and hardware implementations.

## Research question and claim

**Question:** At a fixed gate budget, can deterministic coverage-aware connectivity give a hardened DLGN more useful input diversity than random connectivity?

**Target claim:** Coverage-aware schedules improve the accuracy--gate-count Pareto frontier and reduce seed sensitivity without trainable connections or extra deployed state.

The claim must be supported by both task accuracy and topology measurements. An accuracy improvement alone is insufficient if the proposed schedule cannot be shown to change coverage in the expected way.

## Technical design

For each gate, maintain an ancestry set containing the original input features that can reach it. Represent ancestry sets as packed bitsets so that union, cardinality, and overlap are inexpensive. When selecting two predecessors for a new gate, score a candidate pair using:

$$
S(a,b) = \alpha |A_a \cup A_b| - \beta |A_a \cap A_b| - \gamma P_{\mathrm{fanout}}(a,b) - \delta P_{\mathrm{distance}}(a,b).
$$

Here, $A_a$ and $A_b$ are ancestry sets. The fan-out term discourages hot spots, and the optional distance term preserves locality when regular wiring is important. The score is used only while constructing the fixed network; it introduces no trainable or deployed parameter.

Implement these connection families behind one interface:

1. `random`: current TorchLogix behavior.
2. `random_unique`: random pairs without avoidable duplicate pairs.
3. `local_cyclic`: bounded-neighborhood cyclic wiring.
4. `butterfly`: deterministic stages whose stride increases with depth.
5. `coverage_greedy`: candidate selection using the ancestry score.
6. `coverage_hybrid`: mostly butterfly/local edges plus a configurable fraction of greedy long-range edges. This is the proposed method.

Start with rank-2 dense layers. Add a convolutional version only after the dense generator and metrics are stable. In convolutional layers, apply the schedule across channels while leaving TorchLogix's spatial kernel indexing unchanged.

## Implementation plan

### Milestone 1: Reproduce and measure

1. Reproduce the unmodified TorchLogix MNIST and CIFAR-10 hard-accuracy baselines.
2. Add a topology report containing input coverage, pairwise ancestry overlap, fan-out distribution, number of distinct predecessor pairs, and reachable-gate fraction by depth.
3. Verify that the report is deterministic for a fixed topology seed and does not depend on minibatch order.

### Milestone 2: Connection generators

1. Add a fixed-connection strategy interface in `repos/torchlogix/src/torchlogix/connections.py`.
2. Implement local/cyclic and butterfly schedules first because they are simple deterministic controls.
3. Implement packed ancestry propagation and greedy candidate scoring.
4. Implement the hybrid schedule with only two main controls: long-range edge fraction and candidate-pool size.
5. Store the strategy name, topology seed, and generated indices in checkpoints.

### Milestone 3: Training integration

1. Expose the strategy through the shared experiment configuration.
2. Keep data order, initialization seed, training steps, and gate budget paired across strategies.
3. Evaluate hard inference at every saved checkpoint; do not select a model using only soft accuracy.
4. Run a small width/depth sweep, then freeze the best strategy settings before final seeds.

### Milestone 4: Optional convolutional extension

1. Apply the selected channel schedule to `LogicConv2d` without altering spatial receptive fields.
2. Test shape, index-bound, and deterministic-export behavior.
3. Include this extension only if it improves CIFAR-10 beyond the dense-input version; it is not required for the minimum paper.

## Repository foundation

Use [TorchLogix](../../repos/torchlogix/README.md). Relevant starting points are:

- `repos/torchlogix/src/torchlogix/connections.py` for connection generation;
- `repos/torchlogix/src/torchlogix/layers/dense.py` for dense integration;
- `repos/torchlogix/src/torchlogix/layers/conv.py` for the optional channel extension;
- `repos/torchlogix/experiments/train.py` for baseline reproduction.

Do not use `repos/difflogic-light-master` as the foundation. It is useful as an IWP/Light baseline but does not provide convolutional LGNs. Retaining TorchLogix also permits the same topology to be tested with raw, WARP, and Light gate parameterizations.

## Datasets and protocol

| Dataset | Role | Protocol |
|---|---|---|
| MNIST | Debug and mechanism validation | Original binary/thermometer encoding used by the DLGN baseline; three pilot seeds |
| Fashion-MNIST | Low-cost generalization check | Same encoding and budgets as MNIST; no dataset-specific topology tuning |
| CIFAR-10 | Primary DATE result | TorchLogix convolutional model, standard augmentation, two fixed gate budgets, five final seeds |
| CIFAR-100 | Optional scale test | Run only after CIFAR-10 is positive; use one predeclared compact and one medium budget |

Use a validation split for selecting the hybrid fraction and candidate-pool size. The held-out test set must not guide topology selection.

## Comparisons

### Published-method baselines

| Paper and method | Comparison to implement | Why it is required |
|---|---|---|
| Petersen et al., [*Deep Differentiable Logic Gate Networks*](../../pdfs/deep_differentiable_logic_gate_networks.pdf), NeurIPS 2022 | Original rank-2 DLGN with fixed random connections | Establishes the standard DLGN topology baseline |
| Petersen et al., [*Convolutional Differentiable Logic Gate Networks*](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf), NeurIPS 2024 | Published convolutional fixed-connectivity model on CIFAR-10 | Separates gains from spatial layers from gains due to coverage-aware channel wiring |
| Buehrer et al., [*BitLogic: A Framework for Gradient-Based LUT-Native Neural Networks*](../../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf), TMLR 2026 | BitLogic fixed-connectivity and best-of-space configurations at matched fan-in and width | Provides a recent unified LUT-network comparison rather than only a TorchLogix baseline |
| Mommen et al., [*A Method for Optimizing Connections in Differentiable Logic Gate Networks*](../../pdfs/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.pdf), 2025 | Partial learnable-connectivity method with the paper's candidate counts, initially $N_c=8$ and $N_c=16$ | Directly tests whether a fixed coverage schedule can approach learned routing without routing parameters |
| Fojcik et al., [*LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment*](../../pdfs/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.pdf), 2026 | Top-K learnable-connectivity result on the largest matched model that fits the budget | Represents recent learnable sparse connectivity and is the strongest direct competitor |
| Evci et al., [*Rigging the Lottery: Making All Tickets Winners*](https://proceedings.mlr.press/v119/evci20a.html), ICML 2020 | RigL sparse MLP/CNN with the same nonzero-connection budget, reported as a non-logic contextual baseline | Shows whether task-agnostic fixed coverage is competitive with published dynamic sparse training |

The minimum DATE comparison is original fixed-random DLGN, BitLogic fixed routing, Mommen-style partial learned connections, LILogic Top-K connections, and CoverageDLGN. If the authors' code cannot be integrated, reproduce the paper's connection rule in TorchLogix and document every deviation; do not compare only against reported numbers from a different split.

### Controlled topology ablations

At identical architecture, gate parameterization, initialization, and training budget, compare:

1. fixed random and random-unique connections;
2. local/cyclic and butterfly connections;
3. pure greedy CoverageNet connections;
4. proposed coverage-hybrid connections;
5. TorchLogix learnable connections as an implementation-level check.

Repeat the central comparison with the standard Petersen parameterization and the strongest Light or WARP setting. This establishes that the result is about connectivity rather than one gate relaxation.

## Metrics

Report:

- hardened test accuracy, soft accuracy, and soft-to-hard gap;
- gate count, trainable parameters, deployed connection-index bits, and peak GPU memory;
- training time, inference latency, and topology-construction time;
- original-input coverage at each depth;
- mean and distribution of ancestry overlap;
- fan-out mean, maximum, coefficient of variation, and number of unused outputs;
- accuracy variance across topology/training seeds;
- accuracy versus gate count and accuracy versus routing-storage Pareto curves.

For the central result, report mean, standard deviation, and paired 95% confidence intervals over at least five seeds.

## Minimum DATE experiment matrix

- Two gate budgets and three depths on CIFAR-10.
- Random, butterfly, pure coverage, and hybrid schedules on all cells.
- Five seeds for the medium-budget central cell; three paired seeds elsewhere.
- One generalization cell on Fashion-MNIST or CIFAR-100 without retuning the topology objective.
- One ablation each for overlap penalty, fan-out penalty, and long-range fraction.

## Agent deliverables and tests

The assigned agent must provide:

- deterministic generator unit tests, index-bound tests, and expected small butterfly examples;
- brute-force ancestry checks on tiny networks against the packed-bitset implementation;
- configuration files for all required schedules;
- a topology-only analysis command that does not train a model;
- raw per-seed accuracy and topology CSV files;
- one figure connecting measured coverage to hard accuracy;
- a memory comparison that includes connection indices and any temporary generator state.

## Risks, controls, and kill criterion

- **Risk:** coverage increases but accuracy does not. Control this with overlap and fan-out ablations rather than adding more objectives.
- **Risk:** greedy construction is too slow. Bound the candidate pool and compare against the regular butterfly method.
- **Risk:** gains come from a favorable seed. Pair all non-topology randomness and report five final seeds.
- **Risk:** irregular greedy wiring harms deployment. Make the hybrid regular schedule the proposed method and report pure greedy as analysis.

Stop treating this as a DATE paper if the hybrid method fails to improve the fixed-budget hard accuracy by at least 0.3 percentage points on both Fashion-MNIST and CIFAR-10, or fails to match random accuracy with a clear reduction in routing storage or required gates. A negative topology study can still be retained as internal evidence.

## Definition of done

The project is ready for manuscript writing when the baseline is reproduced, all schedules pass deterministic tests, CIFAR-10 has five-seed results at matched gate budgets, topology metrics explain the observed behavior, and the proposed hybrid has a statistically supported Pareto improvement over random wiring.

## Primary references in this repository

- [Original DLGN paper](../../pdfs/deep_differentiable_logic_gate_networks.pdf)
- [Convolutional DLGN paper](../../pdfs/convolutional_differentiable_logic_gate_networks.pdf)
- [Light DLGN paper](../../pdfs/light_differentiable_logic_gate_networks.pdf)
