# MarginSynth paper comparison plan

## Purpose

This document freezes the intended external-paper comparison set for the
MarginSynth DATE submission. MarginSynth is a post-training, margin-constrained
resynthesis method, so competitors must be grouped according to whether they
start from the same trained checkpoint or train a different model.

The five external methods below are the comparison set. MarginSynth itself is
not counted among the five.

## Selected comparison papers

### 1. Two-Stage Unit Tying

Lee et al.,
[*Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*](../../../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf),
2026.

- Role: mandatory and most direct post-training competitor.
- Comparison: start MarginSynth and Unit Tying from the identical trained and
  hardened checkpoint.
- Match: dataset split, seed, recovery budget, allowed accuracy loss, synthesis
  flow, and hardware library.
- Reproduce: constant tying, Gauss--Newton screening, Binary Split refinement,
  tied-ratio operating points, and recovery fine-tuning.

### 2. Mind the Gap

Yousefi et al.,
[*Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks*](../../../../pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf),
2025.

- Role: recent training-time alternative based on hard-forward/Gumbel gates.
- Question: does MarginSynth remain useful when the source network already has
  a small discretization gap?
- Comparison: train the published formulation, harden it, and evaluate it with
  the same export and synthesis pipeline used for MarginSynth.

### 3. Optimized DLGN connectivity

Mommen et al.,
[*A Method for Optimizing Connections in Differentiable Logic Gate Networks*](../../../../pdfs/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.pdf),
2025.

- Role: recent compact-DLGN method that learns connections during training.
- Question: is post-training circuit resynthesis better than producing a
  smaller network through learned connectivity?
- Comparison: reproduce partial learned-connectivity settings at matched final
  hardware costs, particularly on Fashion-MNIST.

### 4. LILogic Net

Fojcik et al.,
[*LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment*](../../../../pdfs/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.pdf),
2025/2026.

- Role: strong recent compact and hardware-oriented DLGN competitor.
- Question: how does MarginSynth compare with Top-\(K\) learned connectivity and
  basis-projected gate evaluation?
- Comparison: reproduce relevant Top-\(K\) configurations at matched datasets
  and synthesized hardware budgets.

### 5. Silicon-Aware Neural Networks

Fieldhouse and Tang,
[*Silicon-Aware Neural Networks*](../../../../pdfs/silicon_aware_neural_networks.pdf),
2026.

- Role: training-time hardware-cost optimization, especially relevant to DATE.
- Question: can post-training MarginSynth compete with a method that includes
  expected implementation cost directly in training?
- Comparison: use one declared cell-cost library and report accuracy against
  final cost from the same synthesis flow.

## Starting-architecture references

The following papers define the unmodified source architectures. They are
reference architectures, not additional competing simplification methods, and
therefore do not count toward the five-paper limit.

- Petersen et al.,
  [*Deep Differentiable Logic Gate Networks*](../../../../pdfs/deep_differentiable_logic_gate_networks.pdf),
  NeurIPS 2022: default dense architecture for MNIST and Fashion-MNIST.
- Petersen et al.,
  [*Convolutional Differentiable Logic Gate Networks*](../../../../pdfs/convolutional_differentiable_logic_gate_networks.pdf),
  NeurIPS 2024: default convolutional architecture for CIFAR-10.

For each source model, report both the untouched hardened network and its
function-preserving exact-synthesis result.

## Required presentation structure

Do not put all methods into one undifferentiated superiority table.

1. **Same-checkpoint post-training study:** MarginSynth versus Two-Stage Unit
   Tying. This is the strongest causal comparison of the simplification method.
2. **End-to-end accuracy--hardware study:** MarginSynth versus Mind the Gap,
   optimized connectivity, LILogic Net, and Silicon-Aware Neural Networks.
   These methods train different models, so compare complete pipelines rather
   than attributing every difference to post-training simplification.

Numbers copied from papers may appear only in a clearly labelled related-work
table. Statistical superiority claims require local reproduction with matched
data partitions and the same synthesis flow.

## Dataset roles

- **CIFAR-10:** central DATE benchmark because it provides the greatest overlap
  with the selected papers. Use five central seeds.
- **Fashion-MNIST:** secondary dense benchmark and direct overlap with optimized
  connectivity and LILogic Net.
- **MNIST:** correctness, debugging, and low-cost mechanism experiments; it is
  not the central paper claim.
- **CIFAR-100:** optional scalability extension after the CIFAR-10 pipeline is
  stable. Among these competitors, Mind the Gap provides the clearest overlap.

## Fairness requirements

- Use disjoint training, calibration, validation, and held-out test partitions.
- Freeze hyperparameters and operating-point rules before opening test results.
- Use identical Yosys/ABC commands and libraries for every exported circuit.
- Compare methods at matched accuracy-loss and hardware-cost operating points,
  not only at each method's most favorable point.
- Report mean, standard deviation, individual seed values, and paired confidence
  intervals over the same seeds.
- Separate training time, hyperparameter exploration time, frozen-method
  application time, recovery time, and common synthesis time.
- Preserve configs, seeds, checkpoints, split hashes, software versions, raw
  metrics, synthesis logs, failures, and exact commands for every method.

## Common reported metrics

- accuracy and macro-F1;
- global disagreement with the original hardened model;
- worst-class accuracy loss and worst-class disagreement;
- reachable/live gates;
- Yosys/ABC AND nodes and logic levels;
- mapped cell area or FPGA LUT count when a common target library is available;
- compiled-C latency and, where practical, FPGA latency/throughput;
- method, recovery, exploration, and synthesis wall-clock time; and
- peak CPU memory and peak GPU memory.

## Priority

If implementation time becomes limited, comparisons should be completed in
this order: Two-Stage Unit Tying, LILogic Net, Mind the Gap, optimized
connectivity, and Silicon-Aware Neural Networks. Unit Tying must not be omitted.
