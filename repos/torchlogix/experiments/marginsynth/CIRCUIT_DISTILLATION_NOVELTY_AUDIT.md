# Local novelty audit: margin-constrained circuit resynthesis

This audit intentionally uses only papers and paper notes already present in
this repository. It is a design guardrail, not a claim that the worldwide
literature search is complete.

## Proposed method in one sentence

Starting from a trained and hardened DLGN, jointly reassign every eligible
rank-2 gate among all 16 Boolean functions on the GPU, minimizing a
synthesis-oriented circuit cost while preserving the teacher's winner versus
runner-up decision margins; then harden, exactly repair any budget violations,
simplify, and measure the result with the same ABC flow as every baseline.

## What cannot be claimed as novel

| Local paper | Existing idea that overlaps | Consequence for this work |
|---|---|---|
| *Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks* | Treats the original network as a teacher, scores constant ties with a Gauss--Newton approximation to teacher-logit squared error, applies binary-split refinement, and fine-tunes. | Teacher distillation, post-training simplification, constant gates, and approximate sensitivity screening are not novel. Our primary method must not use its shortlist, fixed per-layer tie quota, binary split, or Unit-Tying warm start. |
| *Silicon-Aware Neural Networks* | Adds differentiable expected standard-cell area per neuron to the DLGN training loss. | A differentiable per-gate hardware penalty is not novel by itself. Our differentiator must be post-training circuit resynthesis plus decision constraints, exact repair, and same-flow synthesis validation. |
| *Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks* | Uses discrete/straight-through gate selection to align training and hardened inference. | Straight-through or annealed categorical gate selection is an implementation mechanism, not a contribution. |
| *BitLogic: A Framework for Gradient-Based LUT-Native Neural Networks* | Unifies LUT-native differentiable models and explores broad LUT design choices; its reported scope does not center on post-training pruning/calibration. | Optimizing LUT choices is not enough. The contribution must concern constrained post-training circuit transformation. |
| *Fitting Multilinear Polynomials for Logic Gate Networks* and *Light Differentiable Logic Gate Networks* | Provide alternative continuous parameterizations/training formulations for Boolean gates. | A soft Boolean relaxation is prior machinery, not novelty. |
| *LiLogic-Net* | Learns connectivity to make compact hardware-oriented logic networks. | Compact learned topology and hardware deployment are not unique to our method; here connectivity is frozen and only the hardened circuit is resynthesized. |

## Deliberate separation from Two-Stage Unit Tying

The primary method has no Unit-Tying warm start, no Gauss--Newton shortlist, no
binary split, no predetermined number of removed gates, and no restriction to
constant replacements. All eligible gate functions are optimized jointly.
Unit Tying + Margin Refinement remains in this repository only as an explicitly
labeled ablation/baseline.

## Candidate contribution bundle

The paper claim must be made as the following bundle, not as any isolated
component:

1. Post-training **whole-circuit LUT resynthesis**, including constants,
   bypasses, inversions, and alternative two-input gates.
2. Direct preservation of the teacher's **winner--runner decision boundary**,
   including worst-class and stratified-fold penalties, rather than only MSE on
   all logits.
3. A synthesis-aligned AIG operation cost used during the joint search, with
   final cost always measured by the identical Yosys/ABC flow.
4. Deterministic **exact constraint repair** that restores original LUTs until
   global and per-class accuracy/disagreement budgets are satisfied.
5. A reproducible accuracy--hardware Pareto path with saved checkpoints,
   calibration membership hashes, training traces, and tool versions.

## Evidence locations used

- `notes/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.md`
- `pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf`
- `notes/silicon_aware_neural_networks.md`
- `pdfs/silicon_aware_neural_networks.pdf`
- `notes/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.md`
- `pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf`
- `notes/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.md`
- `pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf`
- `notes/fitting_multilinear_polynomials_for_logic_gate_networks.md`
- `pdfs/fitting_multilinear_polynomials_for_logic_gate_networks.pdf`
- `notes/light_differentiable_logic_gate_networks.md`
- `notes/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.md`

## Publication guardrail

Before submission, this local audit must be followed by a formal external
literature search. Until then, wording should be “our locally differentiated
method” rather than “the first method.”
