# CIFAR-10 DLGN Baseline Reference

This note records the CIFAR-10 accuracy values found in the papers under
`pdfs/` and explains which results are valid comparisons for CoverageDLGN.
It is intended to prevent results with different architectures, gate budgets,
training protocols, or evaluation modes from being treated as equivalent.

No new experiments were run to prepare this comparison.

## Primary comparison for CoverageDLGN

For experiments using the paper-faithful convolutional Small architecture,
the primary published reference is:

| Method | Architecture | Accuracy | Circuit size | Source |
|---|---|---:|---:|---|
| LogicTreeNet-S | Convolutional DLGN, `k=32` | **60.38%** | Approximately 0.40M gates | *Convolutional Differentiable Logic Gate Networks*, Table 1 |

The corresponding Medium reference is LogicTreeNet-M at **71.01%** and
approximately 3.08M gates. Small and Medium use two-bit RGB inputs encoded by
three thresholds per channel, giving nine Boolean input channels.

## Why the two-stage paper reports 58.39%

*Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*
adopts the LogicTreeNet S/M architectures but trains its own checkpoints before
applying unit tying. Its CIFAR-10 results are:

| Model | Untied checkpoint accuracy | Location |
|---|---:|---|
| CIFAR-10(S) | **58.39%** | Appendix B, Table 7 |
| CIFAR-10(M) | **71.57%** | Table 1 |

In that paper, the label `original: 58.39` means the paper's original
**pre-tying checkpoint**. It does not mean the 60.38% result originally
published for LogicTreeNet-S. Therefore, 58.39% is an independent
reimplementation result for the intended S architecture, approximately 1.99
percentage points below the original LogicTreeNet-S result.

For CoverageDLGN:

- **60.38%** is the primary published target for a fully trained,
  paper-faithful LogicTreeNet-S.
- **58.39%** is a useful secondary reproduction reference.
- A short topology pilot must not be presented as a full reproduction of
  either result.

## Accuracy values from other local papers

These results provide useful context, but most are not direct comparisons to
LogicTreeNet-S.

| Paper or method | CIFAR-10 accuracy | Architecture or protocol distinction |
|---|---:|---|
| Deep DLGN small | 51.27% | Non-convolutional; 4 layers × 12K gates = 48K gates |
| Deep DLGN medium | 57.39% | Non-convolutional; 4 layers × 128K gates = 512K gates |
| Deep DLGN large | 60.78% | Non-convolutional; 5 layers × 256K gates = 1.28M gates |
| Deep DLGN large×2 | 61.41% | Non-convolutional; 5 layers × 512K gates = 2.56M gates |
| Deep DLGN large×4 | 62.14% | Non-convolutional; 5 layers × 1.024M gates = 5.12M gates |
| LogicTreeNet-S | **60.38%** | Convolutional S; approximately 0.40M gates |
| LogicTreeNet-M | 71.01% | Convolutional M; approximately 3.08M gates |
| LogicTreeNet-B | 80.17% | Convolutional B; teacher supervision and fixed feature preprocessing |
| LogicTreeNet-L | 84.99% | Convolutional L; teacher supervision and fixed feature preprocessing |
| LogicTreeNet-G | 86.29% | Convolutional G; teacher supervision and fixed feature preprocessing |
| Conv. TTNet small | 50.10% | Different convolutional truth-table architecture |
| Conv. TTNet large | 70.75% | Larger, different convolutional truth-table architecture |
| LILogicNet-S | 55.11% | Non-convolutional learnable connectivity; 8K gates |
| LILogicNet-M | 57.66% | Non-convolutional learnable connectivity; 64K gates |
| LILogicNet-L | 60.98% | Non-convolutional learnable connectivity; 256K gates |
| Multilinear Soft-Mix | 58.13 ± 0.12% | Non-convolutional; 6 layers × 128K; last-10-checkpoint metric |
| Multilinear-CovJac | 58.97 ± 0.26% | Same non-convolutional experiment; different gate parameterization |
| BitLogic DiffLogic reproduction, width 64K | 51.73 ± 0.34% | Shared two-layer protocol; not the published DiffLogic architecture |
| BitLogic best-of-space, width 64K | 58.06 ± 0.14% | Two-layer, four-input LUT configuration |
| Silicon-aware DLGN baseline | 60.07% | Deep feed-forward DLGN; not LogicTreeNet-S |

### Papers without a directly comparable S result

- *Light Differentiable Logic Gate Networks* primarily evaluates the Medium
  DLGN/CDLGN architectures on CIFAR-100. It does not provide a new,
  paper-faithful CIFAR-10 LogicTreeNet-S test result.
- *Mind the Gap* evaluates deep feed-forward networks, including width 256K
  and depth 12. These are not convolutional S models. Its appendix also warns
  that its generic six-layer CIFAR results are not representative of optimal
  performance.
- *WARP Logic Neural Networks* presents CIFAR-10 learning curves for dense and
  custom convolutional architectures but does not tabulate a final,
  paper-faithful LogicTreeNet-S test result.
- *From MNIST to ImageNet: Understanding the Scalability Boundaries of
  Differentiable Logic Gate Networks* reports as much as 65.23% for a
  minimally modified convolutional M configuration. This is neither S nor an
  exact reproduction of the original M protocol; the original
  LogicTreeNet-M result is 71.01%.

## Existing CoverageDLGN Small pilot

The current three-seed, 20K-step pilot used the paper-faithful S architecture
but a deliberately short training budget:

| Topology | Mean hardened test accuracy |
|---|---:|
| Random | 56.14% |
| Semantic-channel V4 | 56.37% |
| Paired V4 − random | +0.23 percentage points |

The seed-level V4 effects were mixed. This pilot is suitable for screening the
topology mechanism, but it is not a reproduction of the fully trained
LogicTreeNet-S result. It must not be used to claim that V4 beats either the
60.38% published result or the 58.39% two-stage reproduction.

## Why apparently similar baselines differ

Before comparing two accuracy values, check all of the following:

1. **Architecture family:** dense DLGN, convolutional LogicTreeNet,
   truth-table network, or higher-input LUT network.
2. **Meaning of the size label:** `S`, `M`, and `L` do not denote common gate
   budgets across papers.
3. **Gate budget:** reported networks range from thousands to tens of millions
   of gates.
4. **Input encoding:** threshold count, fixed versus learnable thresholds,
   per-channel handling, precision, and fixed feature detectors.
5. **Training budget:** iterations or epochs, optimizer, learning rate,
   weight decay, scheduler, and batch size.
6. **Data protocol:** train/validation split, augmentation, normalization, and
   checkpoint-selection procedure.
7. **Evaluation mode:** relaxed versus hardened/discrete inference.
8. **Reported statistic:** validation or test accuracy, best checkpoint,
   final checkpoint, last-checkpoint average, and number of seeds.
9. **Additional supervision:** teacher supervision is used by the larger
   LogicTreeNet B/L/G models.
10. **Circuit accounting:** learned gates, fixed preprocessing gates,
    spatially instantiated operations, synthesized LUTs, and simplified gates
    are not interchangeable cost measures.

## Comparison rule for future CoverageDLGN experiments

A headline accuracy comparison should use the same:

- dataset and train/validation/test split;
- input encoding and augmentation;
- LogicTreeNet architecture and output head;
- learned and deployed gate budget;
- optimizer, number of updates, and batch size;
- checkpoint-selection rule;
- hardened evaluation procedure; and
- seed set.

If any of these differ, report the result as contextual evidence or an
ablation, not as a direct improvement over the published baseline.

## Local paper sources

- `pdfs/convolutional_differentiable_logic_gate_networks.pdf`
- `pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf`
- `pdfs/deep_differentiable_logic_gate_networks.pdf`
- `pdfs/light_differentiable_logic_gate_networks.pdf`
- `pdfs/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networks.pdf`
- `pdfs/warp_logic_neural_networks.pdf`
- `pdfs/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.pdf`
- `pdfs/fitting_multilinear_polynomials_for_logic_gate_networks.pdf`
- `pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf`
- `pdfs/silicon_aware_neural_networks.pdf`
- `pdfs/a_scalable_interpretable_verifiable_differentiable_logic_gate_convolutional_neural_network_architecture_from_truth_tables.pdf`
- `pdfs/from_mnist_to_imagenet_understanding_the_scalability_boundaries_of_differentiable_logic_gate_networks.pdf`
