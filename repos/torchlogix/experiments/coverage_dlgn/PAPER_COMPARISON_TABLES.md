# CoverageDLGN paper comparison tables and experiment plan

**Created:** July 24, 2026  
**Status:** active experiment ledger; populate in place as matched runs finish

This is the canonical working document for the accuracy tables planned for the
CoverageDLGN paper. It distinguishes results produced by our controlled
TorchLogix protocol from numbers reported under other papers' protocols.

## Notation and comparison rules

Every accuracy value must carry one of these provenance labels directly in
its table cell:

- `[TRIED]` is a valid local exploratory, pilot, short-schedule, or superseded
  result that is not eligible as the final comparison value.
- `[REPRODUCED]` is a local result from a paper-faithful implementation and
  protocol.
- `[ADAPTED]` is a local result from a published method adapted to our common
  target architecture or protocol.
- `[OUR-FINAL]` is a frozen final CoverageDLGN result.
- `[REPORTED]` is copied from the cited paper and was not produced locally.
- `[PENDING]` means that the planned local experiment has not finished.
- `[N/A]` means that no valid value exists for that side of the comparison.

`A / R` columns retain the achieved-local/reported-paper ordering, but the
labels are mandatory even when that ordering appears obvious. A local failed
or interrupted attempt is recorded in the experiment log and never receives
an accuracy value. Each `[REPRODUCED]`, `[ADAPTED]`, or `[OUR-FINAL]` value
must point to its machine-readable run summary when populated. Each
`[REPORTED]` value must point to a PDF table or figure in the provenance
ledger.

- A reported architecture or protocol that is not an exact match is context,
  not evidence that one method beats another.
- `[N/A]` on the reported side means that no paper-reported result exists for
  that method and cell.
- Gate count means total deployed logic gates or, for convolutional networks,
  spatially instantiated gate operations. It never means layer width.
- A primary improvement claim may compare only `A` values produced with the
  same data split, encoding, augmentation, gate budget, training effort,
  checkpoint-selection rule, and hardened evaluator.
- Final central cells report mean, standard deviation, and paired 95%
  confidence intervals. Reported values retain the statistic used by their
  source paper.
- Each dataset--architecture cell contains one CoverageDLGN row: the best
  configuration selected by hardened validation accuracy for that cell.
  Alternative configurations remain in the experiment history, not the paper
  table.
- S/M/L labels are local to an architecture family. Gate counts are always
  printed because size labels are not comparable across papers.
- Bold type in the final manuscript will identify the best protocol-matched
  `A` result, never the largest `R-only` value.

## Table 1: Dense MNIST and Fashion-MNIST

Target architecture: six layers, 48K total rank-2 gates, and 768K raw gate
logits during training.

| Method | Matched gates | MNIST A / R | Fashion-MNIST A / R | Match status |
|---|---:|---:|---:|---|
| Deep DLGN fixed random | 48K | **[REPRODUCED] 97.090 +/- 0.180% (n=5) / [REPORTED] 97.69%** | **[TRIED] frozen validation 87.477 +/- 0.183% (n=5; test pending) / [REPORTED] 87.17%** | Exact architecture; MNIST source: `summary/table1_mnist_final.json`; Fashion final validation complete |
| Mommen, \(N_c=16\) | 48K | [ADAPTED] 98.084 +/- 0.066% (n=5) / [REPORTED] 98.14% `[12K, nonmatched]` | [TRIED] frozen validation 88.361 +/- 0.107% (n=3; test pending) / [REPORTED] 87.16% `[8K, nonmatched]` | Adapted to 48K; MNIST source: `summary/table1_mnist_final.json`; Fashion selected \(N_c=8\), depth 3, final validation complete |
| LILogicNet Top-32 | 48K | [ADAPTED] 98.124 +/- 0.029% (n=5) / [REPORTED] 98.95 +/- 0.09% `[32K, nonmatched]` | [TRIED] frozen validation 89.606 +/- 0.118% (n=3; test pending) / [REPORTED] 90.26 +/- 0.11% `[64K, nonmatched]` | Adapted to 48K; MNIST source: `summary/table1_mnist_final.json`; Fashion Top-32, depth 2, tau 30 final validation complete |
| BitLogic best-of-space | 48K | [ADAPTED] 98.204 +/- 0.042% (n=5) / [REPORTED] 97.84 +/- 0.04% `[128K total, nonmatched]` | [TRIED] frozen validation 90.450 +/- 0.277% (n=3; test pending) / [REPORTED] 89.16 +/- 0.08% `[128K total, nonmatched]` | Adapted to 48K; MNIST source: `summary/table1_mnist_final.json`; Fashion final validation complete |
| **CoverageDLGN** | **48K** | **[OUR-FINAL] 97.500 +/- 0.099% (n=5) / [N/A]** | **[TRIED] frozen validation 87.873 +/- 0.271% (n=5; test pending) / [N/A]** | Exact target; MNIST source: `summary/table1_mnist_final.json`; Fashion paired validation gain +0.397 pp, 95% CI [+0.111, +0.683] |

Mommen \(N_c=16\) adds approximately 32 routing logits per gate during
training but retains one hard predecessor per gate input after training.
The selected Fashion-MNIST \(N_c=8\) variant uses 1.536M total trainable
parameters, including 768K training-only routing parameters; its three final
seeds averaged 24.04 minutes and peaked at 0.498 GiB allocated GPU memory.
The selected Fashion-MNIST LILogicNet variant uses 3.84M total trainable
parameters, including 3.072M training-only routing parameters; its three
final seeds averaged 70.84 minutes and peaked at 4.676 GiB.
The adapted Fashion-MNIST BitLogic variant also uses 3.84M total trainable
parameters, including 3.072M training-only routing parameters; its three
final seeds averaged 88.94 minutes and peaked at 3.557 GiB.
On the locked MNIST test set, CoverageDLGN improves its paired fixed-random
baseline by 0.410 percentage points (95% Student-t confidence interval
[+0.108, +0.712] pp). All five methods were evaluated once using the
checkpoint selected solely by hardened validation accuracy.

## Table 2: Dense CIFAR-10 S/M/L

| Architecture | Target gates | Raw training parameters | Method | A / R accuracy | Reported configuration |
|---|---:|---:|---|---:|---|
| S: 4 x 12K | 48K | 0.768M | Deep DLGN random | **[TRIED] 49.06% / [REPORTED] 51.27%** | Exact 48K architecture |
|  |  |  | LILogicNet | [PENDING] / [REPORTED] 55.11% | 8K gates, nonmatched |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates under the BitLogic protocol, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | **[TRIED] 52.36% / [N/A]** | Exact 48K target |
| M: 4 x 128K | 512K | 8.192M | Deep DLGN random | **[TRIED] 54.03% / [REPORTED] 57.39%** | Exact 512K architecture |
|  |  |  | LILogicNet | [PENDING] / [REPORTED] 57.66 +/- 0.17% | 64K gates, nonmatched |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | **[TRIED] 58.28% / [N/A]** | Exact 512K target |
| L: 5 x 256K | 1.28M | 20.48M | Deep DLGN random | [PENDING] / **[REPORTED] 60.78%** | Exact 1.28M architecture |
|  |  |  | LILogicNet-L | [PENDING] / [REPORTED] 60.98 +/- 0.19% | 256K gates, nonmatched |
|  |  |  | WARP-LUT | [PENDING] / [REPORTED] 52.12 +/- 0.01% | 128K total gates, nonmatched |
|  |  |  | BitLogic best-of-space | [PENDING] / [REPORTED] 58.06 +/- 0.14% | 128K total rank-4 gates, nonmatched |
|  |  |  | **CoverageDLGN** | [PENDING] / [N/A] | Exact 1.28M target |

The 128K, 256K, and 384K compression points will be reported in a separate
Pareto table after their validation sweep.

## Table 3: Convolutional CIFAR-10 S/M/L

Gate count follows the convolutional DLGN paper's spatially instantiated
operation accounting.

| Architecture | Gate operations | Method | A / R accuracy | Match status |
|---|---:|---|---:|---|
| LogicTreeNet-S | 0.40M | Original fixed routing | [TRIED] 56.14% pilot / **[REPORTED] 60.38%** | Exact architecture; pilot is short |
|  | 0.40M | Two-stage unit tying, 30% | [PENDING] / [REPORTED] 56.70 +/- 0.08% | Exact S; approximately 0.70 x gates after tying |
|  | 0.57M | Conv. TTNet-S | [N/A] / [REPORTED] 50.10% | Different truth-table architecture |
|  | 0.40M | Light/IWP-LogicTreeNet | [PENDING] / [N/A] | Exact adaptation; no published S result |
|  | 0.40M | WARP-LogicTreeNet | [PENDING] / [N/A] | Exact adaptation; no published S result |
|  | **0.40M** | **CoverageDLGN-Channel** | [TRIED] 56.37% pilot / [N/A] | Exact architecture; pilot is short |
| LogicTreeNet-M | 3.08M | Original fixed routing | [PENDING] / **[REPORTED] 71.01%** | Exact architecture |
|  | 3.08M | Two-stage unit tying, 30% | [PENDING] / [REPORTED] 70.77 +/- 0.07% | Exact M; approximately 0.70 x after tying |
|  | approximately 3.08M | Scalability-boundaries CDLGN-M | [N/A] / [REPORTED] 65.23% | Minimally modified M protocol |
|  | 189M | Conv. TTNet-L | [N/A] / [REPORTED] 70.75% | Different, much larger architecture |
|  | 3.08M | Light/IWP-LogicTreeNet | [PENDING] / [N/A] | Exact adaptation |
|  | 3.08M | WARP-LogicTreeNet | [PENDING] / [N/A] | Exact adaptation |
|  | **3.08M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | Existing 61.56% is `[TRIED]` 5K validation only |
| LogicTreeNet-L | 28.9M | Original fixed routing | [PENDING] / **[REPORTED] 84.99%** | Exact replication requires 5-bit input and teacher |
|  | 189M | Conv. TTNet-L | [N/A] / [REPORTED] 70.75% | Different architecture |
|  | 3.08M | Scalability-boundaries CDLGN-M | [N/A] / [REPORTED] 65.23% | M architecture, not L |
|  | 28.9M | Light/IWP-LogicTreeNet | [PENDING] / [N/A] | New L adaptation |
|  | 28.9M | WARP-LogicTreeNet | [PENDING] / [N/A] | New L adaptation |
|  | **28.9M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | Requires a paper-faithful L baseline first |

## Table 4: Dense CIFAR-100 S/M/L

The initial compact CIFAR-100 ladder follows BitLogic's two-layer common
protocol. The paper reports width per layer, so total gate count is twice its
reported width.

| Architecture | Total gates | Fan-in | Method | A / R accuracy | Source status |
|---|---:|---:|---|---:|---|
| S: 2 x 4K | 8K | 2 | DiffLogic | [PENDING] / [REPORTED] 7.49 +/- 0.21% | BitLogic common-protocol reproduction |
|  | 8K | 2 | WARP-LUT | [PENDING] / [REPORTED] 7.00 +/- 0.04% | BitLogic reproduction; C100 extrapolation |
|  | 8K | 2 | LILogicNet | [PENDING] / [REPORTED] 7.63 +/- 0.01% | BitLogic reproduction; C100 extrapolation |
|  | 8K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 10.19 +/- 0.06%** | Exact BitLogic configuration |
|  | **8K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates; lower fan-in |
| M: 2 x 16K | 32K | 2 | DiffLogic | [PENDING] / [REPORTED] 10.61 +/- 0.08% | BitLogic reproduction |
|  | 32K | 2 | WARP-LUT | [PENDING] / [REPORTED] 10.46 +/- 0.00% | BitLogic reproduction |
|  | 32K | 2 | LILogicNet | [PENDING] / [REPORTED] 10.62 +/- 0.12% | BitLogic reproduction |
|  | 32K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 14.06 +/- 0.04%** | Exact BitLogic configuration |
|  | **32K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates |
| L: 2 x 64K | 128K | 2 | DiffLogic | [PENDING] / [REPORTED] 14.64 +/- 0.09% | BitLogic reproduction |
|  | 128K | 2 | WARP-LUT | [PENDING] / [REPORTED] 14.43% | Single seed in BitLogic |
|  | 128K | 2 | LILogicNet | [PENDING] / [REPORTED] 14.54 +/- 0.04% | BitLogic reproduction |
|  | 128K | 4 | BitLogic best-of-space | [PENDING] / **[REPORTED] 18.82 +/- 0.09%** | Exact BitLogic configuration |
|  | **128K** | **2** | **CoverageDLGN** | [PENDING] / [N/A] | Matched gates |

Additional reported-only dense CIFAR-100 references:

| Method | Accuracy | Gates or parameters | Status |
|---|---:|---:|---|
| Scalability-boundaries dense DLGN | [REPORTED] 22.54 +/- 0.26% | 6 x 64K = 384K gates | Reported only |
| Multilinear Soft-Mix | [REPORTED] 27.92 +/- 0.43% | 6 x 256K = 1.536M gates; 16 parameters/gate | Reported only |
| Multilinear-CovJac | [REPORTED] 28.37 +/- 0.22% | 1.536M gates; 4 parameters/gate | Reported only |
| Multilinear-CovJac large | [REPORTED] 32.72 +/- 0.09% | 6 x 1.28M = 7.68M gates; 4 parameters/gate | Reported only |

## Table 5: Convolutional CIFAR-100 S/M/L

These are transferred LogicTreeNet scales. No local paper reports exact
CIFAR-100 S and L results. Estimated operation counts apply Light's tenfold
final-layer expansion for 100 classes to the rounded CIFAR-10 gate counts.

| Architecture | Estimated gate operations | Method | A / R accuracy | Status |
|---|---:|---|---:|---|
| C100 LogicTreeNet-S | approximately 0.49M | Original/Soft-Mix routing | [PENDING] / [N/A] | New S transfer |
|  | approximately 0.49M | Light/IWP | [PENDING] / [N/A] | No published C100-S result |
|  | approximately 0.49M | WARP | [PENDING] / [N/A] | No published C100-S result |
|  | **approximately 0.49M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | New matched experiment |
| C100 LogicTreeNet-M | approximately 3.82M | Original/Soft-Mix | [PENDING] / [REPORTED] 29.0 +/- 0.6% | R value is fivefold-depth M, not base M |
|  | approximately 5 x M depth | Light/IWP | [PENDING] / **[REPORTED] 38.2 +/- 0.3%** | Fivefold-depth M, nonmatched |
|  | M-derived | Scalability-boundaries CDLGN | [N/A] / [REPORTED] 30.96% | Minimally modified M |
|  | approximately 3.82M | WARP | [PENDING] / [N/A] | No published exact M result |
|  | **approximately 3.82M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | New matched experiment |
| C100 LogicTreeNet-L | approximately 34.8M | Original/Soft-Mix | [PENDING] / [N/A] | New L transfer |
|  | approximately 34.8M | Light/IWP | [PENDING] / [N/A] | No published C100-L result |
|  | approximately 34.8M | WARP | [PENDING] / [N/A] | No published C100-L result |
|  | **approximately 34.8M** | **CoverageDLGN-Channel** | [PENDING] / [N/A] | Requires teacher/input protocol decision |

Convolutional CIFAR-100 S/L therefore cannot yet contain three exact
paper-reported architecture matches. Any dense reported numbers shown beside
them in the manuscript must remain explicitly `R-only`.

## Reported-value provenance

- Deep DLGN: MNIST Table 4, CIFAR-10 Table 5, and architecture Table 6.
- Mommen connection optimization: MNIST and Fashion-MNIST Figures 3 and 4.
- LILogicNet: Tables 3 and 6.
- BitLogic: shared-protocol Table 6 and method mapping Table 8.
- Convolutional DLGN: CIFAR-10 Table 1 and architecture Appendix A.1.
- Two-stage unit tying: CIFAR-10(M) Table 1 and CIFAR-10(S) Table 7.
- Light DLGN: CIFAR-100 fivefold-depth comparison in Figure 8.
- Scalability-boundaries study: CIFAR-10/100 Tables 11 and 12.
- Multilinear-CovJac: cross-dataset Table 1 and scaling Table 15.

Before using a reported value in a manuscript, record its PDF filename, table
or figure number, metric definition, seed count, and whether its gate count is
width, learned gates, spatial operations, or synthesized gates.

## Experiment plan

The goal is one best CoverageDLGN result per dataset and architecture, not
multiple reporting variants. Hyperparameters may be selected independently
for every cell, but the implemented CoverageDLGN mechanism must not change.

### Frozen run-count policy

- Fixed random and CoverageDLGN use five full seeds on every promoted central
  cell.
- Locally reproduced or adapted paper-derived comparators use three full
  seeds. A paper-prescribed recipe is preferred; otherwise a one-seed short
  screen advances only one setting to the three long runs.
- The already-running MNIST Table 1 queue is the only locked exception: all
  five methods retain five final seeds. Fashion-MNIST uses five seeds for
  random/CoverageDLGN and three for Mommen/LILogicNet/BitLogic.
- Dense CIFAR-10 and CIFAR-100 run Mommen for three seeds at each feasible
  S/M/L scale. LILogicNet is run locally for three seeds on S and is
  reported-only on M/L. BitLogic and WARP-LUT use reported values.
- Promoted convolutional cells run fixed random and CoverageDLGN for five
  seeds and Light/IWP and WARP for three seeds. Mommen, LILogicNet, and
  BitLogic are not treated as convolutional comparators unless a separately
  justified generic extension is implemented.
- CIFAR-10 compression points use three paired random/CoverageDLGN seeds.
  Only the selected Pareto crossing is extended to five final seeds.
- Convolutional L and CIFAR-100 M/L obey the promotion/kill conditions below.
- Every numeric table value carries the provenance labels defined above.

### Selection protocol used for every CoverageDLGN cell

1. Rebuild the topology for the exact dataset, encoding, width, depth, and
   topology seed.
2. Search only existing method controls, such as candidate-pool size, swap or
   long-range fraction, novelty/overlap weights, and the supported raw,
   Light/IWP, or WARP gate parameterization. Record the selected
   parameterization in the table.
3. Do not introduce a new score term, dataset-coded branch, layer-specific
   exception, or topology mechanism during the search.
4. Do not select a favorable topology or training seed. Every candidate uses
   the same predeclared paired pilot seeds.
5. Use topology-only analysis to reject invalid, excessively slow, or
   degenerate configurations, but select accuracy configurations by mean
   hardened validation accuracy rather than maximum ancestry coverage.
6. Use a multi-fidelity search:
   - one-seed CUDA smoke and short screen for the broad candidate set;
   - three paired 20K validation seeds for the leading candidates;
   - freeze the best mean hardened-validation configuration; and
   - train the frozen winner for the final seed set and full schedule.
7. Keep the held-out test set locked until the winner and every comparator are
   frozen.
8. Preserve every attempted configuration, learning curve, failure, and
   selection decision in the experiment history. Only the winner populates
   the CoverageDLGN row.
9. Give reproduced competitors a documented validation search or their
   source paper's frozen recipe. Do not give CoverageDLGN more selection seeds
   or optimizer updates than a matched trainable comparator without reporting
   the difference.
10. After the Table 1 Fashion-MNIST cell, use a paper-prescribed fixed
    LILogicNet recipe when directly applicable. Otherwise use the short screen
    to choose one LILogicNet setting and run three paired 20K seeds only for
    that setting. Report this smaller comparator-search budget explicitly;
    do not multiply or reduce the trainable-parameter count based on the
    number of independent runs.

### Plan for Table 1: Dense MNIST and Fashion-MNIST

1. Audit the existing 48K random and CoverageDLGN artifacts; retain an
   achieved value only when its protocol matches the final table.
2. Implement and test Mommen \(N_c=8/16\), LILogicNet Top-32, and the matched
   BitLogic coordinates.
3. Use MNIST CUDA smokes to validate every new implementation.
4. Search CoverageDLGN independently on MNIST 48K and Fashion-MNIST 48K using
   the common selection protocol above.
5. Run the same pilot and selection effort for the reproduced comparators.
6. Retain five final seeds for every frozen MNIST row. On Fashion-MNIST,
   train five final random/CoverageDLGN seeds and three final
   Mommen/LILogicNet/BitLogic seeds.
7. Evaluate test once and replace the current CoverageDLGN values only when
   the newly selected configurations are protocol-complete.

**Completion condition:** each dataset has achieved results for fixed random,
at least three paper-derived comparators, and the single best CoverageDLGN
configuration.

### Plan for Table 2: Dense CIFAR-10 S/M/L and compression

1. Retain the existing S and M results as incumbent CoverageDLGN candidates.
2. Add and test 128K, 256K, and 384K four-layer compression architectures.
3. Search CoverageDLGN independently for S, 128K, 256K, 384K, M, and L.
4. Run fixed random/CoverageDLGN for five seeds and Mommen for three seeds at
   each feasible S/M/L scale. Run LILogicNet for three seeds on S; retain
   LILogicNet M/L, WARP-LUT, and BitLogic as explicitly reported-only values.
5. Use three paired 20K seeds for selection at every budget.
6. Run five full fixed-random/CoverageDLGN seeds for S, M, the selected
   compression crossing, and L after a timing/memory feasibility run.
7. Select the smallest CoverageDLGN budget that matches the larger random
   model within a predeclared 0.3-percentage-point non-inferiority margin.

**Completion condition:** S/M/L contain the five-seed local primary
comparison, the three-seed Mommen adaptation, and at least three clearly
labelled local or reported paper-derived comparisons. The table or companion
figure demonstrates the best observed accuracy--gate-count Pareto frontier.

### Plan for Table 3: Convolutional CIFAR-10 S/M/L

1. Finish a full-schedule paper-faithful fixed-routing S baseline.
2. Search the unchanged convolutional CoverageDLGN mechanism independently
   for S, M, and L; V4 is the incumbent, while V5 and coverage--reuse remain
   separate negative methods and are not candidate configurations.
3. Compare raw, Light/IWP, and WARP parameterizations within the allowed
   CoverageDLGN search and record the winning choice.
4. Reproduce Light/IWP-LogicTreeNet and WARP-LogicTreeNet for three final
   seeds on each promoted scale.
5. Retain two-stage unit tying as a clearly labelled reported value unless a
   later paper need justifies local reproduction.
6. Run three paired 20K selection seeds, five final random/CoverageDLGN
   seeds, and three final Light/IWP/WARP seeds for S and M.
7. Start L only after reproducing its 5-bit input, preprocessing, teacher
   supervision, and gate accounting, followed by a timing/memory feasibility
   run.

**Promotion condition for L:** M must show a positive mean CoverageDLGN gain
over its matched fixed-routing baseline. Otherwise L remains reported-only.

### Plan for Table 4: Dense CIFAR-100 S/M/L

1. Audit the BitLogic two-layer S/M/L common protocol and gate accounting.
2. Retain DiffLogic, WARP-LUT, and BitLogic best-of-space as reported values;
   run a three-seed local LILogicNet adaptation on S and retain its M/L
   reported values.
3. Search rank-2 CoverageDLGN independently at 8K, 32K, and 128K total gates.
4. Keep fan-in and operation count visible because BitLogic best-of-space is
   rank 4.
5. Use five final seeds for fixed random/CoverageDLGN and three for Mommen at
   each scale, subject to timing feasibility.
6. Retain the 384K and Multilinear-CovJac values as reported-only unless their
   exact architectures are separately reproduced.

**Completion condition:** all S/M/L rows contain a five-seed local primary
comparison, a three-seed Mommen adaptation, and at least three clearly
labelled local or reported paper-derived comparators.

### Plan for Table 5: Convolutional CIFAR-100 S/M/L

1. Finalize the 100-class output-head definition and replace estimated gate
   counts with exact learned-gate and spatial-operation counts.
2. Implement and smoke-test the transferred S architecture.
3. Search CoverageDLGN independently for S using random, raw/Light/WARP
   parameterization candidates and three paired 20K validation seeds.
4. Populate five-seed fixed-routing/CoverageDLGN rows and three-seed
   Light/IWP/WARP adapted comparator rows.
5. Promote M only after S produces a positive signal; promote L only after M.
6. For every promoted scale, run five final random/CoverageDLGN seeds, three
   final Light/IWP/WARP seeds, and one locked test evaluation after selection.
7. Leave unpromoted scales and nonmatching literature results explicitly
   reported-only.

**Kill condition:** if a scale fails to improve its matched fixed-routing
baseline after the allowed search, do not launch the next, substantially more
expensive scale.

### Supporting analyses after the five tables

1. On the selected dense CIFAR-10 central cell, compare random, butterfly,
   pure greedy coverage, and full CoverageDLGN.
2. Ablate overlap, fan-out preservation, multiscale semantic pairing, and raw
   source versus bit ancestry without using those ablations to rename or
   redefine the selected method.
3. Evaluate all frozen winners on held-out test exactly once.
4. Report mean, standard deviation, paired 95% intervals, learning curves,
   gates, training parameters, routing bits, construction time, peak GPU
   memory, hardened latency, and circuit-export equivalence.
5. Generate accuracy--gate-count and topology-metric--accuracy figures from
   machine-readable summaries.

## Immediate next work

1. Freeze the per-cell selection protocol and search budget in a
   machine-readable manifest.
2. Implement and smoke-test the Table 1 comparators on MNIST.
3. Search and populate Table 1 on MNIST and Fashion-MNIST.
4. Continue with the Table 2 dense CIFAR-10 compression and S/M/L searches.
5. Do not start convolutional CIFAR-10 L or any convolutional CIFAR-100 scale
   until the preceding promotion condition is satisfied.
