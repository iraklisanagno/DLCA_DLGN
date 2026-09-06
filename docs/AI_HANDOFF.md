# AI handoff: MarginSynth DATE project

- Last reconstructed: 2026-09-06
- Git branch at handoff: `mmarginsynth`
- Implementation root: `repos/torchlogix`
- Experiment root: `repos/torchlogix/experiments/marginsynth`

## Executive status

The long-term goal is a reproducible DATE paper about reducing the deployed
hardware cost of hardened differentiable logic gate networks (DLGNs) while
controlling classification damage. The proposed method is MarginSynth:
post-training, data-aware resynthesis guided by teacher decision margins,
explicit global/per-class behavioral budgets, and hardware cost.

The current code and experimental infrastructure are substantial and well
tested, but the present algorithm is a **research no-go for a state-of-the-art
compression claim**:

- a frozen five-seed Fashion-MNIST comparison shows 10% Two-Stage Unit Tying
  removes much more logic at comparable or smaller accuracy loss;
- a frozen three-seed dense CIFAR-10 transfer shows MarginSynth preserves the
  teacher better, but removes less hardware, is about 17 times slower, and
  violates the worst-class guard on one seed; and
- the latest hardware-aware candidate-ranking experiment did not improve the
  prior MarginSynth result, so its transfer freeze is intentionally
  `not-frozen`.

The evidence supports a narrower scientific result: margin and class-aware
guards improve teacher fidelity, but the current per-gate action generation
does not create enough structurally valuable simplification. Do not present
the current method as state of the art or as a completed DATE result.

The recommended next research direction is a deliberately bounded pivot to
**hardware-budgeted, class-robust cone/cut resynthesis** (working name:
MarginSynth-Cut). This is a recommendation, not implemented work. It should
receive one predeclared three-seed go/no-go study rather than another broad
Bayesian search over the current method.

## Current goal

Produce a defensible, reproducible DATE submission that demonstrates a new
DLGN resynthesis method on a competitive CIFAR-10 architecture. The primary
claim must be supported by matched mapped-hardware budgets, behavior guards,
and paired multi-seed evidence against Two-Stage Unit Tying; a separate
end-to-end comparison must cover recent training-time DLGN alternatives.

The immediate goal is narrower: decide whether cone/cut-level structural
actions can convert MarginSynth's demonstrated fidelity advantage into a
material hardware Pareto advantage. Until that predeclared mechanism gate
passes, do not launch final five-seed experiments or draft a SOTA claim.

## Original research question

The source brief is
[`ideas/date_ideas/margin_synth.md`](../ideas/date_ideas/margin_synth.md).
It asks whether a classifier's winner-versus-runner margin can identify safe
approximate rewrites of a hardened DLGN and yield a better
accuracy--circuit-cost Pareto frontier than exact simplification, random or
activation-only pruning, and published simplification methods.

The intended rank-2 rewrite space was:

1. replace a gate with constant 0 or 1;
2. route or invert either input;
3. replace the gate with another of the 16 two-input Boolean functions; and
4. remove newly dead logic after the replacement.

The original method was a sequential greedy circuit editor using bit-packed
calibration traces and exact simulation of only the affected fan-out cone. It
was later expanded into joint GPU resynthesis of all eligible gates, exact
repair, optional recovery, a second pass, Bayesian multi-objective search,
class/activity policies, and operation-aware hardware ranking.

## Repository and system architecture

This is one Git repository. `repos/torchlogix` is a directory in the root
worktree, not a submodule or nested Git repository.

### Repository-level organization

- `pdfs/`: the local paper corpus. These PDFs are the primary related-work
  sources already selected for the project.
- `notes/`: paper-by-paper summaries. Verify important numerical claims
  against the associated PDF before using them in a manuscript.
- `ideas/date_ideas/`: candidate DATE ideas and the original MarginSynth brief.
- `repos/torchlogix/`: the WARP Logic Neural Networks implementation used as
  the software foundation.
- `repos/torchlogix/experiments/marginsynth/`: all MarginSynth methods,
  protocols, configurations, runners, summaries, and research records.
- `repos/torchlogix/experiments/marginsynth/results/`: approximately 27 GB and
  11,800 generated local files at handoff. This directory contains raw JSON,
  CSV, SQLite, checkpoints, circuits, C/Verilog/BLIF, Yosys/ABC logs, and
  manifests. It is intentionally ignored by Git and must be archived to an
  external artifact store before the paper is released.

### Four-stage experimental pipeline

```text
ordinary GPU DLGN training
        |
        v
hardening -> Circuit export -> exact function-preserving simplification
        |
        v
MarginSynth optimization -> hardening -> exact repair -> independent guard
        |
        v
exact clean-up -> compiled-C equivalence -> Yosys -> Berkeley ABC
```

1. **Source training.** Train an ordinary DLGN using task loss. MarginSynth is
   not currently part of source training.
2. **Hard circuit baseline.** Select one Boolean function per rank-2 gate,
   export to TorchLogix's editable `Circuit`, verify predictions, and run exact
   simplification. Approximate methods must not receive credit for this exact
   reduction.
3. **Post-training resynthesis.** Keep the original hardened model as the
   teacher. Optimize eligible internal gate choices using calibration examples,
   harden the result, and restore changes until all declared budgets pass.
4. **Deployment verification.** Export and exactly simplify the selected
   candidate, independently verify compiled-C predictions, and apply the same
   Yosys/ABC commands to every method.

### Model architectures already used

**MNIST development fixture**

- `DlgnMnistTiny`;
- five rank-2 layers of 1,000 gates, 5,000 nominal gates;
- one fixed binary input threshold;
- correctness/mechanism use only, not a paper-quality model.

**Fashion-MNIST paper-small source**

- `DlgnFashionMnistPaperSmall`;
- six rank-2 layers of 8,000 gates, 48,000 nominal gates;
- fixed random connectivity;
- three thresholds (0.25, 0.50, 0.75), yielding 2,352 Boolean input bits;
- raw 16-function logits, Adam, learning rate 0.01, batch size 100, 108,000
  GPU updates;
- four internal layers (32,000 gates) are eligible for joint resynthesis; the
  first and final logic layers remain fixed.

**Dense CIFAR-10 scalability source**

- `DlgnCifar10Medium`;
- four rank-2 layers of 128,000 gates, 512,000 nominal gates;
- fixed standard-random connectivity and three-bit input encoding;
- 108,000 GPU updates, batch size 100, learning rate 0.01;
- the two middle layers (256,000 gates) are eligible for resynthesis;
- three trained seeds exist, with validation accuracies 54.64%, 55.16%, and
  53.98%.

The dense CIFAR source is useful for same-checkpoint scaling but is not a
competitive end-to-end architecture. A final paper needs an adequately trained
convolutional DLGN/LogicTreeNet-class source.

### Main implementation components

- `src/torchlogix/circuit.py`: editable circuit IR, simplification,
  simulation, serialization, C, and Verilog.
- `src/torchlogix/models/dense.py`: source model definitions, including the
  Fashion and dense CIFAR classes above.
- `experiments/marginsynth/trace.py`: packed calibration values, scores,
  margins, and graph indices.
- `experiments/marginsynth/rewrites.py`: safe rewrite representation,
  application, undo, and serialization.
- `experiments/marginsynth/incremental.py`: exact affected-cone simulation.
- `experiments/marginsynth/search.py` and `search_v2.py`: original sequential
  greedy search and structured variant.
- `experiments/marginsynth/circuit_distillation.py`: joint all-eligible-gate
  straight-through resynthesis, action masks, losses, and exact repair.
- `experiments/marginsynth/unit_tying.py`: local Two-Stage Unit Tying baseline.
- `experiments/marginsynth/recovery_finetune.py`: locked/unlocked short
  recovery experiments.
- `experiments/marginsynth/bayesian_search.py` and `bayesian_protocol.py`:
  resumable Optuna multi-objective exploration.
- `experiments/marginsynth/liveness_activity.py`: topological/algebraic
  liveness and class/fold activity analysis.
- `experiments/marginsynth/hardware_ranking.py`: structural ABC-gain
  estimator and ranking features.
- `experiments/marginsynth/run_component_protocol.py`: tracked, resumable
  multi-stage orchestration.
- `experiments/marginsynth/verify_checkpoint.py`, `verify_search*.py`, and
  `verify_synthesis.py`: fail-closed semantic verification.

## Important design decisions and rationale

### Exact simplification comes first

The source is hardened and exactly simplified before approximate editing. This
separates ordinary dead-code removal, constant propagation, and deduplication
from MarginSynth's contribution. Exact simplification also runs after an
approximate candidate to clean up logic exposed by its edits.

### Exact and approximate claims are kept separate

MarginSynth preserves behavior on sampled data; it does not generally preserve
the Boolean function on all possible inputs. Never describe a MarginSynth
circuit as functionally equivalent to its teacher unless a separate proof or
exhaustive check supports that statement.

### Data firewall

Training, validation, calibration, and official test partitions are disjoint:

- training data updates the source model;
- validation selects the source checkpoint;
- calibration is subdivided for optimization, repair, and guard;
- test is inaccessible until the entire method and operating point are frozen.

The split seed is normally 2027. Exact indices and SHA-256 hashes are saved.
`evaluate_frozen_test.py` is the only MarginSynth command intended to open the
test set, and it verifies frozen hashes before doing so. The old frozen
five-seed Fashion protocol has already been evaluated; this does not authorize
test access for any new method.

### Teacher behavior is measured explicitly

The primary behavior signal is the original hardened teacher's
winner-versus-challenger margin. Accuracy loss and prediction disagreement are
different metrics: disagreement catches behavioral changes even if aggregate
accuracy happens to remain constant. Global, worst-class, and stratified-fold
guards prevent damage from being hidden in a mean.

### Joint hard-forward/soft-backward resynthesis

Each eligible gate has 16 optimization logits. The forward pass executes a
hard selected Boolean function; the backward pass uses a straight-through soft
surrogate. This updates all eligible gates together without enumerating
`16^N` circuits. The original gate is one of the 16 functions, not an extra
17th choice.

### Exact repair is mandatory

Gradient optimization proposes a hard edit set. A disjoint repair subset then
selects a feasible subset, with complete hard-circuit evaluations and a local
scan because edit interactions are not monotonic. A final independent guard
must pass. Infeasible lower-cost circuits remain useful diagnostics but cannot
be reported as selected results.

### The primary method must remain independent of Unit Tying

Unit Tying is the mandatory same-checkpoint competitor. The primary
MarginSynth path does not use its checkpoint, Gauss--Newton shortlist, Binary
Split refinement, warm start, or fixed tie quota. A hybrid implementation
exists but its approximately 0.34% single-seed ABC improvement is too small and
would be perceived as an incremental Unit Tying extension; keep it as an
ablation only.

### Hardware proxies do not establish hardware gains

Gate count, live gates, operation counts, Yosys generic cells, and ABC AND
nodes are diagnostic metrics. The current host has no characterized Liberty
library in the protocol, so `mapped_area` is deliberately null. A DATE claim
requires identical mapped FPGA LUT or ASIC PPA measurements. Do not relabel the
SkyWater operation sum as mapped area.

### GPU and CPU responsibilities

Source training, joint resynthesis, and recovery must run on CUDA. Packed exact
Boolean simulation, circuit export, compiled-C verification, Yosys, and ABC
are CPU/compiler tasks by design; moving them to GPU is not required for
correctness and would not make external EDA GPU-based.

### Reproducibility before convenience

Runs save resolved configs, source revision, dataset hashes, checkpoints,
learned edits, optimization/repair traces, tool versions, wall times, memory,
subprocess commands, failures, circuits, synthesis logs, and artifact hashes.
Do not delete failed or dominated runs. New protocols and result roots should
be versioned rather than overwriting prior evidence.

## Completed work and empirical conclusions

The detailed chronological ledger is
[`repos/torchlogix/experiments/marginsynth/RUN_LOG.md`](../repos/torchlogix/experiments/marginsynth/RUN_LOG.md).

### Correctness fixture

The MNIST fixture proved end-to-end equivalence among the hardened PyTorch
model, Boolean backend, exported circuit, exact-simplified circuit, and packed
compiled C on all 6,000 validation examples. Exact simplification reduced
3,364 exported primitives to 2,009. A 25-edit greedy run reached 1,939 live
gates with zero calibration disagreement. A more aggressive 100-edit point
reached 1,429 gates and reduced ABC nodes from 8,885 to 7,871, but used a 2%
disagreement budget and lost 0.667 accuracy points on calibration. These are
development results only.

### Frozen five-seed Fashion result

At the selected no-recovery operating point:

| Validation mean | MarginSynth | 10% Unit Tying |
|---|---:|---:|
| accuracy loss | 0.390 pp | 0.360 pp |
| disagreement | 2.460% | 2.700% |
| live-gate reduction from exact | 2.895% | 11.094% |
| ABC reduction from exact | 1.587% | 5.894% |

The paired MarginSynth-minus-Unit-Tying ABC-reduction difference is -4.306
percentage points, with an exact-bootstrap 95% interval approximately
[-4.814, -3.912]. On the sealed test set, source/MarginSynth/Unit-Tying mean
accuracies are 86.204%/85.952%/86.054%. MarginSynth's mean disagreement is
2.484% versus 2.784% for Unit Tying. MarginSynth preserves behavior slightly
better but loses on accuracy and hardware.

### Whole-circuit action-space result

On Fashion seed 0, the full-action gate-count variant retained 2,694 learned
changes, including 1,622 nonconstant changes. It reached 87.233% validation
accuracy, 31,440 live gates, and 96,519 ABC nodes. Unit Tying reached 86.967%,
30,405 live gates, and 94,084 nodes. The constants-only ablation reached 95,173
nodes, better than either full-action proxy. Nonconstant per-gate edits were
therefore not hardware-competitive.

### Recovery and repeated resynthesis

Aggressive MarginSynth plus 5,000 recovery updates reached 86.600%, 27,189
live gates, and 87,568 ABC nodes. It was smaller than Unit Tying but 0.367
accuracy points worse. A second pass reached 87.067%, 29,812 gates, and 94,489
nodes—slightly more accurate than Unit Tying and with fewer live gates, but 405
more ABC nodes and an infeasible worst-class guard. Recovery is not a reliable
source of dominance.

### Bayesian Fashion result

The four-case study completed 160/160 acquisition trials; 29 were behaviorally
feasible and 27 were exactly promoted. Guarded-constrained trial 28 is the
strongest single-seed development point: 27,493 live gates and 91,919 ABC nodes
versus Unit Tying's 30,405 and 94,084. It is 2.30% lower in ABC nodes than Unit
Tying and passes the declared worst-class guard, but takes 28.73 seconds versus
2.02 seconds. It retains 1,626 constants and 4,602 routing/inversion actions;
no alternative binary function survives.

This result is not a paper claim: it was selected by seed-0 exploration, has no
five-seed/test confirmation for this protocol, and did not transfer into a
CIFAR hardware win.

### Dense CIFAR transfer

Across three 512K-gate seeds:

| Mean | MarginSynth | 10% Unit Tying |
|---|---:|---:|
| source accuracy loss | 0.320 pp | 0.567 pp |
| disagreement | 2.093% | 5.720% |
| live-gate reduction | 3.606% | 9.508% |
| ABC reduction | 1.110% | 4.403% |
| application time | 199.07 s | 11.75 s |

MarginSynth gains 0.247 accuracy points and 3.627 disagreement points relative
to Unit Tying, but uses about 49,003 more ABC nodes and is 17.15 times slower.
Its seed-2 second pass fails the 1.5-point worst-class guard.

### Latest hardware-aware no-go

The operation/constant-propagation/fan-out estimator ranked the small set of
observed ABC reductions well and predicted the held-out Unit-Tying reduction
within 2.82%. Nevertheless, the new feasible results had 1,475,742 and
1,480,257 nodes versus the prior MarginSynth result's 1,472,970 and Unit
Tying's 1,419,811. The selector correctly emitted a `not-frozen` record; no
seed-1/2 transfer or test evaluation was launched. Candidate generation and
repair interaction—not another fit of this estimator—is the remaining
bottleneck.

## Known bugs, issues, and reporting hazards

1. **The central superiority hypothesis currently fails.** Do not turn the
   fidelity advantage into a generic “better overall” claim.
2. **Bayesian baseline denominator is mislabeled.** The Bayesian report calls
   107,369 ABC nodes and 34,740 gates the “original exact” reference. The
   actual exact-simplified baseline is 99,365 nodes and 33,843 gates. Trial
   28's incremental reduction from exact is 7.49% ABC and 18.76% live gates,
   not 14.39% and 20.861%. Its direct comparison with Unit Tying remains valid.
   Fix the report generator/protocol wording and regenerate summaries before
   using these numbers.
3. **Alternative binary gates do not survive selected runs.** All selected
   Bayesian and dense-CIFAR actions are constants or routing/inversion. The
   broad 16-function novelty is implemented but empirically unsupported.
4. **Calibration guards are noisy.** The Fashion independent guard has 1,200
   examples, roughly 120 per class. A one- or two-example change can move a
   worst-class percentage materially. Use cross-fitting and confidence bounds.
5. **Calibration overfitting has been observed.** Several low-cost component
   variants improve their optimization subset but fail the untouched guard.
6. **Only the old Fashion protocol has a five-seed test result.** Bayesian
   trial 28 and dense CIFAR remain development/validation evidence. Do not open
   test data for a replacement method before a new freeze.
7. **The dense CIFAR source is not end-to-end competitive.** Its roughly
   54--55% validation accuracy is below recent compact/learned-connectivity
   results and far below convolutional DLGNs. Same-checkpoint comparisons are
   fair, but broad SOTA claims are not.
8. **Unit-Tying recovery is not fully reproduced.** The local matched 30,000
   update run uses a different optimizer/objective and degraded. It must not be
   described as evidence that the published recovery fails.
9. **Hardware evidence is incomplete.** ABC AIG nodes/levels are not FPGA LUT
   utilization, standard-cell area, timing, or power. Yosys 0.9 and Berkeley
   ABC 1.01 were used and should be upgraded/frozen for a final flow.
10. **Method runtime is poor.** Frozen MarginSynth application is about 14--17
    times slower than local Unit Tying. Common export/synthesis time must be
    reported separately, but does not erase the application-time gap.
11. **Raw results are local only.** The 27 GB result tree is intentionally not
    in Git. A clean clone has protocols and prose results but not checkpoints
    or machine-readable raw evidence. Create an external immutable archive
    with checksums before artifact release.
12. **Some historical prose is stale.** `MARGINSYNTH_COLLEAGUE_PRESENTATION.md`
    describes the Bayesian Fashion seed-0 stage, not the latest conclusion.
    `THOUGHTS_tmp.md` is a historical liveness brainstorming note. Prefer this
    handoff and `RUN_LOG.md` for current status.
13. **Absolute paths occur in generated artifacts.** Reproduction runners are
    generally config-driven, but portable artifact publication should rewrite
    or contextualize machine-specific `/ssd1/...` paths.

No unresolved semantic correctness failure is known in the checked-in code.
The latest full suite reported 3,384 passes, 3,038 skips, and one pre-existing
tensor-copy warning.

## Unfinished work

- No cone/cut resynthesis implementation exists.
- No adequately trained convolutional CIFAR-10 source has been integrated with
  the MarginSynth circuit-editing pipeline.
- No complete matched-cost Unit-Tying frontier exists on the central CIFAR
  source; the current headline comparison uses 10% tying.
- Mind the Gap, optimized connectivity, LILogic, and from-scratch
  Silicon-Aware baselines have not been reproduced in the same final flow.
- No final FPGA LUT mapping or ASIC physical-design comparison exists.
- Compiled-C latency frontiers required by the paper plan are incomplete.
- Trial 28 has not received a frozen five-seed Fashion transfer/test study.
  Running it may be useful as a secondary confirmation, but it will not fix the
  CIFAR/SOTA weakness by itself.
- The Bayesian exact-baseline reporting label/denominator must be corrected.
- Raw artifacts need external archival and a retrieval manifest.
- The proposed cone/cut pivot needs a novelty audit against approximate logic
  synthesis and the newest DLGN/LUT literature before implementation claims.

## Assumptions and durable constraints

- The target is a paper, so every experiment must be reproducible and every
  failed/infeasible run must remain visible.
- Work on branch `mmarginsynth`; do not delete or overwrite previous methods,
  configs, or result directories.
- Use the existing `repos/torchlogix/venv` for Python packages. Do not install
  PyTorch/CUDA packages into the user account. System Yosys/ABC executables are
  separate from the Python environment.
- Training, resynthesis, and recovery runs must execute on an NVIDIA GPU and
  must fail rather than silently fall back to CPU.
- CPU execution is expected for exact Boolean simulation, C compilation,
  Yosys, and ABC.
- Use `DATASET_PATH=/tmp/torchlogix-datasets` unless a new documented durable
  dataset location is intentionally selected.
- CIFAR-10 is the intended central dataset, Fashion-MNIST is secondary, and
  MNIST is a correctness fixture. CIFAR-100 is optional after CIFAR-10 works.
- Preserve the training/validation/calibration/test firewall. Do not use the
  test set for method selection, hyperparameter search, repair, or fallback.
- Report accuracy changes in percentage points and cost changes as percentages
  with an explicit denominator.
- Use identical export, exact simplification, Yosys/ABC, target library, and
  timing settings for methods in a hardware table.
- Do not call a hardware proxy “area,” “LUT count,” or “PPA.”
- Do not call the primary method a Unit-Tying extension or use a Unit-Tying
  warm start. The hybrid remains an ablation.
- Do not launch another Bayesian sweep until a component-level mechanism shows
  a reproducible improvement over the current candidate-generation bottleneck.

## Important files

| File | Purpose |
|---|---|
| [`ideas/date_ideas/margin_synth.md`](../ideas/date_ideas/margin_synth.md) | Original idea, scope, experiment matrix, and initial kill criterion |
| [`experiments/marginsynth/README.md`](../repos/torchlogix/experiments/marginsynth/README.md) | Full method formulation, novelty boundary, workflow, and go/no-go rules |
| [`RUN_LOG.md`](../repos/torchlogix/experiments/marginsynth/RUN_LOG.md) | Authoritative chronological commands, results, hashes, failures, and conclusions |
| [`PAPER_EVALUATION_REFERENCE.md`](../repos/torchlogix/experiments/marginsynth/PAPER_EVALUATION_REFERENCE.md) | Datasets, metrics, fairness rules, comparison grouping, and success criteria |
| [`PAPER_COMPARISON_PLAN.md`](../repos/torchlogix/experiments/marginsynth/PAPER_COMPARISON_PLAN.md) | Five external paper methods and execution roadmap |
| [`CIRCUIT_DISTILLATION_NOVELTY_AUDIT.md`](../repos/torchlogix/experiments/marginsynth/CIRCUIT_DISTILLATION_NOVELTY_AUDIT.md) | Local novelty boundary for joint resynthesis |
| [`CIRCUIT_DISTILLATION_RESULTS.md`](../repos/torchlogix/experiments/marginsynth/CIRCUIT_DISTILLATION_RESULTS.md) | Whole-circuit seed-0 results and failed promotion gate |
| [`RECOVERY_RESULTS.md`](../repos/torchlogix/experiments/marginsynth/RECOVERY_RESULTS.md) | Recovery, repeated resynthesis, and matched Unit-Tying caveats |
| [`BAYESIAN_SEARCH_PROTOCOL.md`](../repos/torchlogix/experiments/marginsynth/BAYESIAN_SEARCH_PROTOCOL.md) | Four-case search design and logging contract |
| [`HARDWARE_AWARE_PROTOCOL.md`](../repos/torchlogix/experiments/marginsynth/HARDWARE_AWARE_PROTOCOL.md) | Latest estimator, fallback, seed-0 result, and no-go decision |
| [`configs/`](../repos/torchlogix/experiments/marginsynth/configs) | Frozen and development configurations; never infer settings from prose when a config exists |
| [`tests/test_marginsynth.py`](../repos/torchlogix/tests/test_marginsynth.py) | Rewrite, trace, distillation, guard, and protocol tests |
| [`tests/test_hardware_ranking.py`](../repos/torchlogix/tests/test_hardware_ranking.py) | Structural estimator and selection tests |
| [`tests/test_dense_transfer_summary.py`](../repos/torchlogix/tests/test_dense_transfer_summary.py) | Dense transfer aggregation and guard behavior |
| [`pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf`](../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf) | Mandatory direct post-training comparison |
| [`pdfs/silicon_aware_neural_networks.pdf`](../pdfs/silicon_aware_neural_networks.pdf) | Training-time hardware-cost comparison and novelty boundary |

## Environment, build, and test commands

Run Python commands from `repos/torchlogix`.

### Existing environment and GPU preflight

```bash
cd repos/torchlogix
nvidia-smi
venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
yosys -V
berkeley-abc -q "version"
```

The recorded environment used Python 3.12.13, PyTorch 2.9.0+cu130, CUDA 13.0,
an NVIDIA RTX PRO 6000 Blackwell Max-Q GPU, Yosys 0.9, and Berkeley ABC 1.01.

If the existing virtual environment needs repair, install only into it:

```bash
venv/bin/python -m pip install -e ".[dev]"
venv/bin/python -m pip install -r experiments/requirements.txt
```

Do not reinstall a different PyTorch build without first recording the current
CUDA/driver compatibility and updating the experiment environment record.

### Focused tests

```bash
cd repos/torchlogix
PYTHONPATH=. venv/bin/pytest -q \
  tests/test_marginsynth.py \
  tests/test_bayesian_summary.py \
  tests/test_dense_transfer_summary.py \
  tests/test_hardware_ranking.py
```

### Full test suite and source checks

```bash
cd repos/torchlogix
PYTHONPATH=. venv/bin/pytest -q
git diff --check
```

`PYTHONPATH=.` is required because the repository exposes `experiments` as a
namespace from its root.

### Reproduce/resume dense source training

This is a long GPU job. Do not run it merely as a smoke test.

```bash
cd repos/torchlogix
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/train_dense_sources.py \
  --protocol experiments/marginsynth/configs/dense_cifar_standard_random_sources.json \
  --sources small_seed0 medium_seed0 medium_seed1 medium_seed2 --resume
```

### Reproduce/resume component or transfer protocol

```bash
cd repos/torchlogix
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/run_component_protocol.py \
  --protocol PATH_TO_PROTOCOL.json --resume
```

Do not attempt seed transfer from the latest hardware-aware freeze record: it
is intentionally `not-frozen`, and the generator correctly rejects it.

### Aggregate the existing dense transfer

```bash
cd repos/torchlogix
PYTHONPATH=. venv/bin/python experiments/marginsynth/summarize_dense_transfer.py \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed0/marginsynth_comparison/trial28_transfer_v1/dense_comparison.json \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed1/marginsynth_transfer/trial28_transfer_v1/dense_comparison.json \
  --comparison experiments/marginsynth/results/dense_cifar_sources/medium_standard_random_seed2/marginsynth_transfer/trial28_transfer_v1/dense_comparison.json \
  --output-dir experiments/marginsynth/results/dense_cifar_sources/transfer_summary/trial28_transfer_v1
```

### Test-set command—restricted use

`experiments/marginsynth/evaluate_frozen_test.py` is documented in the
MarginSynth README. Do not run it for a new method until all seeds, selected
artifacts, operating-point rules, and hashes have been frozen. The existing
Fashion test result is immutable historical evidence and does not justify
reopening test data during development.

## Recommended next 10 concrete steps

1. **Repair reporting first.** Correct the Bayesian “original exact” label and
   denominators, add separate pre-exact/exact/post-MarginSynth reductions, and
   regenerate the machine-readable summary without changing raw runs.
2. **Freeze a new question before coding.** Define MarginSynth-Cut as:
   maximize teacher fidelity and worst-class accuracy subject to explicit
   mapped hardware budgets. Predeclare success as at least 10% lower mapped
   cost than Unit Tying at no more than 0.5 pp loss, or at least 0.5 pp higher
   accuracy at exactly matched cost.
3. **Implement small cone/cut extraction.** Add deterministic extraction of
   two-to-six-input cuts around eligible gates, stable IDs, overlap detection,
   truth-table evaluation, serialization, and exhaustive tiny-circuit tests.
4. **Generate structurally valuable replacements.** Minimize each cut into a
   constant, routed/inverted input, reduced-input LUT, or cheaper Boolean
   expression. Reject candidates without positive local mapped/AIG gain. Keep
   ordinary per-gate actions as controlled ablations.
5. **Add set-aware selection and repair.** Evaluate compatible edit batches or
   a small beam, account for overlapping/reconvergent cones, and repair by
   behavior recovered per unit of hardware saving lost rather than behavior
   risk alone.
6. **Strengthen behavioral validation.** Use stratified cross-fitting and
   per-class confidence bounds; predeclare fallback to the most aggressive
   independently guard-feasible snapshot.
7. **Run a bounded Fashion/CIFAR mechanism study.** Use one Fashion seed and a
   small CIFAR source only to verify correctness, runtime, nonconstant cone
   contribution, and correlation between local cost and final synthesis. Do
   not start another Bayesian sweep.
8. **Integrate a competitive convolutional CIFAR source.** Establish export,
   exact simplification, compiled-C equivalence, and identical Unit-Tying and
   MarginSynth eligibility on a Conv DLGN/LogicTreeNet-class checkpoint.
9. **Run a three-seed go/no-go frontier.** Compare exact, random/sensitivity,
   full Unit-Tying tie ratios, constant-only, and MarginSynth-Cut at matched
   accuracy and matched mapped-cost budgets. Require all guards to pass and a
   meaningful contribution from multi-gate/nonconstant cuts. Also target
   frozen-method runtime within roughly 3x Unit Tying.
10. **Escalate only after the gate passes.** Freeze five seeds and the test
    policy, reproduce the applicable Mind the Gap/LILogic/optimized-connectivity/
    Silicon-Aware baselines, run a modern identical FPGA or ASIC flow, archive
    raw artifacts externally, and then evaluate the sealed test set once. If
    the three-seed gate fails, stop standalone MarginSynth and reposition the
    infrastructure/results as an artifact or negative empirical study.

## Paper comparison structure

Keep comparisons in two groups:

1. **Same-checkpoint post-training:** exact source, MarginSynth, and Two-Stage
   Unit Tying. This is the causal simplification comparison.
2. **End-to-end accuracy--hardware:** Mind the Gap, optimized connections,
   LILogic Net, and Silicon-Aware training, all measured in the same final
   hardware flow. These methods train different source models and must not be
   presented as same-checkpoint ablations.

The default source references are Deep DLGN for dense MNIST/Fashion and
Convolutional DLGN for CIFAR-10. Recheck the literature immediately before a
“state of the art” claim because the field is moving quickly.
