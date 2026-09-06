# CoverageDLGN AI handoff

State reconstructed on **2026-09-06** from the repository, frozen experiment
artifacts, git history, and the prior project discussion. This is an engineering
and research handoff, not a manuscript draft.

## Snapshot

- Repository branch: `coverageDLGN`.
- HEAD before this handoff was written: `65db1ca` (`Add thorough CoverageDLGN
  project evaluation`).
- The branch was clean and **7 commits ahead of `origin/coverageDLGN`** before
  these two handoff files were added. Nothing was pushed in this session.
- No CoverageDLGN training or queue process was running at inspection time.
- The local experiment directory is about 74 GiB. It contains 2,796 ignored
  `.pt` files totaling about 73.2 GiB. These checkpoints are not tracked by git;
  a fresh clone will not contain them.
- Host `nvidia-smi` sees two NVIDIA RTX PRO 6000 Blackwell GPUs with CUDA 13.0.
  At this snapshot GPU 1 was occupied by a VLLM process. The sandboxed Codex
  process could not initialize CUDA even though the host could see it. This is
  an execution-environment issue, not evidence that the repository venv is
  broken. Do not train unless the venv can allocate at least one GPU.

## Current goal

Develop a reproducible DATE 2027 conference contribution showing that deliberately
designed, fixed DLGN connectivity improves hardened accuracy and the
accuracy/circuit-cost Pareto frontier without learned routing or extra deployed
routing state. TorchLogix is the implementation foundation because it supports
dense and convolutional LGNs, raw/WARP/Light LUT parameterizations, learned
binarization, and circuit export.

The original proposal in
[`ideas/date_ideas/coverage_dlgn.md`](../../../../../ideas/date_ideas/coverage_dlgn.md)
was an ancestry-maximizing `coverage_hybrid`. Experiments rejected that precise
mechanism. The current headline candidate is:

> **CoverageDLGN-U2 (`semantic_multiscale_balanced`)**: deterministic semantic
> source ordering, degree-first balanced routing, and multiscale matching-stage
> selection using normalized ancestry novelty, with zero learned routing.

The scientific story is now that unconstrained random routing creates avoidable
degree imbalance and poor semantic propagation. Maximum ancestry coverage by
itself is not sufficient and can be harmful. Dense V3 is the strongest dense
specialization and belongs in the paper as an ablation/specialized upper point;
U2 is the only method currently defensible as the unified dense/convolutional
method.

The work has a publishable core, but the broad DATE claim is not yet finished.
The decisive gaps are multi-seed full convolutional confirmation, compatibility
with a modern gate-training method, rank-4 comparison, and real hardware
synthesis.

## System architecture

### Training and model path

1. `experiments/train.py` parses a JSON configuration, fixes the data split and
   random seeds, builds the binarizer/model, trains relaxed LUTs, periodically
   evaluates hardened validation accuracy, and stores best/final checkpoints.
2. `src/torchlogix/models/dense.py` builds binarization -> flatten ->
   `LogicDense` stacks -> `GroupSum`.
3. `src/torchlogix/models/conv.py` builds binarization -> four depth-3
   `LogicConv2d`/OR-pool stages -> three `LogicDense` classifier layers ->
   `GroupSum`.
4. `src/torchlogix/connections.py` maps the configured strategy to fixed dense
   indices or convolutional channel groups. Convolutional spatial coordinates
   remain sampled by the existing TorchLogix receptive-field path.
5. `src/torchlogix/topology.py` contains deterministic topology construction,
   packed ancestry propagation, semantic input metadata, and topology metrics.
6. `src/torchlogix/parametrization.py` supplies raw, WARP, and Light LUT
   parameterizations plus soft/hard/Gumbel sampling modes.
7. Run artifacts are written under `experiments/coverage_dlgn/results/`; phase
   summaries and frozen manifests are under `summary/` and `logs/`.

### Principal evaluated architectures

| Family | Architecture | Structure | Input encoding |
|---|---|---|---|
| Dense MNIST/Fashion | paper-small and budget ladders | 6 layers; 48K reference is 6 x 8K | MNIST 1 threshold; Fashion 3 thresholds |
| Dense CIFAR-10 S | `DlgnCifar10Small` | 4 x 12K = 48K rank-2 gates | 3 thresholds per RGB channel; 9,216 Boolean inputs |
| Dense CIFAR-10 M | `DlgnCifar10Medium` | 4 x 128K = 512K gates | 3 thresholds per RGB channel |
| Dense CIFAR-10 L | `DlgnCifar10Large` | 5 x 256K = 1.28M gates | 5 thresholds per RGB channel |
| Conv. CIFAR-10 S | `ClgnCifar10PaperSmall` | `k=32`, tau 20; four conv stages and three dense tail layers | paper-faithful 3 thresholds per RGB channel = 9 channels |
| Conv. CIFAR-10 M | `ClgnCifar10PaperMedium` | `k=256`, tau 40; same stage ratios | paper-faithful 9 channels |
| Legacy/WARP-style M | `ClgnCifar10Medium` | `k=256`, same broad backbone | 2 thresholds per RGB channel = 6 channels; separate protocol |
| Dense CIFAR-100 retained | `DlgnCifar100Budget384kDepth3` | 3 x 128K = 384K gates | 3 thresholds per RGB channel |

For paper-faithful convolutional S/M, the four convolutional channel maps are
`9 -> k -> 4k -> 16k -> 32k`. Every block uses a rank-2, depth-3 logic tree,
3x3 receptive field, padding 1, and 2x2 OR pooling. The dense tail widths are
`128k -> 1280k -> 640k -> 320k`, followed by ten-way GroupSum. S has 83,552
learned LUT functions and 874,496 spatial gate applications; M has 668,416
learned LUT functions. Do not call the legacy six-channel models
paper-faithful.

`ClgnCifar10Large` is not a faithful reproduction of the published larger
LogicTreeNet variants: required output scaling, edge/curvature preprocessing,
teacher/distillation choices, and five-bit protocol details are incomplete.

## Current method definitions and why they are frozen

### V3: dense specialization

- Strategy name: `semantic_balanced_hybrid`.
- Semantic first-layer butterfly avoids pairing thresholds of the same raw
  image source and balances predecessor use.
- Deeper layers perform degree-preserving ancestry/novelty swaps.
- Frozen settings used in the central study: candidate pool 8, swap fraction
  0.25, novelty weight 1.0.
- Rationale for preservation: it gives the strongest dense CIFAR-10 frontier.
  The component ablation shows degree balance accounts for about 93% of its
  gain, while swaps add only about 0.04 pp at CIFAR-10 M.

Do not change V3 behavior, defaults, RNG consumption, or old configs. If a new
idea is needed, add a newly named strategy and regression-test V3 bit identity.

### V4: legacy convolutional specialization

- Strategy name: `semantic_channel_hybrid`.
- Applies the V3-style balanced/swap rule to convolutional channel groups while
  leaving spatial receptive-field coordinates unchanged.
- It was initially promising on a legacy six-channel model and is retained as
  a historical ablation. On the corrected nine-channel architecture its gain
  was smaller.

Do not describe V4 as WARP. A separate WARP-style reproduction used the WARP
architecture/training conventions, but the valid V4 attribution comparison is
against its matched random sampler.

### U1: no-swap unified precursor

- Strategy name: `semantic_degree_balanced`.
- Same semantic/degree-balanced base for dense and convolutional layers, no
  ancestry swaps.
- Five-seed convolutional-S validation gain was +0.760 pp with 4/5 wins, but
  its predeclared +1 pp promotion threshold failed.

### U2: current unified method

- Strategy name: `semantic_multiscale_balanced`.
- First layer uses semantic ordering. For image encodings, correlated threshold
  bits map back to the same raw source; the construction avoids semantically
  redundant first-layer pairs.
- Deeper layers choose entire regular matching stages. The selection priority
  is minimum prospective fan-out spread, then maximum normalized ancestry
  novelty, then the nominal local-to-global multiscale order as a tie-break.
- Individual edges are never greedily swapped. This preserves useful regular
  pair sets and exact/near-exact predecessor balance.
- In convolutional layers, U2 chooses channel pairs only; the established
  random spatial sampler is unchanged. The dense classifier tail also uses U2.
- Construction is offline on CPU. Only ordinary fixed indices are deployed;
  there are no learned routing parameters, new inference operators, or new
  gates.
- Current implementation is rank-2 only.

The no-swap/stage-level design is deliberate: V4 swaps changed pair sets but
barely changed ancestry, and V5 greatly improved ancestry statistics without
improving accuracy. Do not reintroduce per-edge greedy ancestry maximization
under the U2 name.

## Completed work

### Engineering and reproducibility

- Added random-unique, local, butterfly, greedy coverage, hybrid, V3, V4, V5,
  coverage/reuse, class-head, U1, and U2 topology variants.
- Added packed `uint64` ancestry, semantic/raw-source ancestry, overlap,
  fan-out, unused-input, duplicate-pair, group-coverage, and convolutional
  diagnostics.
- Integrated fixed topology into dense and convolutional TorchLogix models and
  checkpoint metadata.
- Added CUDA-only two-GPU queueing with refusal of CPU configs, completed-run
  skipping, and preservation of failed/interrupted attempts.
- Added frozen validation manifests and SHA-256 checks before one-time held-out
  evaluation.
- Added learning-curve aggregation, CUDA inference benchmarks, circuit export,
  simplified-IR counts, compiled CPU snapshots, and bit-exact equivalence.
- Completed 110 second-round CUDA runs and 38 third-round CUDA runs, in addition
  to earlier pilots and long LogicTreeNet-M runs.
- The last recorded complete suite result was 3,412 passed, 3,038 skipped, one
  pre-existing warning in 257.62 seconds. The focused method/protocol suite was
  rerun for this handoff: 215 passed, 2 skipped, and one existing warning in
  117.53 seconds.

### Main empirical results

All gains below are hardened accuracy in percentage points. Preserve the
validation/test and provenance labels in the source tables.

| Coordinate | Local result | Interpretation |
|---|---|---|
| Dense CIFAR-10, 48K to 1.28M | V3 gains +3.302, +4.162, +4.650, +4.487, +4.256, and +5.060 pp over random | Strongest evidence; 3-5 paired seeds per point |
| Dense CIFAR-10 M/L | U2 gains +4.557/+4.593 pp over random, n=3 | Unified transfer; U2 tied V3 at M and trails V3 by 0.610 pp at L |
| Dense MNIST 48K | V3 97.500% vs random 97.090%, +0.410 pp, n=5 | Positive but not headline; learned-routing methods are more accurate |
| Dense Fashion 48K | V3 87.102% vs random 86.308%, +0.794 pp, n=5 | Positive but learned-routing methods are more accurate |
| Compressed MNIST 8K | U2 91.937% vs random 91.273%, +0.663 pp, n=3 | Positive paired CI |
| Compressed Fashion 16K | U2 +0.520 pp; V3 +0.717 pp over random | U2 interval crosses zero; V3 retained for Fashion |
| LogicTreeNet-S, 350K | U2 60.630% test vs random 57.370%, +3.260 pp | Full-resource n=1; 20K U2 pilot n=3 was +2.173 pp with 3/3 wins |
| LogicTreeNet-M, 200K | U2 71.650% test vs random 69.570%, +2.080 pp | Matched n=1; numerically +0.64 pp above paper's 71.01%, not a statistical SOTA claim |
| Dense CIFAR-100, 3 x 128K | V3 21.467% test vs random 20.923%, +0.543 pp, n=3 | Validation gain +0.840 pp significant; test CI crosses zero |
| CIFAR-100 U2 pilot | approximately +0.100 pp | Rejected; do not claim universal transfer |

The dense CIFAR-10 frontier supplies the best compression claim: V3 at 128K
gates (53.910%) exceeds random at 384K (53.657%), and V3 at 256K (56.903%)
exceeds the largest tested 1.28M random model (55.960%). State these as
within-the-evaluated-frontier reductions, not universal compression ratios.

On the published LILogicNet protocol, U2 improved matched fixed random by
+3.533 pp at 64K and +4.860 pp at 256K. Top-32 remained more accurate by
5.297/1.837 pp, but used 5x the trainable parameters, 16-17x peak allocated
training memory, and 4.9-8.1x local hardened inference latency. This is an
accuracy-resource Pareto comparison, not accuracy dominance.

The local BitLogic rank-4 transfer is a negative reproduction: relaxed models
learned, but hardened accuracy collapsed and did not match the reported paper
values. Keep `[REPRODUCED-NEGATIVE]` separate from `[REPORTED]`.

### Cost findings

Within matched random/U2 experiments, nominal gate count, LUT parameters,
routing bits, training steps, optimizer, peak GPU allocation, and measured
hardened runtime are equal or indistinguishable. For full LogicTreeNet-S:

- random/U2 training time: 4.975/4.951 h;
- peak training allocation: 1.831 GiB for both;
- topology construction: 0.217/1.372 s;
- CUDA batch-128 inference: 6.852/6.835 ms (measurement noise);
- simplified IR nodes: 252,936/262,260, so U2 is **3.686% larger** in this
  checkpoint-level snapshot;
- circuit equivalence passed; energy and placed/routed hardware were not
  measured.

Equal abstract gate count does not imply equal ASIC/FPGA area. Do not claim a
physical cost win until synthesis exists.

## Known bugs, issues, and stale state

1. **Topology-only U2 semantic-report bug.** `generate_dense_stack()` in
   `src/torchlogix/topology.py` includes V3 and U1, but omits
   `semantic_multiscale_balanced` from three semantic-strategy sets. Therefore
   `analyze_topology.py` can construct U2's first layer but initializes/reports
   ancestry in the wrong bit universe. The trained-model path in
   `models/dense.py` does include U2 and is not affected. Fix the helper and add
   a regression test without changing any trained U2 topology.
2. **Narrative documents contain stale lines.** `DATE_TABLES.md` still ends
   with “The Markdown update remains uncommitted.” Older `README.md` and
   `RESULTS.md` limitation paragraphs still describe work as pending or quote
   the pre-U2 LogicTreeNet-M result. Use later dated sections and frozen JSON as
   authority, then clean the prose in one dedicated documentation commit.
3. **The evidence audit is incomplete for the latest work.** The current
   `audit_evidence_consistency.py`/JSON validates the legacy and second-round S
   evidence but does not assert the August 14 LogicTreeNet-M U2 result or all
   third-round aggregates. Extend it before treating it as a global audit.
4. **No exact training resume.** Checkpoints do not preserve optimizer,
   dataloader, and RNG state sufficiently to prove exact continuation. Partial
   training must not be called an exact resume; archive it under
   `results/failed/` and restart unchanged unless resume equivalence is first
   implemented and tested.
5. **Queue completion is a weak first check.** `run_gpu_queue.py::is_complete`
   checks only `training_config.json`, `environment.json`, `metrics.csv`, and
   `run_summary.json`. A phase needing checkpoints/topology can still require
   its summarizer/freeze validator before being accepted as complete.
6. **Rank limitations.** U2 and other coverage strategies explicitly reject
   rank other than two. Raw LUT parameterization and parts of circuit export
   are also rank-2-only. A rank-4 U2 must use Light/WARP-style compact
   parameterization and requires new export/cost tests.
7. **CIFAR-100 remains weak.** U2 failed promotion, V3's test gain is
   inconclusive, and 12/24-layer fixed-budget networks collapsed to chance even
   though ancestry saturated. Topology alone does not fix this optimization
   problem.
8. **Convolutional statistical evidence is incomplete.** Full S and M U2
   results are one seed. S has a supporting 20K three-seed pilot; M does not.
9. **Protocol mismatch must remain visible.** Paper-faithful S/M architectures
   use nine Boolean channels, but local S uses a 45K/5K selection split and
   350K updates; M was stopped at the predeclared 200K matched boundary. Do not
   present comparisons to reported values as exact reproductions.
10. **No hardware SOTA evidence.** There is no Yosys/ABC or FPGA place-route
    area, Fmax, power, energy, or routing result. Simplified IR is only a proxy.
11. **Ignored checkpoints are machine-local.** Git contains JSON/CSV metadata
    and small test metrics, but not `.pt` checkpoints. Back up or transfer the
    ignored artifacts before moving machines. Do not delete the 74 GiB results
    tree casually.
12. **Research-overfitting risk.** Many MNIST/Fashion/CIFAR-10 variants were
    explored. Future headline method choices must be frozen before new test
    access, preferably on an untouched dataset/protocol.

## Experimental assumptions and constraints

- Never install with system `pip`. Use only `repos/torchlogix/venv`; use `uv
  pip --python venv/bin/python` or `venv/bin/python -m pip`.
- CUDA 13 is required on this machine. `cuda130_pytorch29.def` records the
  PyTorch 2.9/CUDA 13 container basis if the venv cannot access the driver.
- GPU training is mandatory. Check both `nvidia-smi` and an actual CUDA tensor
  allocation through the repository venv. If allocation fails, stop; do not
  fall back to CPU training.
- Use both GPUs for independent runs when free, but assign only runs that fit
  memory. LogicTreeNet-M used about 14.6 GiB; LILogic Top-32 L used about
  24.9 GiB.
- Do not rerun a completed experiment. Reuse frozen results and let the queue
  skip only after checking required artifacts.
- Keep data split, augmentation, gate budget, training effort, initialization,
  and seeds identical within an attribution pair. Usually `data_split_seed` is
  2027 and method pairs use the same run/topology seed.
- A seed is an independent paired realization of initialization, data order,
  and deterministic topology. Report per-seed values, sample standard
  deviation, paired Student-t 95% CI, and wins.
- Select topology/hyperparameters using validation only. Hash/freeze configs and
  checkpoints before the first held-out test evaluation. Never repeatedly query
  test to choose a method.
- Label every number `[OUR]`, `[REPRODUCED]`, `[ADAPTED]`,
  `[REPRODUCED-NEGATIVE]`, or `[REPORTED]`, and label validation versus test.
- Preserve failed/interrupted attempts and explanations. Never overwrite V3,
  V4, U1, U2, old result directories, or frozen summary manifests.
- Do not use `repos/difflogic-light-master` as the convolutional foundation.
- Do not run the Cartesian product of rank x WARP/Light/Gumbel. Connectivity,
  fan-in, parameterization, and sampling are conceptually separate axes, but
  interactions should be checked with a minimal preregistered matrix.

## Source-of-truth order and important files

When prose conflicts, prefer sources in this order:

1. Frozen configuration/checkpoint hashes, per-run `training_config.json`,
   `run_summary.json`, and `test_metrics.json`.
2. Machine-readable phase summaries under `summary/` and evaluation logs under
   `logs/`.
3. Later dated sections in `RESULTS.md`, frozen protocol documents, and git
   history.
4. Presentation tables. These are convenient but contain some stale status
   prose.

Key files:

- [`coverage_dlgn.md`](../../../../../ideas/date_ideas/coverage_dlgn.md): original
  specification and kill criterion.
- [`THOROUGH_SUMMARY.md`](../../../../../THOROUGH_SUMMARY.md): current scientific
  assessment and publication priorities.
- [`../RESULTS.md`](../RESULTS.md): chronological experiment and failure history.
- [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md): operational run history,
  interruptions, and recoveries.
- [`../SECOND_ROUND_CONCLUSIONS.md`](../SECOND_ROUND_CONCLUSIONS.md): consolidated
  U2/S result and claim boundaries.
- [`../THIRD_ROUND_PROTOCOL.md`](../THIRD_ROUND_PROTOCOL.md): frozen dense M/L,
  LILogicNet, and BitLogic protocol round.
- [`../LOGICTREENET_M_U2_PROTOCOL.md`](../LOGICTREENET_M_U2_PROTOCOL.md): the
  one-seed, 200K paper-faithful M U2 run and freeze/test details.
- [`../PAPER_COMPARISON_TABLES.md`](../PAPER_COMPARISON_TABLES.md) and
  [`../DATE_TABLES.md`](../DATE_TABLES.md): comprehensive comparison tables with
  provenance labels.
- [`../UNIFIED_DEGREE_BALANCED.md`](../UNIFIED_DEGREE_BALANCED.md): U1 and the
  diagnostic reason swaps were removed.
- [`../../../src/torchlogix/topology.py`](../../../src/torchlogix/topology.py):
  topology algorithms and metrics.
- [`../../../src/torchlogix/connections.py`](../../../src/torchlogix/connections.py):
  dense/conv connection integration.
- [`../../../src/torchlogix/models/dense.py`](../../../src/torchlogix/models/dense.py)
  and [`../../../src/torchlogix/models/conv.py`](../../../src/torchlogix/models/conv.py):
  evaluated architectures.
- [`../../../tests/test_coverage_topology.py`](../../../tests/test_coverage_topology.py)
  and [`../../../tests/test_experiment_protocol.py`](../../../tests/test_experiment_protocol.py):
  core invariants and protocol tests.
- `summary/second_round_final_dense.json`: MNIST/Fashion/CIFAR-10 dense U2 finals.
- `summary/second_round_convolutional_final.json`: full S accuracy/resources.
- `summary/second_round_convolutional_deployment.json`: S circuit/runtime snapshot.
- `summary/third_round_results.json`: dense M/L and published-protocol results.
- `summary/cifar10_paper_medium_u2_200k_freeze.json` and
  `logs/cifar10_paper_medium_u2_200k/test/test_evaluation_summary.json`: M freeze
  and exactly-once test evidence.
- `summary/training_source_pre_analysis.tar.gz` plus its `.sha256`: archived
  source state for historical runs.

## Environment, build, and test commands

Run these from `repos/torchlogix` unless noted.

```bash
# Inspect the existing environment; activation is optional when using explicit paths.
source venv/bin/activate
python --version

# GPU preflight. Training is forbidden unless the final command succeeds.
nvidia-smi
venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); x=torch.ones(1, device="cuda:0"); torch.cuda.synchronize(); print(x.device)'

# Install/update only inside the repository venv.
uv pip install --python venv/bin/python -e . pytest

# Focused method and protocol tests.
venv/bin/python -m pytest \
  tests/test_coverage_topology.py \
  tests/test_experiment_protocol.py \
  tests/test_parametrization.py \
  tests/test_warp_fig4_protocol.py -q

# Complete suite (historically about 4.3 minutes, with many intentional skips).
venv/bin/python -m pytest tests -q

# Existing evidence audit. It currently needs extension for M U2/third round.
venv/bin/python experiments/coverage_dlgn/audit_evidence_consistency.py

# Topology-only analysis example; no dataset or GPU needed.
venv/bin/python experiments/coverage_dlgn/analyze_topology.py \
  --config experiments/coverage_dlgn/configs/topology_semantic_balanced_cifar_paper.json

# Single CUDA run. Prefer a new preregistered config; do not rerun this example.
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/train.py --config PATH_TO_NEW_CONFIG.json

# Two-GPU queue. The queue must set cuda_required=true and every config device=cuda.
venv/bin/python experiments/coverage_dlgn/run_gpu_queue.py \
  --queue PATH_TO_NEW_QUEUE.json --gpus 0 1 \
  --data-path /tmp/torchlogix-datasets
```

If CUDA is visible to `nvidia-smi` but not to the venv, retry in the authorized
host execution context or use an Apptainer image based on
`cuda130_pytorch29.def`. Do not repair this by installing packages globally.

## Next concrete steps

1. **Preserve and synchronize the milestone.** Review these handoff files,
   commit them, and push the seven previously local commits plus the handoff
   when the user authorizes pushing. Independently back up the ignored `.pt`
   files needed for future tests.
2. **Repair analysis/reporting without changing U2.** Add U2 to the semantic
   strategy sets in `generate_dense_stack()`, test that topology-only U2 matches
   model construction for a small case, extend the evidence audit for
   LogicTreeNet-M U2 and third-round aggregates, and remove stale pending/
   uncommitted prose. Verify historical topology/checkpoint hashes do not
   change.
3. **Preregister the decisive experiment matrix.** Freeze seeds, validation
   criteria, schedules, cost fields, and early-stop rules before training. Keep
   V3/V4/U1/U2 immutable and avoid a full rank/parameterization Cartesian
   product.
4. **Test parameterization independence at rank 2.** First screen matched
   random versus U2 under WARP on dense CIFAR-10 M and LogicTreeNet-S. Use one
   short paired seed, then three seeds only if positive. Light requires its
   faithful optimizer/temperature recipe; the earlier raw-settings Light screen
   was poor. Exact Mind-the-Gap is raw rank-2 hard Gumbel; treat Gumbel-on-Light
   as a new method, not an exact reproduction.
5. **Implement rank-4 U2 as a new extension.** Generalize balanced multiscale
   matchings to four distinct predecessors, preserve rank-2 U2 bit-for-bit, add
   determinism/bounds/no-duplicate/fan-out/ancestry/cost tests, and document
   export limitations. Do not change the frozen rank-2 strategy.
6. **Run the minimal rank study.** Under one faithful Light/BitLogic training
   recipe compare rank-2 random/U2, rank-4 fixed random/U2, and rank-4
   learnable-16 at the published 2 x 16K and 2 x 64K CIFAR-10 coordinates.
   Report accuracy, LUT/routing parameters, memory, time, and physical cost.
7. **Build a convolutional Pareto ladder and synthesize it.** At fixed S
   architecture ratios, screen `k={16,20,24,28,32}` for random/U2. For selected
   checkpoints run Yosys/ABC and, if available, Vivado with identical tool,
   device, clock, constraints, and place-route seeds; report NAND2/cells, LUTs,
   Fmax, power/energy, routing, and equivalence. Width scaling is the primary
   gate-reduction route; unit tying is a separately credited comparator/stack.
8. **Freeze and test untouched transfer.** Only after the method/training
    combination is frozen, evaluate convolutional CIFAR-100 S, SVHN, or
    CIFAR-10.1 without topology retuning. A failed transfer should narrow the
    claim, not trigger another dataset-specific U2 revision.

The fallback if the unified confirmations fail is still valuable: write a
narrower dense-DLGN paper around frozen V3's five-seed CIFAR-10 gains,
degree-balance ablation, and 67-80% within-frontier gate reductions. Do not
force a universal dense/convolutional claim unsupported by the final seeds.
