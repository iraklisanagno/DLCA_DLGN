# CoverageDLGN AI handoff

State reconstructed on **2026-09-06** from the repository, frozen experiment
artifacts, git history, and the prior project discussion. This is an engineering
and research handoff, not a manuscript draft.

## Current completion state — September 6, 2026

All six authorized fourth-round steps are implemented and the full experimental
matrix is complete: **50/50 cells**, 35 new training runs and 15 reused logical
endpoints, with six additional archived raw references. Deduplicating endpoints
and references gives 50 unique outputs and 100 best/final checkpoints. The
primary queue and bounded assignments finished; do not restart completed work.

Authoritative final evidence:

- `summary/fourth_round_results.json` and `summary/fourth_round_runs.csv`:
  complete, no pending or skipped cells; all raw seeds, curves and resources.
- `summary/fourth_round_statistics_audit.json`: 1,799 independent scalar
  checks, 18 groups, 14 paired effects, 14 WARP/raw comparisons, three transfer
  contrasts; final JSON/CSV hashes recorded, zero new inference.
- `summary/fourth_round_checkpoint_metadata_audit.json`: pass, 100 checkpoints.
- `summary/fourth_round_factorial_wiring_audit.json`: all twelve trained
  factorial checkpoints pass the seven body/head/spatial checks per seed.
- `logs/fourth_round/supplemental_audit_tests.json`: 36 passing supplemental
  tests with source and log hashes. Original preflight remains 3,455 passed,
  3,038 skipped, one existing warning, 16 GPU smokes and ten synthetic loads.
- `FOURTH_ROUND_RESULTS.md`: complete scientific report, including limitations
  and the preserved transfer serialization failure/artifact-only recovery.
- `summary/fourth_round_completion_audit.json`: six-requirement acceptance
  record and hashes of the reconciled documents. The closing focused suite
  passed 133 tests; read-only protocol checks revalidated all smoke artifacts
  and 14 effective WARP/raw recipe pairs, with no new accuracy experiments.

Findings: neither mechanism coordinate establishes the added novelty-selection
benefit. The factorial shows a conditional body effect with the reference-U2
head (+2.140 pp, paired 95% CI [0.674, 3.606], n=3 [OUR, V]), not a
head-independent or additive effect. Dense adapted WARP retains a U2 gain:
+4.700 pp best hard, CI [3.919, 5.481], 3/3 wins [ADAPTED, V]. Convolutional
WARP is inconclusive: +0.087 pp, CI [−1.243, 1.417], one win/tie/loss across
three seeds; final-hard effect −1.213 pp, CI [−11.597, 9.171]. Both WARP
coordinates trail matched raw and use more peak memory despite fewer training
parameters. Frozen CIFAR-10.1 transfer is positive for dense M (+3.733 pp,
CI [2.801, 4.666], n=3), mixed for S/M (−0.050/+1.950 pp, each n=1)
[OUR, T]. No repeat inference or post-hoc recipe changes were made.

Additional full-schedule S seeds, hardware and rank-four extensions remain
deferred. See the current next-step checklist near the end of this document;
the chronological implementation notes below are not an active queue.

## Historical September 6 implementation and scheduling notes

All intermediate running/pending counts in this section are historical and
superseded by the completed evidence above. Before
the current working edits, HEAD was `5f913a3`, clean and equal to the locally
recorded `origin/coverageDLGN`; the handoff and reconciliation were committed.
The user has now authorized mechanism controls, body/classifier attribution,
rank-2 WARP compatibility, and a frozen transfer evaluation. Hardware work,
rank-four extensions, and additional full LogicTreeNet-S seeds are deferred.
All reported bugs must be repaired and tests/GPU smokes pass before full runs.
The user subsequently authorized execution of all six steps. Fourth-round
configs and `protocols/fourth_round.json` now declare 50 cells (35 new, including
12 conditional WARP follow-ups), plus six archived raw WARP references. The
ten transfer checkpoint/config selections were frozen at 06:13:59 UTC before
CIFAR-10.1 data access. Host preflight passed: 3,455 tests passed, 3,038 skipped,
one existing warning; all 16 training-path GPU smokes and ten synthetic
checkpoint-load checks passed. `summary/fourth_round_preregistration.json`
freezes the verified implementation, configs, smoke records and reused artifacts.
The serial accuracy queue started on GPU 1 at approximately 06:25 UTC with
`fourth_mechanism_dense_m_balanced_random_seed0`; inspect `logs/fourth_round/`
and process state before any restart. The dense-M 20K mechanism coordinate is
now complete for seeds 0/1/2. U2's paired gains are +4.313 pp over random (95%
CI [2.475, 6.152]), +0.487 pp over balanced-random (CI [−0.795, 1.768]), and
−0.087 pp against nominal (CI [−1.759, 1.585]); all [OUR, V], exploratory.
Thus the dense study does not establish an incremental novelty benefit. The
verified phase evidence is `summary/fourth_round_dense_mechanism.json`.
The serial queue has advanced to convolutional controls. For its completed and
pending counts, run `venv/bin/python experiments/coverage_dlgn/summarize_fourth_round.py`
from the TorchLogix root; this read-only command verifies frozen hashes before
reporting. Determine the active run from the live process and per-run logs,
not from a potentially stale prose counter. The dense phase does not settle
the pending convolutional, WARP or transfer questions.
The convolutional mechanism coordinate has now also completed all three seeds.
U2 gains +0.213 pp over balanced-random (paired 95% CI [−0.836, 1.263]) and
+0.593 pp over nominal (CI [−1.121, 2.308]), both 2/3 wins [OUR, V]. Its matched
random contrast is +1.707 pp, CI [−2.600, 6.013], 3/3 wins. Thus neither dense
nor convolutional 20K evidence establishes the extra novelty-selection benefit;
do not claim equivalence or retune the frozen method. All 24 mechanism cells
are complete, with source `summary/fourth_round_conv_mechanism.json` alongside
the dense summary. Raw-seed effects and intervals were independently checked.
GPU 1 has moved to factorial body seed 1; GPU 0 continues the bounded seed-2
body/head assignment. Full factorial and promoted WARP confirmation remain pending.
The one-time frozen transfer evaluation is complete: ten checkpoint queries,
no adaptation or repeated inference. Dense-M U2 gains +3.733 pp over random,
n=3, paired 95% CI [2.801, 4.666], 3/3 wins [OUR, T]. S/M gains are −0.050/
+1.950 pp, each descriptive n=1. Retain the S result: convolutional robustness
is not established universally. Predictions ran from 07:59:11 to 08:02:06 UTC
on GPU 1 alongside `fourth_mechanism_conv_s_nominal_seed0`; account for that
overlap when comparing training wall times. After all ten records were saved,
aggregate serialization failed on a NumPy `int64`. The artifact-only recovery
completed at 08:05:55 UTC after seven regression tests and saved-data/hash checks,
with zero new inference. The frozen evaluator source remains unchanged to
preserve the active execution freeze. Never rerun it; use read-only
`venv/bin/python experiments/coverage_dlgn/recover_transfer_report.py` to verify
the recovered summary and hash-linked recovery/completion records.
Hardware and extra full-S seeds stay deferred.

Scheduling update at approximately 08:16 UTC: GPU 0 became free and passed a
fresh CUDA allocation. A bounded second process completed
`fourth_factorial_conv_s_body_seed0` followed by
`fourth_factorial_conv_s_head_seed0` using the existing frozen runner helpers.
All seven saved-wiring checks now pass for the complete seed-0 factorial;
seeds 1/2 are still pending. GPU 1 continues the original serial queue and
skips these cells only after normal completion/hash checks. The assignment is recorded in
`logs/fourth_round/parallel_assignment_gpu0_seed0.json`. Monitor both live
processes; never duplicate a live target or restart an unclassified output.
No training setting or frozen source changed. Further GPU-0 assignments require
checking the primary queue position first to avoid overlapping work.

At approximately 08:52 UTC, after the bounded assignment exited successfully
and GPU 0 was verified free, a new bounded process started the two dense-M WARP
pilot cells (random then U2, seed 0). GPU 1 was on balanced-random convolutional
seed 2. `logs/fourth_round/parallel_assignment_gpu0_warp_pilot.json` records this
scheduling-only assignment. After both pilots pass ordinary artifact checks,
it writes the exact preregistered `warp_promotion.json` decision; it does not
launch conditional follow-ups. The primary runner later reuses that identical
decision and completed cells. Check the live process and promotion receipt;
do not infer a positive outcome from a running pilot.

Both WARP pilots have now completed and the frozen gate passed: best hardened
validation random 53.660%, U2 57.680%, +4.020 pp [ADAPTED, V], paired seed 0.
This is a positive pilot, not three-seed confirmation. All twelve preregistered
follow-ups are now required: six dense-M 108K runs and six convolutional-S 20K
runs. WARP trails matched raw by 1.740/1.180 pp for random/U2 at 20K, so do not
claim that the adapted WARP recipe improves raw accuracy. Source:
`summary/fourth_round_warp_pilot.json` and `logs/fourth_round/warp_promotion.json`.

The GPU-0 pilot assignment exited successfully. After fresh GPU checks, its
next bounded assignment is factorial body seed 2 followed by head seed 2;
GPU 1 was on nominal convolutional seed 2 and then handles factorial seed 1.
See `logs/fourth_round/parallel_assignment_gpu0_seed2.json`. Monitor both live
queues and check their position before assigning more work; never duplicate
active outputs.

The bounded GPU-0 factorial assignment has now completed. Trained seeds 0 and
2 each pass all seven body/head/spatial wiring checks; only head-only seed 1
remains for full factorial certification. GPU 1 is training that cell. GPU 0
was verified free and passed fresh CUDA allocation, then started the promoted
dense-M 108K WARP seed-2 pair (random then U2) from scratch after revalidating
the positive gate. Scheduling receipt:
`logs/fourth_round/parallel_assignment_gpu0_warp_confirm_seed2.json`. The main
queue handles WARP seeds 0/1 before reaching the side-assigned seed 2; monitor
both queues before any new assignment. No frozen source or configuration changed.
All fourteen declared WARP/raw recipe pairs were independently compared after
resolving parser defaults: only parameterization, paths, and the previously
tested inactive fixed-routing `connections_gumbel` flag differ.

The last head-only seed-1 run has now completed. All twelve factorial checkpoints
pass the full saved-wiring audit, and all five paired effects were independently
recomputed. With the U2 head fixed, U2 body wiring gains +2.140 pp (paired 95%
CI [0.674, 3.606]), 3/3 wins [OUR, V]. This is a positive conditional body
contribution, not a head-agnostic or additive claim: the head uses reference
U2-body ancestry. The random-head body effect (+1.207 pp), both head effects
(−0.433/+0.500 pp), and interaction (+0.933 pp) all have intervals crossing zero.
See `summary/fourth_round_factorial.json` and its hash-linked
`summary/fourth_round_factorial_wiring_audit.json` for all raw seeds and checks.
Both GPUs now run promoted dense-M WARP replications: the main GPU-1 queue starts
with seed 0, and the bounded GPU-0 assignment handles seed 2. The twelve WARP
follow-ups and consolidated final reporting are the remaining required work.

Supplemental final-factorial verification is complete in `audit_saved_wiring.py`.
Its eight tests, four saved seed-0 smoke checkpoints, and all twelve trained
factorial checkpoints pass. The trained-checkpoint certificate is
`summary/fourth_round_factorial_wiring_audit.json`; use the auditor's default
read-only mode for subsequent verification, not a new experiment. The
artifact-only auditor and tests are outside the frozen training implementation;
the audit hashes both sources and refuses incomplete certification.

The supplemental `audit_checkpoint_metadata.py` now makes the earlier manual
checkpoint-metadata check reproducible. Its 21 regression tests pass, and its
latest default read-only run verifies all 98 best/final checkpoints from 49 completed
unique runs (including raw references). It checks first-maximum best selection,
final step, exact saved configuration and all four validation accuracy/loss
metrics against CSV, after frozen-artifact verification. It performs no model
inference or dataset access. Once every required WARP cell is complete, use
`--write` to save `summary/fourth_round_checkpoint_metadata_audit.json`; it
refuses incomplete certification. This auditor and its experiment-local test
are outside the frozen implementation and do not alter the active queue.

The first promoted 108K WARP arm, dense-M random seed 2, has completed and
passed both artifact and metadata audits. Best/final hard validation is
52.920%/50.800% [ADAPTED, V]; best selection is at 24K. GPU 0 has started its
already-assigned U2 seed-2 partner, while GPU 1 continues random seed 0. There
are now 39/50 verified matrix cells and eleven pending WARP cells. No paired
108K conclusion is available yet; check live state before acting on this count.

Random seed 0 has also completed 108K and passed artifact/metadata checks:
best hard validation 53.660% at 8K, final hard 50.600% [ADAPTED, V]. GPU 1
has started its assigned U2 seed-0 partner; GPU 0 continues U2 seed 2. The
latest audit verifies 40/50 matrix cells and 80 best/final checkpoints, with
ten WARP cells pending. Both random arms peaked before 108K; retain best and
final values, and wait for complete pairs before any compatibility conclusion.

The seed-2 108K pair is now complete: U2 reaches 57.760% best hard validation
at 10K and 55.420% final hard, versus random 52.920%/50.800%. Gains are
+4.840 pp best and +4.620 pp final [ADAPTED, V], descriptive n=1; the full
three-seed compatibility conclusion remains pending. Both artifact and
metadata audits pass, now covering 41/50 matrix cells and 82 checkpoints.
The bounded GPU-0 dense assignment exited successfully. Fresh host inspection
found GPU 0 free, confirmed GPU 1 still on dense U2 seed 0, and a CUDA tensor
allocation passed. GPU 0 then started the four preregistered convolutional
WARP cells for random/U2 seeds 0/1 at approximately 11:10 UTC. Receipt:
`logs/fourth_round/parallel_assignment_gpu0_warp_conv_seeds01.json`. This
assignment exits after U2 seed 1; convolutional seed 2 is not side-assigned.
The primary queue finishes dense U2 seed 0 and both dense seed-1 arms before
reaching these cells. Monitor both queues to avoid overlap; no frozen source
or configuration changed. Nine WARP cells remain pending at this audit.

Dense U2 seed 0 subsequently completed: 58.000% best hard validation at 22K,
55.400% final hard, gaining +4.340/+4.800 pp over its paired random arm
[ADAPTED, V]. The raw-CSV gains were independently checked. Dense seeds 0/2
are complete and positive, but seed 1 remains required; GPU 1 is now training
dense random seed 1. GPU 0 completed convolutional WARP random seed 0 at 20K
and has started its U2 partner. That random arm reaches only 45.020% best hard
at 8K and 36.060% final hard [ADAPTED, V], versus 55.000% best for its matched
raw arm; retain the decline and negative raw comparison without retuning.
WARP peak allocation is 2.176 GiB, above raw's 1.831 GiB despite fewer trainable
parameters. Both new runs pass artifact/metadata checks. The latest audit covers
43/50 cells and 86 checkpoints; seven WARP cells remain pending, and no
convolutional WARP pair is complete yet.

The first convolutional WARP pair is now complete and verified. At seed 0,
both random and U2 reach 45.020% best hard validation at 8K, but final hard
is 36.060%/30.320%: U2 − random is 0.000 pp best and −5.740 pp final
[ADAPTED, V], descriptive n=1. U2's WARP best is 13.480 pp below matched raw,
and its final relaxed accuracy is 36.920% (6.600 pp above final hard).
Independent CSV reconstruction and artifact/metadata checks pass. Retain this
negative/inconclusive compatibility result and finish the predeclared seeds;
do not tune a rescue recipe. GPU 0 has advanced to convolutional random seed 1;
GPU 1 continues dense random seed 1. The latest audit verifies 44/50 cells and
88 checkpoints, with six WARP cells pending. No full three-seed WARP coordinate
is complete yet.

Both seed-1 random arms have since completed and passed artifact/metadata
checks. Convolutional WARP random seed 1 reaches 43.300% best hard at 4K and
37.600% final; dense WARP random seed 1 reaches 53.160% best hard at 8K and
50.920% final [ADAPTED, V]. GPU 0 now trains convolutional U2 seed 1, the last
run in its bounded assignment; GPU 1 trains dense U2 seed 1, the last unfinished
dense cell. The latest audit verifies 46/50 cells and 92 checkpoints. Four
cells remain: both active U2 partners and convolutional random/U2 seed 2.
Convolutional seed 2 is not currently side-assigned. Before assigning any of it
to GPU 0, wait for that bounded assignment to exit, recheck GPU ownership and
primary progress, and allow enough lead to finish before the primary reaches
the same output. Do not let an incomplete side-owned directory collide with
the serial runner's fail-closed completion checks.

Convolutional U2 seed 1 has now completed and passed both audits: best hard
42.900% at 6K, final hard 37.200% [ADAPTED, V], both −0.400 pp versus its
random partner. Independent CSV reconstruction confirms the effects. U2 WARP
best is 16.640 pp below matched raw. Convolutional seeds 0/1 therefore have
best effects 0.000/−0.400 pp and final effects −5.740/−0.400 pp; retain these
negative/inconclusive results and wait for the predeclared final seed.
The latest audit verifies 47/50 cells and 94 checkpoints. The bounded GPU-0
assignment exited 0; fresh host checks found GPU 0 free, a CUDA allocation
passed, and GPU 1 still trained dense U2 seed 1 (last saved step 42K/108K).
GPU 0 started ONLY random convolutional seed 2 at approximately 12:21 UTC.
Receipt: `logs/fourth_round/parallel_assignment_gpu0_warp_conv_random_seed2.json`.
It exits after this one run. Final convolutional U2 seed 2 remains with the
primary queue: do not side-launch it while the primary can reach its output.
Three cells remain at this snapshot: active dense U2 seed 1, active random
convolutional seed 2, and pending U2 convolutional seed 2. Continue monitoring
the existing sessions and check completion stamps before any action.

Random convolutional seed 2 completed 20K and passed artifact/metadata checks:
44.640% best hard at 10K and 38.840% final hard [ADAPTED, V]. Its bounded
GPU-0 process exited 0. Dense U2 seed 1 subsequently completed 108K: 58.080%
best hard at 12K and 55.580% final hard. This closes all six dense WARP cells.
`summary/fourth_round_warp_dense.json` freezes the complete coordinate, with
all raw seeds, curves, references and independently checked statistics.
Random/U2 best-hard means ± sample SD are 53.247 ± 0.378% and
57.947 ± 0.167%; paired gain +4.700 pp, 95% CI [3.919, 5.481], 3/3 wins
[ADAPTED, V]. Final-hard gain is +4.693 pp, CI [4.459, 4.928], also 3/3.
This demonstrates a dense topology gain within adapted WARP, not superiority
over raw: matched raw best accuracy remains higher by 1.913/1.600 pp for
random/U2, and WARP peak memory is 1.235 versus raw 1.123 GiB. Parameter
savings are WARP's, not U2's; both methods decline after their early best steps.
The primary queue skipped all verified side-completed cells and started the
LAST run, `fourth_warp_conv_conv_s_u2_seed2`, on GPU 1. The latest audit covers
49/50 cells and 98 checkpoints; only this final cell remains before full
reporting. Do not relaunch either completed GPU-0 assignment or the active
final run. Hardware and extra full-S seeds stay deferred.

## Historical pre-implementation snapshot

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

The scientific story concerns structured pairing and semantic propagation, not
a demonstrated correction of dense degree imbalance. Dense native random already
balances encoded-input fan-out. Native grouped convolution cycles adjacent
channel groups with random spatial sampling; U2 changes both channel balance and
pairing structure there. Neither comparison isolates balance alone. The saved
wiring audit is `summary/fourth_round_wiring_audit.json`; it used no dataset.
Maximum ancestry coverage by itself is not sufficient and can be harmful.
Dense V3 is the strongest dense
specialization and belongs in the paper as an ablation/specialized upper point;
U2 is the only method currently defensible as the unified dense/convolutional
method.

The work has a publishable core, but the broad DATE claim is not yet finished.
The authorized mechanism, channel-wiring, modern-training and frozen-transfer
studies are complete with mixed outcomes; see the current completion summary.
Full convolutional seed replication is the next experimental decision, not an
automatic launch. Rank-4 and real hardware synthesis remain deferred.

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
  The component ablation's balanced-butterfly arm accounts for about 93% of its
  gain, while swaps add only about 0.04 pp at CIFAR-10 M. That arm also imposes
  structure, and native dense random already balances fan-out: it does not
  independently identify the contribution of balance.

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
- September 6 repair verification: **3,453 passed, 3,038 skipped**, one
  pre-existing warning, in 244.71 seconds using `OMP_NUM_THREADS=2
  MKL_NUM_THREADS=2 venv/bin/python -m pytest tests -q`. The focused topology,
  handoff-repair and fourth-round tests passed 95/95 in 6.73 seconds; the
  read-only evidence audit passed 41/41 checks. This sandbox suite does not
  substitute for the required host GPU smokes before accuracy runs.

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

1. **Repaired: topology-only U2 semantic ancestry.** The helper now includes
   U2 in all three semantic-strategy sets. The old error could change deeper
   generated indices as well as metrics. The trained model path was unaffected.
   Tests compare layer indices and ancestry against the correct semantic path,
   and pre-edit digests protect frozen V3/V4/U1/U2 behavior.
2. **Presentation reconciliation.** Commit `5f913a3` removed the uncommitted
   marker and corrected M U2 status. The September 6 repair also updates the
   README's selected Fashion numbers and distinguishes historical sections
   from current status. Frozen per-run evidence remains authoritative.
3. **Extended evidence audit.** The audit checks the 190 third-round frozen
   artifacts, per-run summaries, aggregates, paired effects, dense reference
   comparisons, and M U2 hashes/results/query records. It does not claim
   whole-project coverage. Default execution writes no file; `--output`
   accepts only a new report path, preserving the historical audit JSON.
4. **No exact training resume.** Checkpoints do not preserve optimizer,
   dataloader, and RNG state sufficiently to prove exact continuation. Partial
   training must not be called an exact resume; archive it under
   `results/failed/` and restart unchanged unless resume equivalence is first
   implemented and tested.
5. **Strengthened queue completion.** The queue now requires nonempty best
   and final checkpoints in addition to the four original metadata files.
   Fixed-routing models also require topology JSON; learned-only architectures
   intentionally do not emit it. Malformed configuration/summary metadata
   fails closed. Phase validation and frozen hashes remain necessary for
   scientific acceptance; file presence alone is not evidence of integrity.
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
13. **Transfer aggregate serialization recovered without re-query.** The frozen
    evaluator saved all ten prediction records before its NumPy integer win
    count failed JSON serialization. A separate, tested artifact-only recovery
    writes native Python counts and verifies all saved evidence. Its source and
    tests are outside the immutable training implementation; the recovery receipt
    hashes both. Keep the original failure and frozen source as historical
    evidence. This is report recovery, not a second transfer evaluation.

Additional September 6 safeguards: future S freezes include configuration,
checkpoint, environment and summary hashes. Pending evaluation rejects missing
or changed required hashes and records an exclusive started marker before test
access. Completed evaluation is read-only and preserves its historical log;
legacy unhashed freezes are not retroactively represented as hashed evidence.
The fourth-round transfer validator rejects missing/reordered/truncated cells
and missing hashes. These safeguards have synthetic regression tests. Separate
execution evidence at the top of this handoff records the completed host
preflight and one-time transfer; synthetic tests alone would not prove either.

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
- [`../FOURTH_ROUND_RESULTS.md`](../FOURTH_ROUND_RESULTS.md): completed
  fourth-round evidence report, acceptance records and claim boundaries.
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

# Read-only evidence audit, including M U2 and third round.
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

The fourth round, including all twelve promoted WARP runs and final reporting,
is complete. The original planning notes below are historical, not instructions
to repeat completed work. Subsequent priorities, in order:

1. **Decide and preregister additional full-schedule LogicTreeNet-S paired
   seeds.** This is the highest-priority next experiment, not an authorized
   queue addition. Match frozen random/U2 recipes and add seeds 1/2 to the
   existing full-schedule seed 0 if approved. The completed 20K studies do not
   replace full-schedule replication. Freeze selections before any new test
   access; never re-query the existing locked checkpoints.
2. **Write the manuscript around demonstrated effects.** Strong dense results,
   fixed inference cost, dense WARP compatibility and frozen dense transfer
   are supported. Added novelty-selection value, head-agnostic channel gains
   and universal convolutional robustness are not established. Preserve raw
   seeds, exploratory uncertainty and negative WARP/raw comparisons.
3. **Reassess the breadth of the convolutional claim after full-S replication.**
   More full-M seeds would be a separate cost/benefit decision. A narrower
   dense-centered paper remains preferable to an unsupported universal claim.
4. **Preserve artifacts and defer scope expansion.** Keep checkpoints and all
   failure/recovery records; arrange an independent backup before migration.
   Hardware, rank-four extensions and broader sweeps remain deferred. Use
   read-only audits for verification, not repeated completed experiments.

### Historical planning notes (superseded by the checklist above)

1. **Preserve selected artifacts.** The original milestone and handoff are
   already committed and match the local remote-tracking ref. Checkpoints
   remain ignored and need an independent backup decision.
2. **Verify the September 6 repairs before experiments.** Run regression tests
   and the extended audit. Do not rewrite historical checkpoints or manifests.
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
5. **Deferred: rank-4 U2 as a new extension.** Generalize balanced multiscale
   matchings to four distinct predecessors, preserve rank-2 U2 bit-for-bit, add
   determinism/bounds/no-duplicate/fan-out/ancestry/cost tests, and document
   export limitations. Do not change the frozen rank-2 strategy.
6. **Deferred: minimal rank study.** Under one faithful Light/BitLogic training
   recipe compare rank-2 random/U2, rank-4 fixed random/U2, and rank-4
   learnable-16 at the published 2 x 16K and 2 x 64K CIFAR-10 coordinates.
   Report accuracy, LUT/routing parameters, memory, time, and physical cost.
7. **Deferred: convolutional Pareto ladder and synthesis.** At fixed S
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
structured-routing ablations, and 67-80% within-frontier gate reductions. Do not
force a universal dense/convolutional claim unsupported by the final seeds.
