# Fourth-round evidence report

Status: complete, September 6, 2026. All 50 preregistered matrix cells validate,
including all twelve promoted WARP follow-ups. Final evidence is in
[JSON](summary/fourth_round_results.json) and [CSV](summary/fourth_round_runs.csv).
Completion means the experiments answered the questions, not that every
hypothesis was supported. No additional full-S seeds or hardware work was run.

## Completion requirements

| Step | Required evidence | Current disposition |
|---|---|---|
| GPU verification and freeze | Full suite, all training-path smokes, synthetic checkpoint loads, immutable hashes | Passed; `summary/fourth_round_preregistration.json` |
| U2 construction mechanism | Three paired seeds at dense M and convolutional S; U2 versus balanced-random and nominal-multiscale | Complete; neither coordinate establishes an incremental novelty-selection benefit |
| Independent channel wiring | Three-seed body/head factorial with same-head wiring fixed across bodies | Complete; positive body effect with fixed U2 head, other contrasts inconclusive |
| Modern training compatibility | Matched WARP pilot; frozen promotion decision; promoted replications or documented failed gate | Complete; positive dense topology gain within adapted WARP, convolutional gain inconclusive; both trail matched raw |
| Frozen transfer | Ten preselected checkpoints, pinned CIFAR-10.1 v6, one recorded evaluation attempt, saved predictions | Complete; aggregate serialization recovered from saved predictions, zero repeat inference |
| Evidence summary | Complete JSON/CSV, raw seeds, paired uncertainty, resource costs, and retained negative outcomes | Complete; statistics audit passes 1,799 scalar checks and checkpoint audit passes 100/100 |

The protocol is [FOURTH_ROUND_PROTOCOL.md](FOURTH_ROUND_PROTOCOL.md). The
machine-readable matrix has 50 cells: 15 reused, 23 new initial runs and 12
conditional WARP follow-ups. Six archived 108K raw references provide matched
WARP-versus-raw context. Reused endpoints can appear in two experimental
questions; they are not additional independent training runs.

Final checkpoint-metadata acceptance is reproducible with
`venv/bin/python experiments/coverage_dlgn/audit_checkpoint_metadata.py` from
the TorchLogix root. This supplemental, default-read-only auditor has 21 passing
regression tests and verifies 100 best/final checkpoints from 50 unique
completed runs including raw references. It checks first-maximum selection,
final step, saved configuration and all four validation accuracy/loss metrics
against CSV after verifying frozen hashes, without model inference or data
access. The final certificate is
`summary/fourth_round_checkpoint_metadata_audit.json`; subsequent checks are
read-only. `summary/fourth_round_statistics_audit.json` records independent
CSV/resource reconstruction, 18 groups, 14 paired contrasts, 14 WARP/raw
comparisons and three saved-transfer contrasts, with zero additional inference.
The 36 supplemental auditor/recovery tests also pass; their source/log hashes
are recorded in `logs/fourth_round/supplemental_audit_tests.json`.

Final requirement-by-requirement acceptance is recorded in
`summary/fourth_round_completion_audit.json`. The closing focused rerun passed
133 tests (repairs, controls, topology and supplemental audits). Read-only
verification reproduced the final report and both saved checkpoint/wiring
certificates, checked all 16 smoke artifact sets and ten synthetic-load records,
and confirmed all 14 WARP/raw effective recipe pairs. A further independent
numerical recheck passed 1,256 scalar comparisons plus all curves and CSV rows;
this complements, rather than replaces, the earlier 1,799-check statistics audit.
Host inspection found no experiment training/queue processes. An unrelated
VLLM workload remained untouched. No new training or held-out inference was
performed during closeout; remaining full-S/hardware decisions are outside
this completed round.

## Does U2's extra construction logic help?

Baseline interpretation, verified without dataset access against saved wiring
and source (`summary/fourth_round_wiring_audit.json`): native dense random is
already fan-out balanced, not iid unconstrained edge sampling. Across the three
dense-M seeds its first-layer fan-out is 27–28; deeper layers have exactly 2.
U2 and both new structured controls have first-layer fan-out 27–29 and exactly
2 deeper down. Thus this study does not establish that dense gains come from
fixing degree imbalance. The shared semantic first layer is not covered by
the deeper matching construction's fan-out-spread-at-most-one guarantee.

Native convolutional random instead cycles adjacent two-channel groups and
randomizes spatial samples. The four smoke checkpoints have channel-pair
endpoint fan-out ranges, by block: random 4–8, 4–10, 4–10, 2–6; each structured
arm 6–8, 8–8, 8–8, 4–4. These count channel-pair endpoints, not repeated spatial
leaf slots. The structured arms change both channel balance and pair structure;
their contrast against native random does not isolate either factor alone.
These structural diagnostics are not additional accuracy trials.

The primary contrasts are U2 minus balanced-random and U2 minus nominal-
multiscale, at fixed 20K training updates, with paired seeds 0/1/2. Both new
controls preserve U2's semantic first layer. Balanced-random changes deeper
matching structure; nominal-multiscale removes novelty selection while retaining
the candidate stages and degree priority. The latter directly isolates the
additional stage-selection rule more closely than the former.

Report the three seed values, paired mean gain, sample SD, Student-t 95% interval,
wins, and paired offline construction-time difference for each coordinate.
Do not describe a confidence interval crossing zero as proof of equivalence.
If a simpler control performs as well or better, retain that outcome and narrow
the justification for U2 rather than revise U2 after seeing these results.

Complete dense-M coordinate: 20K updates, best hardened validation (V), seeds
0/1/2. Verified source: `summary/fourth_round_dense_mechanism.json`.

| Wiring | Raw seed accuracies (%) | Mean ± sample SD (%) | Mean layer construction (s) | Provenance |
|---|---|---:|---:|---|
| Explicit random, archived matched control | 55.400, 54.700, 54.360 | 54.820 ± 0.530 | 5.461 | REPRODUCED |
| Balanced-random | 58.800, 58.420, 58.720 | 58.647 ± 0.200 | 11.582 | OUR |
| Nominal-multiscale | 59.220, 58.800, 59.640 | 59.220 ± 0.420 | 13.648 | OUR |
| U2 | 58.860, 59.480, 59.060 | 59.133 ± 0.316 | 14.110 | OUR |

All contrasts below are [OUR, V], n=3, with exploratory paired Student-t 95%
intervals. They are not multiplicity-adjusted.

| Contrast | Paired gain (pp) | Paired 95% CI (pp) | Wins | Extra layer construction (s) |
|---|---:|---|---:|---:|
| U2 − random | +4.313 | [2.475, 6.152] | 3/3 | +8.649 |
| U2 − balanced-random | +0.487 | [−0.795, 1.768] | 3/3 | +2.528 |
| U2 − nominal | −0.087 | [−1.759, 1.585] | 1/3 | +0.462 |

At this dense coordinate and budget, U2 improves random, but the experiment
does **not establish an incremental benefit from novelty selection** over the
simpler nominal multiscale rule. The advantage over balanced-random is also
inconclusive. An interval crossing zero is not proof of equivalence. All arms
have the same declared cost and 1.123 GiB peak allocated GPU memory; mean run
wall times are 7.685/7.749/7.751/7.754 minutes in the table's method order.
These are 20K mechanism results, not conclusions about longer training budgets.

Complete convolutional-S coordinate: 20K updates, best hardened validation (V),
seeds 0/1/2. Verified source: `summary/fourth_round_conv_mechanism.json`.

| Wiring | Raw seed accuracies (%) | Mean ± sample SD (%) | Mean layer construction (s) | Provenance |
|---|---|---:|---:|---|
| Explicit random, archived matched control | 55.000, 57.960, 58.460 | 57.140 ± 1.870 | 0.201 | REPRODUCED |
| Balanced-random | 58.680, 58.880, 58.340 | 58.633 ± 0.273 | 0.968 | OUR |
| Nominal-multiscale | 58.600, 58.940, 57.220 | 58.253 ± 0.911 | 1.120 | OUR |
| U2 | 58.500, 59.540, 58.500 | 58.847 ± 0.600 | 1.380 | OUR |

All following contrasts are [OUR, V], n=3, with exploratory paired Student-t
95% intervals, without multiplicity adjustment.

| Contrast | Paired gain (pp) | Paired 95% CI (pp) | Wins | Extra layer construction (s) |
|---|---:|---|---:|---:|
| U2 − random | +1.707 | [−2.600, 6.013] | 3/3 | +1.178 |
| U2 − balanced-random | +0.213 | [−0.836, 1.263] | 2/3 | +0.411 |
| U2 − nominal | +0.593 | [−1.121, 2.308] | 2/3 | +0.259 |

All arms have 83,552 LUT functions and 1,336,832 trainable raw parameters, no
learned routing, and approximately 1.831 GiB peak allocated GPU memory. Mean
wall times in table order are 17.001/17.105/17.200/17.018 minutes; nominal
seed 0 overlapped the transfer evaluation, as disclosed below. These timings
are shared-host observations, not isolated performance benchmarks.

Across the two completed coordinates, U2's extra novelty-selection rule has
**not demonstrated an incremental accuracy benefit** over nominal multiscale.
Its advantages over balanced-random also remain inconclusive. The convolutional
point estimates favor U2, but even the matched random contrast has a wide
interval crossing zero at this 20K budget. Do not claim equivalence, universal
superiority, or use the small construction-time increment as evidence of an
accuracy benefit. Retain frozen U2 and all controls; these outcomes narrow the
paper's mechanism claim without authorizing post-hoc method changes. Both
coordinates' paired effects/SD/intervals/wins were independently reconstructed
from the saved raw seeds.

Structural check on the seed-0 GPU smoke checkpoints: nominal and U2 have
identical semantic first-layer dense indices but differ in all three deeper
dense-M layers. For convolutional S, their first two convolutional blocks match;
the last two blocks and final classifier layer differ, while the first two
classifier layers match. The spatial samples remain matched by the preflight
invariants. Thus the nominal ablation is not a duplicate wiring configuration
at either tested coordinate. This check is not an accuracy result.

## Does channel wiring contribute independently?

The four cells are random/random, U2/random, random/U2, and U2/U2 for body/head.
The opt-in reference-ancestry construction keeps U2 classifier indices identical
across the two bodies; otherwise changing the body would also change that head.
Spatial samples, initial gate parameters and declared model costs are checked.

Supplemental saved-checkpoint verification uses `audit_saved_wiring.py`, separate
from the immutable training implementation. Eight regression tests pass; the
four seed-0 smoke checkpoints pass all seven body/head/spatial independence
checks (`summary/fourth_round_factorial_smoke_wiring_audit.json`). That smoke
evidence was followed by the full trained-checkpoint audit below, which certifies
every seed and refuses incomplete certification. Subsequent checks can use the
auditor's default read-only mode. It loads fixed indices on CPU, without model
inference or dataset access.

All twelve trained checkpoints now pass all seven checks for every seed.
`summary/fourth_round_factorial_wiring_audit.json` certifies matching body/head
indices and spatial samples, with genuinely distinct treatment levels. The
statistical snapshot `summary/fourth_round_factorial.json` hashes that audit.

Report body effects separately with each head, head effects separately with
each body, and the per-seed interaction:

`U2/U2 - U2/random - random/U2 + random/random`.

These are 20K attribution pilots, not substitutes for multi-seed full-schedule
LogicTreeNet-S/M confirmation. The explicit-random endpoints differ from an
older random control; do not substitute the historical +2.173 pp pilot contrast
for the +1.707 pp matched-controlled contrast (both [OUR, V]).

Complete results, best hardened validation at 20K, seeds 0/1/2. Random/random
is [REPRODUCED, V]; the other cells and all effects are [OUR, V].

| Body / classifier | Raw seed accuracies (%) | Mean ± sample SD (%) | Layer construction (s) | Wall (min) |
|---|---|---:|---:|---:|
| Random / random | 55.000, 57.960, 58.460 | 57.140 ± 1.870 | 0.201 | 17.001 |
| U2 / random | 58.240, 58.180, 58.620 | 58.347 ± 0.239 | 0.217 | 17.053 |
| Random / U2 | 56.960, 56.820, 56.340 | 56.707 ± 0.325 | 1.373 | 17.063 |
| U2 / U2 | 58.500, 59.540, 58.500 | 58.847 ± 0.600 | 1.380 | 17.018 |

All cells retain 83,552 LUT functions, 1,336,832 trainable raw parameters and
approximately 1.831 GiB peak GPU allocation, with no learned routing. The
layer construction timers exclude off-layer reference-ancestry work; their
differences are not complete end-to-end construction overhead measurements.

| Controlled contrast | Per-seed effects (pp) | Mean effect (pp) | Paired 95% CI (pp) | Positive seeds |
|---|---|---:|---|---:|
| Body: U2 − random, random head fixed | +3.240, +0.220, +0.160 | +1.207 | [−3.169, 5.582] | 3/3 |
| Body: U2 − random, U2 head fixed | +1.540, +2.720, +2.160 | +2.140 | [0.674, 3.606] | 3/3 |
| Head: U2 − random, random body fixed | +1.960, −1.140, −2.120 | −0.433 | [−5.725, 4.858] | 1/3 |
| Head: U2 − random, U2 body fixed | +0.260, +1.360, −0.120 | +0.500 | [−1.410, 2.410] | 2/3 |
| Body–head interaction | −1.700, +2.500, +2.000 | +0.933 | [−4.766, 6.633] | 2/3 |

Paired sample SDs in table order are 1.761, 0.590, 2.130, 0.769 and 2.294 pp.
All intervals are exploratory Student-t intervals, without multiplicity
adjustment. Each effect, SD, confidence bound and positive-seed count was
independently reconstructed from the raw seed values.

The positive fixed-U2-head contrast shows that convolutional channel wiring
contributes without changing classifier indices. Its scope is conditional:
that U2 head was constructed from reference U2-body ancestry. The result is
not proof of a head-agnostic body advantage, statistical additivity, or the
same effect under WARP or a full training schedule. The random-head body
contrast, both head contrasts and interaction remain inconclusive; retain all
of them and do not interpret a crossing-zero interval as equivalence.

## Is the topology gain compatible with WARP?

This is [ADAPTED, V] use of the authors' rank-two Walsh parameterization on the
matched local architecture/training recipe, not an exact reproduction of all
WARP-paper hyperparameters or architectures. The only connectivity difference
within each WARP pair is random versus U2.

The dense-M seed-0 pilot must achieve at least 40% best hardened validation for
random and a strictly positive U2-minus-random gain. Only then are the six
108K dense-M and six 20K convolutional-S follow-ups required. A failed gate is a
reported negative/inconclusive screen, not permission to tune another recipe.

Report best/final hardened and relaxed validation, their gaps, paired topology
effects, matched-budget raw references, and training/construction costs.
WARP's parameter-count change relative to raw must not be credited to U2.

The bounded GPU-0 pilot pair started at approximately 08:52 UTC while GPU 1
continued the convolutional controls. This is scheduling only, not a change to
the frozen matrix or promotion criterion. Both pilot runs are complete and the
frozen gate passed: random reaches 53.660% and U2 57.680% best hardened
validation, a +4.020 pp gain. All values and comparisons here are [ADAPTED, V],
one paired seed (0), at 20K updates; no training-seed interval is estimable.

| WARP wiring | Best hard (%) | Selected step | Relaxed at selected step (%) | Final hard (%) | Final relaxed (%) | Final gap (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 53.660 | 8K | 54.480 | 52.280 | 53.780 | 1.500 |
| U2 | 57.680 | 20K | 58.800 | 57.680 | 58.800 | 1.120 |

Both arms have 512K LUT functions, 2.048M trainable WARP parameters, zero
learned routing, and 1.235 GiB peak allocated GPU memory. Random/U2 wall times
are 7.690/7.685 minutes and layer construction times are 5.028/14.124 seconds.
The matching raw seed-0 arms use 8.192M parameters and 1.123 GiB; WARP's lower
parameter count is not a U2 benefit and does not imply lower peak memory here.
At the same 20K budget WARP trails raw by 1.740 pp for random and 1.180 pp for
U2. Thus this pilot supports a connectivity gain within WARP, not superiority
of this adapted WARP recipe over raw training.

`summary/fourth_round_warp_pilot.json` preserves the complete pilot evidence,
raw references and hashes. `logs/fourth_round/warp_promotion.json` records the
verified gate. All six 108K dense-M and six 20K convolutional-S follow-ups are
complete. The pilot alone was not completed compatibility confirmation.
Expansion is conditional on that positive pilot, and seed 0 is reused at the
longer budget alongside seeds 1/2. The 20K pilot and 108K seed-0 run are not
independent seed realizations and must not be pooled as extra replicates.
The promoted study is exploratory, not a selection-independent confirmatory test.

### Complete dense-M WARP study

All six runs completed 108K from scratch with paired seeds 0/1/2 and passed
artifact and checkpoint-metadata checks. Source:
`summary/fourth_round_warp_dense.json`, which includes all curves, raw references,
costs and hashes. All following WARP values and effects are [ADAPTED, V].

| Wiring | Seed | Best hard (%) | Selected step | Selected relaxed (%) | Final hard (%) | Final relaxed (%) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0 | 53.660 | 8K | 54.480 | 50.600 | 52.400 |
| Random | 1 | 53.160 | 8K | 54.480 | 50.920 | 51.920 |
| Random | 2 | 52.920 | 24K | 54.040 | 50.800 | 52.680 |
| U2 | 0 | 58.000 | 22K | 58.660 | 55.400 | 56.700 |
| U2 | 1 | 58.080 | 12K | 59.580 | 55.580 | 56.720 |
| U2 | 2 | 57.760 | 10K | 59.000 | 55.420 | 56.760 |

Random/U2 best-hard means ± sample SD are 53.247 ± 0.378% and
57.947 ± 0.167%. Paired gains are +4.340/+4.920/+4.840 pp, mean +4.700 pp,
sample SD 0.314 pp, exploratory paired Student-t 95% CI [3.919, 5.481], 3/3
wins. Final-hard means are 50.773 ± 0.162% and 55.467 ± 0.099%; paired final
gains are +4.800/+4.660/+4.620 pp, mean +4.693 pp, sample SD 0.095 pp,
95% CI [4.459, 4.928], 3/3 wins. Final is a reported secondary endpoint, not a
new selection rule. Every run peaked before 108K; the late accuracy decline
is retained in the curves and table.

Mean selected relaxed-minus-hard gaps are 1.087/1.133 pp for random/U2;
final gaps are 1.560/1.260 pp. Both methods have 512K LUT functions, 2.048M
trainable WARP parameters, zero learned routing and 1.235 GiB peak allocation.
Mean wall times are 41.827/41.667 minutes and layer construction 4.980/14.221
seconds. U2's paired extra layer construction is 9.241 seconds. These are
shared-host timings, not a speed or physical-cost claim.

Matched raw 108K best-hard reference values and WARP-minus-raw differences:

| Wiring | Raw seeds 0/1/2 (%) | WARP − raw seeds 0/1/2 (pp) | Mean difference (pp) | Paired 95% CI (pp) |
|---|---|---|---:|---|
| Random | 55.400, 55.340, 54.740 | −1.740, −2.180, −1.820 | −1.913 | [−2.496, −1.331] |
| U2 | 59.320, 59.780, 59.540 | −1.320, −1.700, −1.780 | −1.600 | [−2.211, −0.989] |

Raw random is [REPRODUCED, V], raw U2 [OUR, V], and parameterization contrasts
[ADAPTED, V]. Paired sample SDs are 0.234/0.246 pp; WARP wins 0/3 in both
families. Raw uses 8.192M parameters and 1.123 GiB peak allocation at the same
512K LUT count. Thus U2's dense connectivity gain survives this adapted modern
parameterization, but WARP does not improve raw accuracy or peak memory here.
The parameter-count reduction belongs to WARP, not U2. All group statistics,
paired effects, construction deltas and raw comparisons were independently
reconstructed. These dense results do not establish a convolutional gain;
the completed convolutional study is reported separately below.

### Complete convolutional-S WARP study

All six runs completed 20K, with paired seeds 0/1/2, unchanged nine-channel
architecture and frozen recipe. Source: `summary/fourth_round_warp_conv.json`.
All WARP values and contrasts below are [ADAPTED, V].

| Wiring / seed | Best hard (%) | Selected step | Selected relaxed (%) | Final hard (%) | Final relaxed (%) |
|---|---:|---:|---:|---:|---:|
| Random / 0 | 45.020 | 8K | 45.880 | 36.060 | 38.980 |
| Random / 1 | 43.300 | 4K | 46.200 | 37.600 | 39.200 |
| Random / 2 | 44.640 | 10K | 44.940 | 38.840 | 40.200 |
| U2 / 0 | 45.020 | 8K | 45.660 | 30.320 | 36.920 |
| U2 / 1 | 42.900 | 6K | 44.220 | 37.200 | 39.240 |
| U2 / 2 | 45.300 | 4K | 46.460 | 41.340 | 42.820 |

Best-hard means ± sample SD are 44.320 ± 0.904% random and
44.407 ± 1.312% U2. Paired gains are 0.000/−0.400/+0.660 pp: mean
+0.087 pp, sample SD 0.535 pp, paired Student-t 95% CI [−1.243, 1.417],
one win, one tie and one loss. Final-hard means are 37.500 ± 1.393% and
36.287 ± 5.566%; paired gains −5.740/−0.400/+2.500 pp, mean −1.213 pp,
SD 4.180 pp, CI [−11.597, 9.171], 1/3 wins. These exploratory unadjusted
intervals establish neither a positive convolutional effect nor equivalence.
Every run peaks before the endpoint; retain the late declines. Selected
relaxed-minus-hard mean gaps are 1.353/1.040 pp, final gaps 1.960/3.373 pp.

Both arms have 83,552 whole-model LUT functions, 334,208 WARP parameters,
zero learned routing and approximately 2.176 GiB peak allocation. Mean wall
times are 17.123/17.152 minutes for random/U2; mean layer construction is
0.200/1.375 seconds. U2 adds 1.175 seconds (paired SD 0.004 seconds,
95% CI [1.164, 1.186]). These shared-host observations are not speed claims.
The 874,496 LUT applications per image exclude pooling/output aggregation.

| Wiring | Matched raw best seeds 0/1/2 (%) | WARP − raw seeds 0/1/2 (pp) | Mean (pp) | Paired 95% CI (pp) |
|---|---|---|---:|---|
| Random | 55.000, 57.960, 58.460 | −9.980, −14.660, −13.820 | −12.820 | [−19.019, −6.621] |
| U2 | 58.500, 59.540, 58.500 | −13.480, −16.640, −13.200 | −14.440 | [−19.186, −9.694] |

Raw references are [REPRODUCED, V] random and [OUR, V] U2. WARP-minus-raw
sample SDs are 2.495/1.910 pp and both win 0/3. Raw uses 1,336,832 parameters
but lower peak memory (approximately 1.831 GiB). The parameter reduction
belongs to WARP, not U2. This adapted recipe supplies no established
convolutional topology gain and performs substantially worse than raw. Do not
retune a rescue recipe or generalize the positive dense result to convolution.

### Historical execution notes

The incremental notes below preserve run order and partial observations. Their
pending counters are historical; the completed coordinate summaries above and the
top-level checklist give the current state. Operational details are also in
`EXPERIMENT_LOG.md`.

The seed-2 108K pair has started on GPU 0 while GPU 1 finishes attribution.
The random seed-2 108K run has since completed and passed artifact and metadata
checks: best hard validation 52.920% at 24K (relaxed 54.040%), final hard/relaxed
50.800%/52.680%, 41.503 minutes and 1.235 GiB peak allocation [ADAPTED, V].
Its already-assigned U2 partner is now running. This single completed arm is
execution evidence, not a paired or three-seed compatibility conclusion.
Random seed 0 has also completed and passed the same audits: best hard/selected
relaxed validation 53.660%/54.480% at 8K, final hard/relaxed 50.600%/52.400%,
41.984 minutes and 1.235 GiB peak allocation [ADAPTED, V]. Its assigned U2
partner is running on GPU 1. Both completed random runs peaked before their
108K endpoints; preserve the final accuracies rather than reporting only the
selected best.
The seed-2 U2 partner has now completed and passed artifact/metadata checks:
best hard/selected relaxed validation 57.760%/59.000% at 10K, final hard/relaxed
55.420%/56.760%, 41.321 minutes, 1.235 GiB peak allocation and 14.248 seconds
layer construction [ADAPTED, V]. The complete seed-2 pair gains +4.840 pp in
best hard validation and +4.620 pp at the final endpoint. This is descriptive
n=1, without a training-seed interval; the three-seed conclusion remains pending.
Both arms peaked before 108K. The bounded GPU-0 dense assignment exited 0;
after fresh GPU/CUDA and gate checks, GPU 0 started the four preregistered
convolutional WARP cells for seeds 0/1 while GPU 1 continues the dense study.
Receipt: `logs/fourth_round/parallel_assignment_gpu0_warp_conv_seeds01.json`.
Dense U2 seed 0 has also completed and passed the same checks: best hard/
selected relaxed validation 58.000%/58.660% at 22K, final hard/relaxed
55.400%/56.700%, 41.880 minutes, 1.235 GiB peak allocation and 14.233 seconds
layer construction [ADAPTED, V]. Its paired gains are +4.340 pp best hard and
+4.800 pp final hard, independently reconstructed from CSV. Both completed
dense pairs (seeds 0/2) are positive; the seed-1 pair remains required before
the three-seed report. GPU 1 has advanced to dense random seed 1.

The first convolutional WARP arm, random seed 0 at 20K, has completed and
passed artifact/metadata checks: best hard/selected relaxed validation
45.020%/45.880% at 8K, final hard/relaxed 36.060%/38.980%, 17.124 minutes,
2.176 GiB peak allocation and 0.199 seconds layer construction [ADAPTED, V].
Best hard accuracy is 9.980 pp below the matched raw seed-0 arm (55.000%);
final hard accuracy is 8.960 pp below its own selected best. The WARP arm uses
334,208 trainable parameters versus raw's 1,336,832, but more peak memory than
raw's approximately 1.831 GiB. These parameter/memory differences are not U2
benefits. Retain this negative raw-context result and the full curve; do not
retune the preregistered recipe.
Its U2 partner has now completed and passed artifact/metadata checks: best hard/
selected relaxed validation 45.020%/45.660% at 8K, final hard/relaxed
30.320%/36.920%, 17.103 minutes, 2.176 GiB peak allocation and 1.369 seconds
layer construction [ADAPTED, V]. The seed-0 pair therefore ties on best hard
validation (0.000 pp) and favors random at the final endpoint (U2 − random
−5.740 pp). U2's adapted WARP best is 13.480 pp below its matched raw arm
(58.500%). Both best/final paired effects were independently reconstructed
from CSV. This is descriptive n=1 with no training-seed interval; the remaining
two paired seeds are required. It supplies no positive convolutional WARP
compatibility evidence yet, and its decline must not be hidden by best-only
reporting. GPU 0 has advanced to convolutional random seed 1 without retuning.
Both seed-1 random arms have now completed and passed artifact/metadata checks.
Convolutional S at 20K reaches 43.300% best hard at 4K (selected relaxed
46.200%), final hard/relaxed 37.600%/39.200%, 17.115 minutes, 2.176 GiB peak
allocation and 0.200 seconds layer construction [ADAPTED, V]. Dense M at 108K
reaches 53.160% best hard at 8K (selected relaxed 54.480%), final hard/relaxed
50.920%/51.920%, 41.994 minutes, 1.235 GiB and 4.979 seconds construction
[ADAPTED, V]. Both queues have advanced to their U2 seed-1 partners. These are
individual-arm execution results; neither full three-seed WARP coordinate is
complete. Four cells remain: the two active U2 partners and convolutional seed 2.
Convolutional U2 seed 1 has since completed and passed artifact/metadata checks:
best hard/selected relaxed 42.900%/44.220% at 6K, final hard/relaxed
37.200%/39.240%, 17.100 minutes, 2.176 GiB peak allocation and 1.378 seconds
layer construction [ADAPTED, V]. Its best and final U2-minus-random gains are
both −0.400 pp, independently reconstructed from CSV. Best WARP accuracy is
16.640 pp below its matched raw arm. The completed convolutional pairs have
best effects 0.000/−0.400 pp and final effects −5.740/−0.400 pp (seeds 0/1).
Neither supplies a positive topology effect under this adapted recipe; seed 2
is still required before the three-seed report.
The bounded GPU-0 seed-0/1 assignment exited 0. After fresh GPU/CUDA and queue-
position checks, GPU 0 started only random convolutional seed 2 at approximately
12:21 UTC, while GPU 1 continues dense U2 seed 1. The primary's last recorded
step was 42K/108K, leaving sufficient observed lead for the roughly 17-minute
side run. The final convolutional U2 seed 2 remains with the primary, not this
side assignment. Receipt:
`logs/fourth_round/parallel_assignment_gpu0_warp_conv_random_seed2.json`.
All fourteen declared WARP/raw pairs were independently compared with parser
defaults resolved: training recipes match apart from parameterization, output/
config paths and the documented inactive fixed-routing Gumbel flag. This checks
the declared comparisons; each training run still requires artifact validation.

## Does the frozen model survive an untouched distribution shift?

CIFAR-10.1 v6 is a frozen distribution-shift evaluation on the same ten classes,
not evidence of new-task training or universal cross-dataset transfer. The ten
existing raw checkpoints were selected independently of fourth-round outcomes:
dense M random/U2 seeds 0/1/2, full S seed 0, and full M seed 0. Configuration
and checkpoint hashes preceded any new dataset download or read.

Use checkpoint thresholds and the original evaluation input scale, without
augmentation, threshold calibration, fine-tuning, or adaptation. Record all
predictions and paired per-example correctness. Report seed-level uncertainty
for dense M and descriptive n=1 S/M effects; example-level comparisons do not
replace independent training seeds. A failed evaluation attempt must be audited
before any retry, never silently restarted.

Evaluation started at 07:59:11 UTC on September 6, after all frozen checks,
using GPU 1 concurrently with `fourth_mechanism_conv_s_nominal_seed0` training.
The selection is independent of fourth-round outcomes; this execution order
does not change the frozen protocol. Account for the overlap in training-time
comparisons: the last prediction was saved at 08:02:06 UTC. No new CIFAR-10
test queries are part of this round.

Complete CIFAR-10.1 v6 hardened test results, 2,000 examples per checkpoint.
Random accuracies are [REPRODUCED, T]; U2 accuracies and paired gains are
[OUR, T]. All models use their original frozen CIFAR-10 training recipe.

| Coordinate | Seed | Random (%) | U2 (%) | U2 − random (pp) |
|---|---:|---:|---:|---:|
| Dense M | 0 | 42.100 | 45.450 | +3.350 |
| Dense M | 1 | 41.850 | 45.950 | +4.100 |
| Dense M | 2 | 41.950 | 45.700 | +3.750 |
| Convolutional S | 0 | 46.000 | 45.950 | −0.050 |
| Convolutional M | 0 | 54.100 | 56.050 | +1.950 |

Dense M means ± sample SD are 41.967 ± 0.126% for random and 45.700 ± 0.250%
for U2. The paired gain is +3.733 pp, paired sample SD 0.375 pp, exploratory
Student-t 95% CI [2.801, 4.666], with 3/3 wins. S/M effects are descriptive n=1;
no training-seed confidence interval is estimable. U2-only/random-only correct
counts are 216/149, 215/133, 217/142 for dense seeds 0/1/2, 262/263 for S,
and 253/214 for M. These example-level counts do not supply extra training seeds.

The dense gain survives this shift; S's original CIFAR-10 gain does not persist
in this one frozen checkpoint pair. M is directionally positive but unreplicated.
Do not claim universal distribution-shift robustness or reinterpret S as proof
of equivalence. No method, checkpoint, or training recipe was changed in response.

Reporting failure and recovery: all ten queries and per-model JSON records
completed, then the evaluator's NumPy-valued win count failed aggregate JSON
serialization (`int64`). The original frozen evaluator and its started marker
were preserved. `recover_transfer_report.py` reconstructs the summary entirely
from saved predictions, verifies dataset/checkpoint/record hashes, prediction
correctness and freeze timing, and writes a separate recovery receipt. Its
seven regression tests pass. Recovery completed at 08:05:55 UTC with zero new
inference queries. Source: `summary/fourth_round_transfer_results.json` and
`logs/fourth_round_transfer/{dataset_receipt,recovery,completed}.json`; per-model
prediction records are in that same log directory. Read-only verification:
`venv/bin/python experiments/coverage_dlgn/recover_transfer_report.py`.
Never invoke the original evaluator again for these locked checkpoints/data.

## Resource and statistical claim boundaries

- All topology comparisons hold the architecture and parameterization fixed.
  No learned routing or extra inference operators are introduced.
- Offline construction time, training wall time and peak allocated GPU memory
  are separate quantities. Shared-host contention limits timing comparisons;
  training wall time is not an isolated inference-latency benchmark.
- `topology_seconds` sums layer-reported construction timers. It is not the
  complete model-initialization time and does not include every off-layer
  ancestry operation, including the factorial reference-body reconstruction.
  Do not present that sum as a comprehensive end-to-end construction benchmark.
- In the current `cost` dictionaries, `dense_gate_count` and deployed-routing
  fields cover only dense layers, including the classifier tail for convolutional
  models. Trainable-parameter counts cover the entire model. Do not mislabel
  dense-tail routing bits as total convolutional routing cost or physical area.
- Full nominal budgets, checked against the architecture regression tests:
  dense M has `4 * 128000 = 512000` LUT functions. Convolutional S has
  `7 * (32 + 128 + 512 + 1024) + 71680 = 83552` distinct LUT functions, and
  `7 * (32 * 32^2 + 128 * 16^2 + 512 * 8^2 + 1024 * 4^2) + 71680 = 874496`
  LUT applications per image. The latter excludes pooling and output aggregation
  and is not a physical cell count. These budgets are unchanged by the topology
  and raw/WARP choices in this round.
- Three-seed Student-t intervals are exploratory and not multiplicity-adjusted.
  Preserve raw seeds, sample SD, effect direction and uncertainty; do not select
  only favorable contrasts or checkpoints.
- Additional full LogicTreeNet-S seeds remain a later decision. Hardware,
  rank-four extensions and physical area/energy claims are deferred.
