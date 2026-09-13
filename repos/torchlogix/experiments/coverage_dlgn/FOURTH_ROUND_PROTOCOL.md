# Fourth round: mechanism, component attribution, WARP, and frozen transfer

Execution is allowed only when `summary/fourth_round_preregistration.json`
exists and its code/configuration/evidence hashes verify. That manifest records
the completed full suite and GPU smoke gate; this document alone is not proof
of verification or completed runs.

Authorized September 6, 2026. No hardware work, rank-four extension, or new
full-schedule LogicTreeNet-S seed belongs to this round. Fix handoff bugs and
pass full tests plus every family GPU smoke before any accuracy run.

## Immutable method and hypotheses

U2/V3/V4/U1 remain unchanged, protected by pre-edit golden digests.
`semantic_random_balanced` is a NEW ablation: keep U2's semantic first layer;
subsequent layers use random disjoint matchings, selecting least-used nodes
before random tie breaking. Fan-out spread is at most one and predecessor
slots are distinct. It does not claim uniform sampling over regular graphs.
`semantic_multiscale_nominal` is a NEW ablation: retain U2 candidate stages,
degree priority, and nominal tie order, but disable ancestry novelty selection.
The U2 helper's semantic-ancestry bug must be repaired before analysis.

Primary mechanism contrasts are U2 minus nominal and U2 minus balanced-random;
native random is context. Dense CIFAR-10 M and paper-faithful nine-channel S
use three paired seeds 0/1/2 and 20K updates, evaluation every 2K, split 2027.
Dense uses its frozen Adam 0.01/batch-100/no-augmentation recipe; S uses its
AdamW 0.02/decay-0.002/batch-128/crop-flip/residual-0.951 recipe.
Semantic first-layer pairs are identical across the three structured arms.
Report effect sizes and Student-t paired 95% intervals as exploratory evidence,
not multiplicity-adjusted confirmatory significance. Record all outcomes.

## Body/classifier factorial

S uses random/random, U2/random, random/U2, U2/U2, each at seeds 0/1/2 and 20K.
Existing explicit-random and U2/U2 endpoints are reused after artifact checks.
For the two mixed arms, hold the classifier wiring fixed for each classifier
method across body variants. The explicit `classifier_reference_u2` option
constructs U2-head indices from the offline U2-body reference ancestry. It
does not add a trained reference model or inference operator. U2/U2 indices
are bit-identical to the frozen model. Without this control, body changes
would silently rebuild U2 classifier wiring and confound attribution.
Report body effects with each head, head effects with each body, and the
per-seed interaction (both - body - head + random). Verify initial parameters,
post-construction RNG state, spatial coordinates, and same-head indices.

## Rank-two WARP compatibility

Use the authors' TorchLogix Walsh implementation (`parametrization=warp`),
soft sampling and temperature 1, with the exact matched coordinate's existing
optimizer/initialization recipe. Dense M first runs random/U2 seed 0 for 20K.
This is [ADAPTED], not an exact numerical reproduction of a WARP paper. It
isolates connectivity within WARP and compares against the matching raw arm;
it does not transplant JSC-specific settings or claim WARP's speedup as U2's.
Reference: https://arxiv.org/abs/2602.03527 and local
`pdfs/warp_logic_neural_networks.pdf`; authors' source is this TorchLogix base.

Promote only if random reaches at least 40% hardened validation and U2 has a
strictly positive best-hard-validation gain at the 20K endpoint. Otherwise
report the negative/inconclusive compatibility screen; do not tune on test.
On promotion, dense M random/U2 seeds 0/1/2 run 108K from scratch, and S
random/U2 seeds 0/1/2 run 20K. A checkpoint restart is not an exact resume.
Report hardened/relaxed curves, gaps, final and best validation, wall time,
peak memory, parameters and topology time. No new CIFAR-10 test queries.
Hash the six existing 108K raw dense-M reference runs alongside the reused
pilots, so WARP-versus-raw context is checked at the same update budget.

## Frozen transfer: CIFAR-10.1 v6

Use the recommended class-balanced v6 dataset (2,000 images) from
https://github.com/modestyachts/CIFAR-10.1. Evaluate existing raw U2/random
best-validation checkpoints: dense M seeds 0/1/2, full S seed 0 and M seed 0.
These ten checkpoints are fixed independently of fourth-round results. Freeze
checkpoint/config hashes and exact dataset revision before downloading or
opening image/label arrays. Use checkpoint thresholds, NHWC uint8 -> NCHW
float/255, no augmentation, recalibration, or adaptation. No CIFAR-10 test
re-evaluation. Store predictions and per-example paired correctness for all
predeclared models. A durable started marker precedes label access; interruption
requires manual accounting, not an automatic repeat of the locked test.
Before any new dataset access, load all ten selected checkpoints and verify
finite hardened outputs on synthetic inputs as part of the GPU smoke gate.
Report seed-level paired intervals for dense M and descriptive n=1 S/M effects;
an example-level test does not replace training-seed replication. Retain every
result; no transfer-based method or checkpoint selection.

## Verification, provenance, and reuse

Prepare configs idempotently; freeze code/protocol/config hashes only after
verification. GPU smokes must complete with finite metrics and checkpoints.
Paired construction checks must show equal gate weights, spatial samples and
RNG states; factorial same-head indices must be equal. No CPU training.
Before accepting reused outputs require checkpoints, complete 20K validation
history, config equality apart from declared topology/metadata fields, and
frozen content hashes. All new outputs have fourth-round names and must not
overwrite historical directories. Abort the current stage on any failure.
Historical fixed-routing endpoints have different `connections_gumbel` metadata;
this option is consumed only by learned routing and has no effect here. Keep
the archived values visible and regression-test fixed-model bit identity with
both settings. Do not silently normalize historical configurations.
Raw seed values, sample SD, paired Student-t intervals and wins are required.
There is no automatic full-S seed promotion in this protocol.
