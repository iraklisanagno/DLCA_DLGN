# Next steps: bounded methodological refinement before the paper

Created September 6, 2026, following the completed fourth round.

Status: proposed plan, not an execution authorization or frozen preregistration.
No new training, checkpoint analysis, test access or method change is performed
by creating this document. It supplements the completed handoff; it does not
modify the fourth-round protocol, results or completion certificate.

## Objective and boundaries

Identify which fixed-wiring structure contributes to accuracy, determine
whether a simpler construction is sufficient, and strengthen the reliability
of the convolutional result. The aim is a clearer, defensible method, not a
guaranteed positive outcome or an indefinitely expanding search.

Evidence motivating this plan:

- U2 improves matched dense random, but native dense random already balances
  fan-out. Balance alone is not an established explanation.
- Neither completed 20K mechanism coordinate establishes an incremental
  novelty-selection benefit over nominal multiscale. This is not equivalence.
- The positive channel/body effect is conditional on fixed reference-U2 head
  indices; other factorial effects remain inconclusive.
- Dense adapted WARP and frozen dense transfer are positive; convolutional
  WARP and transfer do not support universal compatibility or robustness.
- Full-schedule LogicTreeNet-S/M results still have one paired seed each.

Sources: [fourth-round report](../FOURTH_ROUND_RESULTS.md),
[final validation evidence](../summary/fourth_round_results.json),
[transfer evidence](../summary/fourth_round_transfer_results.json), and
[completion audit](../summary/fourth_round_completion_audit.json).

## Sequence and proposed scope

| Stage | Work | Proposed accuracy-training budget | Exit deliverable |
|---|---|---|---|
| 0 | Preserve evidence; define analysis and experimental contracts | None | Versioned inputs, audit receipt and draft preregistration |
| 1 | Analyze functional dependencies in saved hardened dense circuits | None | Tested analyzer, per-layer/per-seed JSON/CSV and descriptive report |
| 2 | Exact-degree first-layer/deeper-layer structural factorial | 12 logical cells at dense M, 20K; normally 9 new and 3 verified reused U2 cells | All five paired effects, controls audit and decision report |
| 3A | Bit-identical ancestry-free nominal builder, if feasible | None | Equivalence tests, scoped guarantees and construction benchmark |
| 3B | Longer-budget nominal comparison, only after a separate decision | Proposed 6 cells at 108K, normally 3 new nominal and 3 reused U2 references | Accuracy/construction trade-off; not an automatic replacement |
| 4 | Separate topology from training variation, only if warranted | At most 2 methods × 3 topology seeds × 3 training seeds = 18 logical cells | Descriptive crossed-seed variance report |
| 5 | Full-schedule LogicTreeNet-S replication, separately approved | 4 new runs: random/U2 × seeds 1/2, 350K; retain existing seed 0 | Three paired full-schedule results and revised claim boundary |
| 6 | Stop exploration and assemble the paper evidence | None | Claim-to-evidence map and final figure/table specifications |

Recommended default methodological package: stages 0–2 only. Stage 3A is a
bounded engineering opportunity; stages 3B and 4 are conditional, not automatic
queue additions. Stage 5 is the next replication priority and must not depend
on obtaining favorable mechanism findings. Do not let optional stages delay it
indefinitely. All budgets exclude small, mandatory synthetic/GPU preflight tests.

## Stage 0 — preserve and preregister

1. Revalidate completed artifacts without retraining or new held-out inference.
   Preserve source/config/checkpoint hashes, query history and failure records.
2. Before implementation changes, archive a verifiable copy of the exact source
   needed by the old auditors. Their implementation hash includes source and
   tests: even a legitimate additive change can invalidate verification against
   the current checkout. Verify the archived source matches the old digest;
   give new work a separate implementation/protocol freeze. Never rewrite an
   old digest to certify a different implementation.
3. Freeze the stage-1 checkpoint list and metric definitions before examining
   those new metrics. Draft stage 2 before training: architectures, seeds,
   null generator, endpoints, all contrasts, cost fields and stopping rules.
4. Use separate artifact names and exclusive/idempotent writes. Reuse only
   after effective-config, initialization, wiring and artifact checks. No
   completed-run replay, automatic failed-run restart or claimed exact resume.
5. All Python uses `repos/torchlogix/venv`. Before any accuracy run, repair new
   bugs, pass focused/full tests and each new GPU path, verify actual CUDA
   allocation and check GPU ownership. Do not displace unrelated workloads.

Acceptance: reproducible manifests and passing invariants, with an explicit
approved execution scope. This planning document alone does not satisfy that
gate or grant permission to access another test set.

## Stage 1 — what does the hardened circuit use?

Start with all twelve dense-M 20K mechanism outputs: native random,
balanced-random, nominal and U2, seeds 0/1/2. Analyze both the already-selected
best checkpoint and final checkpoint (24 checkpoints), without reselection.
No dataset, predictions or model-accuracy inference is needed. Do not expand
to convolutional pooling/aggregation until dense definitions are validated.

Extract the actual hardened truth table using the model's established gate
selection and tie convention. For each LUT, an input is locally relevant if
flipping that input changes the output for at least one assignment of the other
input. Measure:

- Fractions of constant, one-input and two-input gates, per layer.
- Per-predecessor counts of locally relevant outgoing edges, including zeros.
- Raw-image-source path support after dropping locally irrelevant edges;
  count thermometer thresholds from one source consistently.
- Backward reachability to classifier aggregation inputs, distinguishing
  disconnected paths from merely constant-valued gate contributions.
- Differences between structural ancestry and dependency-pruned support, and
  best-versus-final changes, retaining every method and seed.

Local truth-table relevance is exact for that LUT over Boolean assignments.
Path-union support is only an upper bound on global functional dependence:
reconvergent cancellation, input constraints and output aggregation can remove
dependencies. Do not call it exact classifier support, physical gate removal,
or evidence that more two-input gates necessarily improve accuracy.

Tests: all 16 rank-two truth tables; constant/unary/binary cases; hardening
ties; fixed indices and source encoding; small synthetic networks with exhaustive
enumeration to verify the support upper bound; reconvergent cancellation;
checkpoint hash preservation and rejection of unsupported layer semantics.

Deliverables: a read-only analyzer, tests, input/source hashes, machine-readable
metrics and one descriptive plot. Accuracy associations use existing validation
values, not new queries. With three seeds, do not fit an elaborate predictive
model or present pooled gates/checkpoints as independent training replicates.

Decision: retain positive, negative and null associations. They motivate
interpretation, not a new learned routing objective or a favorable-case filter
for the already-planned structural factorial.

## Stage 2 — exact-degree structural intervention

Use dense CIFAR-10 M, rank two, raw LUTs, the established 20K recipe and split,
evaluation every 2K, paired training/topology seeds 0/1/2. No architecture,
optimizer, initialization, encoding, augmentation or gate-budget changes.

For each seed, first materialize the entire frozen U2 reference stack. Create
one degree-preserving randomized version of its first layer and one randomized
version of its deeper layers, using an isolated null-generator RNG stream.
Construct the four arms from these same materialized components:

| Arm | First-layer indices | Deeper-layer indices |
|---|---|---|
| SS | Structured U2 reference | Structured U2 reference |
| SR | Same structured first layer as SS | Randomized deeper component |
| RS | Randomized first-layer component | Same structured deeper layers as SS |
| RR | Same randomized first layer as RS | Same randomized deeper layers as SR |

Crucial: do not regenerate structured deeper indices from the changed first
layer's ancestry. That would change both factors at once. The reference-stack
construction is offline and adds no trainable reference model or inference
operators. Likewise, each randomized component must be held identical wherever
it is reused. The reference SS cell should match the archived U2 cell bit-for-bit.

The null generator swaps predecessor occurrences between gates, rejecting
out-of-bounds or repeated-parent proposals. Prefer within-slot swaps so that
each predecessor's use count in each input slot is also preserved. Fix the
proposal schedule, proposal count, seed mapping and failure rules using
graph-only checks before accuracy runs. Report accepted swaps, unchanged-edge
fraction and pair/motif diagnostics; do not claim uniform graph sampling.
Preserve the exact per-node degree sequence, not just its histogram or spread.
Check this changes pairings, rather than applying a consistent neuron relabeling.
No null-generation parameter search on validation accuracy is allowed.

First-layer randomization changes several pairing properties together, including
semantic pairing and locality. Its contrast isolates the specified first-layer
structure, not semantic ordering alone. This null is also not the native-random
baseline: retain native-random results separately as context.

Primary endpoint: best hardened validation under the unchanged selection rule.
Report final hard/relaxed accuracy and curves secondarily. Required contrasts:

1. First-layer structure with structured deeper layers: SS − RS.
2. First-layer structure with randomized deeper layers: SR − RR.
3. Deeper structure with structured first layer: SS − SR.
4. Deeper structure with randomized first layer: RS − RR.
5. Interaction: SS − SR − RS + RR.

Report all raw paired values, sample SD, paired Student-t 95% intervals and
wins/ties/losses. Treat the five intervals as exploratory and unadjusted;
three seeds do not guarantee useful power. Never select a favorable contrast
afterward and relabel it confirmatory. Do not pool head/layer conditions as
independent seeds.

Acceptance before launch: golden V3/V4/U1/U2 identity tests; exact node/slot
degree checks; bounds/distinct parents; deterministic null generation and RNG
isolation; identical gate parameters, budgets and unaffected components across
arms; tiny synthetic tests plus four GPU smokes. Match archived SS initialization
and effective settings before counting it as reusable. An incompatible archived
cell triggers a protocol/reuse decision, not a silent rerun.

Stopping rule: complete the fixed matrix, then report once. Positive intervals
support only the corresponding conditional effects. If effects are inconclusive
or negative, stop this mechanism expansion and narrow the claim; do not add
seeds or alternative nulls until one produces a favorable result. Stage 3B or 4
requires a separate, evidence-based approval and a new frozen protocol.

## Stage 3 — investigate simplification without rewriting history

### 3A: implementation-equivalent nominal construction

Inspect whether all decisions in the existing nominal strategy depend only on
dimensions, semantic first-layer metadata, seed and schedule, not ancestry
contents. If so, implement a separately identifiable ancestry-free construction
path that reproduces those indices exactly. Preserve the historical nominal
control and U2. If removing ancestry changes any decision, it is a new method,
not a performance optimization; stop and request a revised scope.

Prove or explicitly delimit guarantees for even/odd dimensions, partial stages,
fan-out, bounds, distinct inputs and determinism. Separate first-layer guarantees
from deeper-layer guarantees; do not extend the deeper balance bound to the
semantic first layer without proof. Exhaustive small cases and randomized
boundary tests supplement, rather than replace, mathematical justification.

Benchmark end-to-end CPU construction time and peak construction memory with
and without optional diagnostics, with equivalent cache conditions. Do not
compare a diagnostics-free builder to a fully instrumented U2 run without
disclosing that distinction. Report output hashes and whether full ancestry
metrics are requested; producing those metrics can still require ancestry work.

### 3B: conditional longer-budget accuracy check

If a simpler construction remains scientifically promising and stage 3A passes,
propose nominal versus frozen U2 at dense M, 108K, seeds 0/1/2. Normally this
requires three new nominal runs and reuse of three matched archived U2 runs;
native-random references provide context. Never replay a completed reference.

This is validation-based follow-up selected after earlier exploratory results,
not independent confirmation of a pre-existing headline method. A paired CI
crossing zero does not justify equivalence. If noninferiority is the intended
claim, agree a scientifically meaningful margin and power/sample-size plan
before execution; if the required budget is unavailable, report the trade-off
without a noninferiority claim. Do not assume three seeds are sufficient.

No replacement of U2 or reuse of old frozen-transfer results as evidence for
the simplified method. Any new headline method requires a separate freeze and
prospectively chosen evaluation; previously accessed benchmarks are not newly
untouched merely because the method name changed.

## Stage 4 — conditional topology/training variability study

Only propose this if a retained contrast is promising and its variation matters
to the claim. Default comparison: structured SS versus randomized RR from
stage 2, unless another comparison is explicitly preregistered before launch.
Use one coordinate/recipe, three isolated topology seeds crossed with three
training seeds, paired across methods: 18 logical cells, not a broad sweep.

Separate RNG streams for topology, weights and data order. Reuse existing
diagonal cells only if the new seed isolation reproduces their exact effective
configuration and initialization; do not assume reuse from seed labels alone.
Otherwise request a revised budget and keep archived runs untouched.

Show the full grid and topology-averaged/training-averaged paired differences.
Any variance-component or hierarchical interval analysis must account for the
crossed design. Nine cells are not nine independent topology draws, and three
levels per axis give imprecise variance estimates. Never search for the best
graph. Stop after the declared grid; null variance findings are reportable.

## Stage 5 — full LogicTreeNet-S replication

After approving the replication protocol, add random/U2 seeds 1/2 at the frozen
nine-channel 350K recipe, keeping the existing seed-0 evidence. This is four
new runs, independent of whether a simpler dense method is considered. Do not
replace U2 in this comparison following the dense analysis.

Select using hardened validation, then freeze all new checkpoint/config hashes
before one-time test evaluation. Preserve the existing seed-0 test record;
never evaluate it again. CIFAR-10 is a historically used benchmark, so this is
replication, not an untouched transfer claim. Do not re-query CIFAR-10.1 or
adapt the method using its completed outcomes.

Report all three paired seeds and uncertainty regardless of direction. If the
full-S effect remains uncertain, narrow the convolutional claim rather than
automatically adding M seeds. Full-M replication is a separate later decision.

## Stage 6 — mandatory stopping point and paper handoff

After the approved package completes, produce:

- A claim-to-evidence map distinguishing observed effects, descriptive mechanism
  associations, proven construction properties and unresolved hypotheses.
- A functional-dependency diagnostic, a structural-factorial effects plot,
  construction-cost comparison if stage 3A succeeds, and full-S replication
  table if stage 5 is executed.
- All raw seeds, fixed manifests, resource definitions, negative outcomes and
  audit commands, with clear OUR/REPRODUCED/ADAPTED and V/T labels.
- A method decision: preserve U2, explicitly introduce a validated simpler
  candidate, or write a narrower dense-centered paper. No retrospective
  relabeling of existing method results.

Hardware synthesis/place-route, rank-four extensions, learned routing,
dataset-specific topology tuning, broad WARP/Light/Gumbel rescue sweeps and
additional transfer datasets are outside this plan. No guaranteed publication,
accuracy improvement, causal explanation or physical-cost saving is assumed.
