# UR1: Unified Refinement — additive execution contract

2026-09-06. Authorized by the user's request to implement as many planned
steps as possible while leaving all previous work unchanged. This is a new
experiment namespace, not a replacement method or a rewrite of old results.

## Preservation and unified paper methodology

CoverageDLGN-U2 remains the incumbent across dense and convolutional models:
semantic input ordering, degree-first multiscale matching, fixed deployment
indices, no learned routing. UR1 diagnoses that SAME methodology. Dense-only
success does not authorize selecting a different convolutional recipe and
calling the combination unified. UR1-N is a separately named index-equivalent
implementation of the existing nominal control, not evidence for a new winner.

All pre-existing files, including dirty/untracked source, reports, configuration
and checkpoint files, remain byte-identical. The verified recovery archive and
per-file manifest are in workspace `preservation/unified_refinement_20260906`.
It is same-disk recovery, not an off-machine backup. All new project artifacts
are confined to this directory. Old implementation hashes are never updated.
Old auditors remain executable against the unchanged live implementation.

## Stage 1: checkpoint-only dependency diagnostics

Before examining new metrics, freeze all twelve dense-M 20K mechanism runs
(native random, balanced random, nominal, U2; seeds 0/1/2), both best and final:
24 checkpoints, source/protocol/configuration/checkpoint hashes. No datasets,
predictions, reselection, or held-out inference. Harden with the established raw
rank-two argmax/tie convention. Analyze all 16 truth tables in tests first.

Report per-layer constants/unary/binary gates, relevant outgoing degree including
zeros, structural and dependency-pruned raw-source support, and backward path
reachability with every classifier aggregation input a root. Local relevance
is exact only for one LUT over Boolean assignments. Pruned path unions are an
UPPER BOUND on global dependency: reconvergence, thermometer constraints and
aggregation can cancel dependencies. Constants still contribute to sums. No
physical pruning, area or energy claim. Keep all seeds/checkpoints; n=3 training
replicates, not 24 independent samples. Plots/associations are descriptive.

## Stage 2: first/deeper exact-degree factorial

Use the frozen dense-M raw rank-two 20K recipe, evaluation every 2K, split seed
2027, training/topology seeds 0/1/2. Twelve logical cells: reuse three verified
archived U2 SS runs; train nine SR/RS/RR cells. SS/RS share IDENTICAL complete
structured deeper indices; SR/RR share identical randomized deeper indices.
SS/SR share the structured first layer and RS/RR its randomized counterpart.
Materialize the entire U2 stack BEFORE substitutions. Do not rebuild later
structured layers from changed ancestry. Native random remains separate context.

The fixed null uses 40 disjoint random-pair swap sweeps per input slot. Each
sweep samples a permutation of output gates; its two halves form disjoint
proposals. Swap same-slot predecessors only if both new gates retain distinct
parents. Seed is the uint32 first state of NumPy SeedSequence with entropy
[20260906, topology_seed, layer_index]. Preserve each named predecessor's
degree in EACH slot, not just a histogram. Report proposals, acceptance,
nontrivial swaps, unchanged edges and changed unordered-pair multiset. Require
changed pair multisets at actual experiment dimensions. This is neither uniform
graph sampling nor proof of graph non-isomorphism; unchanged labelled inputs
and held-fixed adjoining layers define the intervention. No accuracy-tuned null.

Primary endpoint is best hardened validation, unchanged selection rule. Report
final hard/relaxed values secondarily, all raw seeds, sample SD and all five
paired Student-t 95% intervals: SS-RS, SR-RR, SS-SR, RS-RR, SS-SR-RS+RR.
Include wins/ties/losses. Intervals are exploratory and unadjusted. First-layer
effects combine semantic/locality/pairing properties, not semantics alone.
Complete the fixed matrix regardless of sign; no favorable-contrast filter,
additional null search or automatic seed expansion.

## Stage 3A: shared ancestry-free nominal construction

UR1-N uses the same dimension/seed/matching primitive for dense predecessor sets
and convolutional channel sets, with explicit semantic metadata where the
incumbent uses it. Require index identity against frozen nominal construction
on small boundaries and both actual model families. Frozen nominal's selector
does not read ancestry contents when novelty selection is disabled. The new
index-only path omits ancestry diagnostics, which must be separately costed.
No changed spatial sampler, optimizer, parameters, gate budget or inference.

Benchmark separate processes with identical dimensions/seeds and cache policy;
report wall time and process peak RSS, output hashes, scope and diagnostic mode.
Do not label an index-only microbenchmark as end-to-end model construction.
Bounds/distinct parents/determinism apply to all supported dimensions; small
exhaustive tests are not a proof of unrestricted degree guarantees. Even full
matching stages cover each predecessor once; partial/odd/semantic cases require
their own measured degree records, not extrapolation of that guarantee.

## Gates, optional work and replication

Before accuracy: original full GPU regression suite, new unit tests, matched
parameter/cost/RNG and component-identity checks, archived SS/config matching,
four new GPU smokes, actual CUDA allocation and GPU ownership checks. Freeze
new source, configurations, reuse hashes and receipts after preflight. Never
overwrite outputs, replay completed runs, silently restart failures, or call
checkpoint restart exact resume. Use only the TorchLogix venv; no CPU training.

Full LogicTreeNet-S replication is the next independent priority: prepare four
new random/U2 seeds 1/2 at the unchanged nine-channel 350K recipe, reusing seed
0 without retraining or re-querying it. Select using validation and freeze all
new checkpoint/configuration hashes BEFORE any one-time CIFAR-10 test query.
Test evaluation requires its own execution receipt; training alone is not a
test result. This is replication on a historically used benchmark, not new
untouched transfer. No CIFAR-10.1 access anywhere in UR1.

Stages 3B (108K nominal) and 4 (crossed seeds) remain conditional decision items,
not automatic jobs: inspect the complete fixed factorial, report negatives,
and make an explicit evidence-based scope/budget decision first. No universal
compatibility/robustness claim; preserve prior negative convolutional results.
No hardware work, rank-four extension, learned routing, broad rescue sweeps or
new datasets. Final handoff must distinguish implemented, verified, queued,
running, complete and deferred; map claims to evidence without relabelling
historical results. New reporting stays here to honor strict preservation.
