# UR1 — Unified Refinement

Additive methodological work requested September 6, 2026. **U2 is unchanged and
remains the unified incumbent.** All new implementation, tests, protocols and
evidence live here. Do not edit the historical subtree to update these results.

## Preservation

Before adding this package, 11,035 pre-existing files/symlinks (78.798 GiB) were
archived and verified against a per-file manifest, including ignored checkpoints
and the dirty/untracked working tree. Workspace location:
`preservation/unified_refinement_20260906/{workspace.tar,manifest.json}`.
Archive SHA256: `060a662599204b5032bad6f3a8deb1a928c5415491b317764c2be26a3e86e804`.
This is same-disk recovery, not an off-machine backup. Git, environments and
transient caches were excluded; see the manifest for the exact list.

Do not extract over the live workspace. If recovery is ever needed, first
verify the manifest/archive and extract the required members to a NEW separate
directory for comparison. No recovery operation is needed to continue using U2:
its original source, configurations, checkpoints and reports remain in place.

Old implementation hash is deliberately unchanged. New files are nested outside
the old auditor's source/test glob. Never edit either frozen source digest to
certify a different implementation. No commit or forced addition of model
binaries is part of UR1.

## Implemented components and boundaries

- [Protocol](PROTOCOL.md): preservation, unified-method interpretation, fixed
  factorial, inference boundaries and stopping rules.
- `functional_support_v2.py`: read-only analysis of the same 24 frozen dense
  checkpoints. [JSON](summary/functional_support.json),
  [CSV](summary/functional_support.csv), [report](summary/FUNCTIONAL_SUPPORT.md),
  [plot](summary/functional_support.svg). No dataset or accuracy inference.
- `construction.py`: isolated-RNG, exact per-node per-slot degree null; shared
  ancestry-free nominal index primitive for dense and convolutional channels.
- `train_adapter.py`: process-local index substitution before optimization;
  original trainer/source untouched. Complete U2 reference stack is constructed
  before combining first/deeper components. Weights, RNG and cost are checked.
  New `ur1_arm` configuration and construction receipts identify actual wiring.
  `connections_init_method=semantic_multiscale_balanced` describes the reference
  construction, NOT every final index in a randomized arm. Check the receipt and
  actual saved indices; randomized layers are labelled `ur1_degree_null`.
- `prepare.py`: 12 logical factorial cells (9 new, 3 reused U2) and four new
  full-S 350K random/U2 replications, with separate paths and exact old recipes.
- `verify.py`: graph/initialization audits, archival identity, six tiny GPU
  smokes, checkpoint metadata/wiring audits and independent execution freeze.
- `queue.py`: two-GPU bounded training and per-run completion receipts; no
  automatic restart, no test queries. Reports each complete phase independently.
- `benchmark.py`: fresh-process dense index-stack timing/RSS benchmark. This
  is NOT end-to-end model initialization or a full diagnostic-mode benchmark.
- `report.py`: all five exploratory paired factorial contrasts and additive
  descriptive evidence, with no edits to old paper tables.

`functional_support.py` and `analysis_selection.json` preserve the aborted first
analyzer version: it rejected native-random repeated parents before producing
any completed checkpoint metrics. Use **v2**, which retains those actual edges;
its frozen list contains the SAME 24 checkpoints, not a favorable reselection.
Boundary-test setup mistakes and this failure are recorded in
[development_failures.json](logs/development_failures.json).
An initial GPU preflight omitted `DATASET_PATH` and started a duplicate download
before any training update. That attempt, its logs and partial download were
moved intact to `failed/preflight_missing_dataset_path`. The adapter now refuses
missing/corrupt datasets and requires an explicit path; nothing was deleted.

## Verification and execution

Run from `repos/torchlogix`, using only its venv:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest experiments/coverage_dlgn/unified_refinement -q -p no:cacheprovider
DATASET_PATH=/tmp/torchlogix-datasets PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m experiments.coverage_dlgn.unified_refinement.verify
```

Preflight must pass before any accuracy queue. Inspect `execution_freeze.json`,
`logs/graph_preflight.json`, smoke receipts and the original full-test log in the
preservation directory. The original suite passed 3,455 tests, 3,038 skipped.
Never launch another queue while the current queue is running. Inspect
`logs/queue_launch.json`, `logs/queue.log`, `*.started.json`, `*.complete.json`
and the host processes first. A started receipt is NOT proof of completion.

After a verified freeze, the bounded queue command is:

```bash
DATASET_PATH=/tmp/torchlogix-datasets PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m experiments.coverage_dlgn.unified_refinement.queue --gpus 0 1 --phases factorial full_s
```

Do not run it blindly: it requires host CUDA allocation and GPU ownership checks.
Unrelated GPU workloads must stay untouched. Existing outputs are never replaced.
The full-S phase is roughly five hours per run based on old timings; four runs
on two GPUs require roughly ten hours, after the shorter factorial. These are
estimates, not measured UR1 completion times.

No test evaluator is invoked by this package. File-integrity checks include the
existing CIFAR-10 archive files; the unchanged legacy loader also constructs an
unused test loader. Neither step generates test predictions. New full-S best checkpoints must
be frozen together before a separately recorded, one-time CIFAR-10 evaluation.
Keep the old seed-0 test record; do not query it again. No CIFAR-10.1 access.
Stages 3B/4 remain deferred pending a complete factorial and an explicit decision.

## Paper-facing interpretation

The paper still presents one fixed-wiring U2 construction across dense and
convolutional models. UR1 is a diagnostic/replication package, not a collection
of separately chosen winning methods. Do not rename historical nominal results
as results of a new trained method. Index equivalence is an implementation fact,
not new accuracy, noninferiority, robustness or physical-cost evidence.

The new dependency analysis shows nearly saturated structural ancestry in all
four dense controls and much smaller dependency-pruned path support. U2 and
nominal are very similar on these descriptive measures. This cautions against
maximum-coverage explanations; it does not prove which wiring feature causes
accuracy. The exact-degree first/deeper factorial is the planned intervention.

The scoped benchmark finds little timing difference (legacy 13.691 s vs
index-only 13.433 s mean; n=3 fresh processes), but lower total peak RSS when
ancestry is omitted (779.23 vs 627.24 MiB). With ancestry retained, UR1 is
13.630 s and 776.88 MiB. Imports count toward RSS. Do not call this a major
end-to-end speedup or compare omitted diagnostics as though work were equal.

Use [HANDOFF.md](HANDOFF.md) for the latest dated execution snapshot. Machine
readable completion receipts and result manifests take precedence over prose.
