# UR1 execution handoff — 2026-09-06, 23:09 UTC

## User contract

Implement as much of the methodological plan as practical, preserve ALL prior
work, retain U2 if new changes are unhelpful, and present a unified methodology.
All pre-existing project files are unchanged. All new project work is in UR1;
the separate preservation directory holds the recovery archive/manifest.
No historical report, handoff, source, test, configuration, checkpoint or frozen
digest was edited. No commit/push/deletion. Unrelated VLLM workload untouched.

## Status at this snapshot

| Stage | State | Evidence / remaining work |
|---|---|---|
| 0 preservation/preregistration | Complete | Verified 78.798 GiB archive, 11,035-file manifest; full post-change recheck passed; new protocol and execution freeze |
| 1 functional dependencies | Complete | Same 24 preregistered dense best/final checkpoints; JSON/CSV/report/SVG; zero accuracy inference |
| 2 exact-degree factorial | Implemented, verified, RUNNING | 12 logical cells: 3 archived SS reused; 9 new SR/RS/RR runs queued; no completed accuracy claim yet |
| 3A ancestry-free nominal | Primitive and equivalence checks complete; benchmark scoped | Bit-identical on small boundaries and dense/conv model seeds 0/1/2; dense index-stack benchmark complete; NOT an end-to-end model-construction/diagnostic-mode benchmark |
| 3B long nominal | Deferred | No automatic promotion; inspect complete factorial before an explicit decision |
| 4 crossed seeds | Deferred | No automatic grid; requires a retained question and explicit decision |
| 5 full-S replication | Prepared, verified, QUEUED | Four unchanged random/U2 350K seeds 1/2, after factorial; no test evaluation implemented/invoked here |
| 6 paper handoff | Preliminary only | CLAIM_MAP.md and descriptive diagnostics available; final factorial/full-S evidence awaits completion |

## Running queue — inspect before doing anything

Queue PID **636295**, launched **2026-09-06 23:08:46 UTC**. This PID is a
snapshot, not a guarantee it remains live. Check the command line and receipts.
At launch GPU 0 began `ur1_factorial_dense_m_sr_seed0`; GPU 1 began
`ur1_factorial_dense_m_rs_seed0`. Both construction receipts were present at
23:09 UTC. Never launch another copy while the queue is running.
The subsequent launch-health check found both runs at 4,000 updates with two
finite validation rows and no errors. This is startup verification, not a final
accuracy result or grounds for changing the fixed matrix.

- Launch receipt: `logs/queue_launch.json`.
- Overall log: `logs/queue.log`.
- Per-run logs and `.started.json` / `.complete.json`: `logs/`.
- New artifacts: `results/ur1_*`.
- Exact configs/assignments: `matrix.json`, `configs/`.
- On complete factorial: `summary/factorial_results.json`, all five paired
  exploratory contrasts for best hard, final hard and final relaxed validation.
- On complete full-S training: `summary/full_s_results.json`, four new V rows.

The queue skips only audited completed receipts, refuses nonempty unclassified
outputs, and never restarts failed jobs. A subprocess/audit failure prevents new
launches after currently running work and prevents advancing to the next phase.
Do not call a checkpoint restart exact resume. If interrupted, preserve the
attempt and decide explicitly; do not overwrite or silently reuse its directory.

Estimated workload from historical timings: roughly 40 minutes for the nine
dense runs on two GPUs, then about ten hours for four full-S runs. Actual times
may differ. Neither phase is complete merely because its estimated time elapsed.

## Passed verification

- Original full GPU suite: **3,455 passed, 3,038 skipped, one warning**.
  Log: workspace `preservation/unified_refinement_20260906/baseline_full_tests.log`.
- New suite: **60 passed**; `logs/unit_tests.log`.
- Three factorial seeds: archived SS index identity, exact per-node/per-slot
  degree preservation, isolated RNG, identical weights/cost, shared-factor
  component identity, changed pair multisets; `logs/graph_preflight.json`.
- Six nominal model/seed checks: dense M and conv S, seeds 0/1/2, actual model
  indices and saved wiring match the new shared index primitive exactly.
- Six CUDA 10-update smokes: four factorial arms and full-S random/U2 paths;
  best/final metadata, actual saved wiring, initial weight/RNG/cost pairing.
- SS 10-update best/final state dictionaries reproduce the archived U2 smoke
  **bit-for-bit**. Completed reference accuracy runs were not retrained.
- Full preservation recheck: all **11,035** original files match the pre-change
  manifest; `logs/preservation_recheck.json`.

Frozen old implementation:
`3d1c59d3b2d0948235c66a78b5f4861cc0b55844f5291f4e602a7ae8dc519a0b`.
UR1 source freeze:
`04674a5678f891ce08cb960bf1c63c6fd85427da8ef0483e794515bc3c3bdfdc`.
Execution-freeze file SHA256:
`77bd84bc23601c5e127b81a0d4108ee4e32259faaab2de00d7ce9bbdf4399ae1`.

Do not edit/add any Python file under UR1 or edit PROTOCOL.md after this freeze:
the new auditor hashes that whole Python namespace. New reporting documents are
allowed, but code evolution requires a separately named, non-overlapping source
namespace and explicit provenance, not changing this execution's hash.

## Failure history and repairs BEFORE accuracy training

1. Initial new boundary tests omitted the old partial-coverage flag: corrected
   only new tests; no old builder edit.
2. Analyzer v1 required distinct parents and rejected native random before any
   completed checkpoint metrics. Preserve `functional_support.py` and its
   selection. Use `functional_support_v2.py` and `analysis_selection_v2.json`.
   The two frozen row lists are identical. Repeated-parent regression passes.
3. Initial GPU preflight omitted DATASET_PATH, triggering a duplicate download
   before any update. Interrupted only our child process. The logs, empty output
   and partial download were moved intact to
   `failed/preflight_missing_dataset_path`; nothing deleted. The adapter now
   requires explicit existing data and verifies official file integrity before
   output creation, so it fails before any future automatic download.
4. Static review caught old metric-label canonicalization rejecting the new null
   name. The process-local adapter uses the old formulas on actual indices and
   restores/serializes the correct `ur1_degree_null` labels. Regression and GPU
   smokes pass; historical source was not changed.

## Dataset and evaluation boundaries

Use only the TorchLogix venv and
`DATASET_PATH=/tmp/torchlogix-datasets`. Dataset file hashes are frozen; no package
install or CPU training. The old loader constructs an unused test loader and
integrity checks hash the existing files, but **no new test predictions or test
accuracy queries occur in UR1**. Stage 1 did not load any dataset at all.

After full-S training, select the already-validation-selected best checkpoints,
freeze all four new configurations/checkpoint hashes together, then arrange a
separately recorded one-time CIFAR-10 test evaluation. Never re-query old seed 0.
Combine its existing record with new seeds for the eventual three-pair report.
The current full-S report intentionally reports validation only and does not
pretend this evaluation is complete. No CIFAR-10.1 queries or new transfer claim.

## What the completed diagnostics support

Best-checkpoint final-layer structural raw-source support is approximately
15.96–15.97 for all four dense controls. Dependency-pruned path-support means
are random 6.627, balanced random 6.684, nominal 6.763, U2 6.764. These are
descriptive layer/gate means across three training seeds, not exact global
classifier dependence or physical pruning. Do not infer causality from them.

Nominal index-stack benchmark (three fresh processes each): legacy with ancestry
13.691 s / 779.23 MiB peak process RSS; UR1 index-only 13.433 s / 627.24 MiB;
UR1 retaining ancestry 13.630 s / 776.88 MiB. Memory benefit is specific to omitted
ancestry, and timing barely changes. Do not advertise a major end-to-end speedup.

U2 stays the unified incumbent regardless of these outcomes. The exact-degree
factorial tests its first/deeper structure; full-S replicates the same method
independently. Keep prior negative WARP/transfer outcomes and all null/negative
UR1 results visible. Hardware remains deferred.
