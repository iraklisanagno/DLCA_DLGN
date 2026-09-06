# Durable CoverageDLGN project instructions

Read `docs/AI_HANDOFF.md` before modifying code, protocols, results, or tables.
Although this `AGENTS.md` is located in the experiment subtree, its method and
evidence rules also apply when this project changes `src/torchlogix/`,
`experiments/train.py`, or `tests/`.

## Research scope

- Use `repos/torchlogix` as the implementation foundation. Do not move the
  convolutional work to `difflogic-light-master`; that repository lacks the
  required convolutional LGN support.
- The headline method is frozen **CoverageDLGN-U2**, strategy
  `semantic_multiscale_balanced`: semantic input ordering plus degree-first,
  deterministic multiscale matching-stage selection and no learned routing.
- Dense V3 (`semantic_balanced_hybrid`) is the strongest dense specialization.
  V4 (`semantic_channel_hybrid`), U1 (`semantic_degree_balanced`), V5, the
  channel-spatial adapter, class-head refinement, task-aware rewiring, and
  coverage/reuse are preserved ablations or negative results.
- Never silently change V3, V4, U1, or U2. Add a new strategy name for a new
  mechanism and regression-test the frozen methods for bit identity.
- Do not revive maximum ancestry coverage as the main explanation. The evidence
  says degree-balanced structured propagation drives most of the gain; greedy
  swaps and maximum ancestry can be neutral or harmful.

## Experimental integrity

- Reproduce the fixed-random baseline first. In an attribution pair, change
  only the topology strategy. Match architecture, dataset/split, preprocessing,
  gate budget, rank, parameterization, optimizer, updates, evaluation cadence,
  initialization, and seeds.
- Use hardened validation for selection. Freeze configuration and checkpoint
  hashes before held-out test access. Do not use test accuracy for tuning or
  repeatedly evaluate a locked test set.
- Report every number with provenance (`OUR`, `REPRODUCED`, `ADAPTED`,
  `REPRODUCED-NEGATIVE`, or `REPORTED`) and scope (`V` or `T`). Never compare a
  six-channel legacy/WARP-style model as though it were the nine-channel
  paper-faithful LogicTreeNet.
- Pair seeds across methods. Report raw seeds, mean, sample standard deviation,
  paired Student-t 95% confidence interval, and number of wins. Expensive
  comparators may use one local seed only when the protocol says so; retain the
  paper's reported multi-run result separately.
- Do not rerun completed jobs. Check the required artifacts and the phase
  summarizer/freeze manifest; `run_gpu_queue.py::is_complete` alone is not a
  sufficient archival check.
- Never overwrite result directories or frozen manifests. Move incomplete or
  interrupted attempts to a descriptive `results/failed/` location and record
  what happened in `EXPERIMENT_LOG.md`.
- Do not call a restart from a checkpoint an exact resume. The current harness
  does not preserve enough optimizer, dataloader, and RNG state to prove exact
  continuation.
- Keep reported, adapted, and reproduced values separate. A local failed
  reproduction does not replace the published number.

## Environment and GPU rules

- All Python and package operations must use `repos/torchlogix/venv`.
- Never run `pip install` outside a virtual environment. Prefer
  `uv pip install --python venv/bin/python ...` or
  `venv/bin/python -m pip ...`.
- This host requires CUDA 13. The validated stack is Python 3.12, PyTorch
  2.9.0+cu130, and torchvision 0.24.0+cu130. Use
  `repos/torchlogix/cuda130_pytorch29.def` as the Apptainer basis if needed.
- Training must run on GPU. Before a queue, check both `nvidia-smi` and an
  actual tensor allocation with `venv/bin/python`. If CUDA allocation fails,
  stop and report it; never fall back to CPU training.
- Use both GPUs for independent runs when available. Check existing processes
  and memory first. Do not displace unrelated workloads.

## Method and architecture invariants

- U2 is currently rank-2 only. A rank-r version is a separately named extension
  until it is validated; preserve rank-2 indices exactly.
- U2 construction is offline CPU preprocessing and deploys only fixed indices.
  It must not add trainable routing parameters, inference operators, gates, or
  hidden dataset-specific controls.
- Dense semantic inputs collapse thermometer thresholds to their raw
  `(channel, y, x)` source. Convolutional U2 changes channel pairing while
  preserving the established spatial receptive-field sampler.
- Paper-faithful convolutional CIFAR-10 S/M use three thresholds per RGB
  channel (nine Boolean channels). Legacy `ClgnCifar10Small/Medium` use six
  channels and are a separate protocol.
- `ClgnCifar10Large` is not yet a faithful published large-model reproduction.
- Do not run a Cartesian product of fan-in/rank, WARP/Light/Gumbel, and
  connectivity. Use a preregistered minimal matrix that isolates each axis and
  only expands positive cells.

## Evidence and artifact safety

- The result tree is large and mostly ignored by git. `.pt`, `.pth`, `.ckpt`,
  `.onnx`, arrays, archives, venvs, and datasets must remain ignored. Never add
  them with `git add -f`.
- Do not delete local checkpoints without explicit user approval and a backup
  decision. JSON/CSV summaries in git do not replace the model binaries.
- Treat frozen per-run artifacts and machine-readable summaries as more
  authoritative than prose tables. Some older README/RESULTS/DATE table lines
  are stale; update all affected documents together after changing evidence.
- Preserve failure history. Negative outcomes on CIFAR-100, V5, the
  channel-spatial adapter, class-head refinement, task-aware rewiring,
  coverage/reuse, and local rank-4 BitLogic are scientifically important.
- Never claim physical area/energy from nominal gate count or simplified IR.
  Hardware claims require identical synthesis/place-route flows and bit-exact
  equivalence.

## Coding and verification

- Use `apply_patch` for hand edits and preserve unrelated user changes.
- Add focused tests for determinism, index bounds, unique predecessors,
  fan-out balance, RNG isolation, ancestry propagation, checkpoint metadata,
  architecture/cost equality, and export equivalence as applicable.
- Known issue to fix before relying on topology-only U2 reports:
  `generate_dense_stack()` omits `semantic_multiscale_balanced` from its
  semantic ancestry strategy sets. The model-construction path is correct.
- The current evidence audit does not cover all third-round and LogicTreeNet-M
  U2 facts. Extend it before calling it a whole-project consistency audit.
- Focused verification from `repos/torchlogix`:

  ```bash
  venv/bin/python -m pytest \
    tests/test_coverage_topology.py \
    tests/test_experiment_protocol.py \
    tests/test_parametrization.py \
    tests/test_warp_fig4_protocol.py -q
  ```

- Full verification:

  ```bash
  venv/bin/python -m pytest tests -q
  ```

## Reporting style

- Keep `RESULTS.md` chronological and record failures as well as successes.
- Keep `PAPER_COMPARISON_TABLES.md` comprehensive and provenance-labeled.
- Keep `DATE_TABLES.md` concise and paper-facing, but do not hide unmatched
  protocols, seed counts, validation/test status, or resource trade-offs.
- Lead with the defensible claim: U2 improves fixed random on dense and
  convolutional CIFAR-10 at matched declared cost; V3 gives the strongest dense
  frontier. Do not claim universal dataset gains, state-of-the-art accuracy, or
  lower physical area yet.
