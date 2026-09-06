# Durable instructions for AI contributors

Read [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md) before changing or running the
MarginSynth project. It reconstructs the method, evidence, no-go status,
commands, known issues, and next research gate.

## Project and scope

- The active research target is a reproducible DATE paper about post-training
  hardware simplification of hardened DLGNs.
- The implementation lives in `repos/torchlogix/experiments/marginsynth` on
  branch `mmarginsynth`. `repos/torchlogix` is part of the root Git worktree,
  not a submodule.
- The current per-gate/hardware-aware MarginSynth is a documented no-go for a
  state-of-the-art compression claim. Do not describe it as superior overall.
- The proposed next direction is hardware-budgeted cone/cut resynthesis. It has
  not been implemented or validated. Do not silently present it as completed.
- Keep the primary method independent of Unit Tying. The existing hybrid is an
  ablation, not the paper's primary method.

## Research integrity and data firewall

- Treat this as paper work: save resolved configs, seeds, split indices and
  hashes, code revision, checkpoints, environment/tool versions, commands,
  timings, memory, raw metrics, failures, rewrite/repair traces, circuits, and
  synthesis logs.
- Never delete or overwrite historical methods or results. Use new versioned
  configs and result directories; preserve failed and infeasible runs.
- Training selects model parameters from training data; validation selects the
  source checkpoint; calibration drives resynthesis/repair/guards; test stays
  sealed until the full method, seeds, operating point, and artifact hashes are
  frozen.
- Do not run `evaluate_frozen_test.py` for a new method during development.
- Report accuracy changes in percentage points. Report cost reductions with an
  explicit denominator (pre-exact source, exact baseline, or competitor).
- Exact simplification and approximate resynthesis are distinct. Never call an
  approximate circuit functionally equivalent without a separate proof.
- Do not call ABC nodes, generic cells, or an operation-cost sum mapped LUTs,
  silicon area, timing, power, or PPA.
- Numbers in `notes/` are navigation aids. Verify manuscript-critical claims
  in the corresponding `pdfs/` paper. Perform a current literature check before
  claiming novelty or state of the art.

## Environment and execution

- Run Python commands from `repos/torchlogix` with its existing `venv`.
- Install Python dependencies only into `repos/torchlogix/venv`, never into the
  user account. Do not replace the PyTorch build without recording CUDA/driver
  compatibility.
- Source training, joint resynthesis, and recovery must run on CUDA and must
  fail instead of silently falling back to CPU. Check `torch.cuda.is_available()`
  before launching them.
- Exact packed simulation, circuit export, compiled-C verification, Yosys, and
  Berkeley ABC are intentionally CPU/compiler stages.
- The customary dataset root is `/tmp/torchlogix-datasets`; record any change.
- Yosys and Berkeley ABC are system tools, not Python packages in the venv.
- The latest recorded stack is Python 3.12.13, PyTorch 2.9.0+cu130, CUDA 13.0,
  Yosys 0.9, and Berkeley ABC 1.01. A final paper flow needs a modern frozen
  toolchain and a declared FPGA target or characterized cell library.

## Git and artifacts

- The generated `repos/torchlogix/experiments/marginsynth/results/` tree is
  intentionally Git-ignored; it was about 27 GB at handoff. Do not force-add
  it. Use an external immutable artifact archive plus checksums for release.
- Checkpoints and large arrays are ignored globally. Do not weaken these rules
  merely to make a run appear self-contained.
- Commit source, tests, configs, compact summaries, protocols, and research
  decisions. Keep generated binaries/netlists/logs outside normal Git history.
- Before committing, inspect unrelated user changes, run relevant tests and
  `git diff --check`, and avoid destructive Git commands.

## Implementation expectations

- Generic graph/circuit functionality belongs in `src/torchlogix/circuit.py`
  or a reusable source module. Paper-specific objectives, orchestration, and
  baselines stay under `experiments/marginsynth`.
- Every new rewrite needs stable serialization, safe apply/undo, deterministic
  replay, exhaustive tiny truth-table tests, and incremental-versus-full
  simulation checks.
- Every selected circuit must pass model/circuit and compiled-C equivalence to
  its own hardened candidate before synthesis.
- Use identical exact simplification and hardware commands for all methods in
  one comparison.
- No new Bayesian sweep should start until a component-level experiment shows
  that a new structural mechanism improves candidate generation. More tuning
  of the current per-gate method is not the next step.
- For the proposed cone/cut pivot, predeclare the hardware budget, behavior
  guards, fallback rule, success threshold, seeds, and action-contribution
  ablations before expensive runs.

## Routine checks

From `repos/torchlogix`:

```bash
PYTHONPATH=. venv/bin/pytest -q \
  tests/test_marginsynth.py \
  tests/test_bayesian_summary.py \
  tests/test_dense_transfer_summary.py \
  tests/test_hardware_ranking.py

PYTHONPATH=. venv/bin/pytest -q
git diff --check
```

Use `DATASET_PATH=/tmp/torchlogix-datasets` and an explicit
`CUDA_VISIBLE_DEVICES` value for GPU experiment commands. Long reproduction
commands and their expected outputs are in `docs/AI_HANDOFF.md` and
`repos/torchlogix/experiments/marginsynth/RUN_LOG.md`.
