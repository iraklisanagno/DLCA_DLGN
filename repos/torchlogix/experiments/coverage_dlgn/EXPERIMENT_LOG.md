# CoverageDLGN experiment history after the pre-experiment checkpoint

This log records operational failures and protocol decisions made after commit
`7332d91`. Raw run artifacts remain the source of truth for numerical results.

## July 24, 2026: Table 1 startup

- The first GPU check failed with `Failed to initialize NVML: Driver/library
  version mismatch`: the loaded kernel module was 580.159.03 while the NVML
  userspace library was 580.173.
- PyTorch 2.9.0+cu130 could still enumerate both RTX PRO 6000 GPUs and complete
  a matrix multiplication, but no training was launched while the host state
  was inconsistent.
- The machine was restarted by the user. After restart, `nvidia-smi` and
  PyTorch both report driver 580.173.02, CUDA 13.0, two GPUs, and no active GPU
  workload.
- All Python commands use `repos/torchlogix/venv/bin/python`; no package was
  installed.
- CoverageDLGN v3 (`semantic_balanced_hybrid`) is frozen. Comparator work adds
  learnable-routing controls and architectures but does not change v3 topology
  construction or scoring.
- The search budget and Table 1 comparator adaptations are frozen in
  `experiment_manifest.json` before accuracy screening.
- The first Mommen and LILogicNet CUDA smokes revealed that TorchLogix's
  historical `--connections-gumbel` declaration used `store_false`, making
  Gumbel noise active when the option was omitted. Both papers use ordinary
  softmax routing. Those artifacts were classified as failed protocol checks,
  the parser was corrected to an explicit boolean option defaulting to false,
  and every comparator configuration now records `connections_gumbel: false`.
- The first 5K queue launch stopped before creating a process because the
  config-parser validation pass had created empty output directories. The
  restart guard was tightened to permit an existing directory only when it is
  empty; a directory containing any partial artifact still requires manual
  classification before retry.
- The committed MNIST 5K screen completed all 26 runs with no process failure.
  Every run used source revision `51b1f52` and training implementation hash
  `8dccfbbecf87e07e15256812a2cbd15551a26c22eaffd76f4eb7d5be33d02d8a`.
  V3 incumbent achieved 96.58% hardened validation accuracy versus 96.00% for
  fixed random. LILogicNet Top-32 depth 3 with GroupSum temperature 15 led the
  comparator screen at 97.87%. These are seed-0 screening values, not final
  estimates.
- Python's default CSV dialect emitted CRLF records, which `git diff --check`
  reports as trailing whitespace. The metric, threshold, and topology writers
  now explicitly use LF records; existing screen CSVs were mechanically
  normalized without changing values.
- The committed MNIST 20K selection queue completed all 33 predeclared runs
  with no failures, skips, or nonzero return codes in 14,975 seconds using two
  GPUs. All 33 artifacts use source revision `2e499f2` and training
  implementation hash
  `f5bef4c78c6540e5c783bfa8c1033e61dd89bf9a50e2bb52897aa0d8272ac444`.
  The held-out test set was not used.
- Mean best hardened validation accuracy over paired seeds 0, 1, and 2 selected
  the following frozen MNIST candidates: fixed random 96.894% +/- 0.042%;
  CoverageDLGN v3 incumbent 97.233% +/- 0.101%; Mommen `Nc=16`, depth 2,
  97.756% +/- 0.134%; LILogicNet Top-32, depth 3, GroupSum temperature 15,
  97.956% +/- 0.142%; and BitLogic 98.083% +/- 0.060%. The `+/-` values are
  sample standard deviations, not confidence intervals.
- CoverageDLGN v3 incumbent and the pool-16 candidate were numerically tied at
  97.233330% to six decimal places. The incumbent was retained because the
  exact difference was below `1e-8` and it has the smaller routing/search
  representation. No V3 score, topology rule, or implementation was changed.
- The MNIST winner-selection source of truth is
  `summary/table1_mnist_selection.json`; its CSV companion preserves all
  candidate means, standard deviations, costs, runtimes, memory measurements,
  and hardened/relaxed metrics. These are validation-selection values, not
  final five-seed or held-out-test results.

## July 24, 2026: Table 1 Fashion-MNIST startup

- The first launch of the committed 29-run Fashion-MNIST screen failed before
  model construction because the machine restart had cleared the temporary
  dataset cache. `torchvision` attempted to download Fashion-MNIST, but the
  sandbox could not resolve the dataset host. The queue was interrupted after
  28 identical pre-training failures; the final BitLogic entry had not
  launched. No checkpoint, metric, or partial training artifact was created.
- The 28 tracebacks are preserved locally under
  `logs/failed/table1_screen_fashion_missing_cache_attempt1/`. The 28 verified
  empty result placeholders were removed so the queue restart guard can
  distinguish the clean retry.
- Fashion-MNIST was then downloaded once into
  `/tmp/torchlogix-datasets/data-fashion-mnist` using the repository venv.
  A download-disabled read verified 60,000 training and 10,000 test examples.
  The held-out test set remains unused for screening and selection.
- The next queue launch loaded Fashion-MNIST and calculated the expected
  thresholds `[0.25, 0.5, 0.75]`, but every sandboxed child failed at CUDA
  initialization with `No CUDA GPUs are available`. The queue was again
  interrupted after 28 pre-training failures, with no checkpoints or metrics.
  Tracebacks are preserved locally under
  `logs/failed/table1_screen_fashion_no_cuda_attempt2/`; verified empty result
  placeholders were removed.
- An unsandboxed check using `venv/bin/python` then confirmed PyTorch
  2.9.0+cu130, CUDA 13.0, both RTX PRO 6000 devices, and a finite CUDA tensor.
  The queue therefore requires the same GPU-enabled execution context.
- The GPU-enabled Fashion-MNIST screen then completed all 29 predeclared 5K
  runs with no failures, skips, or nonzero return codes in 4,375 seconds.
  Every valid artifact uses source revision `bbabe4d` and training
  implementation hash
  `f5bef4c78c6540e5c783bfa8c1033e61dd89bf9a50e2bb52897aa0d8272ac444`.
- The one-seed screen selected V3 swap fractions 0.125 (87.23%) and 0.5
  (87.18%) plus the 0.25 incumbent (86.97%); Mommen `Nc=8` depth 3 (87.82%),
  `Nc=16` depth 3 (87.68%), and `Nc=16` depth 2 (87.52%); and LILogicNet
  Top-32 depth 2/tau 30 (89.22%), depth 3/tau 15 (89.22%), and depth 3/tau
  20 (89.08%). Fixed random reached 85.37% and BitLogic reached 87.90%.
  These values only determine which candidates enter paired selection.
- The resulting Fashion-MNIST selection queue contains 33 predeclared 20K
  runs: seeds 0/1/2 for fixed random, each of the three candidates per
  tunable method, and BitLogic. Topology seed equals training seed, Gumbel is
  disabled, and the held-out test set remains locked.
- The committed Fashion-MNIST 20K selection queue completed all 33 runs with
  no failures, skips, or nonzero return codes in 15,160 seconds using two
  GPUs. All artifacts use source revision `47bb631` and training
  implementation hash
  `f5bef4c78c6540e5c783bfa8c1033e61dd89bf9a50e2bb52897aa0d8272ac444`.
- Mean best hardened validation accuracy over seeds 0/1/2 selected fixed
  random at 86.928% +/- 0.271%; CoverageDLGN v3 with swap fraction 0.5 at
  87.911% +/- 0.460%; Mommen `Nc=8`, depth 3, at 88.106% +/- 0.149%;
  LILogicNet Top-32, depth 2, GroupSum temperature 30, at 89.594% +/-
  0.035%; and BitLogic at 89.956% +/- 0.302%. Values after `+/-` are sample
  standard deviations. V3 improves its matched random baseline by 0.983
  percentage points without trainable routing parameters.
- The Fashion winner-selection source of truth is
  `summary/table1_fashion_selection.json`. These are validation-selection
  values, not final five-seed or held-out-test results.
- The nine long LILogicNet selection runs were retained for the already-frozen
  Fashion protocol. Starting with the next dataset/architecture cell, use a
  paper-prescribed fixed LILogicNet recipe when directly applicable;
  otherwise run the short screen and advance only its single best setting to
  three paired 20K seeds. This reduces long LILogicNet selection from nine
  runs to three per cell. The reduced search effort must remain visible in
  the protocol and does not change per-model trainable-parameter accounting.
