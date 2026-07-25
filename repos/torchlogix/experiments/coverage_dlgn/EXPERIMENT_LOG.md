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
