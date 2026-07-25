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
- The frozen Table 1 winners were expanded into separate MNIST and
  Fashion-MNIST final queues with five seeds per method. Final training effort
  is matched at 200 effective epochs over the 54,000-example training split:
  108,000 iterations for batch size 100, 84,400 for batch size 128, and
  42,200 for batch size 256. This gives every method the same number of data
  passes while preserving its selected batch size. Validation remains the
  only evaluation source during training; held-out test evaluation is still
  locked.

## July 25, 2026: Table 1 final-queue divisibility recovery

- The first MNIST final queue completed all five fixed-random, all five
  CoverageDLGN v3, and all five Mommen runs successfully. The five LILogicNet
  and five BitLogic entries then failed before model or dataset construction:
  the generator assigned every method `eval_freq=2000`, but their batch-size
  specific totals of 42,200 and 84,400 iterations are not divisible by 2,000.
  The trainer's protocol assertion rejected each invalid configuration after
  about five seconds. These are configuration failures, not training results;
  their empty output directories are safe to reuse.
- The final-queue generator now preserves the 2,000-step cadence when it
  divides the run length and otherwise evaluates every four effective epochs.
  This produces valid intervals of 844 steps for batch size 256 and 1,688
  steps for batch size 128, with 50 validations over 200 epochs. A regression
  test covers all three Table 1 batch sizes, and both MNIST and Fashion final
  queues were regenerated before further launches.
- Corrected LILogicNet seeds 0 and 1 were recovered on separate GPUs without
  changing the training implementation. They completed all 42,200 iterations
  with best hardened validation accuracies of 97.900% and 97.883%,
  respectively. Each used about 4.50 GB peak allocated GPU memory and
  4,898--4,950 seconds of wall time. The held-out test set remains unused.
- A temporary internet interruption occurred after these jobs had launched.
  It did not affect either run because MNIST was already cached locally. A
  post-interruption audit found both corrected artifacts complete, both GPUs
  healthy, and no partial trained checkpoint.
- The corrected recovery queue then skipped the 17 complete artifacts and
  finished exactly the remaining eight runs with zero failures in 21,110
  seconds. All 25 MNIST final artifacts use the same training implementation
  hash,
  `f5bef4c78c6540e5c783bfa8c1033e61dd89bf9a50e2bb52897aa0d8272ac444`.
  Their source revisions are split 17/2/6 across `6c06d0d`, `93a956b`, and
  `9dcf7e7`; the later commits changed queue/protocol documentation and the
  evaluation-cadence generator, not the hashed training implementation.
- At 200 effective epochs, five-seed mean best hardened validation accuracy
  is fixed random 97.157% +/- 0.043%, CoverageDLGN v3 97.403% +/- 0.114%,
  Mommen 97.947% +/- 0.121%, LILogicNet 97.927% +/- 0.143%, and BitLogic
  98.123% +/- 0.042%. V3 improves its paired fixed-random baseline by 0.247
  percentage points with a 95% Student-t interval of [+0.078, +0.416] pp.
  The machine-readable source of truth is
  `summary/table1_mnist_final.json`; the held-out test set was still unused
  when this validation summary was frozen.
- Mean training wall time per final seed is 15.0 minutes for fixed random and
  V3, 35.9 minutes for Mommen, 82.0 minutes for LILogicNet, and 89.2 minutes
  for BitLogic. Peak allocated GPU memory is 0.099, 0.099, 1.002, 4.187, and
  3.557 GiB, respectively. V3 therefore preserves the random model's training
  and deployment cost while the adapted trainable-routing comparators trade
  substantial training cost for higher raw validation accuracy.
- The post-MNIST run-count policy reduces the Fashion final queue from 25 to
  19 entries: five paired seeds for fixed random and CoverageDLGN v3 and
  three seeds for each adapted comparator. The six unlaunched superseded
  seed-3/4 comparator configs were removed before the queue was committed.
