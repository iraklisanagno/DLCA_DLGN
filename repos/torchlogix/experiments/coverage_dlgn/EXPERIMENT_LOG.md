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

## July 25, 2026: Table 1 MNIST held-out final evaluation

- The guarded evaluator preflighted all 25 frozen best-validation checkpoints,
  confirmed that no prior test metric existed, and distributed the one-time
  held-out evaluation across both GPUs. All 25 commands completed successfully
  with zero failures or missing artifacts. The execution audit is
  `logs/table1_final_mnist/test/test_evaluation_summary.json`.
- Five-seed mean hardened test accuracy is fixed random `[REPRODUCED]`
  97.090% +/- 0.180%, CoverageDLGN v3 `[OUR-FINAL]` 97.500% +/- 0.099%,
  Mommen `[ADAPTED]` 98.084% +/- 0.066%, LILogicNet `[ADAPTED]`
  98.124% +/- 0.029%, and BitLogic `[ADAPTED]` 98.204% +/- 0.042%.
  Values after `+/-` are sample standard deviations.
- CoverageDLGN v3 improves its paired fixed-random baseline by 0.410
  percentage points on the held-out test set. The 95% Student-t confidence
  interval is [+0.108, +0.712] pp, and all five per-seed differences are
  positive. V3 and random have identical 48K deployed-gate budgets, 768K
  trainable gate logits, and no trainable routing parameters.
- `summary/table1_mnist_final.json` and its CSV export are now the final
  machine-readable sources of truth with `test_set_used=true`. The held-out
  test set will not be queried again for this Table 1 MNIST experiment.

## July 25--26, 2026: Table 1 Fashion-MNIST final validation

- The reduced final queue completed all 19 predeclared runs with zero
  failures or skips in 22,275 seconds using two GPUs: five fixed-random
  seeds, five CoverageDLGN v3 seeds, and three seeds each for Mommen,
  LILogicNet, and BitLogic. Every artifact uses source revision `dd194c0`
  and training implementation hash
  `f5bef4c78c6540e5c783bfa8c1033e61dd89bf9a50e2bb52897aa0d8272ac444`.
- Mean best hardened validation accuracy is fixed random `[REPRODUCED]`
  87.477% +/- 0.183% (n=5), CoverageDLGN v3 `[OUR-FINAL]`
  87.873% +/- 0.271% (n=5), Mommen `[ADAPTED]`
  88.361% +/- 0.107% (n=3), LILogicNet `[ADAPTED]`
  89.606% +/- 0.118% (n=3), and BitLogic `[ADAPTED]`
  90.450% +/- 0.277% (n=3). Values after `+/-` are sample standard
  deviations.
- V3 improves its paired fixed-random validation baseline by 0.397
  percentage points with a 95% Student-t confidence interval of
  [+0.111, +0.683] pp. All five per-seed differences are positive.
- Mean training wall time is 14.97 minutes for random, 15.02 for V3, 24.04
  for Mommen, 70.84 for LILogicNet, and 88.94 for BitLogic. Peak allocated
  GPU memory is 0.099, 0.099, 0.498, 4.676, and 3.557 GiB, respectively.
  V3 preserves the random model's 768K trainable parameters and zero
  training-routing parameters; the adapted comparators use 1.536M/768K,
  3.84M/3.072M, and 3.84M/3.072M total/routing parameters.
- The validation source of truth is
  `summary/table1_fashion_final.json` and its CSV export with
  `test_set_used=false`. The held-out Fashion-MNIST test set remains locked
  until this validation summary is committed and then will be evaluated once
  using the guarded evaluator.

## July 26, 2026: Table 1 Fashion-MNIST held-out final evaluation

- Validation results and the checkpoint-selection state were frozen in commit
  `354ed1a` before test evaluation. The guarded evaluator preflighted all 19
  best-validation checkpoints, verified that no Fashion test metric existed,
  and distributed the exactly-once evaluation across both GPUs. All 19
  commands completed successfully with zero failures or missing artifacts.
- Mean hardened test accuracy is fixed random `[REPRODUCED]`
  86.308% +/- 0.186% (n=5), CoverageDLGN v3 `[OUR-FINAL]`
  87.102% +/- 0.357% (n=5), Mommen `[ADAPTED]`
  87.260% +/- 0.282% (n=3), LILogicNet `[ADAPTED]`
  88.437% +/- 0.159% (n=3), and BitLogic `[ADAPTED]`
  89.740% +/- 0.243% (n=3).
- CoverageDLGN v3 improves its paired fixed-random test baseline by 0.794
  percentage points. The 95% Student-t confidence interval is
  [+0.471, +1.117] pp, and all five per-seed differences are positive. The
  test gain is larger than the frozen validation gain of 0.397 pp.
- Mommen is 0.158 pp above V3 in the unpaired method means but uses twice
  the trainable parameters, 1.60 times the mean training time, and 5.02 times
  the peak allocated GPU memory. LILogicNet and BitLogic achieve higher raw
  accuracy but use five times the trainable parameters and approximately
  4.72/5.92 times the training time and 47.1/35.8 times the peak memory.
- `summary/table1_fashion_final.json` and its CSV export are now the final
  machine-readable sources of truth with `test_set_used=true`. The held-out
  Fashion-MNIST test set will not be queried again for this Table 1
  experiment.

## July 26, 2026: dense CIFAR-10 compression screen

- The first two-GPU launch began before the CIFAR-10 cache existed. Both
  workers therefore attempted to download the same official archive into the
  same path. They were stopped before model construction or training to avoid
  a corrupted shared cache. The two pre-training logs are preserved under
  `logs/failed/table2_screen_cifar10_duplicate_download_attempt1/`; the
  incomplete 8.8 MB archive was quarantined in the dataset cache with the
  suffix `.duplicate-download-partial`.
- CIFAR-10 was then downloaded exactly once through torchvision. The official
  archive checksum passed, extraction completed, and the cache was verified
  to contain 50,000 training and 10,000 test objects. Only dataset lengths
  were inspected; no model was evaluated on the held-out test set.
- The clean one-seed, 5K-step screen completed all 30 declared runs on two
  GPUs with zero failures in 1,880 seconds. At 128K gates, fixed random
  reached 49.780% best hardened validation accuracy and the best V3 candidate
  reached 54.300% (raw parameterization, candidate pool 4). The frozen
  advancement rule retained pool 4, swap fraction 0.50, and the raw
  incumbent.
- At 256K gates, fixed random reached 51.600%. The raw incumbent and novelty
  weights 0.5 and 2.0 tied at 55.940%, so all three advance under the
  predeclared second-place-tie rule. At 384K gates, fixed random reached
  52.280%, WARP-V3 reached 57.520%, and the raw incumbent and novelty-0.5
  variants tied at 57.420%; those three advance.
- Light parameterization was not competitive in this screen
  (44.24--44.58%). This is retained as a negative result. The screen source of
  truth is `summary/table2_cifar10_compression_screen.json`; all values are
  `[TRIED]` validation-selection measurements, not paper-final results.
- The next queue contains 36 paired 20K runs: random plus three advanced V3
  candidates at each of 128K, 256K, and 384K, each with seeds 0, 1, and 2.
  The held-out CIFAR-10 test set remains locked.

## July 26, 2026: dense CIFAR-10 compression selection

- The paired 20K selection completed all 36 declared runs on two GPUs with
  zero failures or skips in 5,020 seconds. At 128K gates, raw V3 with swap
  fraction 0.50 won at 54.853% +/- 0.234% mean best hardened validation
  accuracy, versus 50.000% +/- 0.390% for fixed random. Its paired mean gain
  is +4.853 percentage points.
- At 256K, the raw incumbent won at 57.513% +/- 0.609%, versus
  52.567% +/- 0.031% random, for a +4.947 pp paired gain. Novelty weights
  0.5, 1.0, and 2.0 generated exactly identical per-seed accuracies, showing
  that this weight did not alter the discrete selected topology in this cell.
  The deterministic tie-break therefore retained the incumbent.
- At 384K, the raw incumbent and novelty-0.5 again generated identical
  results, 58.980% +/- 0.485%. The incumbent was retained by the tie-break.
  WARP reached 57.433% +/- 0.546%, while fixed random reached
  54.400% +/- 0.242%; the selected incumbent's paired gain is +4.580 pp.
- These are `[TRIED]` validation-selection values, not held-out test results
  or paper-final accuracies. The source of truth is
  `summary/table2_cifar10_compression_selection.json`.
- The frozen winners advance to 108,000-step matched training with three
  paired seeds at each budget (18 runs total). Only after this full-effort
  validation stage will the predeclared compression noninferiority crossing
  be determined and extended from three to five seeds.

## July 26, 2026: dense CIFAR-10 compression full-effort validation

- All 18 frozen 108,000-step runs completed successfully with zero failures
  or skips. At 128K gates, fixed random reached 50.460% +/- 0.240% and
  CoverageDLGN reached 54.927% +/- 0.153% mean best hardened validation
  accuracy. The paired gain is +4.467 pp with a three-seed 95% Student-t
  interval of [+3.495, +5.438] pp.
- At 256K gates, random reached 53.073% +/- 0.341% and CoverageDLGN reached
  57.800% +/- 0.410%. The paired gain is +4.727 pp with 95% interval
  [+3.120, +6.333] pp.
- At 384K gates, random reached 54.920% +/- 0.251% and CoverageDLGN reached
  59.313% +/- 0.397%. The paired gain is +4.393 pp with 95% interval
  [+3.184, +5.603] pp.
  Exact per-seed values, standard deviations, costs, and timings are stored
  in `summary/table2_cifar10_compression_final3.json`.
- The predeclared reference is the existing five-seed 512K random validation
  mean of 55.084%, giving a 54.784% noninferiority threshold. The 128K
  CoverageDLGN mean of 54.927% exceeds that threshold by 0.143 pp and is the
  smallest evaluated budget to qualify. It is therefore the frozen
  compression crossing and will be extended from three to five paired seeds.
- The held-out CIFAR-10 test set remains locked. All values in this section
  are `[TRIED]` validation results.

## July 26, 2026: 128K compression-crossing extension

- The first extension attempt was interrupted after random seeds 3 and 4
  reached completed validation step 98,000 but before either produced a final
  `run_summary.json`. No V3 extension run had started. The incomplete
  artifacts were archived under
  `results/failed/table2_crossing_extension_interrupted_attempt1/`, and a
  compact failure record is preserved under
  `logs/failed/table2_crossing_extension_interrupted_attempt1/`.
- Exact resume equivalence had not been proven, so the partial checkpoints
  were not resumed or counted. Only the two incomplete random seeds were
  restarted from step zero; completed seeds 0--2 were not rerun. V3 seeds
  3--4 then ran for the first time. The clean recovery queue completed all
  four declared runs with zero failures or skips.
- Across five paired seeds, fixed random reaches
  `[REPRODUCED]` 50.760% +/- 0.446% and CoverageDLGN reaches
  `[OUR-FINAL]` 55.140% +/- 0.336% mean best hardened validation accuracy.
  The paired gain is +4.380 pp with 95% Student-t interval
  [+3.989, +4.771] pp; all five per-seed differences are positive.
- The frozen source of truth is
  `summary/table2_cifar10_compression_crossing_final.json` with
  `test_set_used=false`. The held-out test will be evaluated exactly once
  only after this validation result is committed.

## July 26, 2026: 128K compression-crossing held-out evaluation

- The five-seed validation state was frozen in commit `b4a1f5f` before test
  evaluation. The guarded evaluator verified all 10 checkpoints, rejected
  any pre-existing test output, and distributed the exactly-once evaluation
  across both GPUs. All commands completed successfully with zero failures
  or missing artifacts.
- Fixed random reaches `[REPRODUCED]` 49.748% +/- 0.141% hardened test
  accuracy and CoverageDLGN reaches `[OUR-FINAL]` 53.910% +/- 0.282%.
  CoverageDLGN's paired test gain is +4.162 pp with a 95% Student-t interval
  of [+3.759, +4.565] pp. All five per-seed test differences are positive.
- `summary/table2_cifar10_compression_crossing_final.json` and its CSV export
  are now the final machine-readable sources with `test_set_used=true`.
  This crossing's held-out test set will not be queried again.

## July 26, 2026: dense CIFAR-10 S comparator screen

- Two comparator-only model wrappers expose learnable routing on the exact
  existing 48K S and 512K M architectures. Tests verify that their gate
  counts, four-layer depths, widths, output temperatures, and three-threshold
  input encoding are unchanged. No fixed-topology random or CoverageDLGN V3
  implementation was modified.
- The reduced S screen contained only three one-seed, 5K runs. Mommen
  \(N_c=8\) reached 49.000% best hardened validation accuracy using 1.536M
  total trainable parameters and 0.768M routing parameters. \(N_c=16\)
  reached 48.580% using 2.304M/1.536M total/routing parameters. The frozen
  winner is \(N_c=8\).
- The single fixed LILogicNet Top-32, tau-30 matched-48K adaptation reached
  50.460%, using 3.840M total and 3.072M routing parameters. It required
  618 seconds and 3.95 GiB peak allocated GPU memory for the 5K screen,
  compared with 77 seconds and 0.48 GiB for Mommen \(N_c=8\).
- These values are `[TRIED]` validation screens only. The selected Mommen
  configuration and the sole LILogicNet configuration advance directly to
  three full seeds each; no three-candidate 20K comparator selection is run.
  This reduced search budget is explicit and does not change per-model
  parameter accounting. The held-out test remains locked.

## July 26, 2026: conditional seed policy for deep comparators

- Future expensive deep/L comparator adaptations start with one full seed,
  reported as `[TRIED, n=1]` without a variance estimate. They are extended
  to three seeds only if competitive with the central fixed-routing methods
  or required for a specific scientific claim.
- This change does not affect already completed or running S/M experiments.
  Central fixed-random and frozen CoverageDLGN v3 comparisons retain five
  seeds. The CoverageDLGN v3 mechanism is unchanged.

## July 26, 2026: CIFAR-10 M Mommen seed decision

- The exact-512K CIFAR-10 M Mommen adaptation starts with one full seed
  because its trainable routing and runtime scale substantially beyond S.
  It advances to three seeds only if competitive with the completed matched
  fixed-routing results or required for a specific paper claim.
- LILogicNet and BitLogic remain clearly labelled reported-only comparisons
  on CIFAR-10 M. This decision does not alter any model or CoverageDLGN v3.

## July 26, 2026: dense CIFAR-10 S comparator validation

- All six frozen full-schedule comparator runs completed successfully:
  three Mommen \(N_c=8\) seeds and three LILogicNet Top-32, tau-30 seeds.
  Every run used source revision `e4327c6` and training implementation hash
  `7904ccc39c63e188c4b403ee529a3d7634642a4fcd9153b4f480f2f8f5827ee5`.
- Mean best hardened validation accuracy is Mommen `[ADAPTED]`
  51.753% +/- 0.386% and LILogicNet `[ADAPTED]` 51.013% +/- 0.450%, with
  three seeds per method. Values after `+/-` are sample standard deviations.
- Mommen averaged 1,587 seconds per seed and peaked at 515,224,576 allocated
  GPU bytes. LILogicNet averaged 5,212 seconds per seed and peaked at
  4,239,894,016 bytes. Both deploy the same exact 48K rank-2 gate circuit,
  but use 0.768M and 3.072M training-only routing parameters respectively.
- An accidental Ctrl+C interrupted the first LILogic attempts at recorded
  steps 34,604 and 21,944. Their partial artifacts and logs are preserved
  under the `failed/table2_s_lilogic_interrupted_ctrl_c_attempt1` directories.
  Exact resume was not used because equivalence is unproven. The three
  completed Mommen runs were skipped, and only unfinished LILogic work was
  restarted.
- `summary/table2_s_comparator_final.json` and its CSV export are the frozen
  validation sources with `test_set_used=false`. The held-out test remains
  locked until this validation state is committed.

## July 26, 2026: dense CIFAR-10 S comparator held-out test

- Commit `6d63a4b` froze all six comparator validation checkpoints before
  test evaluation. Every checkpoint was then evaluated exactly once with
  zero failures.
- Held-out hardened test accuracy is Mommen `[ADAPTED]`
  50.950% +/- 0.244% and LILogicNet `[ADAPTED]` 50.743% +/- 0.574%, with
  three seeds per method. The existing exact-48K CoverageDLGN v3 result is
  53.116% (n=5), 2.166 and 2.373 percentage points higher respectively;
  these cross-method differences are not paired confidence intervals.
- `summary/table2_s_comparator_final.json` and its CSV export now have
  `test_set_used=true` and are the final machine-readable sources. These six
  comparator checkpoints will not be queried on the held-out test again.

## July 26, 2026: dense CIFAR-10 L feasibility

- The paper-faithful `DlgnCifar10Large` architecture was audited as five
  layers of 256K rank-2 gates, 1.28M gates total, five input bits per channel,
  and output temperature 100. The focused protocol suite now tests these
  properties explicitly.
- A timing-only fixed-random run completed 1K steps in 64 seconds of measured
  training time and peaked at 2,917,583,360 allocated GPU bytes. Its short-run
  accuracy is not eligible for any comparison table.
- L advances to a reduced one-seed 5K screen containing fixed random and five
  unchanged V3 configurations: incumbent, pool 4, swap 0.50, Light, and WARP.
  These vary only existing topology controls or gate parameterization; the
  frozen V3 mechanism is unchanged and the held-out test remains locked.

## July 27, 2026: dense CIFAR-10 M Mommen and L screen validation

- The conditional exact-512K Mommen \(N_c=8\) run completed its single
  predeclared full seed in 17,441 seconds, peaked at 5,280,565,760 allocated
  GPU bytes, and reached 55.000% best hardened validation accuracy.
- Existing exact-512K validation references are fixed random
  55.084% +/- 0.279% (n=5) and CoverageDLGN v3
  59.560% +/- 0.246% (n=5). Mommen is effectively level with random but
  4.560 pp below V3, so it is not promoted to three seeds.
- All six exact-1.28M L screen runs completed without failures. Raw random
  reached 53.040%; raw V3 incumbent 56.840%; pool-4 56.520%; raw swap-0.50
  58.780%; Light 36.020%; and V3+WARP 59.080%.
- Raw swap-0.50 advances as the primary V3 topology-only candidate. WARP
  advances only as a secondary matched pair that adds a random-topology+WARP
  control at 20K. The V3 algorithm is unchanged and the held-out test remains
  locked.
- Machine-readable sources are `summary/table2_m_mommen_final.json` and
  `summary/table2_l_screen.json`, both with `test_set_used=false`.

## July 27, 2026: dense CIFAR-10 M Mommen held-out test

- Commit `c4c1c4a` froze the one-seed Mommen M validation checkpoint before
  test evaluation, and commit `54e2140` froze the exactly-once guard.
- The checkpoint reached `[TRIED]` 54.420% hardened held-out test accuracy.
  This is a single-seed adapted result without a variance estimate. The
  existing exact-512K five-seed results are fixed random 54.028% and
  CoverageDLGN v3 58.284%.
- `summary/table2_m_mommen_final.json` and its CSV export now have
  `test_set_used=true`. This checkpoint will not be queried on test again.

## July 27, 2026: dense CIFAR-10 L paired 20K selection

- All 12 predeclared runs completed with zero failures and one identical
  training implementation: three seeds each for raw random, raw V3
  swap-0.50, matched random+WARP, and V3+WARP.
- Raw random reached 55.627% +/- 0.502% best hardened validation accuracy;
  raw V3 reached 60.907% +/- 0.310%. The paired gain is +5.280 pp with a
  95% Student-t interval of [+3.769, +6.791], and all three seed differences
  are positive.
- Matched random+WARP reached 54.207% +/- 0.133%; V3+WARP reached
  59.813% +/- 0.242%. Its paired gain is +5.607 pp with a 95% interval of
  [+4.769, +6.444], again positive for all seeds.
- The primary final is frozen as raw random versus raw V3 swap-0.50 because
  it isolates the CoverageDLGN contribution and raw V3 has higher absolute
  validation accuracy than V3+WARP. WARP remains a secondary complementary
  result; Light remains rejected.
- `summary/table2_l_selection.json` and its CSV export are the
  machine-readable selection sources with `test_set_used=false`. The
  held-out test remains locked.

## July 27, 2026: dense CIFAR-10 L five-seed full validation

- All ten frozen 108K runs completed with zero failures in 35,906 seconds of
  two-GPU queue wall time: five raw fixed-random seeds and five raw
  CoverageDLGN v3 swap-0.50 seeds. Every run used source revision `b73307a`
  and training implementation hash
  `7904ccc39c63e188c4b403ee529a3d7634642a4fcd9153b4f480f2f8f5827ee5`.
- Fixed random `[REPRODUCED]` reached 56.468% +/- 0.210% mean best hardened
  validation accuracy. CoverageDLGN v3 `[OUR-FINAL]` reached
  61.748% +/- 0.277%.
- Per-seed V3 gains are +5.440, +4.920, +4.980, +5.280, and +5.780 pp.
  The paired mean is +5.280 pp with a 95% Student-t interval of
  [+4.843, +5.717]; all five differences are positive.
- Both methods use the identical 1.28M deployed-gate and 20.48M trainable
  gate-logit budgets with no trainable routing parameters. Training averaged
  6,808 seconds per random seed and 6,793 seconds per V3 seed; peak allocated
  GPU memory was 2,917,583,360 bytes for both.
- `summary/table2_l_final.json` and its CSV export are the frozen validation
  sources with `test_set_used=false`. The held-out test remains locked until
  the single L Mommen run is complete and all validation is committed.

## July 28, 2026: dense CIFAR-10 L one-seed Mommen validation

- The exact-1.28M Mommen \(N_c=8\) comparator completed its single
  predeclared 108K seed with zero failures. It reached `[TRIED]` 54.340%
  best hardened validation accuracy; its final-step value was 53.680%.
- The run required 48,206 seconds (13.39 hours) and peaked at
  12,781,914,624 allocated GPU bytes. It used 40.96M total trainable
  parameters, including 20.48M training-only routing parameters, while
  deploying the same 1.28M-gate circuit budget.
- Mommen is 2.128 pp below the five-seed random validation mean and
  7.408 pp below CoverageDLGN v3. Per the conditional policy it is not
  promoted beyond one seed.
- `summary/table2_l_mommen_final.json` and its CSV export are the
  machine-readable sources with `test_set_used=false`. The held-out test
  remains locked pending a separate exactly-once final evaluation decision.

## July 28, 2026: dense CIFAR-10 L held-out test

- Commits `acc360d` and `7eaea55` froze the central and Mommen validation
  checkpoints, and commit `435dd00` froze the exactly-once guard.
- All 11 checkpoints completed held-out evaluation with zero failures.
  Fixed random `[REPRODUCED]` reached 55.960% +/- 0.251% test accuracy;
  CoverageDLGN v3 `[OUR-FINAL]` reached 61.020% +/- 0.336%.
- Per-seed V3 test gains are +4.720, +5.720, +5.170, +4.900, and +4.790 pp.
  Their mean is +5.060 pp with a 95% Student-t interval of
  [+4.555, +5.565], and all five are positive.
- The one-seed Mommen adaptation reached `[TRIED]` 54.340% test accuracy.
  `summary/table2_l_final.json` and `summary/table2_l_mommen_final.json`
  now have `test_set_used=true`; none of these checkpoints will be tested
  again.

## July 28, 2026: experiment-order decision

- Convolutional experiments are paused by user direction. No convolutional
  S, M, or L run will be launched in the current phase.
- The next phase is dense CIFAR-100, beginning with the exact S protocol and
  advancing to M/L under the frozen promotion and comparator policies.
- V4 will be revisited after dense CIFAR-100 S/M/L is complete.

## July 28, 2026: dense CIFAR-100 protocol freeze

- The BitLogic Table 6 common protocol was audited before implementation.
  Dense CIFAR-100 uses a two-layer width ladder of 4K, 16K, and 64K gates
  per layer (8K, 32K, and 128K total), rank-2 LUTs for the matched
  DiffLogic/CoverageDLGN coordinate, three linear thermometer thresholds,
  a 90/10 train/validation split, reflect-padded random crop plus horizontal
  flip, AdamW at 0.01, batch size 128, and 100 epochs (35,100 iterations).
- The initial S architecture regression test failed before any training:
  9,216 encoded inputs exceed its 8,000 first-layer input slots. TorchLogix
  previously required every encoded input to occur at least once. Changing
  the widths would have violated the paper protocol, so the exact two-by-4K
  architecture was retained.
- An opt-in partial-coverage mode now selects a uniform maximal subset of
  first-layer inputs when slots are insufficient. It is enabled only by the
  new CIFAR-100 common-protocol models; all existing models retain the prior
  full-coverage assertion. Both paired random and V3 runs use the same
  architecture and slot constraint. V3's topology construction, scoring,
  and refinement are unchanged.
- A subsequent pre-run comparator audit found the same legacy assertion in
  bounded learnable routing. The CIFAR-100 models now pass the opt-in through
  fixed and learnable connections alike. Candidate tensors retain their full
  Top-K coverage, while deployed routing remains subject to the same 8,000
  S-layer slots as fixed routing.
- The S/M/L architecture and budget checks, partial-coverage behavior, and
  existing topology/LGN tests pass: 95 tests total. The frozen machine-readable
  protocol is `protocols/table4_dense_cifar100.json`.

## July 28, 2026: dense CIFAR-100 S screen

- The official CIFAR-100 archive initially transferred at about 0.1 MB/s.
  The single transfer was interrupted at 49,774,592 bytes and retained as
  `/tmp/torchlogix-datasets/data-cifar-100/cifar-100-python.interrupted.tar.gz`.
  Eight parallel byte ranges were assembled and accepted only after matching
  the official MD5 `eb9058c3a382ffc7106e4002c42a8d85`.
- Both 10-step CUDA smokes passed. They recovered fixed thresholds
  `{0.25, 0.5, 0.75}`, two 4K layers, 8K total gates, 128K trainable LUT
  parameters, and hardened validation for random and frozen V3.
- All eight one-seed 5K screen runs completed with identical training
  implementation hash
  `51efea4e869a3b2ce3861afb9b8d5011ee6770f4ade3166afe89a815363e7f67`.
  Random reached 9.420% best hardened validation accuracy. The best V3
  control was the existing swap fraction 0.125 at 8.140%, a preliminary
  -1.280 pp screen gap. This is `[TRIED]`, not a final or held-out result.
- Per the frozen protocol, random and `swap0125` advance to the paired
  three-seed 20K confirmation. No held-out test has been used.

## July 28, 2026: dense CIFAR-100 S selection and topology diagnosis

- All six paired 20K runs completed. Fixed random reached
  `[TRIED]` 8.780% +/- 0.381% best hardened validation accuracy; frozen V3
  `swap0125` reached `[TRIED]` 7.940% +/- 0.302%.
- Per-seed V3 gains were -1.100, -0.680, and -0.740 pp. The paired mean is
  -0.840 pp with a 95% Student-t interval of [-1.404, -0.276]. The frozen
  S-to-M promotion condition therefore failed. No full S or held-out-test
  run is authorized by this result.
- The topology report explains why compact S is a distinct regime. Its
  8,000 first-layer slots are fewer than 9,216 encoded inputs. Random uses
  all 8,000 slots and covers 99.740% of raw RGB spatial sources. Frozen V3's
  truncated first semantic-butterfly stage covers 7,905 encoded bits and
  98.828% of raw sources; 91.650% of its pairs retain the same channel and
  threshold because the 4K output prefix ends before later semantic-axis
  stages. Random's corresponding fraction is about 34%.
- M and L do not have this undersubscribed first layer: they provide 32K and
  128K first-layer slots for the same 9,216 inputs. In order to complete the
  user-directed dense study without changing V3, M receives a separately
  labelled one-seed 5K exploratory screen. This is a documented diagnostic
  exception, not an assertion that S met the promotion condition. M advances
  further only if its own paired signal is positive.

## July 28, 2026: dense CIFAR-100 M diagnostic screen

- All eight one-seed 5K diagnostic runs completed without touching held-out
  test. Random reached 9.880% best hardened validation accuracy.
- Every V3 control except `swap0125` matched or exceeded random. The selected
  existing control is `swap0500` at 10.380%, a preliminary +0.500 pp.
- The positive M-specific signal authorizes a paired three-seed 20K
  confirmation. This does not retroactively change the failed S promotion
  decision and does not change the V3 mechanism.

## July 28, 2026: dense CIFAR-100 M selection and L kill

- All six paired M 20K runs completed. Fixed random reached
  `[TRIED]` 10.060% +/- 0.220% best hardened validation accuracy; frozen V3
  `swap0500` reached `[TRIED]` 9.707% +/- 0.076%.
- Per-seed V3 gains were -0.100, -0.660, and -0.300 pp. The paired mean is
  -0.353 pp with a 95% Student-t interval of [-1.058, +0.352]. All three
  point estimates are negative, and the positive-mean promotion condition
  failed.
- Dense CIFAR-100 L is therefore stopped before screen, full training, or
  held-out test. S and M remain validation-only `[TRIED]` results; L retains
  reported-only paper values. No CIFAR-100 held-out test was used.
- This closes dense CIFAR-100 under the frozen promotion/kill protocol.
  Convolutional V4 may now be revisited as a separate phase.

## July 28, 2026: dense CIFAR-100 deep-architecture extension

- At the user's direction, dense CIFAR-100 was reopened for the deeper
  architectures used by the scalability and multilinear papers:
  six layers by 64K gates (384K total, three thresholds, temperature 10) and
  six layers by 256K gates (1.536M total, 31 thresholds, temperature 1).
  These are separate paper coordinates from the compact BitLogic S/M/L
  ladder. The frozen V3 implementation was not modified.
- The machine-readable protocol is
  `protocols/table4_dense_cifar100_deep.json`. It predeclares a seed-0 5K
  screen of fixed random and existing V3 swap fractions 0.125, 0.25, and
  0.5; a positive winner advances to paired seeds 0--2 at 20K; a positive
  paired mean advances to the paper-length schedule. Held-out test remains
  locked until the full winner is frozen.
- Both CUDA smoke tests passed. The 6-by-64K model has 384,000 gates,
  6,144,000 trainable LUT parameters, and about 0.795 GiB peak allocated GPU
  memory. The 6-by-256K model has 1,536,000 gates, 24,576,000 trainable LUT
  parameters, and about 14.18 GiB peak allocated GPU memory.
- The first two-GPU screen supervisor was externally terminated while
  6-by-64K `swap0500` and 6-by-256K random were incomplete. Their artifacts
  and logs were moved intact to
  `results/failed/table4_cifar100_deep_screen_supervisor_sigterm_attempt1/`.
  The idempotent restart skipped the three already complete runs and
  completed the remaining five. No incomplete result entered a summary.
- The selection generator initially raised a `KeyError` before launching
  training because it read the screen field `candidate` as `label`. The
  report-only generator was corrected and rerun. This created no checkpoint
  or accuracy result and did not alter training or V3.

## July 28, 2026: dense CIFAR-100 deep results

- In the 6-by-64K 5K screen, random reached `[TRIED]` 19.340% hardened
  validation accuracy. V3 swap fractions 0.125, 0.25, and 0.5 reached
  19.980%, 19.700%, and 19.580%; `swap0125` advanced at +0.640 pp.
- Its paired three-seed 20K confirmation reached `[TRIED]`
  21.580% +/- 0.100% for V3 versus 21.040% +/- 0.420% for random. Per-seed
  gains were +1.060, +0.540, and +0.020 pp; the mean +0.540 pp satisfied the
  frozen promotion rule.
- The exact six-by-64K paper-length schedule used 100 epochs, a fixed 80/20
  train/validation split, batch size 100, Adam at 0.01, three thresholds, no
  augmentation, and three paired seeds. Full validation was
  `[OUR-FINAL]` 21.577% +/- 0.067% for V3 versus
  `[REPRODUCED]` 20.943% +/- 0.311% for random, a paired +0.633 pp
  (95% CI [-0.193, +1.459]); all three validation gains were positive.
- After freezing those checkpoints, each was evaluated exactly once on the
  held-out test set. V3 reached `[OUR-FINAL]` 21.010% +/- 0.131% versus
  `[REPRODUCED]` 20.677% +/- 0.522% for random. Per-seed gains were
  -0.200, +0.630, and +0.570 pp, for +0.333 pp paired mean with 95% CI
  [-0.816, +1.483]. The positive mean is not statistically conclusive at
  three seeds. The paper reports 22.54% +/- 0.26% for its random baseline.
- For 6-by-64K, random topology construction averaged about 2.07 seconds and
  V3 `swap0125` about 37.46 seconds; peak allocated GPU memory was identical.
  Each full training run took about 11.4 minutes excluding topology.
- In the 6-by-256K 5K screen, random reached `[TRIED]` 11.680%. V3 swap
  fractions 0.125, 0.25, and 0.5 reached 10.320%, 10.180%, and 11.060%.
  Since every frozen V3 control was negative, the branch stopped before
  multi-seed confirmation, paper-length training, or held-out test. The
  screen is not directly comparable to the papers' 100K-step reported
  Soft-Mix/CovJac results.
- For 6-by-256K, topology construction ranged from 151.47 seconds at
  `swap0125` to 554.86 seconds at `swap0500`; fixed random took 34.60
  seconds. Each 5K training run took about 31.7 minutes excluding topology.
  All learning curves, checkpoints, manifests, summaries, and stopped/failed
  attempts are retained.

## July 28, 2026: fixed-384K CIFAR-100 depth ablation

- Three controlled architectures were added without changing V3:
  3-by-128K, 12-by-32K, and 24-by-16K. Each has exactly 384,000 rank-2
  gates, 6,144,000 trainable LUT parameters, three input thresholds,
  GroupSum temperature 10, batch size 100, Adam at 0.01, no augmentation,
  and the fixed 90/10 split with split seed 2027.
- The frozen protocol is
  `protocols/table4_dense_cifar100_depth384k.json`. It predeclares paired
  seed-0 20K pilots using random versus unchanged V3 pool 8, swap fraction
  0.125, and novelty weight 1.0. Only a gain of at least +1.0 pp authorizes
  confirmation seeds 1 and 2. Held-out test access is prohibited.
- Architecture and CUDA smokes passed. All six 20K runs completed with zero
  queue failures. No prior completed experiment was rerun.
- The shallow/wide 3-by-128K control reached `[TRIED]` 21.080% best hardened
  validation accuracy for random and `[TRIED]` 21.860% for V3, a positive
  +0.780 pp. It did not meet the predeclared +1 pp confirmation threshold.
  Its final-step hard accuracies were 20.000% and 21.860%, respectively.
- Both deeper coordinates failed to optimize. At 12-by-32K, random and V3
  both peaked at 1.180%; final hard/relaxed accuracy was 1.060%/0.980% for
  random and 0.780%/1.380% for V3. At 24-by-16K, both peaked at 1.200%;
  final hard/relaxed accuracy was 1.100%/0.740% for random and
  1.020%/0.740% for V3. Losses remained finite, so these are optimization
  failures rather than crashes or numerical divergence.
- Topology ancestry had already saturated before it could supply a useful
  distinction. At depth 12, final mean raw-source ancestry was 2,262.25 for
  random and 2,288.53 for V3 out of 3,072 sources; global source coverage
  was complete. At depth 24, both methods gave every final gate all 3,072
  sources and cross-gate source-ancestry Jaccard 1.0.
- Offline topology construction was 3.34 versus 47.98 seconds at depth 3,
  1.12 versus 33.79 seconds at depth 12, and 0.56 versus 42.43 seconds at
  depth 24 for random versus V3. Peak GPU allocation remained matched
  within each pair (0.890, 0.750, and 0.723 GiB).
- No architecture crossed the promotion threshold. Consequently, no
  confirmation seed, full schedule, or held-out-test evaluation was run.
  The machine-readable result is
  `summary/table4_cifar100_depth384k_pilot.json`.

## July 29, 2026: class-conditional coverage-head diagnosis

- Frozen CIFAR-10 S/L and CIFAR-100 6-by-64K checkpoints were analyzed
  offline without using weights, labels, or held-out data. CIFAR-100 V3
  already gave every class complete raw-source coverage, but seed-0 mean
  per-class source-usage CV was 0.24650 versus 0.05289 for CIFAR-10 L V3.
  Within-class ancestry Jaccard was 0.01042 versus 0.00502.
- This rejected a simple additional-coverage objective. A separate
  `class_conditional_coverage` final-layer strategy was implemented instead.
  It starts from the exact V3 final-layer base and uses strictly improving
  cross-class, degree-preserving two-edge swaps to favor sources underused by
  each affected class.
- Frozen V3 and V4 remain separate unchanged strategies. With no classifier
  override, model-level tests recover identical V3 indices and initialization.
  The new strategy changes no backbone layer, spatial coordinate, gate,
  parameter, routing-bit, or inference-cost budget.
- Seed-0 offline/CUDA checks changed 15,088 of 64,000 classifier gates, kept
  the exact predecessor-degree vector, and reduced mean source-usage CV from
  0.24650 to 0.22732. Circuit export and functional equivalence passed.
- The machine-readable protocol
  `protocols/table4_cifar100_class_head.json` froze one setting before
  accuracy training: V3 pool 8, V3 swap 0.125, novelty 1.0, and a maximum
  class-head change fraction of 0.25. Promotion required both +2 pp over
  random and +1 pp over V3 across paired seeds 0--2 at 20K.
- To honor the no-rerun policy, exact completed random and V3 20K artifacts
  were reused. Only the three missing class-head arms were trained. They
  completed with zero failures; no held-out test was accessed.
- Random reached `[TRIED]` 21.040% +/- 0.420%, frozen V3 reached
  `[TRIED]` 21.580% +/- 0.100%, and V3 plus the class head reached
  `[TRIED]` 21.593% +/- 0.133% best hardened validation accuracy.
  Head-minus-random was +0.553 pp (95% CI [-0.565, +1.671]); head-minus-V3
  was +0.013 pp (95% CI [-0.385, +0.412]).
- Across three seeds, the head reduced mean class source-usage CV from
  0.25655 for V3 to 0.23475 and slightly reduced within-class Jaccard from
  0.01051 to 0.01042. The intended topology effect therefore occurred, but
  it did not translate into accuracy.
- Mean topology construction was 2.07 seconds for random, 37.47 seconds for
  V3, and 49.50 seconds for V3 plus the head. All three used 0.795 GiB peak
  GPU allocation and identical deployment costs.
- Both promotion thresholds failed. The full schedule, held-out test,
  CIFAR-10/Fashion transfer, and convolutional V4 combination were stopped
  by the predeclared rule. Details are in
  `CLASS_CONDITIONAL_HEAD.md` and
  `summary/table4_cifar100_class_head.json`.

## July 29, 2026: CIFAR-10 compression held-out completion

- Before evaluation, all 12 frozen 256K/384K random/V3 checkpoint directories
  were audited: every checkpoint existed and none contained
  `test_metrics.json`.
- `evaluate_table2_compression_remaining.py` assigned the checkpoints across
  both GPUs and refuses to overwrite any existing held-out result. All 12
  evaluations completed with zero failures. No training was rerun.
- At 256K gates, V3 reached `[OUR-FINAL]` 56.903% +/- 0.134% versus
  `[REPRODUCED]` 52.253% +/- 0.058% for random. Every paired gain was
  positive; the mean was +4.650 pp with 95% Student-t CI
  [+4.174, +5.126].
- At 384K gates, V3 reached `[OUR-FINAL]` 58.143% +/- 0.153% versus
  `[REPRODUCED]` 53.657% +/- 0.328% for random. Every paired gain was
  positive; the mean was +4.487 pp with 95% Student-t CI
  [+3.515, +5.458].
- These checkpoints are closed to additional test queries. The immutable
  summary is `summary/table2_cifar10_compression_remaining_test.json`.

## July 29, 2026: CIFAR-100 baseline reproduction audit

- `audit_cifar100_baseline.py` verified the six completed random/V3 runs
  against the scalability paper: dataset, no augmentation, 20% validation,
  exact 0.25/0.50/0.75 thresholds, 6-by-64K architecture, temperature 10,
  Adam at 0.01, batch 100, 100 epochs, and three seeds all match.
- The 1.863 pp gap between the local random mean (20.677%) and the paper
  mean (22.54%) cannot be treated as an exact numerical reproduction gap.
  Canonical difflogic random routing uses two Torch `randperm` calls, while
  the paired local study uses an independent NumPy topology seed. The latter
  is necessary for matched random/V3 weight initialization but produces a
  different fixed graph and RNG stream.
- The paper's validation split seed and final-versus-validation-selected
  checkpoint policy are also unresolved. No checkpoint was retrained or
  evaluated on test during this audit. Details are in
  `CIFAR100_BASELINE_AUDIT.md` and
  `summary/cifar100_baseline_audit.json`.

## July 29, 2026: frozen V3 component ablation on CIFAR-10 M

- The machine-readable protocol
  `protocols/cifar10_medium_v3_components.json` decomposes the unchanged V3
  into fixed random, balanced butterfly fan-out, semantic first layer with
  swaps disabled, and full frozen V3.
- Existing three-seed 20K random and full-V3 controls were reused. Only the
  six missing balanced-backbone and semantic/no-swap arms were trained.
  All six completed successfully on two GPUs; no held-out test was accessed.
- The first task-aware config-generation attempt failed before training
  because only the original seed-0 V3 pilot config file exists. The generator
  was corrected to clone that frozen template and override both `seed` and
  `topology_seed`. This failure created no result directory.
- The first component-summary attempt failed because old pilot controls
  predate the `run_summary.json` cost field. The summarizer was corrected to
  recover the analytically fixed medium cost (verified against all six new
  arms) and mark recovered rows. No accuracy or checkpoint changed.
- Random reached 54.820% +/- 0.530%. Balanced butterfly reached
  58.980% +/- 0.548%, a paired +4.160 pp with 95% CI
  [+3.988, +4.332].
- Semantic-first/no-swaps reached 59.253% +/- 0.153%, only +0.273 pp over
  balanced butterfly (95% CI [-0.780, +1.326]).
- Full V3 reached 59.293% +/- 0.214%. Its ancestry-swap increment over the
  semantic/no-swap arm was +0.040 pp (95% CI [-0.434, +0.514]), while the
  complete V3 gain over random remained +4.473 pp
  (95% CI [+3.624, +5.323]).
- The dominant measured mechanism at this coordinate is therefore balanced
  fan-out. Semantic pairing and ancestry swaps remain parts of the frozen V3
  algorithm but do not have independently significant incremental effects in
  this three-seed pilot.

## July 29, 2026: one-shot task-aware rewiring negative result

- A separate optional method was implemented in `torchlogix/task_aware.py`;
  frozen `semantic_balanced_hybrid` V3 in `topology.py` was not edited.
  The feature is disabled by default.
- On the ordinary step-10K training batch, the method records
  class-conditional absolute activation-gradient signatures, performs
  strictly improving degree-preserving two-edge swaps once, discards the
  calibration state, and resumes training. It adds no optimizer step,
  trainable routing parameter, deployed routing bit, gate, or LUT parameter.
- Focused tests passed, followed by a 10-step CUDA smoke. The three-seed
  CIFAR-10 M protocol is
  `protocols/cifar10_medium_task_aware.json`; promotion required both +2 pp
  over random and +1 pp over V3 at 20K.
- The first post-run export-equivalence test used a full `Dlgn` fixture whose
  fixed binarizer emits floats; export-mode `LogicDense` requires Boolean
  inputs, so the fixture failed before testing rewired routing. It was
  replaced by the direct-`LogicDense` Boolean fixture used by the circuit
  suite. Rewired model/circuit equivalence then passed. This test-only failure
  did not touch any experiment artifact.
- Existing random and V3 controls were reused. Only three task-aware arms
  were trained. Pre-event layer-index hashes exactly matched the old frozen
  V3 checkpoints for all seeds.
- The event changed 57,818, 57,582, and 57,722 of 512K gates and took
  3.23, 3.22, and 3.09 seconds of swap computation. Exact predecessor
  degrees were preserved.
- Task-aware V3 reached `[TRIED]` 59.093% +/- 0.234%. This is +4.273 pp over
  random (95% CI [+3.421, +5.126]) but -0.200 pp versus V3
  (95% CI [-0.679, +0.279]); all three paired V3 differences were negative.
- The method failed the +1 pp-over-V3 gate. No full schedule, held-out test,
  dataset transfer, or convolutional run is authorized. Details are in
  `TASK_AWARE_REWIRING.md` and
  `summary/cifar10_medium_task_aware.json`.

## July 29, 2026: convolutional V4 components and channel-spatial negative result

- Frozen dense V3 and convolutional V4 were not modified. The new
  `semantic_channel_spatial_hybrid` strategy is a separate candidate that
  preserves V4 channel pairs and spatial coordinates but forces each
  bottom-level LUT to mix the selected channel pair.
- The original adapter draft balanced spatial offsets. Before training, the
  project specification was re-audited and its requirement to leave spatial
  receptive-field indexing unchanged was found. The implementation was
  corrected before any adapter pilot completed; four full-architecture spatial
  hashes match V4 exactly.
- An initial explicit classifier-override attempt exposed a second protocol
  issue. The override gave dense routing an independent topology RNG, changing
  later dense parameter initialization relative to historical V4. Two
  completed no-swap runs and two partial runs were excluded and preserved
  under
  `results/failed/cifar10_conv_small_explicit_classifier_rng_attempt1`.
- Corrected configs use the historical component-wide selector. A full-model
  regression test proves bit-identical trainable parameters, dense classifier
  indices, V4 channel pairs, and spatial coordinates for each paired seed.
- All six corrected runs completed with zero queue failures. Existing
  historical random and V4 controls were reused; neither was retrained and the
  held-out test set was not accessed.
- Balanced channels without swaps reached 58.013%, +1.340 pp over random
  (95% CI [-0.988, +3.668]) and +0.827 pp over V4
  (95% CI [-2.394, +4.047]). The positive direction is inconclusive.
- Forced channel-spatial leaf pairing reached 57.033%, +0.360 pp over random
  (95% CI [-0.501, +1.221]) and -0.153 pp versus V4
  (95% CI [-2.714, +2.407]).
- The adapter failed both +2 pp-over-random and +1 pp-over-V4 gates. No
  CIFAR-10 M run or held-out evaluation was launched.

## July 30, 2026: unified semantic degree-balanced candidate and stop decision

- Commit `0d22b8d` captured the full frozen V3/V4 milestone, corrected
  protocols, negative adapters, completed results, and failure history before
  the new candidate was implemented.
- `analyze_cifar10_conv_no_swap.py` loaded the exact three-seed S
  checkpoints. V4 preserved the complete fan-out vector and spatial indices
  but changed 8.33%, 20.31%, 10.68%, and 2.08% of output pairs by layer.
  It reduced duplicate pairs and increased span, while the maximum mean change
  in predecessor Jaccard was only 0.00246 and raw ancestry size was
  effectively unchanged.
- The earlier “round-robin” description was corrected: the successful
  no-swap base is an affine-ordered balanced butterfly. Its hardened
  validation curve overtook V4 at 12K and ended +1.173 pp above V4 at 20K
  when averaged over the original three seeds.
- A separate `semantic_degree_balanced` U1 strategy was implemented. Dense
  and convolutional forms share deterministic semantic/affine butterfly
  pairing, preserve the declared base degree schedule, do not force
  convolutional bottom-level leaf pairs, and contain no ancestry-swap call.
  Legacy swap/candidate controls cannot change U1.
- Frozen `semantic_balanced_hybrid` V3 and `semantic_channel_hybrid` V4 code
  paths were not edited. A full-model regression test proves U1 is bitwise
  identical to historical convolutional V4/no-swaps for a matched seed,
  including trainable parameters, convolutional indices, dense indices, and
  spatial coordinates.
- The final full TorchLogix suite passed with 3,367 tests passed, 3,038
  skipped, and one pre-existing warning in 244.56 seconds.
- The five-seed S protocol fixed a strict promotion rule before training:
  at least +1.0 pp paired mean over random and positive gains on at least four
  of five seeds. Existing seeds 0-2 were reused under proven equivalence.
- Six genuinely missing seed-3/4 random, frozen-V4, and U1 runs completed on
  both GPUs in 51.5 minutes with zero failures and no held-out test access.
- Per-seed random accuracies were [56.10, 57.40, 56.52, 56.88, 57.42]%;
  V4 was [57.60, 57.80, 56.16, 57.12, 58.56]%; U1 was
  [57.92, 57.66, 58.46, 57.56, 56.52]%.
- U1 reached 57.624% versus 56.864% random: +0.760 pp paired mean,
  95% CI [-0.700, +2.220], positive on four of five seeds. The consistency
  gate passed, but the +1.0 pp mean gate failed. U1 exceeded V4 by only
  +0.176 pp on average.
- On new seeds 3/4, topology construction averaged 0.165 seconds for U1,
  0.414 seconds for V4, and 0.176 seconds for random. Wall time, peak GPU
  allocation, gate/parameter count, and deployed routing bits were equal
  within instrumentation precision.
- The locked stop rule was applied. No convolutional CIFAR-10 M, L,
  CIFAR-100 transfer, full schedule, held-out test, or extended deployment
  study was launched for U1. This avoids post-hoc escalation of a candidate
  that did not meet its declared effect-size threshold.
- Detailed evidence is in `UNIFIED_DEGREE_BALANCED.md`,
  `summary/cifar10_conv_small_no_swap_diagnostics.json`, and
  `summary/cifar10_conv_small_unified_five_seed.json`.

## July 31, 2026: nine-channel LogicTreeNet-M long-run launch preparation

- A single-seed long run was frozen for the paper-faithful nine-channel
  `ClgnCifar10PaperMedium` architecture and unchanged Legacy V4
  `semantic_channel_hybrid` topology. V3 and V4 source code were not edited.
- The budget is 350,000 updates, or 997.15 effective epochs over the 45,000
  training examples at batch size 128. This is the largest multiple of the
  paper's 2,000-step validation interval below the approximately 1,000-epoch
  extent of its long-training plots.
- The existing nine-channel pilot's training-only crop/flip augmentation is
  retained and declared as an adaptation because the original paper does not
  specify augmentation. The held-out test set remains locked until the best
  hardened-validation checkpoint is frozen.
- No earlier 5K checkpoint is resumed: the training harness does not preserve
  optimizer, data-loader, and RNG states needed to prove exact continuation.
  The long run will start cleanly from step zero.
- Launch was attempted only through CUDA availability preflight. It produced
  no output directory or training update because `nvidia-smi` could not
  communicate with the driver and PyTorch 2.9.0+cu130 reported zero CUDA
  devices. The frozen configuration is
  `configs/full_conv_cifar10_paper_medium_legacy_v4_seed0.json`; protocol and
  blocker details are in `protocols/cifar10_paper_medium_long_v4.json`.
- The first focused architecture-test command was invoked from the parent
  repository and failed collection because `experiments` was not on the
  import path. Re-running unchanged from `repos/torchlogix` passed the two
  selected nine-channel architecture/topology tests; no source fix was needed.
- GPU access was restored at 16:10 UTC. Physical GPU 0 had 97,247 MiB free
  and was selected for the long run; GPU 1 remained occupied by an unrelated
  VLLM process and was left untouched.

## August 1, 2026: LogicTreeNet-M matched 200K control decision

- The one-seed nine-channel Legacy V4 run showed a hardened-validation
  plateau after its early maximum. The global best through 184K was 71.220%
  at 68K; the 100K--184K hardened values had not exceeded 71.080%.
- To prioritize a direct causal comparison, the declared run length was
  revised from 350K to 200K for both Legacy V4 and an otherwise identical
  original fixed-random control. The V4 source and topology are unchanged.
- V4 will be interrupted only after the completed 200K validation/checkpoint
  write is visible. Exact continuation will not be claimed because optimizer,
  data-loader, and RNG states are not stored in checkpoints.
- The paired protocol is
  `protocols/cifar10_paper_medium_200k_paired.json`; the random-control config
  is `configs/full_conv_cifar10_paper_medium_random_seed0_200k.json`.
- Both methods use seed 0, split seed 2027, topology seed 0, 45K/5K split,
  batch 128, AdamW at 0.02 with 0.002 weight decay, standard training-only
  crop/flip, raw rank-2 gates, residual probability 0.951, and evaluation
  every 2K updates. The topology is the only intended method difference.
- Every intermediate evaluation, threshold snapshot, topology diagnostic,
  environment record, training configuration, and best-validation checkpoint
  is retained. The held-out test set remains locked until both validation
  trajectories are complete and their checkpoints are frozen.
- Legacy V4 stopped as declared after its 200K metrics and thresholds had
  been written. It completed 100 validation evaluations. Its best hardened
  validation was 71.260% at 194K, best relaxed validation was 72.960% at
  146K, and final 200K values were 70.280% hardened and 72.900% relaxed.
  The controlled `SIGINT` means normal-final artifacts were not emitted, but
  the validation-selected `best_checkpoint.pt` and `best_model.pt` are intact.
  Full termination metadata is stored in the run's `early_stop.json`.
- The fixed-random control passed its 200K/configuration guard and launched
  automatically on physical GPU 0. Its output is
  `results/full_conv_cifar10_paper_medium_random_seed0_200k/`.

## August 3, 2026: LogicTreeNet-M paired completion and held-out test

- The fixed-random control completed all 200K updates normally in 91,690.33
  seconds, with all 100 validation evaluations, final checkpoint, run summary,
  topology diagnostics, thresholds, configuration, and environment retained.
- V4's best hardened validation was 71.260% at 194K versus 70.680% for random
  at 164K, a +0.580 pp selected-checkpoint gain. V4 led on 97/100 matched
  hardened evaluations with a +1.055 pp mean curve advantage and reached 70%
  at 36K rather than 66K.
- `freeze_cifar10_paper_medium_200k.py` verified the complete curves and
  matched training settings, then froze both best checkpoints by SHA-256
  before test access. The V4 digest begins `2fa5ccd955a7`; random begins
  `19d5a2fa50da`.
- The two checkpoints were each evaluated exactly once on all 10,000 held-out
  CIFAR-10 examples. V4 reached 69.960% hardened and 72.180% relaxed; random
  reached 69.570% hardened and 71.280% relaxed. The one-seed gains are +0.390
  and +0.900 pp, respectively. Neither checkpoint will be queried on this test
  set again.
- The reported LogicTreeNet-M value is 71.01% test. V4 is -1.05 pp and random
  -1.44 pp from that number. The controlled local comparison is not an exact
  paper-protocol reproduction because it uses a 45K/5K selection split and
  explicitly adapted training-only crop/flip augmentation.
- Offline topology construction totaled 6.182 seconds for V4 and 2.478 seconds
  for random, a +3.704-second V4 cost. The matched hardened GPU benchmark was
  74.573/74.891 ms per batch of 128 and 2,949,915,136/2,949,698,048 bytes peak
  allocation (V4/random). These tiny differences are timing/allocation noise;
  no runtime advantage is claimed.
- Gate/parameter/routing cost is identical: approximately 3.08M reported gate
  operations, 10,694,656 trainable parameters, zero training routing
  parameters, and 2,375,680 packed routing bytes. Random recorded
  15,691,860,480 bytes peak training allocation. V4's exact training peak is
  unavailable because controlled interruption bypassed the normal finalizer.
- Reproducible sources are
  `summary/cifar10_paper_medium_200k_freeze.json`,
  `summary/cifar10_paper_medium_200k_paired.json`, and
  `summary/cifar10_paper_medium_200k_curve.csv`.
- Two pre-test commands failed without accessing held-out data or writing a
  freeze manifest: the first used `repos/torchlogix/venv/bin/python` while
  already inside `repos/torchlogix`; the second strict guard exposed the
  expected saved `config`-path difference. The invocation and normalizer were
  corrected, and all subsequent freeze, test, benchmark, and summary commands
  succeeded.
- Python compilation of the four evaluation/summary scripts passed. The first
  focused pytest invocation reproduced the repository's known import-path
  collection error because `PYTHONPATH` was omitted. Re-running the unchanged
  suite with `PYTHONPATH=.` passed **103 tests** in 98.80 seconds.
- The first commit attempt was blocked by `git diff --check` because Python's
  default CSV writer emitted CRLF line endings, which Git reported as trailing
  whitespace. No commit was created. The summarizer now explicitly emits Unix
  newlines; the curve was regenerated before the successful commit attempt.

## August 4, 2026: convolutional evidence freeze and deployment accounting

- No accuracy training or dataset evaluation was launched. A new immutable
  evidence manifest SHA-256-froze configurations, metrics, checkpoints, test
  records, topology records, and inference records across 22 paper-faithful
  S/M and WARP-style run directories.
- The architecture audit confirmed that paper-faithful S and M use the same
  three-threshold-per-RGB, nine-Boolean-channel principle and four depth-3
  convolutional-stage pattern. Their scale parameters are `k_num=32`, tau 20
  and `k_num=256`, tau 40. WARP-style Medium is explicitly separate at two
  thresholds/six channels.
- The five-seed paper-S validation cohort and historical three-seed test
  cohort were separated. U1 has no held-out test result; no new test query was
  made. The dense CIFAR-10 S ledger was corrected from mislabeled validation
  values 49.692/53.116% to the frozen test values 49.056/52.358%.
- Frozen seed-0 S random, V4, and U1 checkpoints exported successfully on
  deterministic synthetic Boolean inputs. Hardened class, export-mode,
  Python Circuit, simplified Circuit, and generated C outputs agreed.
- The first random-S `gcc -O1` compile began at 21:36:45 UTC and was manually
  interrupted at 21:45:31 UTC after approximately 526 seconds. It had not
  failed, but the unbounded compile was not a practical measurement. The
  benchmark was changed to persist every stage and bound the compiler
  subprocess; no checkpoint or experimental artifact was modified.
- Bounded `gcc -O0`, 64-way bit-packed S compiles completed in 37.75, 39.75,
  and 44.72 seconds for random, V4, and U1. Mean batch-128 latencies were
  3.057, 3.076, and 3.095 ms over ten timed batches. These single-pass
  differences do not establish a speed claim.
- Exact nine-channel M and six-channel WARP-style M random/V4 pairs passed
  synthetic trace, Python-Circuit equivalence, and simplification. Fully
  unrolled M C compilation was not attempted after the S feasibility result.
- Detailed records are in `summary/convolutional_evidence_*`,
  `summary/convolutional_deployment.*`, and `summary/deployment/`.
- The final evidence-consistency audit passed all 16 checks. Focused
  convolutional, circuit, and experiment-protocol verification completed with
  **1,933 passed and 1,660 skipped in 228.11 seconds**.
# August 9, 2026: DATE second-round preflight

- Both RTX PRO 6000 Blackwell GPUs were visible to `nvidia-smi` and PyTorch
  2.9.0+cu130 in `repos/torchlogix/venv`; CUDA 13.0 reported two devices.
- The focused topology and protocol suite passed (117 tests).
- The 100-step dense U2 CUDA smoke test completed on GPU 0 and retained its
  complete artifacts in `results/smoke_second_u2_mnist_8k_seed0`.
- The first convolutional U2 smoke attempt stopped before training. Its 1x1
  semantic channel abstraction had channel structure but no spatial stage,
  and `_semantic_butterfly_indices` rejected the empty stage list. The failed
  directory is retained as `results/smoke_second_u2_conv_cifar10_s_seed0`.
  The scheduler was corrected to keep a channel level for singleton spatial
  dimensions and to ignore empty semantic axes; the retry uses a new output
  name rather than overwriting the failed attempt.

## August 9, 2026: architecture-matched BitLogic calibration correction

- The first MNIST 6 x 8K BitLogic run completed normally on CUDA but reached
  only 11.417% best hardened validation (9.733% final). A protocol audit found
  that the generated config used one distributive-threshold calibration batch,
  whereas the prior successful 48K BitLogic reproduction used 100 batches.
- Mommen and LILogic exact-comparator configs matched their prior reproduced
  recipes; the mismatch was isolated to MNIST BitLogic. Fashion-MNIST already
  used 100 calibration batches.
- The completed seed-0 artifacts were preserved under
  `results/failed/second_round_mnist_bitlogic_calibration1/` and classified as
  invalid-protocol evidence, not an accuracy result. The seed-1 attempt was
  interrupted after 240 seconds with a targeted `SIGINT`; its partial artifacts
  were preserved beside seed 0. The valid concurrent LILogic process was not
  interrupted, and the queue immediately launched Mommen seed 2 on CUDA.
- The generator and all three canonical MNIST BitLogic configs now explicitly
  use 100 calibration batches. A regression test locks this invariant. Corrected
  seeds 0 and 1 will be scheduled separately; seed 2 will use the corrected
  config when reached by the live queue. No failed directory is overwritten.

## August 9, 2026: U2 pre-pilot fan-out invariant correction

- A structural audit of the successful convolutional U2 smoke artifact found
  that a non-power-of-two cyclic stage was safe only when consumed in full.
  The first classifier reduction consumed half of a degree-two cyclic graph,
  leaving 12,288 of 40,960 inputs unused (`fanout_cv=0.775`). This violated
  U2's declared degree-balance rule, so the smoke is retained as pre-fix
  diagnostic evidence and is not an accuracy result.
- Only the separate, not-yet-promoted U2 strategy was corrected. Non-power-of-
  two regular stages now use deterministic perfect matchings at even widths
  (near-perfect matchings at odd widths), with coprime cyclic steps preserving
  the local-to-global scale schedule. Every partial-stage prefix uses an input
  at most once; degree spread is minimized before ancestry novelty, and odd-
  width byes rotate deterministically. Frozen V3, V4, and U1 code paths were
  not changed.
- The full topology suite passed 54 tests. An exhaustive audit over even widths
  4--64 and six partial/full stage sizes found maximum fan-out spread one and
  no violations; the nine-input/32-output odd-width channel case also reached
  the optimum spread of one. A fresh convolutional CUDA smoke will be required
  before U2 pilots launch.
- The regenerated post-balance dense/convolutional CUDA smoke queue is separate
  from the 20K pilot queue and rejects CPU configs. The full experiment-
  protocol suite passed 75 tests after adding this invariant and the live
  provenance ledger.
- Both post-balance smokes completed on GPU 1. The dense run used 1.536 seconds
  of training wall time and the convolutional run used 6.147 seconds (100
  updates each). All dense layers and all convolutional channel topologies had
  zero unused inputs. Most importantly, the convolutional classifier reductions
  now have `fanout_cv=0.0` and zero unused inputs at 40,960 -> 20,480 and
  20,480 -> 10,240, replacing the pre-fix values of 0.775/12,288 unused and
  0.707/5,112 unused. U2 is therefore structurally eligible for the 20K pilots.

## August 9, 2026: exact 6 x 8K comparator phase complete

- All 18 architecture-matched comparator runs completed on CUDA: three seeds
  each for Mommen, LILogicNet, and BitLogic on MNIST and Fashion-MNIST. The
  corrected queue rerun skipped 16 existing valid results and trained only the
  two missing MNIST BitLogic seeds; no completed run was repeated.
- Corrected MNIST BitLogic rank-4 reached 11.417 +/- 0.000% best hardened
  validation across all three seeds (112.75 minutes mean training wall time,
  3.527 GiB peak allocation). Together with Fashion-MNIST's
  10.867 +/- 0.000%, this is recorded as a reproduced-negative transfer of the
  two-layer method to the matched six-layer topology, not as missing data.
- The superseded one-calibration-batch seed and interrupted partial run remain
  isolated under `results/failed/second_round_mnist_bitlogic_calibration1/`;
  neither contributes to the reported aggregate.

## August 10, 2026: dense compression ladder complete

- All 48 planned MNIST/Fashion-MNIST compression runs completed on CUDA. The
  main queue reported 47 pending and skipped the previously completed MNIST
  4K random seed 0, so no finished run was repeated.
- Frozen V3 has a positive mean paired effect in every new budget cell and
  wins 22 of 24 paired runs. MNIST gains are +0.528, +0.494, +0.378, and
  +0.317 pp at 4K, 8K, 16K, and 32K total gates. Fashion-MNIST gains are
  +0.211, +0.689, +0.317, and +0.767 pp at 8K, 16K, 32K, and 64K.
- Fashion-MNIST 8K is individually significant at n=3 (95% CI
  [+0.085, +0.338]); all other new cell-wise intervals cross zero and are
  retained as directional compression evidence. The reused five-seed 48K
  validation references are positive on both datasets.
- Gate count, LUT parameter count, mean training wall time, and peak PyTorch
  allocation are matched between random and V3 at every budget. The method's
  only added cost remains offline deterministic topology construction.

## August 10, 2026: full LogicTreeNet-S random/V4 arms complete

- The exact nine-channel LogicTreeNet-S fixed-random and frozen legacy-V4
  seed-0 arms completed all 350K updates on separate CUDA GPUs. The held-out
  test set was not queried; selection uses best hardened validation only.
- Fixed random reached 58.680% best hard validation (58.500% final) in
  4.975 hours with 1.831 GiB peak allocation. Frozen V4 reached 59.860% best
  hard validation (58.460% final) in 4.957 hours with 1.831 GiB peak
  allocation, a +1.180 pp selected gain.
- Both arms have 83,552 learned LUT functions (including 71,680 classifier
  functions), 1.337M training parameters, zero training-only routing
  parameters, and identical deployed routing cost. V4's
  offline construction took 0.449 seconds versus 0.217 seconds for random.
- The full U1 arm started on the freed GPU and remains pending. It must finish
  before the three-arm full-schedule conclusion or any held-out test query.

## August 10, 2026: strengthened dense CIFAR-100 phase complete

- All 14 planned CUDA jobs completed without failures: six baseline-only 5K
  recipe screens, four missing members of the 3 x 128K random/V3 cohort, and
  two seed-0 same-384K allocation pairs. Previously completed seed-0 cohort
  runs were reused and not repeated.
- The short baseline screen selected temperature 20, learning rate 0.01, and
  no augmentation at 22.700% hardened validation. Screen results are kept
  separate from matched topology evidence.
- The original-recipe 3 x 128K cohort is now complete: random reached
  21.093 +/- 0.101% and frozen V3 reached 21.933 +/- 0.110% best hardened
  validation. The paired +0.840 pp gain has a 95% CI of
  [+0.351, +1.329] and all three seeds favor V3. Gate count (384K), LUT
  parameters (6.144M), training effort, memory, and deployed routing cost are
  matched. Mean training time is 5.35/5.34 minutes; mean offline topology
  construction is 3.36/71.03 seconds for random/V3.
- Moving the same gate budget toward the class head did not improve absolute
  accuracy. The 96K/96K/192K split produced 19.960% random and 20.620% V3;
  64K/64K/256K produced 20.380% random and 20.240% V3. These are retained as
  allocation ablations. No held-out test was queried.
- With GPU visibility reconfirmed, the freed GPU was assigned to the separate
  three-seed U2 pilot queue. The full U1 convolutional run continues on the
  other GPU.

## August 10, 2026: frozen U2 pilot complete and promoted

- All 15 U2 pilots completed on CUDA with no queue failures. The topology rule
  was frozen before these results and was not tuned between datasets.
- U2 versus random reached +0.756 pp on MNIST-8K (95% CI
  [+0.029, +1.483]), +0.600 pp on Fashion-16K, +3.293 pp on dense CIFAR-10 S
  (CI [+1.803, +4.784]), +0.100 pp on dense CIFAR-100 3 x 128K, and
  +2.173 pp on convolutional CIFAR-10 S (CI [+1.647, +2.700]). It won 3/3
  pairs in every coordinate except CIFAR-100 (2/3).
- On convolutional S, U2 reached 58.847 +/- 0.600% hardened validation and
  won 3/3 against frozen V4 (+1.660 pp) and U1 (+0.833 pp). Against the
  separate explicit controlled-random cohort it gained +1.707 pp with 3/3
  wins, although the n=3 interval crosses zero. Both random cohorts remain
  explicitly labeled.
- U2 was 0.740 pp below frozen V3 on CIFAR-100 with all three pairs negative;
  that coordinate did not promote. The statistically positive V3 3 x 128K
  result remains protected and unchanged.
- MNIST-8K, Fashion-16K, dense CIFAR-10 S, and convolutional CIFAR-10 S
  promoted to an unchanged U2 full-effort queue. Dense coordinates use three
  108K seeds. The expensive convolutional coordinate uses one 350K seed to
  match the already completed full-schedule random/V4/U1 resource cohort; its
  multi-seed evidence is the completed 20K pilot. The calibrated fallback was
  not authorized because convolutional U2 succeeded.

## August 10, 2026: dense final tests and full U1 complete

- The full 350K paper-faithful LogicTreeNet-S U1 arm completed on CUDA at
  59.880% best hardened validation, 58.580% final hardened validation,
  4.980 hours, and 1.831 GiB peak allocation. It gains +1.200 pp over the
  full matched random control and narrowly exceeds legacy V4 by +0.020 pp.
- The 33 dense validation selections were frozen before test access. Twenty-
  seven missing test records were evaluated once on CUDA; six existing dense
  CIFAR-10 random/V3 records were reused. No failures occurred.
- U2 held-out test gains versus random are +0.663 pp on MNIST-8K (95% CI
  [+0.015, +1.311]), +0.520 pp on Fashion-16K, and +3.183 pp on dense
  CIFAR-10 S; every paired seed is positive. Frozen V3 remains the stronger
  selected method on Fashion and dense CIFAR-10, while U2 ties V3 on MNIST.
- The strengthened CIFAR-100 3 x 128K V3 cohort reached 21.467 +/- 0.410%
  held-out test versus 20.923 +/- 0.352% random, a +0.543 pp paired gain with
  3/3 wins and an inconclusive n=3 CI [-0.141, +1.227].
- Machine-readable provenance is in
  `summary/second_round_final_validation_freeze.json` and
  `summary/second_round_final_dense.json`. At this log milestone the final
  convolutional U2 run was still training and convolutional test was locked;
  the later completion entry below supersedes that transient status.
- Some full-tree hashes differ because result documentation and queue scripts
  changed during sequential execution. The dedicated training-implementation
  hash is identical across all frozen dense runs, confirming that model,
  topology, optimizer, and data code did not change.
- The first attempt to apply the new synthetic GPU inference benchmark to
  dense checkpoints stopped before writing a result: timing events were
  created on the default CUDA device while the model ran on GPU 1. The helper
  was corrected to create and synchronize events inside the explicit device
  context. This benchmark-only failure did not alter any model or training.
- The three earlier convolutional synthetic benchmark files were generated
  before the explicit-device correction. Although they completed, they are
  conservatively retained as version-1 history and excluded. Corrected
  evidence is written separately as `synthetic_inference_benchmark_v2.json`.

## August 10, 2026: second round complete, full U2 frozen test and trade-offs

- The unchanged paper-faithful nine-channel LogicTreeNet-S U2 seed-0 run
  completed all 350K CUDA updates in 4.951 hours. The complete second-round
  ledger is 110/110 runs with no pending job and no completed run repeated.
- U2 reached 61.000% best hardened validation at step 294K and 60.280% at the
  final 350K step. This is +2.320 pp over the full matched random best and
  +1.140/+1.120 pp over full V4/U1.
- The validation manifest was written before test access and records
  `test_metrics_existing_at_freeze=false` for random, V4, U1, and U2. Each
  best-validation checkpoint was then evaluated exactly once on GPU 1.
  Hard test was 57.370/58.930/58.800/60.630% for random/V4/U1/U2; U2 gains
  +3.260 pp over random, +1.700 over V4, and +1.830 over U1. Its 60.630% is
  0.250 pp above the paper-reported 60.38% S test result. The full cohort is
  one seed; the three-seed 20K pilot remains the replication evidence.
- U2's mean hardened validation over the full learning curve is 59.562%
  versus 57.233% random. U2 reaches 59.5% at 34K updates; V4 first reaches it
  at 212K, U1 at 252K, and random never reaches it.
- All four methods retain exactly 83,552 learned LUT functions, 874,496
  spatial gate applications, 1,336,832 trainable LUT parameters, zero
  trainable routing parameters, and 1,945,600 deployed routing bits. Training
  time and peak GPU allocation are matched; U2 adds only 1.155 seconds of
  offline construction relative to random.
- Corrected synthetic hardened GPU inference is 6.852/6.856/6.855/6.835 ms
  per batch 128 for random/V4/U1/U2 with 0.3462 GiB peak allocation throughout.
  Sub-percent deltas are not speed claims. Compiled `gcc -O0` CPU circuit
  latency is 3.230/3.163/3.136/3.185 ms. U2's simplified IR is 262,260 nodes,
  +3.686% over random; functional equivalence passed. Energy was not measured.
- Machine-readable final artifacts are
  `summary/second_round_convolutional_validation_freeze.json`,
  `summary/second_round_convolutional_final.json`,
  `summary/second_round_convolutional_curves.{json,csv}`, and
  `summary/second_round_convolutional_deployment.json`.
- The calibrated fallback was not run: the frozen protocol permitted it only
  if both CIFAR-100 and convolutional U2 failed, whereas convolutional U2
  passed its promotion gate decisively.

## August 12, 2026: U2 published-protocol round completed

- Implemented the frozen third-round matrix without modifying U2, V3, or V4:
  six current dense M/L U2 runs, 14 LILogic M/L runs, and 18 BitLogic S/M/L
  runs. All 38 full trainings used CUDA through `venv` and completed. The six
  construction smokes are separate and are not accuracy results.
- Existing dense U2 hard-test accuracy is 58.653 +/- 0.168% on M and 60.463
  +/- 0.348% on L (three seeds). Paired gains over random are +4.557 and
  +4.593 pp, 3/3 wins in both cells. U2 is tied with V3 on M and trails V3 by
  0.610 pp on L, so V3 remains the best dense specialization.
- On the LILogic protocol, U2 reaches 52.543 +/- 0.296% on M and 60.193 +/-
  0.286% on L, gaining +3.533 and +4.860 pp over fixed random with 3/3 paired
  wins. One-seed Top-32 reaches 57.840% and 62.030%, close to or above the
  reported 57.28 +/- 0.30% and 60.98 +/- 0.19% coordinates.
- U2 uses the same fixed-random cost and 5x fewer training parameters than
  Top-32. Peak allocated memory is 0.474 versus 8.100 GiB on M and 1.557
  versus 24.861 GiB on L; local hardened latency is 0.934 versus 7.552 ms and
  4.303 versus 20.893 ms per batch 128. This is an accuracy--resource Pareto
  result because Top-32 remains more accurate.
- On the BitLogic common ladder, paired U2 gains are +2.260 pp at S (2/2),
  +0.095 pp at M (1/2), and +0.770 pp at L (1/2). Only S is defensibly
  positive; all n=2 intervals are underpowered.
- Local rank-4 BitLogic transfers reached only 27.500%, 16.625%, and 13.160%
  hard test at S/M/L, far below reported 38.93%, 49.22%, and 58.06%. The
  relaxed models learn signal (M final relaxed test 57.60% versus 14.70%
  hard), isolating a hardening/protocol mismatch. These are labeled
  `[REPRODUCED-NEGATIVE]`, never as faithful paper reproductions.
- Before test access, `summary/third_round_validation_freeze.json` hashed 38
  run artifact sets and 76 checkpoints. Both predeclared checkpoints were
  evaluated once per run on GPUs 0/1: 38/38 succeeded, zero failed, zero were
  missing. Synthetic benchmarks and test checkpoint hashes match the freeze.
- Added restart-safe finalization, immutable freeze, dual-checkpoint CUDA
  evaluation, aggregation, CSV exports, and provenance tests. The focused
  topology/protocol suite passes 145 tests; six third-round-specific tests
  pass after final aggregation. The complete suite passes with 3,412 passed,
  3,038 skipped, and one pre-existing warning in 257.62 seconds.
