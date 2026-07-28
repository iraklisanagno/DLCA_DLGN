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
