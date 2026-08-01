# MarginSynth constrained Bayesian exploration protocol

## Purpose and status

This protocol implements the prespecified hyperparameter exploration for four
cases. It is development/model-selection infrastructure, not a test-set
evaluation and not itself a paper result.

| Study case | Executed method | Disagreement policy |
|---|---|---|
| `guarded_constrained` | Independent resynthesis followed by a locked, guarded second resynthesis | Overall and worst-class disagreement are constraints |
| `guarded_unconstrained` | The identical two-pass method | Disagreement is measured but is not a constraint |
| `aggressive_constrained` | Aggressive unrepaired resynthesis followed by locked short recovery | Overall and worst-class disagreement are constraints |
| `aggressive_unconstrained` | The identical aggressive/recovery method | Disagreement is measured but is not a constraint |

Accuracy, worst-class accuracy, and locked-function safety are never removed.
The aggressive cases additionally constrain the selected recovery point to at
most 3,000 updates. A 5,000-update choice remains in the search space as a
diagnostic infeasible point.

The frozen machine-readable protocol is
[`configs/bayesian_exploration_fashion_seed0.json`](configs/bayesian_exploration_fashion_seed0.json).
Changing a range, budget, objective, split, seed, or reference point requires a
new protocol name and a new output directory.

## What each trial runs

### Guarded two-pass MarginSynth

1. Start from the original hardened checkpoint, never a Unit-Tying checkpoint.
2. Jointly optimize heterogeneous rank-2 LUT functions in the first pass.
3. Harden and exactly repair the learned rewrite prefix.
4. Start a second joint resynthesis from that first-pass checkpoint.
5. Lock every function accepted in pass one and optimize only the remaining
   eligible LUT rows.
6. Harden and exactly repair the second prefix.
7. Independently reevaluate the result on the reserved calibration guard.
8. If behaviorally feasible, export it and measure live gates, ABC AND nodes,
   and ABC levels with the same Yosys/ABC flow used by the baselines.

Both passes use the same stratified 60/20/20 calibration partition. The 20%
guard is therefore unseen by gradients and exact repair in both passes. The
optimizer seed for pass two is different, while `partition_seed` is fixed so
the data firewall does not change.

### Aggressive MarginSynth plus short recovery

1. Start from the original hardened checkpoint.
2. Jointly optimize all eligible LUT functions without exact prefix repair.
3. Reserve the same 20% calibration guard without using it in this aggressive
   pass.
4. Infer every changed hard function and lock those rows.
5. Recover only unlocked LUT rows using the original training partition and a
   disjoint training-monitor holdout.
6. Select the earliest feasible recovery snapshot without loading calibration,
   validation, or test examples.
7. Independently score the recovered checkpoint on the reserved calibration
   guard and its four deterministic reporting folds.
8. If behaviorally feasible, run the identical exact export and Yosys/ABC flow.

The final validation partition is not loaded anywhere in Bayesian trial
selection. It remains available for a single evaluation after the search and
selection rule have been frozen. Test remains sealed until the later paper
protocol explicitly authorizes it.

## Optimizer and objectives

Optuna 4.6.0 multi-objective TPE (MOTPE) is pinned in
[`experiments/requirements.txt`](../requirements.txt). MOTPE is used because
the search spaces combine log-scaled floats, integers, categorical choices,
and method-dependent parameters. Each study has its own constrained Pareto
model but uses the same sampler seed.

The final paper Pareto front has only two minimized objectives:

1. accuracy loss relative to the original hardened teacher on the calibration
   guard; and
2. exact ABC AND-node count.

The Bayesian acquisition stage uses guard accuracy loss and a calibrated,
operation-aware ABC estimate. The estimate exports and exactly simplifies the
hard circuit without loading any dataset, extracts operation mix, live gates,
connections, sum inputs, and depth, and applies the frozen same-flow estimator
in `synth_cost_model.json`. Exact measurements and proxy values are always
stored in separate fields.

After all 40 suggestions, a deterministic promotion stage selects at most ten
feasible candidates per study: all proxy-Pareto points (evenly thinned if
necessary), followed by joint accuracy/hardware-rank diversity fillers. Only
those promoted candidates run full export, semantic verification, Yosys, and
ABC. Only feasible candidates with an exact ABC result enter `pareto.json`.
This limits the measured 190-second export bottleneck without selecting exact
measurements after seeing validation or test results. `--synthesize-infeasible`
is reserved for explicit integration/diagnostic runs. Smoke trials remain
excluded from the paper Pareto front.

The initial reference configuration is enqueued before the space-filling
startup trials. Each full study contains 40 trials: the reference/startup
phase followed by MOTPE suggestions. Exploration uses seed 0 only. The best
three feasible Pareto configurations are subsequently repeated on seeds 1 and
2; one configuration is then frozen before the five-seed paper study.

## Reproducibility and meta-analysis record

No failed or infeasible trial is deleted. Each trial directory contains:

- a lifecycle `trial_record.json`, including status and failure traceback;
- exact suggested parameters and fully resolved stage configurations;
- one console log and timing/resource record for every subprocess;
- first/second-pass optimization traces, learned changes, repair logs, data
  split hashes, software versions, checkpoints, and artifact manifests;
- recovery snapshots, optimizer state, training curves, selected step,
  examples processed, GPU time/memory, and lock violations when applicable;
- independent guard metrics, per-class arrays, four fold records, and sample
  hashes; and
- complete export, Yosys, and ABC inputs, commands, logs, versions, timings,
  circuits, and summaries when synthesis runs.

The study root contains the frozen protocol, source-checkpoint hashes, Python
environment, `pip freeze`, Git status/revision, CUDA/GPU probe, tool versions,
Optuna SQLite database, optimizer log, and append-only lifecycle events.

After every trial the runner regenerates tidy analysis files:

- `trials.csv` and `trials.json`;
- `parameters_long.csv`;
- `constraints_long.csv`;
- `metrics_long.csv`;
- `stages.csv`; and
- exact-feasible-only `pareto.csv` and `pareto.json`.

These tables retain infeasible and failed observations for censored-runtime,
failure-mode, sensitivity, ablation, and later meta-analysis. The Optuna
database remains the authoritative optimizer state; the tables are portable
analysis exports.

## Commands

Run from `repos/torchlogix` with the existing virtual environment.

Validate paths and the frozen schema without creating a study:

```bash
venv/bin/python experiments/marginsynth/bayesian_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --validate-only
```

Run one isolated GPU smoke trial for every case without ABC:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/bayesian_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study all --n-trials 1 --smoke
```

Run the complete four-study Bayesian acquisition stage:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/bayesian_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study all
```

Promote at most ten feasible candidates per study to exact hardware
measurement after all four searches finish:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/promote_bayesian_trials.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study all
```

Resume safely after interruption, adding the requested number of trials to
each selected study:

```bash
DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/bayesian_search.py \
  experiments/marginsynth/results/pilot_fashion_mnist_paper_small_raw_seed0 \
  --protocol experiments/marginsynth/configs/bayesian_exploration_fashion_seed0.json \
  --study guarded_constrained --n-trials 5 --resume
```

CUDA is mandatory for all optimization, recovery, and guard-evaluation
stages. Exact export/simulation and Yosys/ABC remain CPU stages because they
exercise the deployed Boolean circuit and external synthesis tools.

## Fair runtime reporting

The one-time four-study optimizer cost is reported separately from the cost of
applying a frozen method. For the paper, report:

- total Bayesian search CPU/GPU hours and synthesis calls;
- per-trial failure/infeasibility rates;
- frozen per-model resynthesis, recovery, export, and synthesis wall time;
- recovery steps, examples, equivalent epochs, and GPU memory; and
- the same per-model processing statistics for Unit Tying.

The Bayesian cost is not charged to every deployed model only if the frozen
configuration transfers across seeds. If a separate search is required for
each checkpoint, its search time must be counted as method overhead.
