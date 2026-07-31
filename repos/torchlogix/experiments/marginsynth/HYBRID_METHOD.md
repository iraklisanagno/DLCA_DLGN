# Baseline: Unit Tying + Margin Refinement

> **Status:** preserved comparison baseline. This is not the proposed
> MarginSynth method because it inherits Two-Stage Unit Tying's structured
> constant-tie action, candidate shortlist, and fixed tie quota.

## Method

The redesign keeps the fast, regular transformation used by Two-Stage Unit
Tying: selected DLGN units are forced to constant zero or one. It changes how
the tied set is chosen and optionally follows it with a very small circuit
cleanup.

1. Use the original Gauss--Newton score to form a small oversampled candidate
   pool in every eligible layer.
2. On a separate calibration sample, project the teacher winner-versus-runner
   decision margin through each possible constant tie.
3. Score mean margin risk, worst-class risk, worst-fold risk, fold variance,
   projected decision flips, and direction-aware structural benefit.
4. Start from the published Unit-Tying set and test a small number of
   deterministic swaps between its riskiest ties and safer shortlisted ties.
5. Accept swaps only when exact full-calibration evaluation improves the
   accuracy-first objective under all global and per-class budgets.
6. Export and synthesize through the identical Circuit/Yosys/ABC flow.
7. Optionally run at most 16 coordinated MarginSynth residual rewrites.

The full tie selector is GPU batched. Validation is used only after selection;
test data is not accessed by development commands.

## Seed-0 development result

| Method | Validation accuracy | Loss from exact baseline | Disagreement | Live gates | ABC nodes | Selection time |
|---|---:|---:|---:|---:|---:|---:|
| Unit Tying, 10% | 86.967% | 0.333 pp | 2.483% | 30,405 | 94,084 | 2.0 s |
| Hybrid swaps, 10% | 87.033% | 0.267 pp | 2.450% | 30,384 | 94,070 | 3.2 s |
| Hybrid + 16 residual rewrites | 87.017% | 0.283 pp | 0.750% relative to hybrid | 30,207 | 93,763 | 214 s total |

The hybrid strictly dominates Unit Tying on this seed, but the advantage is
small: 321 ABC nodes (0.34%) and 0.05 percentage points of validation accuracy
after residual cleanup. This is evidence that the direction is viable, not a
paper claim. Five-seed evaluation, a second dataset, matched recovery training,
and stronger swap-group construction remain required.

## Reproduction

```bash
DATASET_PATH=/tmp/torchlogix-datasets CUDA_VISIBLE_DEVICES=0 \
venv/bin/python experiments/marginsynth/margin_aware_tying.py RUN_DIR \
  --config experiments/marginsynth/configs/margin_aware_tying_fashion_seed0_matched_cost.json

DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/export_tied_method.py RUN_DIR METHOD_DIR \
  --prepare-residual-trace

DATASET_PATH=/tmp/torchlogix-datasets \
venv/bin/python experiments/marginsynth/search_v2.py METHOD_DIR/export_run \
  --config experiments/marginsynth/configs/hybrid_residual16_fashion_seed0.json
```

Every stage records configurations, sampled indices and folds, candidate
ranking, exact round decisions, swaps, checkpoints, circuit hashes, replay
verification, validation metrics, and synthesis logs.
