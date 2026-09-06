# Historical liveness design notes

> These scratch notes predate the implemented liveness/activity component
> study. Safe topological liveness removed no eligible gates on the tested
> Fashion-MNIST or dense-CIFAR sources, and the latest conclusion is recorded
> in `RUN_LOG.md` and the root `docs/AI_HANDOFF.md`.

## What the original system caught

   Situation                                             Caught now?          How?
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Gate disconnected from every output                   Yes, after export    Exact dead-gate elimination
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Input logically constant 0/1                          Yes, after export    Exact constant propagation
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Gate output logically constant                        Yes, after export    Exact Boolean simplification
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Gate output constant on all calibration examples      Indirectly           MarginSynth may learn a constant replacement
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Gate output constant 90% of the time                  Indirectly           Gradient optimization plus repair/guard
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Explicit activation-frequency candidate generation    No                   Not currently implemented
  ────────────────────────────────────────────────────  ───────────────────  ──────────────────────────────────────────────
   Pre-MarginSynth liveness masking                      No                   Proposed improvement

  ## How the liveness-mask extension should work

  We should separate three kinds of liveness:

  1. Topologically dead gates
     Cannot reach any output even if gate functions change. These can always be excluded safely.

  2. Functionally dead gates
     Dead because a current constant or unary gate blocks their path. These could become live again if MarginSynth changes that downstream gate to use both inputs.

  3. Empirically inactive gates
     Constant or nearly constant only on calibration examples. Removing these is approximate and requires behavioral checks.

  A safe first implementation would:

  1. Export the hardened original circuit.
  2. Run exact simplification and map surviving nodes back to DLGN gate IDs.
  3. Permanently mask only topologically dead gates.
  4. Record functionally dead gates separately.
  5. Compute activation rates for every remaining gate:
      - overall;
      - per class;
      - across stability folds.

  6. Use empirical constancy as a candidate prior, not an automatic deletion rule.
  7. Rank constant candidates using local mismatch and downstream margin impact.
  8. Continue using disjoint repair and guard subsets for acceptance.

  For example:

  Gate output is 1 on:
  overall: 98%
  class 0: 100%
  class 1: 81%
  other classes: 99%

  The 98% aggregate statistic looks attractive, but the class-1 statistic warns that replacing it with 1 could disproportionately affect that class.

  The strongest extension is therefore not merely a liveness mask. It is:

  > Exact structural liveness masking plus class-aware empirical activity priors.

  That could both accelerate optimization and make the constant/routing choices more deliberate, while keeping exact and approximate simplifications clearly separated.


   B0     Original hardened circuit + exact simplification    Function-preserving reference
  ─────  ──────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   B1     Two-Stage Unit Tying                                Direct post-training competitor
  ─────  ──────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   B2     Silicon-Aware loss from scratch                     End-to-end hardware-aware competitor
  ─────  ──────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   B3     Silicon-Aware-style post-training control           Isolate whether our margins/repair add value
  ─────  ──────────────────────────────────────────────────  ─────────────────────────────────────────────────────
