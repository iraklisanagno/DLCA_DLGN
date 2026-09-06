> **Historical presentation snapshot:** this deck describes the Bayesian
> Fashion-MNIST seed-0 stage. It predates the three-seed dense-CIFAR transfer
> and the hardware-aware no-go. Do not use it as the current paper conclusion;
> see `RUN_LOG.md` and the root `docs/AI_HANDOFF.md`. Its “original reference”
> includes logic later removed by exact simplification, so those totals must
> not be labelled as the exact-simplified baseline.

# Slide 1 — MarginSynth

## Post-training, margin-constrained resynthesis of DLGNs

- **Starting point:** an already trained Differentiable Logic Gate Network
  (DLGN), hardened into a Boolean circuit.
- **Goal:** reduce the deployed circuit cost while explicitly limiting changes
  in predictions and in the behavior of individual classes.
- **Core idea:** reopen the Boolean-function choice of internal gates for a
  short optimization stage that balances decision margins and hardware cost.
- **Current evidence:** one Fashion-MNIST seed-0 development experiment.
- **Important:** validation was used for ordinary training checkpoint selection,
  but not for MarginSynth selection; the official test set remains sealed.

---

# Slide 2 — The complete process at a glance

```text
Stage A: ordinary DLGN training
    training images + classification loss
                    ↓
        trained soft DLGN checkpoint
                    ↓
Stage B: harden and freeze the original model
    choose one of 16 Boolean functions per gate
                    ↓
Stage C: MarginSynth post-training resynthesis
    calibration data + margin/hardware loss
                    ↓
    harden → exact repair → second pass → final guard
                    ↓
Stage D: deployment verification
    exact simplify → compiled C → Yosys → ABC
```

- We do **not** train the DLGN from scratch using the MarginSynth loss.
- Only Stage C is our proposed method.

---

# Slide 3 — Stage A1: prepare the data

Fashion-MNIST contains 60,000 official training images and 10,000 official test
images. We deterministically split the 60,000-image partition:

| Partition | Images | Purpose | Why separate it? |
|---|---:|---|---|
| Training | 48,000 (80%) | Learn the original DLGN | The only images used by ordinary training gradients |
| Validation | 6,000 (10%) | Select the original checkpoint | Prevents selecting a checkpoint on its training data |
| Calibration | 6,000 (10%) | Run MarginSynth after training | Prevents resynthesis from reusing training/validation examples |
| Official test | 10,000 | Final paper evaluation | Kept sealed until the full protocol is frozen |

- The 80/10/10 split is a conventional compromise: most data train the model,
  while two sizeable disjoint sets support checkpoint selection and
  post-training resynthesis.
- Split seed **2027** and partition hashes are stored for exact reproduction.

---

# Slide 4 — Stage A2: train the original DLGN

- **Input encoding:** each (28\times28) image is thresholded at 0.25, 0.50,
  and 0.75.
  - Three thresholds retain coarse grayscale information.
  - (28\times28\times3=2{,}352) Boolean input bits.
- **Architecture:** six layers × 8,000 rank-2 gates = **48,000 gates**.
  - This is the published-scale dense Fashion-MNIST architecture used by the
    optimized-connectivity comparison paper.
  - Rank 2 means that every gate has two inputs and one of 16 Boolean functions.
- **Connectivity:** fixed random wires; training learns gate functions, not
  connections.
- **Parameters:** (48{,}000\times16=768{,}000) gate logits.
- **Training:** Adam, learning rate 0.01, batch size 100, 108,000 GPU updates.
  - These were frozen original-model training settings, not tuned by
    MarginSynth.
  - Validation was checked every 2,000 updates; the best hard checkpoint was
    step 36,000 with 87.300% validation accuracy.

---

# Slide 5 — Stage B: harden and freeze the teacher

- Ordinary DLGN training gives 16 scores for each gate.
- Hardening selects the largest score:

  ```text
  learned gate scores: AND 3.7, XOR 0.8, OR 0.1, ...
  hardened function:  AND
  ```

- The hardened original model becomes a frozen **teacher**:
  - its connections never change;
  - its thresholds and input encoder never change;
  - its class scores define the reference decisions and margins;
  - its weights receive no MarginSynth gradients.
- The frozen Bayesian-study hardware reference has:
  - 34,740 live/reachable gates after normalization;
  - 107,369 ABC AND nodes; and
  - 79 logic levels.
- “Live gates” and “ABC nodes” differ because ABC decomposes and shares Boolean
  logic after export; one DLGN gate is not necessarily one AIG node.

---

# Slide 6 — Stage C1: reopen the 16 gate choices

There are exactly 16 Boolean functions of two inputs. MarginSynth groups them
by the type of simplification they can create:

| Category | Count | Examples | Why potentially cheap? |
|---|---:|---|---|
| Constants | 2 | 0, 1 | Trigger constant propagation through later logic |
| Unary/routing | 4 | A, ¬A, B, ¬B | Remove dependence on one input and possibly its upstream cone |
| Genuine binary | 10 | AND, OR, XOR, NAND, implications, … | May alter downstream simplification or use a cheaper Boolean form |

- “Keep the original gate” is **not** a 17th option; the original function is
  one of these 16.
- Original training asked: “Which function best improves task accuracy?”
- MarginSynth asks: “Can another function preserve important decisions while
  producing a cheaper complete circuit?”

---

# Slide 7 — Why the (16^N) combinations are manageable

- For (N) editable gates, exhaustive search would require (16^N) circuits.
  That is impossible even for a few dozen gates.
- Instead, each gate receives a 16-element trainable logit vector:

  \[
  z_g=[z_{g,0},z_{g,1},\ldots,z_{g,15}].
  \]

- We edit internal layers 1–4: (4\times8{,}000=32{,}000) gates.
  - The first and final logic layers are conservatively excluded.
  - This matches the eligible-layer policy used for Unit Tying.
- The optimization therefore stores:

  \[
  32{,}000\times16=512{,}000\text{ logits},
  \]

  not (16^{32{,}000}) candidate circuits.
- GPU backpropagation updates all logits together. Runtime grows roughly with
  gates × functions × minibatches × update steps—not exponentially.

---

# Slide 8 — Hard forward, soft backward

For each gate (g):

\[
p_g=\operatorname{softmax}(z_g/T),\qquad
k_g^*=\arg\max_k p_{g,k}.
\]

- **Forward pass:** execute only function (k_g^*). The circuit therefore
  behaves like a discrete gate network during optimization.
- **Backward pass:** use the derivative of the soft probabilities (p_g).
  This is a straight-through estimator for the non-differentiable `argmax`.
- Example:

  ```text
  forward:  gate executes AND
  backward: OR and A-routing can still receive useful gradient signals
  ```

- Temperature starts at **1.0** and decreases.
  - A lower temperature sharpens the probabilities toward one function.
  - Starting at 1.0 avoids an artificially sharp distribution on the first
    update.
- This estimator is efficient but biased; it does not guarantee the globally
  best circuit. Exact repair and synthesis remain necessary.

---

# Slide 9 — Stage C2: the joint circuit-level loss

The same minibatch passes through the entire candidate circuit. Only eligible
gate logits receive gradients.

\[
L=\lambda_dL_{margin}
+\lambda_cL_{worst\ class}
+\lambda_fL_{worst\ fold}
+\lambda_yL_{label}
+\lambda_hL_{hardware}.
\]

- **Margin loss:** preserve the teacher winner’s lead over its runner-up.
- **Worst-class loss:** prevent the average margin damage from concentrating in
  one Fashion-MNIST class.
- **Worst-fold loss:** divide calibration examples into four stratified folds
  and penalize the least stable fold.
- **Label loss:** retain some direct pressure toward the correct label.
- **Hardware loss:** favor functions expected to be cheaper.
- “Joint” means that a change in an early gate affects downstream gates and
  class scores in the same computation graph, so multiple gates can co-adapt.

---

# Slide 10 — Margin loss: a numerical example

For each example, use the teacher’s winning and runner-up classes:

\[
m_T=s^T_{winner}-s^T_{runner},\qquad
m_S=s^S_{winner}-s^S_{runner}.
\]

Trial 28 uses:

- margin retention = **0.5742**;
- minimum reserve = **0.3011**; and
- target cap = **2.0**.

The target is:

\[
m_{target}=\max(0.3011,\min(0.5742m_T,2.0)).
\]

- If the teacher margin is 1.0, target margin = 0.5742.
- If the teacher margin is 0.2, target margin = 0.3011; even difficult examples
  receive a small safety reserve.
- If the teacher margin is 10, target margin = 2.0; very easy examples do not
  dominate the loss.
- The penalty is (\max(0,m_{target}-m_S)).
- These exact values were selected by seed-0 Bayesian development from frozen
  ranges; they are not universal constants and must now be tested unchanged.

---

# Slide 11 — Hardware and robustness terms

- **Gate-count proxy:** constants and unary routing/inversion receive cost 0;
  genuine two-input functions receive cost 1.
  - It is fast and differentiable.
  - It does not model all downstream ABC sharing or constant propagation.
- The hardware penalty is gradually enabled after a warm-up so behavior is not
  destroyed immediately.
- Trial-28 loss weights:

| Term | Pass 1 | Pass 2 | Interpretation |
|---|---:|---:|---|
| Decision margin | 0.6622 | 0.6622 | Main teacher-behavior pressure |
| Worst class | 0.2152 | 0.1010 | Protect class groups |
| Worst fold | 1.0383 | 0.4825 | Protect stability across data subsets |
| Correct label | 0.1735 | 0.0830 | Keep task supervision |
| Hardware | 1.6233 | 1.8255 | Encourage cheaper functions |

- These weights were chosen by Bayesian multi-objective exploration. Their
  precision records the exact trial; it should not be interpreted as a claim
  that four decimal places are scientifically meaningful.

---

# Slide 12 — Stage C3: three-way calibration firewall

The 6,000 calibration images are class-stratified into:

| Subset | Actual size | Fraction | Used for |
|---|---:|---:|---|
| Optimization | 3,599 | ≈60% | Gradient updates |
| Repair | 1,201 | ≈20% | Decide how many hardened changes to retain |
| Guard | 1,200 | 20% | Evaluate the already selected final candidate |

- Counts differ slightly from 3,600/1,200/1,200 because rounding is performed
  separately inside each of the ten classes.
- The split seed is **0**, and every index hash is logged.
- Four stratified stability folds are used inside optimization: four provides
  multiple robustness views while retaining hundreds of examples per fold.
- Validation is not loaded by the Bayesian MarginSynth run.
- Test is never loaded.
- This separation limits overfitting: gradients do not see repair or guard,
  and repair decisions do not see guard.

---

# Slide 13 — What the behavioral budgets mean

The constrained study declares four maximum losses before evaluating trials:

| Budget | Limit | Intuition on the 1,200-example guard |
|---|---:|---|
| Global accuracy loss | 0.333 percentage points | About four additional errors at most |
| Global disagreement | 3.0% | At most about 36 changed predictions |
| Worst-class accuracy loss | 1.5 percentage points | Prevent one class absorbing the global error budget |
| Worst-class disagreement | 7.5% | Roughly nine changed predictions in a 120-example class |

- The 0.333-point global level matches the observed accuracy-loss scale of the
  10% Unit-Tying operating point used as the direct baseline.
- Disagreement counts any changed teacher prediction, even when accuracy does
  not change. It detects behavioral drift hidden by aggregate accuracy.
- Per-class limits prevent apparently good global results from damaging one
  class disproportionately.
- These are development engineering budgets, not universal statistical laws;
  they must transfer across seeds before becoming the paper protocol.

---

# Slide 14 — Stage C4: first resynthesis pass

Trial-28 pass-1 settings and their roles:

| Choice | Value | Why this value appears |
|---|---:|---|
| Editable gates | 32,000 | Four conservative internal 8,000-gate layers |
| Batch size | 256 | GPU-vectorized updates within memory limits |
| Updates | 400 | Bayesian choice from {200, 400, 600} |
| Learning rate | 0.03669 | Bayesian choice from 0.008–0.04 |
| Initial original-gate logit gap | 4.577 | Strongly starts from the trained circuit; ≈86.6% initial soft probability on the original function at (T=1) |
| Temperature | 1.0 → 0.386 | Bayesian endpoint from 0.15–0.60; sharpens choices gradually |
| Hardware warm-up | 42 updates | 10.5% of 400; behavior adapts before full cost pressure |
| Gradient clip | 5.0 | Fixed protection against unstable updates |

- With 3,599 optimization examples, (400\times256/3599\approx28.5)
  equivalent passes through that subset.
- Gradient optimization proposed **8,884** gate changes.
- This number is a proposal, not the final circuit; exact repair follows.

---

# Slide 15 — Stage C5: exact repair after pass 1

Gradient suggestions are hardened and ranked by:

1. estimated cost benefit;
2. learned preference over the original function; and
3. deterministic layer/gate order for reproducibility.

Then the repair stage:

- tests the complete hardened circuit on all 1,201 repair examples;
- binary-searches for a large feasible prefix of the ranked changes; and
- scans the next **64** prefix sizes because interacting Boolean changes mean
  feasibility is not perfectly monotonic.

Example:

```text
8,884 proposed changes
4,211 retained because the larger tested prefixes violated repair budgets
```

- The 4,211 retained changes pass the repair subset.
- They do **not** pass the independent first-pass guard: global loss is 0.417
  points and worst-class loss is 2.632 points.
- Therefore pass 1 is an intermediate source, not a publishable operating
  point. Continuing permits later coordinated changes to restore behavior.

---

# Slide 16 — Stage C6: locked second resynthesis pass

- Start from the pass-1 repaired checkpoint.
- Lock its 4,211 retained changes so pass 2 searches for complementary changes
  rather than simply reversing pass 1.
- Keep the original trained network as the teacher and use the same cumulative
  behavioral budgets.

| Choice | Value | Explanation |
|---|---:|---|
| Updates | 600 | Bayesian choice from {200, 400, 600}; more adaptation after locking |
| Learning rate | 0.01320 | Lower Bayesian-selected rate for the constrained second stage |
| Initial gap | 5.006 | ≈90.9% initial soft probability on each current function at (T=1) |
| Temperature | 1.0 → 0.497 | Sharpening without forcing an extremely cold endpoint |
| Hardware warm-up | 63 updates | Again 10.5% of the pass |
| Sampling seed | 1009 | Fixed offset from seed 0 to decorrelate minibatch order reproducibly |

- Pass 2 proposes 6,148 new changes and exact repair retains **2,017**.
- Final cumulative retained changes: (4{,}211+2{,}017=6{,}228).
- The final guard now passes all budgets, despite the intermediate first-pass
  guard failure. “Guarded” therefore describes final trial acceptance.

---

# Slide 17 — Stage D: export, verify, and synthesize

1. Save the hardened PyTorch checkpoint and all selected LUT IDs.
2. Export the DLGN as an editable Boolean `Circuit`.
3. Apply function-preserving circuit simplification.
4. Compile generated C and verify circuit predictions against the hardened
   PyTorch model on calibration data.
5. Export Verilog/BLIF and run the same Yosys/ABC flow used for every method.
6. Record live gates, ABC AND nodes, logic levels, commands, hashes, versions,
   time, CPU memory, and GPU memory.

Tools used:

- PyTorch 2.9.0+cu130 on an NVIDIA RTX PRO 6000 for resynthesis;
- TorchLogix for the DLGN and circuit export;
- compiled C for independent semantic checking;
- Yosys 0.9 and Berkeley ABC 1.01 for normalized hardware measurement; and
- JSON/CSV/SQLite logs for replay, meta-analysis, and ablations.

Exact promotion took 172.98 s for trial 28; this common verification/synthesis
cost is reported separately from the 28.73 s MarginSynth method time.

---

# Slide 18 — How trial 28 was selected

We did not hand-pick its many hyperparameters after seeing the final circuit.

- Four study cases:
  - guarded two-pass with/without disagreement constraints;
  - aggressive resynthesis plus recovery with/without disagreement constraints.
- **40 trials per case** = 160 trials total.
  - 12 startup trials explore broadly, including one declared reference.
  - 28 constrained TPE trials use earlier observations to suggest settings.
  - Forty was a bounded seed-0 development budget, balancing coverage and
    several hours of compute; it is not part of frozen deployment runtime.
- Objectives: minimize guard accuracy loss and predicted ABC nodes.
- At most **10** feasible Pareto/diversity candidates per case receive costly
  exact synthesis; this avoids synthesizing all 160 trials.
- Results: 29 feasible acquisition trials, 27 exact promotions, and 27/27
  successful export/compiled-C/Yosys/ABC checks.
- Trial 28 is the lowest exact ABC-node point among the promoted feasible
  guarded-constrained candidates.

---

# Slide 19 — Current seed-0 result and trade-offs

| Metric | Original reference | Unit Tying, 10% | MarginSynth trial 28 |
|---|---:|---:|---:|
| Guard accuracy loss | 0.000 pp | +0.083 pp | **-0.250 pp** |
| Global disagreement | 0.000% | **2.000%** | 2.083% |
| Worst-class accuracy loss | 0.000 pp | 2.632 pp | **0.893 pp** |
| Worst-class disagreement | 0.000% | 5.042% | **4.464%** |
| Live gates | 34,740 | 30,405 | **27,493** |
| ABC AND nodes | 107,369 | 94,084 | **91,919** |
| ABC levels | 79 | 79 | 79 |
| Method time before common synthesis | — | **2.02 s** | 28.73 s |

- Versus Unit Tying: 9.58% fewer live gates and 2.30% fewer ABC nodes, with
  the same depth and much lower worst-class accuracy loss.
- Cost: MarginSynth is currently about 14.3× slower than Unit Tying.
- Its global disagreement is worse by one guard example.
- Negative guard accuracy loss is calibration variation, not evidence that
  MarginSynth improves unseen test accuracy.

---

# Slide 20 — Interpretation, Unit Tying, and the paper path

Why this is methodologically different from Unit Tying:

- Unit Tying ranks gates and forces a fixed ratio to constants.
- MarginSynth jointly reoptimizes all 16 functions under margin, group, label,
  and cost terms; then exact repair decides how many changes survive.
- It starts from the original checkpoint and does not use a Unit-Tying
  shortlist, Binary Split, checkpoint, or warm start.

What the current result teaches us:

- The winner retained 1,626 constants and 4,602 routing/inversion changes.
- No alternative genuine binary function survived exact repair.
- Thus the general framework is broader than Unit Tying, but the paper must
  prove that routing, joint optimization, and guarding—not unnecessary action
  space—produce the gain.

Next steps:

1. Apply trial 28 unchanged to Fashion-MNIST seeds 1 and 2.
2. Freeze the method only if it transfers; then run five central seeds.
3. Run constant-only, routing-only, one-pass, no-guard, no-repair, and cost
   ablations.
4. Make CIFAR-10 the central DATE benchmark and compare with Unit Tying and the
   other frozen paper baselines.
5. Reduce frozen runtime through batching, caching, and fewer repair checks.

**Question for discussion:** is the strongest contribution the general
all-function resynthesis framework, or the emerging margin-guarded
constant-and-routing specialization?
