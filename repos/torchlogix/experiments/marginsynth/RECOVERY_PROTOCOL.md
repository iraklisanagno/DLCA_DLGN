# MarginSynth short-recovery protocol

This protocol alternates circuit simplification and short task-guided recovery.
Its purpose is to test whether a heterogeneous full-LUT resynthesis method can
recover useful accuracy in far fewer updates than long post-pruning fine-tuning.

## Fixed sequence

1. Start from the original hardened DLGN, which remains the cumulative teacher.
2. Run first-pass margin-constrained full-LUT resynthesis.
3. Infer the hard LUT rows changed in pass one and lock their selected Boolean
   functions. Connections and binarization thresholds remain fixed.
4. Reset eligible LUT logits to a moderate gap without changing any hard gate.
5. Train only unlocked LUT rows with a straight-through hard forward on a
   deterministic subset of the original training partition.
6. Select a recovery snapshot using only a disjoint holdout from that training
   partition. Calibration, validation, and test do not select the snapshot.
7. Run a second full-LUT resynthesis from the recovered checkpoint. Continue to
   compare behavior with the original teacher, lock the first-pass functions,
   use separate calibration optimization and repair sets, and evaluate once on
   an untouched calibration guard set.
8. Harden, exactly simplify, and run the identical Yosys/ABC flow.

## Recovery curves and fairness

Report snapshots at 0, 250, 500, 1,000, 2,000, 3,000, and 5,000 updates. Report
updates, examples processed, equivalent epochs, GPU time, peak memory, accuracy,
macro F1, total and per-class disagreement, hard/relaxed disagreement, changed
unlocked rows, lock violations, hardware proxy, live gates, ABC AND nodes, and
ABC levels. The primary efficiency target is at most 3,000 updates; 5,000 is a
diagnostic extension.

Apply the identical recovery code, optimizer, data split, batch size, loss, and
snapshot schedule to the local 10% Two-Stage Unit-Tying checkpoint. A 30,000
update point is a separate reference, not evidence that every Unit-Tying ratio
requires 30,000 updates. Compare both update count and examples processed.

## Required ablations

- no recovery;
- label-only recovery;
- label plus original-teacher decision-margin recovery;
- no hardware ceiling;
- soft-forward recovery;
- locked versus unlocked first-pass functions;
- one versus two resynthesis passes;
- recovery length.

The proposed contribution is not fine-tuning by itself. It is the recovery-
efficient alternating optimization of heterogeneous Boolean functions under a
hardware ceiling, locked accepted transformations, cumulative original-teacher
constraints, per-class/fold robustness, and exact held-out repair.
