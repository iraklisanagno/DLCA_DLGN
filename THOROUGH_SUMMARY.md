## Bottom line

The project has produced a valid and potentially publishable contribution—but not exactly the one proposed initially.

- The original `coverage_hybrid` idea, based primarily on maximizing ancestry coverage through greedy swaps, did not survive experimentation.
- The broader hypothesis—that fixed random wiring wastes capacity and that deterministic topology design can improve DLGNs—was strongly validated.
- The most defensible method is Unified U2: deterministic semantic ordering, degree-balanced routing, and multiscale pairing, with no learned routing.
- The current results are paper-quality, but I would not submit them to DATE yet as a definitive state-of-the-art paper. The main convolutional results need multi-seed confirmation, modern parameterizations, and synthesized hardware measurements.

The project is therefore a scientific success, but the paper story needs to be revised.

## 1. The original idea

The original specification in [coverage_dlgn.md](/ssd1/users/ianagno/my-repos/DLCA_DLGN_coverageDLGN/ideas/date_ideas/coverage_dlgn.md:24) asked:

> At a fixed gate budget, can deterministic coverage-aware connectivity give a hardened DLGN more useful input diversity than random connectivity?

The target claims were:

1. Improve hardened accuracy at the same gate budget.
2. Move the accuracy–gate-count Pareto frontier.
3. Reduce sensitivity to random topology seeds.
4. Require no learned routing or additional deployed state.
5. Explain the gains through improved ancestry coverage and reduced overlap.
6. Apply the same principle to dense and convolutional DLGNs.

The proposed method was originally `coverage_hybrid`: mostly local/butterfly connections combined with greedy long-range ancestry-based edges.

The specification also included a kill criterion: stop treating the original hybrid as a DATE paper if it failed to improve Fashion-MNIST and CIFAR-10 by at least 0.3 percentage points, or could not reduce gates/storage.

Strictly speaking, the original hybrid failed that criterion.

## 2. What we implemented and evaluated

The project went substantially beyond a single topology experiment.

### Infrastructure and reproducibility

We added:

- A shared fixed-connectivity strategy interface.
- Random-unique, local, butterfly, greedy-coverage, hybrid, V3, V4, U1, and U2 generators.
- Packed ancestry propagation and raw-source/semantic ancestry diagnostics.
- Fan-out, duplicate-pair, overlap, source coverage, and unused-input measurements.
- Deterministic topology generation and stored topology metadata.
- Dense and convolutional TorchLogix integration.
- Circuit export and equivalence checks.
- Learning-curve aggregation.
- Runtime, GPU memory, topology-construction, compiled CPU and simplified-circuit measurements.
- Frozen protocols, validation selection and one-time held-out testing.
- Machine-readable per-run results and provenance.

At least 110 second-round and 38 third-round CUDA runs were completed, in addition to the earlier pilots and LogicTreeNet-M experiments. The complete TorchLogix test suite reached 3,412 passed tests. The full history is documented in [RESULTS.md](/ssd1/users/ianagno/my-repos/DLCA_DLGN_coverageDLGN/repos/torchlogix/experiments/coverage_dlgn/RESULTS.md), with consolidated conclusions in [SECOND_ROUND_CONCLUSIONS.md](/ssd1/users/ianagno/my-repos/DLCA_DLGN_coverageDLGN/repos/torchlogix/experiments/coverage_dlgn/SECOND_ROUND_CONCLUSIONS.md).

### Method evolution

1. `random_unique`, local and early butterfly schedules did not reliably improve accuracy.
2. Pure greedy ancestry coverage was close to random but not consistently better.
3. Hybrid V1/V2 improved topology metrics, but accuracy gains were weak.
4. The original bit-level ancestry metric was found to be misleading for thermometer-encoded inputs because different thresholds of the same image channel were counted as independent sources.
5. V3 introduced semantic first-layer ordering, degree-balanced butterfly routing and deeper ancestry swaps. It produced large dense CIFAR-10 gains.
6. V4 adapted the concept to convolutional channels, but the original implementation was tested first on a non-paper-faithful six-channel architecture.
7. Correcting the architecture to the paper’s nine channels reduced V4’s gains.
8. V5 maximized ancestry diversity more aggressively. It improved topology metrics dramatically but did not improve accuracy.
9. U1 removed ancestry swaps and kept balanced semantic connectivity. It was positive but not statistically strong enough in convolutional S.
10. U2 introduced a genuinely unified dense/convolutional rule: semantic ordering, complete degree-balanced matching stages, and multiscale ancestry-aware stage selection, without individual greedy swaps.

V3 and V4 were preserved rather than overwritten.

## 3. Main experimental findings

### Dense CIFAR-10: the strongest evidence

| Gates | Random test | V3 test | Gain | Seeds |
|---:|---:|---:|---:|---:|
| 48K | 49.056% | 52.358% | +3.302 pp | 5 |
| 128K | 49.748% | 53.910% | +4.162 pp | 5 |
| 256K | 52.253% | 56.903% | +4.650 pp | 3 |
| 384K | 53.657% | 58.143% | +4.487 pp | 3 |
| 512K | 54.028% | 58.284% | +4.256 pp | 5 |
| 1.28M | 55.960% | 61.020% | +5.060 pp | 5 |

These are strong, consistent, statistically supported fixed-budget improvements.

They also establish a real compression/Pareto result:

- V3 with 128K gates reaches 53.910%, slightly exceeding random with 384K gates at 53.657%: approximately 67% fewer gates.
- V3 with 256K gates reaches 56.903%, exceeding the largest tested random model with 1.28M gates at 55.960%: approximately 80% fewer gates among the evaluated points.

Unified U2 also transfers strongly:

- CIFAR-10 M: +4.557 pp over random, three seeds.
- CIFAR-10 L: +4.593 pp over random, three seeds.
- U2 is slightly below V3, particularly at L, but U2 is the same method used for convolutional models.

### What caused the dense gain?

The CIFAR-10 M component ablation is decisive:

| Construction | Validation | Increment |
|---|---:|---:|
| Random | 54.820% | — |
| Degree-balanced butterfly | 58.980% | +4.160 pp |
| + semantic first layer | 59.253% | +0.273 pp |
| + ancestry swaps, full V3 | 59.293% | +0.040 pp |

Approximately 93% of the observed V3 improvement came from degree-balanced routing. The semantic ordering helped modestly. Individual ancestry swaps contributed almost nothing measurable.

This rejects the strongest form of the original explanation. The result is not “more ancestry is always better.” It is:

> Regular, degree-balanced and semantically structured information propagation is better than unconstrained random wiring; ancestry is useful as a constrained secondary criterion, not as an objective to maximize.

### MNIST and Fashion-MNIST

The gains are positive but much smaller:

- MNIST U2, 8K gates: 91.273% → 91.937%, +0.663 pp, three seeds, positive confidence interval.
- Fashion-MNIST U2, 16K gates: +0.520 pp, but the confidence interval crosses zero.
- V3 was positive in all eight MNIST/Fashion compression-ladder cells and won 22/24 paired runs.
- Most individual ladder cells were statistically underpowered.

These datasets support consistency under compression, but they should not be headline datasets. MNIST is saturated and current learned-connectivity methods are much more accurate at comparable nominal gate counts.

### Paper-faithful convolutional CIFAR-10

Unified U2 finally produced credible convolutional gains:

| Architecture | Random test | U2 test | Gain | Replication |
|---|---:|---:|---:|---|
| LogicTreeNet-S, 350K updates | 57.370% | 60.630% | +3.260 pp | Full run, one seed |
| LogicTreeNet-M, 200K updates | 69.570% | 71.650% | +2.080 pp | Full run, one seed |

Supporting S-scale 20K pilots gave +2.173 pp over random with three out of three wins and a positive confidence interval.

The original convolutional paper reports 60.38% for S and 71.01% for M. U2 is therefore numerically above both reported values by +0.25 and +0.64 pp, respectively. However, these are one-seed full runs and our protocol is not identical in every training detail. They are promising numerical results, not yet state-of-the-art proof. The original convolutional paper also reaches 86.29% using its much larger G architecture. [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/db988b089d8d97d0f159c15ed0be6a71-Paper-Conference.pdf)

### CIFAR-100

CIFAR-100 is the main negative result:

- V3 at 3 × 128K improved test accuracy from 20.923% to 21.467%, +0.543 pp over three seeds.
- The validation gain of +0.840 pp was significant, but the test confidence interval crossed zero.
- U2 gained only +0.100 pp in its pilot and was rejected.
- Increasing depth to 12 or 24 layers caused chance-level optimization collapse.
- Wider short experiments did not promote.

Thus U2 is not currently a universal topology rule across datasets. CIFAR-100 exposes optimization and representation limitations that topology alone cannot repair.

## 4. Cost and deployment findings

Within matched experiments, U2 preserves:

- The number of LUT functions.
- The number of spatial gate operations.
- Trainable LUT parameters.
- Zero trainable routing parameters.
- Deployed routing storage.
- Training steps and optimizer.
- GPU training memory.
- Hardened CUDA inference runtime within measurement noise.

For full LogicTreeNet-S:

- Random and U2 both used 1.831 GiB peak training memory.
- Training time was approximately 4.95 hours for both.
- U2 topology construction took 1.372 seconds versus 0.217 seconds for random.
- Hardened CUDA inference was effectively identical.
- U2’s simplified circuit contained 3.686% more IR nodes than random.

Therefore, we can claim matched declared cost and runtime, but not yet superior physical hardware cost. The larger simplified IR is an honest trade-off and makes FPGA/ASIC synthesis essential.

## 5. Is the original idea valid?

A claim-by-claim assessment:

| Claim | Verdict |
|---|---|
| Random fixed routing wastes capacity | Strongly supported |
| Deterministic topology improves fixed-budget accuracy | Strongly supported on CIFAR-10 |
| Accuracy–gate-count Pareto improves | Strongly supported for dense CIFAR-10 |
| Maximum ancestry coverage causes the gain | Rejected |
| Ancestry swaps are important | Not supported |
| Degree/fan-out balance matters | Strongly supported |
| One topology rule works for dense and convolutional DLGNs | Supported by U2 on CIFAR-10 |
| The method works universally across datasets | Rejected by CIFAR-100 |
| The method reduces topology-seed sensitivity | Not sufficiently supported |
| No learned routing or extra deployed routing state | Supported |
| No physical circuit-cost penalty | Not established; simplified IR increased 3.686% |

So:

- The original `coverage_hybrid` mechanism is not valid as the paper’s final method.
- The broader CoverageDLGN research idea is valid.
- U2 is the defensible revised method.
- The paper should explain that experiments changed our understanding of “coverage” from maximum ancestry diversity to structured, balanced information propagation.

## 6. Are the results publishable?

### What is already publishable

The following form a legitimate contribution:

1. A deterministic, topology-only method with zero learned routing.
2. Strong five-seed dense CIFAR-10 gains from +3.3 to +5.1 pp.
3. A substantial accuracy–gate-count Pareto improvement.
4. A single U2 rule that works in dense and convolutional CIFAR-10.
5. Numerical improvements over reported LogicTreeNet-S/M.
6. Detailed topology ablations showing why naive ancestry maximization fails.
7. Strong reproducibility, circuit export and cost accounting.
8. An important negative result: coverage saturation does not repair deep-network optimization collapse.

### Why I would not submit the full DATE paper yet

A reviewer could reasonably object that:

- Full convolutional S and M results have only one seed.
- The numerical margins over published S/M are only +0.25 and +0.64 pp.
- The strongest modern parameterizations have not been combined with U2.
- The method does not transfer convincingly to CIFAR-100.
- Physical FPGA/ASIC area, timing and energy have not been measured.
- U2’s simplified circuit was modestly larger despite identical declared gate counts.
- The original ancestry narrative is contradicted by the ablation.
- Repeated development on MNIST/CIFAR test benchmarks creates a research-overfitting concern, even though the final checkpoints were validation-selected and tested once.
- Several comparisons mix reported results, adapted implementations and exact reproductions, although our tables now label those distinctions correctly.

My assessment is:

- Publishable research core: yes.
- Ready for a strong DATE submission today: not yet.
- Likely strong after a small number of decisive experiments: yes.

## 7. Where we stand against current state of the art

We are not currently the overall accuracy or hardware-efficiency state of the art.

LILogicNet reports 60.98% on CIFAR-10 with 256K gates using learned Top-K connectivity. In our direct reproductions it remained more accurate than U2, although it required roughly five times more training parameters, 16–17 times more peak training memory and substantially slower inference. [LILogicNet](https://arxiv.org/abs/2511.12340)

The current July 2026 BitLogic v2 comparison deliberately uses two-layer networks and separates encoder, connectivity, fan-in, node parameterization and head into independent axes. Its best-of-space model reports 58.06% on CIFAR-10 and 18.82% on CIFAR-100 at its largest evaluated width. Crucially, BitLogic identifies connectivity as an independent design axis and fan-in as the dominant hardware knob. This creates an excellent protocol for evaluating U2 directly. [BitLogic v2](https://arxiv.org/html/2602.07400v2)

Modern training methods also address limitations that topology cannot:

- Light DLGNs reduce parameter storage, accelerate backward propagation and remain more stable on CIFAR-100. [Light DLGN](https://arxiv.org/abs/2510.03250)
- Mind the Gap reports 4.5× faster training, a 98% reduction in the discretization gap and elimination of unused gates using Gumbel straight-through training. [Mind the Gap](https://arxiv.org/abs/2506.07500)
- WARP-LUT offers faster convergence and fewer training parameters while retaining comparable accuracy. [WARP-LUT](https://arxiv.org/abs/2510.15655)

The important point is that these methods modify parameterization or optimization, while U2 modifies connectivity. They should be complementary.

## 8. The recommended path to a strong DATE paper

### Priority 1: Freeze the method

Do not continue modifying U2 per dataset. Preserve V3, V4 and all negative results.

The paper’s only headline method should be:

> CoverageDLGN-U2: deterministic semantic degree-balanced multiscale routing.

V3 should appear as a dense specialization or ablation, not as a second headline method. V4/V5/U1 should remain development ablations.

### Priority 2: Confirm the convolutional result statistically

Run:

- LogicTreeNet-S random versus U2 to five full seeds.
- LogicTreeNet-M random versus U2 to three full seeds.

Suggested promotion criteria:

- At least +1 pp mean hardened-test gain.
- Positive paired confidence interval.
- U2 wins at least four of five S seeds and all or most M seeds.

Without this, the convolutional claim remains vulnerable.

### Priority 3: Combine U2 with a modern gate-training method

Evaluate random versus U2 while holding everything else fixed under:

1. LightLUT/Light parameterization.
2. Mind-the-Gap/Gumbel training, or WARP where convolutional support is reliable.

The most informative cells are:

- Dense CIFAR-10 M.
- LogicTreeNet-S.
- One CIFAR-100 convolutional coordinate.

This determines whether U2 is genuinely orthogonal to modern optimization. It is also the best chance of fixing CIFAR-100 without corrupting the topology method.

### Priority 4: Add a rank-\(r\) U2 extension

BitLogic’s strongest configuration uses fan-in four, while current U2 is rank two. Generalize the same principle—not a new dataset-specific heuristic—to rank \(r\):

- balanced predecessor degree;
- semantically ordered sources;
- disjoint or minimally overlapping predecessor groups;
- deterministic multiscale group selection;
- zero learned routing.

Then insert it as the connectivity axis in the current BitLogic protocol while keeping its encoder, LightLUT parameterization, fan-in, head and training recipe unchanged.

A very strong result would be:

- Match or exceed BitLogic’s 58.06% CIFAR-10 result at the 128K-node coordinate.
- Remove learned routing parameters.
- Reduce training memory or time.
- Preserve or improve synthesized hardware cost.

This would be a genuinely current, protocol-matched state-of-the-art comparison.

### Priority 5: Target Pareto superiority, not only raw accuracy

Trying to beat the 86.29% large convolutional model directly would require enormous models and training time. The more realistic DATE contribution is a better Pareto point.

Targets could be:

- Match 60.38% LogicTreeNet-S accuracy with materially fewer convolutional operations.
- Match 71.01% LogicTreeNet-M after at least 30% tying/pruning.
- Match LILogicNet’s 60.98% under its 256K dense protocol with much lower training memory and no learned routing.
- Match BitLogic at synthesized NAND2-equivalent or FPGA-LUT cost rather than nominal “gate count.”

The existing dense frontier suggests compression is plausible.

### Priority 6: Add real hardware synthesis

For random and U2 S checkpoints, measure:

- Yosys/ABC NAND2-equivalent area.
- Vivado post-route LUT count.
- Fmax.
- Latency and throughput.
- Dynamic/static power.
- Energy per sample.
- Routing congestion or wire-length proxy.
- Bit-exact equivalence.

This is necessary because identical abstract gate count did not yield identical simplified circuits.

### Priority 7: Add untouched generalization

Once U2 and its training parameterization are frozen, evaluate an untouched dataset or protocol:

- Convolutional CIFAR-100 S.
- SVHN.
- CIFAR-10.1 as distribution-shift confirmation.

Do not retune U2 for the dataset. A successful frozen transfer would address the research-overfitting concern.

## Final recommendation

Continue with CoverageDLGN, but change the scientific claim.

The strongest defensible narrative is:

> Random DLGN wiring is not merely noisy; it creates avoidable degree imbalance and semantically poor information propagation. Maximum ancestry diversity is also insufficient and can be harmful. CoverageDLGN uses deterministic semantic, degree-balanced, multiscale routing to improve hardened accuracy and the accuracy–circuit-size Pareto frontier without learned routing. The same connectivity principle transfers from dense to convolutional DLGNs.

If the multi-seed convolutional gains survive, the gains remain under Light/Gumbel/WARP training, and synthesis shows a non-dominated accuracy–hardware point, this becomes a strong DATE contribution.

If those confirmations fail, the honest fallback is a narrower dense-DLGN paper centered on V3’s five-seed CIFAR-10 gains and 67–80% demonstrated gate reductions—not a universal dense/convolutional claim.
