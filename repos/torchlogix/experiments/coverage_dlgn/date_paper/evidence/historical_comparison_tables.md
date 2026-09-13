# Historical comparison catalog

This is the pre-existing September 6 comparison document, preserved for the broader dataset, topology, method, and cost inventory. Its numeric contents are historical; they were not all re-audited for this manuscript. The current paper evidence and full-S replications are in `paper_evidence.json` and `raw_results.md`. Use the manuscript and current evidence for its headline claims. Relative source links below point back to the original experiment records.

Source: `../PAPER_DATE_SEPT_6_w_reported.md` relative to the paper directory. SHA-256: `bd4f593ab9605af4c56ec5f2f10a072923259d14c8e16e5aaccad6814efe9ad4`.

---

# CoverageDLGN — DATE presentation evidence with reported values, September 6, 2026

The presentation centers on **one methodology: CoverageDLGN, with U2 as the unified dense/convolutional construction and V3 as its strongest dense specialization**.

The tables below cover completed **MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and CIFAR-10.1** evidence, including architecture variants, competing methods, resource costs, and unsuccessful topology ablations. This document preserves the September 6 presentation snapshot and excludes the ongoing unified-refinement experiments. It was saved on September 7, 2026, from the presentation prepared in the conversation; no new experiments were performed for this document. This version adds the paper-reported references already recorded in the project comparison ledgers, with local-PDF checks where needed. Achieved results retain the September 6 snapshot.

**Methodology — presentation-ready bullets**

- **Objective:** improve hardened classification accuracy at a fixed logic-gate budget by designing the network’s fixed connectivity before training.
- **Represent the network as a Boolean computation graph:** each rank-2 gate receives two predecessors; training learns the gate function while keeping the connection indices fixed.
- **Preserve input semantics:** account for channel, spatial position, and threshold ordering. When measuring semantic ancestry, multiple thermometer thresholds of the same pixel/channel represent one underlying source.
- **Construct structured connections at multiple scales:** combine nearby and more distant predecessors through deterministic matching stages.
- **Prioritize balanced predecessor use:** choose stages that minimize the resulting fan-out spread; handle incomplete stages using disjoint pairs involving less-used predecessors.
- **Use ancestry novelty as a secondary stage-selection criterion:** among equally balanced candidate stages, U2 favors less-overlapping predecessor ancestry, with deterministic local-to-global tie-breaking.
- **Apply the same principle to both architecture families:**
  - Dense networks: construct semantic first-layer connections and multiscale connections between subsequent layers.
  - Convolutional networks: construct channel pairings while preserving the architecture’s spatial receptive-field sampler; apply the fixed-routing construction to the classifier as well.
- **Keep topology construction offline:** U2 adds no trainable routing parameters, gates, or inference operators. Deployment stores ordinary fixed connection indices.
- **Train and harden using the existing DLGN pipeline:** optimize relaxed gate functions, select checkpoints by hardened validation accuracy, and export discrete Boolean functions for inference.
- **Treat V3 as a dense specialization:** it uses semantic ordering, structured butterfly pairing, and ancestry-based swaps. U2 replaces individual swaps with whole-stage selection.
- **Evaluate topology through matched comparisons:** hold architecture, encoding, gate count, gate parameterization, optimizer, training budget, split, and paired seeds fixed.
- **Measure accuracy and cost together:** report hardened test accuracy, validation ablations, paired uncertainty, gate count, training parameters, GPU memory, training time, construction overhead, and deployment measurements.
- **Interpret the mechanism conservatively:** native dense random routing already balances fan-out. The evidence supports useful structured connectivity; it does **not** establish that degree balancing alone, maximum ancestry, or U2’s novelty criterion causes the improvement.

Method sources: [frozen U2 protocol](../../SECOND_ROUND_PROTOCOL.md), [matching-stage implementation](../../../../src/torchlogix/topology.py), [current interpretation](../../FOURTH_ROUND_RESULTS.md).

For the slides, use these conventions:

- **T:** hardened held-out test accuracy; **V:** best hardened validation accuracy.
- Accuracy is in **%**; gains are **percentage points**, abbreviated **pp**.
- **±** denotes sample standard deviation across training seeds; confidence intervals are paired Student-t 95% intervals.
- **A / achieved:** measured locally. **OUR:** CoverageDLGN or our ablation; **REP:** locally reproduced baseline; **ADAPT:** adapted external method; **NEG:** unsuccessful local reproduction.
- **R / REPORTED:** published by a source paper, including that paper’s own reproductions of other methods; these are not our measurements. Reference codes link to the local PDFs and are explained in the reported-source ledger at the end.
- **†:** the paper reference differs in architecture, gate budget, fan-in, training, or selection protocol. An exact architecture match alone does not establish an exact training-protocol match. Reported values are contextual unless a matched comparison is explicitly established.
- Reported spreads retain the source paper’s statistic. Local seed counts, parameters, training times, GPU memory, and deployment costs apply only to the achieved result unless explicitly marked R. No paper GPU-memory or timing value is inferred from an accuracy result.
- A reported baseline value belongs to the named published method, never to V3/U2. **—** on the reported side means no applicable published value is recorded for that method/coordinate in the source ledgers.
- Time is recorded wall time **per run**, including the harness’s evaluation overhead. Memory is peak PyTorch GPU allocation, not total device occupancy. Historical timer scopes differ; construction measurements should not automatically be added to archived totals.
- Missing entries mean that the corresponding completed measurement is unavailable. They are not zeros.

**Table 1. Dataset and architecture coverage**

| Dataset | Completed architecture families | Completed evidence |
|---|---|---|
| **MNIST** | Six-layer dense models at 4K, 8K, 16K, 32K, 48K gates; two-/three-layer learned-routing adaptations at 48K; six-layer comparator controls; early small-model topology pilots | V3, U2, random, Mommen, LILogicNet, BitLogic, historical fixed-topology controls |
| **Fashion-MNIST** | Six-layer dense models at 8K, 16K, 32K, 48K, 64K gates; two-/three-layer learned-routing adaptations at 48K; six-layer comparator controls | V3, U2, random, Mommen, LILogicNet, BitLogic, hybrid/fraction ablations |
| **CIFAR-10** | Dense S/M/L and intermediate widths; 4-/8-/12-layer depth studies; LILogic and BitLogic protocol architectures; convolutional S/M with nine-channel input; legacy six-channel convolutional S/M | Main accuracy frontier, unified U2, learned-routing trade-offs, convolutional transfer, mechanism studies, WARP compatibility, deployment |
| **CIFAR-100** | Two-layer 8K/32K pilots; 384K models with 3/6/12/24 layers and alternative output allocations; six-layer 1.536M screen | V3/U2 transfer boundary, depth failure, class-head and width ablations |
| **CIFAR-10.1 v6** | Frozen CIFAR-10 dense-M and convolutional-S/M checkpoints | Distribution-shift test without adaptation |

There are **no completed convolutional MNIST or Fashion-MNIST results in this evidence set**.

**Table 2. Architecture definitions used in the comparisons**

| Family | Layer layout | Total gate functions | Important distinction |
|---|---|---:|---|
| MNIST/Fashion main dense | 6 × 8K | 48K | Random/V3 comparison |
| MNIST/Fashion compression | Six layers, approximately equal widths | 4K–64K | Exact totals; final widths respect classifier grouping |
| MNIST/Fashion shallow comparators | 2 × 24K or 3 × 16K | 48K | Same gate count, different depth |
| CIFAR-10 dense S | 4 × 12K | 48K | Three thresholds per RGB channel |
| CIFAR-10 intermediate dense | 4 × 32K / 64K / 96K | 128K / 256K / 384K | Width-scaling frontier |
| CIFAR-10 dense M | 4 × 128K | 512K | Three thresholds per RGB channel |
| CIFAR-10 dense L | 5 × 256K | 1.28M | Five thresholds per RGB channel |
| CIFAR-10 depth controls | 8 or 12 layers | 48K or 512K | Fixed total budget; widths change |
| LILogic M protocol | 1 × 64K | 64K | Seven thresholds per RGB channel |
| LILogic L protocol | 2 × 128K | 256K | Seven thresholds per RGB channel |
| BitLogic protocol ladder | 2 × 4K / 16K / 64K | 8K / 32K / 128K | Rank-2 random/U2 versus a separate rank-4 learned-routing configuration |
| LogicTreeNet-S | Four convolution blocks + three dense layers; k=32 | **83,552** | **874,496 gate applications/image** |
| LogicTreeNet-M | Same architectural ratios; k=256 | **668,416** | **6,995,968 gate applications/image** |

The nine-channel convolutional S/M models use three thermometer thresholds per RGB channel. Legacy six-channel models belong to a separate protocol. Gate functions, spatial gate applications, simplified circuit nodes, and physical hardware cells are different quantities.

Sources: [dense architectures](../../../../src/torchlogix/models/dense.py), [convolutional architecture audit](../../ARCHITECTURE_AUDIT.md).

**Table 3. MNIST and Fashion-MNIST: all main 48K-gate methods**

These are completed **test results**. Shallow learned-routing methods are architecture adaptations, so this table compares practical accuracy/resource choices rather than isolating topology alone.

| Dataset | Method / provenance | Layers | Rank | T accuracy | Seeds | Trainable parameters | Time, min | GPU, GiB | Reported T accuracy — REPORTED; source configuration |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | Random — REP | 6 | 2 | 97.090 ± 0.180 | 5 | 0.768M | 15.03 | 0.099 | 97.69 [D]; 6 × 8K |
| MNIST | **V3 — OUR** | 6 | 2 | **97.500 ± 0.099** | 5 | **0.768M** | **15.02** | **0.099** | — |
| MNIST | Mommen — ADAPT | 2 | 2 | 98.084 ± 0.066 | 5 | 2.304M | 35.92 | 1.002 | 98.14 [M] †; 3 × 4K = 12K |
| MNIST | LILogicNet — ADAPT | 3 | 2 | 98.124 ± 0.029 | 5 | 3.840M | 82.02 | 4.187 | 98.95 ± 0.09 [L] †; 2 × 16K = 32K |
| MNIST | BitLogic — ADAPT | 2 | 4 | **98.204 ± 0.042** | 5 | 3.840M | 89.18 | 3.557 | 97.84 ± 0.04 [B] †; 2 × 64K = 128K |
| Fashion | Random — REP | 6 | 2 | 86.308 ± 0.186 | 5 | 0.768M | 14.97 | 0.099 | 87.17 [M]; 6 × 8K |
| Fashion | **V3 — OUR** | 6 | 2 | **87.102 ± 0.357** | 5 | **0.768M** | **15.02** | **0.099** | — |
| Fashion | Mommen — ADAPT | 3 | 2 | 87.260 ± 0.282 | 3 | 1.536M | 24.04 | 0.498 | 87.16 [M] †; 2 × 4K = 8K |
| Fashion | LILogicNet — ADAPT | 2 | 2 | 88.437 ± 0.159 | 3 | 3.840M | 70.84 | 4.676 | 90.26 ± 0.11 [L] †; 2 × 32K = 64K |
| Fashion | BitLogic — ADAPT | 2 | 4 | **89.740 ± 0.243** | 3 | 3.840M | 88.94 | 3.557 | 89.16 ± 0.08 [B] †; 2 × 64K = 128K |

V3’s paired improvements over random are:

| Dataset | T gain | Paired 95% CI | Wins |
|---|---:|---:|---:|
| MNIST, 48K | **+0.410 pp** | [0.108, 0.712] | 5/5 |
| Fashion-MNIST, 48K | **+0.794 pp** | [0.471, 1.117] | 5/5 |

**Presentation message:** V3 improves both datasets at essentially unchanged measured training cost. The stronger shallow learned-routing configurations achieve higher accuracy but consume substantially more training memory and time.

Reported sources: Deep DLGN [D], Mommen [M], LILogicNet [L], and BitLogic [B]. The Fashion fixed-random 87.17% reference comes from Mommen’s experiment, not the original Deep DLGN MNIST table. The layer layouts and resource columns above describe the local models; the R cells state the paper’s own gate budgets.

Sources and raw seeds: [MNIST finals](../../summary/table1_mnist_final.json), [Fashion finals](../../summary/table1_fashion_final.json).

**Table 4. MNIST and Fashion-MNIST: six-layer comparator controls**

All methods use **6 × 8K gates**, with the same 200-epoch effort. These are **achieved validation results**. BitLogic still changes fan-in and input encoding. The paper test references are in Table 3; they are not substituted into these validation controls. No matching reported V result is recorded for these specific control configurations.

| Method / provenance | MNIST V | Seeds | MNIST min / GiB | Fashion V | Seeds | Fashion min / GiB |
|---|---:|---:|---:|---:|---:|---:|
| Random — REP | 97.157 ± 0.043 | 5 | 15.03 / 0.099 | 87.477 ± 0.183 | 5 | 14.97 / 0.099 |
| **V3 — OUR** | **97.403 ± 0.114** | 5 | **15.02 / 0.099** | **87.873 ± 0.271** | 5 | **15.02 / 0.099** |
| Mommen — ADAPT | 95.683 ± 0.404 | 3 | 51.90 / 0.809 | 87.400 ± 0.928 | 3 | 29.09 / 0.453 |
| LILogicNet — ADAPT | 95.717 ± 0.351 | 3 | 92.44 / 3.706 | 84.267 ± 1.636 | 3 | 92.05 / 3.706 |
| BitLogic rank-4 — NEG | 11.417 ± 0.000 | 3 | 112.75 / 3.527 | 10.867 ± 0.000 | 3 | 112.74 / 3.527 |

**Presentation message:** the ranking changes when depth is controlled. The six-layer BitLogic failure is a local transfer result; it does not invalidate its successful shallow configuration.

Source: [architecture-matched comparison table](../../DATE_TABLES.md).

**Table 5. Complete MNIST/Fashion compression ladders**

All rows are **V, three paired seeds**, six layers, random — REP versus V3 — OUR.

| Dataset | Gates | Random V | V3 V | Gain, pp | Paired 95% CI | Wins | Random / V3 min | GPU, GiB, both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 4K | 85.539 ± 0.495 | 86.067 ± 0.159 | +0.528 | [−1.087, 2.142] | 2/3 | 14.78 / 14.92 | 0.009 |
| MNIST | 8K | 91.461 ± 0.286 | 91.956 ± 0.113 | +0.494 | [−0.479, 1.467] | 3/3 | 15.06 / 15.09 | 0.017 |
| MNIST | 16K | 95.100 ± 0.161 | 95.478 ± 0.444 | +0.378 | [−0.925, 1.681] | 3/3 | 15.06 / 15.01 | 0.034 |
| MNIST | 32K | 96.694 ± 0.135 | 97.011 ± 0.129 | +0.317 | [−0.334, 0.967] | 3/3 | 15.09 / 14.97 | 0.066 |
| Fashion | 8K | 83.433 ± 0.148 | 83.644 ± 0.158 | **+0.211** | **[0.085, 0.338]** | 3/3 | 15.09 / 15.10 | 0.017 |
| Fashion | 16K | 86.194 ± 0.250 | 86.883 ± 0.192 | +0.689 | [−0.344, 1.722] | 3/3 | 15.09 / 14.96 | 0.034 |
| Fashion | 32K | 87.461 ± 0.234 | 87.778 ± 0.327 | +0.317 | [−1.071, 1.704] | 2/3 | 15.06 / 15.08 | 0.066 |
| Fashion | 64K | 87.333 ± 0.557 | 88.100 ± 0.200 | +0.767 | [−0.524, 2.057] | 3/3 | 15.09 / 14.89 | 0.132 |

V3 has a positive mean in **8/8 coordinates** and wins **22/24 seed pairs**. Most coordinate-level intervals remain inconclusive. Compression reduces memory strongly, but these small models show little wall-time reduction under the fixed training schedule.

Source: [complete compression table](../../DATE_TABLES.md).

**Table 6. Unified U2 across MNIST, Fashion-MNIST, and dense CIFAR-10**

These are **T results on the same three seeds, 0/1/2**. Random — REP; V3/U2 — OUR. This table avoids comparing U2’s three-seed cohort against a five-seed baseline mean.

| Dataset / architecture | Gates | Random T | V3 T | U2 T | U2−random, pp | Paired 95% CI | Reported random-baseline T — REPORTED |
|---|---:|---:|---:|---:|---:|---:|---|
| MNIST, six layers | 8K | 91.273 ± 0.217 | 91.907 ± 0.307 | **91.937 ± 0.137** | **+0.663** | [0.015, 1.311] | —; no exact 8K six-layer reference recorded |
| Fashion, six layers | 16K | 85.197 ± 0.261 | **85.913 ± 0.356** | 85.717 ± 0.453 | +0.520 | [−1.016, 2.056] | —; no exact 16K six-layer reference recorded |
| CIFAR-10 S | 48K | 48.913 ± 0.346 | **52.390 ± 0.295** | 52.097 ± 0.630 | **+3.183** | [0.772, 5.595] | 51.27 [D]; 48K |
| CIFAR-10 M | 512K | 54.097 ± 0.171 | 58.357 ± 0.253 | **58.653 ± 0.168** | **+4.557** | [3.781, 5.332] | 57.39 [D]; 512K |
| CIFAR-10 L | 1.28M | 55.870 ± 0.308 | **61.073 ± 0.463** | 60.463 ± 0.348 | **+4.593** | [3.721, 5.466] | 60.78 [D]; 1.28M |

U2 wins **3/3 pairs against random in each row**. Against V3, U2 is +0.297 pp at M with an interval crossing zero, and **−0.610 pp at L**, CI [−0.983, −0.237].

| Coordinate | Random / V3 / U2 training, min | GPU, GiB, all methods | U2 construction, s |
|---|---:|---:|---:|
| MNIST 8K | 15.06 / 15.09 / 14.62 | 0.017 | 0.100 |
| Fashion 16K | 15.09 / 14.96 / 14.59 | 0.034 | 0.231 |
| CIFAR-10 S | 12.19 / 12.24 / 12.11 | 0.106 | 0.916 |
| CIFAR-10 M | Approximately 41–42 each | 1.123 | 14.333 |
| CIFAR-10 L | Approximately 113–114 each | 2.717 | 31.218 |

Sources and raw seeds: [compressed and S U2 finals](../../summary/second_round_final_dense.json), [M/L paired comparisons](../../summary/third_round_results.json).

**Table 7. Complete dense CIFAR-10 V3 accuracy–cost frontier**

These are **T results**, random — REP versus V3 — OUR.

| Architecture | Gates | Seeds | Random T | V3 T | Gain, pp | Paired 95% CI | Random / V3 min | GPU, GiB, both | Reported random-baseline T — REPORTED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 × 12K | 48K | 5 | 49.056 ± 0.356 | **52.358 ± 0.282** | +3.302 | [2.767, 3.837] | 12.24 / 12.26 | 0.106 | 51.27 [D] |
| 4 × 32K | 128K | 5 | 49.748 ± 0.141 | **53.910 ± 0.282** | +4.162 | [3.759, 4.565] | 12.44 / 12.44 | 0.281 | — |
| 4 × 64K | 256K | 3 | 52.253 ± 0.058 | **56.903 ± 0.134** | +4.650 | [4.174, 5.126] | 19.14 / 19.17 | 0.562 | — |
| 4 × 96K | 384K | 3 | 53.657 ± 0.328 | **58.143 ± 0.153** | +4.487 | [3.515, 5.458] | 29.83 / 29.82 | 0.845 | — |
| 4 × 128K | 512K | 5 | 54.028 ± 0.160 | **58.284 ± 0.259** | +4.256 | [3.851, 4.661] | 41.55 / 41.48 | 1.123 | 57.39 [D] |
| 5 × 256K | 1.28M | 5 | 55.960 ± 0.251 | **61.020 ± 0.336** | +5.060 | [4.555, 5.565] | 113.47 / 113.22 | 2.717 | 60.78 [D] |

**Presentation message:** every evaluated fixed-budget CIFAR-10 frontier point improves over the achieved random baseline, with positive paired confidence intervals. The reported S/M/L values [D] are separate paper baselines; no reported V3 result exists, and no exact reported intermediate-width reference is recorded here.

Sources: [48K results](../../summary/paper_cifar10_semantic_v3.json), [128K finals](../../summary/table2_cifar10_compression_crossing_final.json), [256K/384K tests](../../summary/table2_cifar10_compression_remaining_test.json), [512K results](../../summary/followup_summary.json), [1.28M finals](../../summary/table2_l_final.json).

**Table 8. Observed cross-budget trade-offs**

Savings below are calculated from the completed measurements.

| Smaller V3 model — OUR | Larger random model — REP | T accuracy difference | Fewer gates | Lower peak GPU memory | Lower recorded training time |
|---|---|---:|---:|---:|---:|
| **128K: 53.910%** | 384K: 53.657% | +0.253 pp | **66.7%** | **66.7%** | **58.3%** |
| **256K: 56.903%** | 1.28M: 55.960% | +0.943 pp | **80.0%** | **79.3%** | **83.1%** |

These are **observed mean-accuracy crossings among the tested models**, not formal accuracy-equivalence tests. The 1.28M model also changes depth and encoding, so that row is an architecture-frontier comparison.

**Table 9. CIFAR-10: Mommen and LILogicNet trade-offs**

All accuracies are **T**. The protocol column is essential: the LILogic architectures differ from the standard dense S/M/L family.

| Protocol | Method / provenance | Gates | Seeds | T accuracy | Total / routing parameters | Time, min | GPU, GiB | Reported T accuracy — REPORTED; match note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Standard dense S | Mommen — ADAPT | 48K | 3 | 50.950 ± 0.244 | 1.536M / 0.768M | 26.45 | 0.480 | — |
| Standard dense S | LILogicNet — ADAPT | 48K | 3 | 50.743 ± 0.574 | 3.840M / 3.072M | 86.87 | 3.949 | 55.11 [L] †; 8K fully learned 1L, not local 48K Top-32 |
| Standard dense S | **V3 — OUR** | 48K | 5 | **52.358 ± 0.282** | **0.768M / 0** | **12.26** | **0.106** | — |
| Standard dense M | Mommen — ADAPT | 512K | 1 | 54.420 | 16.384M / 8.192M | 290.4 | 4.918 | — |
| Standard dense L | Mommen — ADAPT | 1.28M | 1 | 54.340 | 40.960M / 20.480M | 803.4 | 11.904 | — |
| LILogic M, 1 × 64K | Random — REP | 64K | 3 | 49.010 ± 0.426 | 1.024M / 0 | 14.62 | 0.474 | 49.17 [L]; fixed 1F, 1 × 64K |
| LILogic M, 1 × 64K | **U2 — OUR** | 64K | 3 | **52.543 ± 0.296** | **1.024M / 0** | **14.59** | **0.474** | — |
| LILogic M, 1 × 64K | Top-32 — REP | 64K | 1 | **57.840** | 5.120M / 4.096M | 37.81 | 8.100 | 57.28 ± 0.30 [L]; 1Top32, 1 × 64K |
| LILogic L, 2 × 128K | Random — REP | 256K | 3 | 55.333 ± 0.469 | 4.096M / 0 | 15.20 | 1.557 | 54.76 [L]; fixed 2F, 2 × 128K |
| LILogic L, 2 × 128K | **U2 — OUR** | 256K | 3 | **60.193 ± 0.286** | **4.096M / 0** | **18.47** | **1.557** | — |
| LILogic L, 2 × 128K | Top-32 — REP | 256K | 1 | **62.030** | 20.480M / 16.384M | 350.64 | 24.861 | 60.98 ± 0.19 [L]; 2Top32, 2 × 128K |

Within the LILogic protocols, U2 improves random by **+3.533 pp at M**, CI [1.797, 5.270], and **+4.860 pp at L**, CI [3.083, 6.637], winning 3/3 pairs at both scales.

Relative to the local Top-32 measurements:

| U2 trade-off | M protocol | L protocol |
|---|---:|---:|
| Accuracy below Top-32 | 5.297 pp | 1.837 pp |
| Fewer training parameters | 80% | 80% |
| Peak-memory ratio, Top-32/U2 | 17.09× | 15.97× |
| Recorded time ratio, Top-32/U2 | 2.59× | 18.98× |

These resource ratios are descriptive and compare achieved measurements; Top-32 has one local seed. In the paper, the 64K fully learned **1L** model reports **57.66 ± 0.17% T [L]**, while the corresponding **1Top32** model reports **57.28 ± 0.30% T [L]**. Only the latter is the appropriate reported routing configuration for the local Top-32 row.

Sources: [standard comparator tables](../../DATE_TABLES.md), [LILogic raw results](../../summary/third_round_results.json).

**Table 10. CIFAR-10: complete BitLogic-protocol transfer ladder**

All rows use **two seeds and T accuracy**. Rank-2 random — REP; U2 — OUR; rank-4 learned-16 — **NEG**, because the local reproduction did not recover the expected published performance.

| Gates | Method | Rank | T accuracy | Training parameters | Time, min | GPU, GiB | Reported T accuracy — REPORTED |
|---|---|---:|---:|---:|---:|---:|---|
| 8K | Random | 2 | 26.175 ± 0.445 | 0.128M | 8.50 | 0.028 | —; see Table 10a |
| 8K | U2 | 2 | **28.435 ± 1.648** | 0.128M | 8.47 | 0.028 | — |
| 8K | Learned-16 | 4 | 27.500 ± 0.113 | 0.640M | 9.92 | 0.611 | 38.93 ± 0.19 [B] |
| 32K | Random | 2 | 25.945 ± 0.870 | 0.512M | 8.58 | 0.104 | —; see Table 10a |
| 32K | U2 | 2 | 26.040 ± 0.014 | 0.512M | 8.64 | 0.104 | — |
| 32K | Learned-16 | 4 | 16.625 ± 0.078 | 2.560M | 41.21 | 2.372 | 49.22 ± 0.26 [B] |
| 128K | Random | 2 | 25.160 ± 0.156 | 2.048M | 8.52 | 0.414 | —; see Table 10a |
| 128K | U2 | 2 | 25.930 ± 2.871 | 2.048M | 8.56 | 0.414 | — |
| 128K | Learned-16 | 4 | 13.160 ± 0.240 | 10.240M | 164.47 | 9.456 | 58.06 ± 0.14 [B] |

U2’s best-checkpoint gains are +2.260/+0.095/+0.770 pp; all three paired intervals cross zero. These results belong in the appendix as **protocol-transfer limitations**, not as evidence of superiority over BitLogic.

Source: [third-round results and raw seeds](../../summary/third_round_results.json).

**Table 10a. Reported CIFAR-10 common-protocol ladder**

All accuracy entries below are **REPORTED T**, from BitLogic Table 6 [B]. These are that paper’s retrainings of the named methods on a common two-layer backbone, not the original Deep DLGN S/M/L results. The failed local rank-4 results remain alongside the corresponding reported BitLogic values in Table 10.

| Backbone | Total gates | DiffLogic R | LILogicNet R | WARP-LUT R | BitLogic rank-4 R | Achieved evidence |
|---|---:|---:|---:|---:|---:|---|
| 2 × 4K | 8K | 33.72 ± 0.16 | 33.83 ± 0.13 | 33.86 ± 0.10 | 38.93 ± 0.19 | Table 10: local rank-2 random/U2 and rank-4 learned-16 |
| 2 × 16K | 32K | 42.55 ± 0.02 | 42.36 ± 0.20 | 42.92 ± 0.29 | 49.22 ± 0.26 | Table 10: same local method families |
| 2 × 64K | 128K | 51.73 ± 0.34 | 51.67 ± 0.09 | 52.12 ± 0.01 | 58.06 ± 0.14 | Table 10: same local method families |

The repository’s [third-round protocol](../../THIRD_ROUND_PROTOCOL.md) records a fan-in inconsistency between BitLogic’s shared-protocol prose and its method-mapping table. Local fixed random/U2 use rank 2; do not describe their comparison with the reported DiffLogic row as an exact numerical reproduction. No completed local WARP-LUT or LILogicNet measurement exists at these specific common-protocol coordinates. The actual adapted WARP experiment in Table 16 uses different architectures/budgets.

**Table 11. CIFAR-10: complete full-schedule convolutional results**

Nine-channel architectures. Random — REP; V4/U1/U2 — OUR. **Every row has one training seed.**

| Architecture | Updates | Method | V accuracy | T accuracy | T gain over random | Time, h | GPU, GiB | Reported T accuracy for this method — REPORTED |
|---|---:|---|---:|---:|---:|---:|---:|---|
| LogicTreeNet-S | 350K | Random | 58.680 | 57.370 | — | 4.975 | 1.831 | 60.38 [C] |
| LogicTreeNet-S | 350K | V4 | 59.860 | 58.930 | +1.560 pp | 4.957 | 1.831 | — |
| LogicTreeNet-S | 350K | U1 | 59.880 | 58.800 | +1.430 pp | 4.980 | 1.831 | — |
| LogicTreeNet-S | 350K | **U2** | **61.000** | **60.630** | **+3.260 pp** | **4.951** | **1.831** | — |
| LogicTreeNet-M | 200K | Random | 70.680 | 69.570 | — | 25.47 | 14.615 | 71.01 [C] |
| LogicTreeNet-M | 200K | V4 | 71.260 | 69.960 | +0.390 pp | ≈25.49 | Not recorded | — |
| LogicTreeNet-M | 200K | **U2** | **72.380** | **71.650** | **+2.080 pp** | **35.686** | **14.614** | — |

At each scale, random/U2 preserve the gate-function count, trainable gate parameters, and nominal inference operations. S has 1.337M raw training parameters; M has 10.695M.

**The M timing must remain visible:** U2’s recorded wall time is approximately **40.1% longer**. Different execution conditions prevent attributing that difference to topology alone, but the completed evidence does not support a blanket “identical training time” claim.

U2 is numerically **+0.250 pp above reported LogicTreeNet-S** and **+0.640 pp above reported LogicTreeNet-M** [C]. Those cross-paper differences are descriptive: the local full runs have one seed and use the local validation-selection protocol. They are not paired superiority or state-of-the-art claims.

Sources: [full S results](../../summary/second_round_convolutional_final.json), [completed M protocol and test](../../LOGICTREENET_M_U2_PROTOCOL.md).

**Table 11a. Additional reported convolutional CIFAR-10 references**

| Method / paper | Architecture or reported cost | Achieved A for this method | Reported R accuracy | Comparison scope |
|---|---|---|---|---|
| Two-stage unit tying, untied checkpoint [U] | LogicTreeNet-S | — | 58.39 V | Paper’s own pre-tying checkpoint; not the 60.38 T original reference |
| Two-stage unit tying, 30% + fine-tuning [U] | S; approximately 70% of source gates retained | — | 56.70 ± 0.08 V | Different gate budget and validation metric |
| Two-stage unit tying, untied checkpoint [U] | LogicTreeNet-M | — | 71.57 V | Paper’s own pre-tying checkpoint |
| Two-stage unit tying, 30% + fine-tuning [U] | M; approximately 70% of source gates retained | — | 70.77 ± 0.07 V | Different gate budget and validation metric |
| Conv. TTNet-S [C] | 0.57M reported gates | — | 50.10 T | Different truth-table architecture; quoted in Convolutional DLGN Table 1 |
| Scalability CDLGN-M [S] | M-derived; approximately 3.08M reported operations | — | 65.23 T | Modified M protocol |
| Conv. TTNet-L [C] | 189M reported gates | — | 70.75 T | Much larger, different architecture; quoted in Convolutional DLGN Table 1 |
| LogicTreeNet-B [C] | 16.0M reported operations | — | 80.17 T | Teacher supervision and fixed feature preprocessing; no faithful local result |
| LogicTreeNet-L [C] | Approximately 28.9M reported operations | — | 84.99 T | Teacher supervision and fixed feature preprocessing; no faithful local result |
| LogicTreeNet-G [C] | Larger teacher-supervised architecture | — | 86.29 T | Context only; no local G result |

Every populated R cell is **REPORTED**, not locally achieved. No applicable paper-reported value is recorded for U2, V3/V4/U1, or the local adapted nine-channel WARP/Light variants. Published operation counts retain the source convention and must not be substituted for the local LUT-function/application counts.

**Table 12. CIFAR-10 depth studies: all completed 4-/8-/12-layer coordinates**

These are **V results at 20K updates, three paired seeds**. Random — REP; V3 — OUR.

| Gates | Layers | Random V | V3 V | Gain, pp | Paired 95% CI | Random / V3 min | GPU, GiB, both |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48K | 4 | 49.040 | 52.193 | +3.153 | [0.985, 5.322] | ≈2.3 / ≈2.3 | 0.106 |
| 48K | 8 | 44.907 ± 0.293 | 47.553 ± 0.982 | +2.647 | [0.013, 5.280] | 3.49 / 3.52 | 0.098 |
| 48K | 12 | 14.087 ± 5.386 | 12.367 ± 1.230 | −1.720 | [−15.325, 11.885] | 4.76 / 4.74 | 0.097 |
| 512K | 4 | 54.820 ± 0.530 | 59.293 ± 0.214 | +4.473 | [3.624, 5.323] | 7.69 / 7.70 | 1.123 |
| 512K | 8 | 48.413 ± 0.522 | 52.007 ± 0.070 | +3.593 | [2.151, 5.035] | 7.71 / 7.71 | 1.028 |
| 512K | 12 | 11.807 ± 0.420 | 11.260 ± 0.941 | −0.547 | [−3.873, 2.779] | 8.32 / 8.33 | 0.996 |

**Presentation message:** structured connectivity helps at four and eight layers, but does not rescue the severe optimization failure at twelve layers.

Source: [depth comparisons](../../summary/followup_summary.json); timings were aggregated read-only from the corresponding completed run summaries.

**Table 13. Which parts of V3 matter?**

Dense CIFAR-10 M, **V, 20K updates, three paired seeds**.

| Construction | V accuracy | Incremental gain | Paired 95% CI | Construction, s |
|---|---:|---:|---:|---:|
| Random — REP | 54.820 ± 0.530 | — | — | 5.46 |
| Structured balanced butterfly — OUR | 58.980 ± 0.548 | **+4.160 pp** | **[3.988, 4.332]** | 5.54 |
| + semantic first layer — OUR | 59.253 ± 0.153 | +0.273 pp | [−0.780, 1.326] | 12.25 |
| + ancestry swaps: V3 — OUR | 59.293 ± 0.214 | +0.040 pp | [−0.434, 0.514] | 107.87 |

All arms use approximately 7.7 training minutes and 1.123 GiB GPU memory.

The butterfly arm recovers approximately **93% of V3’s observed gain**. Because the random baseline already balances fan-out, this supports the changed pairing structure; it does not isolate a degree-balance effect.

Source: [V3 component ablation](../../summary/cifar10_medium_v3_components.json).

**Table 14. U2 mechanism controls across dense and convolutional models**

All entries are **V, 20K updates, three seeds**. Random — REP; other constructions — OUR.

| Wiring | Dense-M V | Conv-S V | Dense construction, s | Conv construction, s |
|---|---:|---:|---:|---:|
| Explicit random | 54.820 ± 0.530 | 57.140 ± 1.870 | 5.461 | 0.201 |
| Semantic balanced-random | 58.647 ± 0.200 | 58.633 ± 0.273 | 11.582 | 0.968 |
| Semantic nominal multiscale | **59.220 ± 0.420** | 58.253 ± 0.911 | 13.648 | 1.120 |
| **U2** | 59.133 ± 0.316 | **58.847 ± 0.600** | 14.110 | 1.380 |

| U2 contrast | Dense-M gain and 95% CI | Conv-S gain and 95% CI |
|---|---:|---:|
| Versus explicit random | **+4.313 [2.475, 6.152]** | +1.707 [−2.600, 6.013] |
| Versus balanced-random | +0.487 [−0.795, 1.768] | +0.213 [−0.836, 1.263] |
| Versus nominal multiscale | −0.087 [−1.759, 1.585] | +0.593 [−1.121, 2.308] |

Dense arms use 1.123 GiB and approximately 7.7 minutes; convolutional arms use 1.831 GiB and approximately 17 minutes.

**The added benefit of ancestry-based stage selection is not established.** Intervals crossing zero also do not establish equivalence.

Sources: [dense mechanism](../../summary/fourth_round_dense_mechanism.json), [convolutional mechanism](../../summary/fourth_round_conv_mechanism.json).

**Table 15. Convolutional body versus classifier contribution**

LogicTreeNet-S, **V, 20K updates, seeds 0/1/2**.

| Body / classifier | Raw seed accuracies | Mean ± SD | Provenance |
|---|---|---:|---|
| Random / random | 55.000, 57.960, 58.460 | 57.140 ± 1.870 | REP |
| U2 / random | 58.240, 58.180, 58.620 | 58.347 ± 0.239 | OUR |
| Random / U2 | 56.960, 56.820, 56.340 | 56.707 ± 0.325 | OUR |
| U2 / U2 | 58.500, 59.540, 58.500 | **58.847 ± 0.600** | OUR |

With the reference-U2 classifier fixed, switching the body to U2 gains **+2.140 pp**, CI **[0.674, 3.606]**, with 3/3 wins. The body effect with a random classifier, both classifier effects, and the interaction remain inconclusive.

Source: [factorial results and wiring audit](../../summary/fourth_round_factorial.json).

**Table 16. U2 under an alternative gate parameterization**

All accuracies are **achieved V, three seeds**. Raw random — REP; raw U2 — OUR; WARP arms — ADAPT. No matching paper-reported accuracy is recorded for this adapted parameterization experiment. Reported WARP-LUT test values use the separate two-layer ladder in Table 10a; reported WARP Figure 4 validation curves use the six-channel protocol in Table 20.

| Architecture / budget | Parameterization | Wiring | Best V | Final V | Parameters | GPU, GiB | Time, min |
|---|---|---|---:|---:|---:|---:|---:|
| Dense M, 108K updates | Raw | Random | 55.160 | — | 8.192M | 1.123 | ≈41.5 |
| Dense M, 108K updates | Raw | U2 | 59.547 | — | 8.192M | 1.123 | 41.39 |
| Dense M, 108K updates | WARP | Random | 53.247 ± 0.378 | 50.773 ± 0.162 | 2.048M | 1.235 | 41.83 |
| Dense M, 108K updates | WARP | U2 | **57.947 ± 0.167** | **55.467 ± 0.099** | 2.048M | 1.235 | 41.67 |
| Conv S, 20K updates | Raw | Random | 57.140 ± 1.870 | 56.307 ± 1.341 | 1.337M | 1.831 | 17.00 |
| Conv S, 20K updates | Raw | U2 | 58.847 ± 0.600 | 58.813 ± 0.631 | 1.337M | 1.831 | 17.02 |
| Conv S, 20K updates | WARP | Random | 44.320 ± 0.904 | 37.500 ± 1.393 | 0.334M | 2.176 | 17.12 |
| Conv S, 20K updates | WARP | U2 | 44.407 ± 1.312 | 36.287 ± 5.566 | 0.334M | 2.176 | 17.15 |

- Dense U2-minus-random under WARP: **+4.700 pp**, CI **[3.919, 5.481]**, 3/3 wins.
- Convolutional effect under WARP: **+0.087 pp**, CI **[−1.243, 1.417]**; final-checkpoint effect is −1.213 pp.
- WARP uses fewer parameters but **more measured peak memory** here. Its parameter savings belong to WARP; the topology contribution is the within-parameterization comparison.

Sources: [dense WARP](../../summary/fourth_round_warp_dense.json), [convolutional WARP](../../summary/fourth_round_warp_conv.json).

**Table 17. Additional datasets: CIFAR-100**

Random — REP; V3/U2/refinements — OUR.

| Architecture | Gates | Scope / seeds | Random | V3 or stated variant | Gain | Random / variant min | GPU, GiB | Reported reference R — REPORTED; method / scope |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 2 × 4K | 8K | V, 20K / 3 | 8.780 ± 0.381 | V3: 7.940 ± 0.302 | −0.840 pp | — | — | 7.49 ± 0.21 T [B] †; DiffLogic full protocol, not local 20K V |
| 2 × 16K | 32K | V, 20K / 3 | 10.060 ± 0.220 | V3: 9.707 ± 0.076 | −0.353 pp | — | — | 10.61 ± 0.08 T [B] †; DiffLogic full protocol, not local 20K V |
| 3 × 128K | 384K | V, 20K / 3 | 21.093 ± 0.101 | V3: **21.933 ± 0.110** | **+0.840 pp** | 5.35 / 5.34 | 0.890 | — |
| 3 × 128K | 384K | T / 3 | 20.923 ± 0.352 | V3: **21.467 ± 0.410** | +0.543 pp | Same cohort | 0.890 | — |
| 3 × 128K | 384K | V, 20K / 3 | 21.093 ± 0.101 | U2: 21.193 | +0.100 pp | 5.35 / 5.34 | 0.890 | — |
| 6 × 64K | 384K | T / 3 | 20.677 ± 0.522 | V3: 21.010 ± 0.131 | +0.333 pp | 11.37 / 11.49 | 0.795 | 22.54 ± 0.26 T [S]; random baseline only |
| 96K + 96K + 192K | 384K | V, 20K / 1 | 19.960 | V3: 20.620 | +0.660 pp | 5.54 / 5.52 | 0.966 | — |
| 64K + 64K + 256K | 384K | V, 20K / 1 | 20.380 | V3: 20.240 | −0.140 pp | 5.97 / 5.95 | 1.060 | — |
| 12 × 32K | 384K | V, 20K / 1 | 1.180 | V3: 1.180 | 0.000 pp | 6.82 / 6.85 | 0.750 | — |
| 24 × 16K | 384K | V, 20K / 1 | 1.200 | V3: 1.200 | 0.000 pp | 9.55 / 9.57 | 0.723 | — |
| 6 × 256K | 1.536M | V, 5K screen / 1 | 11.680 | Best screened V3: 11.060 | −0.620 pp | 32.03 / 31.59 | 14.181 | 27.92 ± 0.43 T [P] †; Soft-Mix, full 100K schedule |

The 3-layer V3 test gain has CI **[−0.141, 1.227]**; the 6-layer test gain has CI **[−0.816, 1.483]**. U2’s pilot CI is **[−0.473, 0.673]**, and it was not promoted.

The class-conditional head reaches **21.593% V**, versus **21.580% V** for V3, despite improving its source-balance diagnostic.

Sources: [CIFAR-100 completed tables](../../PAPER_COMPARISON_TABLES.md), [U2 pilot](../../summary/second_round_u2_pilot.json).

**Table 17a. Reported dense CIFAR-100 common-protocol ladder**

All R values are **REPORTED T** from BitLogic Table 6 [B]. The local A entries are the completed **20K-update V pilots**, not full-protocol tests. Published rank assignments follow the repository’s existing comparison ledger and the paper’s method-mapping table; the fan-in caveat in Table 10a also applies.

| Backbone / total gates | Method | Rank | Achieved A | Reported R |
|---|---|---:|---|---:|
| 2 × 4K / 8K | DiffLogic | 2 | Random pilot: 8.780 ± 0.381 V, n=3, REP † | 7.49 ± 0.21 T |
| 2 × 4K / 8K | WARP-LUT | 2 | — | 7.00 ± 0.04 T |
| 2 × 4K / 8K | LILogicNet | 2 | — | 7.63 ± 0.01 T |
| 2 × 4K / 8K | BitLogic best-of-space | 4 | — | 10.19 ± 0.06 T |
| 2 × 16K / 32K | DiffLogic | 2 | Random pilot: 10.060 ± 0.220 V, n=3, REP † | 10.61 ± 0.08 T |
| 2 × 16K / 32K | WARP-LUT | 2 | — | 10.46 ± 0.00 T |
| 2 × 16K / 32K | LILogicNet | 2 | — | 10.62 ± 0.12 T |
| 2 × 16K / 32K | BitLogic best-of-space | 4 | — | 14.06 ± 0.04 T |
| 2 × 64K / 128K | DiffLogic | 2 | — | 14.64 ± 0.09 T |
| 2 × 64K / 128K | WARP-LUT | 2 | — | 14.43 T; one reported seed |
| 2 × 64K / 128K | LILogicNet | 2 | — | 14.54 ± 0.04 T |
| 2 × 64K / 128K | BitLogic best-of-space | 4 | — | 18.82 ± 0.09 T |

CoverageDLGN has no R values on this ladder. Its achieved S/M pilots are retained in Table 17; no local L result was promoted.

**Table 17b. Additional reported dense CIFAR-100 references**

| Method / source | Architecture / gates | Achieved A | Reported R — REPORTED | Match note |
|---|---|---|---|---|
| Scalability dense DLGN [S] | 6 × 64K / 384K | 20.677 ± 0.522 T, n=3, REP | 22.54 ± 0.26 T | Exact architecture; local routing and checkpoint-selection details differ |
| Multilinear Soft-Mix [P] | 6 × 256K / 1.536M | Random screen: 11.680 V at 5K, n=1, REP | 27.92 ± 0.43 T | Paper uses full 100K training and last-10-checkpoint accuracy; not comparable to the local screen |
| Multilinear-CovJac [P] | 6 × 256K / 1.536M | — | 28.37 ± 0.22 T | Four training parameters/gate; paper’s last-10-checkpoint metric |
| Multilinear-CovJac large [P] | 6 × 1.28M / 7.68M | — | 32.72 ± 0.09 T | Different budget; paper’s last-10-checkpoint metric |

**Table 17c. Reported convolutional CIFAR-100 context**

| Method / source | Actual reported architecture | Achieved A | Reported R — REPORTED | Match note |
|---|---|---|---|---|
| Original/Soft-Mix [I] | Fivefold-depth M-derived convolutional model | — | 29.0 ± 0.6 T | Not base LogicTreeNet-M |
| Light/IWP [I] | Same fivefold-depth M-derived model | — | 38.2 ± 0.3 T | Different parameterization; no local matching run |
| Scalability CDLGN [S] | Modified M-derived model | — | 30.96 T | Separate protocol |

No completed local convolutional CIFAR-100 result is included in this snapshot, and the existing ledgers provide no exact S/L paper reference for the planned local architectures.

**Table 18. Frozen distribution-shift evaluation: CIFAR-10.1**

These are **T results on 2,000 images**, with no fine-tuning, threshold recalibration, or checkpoint reselection.

| Architecture | Seeds | Random — REP | U2 — OUR | Gain | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Dense M | 3 | 41.967 ± 0.126 | **45.700 ± 0.250** | **+3.733 pp** | **[2.801, 4.666]** |
| Convolutional S | 1 | 46.000 | 45.950 | −0.050 pp | Not estimable |
| Convolutional M | 1 | 54.100 | **56.050** | +1.950 pp | Not estimable |

The dense gain survives this distribution shift. Convolutional transfer is mixed.

Source: [frozen transfer results and raw seeds](../../summary/fourth_round_transfer_results.json).

**Table 19. Remaining topology families and historical ablations**

This retains the unsuccessful constructions and earlier variants. Values belong to the specified cohort; they should not be mixed into the main test tables.

| Topology / variant — OUR unless stated | Dataset / protocol | Completed result | Interpretation |
|---|---|---|---|
| Native random — REP | MNIST, 1K-step pilot, seed 0, V | 83.14% | Historical control |
| Random-unique | Same pilot | 81.28% V | Below control |
| Local cyclic V1 | Same pilot | 38.71% V | Poor mixing |
| Unshuffled butterfly V1 | Same pilot | 42.78% V | Output-order failure |
| Pure greedy coverage | Same pilot | 81.76% V | Maximum ancestry did not improve accuracy |
| Hybrid V1, fraction 0.25 / 0.75 / 0.90 | Same pilot | 60.53 / 81.27 / 83.17% V | Strong sensitivity to regular-base fraction |
| Affine-ordered butterfly V2 | Same pilot | 83.08% V | Ordering repair recovers baseline-like performance |
| Hybrid V2, fraction 0.25 | Same pilot | 82.97% V | Near control |
| Hybrid V2 | MNIST, full 48K, 3 seeds | 97.253% T versus 97.183% random | +0.070 pp |
| Hybrid V2 | Fashion, full 48K, 3 seeds | 86.350% T versus 86.303% random | +0.047 pp |
| Hybrid V2 | CIFAR-10 S, full, 3 seeds | 50.670% T versus 48.913% random | +1.757 pp |
| V3 + task-aware rewiring | Dense CIFAR-10 M, 20K, 3 seeds | 59.093% V versus 59.293% V3 | −0.200 pp versus V3 |
| V4 | Nine-channel conv S, 20K, 5 seeds | 57.448% V versus 56.864% random | +0.584 pp; inconclusive |
| U1, no swaps | Same five-seed cohort | 57.624% V | +0.760 pp; CI [−0.700, 2.220]; 4/5 wins |
| Channel-spatial adapter | Conv S, 20K, 3 seeds | 57.033% V versus 56.673% random | +0.360 pp; not promoted |
| V5 ancestry-channel construction | Conv S, 20K, 3 seeds | 57.053% V versus 57.140% controlled random | −0.087 pp; not promoted |
| V4 + V3 classifier | Conv S, 5K, seed 0 | 53.940% V versus 52.520% random | Early classifier screen |
| V5, random classifier | Same 5K screen | 55.360% V | Early gain did not survive confirmation |
| V5 + V3 classifier | Same 5K screen | 55.160% V | No benefit from adding that classifier |
| Coverage/reuse refinement | Conv S, 5K, seed 0 | 55.400% V versus 54.920% V4 | +0.480 pp |
| Coverage/reuse refinement | Conv M, 5K confirmation, seed 0 | 58.440% V versus 61.560% V4 | −3.120 pp; not promoted |
| Class-conditional head | CIFAR-100, 6 × 64K, 20K, 3 seeds | 21.593% V versus 21.580% V3 | Diagnostic improvement without accuracy gain |

The semantic balanced-random, nominal-multiscale, and body/classifier combinations are covered in Tables 14–15. Hyperparameter screens within a topology family are grouped here rather than presented as independent methods.

Sources: [historical topology experiments](../../RESULTS.md), [task-aware results](../../summary/cifar10_medium_task_aware.json), [U1 five-seed results](../../summary/cifar10_conv_small_unified_five_seed.json), [coverage/reuse results](../../summary/paper_conv_coverage_reuse_screen.json).

**Table 20. Legacy six-channel convolutional evidence**

These architectures are separate from nine-channel LogicTreeNet-S/M.

| Protocol | Method / provenance | Scope / seeds | Accuracy | Time | GPU, GiB | Reported R for this method — REPORTED |
|---|---|---|---:|---:|---:|---|
| Legacy conv S, 20K | Random — REP | T / 3 | 55.420 | — | — | —; six-channel S is not reported nine-channel S |
| Legacy conv S, 20K | V4 — OUR | T / 3 | **57.420** | — | — | — |
| WARP-style legacy M, 30K | Matched random, uniform thresholds — REP | V / 1 | 64.580 | 3.87 h | 14.614 | —; matched sampler differs from public random-unique |
| WARP-style legacy M, 30K | V4, uniform thresholds — OUR | V / 1 | **66.230** | 3.85 h | 14.614 | — |
| WARP-style legacy M, 30K | Learnable thresholds — ADAPT | V / 1 | 65.880 | 4.29 h | 14.619 | ≈66.6 V [W]; 50K endpoint, three-seed curve † |

The legacy-S paired test gain is **+2.000 pp**, CI **[0.058, 3.942]**. The WARP-style M runs above use **raw LUT parameterization**; they are distinct from the actual U2+WARP parameterization study in Table 16. Interrupted uniform/distributive runs are excluded.

Additional **REPORTED-PLOT** references from WARP Figure 4 [W] are approximately **64.0% V** for fixed uniform thresholds and **65.0% V** for fixed distributive thresholds, both read from the 50K-step three-seed curves. Their local attempts were interrupted after the frozen 30K boundary and therefore have no eligible completed A entry in this presentation. The complete learnable-threshold A result above is best validation at 30K, whereas its R value is an approximate endpoint at 50K; do not subtract these as a matched gain.

Sources: [legacy convolutional results](../../summary/followup_summary.json), [WARP-style protocol results](../../summary/warp_fig4_cifar10_medium.json).

**Table 21. Construction and deployment trade-offs**

Representative offline construction measurements:

| Coordinate | Random construction | V3 / V4 construction | U2 construction |
|---|---:|---:|---:|
| MNIST 8K | 0.001 s | V3: 0.866 s | **0.100 s** |
| Fashion 16K | 0.005 s | V3: 3.358 s | **0.231 s** |
| Dense CIFAR-10 S | 0.045 s | V3: 9.313 s | **0.916 s** |
| Dense CIFAR-10 M | Approximately 5 s | V3: approximately 107 s | **14.333 s** |
| Dense CIFAR-10 L | Not recorded for cached final cohort | Not recorded for cached final cohort | 31.218 s |
| Full conv S | 0.217 s | V4: 0.449 s | 1.372 s |
| Full conv M | 2.478 s | V4: 6.182 s | 18.908 s |

These are recorded construction timers, sometimes summed over layers; they are not comprehensive end-to-end model-initialization benchmarks.

Full nine-channel convolutional-S deployment, **one checkpoint per method**:

| Method | T accuracy | Simplified IR nodes | Change vs random | CPU batch-128 latency | Peak export RAM |
|---|---:|---:|---:|---:|---:|
| Random — REP | 57.370 | 252,936 | — | 3.230 ms | 1.114 GiB |
| V4 — OUR | 58.930 | 241,262 | −4.62% | 3.163 ms | 1.115 GiB |
| U1 — OUR | 58.800 | 251,693 | −0.49% | 3.136 ms | 1.124 GiB |
| **U2 — OUR** | **60.630** | **262,260** | **+3.69%** | **3.185 ms** | **1.138 GiB** |

For random/U2, synthetic hardened CUDA batch-128 latency is **6.852/6.835 ms**. These snapshots support similar measured runtime, not an established speedup.

U2 has the same declared gate budget but a **larger simplified circuit** at this checkpoint. FPGA/ASIC area, frequency, power, and energy have not been established.

Sources: [dense resources](../../summary/third_round_results.json), [full S deployment](../../summary/second_round_convolutional_deployment.json).

**Table 22. Further reported dense references retained for context**

These R-only rows preserve additional references from [CIFAR10_BASELINE_REFERENCE.md](../../CIFAR10_BASELINE_REFERENCE.md), plus the corresponding MNIST scale from Deep DLGN. They are not achieved CoverageDLGN results or matched frontier points.

| Dataset | Method / source | Architecture / total gates | Achieved A at this coordinate | Reported R — REPORTED | Match note |
|---|---|---|---|---|---|
| MNIST | Deep DLGN [D] | 384K gates | — | 98.47 ± 0.05 T | Larger than the main 48K local model |
| CIFAR-10 | Deep DLGN large×2 [D] | 5 × 512K / 2.56M | — | 61.41 ± 0.02 T | Larger than the largest local frontier point |
| CIFAR-10 | Deep DLGN large×4 [D] | 5 × 1.024M / 5.12M | — | 62.14 ± 0.02 T | Larger than the largest local frontier point |
| CIFAR-10 | LILogicNet fully learned M [L] | 1 × 64K / 64K | — for 1L | 57.66 ± 0.17 T | Distinct from achieved Top-32, 57.840 T in Table 9 |
| CIFAR-10 | Multilinear Soft-Mix [P] | 6 × 128K / 768K | — | 58.13 ± 0.12 T | Last-10-checkpoint average; different depth/budget |
| CIFAR-10 | Multilinear-CovJac [P] | 6 × 128K / 768K | — | 58.97 ± 0.26 T | Last-10-checkpoint average; four parameters/gate |
| CIFAR-10 | Silicon-aware DLGN baseline [A] | Deep feed-forward DLGN | — | 60.07 accuracy | Separate hardware study; evaluation split is not specified in the imported ledger entry |

**Table 23. Reported training-time context**

These are **paper-reported training times**, not estimates of how long those experiments would take on our hardware. Deep DLGN Appendix A.3 explicitly says its times use the original experimental code and notes that a faster implementation would be released. Do not turn cross-paper time ratios into a CoverageDLGN speedup claim.

| Dataset / architecture | Achieved random T — REP | Reported T — REPORTED [D] | Achieved random training | Reported training [D] | Reported peak GPU memory |
|---|---:|---:|---:|---:|---|
| MNIST, 48K | 97.090 ± 0.180 | 97.69 ± 0.11 | 15.03 min | 1.8 h | Not recorded here |
| MNIST, 384K | — | 98.47 ± 0.05 | — | 5.3 h | Not recorded here |
| CIFAR-10 S, 48K | 49.056 ± 0.356 | 51.27 ± 0.26 | 12.24 min | 1.3 h | Not recorded here |
| CIFAR-10 M, 512K | 54.028 ± 0.160 | 57.39 ± 0.13 | 41.55 min | 7.4 h | Not recorded here |
| CIFAR-10 L, 1.28M | 55.960 ± 0.251 | 60.78 ± 0.12 | 113.47 min | 24.2 h | Not recorded here |
| CIFAR-10 large×2, 2.56M | — | 61.41 ± 0.02 | — | 45.6 h | Not recorded here |
| CIFAR-10 large×4, 5.12M | — | 62.14 ± 0.02 | — | 90.3 h | Not recorded here |

Source: Deep DLGN [D], Tables 8–9, PDF page 15. The main tables retain the point estimates used in the earlier comparison ledgers; this resource table includes the additional published standard deviations. All other time/memory columns in this document remain local A measurements.

For the main presentation, use the methodology followed by Tables **3, 6, 7, 8, 11, 13, and 21**. The remaining tables provide the complete comparison and ablation backup.

Additional useful slides, using the existing measurements:

| Visual | What it would show |
|---|---|
| Three accuracy-versus-gates panels: MNIST, Fashion, CIFAR-10 | Small consistent gains on grayscale datasets; much larger CIFAR-10 gains |
| Accuracy versus peak GPU memory | CoverageDLGN’s resource trade-off against learned routing |
| Accuracy versus recorded training time | The dense frontier and expensive comparator points |
| V3 component waterfall | Most observed dense gain appears before ancestry swaps |
| Depth versus accuracy at fixed gate count | Connectivity improvements do not eliminate optimization collapse |
| Full-S accuracy versus simplified circuit size | Accuracy improvement with a modest circuit-size penalty |
| Raw-seed gain plot | Replication strength and uncertainty without hiding individual runs |

The defensible paper claims are **improved fixed-budget dense accuracy, a better observed dense accuracy–gate-count frontier, and a unified fixed-routing method with promising convolutional results**. Universal dataset gains, a proven ancestry-based mechanism, lower physical hardware cost, and replicated full-schedule convolutional superiority remain unsupported by the completed evidence.

**Reported-source ledger**

Reported accuracy references were imported from [PAPER_COMPARISON_TABLES.md](../../PAPER_COMPARISON_TABLES.md), [DATE_TABLES.md](../../DATE_TABLES.md), and [CIFAR10_BASELINE_REFERENCE.md](../../CIFAR10_BASELINE_REFERENCE.md). Local-PDF spot checks resolved routing variants and supplied the explicitly identified supplementary values. This is a documentation update, not a new literature search or a re-evaluation of any checkpoint.

| Code | Paper | Table / figure locator | Values used and metric caveat |
|---|---|---|---|
| [D] | Deep Differentiable Logic Gate Networks | Tables 4–6; Appendix A.3, Tables 8–9 (PDF page 15) | MNIST and dense CIFAR-10 test accuracy; original-code training time and published SD |
| [M] | A Method for Optimizing Connections in Differentiable Logic Gate Networks | Figures 3–4 and accompanying MNIST/Fashion text | Mommen learned-routing accuracies; Fashion fixed-random 87.17 reference; paper gate budgets differ from local adaptations |
| [L] | LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment | Tables 3 and 6; appendix Table 8 (PDF page 14) | Top-32 versus fully learned 1L distinguished; 1F/2F random references 49.17/54.76; Table 8 reports mean ± SD over five runs |
| [B] | BitLogic: A Framework for Gradient-Based LUT-Native Neural Networks | Shared-protocol Table 6 and method mapping Table 8 | Test accuracies from BitLogic’s own common-protocol retrainings; width must be doubled for two-layer total gate counts; CIFAR-100 WARP-LUT L is one seed |
| [C] | Convolutional Differentiable Logic Gate Networks | CIFAR-10 Table 1 (PDF page 8), architecture Appendix A.1 | LogicTreeNet test accuracies and source operation counts; TTNet entries are explicitly secondary quotations in this table |
| [U] | Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks | Table 1 (M) and Appendix B Table 7 (S) | Pre-tying checkpoints and 30%-tied results with fine-tuning; V labeling retained from the project ledger |
| [I] | Light Differentiable Logic Gate Networks | Figure 8 (PDF page 14) | Fivefold-depth CIFAR-100 convolutional comparison; discretized test accuracy, not base M |
| [S] | From MNIST to ImageNet: Understanding the Scalability Boundaries of Differentiable Logic Gate Networks | Tables 11–12 (PDF page 22) | Dense and modified-convolutional test references; source protocols differ from the local matched controls |
| [P] | Fitting Multilinear Polynomials for Logic Gate Networks | Table 1, Table 15, and CIFAR-10 scaling Table 21 | Hardened test accuracy averaged over last ten checkpoints; different widths/depths/training and parameterization |
| [W] | WARP Logic Neural Networks | Figure 4 | Approximate three-seed validation-curve endpoints at 50K; not exact tabulated values and not Table 16’s adapted WARP experiment |
| [A] | Silicon Aware Neural Networks | Table II | 60.07% CIFAR-10 baseline from a separate hardware study; no matched local A or imported split definition |

A missing R cell means that no applicable reference is recorded here, not that the method is absent from all literature. A missing A cell means no eligible completed local result is available for that particular method and protocol in the September 6 snapshot. Neither absence authorizes filling the cell with a result from another architecture, an interrupted run, or ongoing work.

[D]: ../../../../pdfs/deep_differentiable_logic_gate_networks.pdf
[M]: ../../../../pdfs/a_method_for_optimizing_connections_in_differentiable_logic_gate_networks.pdf
[L]: ../../../../pdfs/lilogic_net_compact_logic_gate_networks_with_learnable_connectivity_for_efficient_hardware_deployment.pdf
[B]: ../../../../pdfs/bitlogic_a_framework_for_gradient_based_lut_native_neural_networks.pdf
[C]: ../../../../pdfs/convolutional_differentiable_logic_gate_networks.pdf
[U]: ../../../../pdfs/two-stage_unit_tying_for_simplifying_differentiable_logic_gate_networks.pdf
[I]: ../../../../pdfs/light_differentiable_logic_gate_networks.pdf
[S]: ../../../../pdfs/from_mnist_to_imagenet_understanding_the_scalability_boundaries_of_differentiable_logic_gate_networks.pdf
[P]: ../../../../pdfs/fitting_multilinear_polynomials_for_logic_gate_networks.pdf
[W]: ../../../../pdfs/warp_logic_neural_networks.pdf
[A]: ../../../../pdfs/silicon_aware_neural_networks.pdf
