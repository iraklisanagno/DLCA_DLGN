# Experimental Evaluation

## Evaluation protocol

We evaluate whether fixed graph design improves accuracy (RQ1), changes the training-resource trade-off (RQ2), and yields identifiable structural effects (RQ3). Compatibility and transfer tests examine the boundary of those findings (RQ4). Table \ref{tab:protocol} specifies the main MNIST [@lecun1998], Fashion-MNIST [@xiao2017], and CIFAR-10 [@krizhevsky2009] configurations. Each attribution pair matches architecture, input encoding, gate parameterization, optimizer, updates, evaluation cadence, initialization, and training seed. A 10% validation split uses split seed 2027. Checkpoints are selected by hardened validation accuracy and frozen before held-out evaluation.

{{table:protocol}}

Experiments use RTX PRO 6000 Blackwell Max-Q GPUs with PyTorch 2.9.0 and CUDA 13.0. We report hardened accuracy, peak PyTorch GPU allocation, trainable parameters, CPU construction time, and training wall time including evaluation. Paired experiments use seeds 0, 1, and 2 unless stated otherwise. Tables report means and sample standard deviations; effect intervals use paired Student-$t$ statistics over training seeds. The labels OUR, REP, and R distinguish our constructions, locally reproduced baselines, and reported literature values; V/T distinguish validation/test accuracy.

LogicConnect (LC) denotes the unified method of Section III. LogicConnect-D (LC-D) is a separate dense specialization using semantic butterfly pairing and ancestry-based swaps. Its measurements remain separate from the unified method. Boldface marks the highest local mean, not statistical superiority or an evaluated variant-selection policy.

## Accuracy at a fixed architectural budget

Table \ref{tab:dense} answers RQ1 across five dense configurations. LogicConnect improves CIFAR-10 accuracy by 3.18, 4.56, and 4.59 points at S, M, and L. All three paired confidence intervals exclude zero, and every seed pair improves. Figure \ref{fig:dense_results} shows the advantage across sizes and paired seeds. MNIST improves by 0.66 points with a positive interval; Fashion-MNIST gains 0.52 points, but its interval includes zero.

{{table:dense}}
{{figure:dense_results}}

The dense specialization provides a useful qualification. LC-D has the larger mean at Fashion-MNIST and CIFAR-10 S/L, while unified LC leads at MNIST and CIFAR-10 M. At L, LC-D improves on LC by 0.61 points. Structure improves on native random wiring, but the unified constructor does not dominate every specialization. Published values use different protocols and remain separate.

Table \ref{tab:conv} reports convolutional S/M test gains of 3.26/2.08 points at unchanged architecture, using one seed each. Full-S validation replication gives 61.05% for LC versus 59.82% for random, with a paired gain of 1.23 points and interval $[-3.36,5.82]$. LC wins two of three seeds. The held-out comparisons demonstrate positive operating points, while validation replication does not establish a consistent gain. The additional checkpoints have no held-out evaluations.

{{table:conv}}

## Accuracy versus training resources

Table \ref{tab:routing} addresses RQ2 under the two LILogic architectures and their seven-threshold encoding. At M/L, LC attains 52.54/60.19% test accuracy, improving on matched fixed random by 3.53/4.86 points. Learned Top-32 routing remains more accurate at 57.84/62.03%, but its peak allocations are 8.10/24.86 GiB versus 0.474/1.557 GiB for LC. The fixed constructor therefore uses 94.2%/93.7% less peak GPU memory and 80% fewer trainable parameters, at an accuracy cost of 5.30/1.84 points.

{{table:routing}}
{{figure:routing_tradeoff}}

Figure \ref{fig:routing_tradeoff} presents this trade-off in memory and time. Observed Top-32-to-LC training-time ratios are 2.59/18.98 at M/L; they describe recorded executions, not isolated algorithmic speedups. At L, LC recovers 4.86 of the 6.70 accuracy points that Top-32 gains over random, approximately 73%, using about one-sixteenth of its peak allocation. This operating point suits constrained training memory, while retaining the accuracy cost explicitly.

At dense M/L, CPU construction takes 14.33/31.22 seconds, compared with 41.39/113.53 minutes of training. These construction times are below 0.6% of the corresponding recorded training durations. Dense matched runs retain the same peak GPU allocation as random. Full-S convolutional training averages approximately 4.97 hours for both methods at 1.83 GiB, whereas full-M records 35.69 hours for LC and 25.47 for random. Different execution conditions prevent attributing the M timing difference solely to connectivity. No physical area, energy, or placement benefit is inferred from equal gate counts.

## Structural attribution

RQ3 asks whether the gain survives controls that preserve predecessor use. At dense M with 20K updates, we independently keep or randomize first-layer and deeper-layer wiring. Randomization preserves each predecessor's degree in each input slot. The SS, SR, RS, and RR arms reach 59.13, 58.61, 54.64, and 54.93% validation accuracy, respectively, where S denotes LC structure.

{{figure:structural_ablation}}

Figure \ref{fig:structural_ablation} separates the paired effects. First-layer structure adds 4.49 points with structured deeper layers and 3.69 with randomized deeper layers; both intervals exclude zero. Adding deeper structure to the structured first layer gives 0.52 points with interval $[-0.52,1.56]$. The interaction interval also includes zero. The full construction leads in mean accuracy; first-layer pairing has the strongest identified effect.

Simpler stage-selection controls bound the explanation further. At 20K updates, LC-minus-nominal gains are $-0.09$ points for dense M and $+0.59$ for convolutional S, with intervals $[-1.76,1.59]$ and $[-1.12,2.31]$. Neither result establishes an incremental ancestry-selection advantage. A convolutional body/classifier ablation finds a 2.14-point body effect with a fixed structured classifier, interval $[0.67,3.61]$. That classifier retains reference-body ancestry, so the result is conditional rather than a head-independent body effect. All ablation intervals are exploratory and unadjusted.

## Compatibility and transfer boundaries

RQ4 tests whether benefits survive changes outside the main graph comparison. Matched adapted-WARP experiments retain a dense LC gain of 4.70 points, interval $[3.92,5.48]$, but the convolutional gain is 0.09 with interval $[-1.24,1.42]$. Both adapted recipes trail their matched raw parameterizations. Parameter reduction alone does not establish a better accuracy/memory result.

Frozen CIFAR-10.1 evaluation [@recht2018] provides a separate distribution test without adaptation. Dense M gains 3.73 points, interval $[2.80,4.67]$, across three seeds. Convolutional S/M changes are $-0.05$/$+1.95$ points, one seed each, while CIFAR-100 pilots do not establish an LC benefit. The dense transfer gain extends beyond the original test set; the other results bound its applicability.
