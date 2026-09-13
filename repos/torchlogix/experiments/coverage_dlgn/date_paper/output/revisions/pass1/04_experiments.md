# Experimental Evaluation

## Protocol and main accuracy results

We evaluate MNIST, Fashion-MNIST, and CIFAR-10 using the saved configurations summarized in Table \ref{tab:protocol}. Each attribution pair shares the architecture, encoding, optimizer, training budget, split, and seed. We reserve 10% of the original training set for validation using split seed 2027. Checkpoints are selected by hardened validation accuracy before held-out evaluation. Tables distinguish locally reproduced baselines (REP), our methods (OUR), and paper-reported references (R); V and T denote validation and test accuracy. Intervals use paired Student-$t$ statistics over training seeds, with sample standard deviations reported alongside means.

{{table:protocol}}

Table \ref{tab:dense} compares the same three seeds for random, V3, and U2. U2 improves dense CIFAR-10 test accuracy by 3.18, 4.56, and 4.59 percentage points at S, M, and L, with positive paired intervals. MNIST also improves, whereas the Fashion-MNIST interval includes zero. V3 remains more accurate at L, preventing a claim that the unified construction dominates every dense specialization. Published baselines differ from local results and provide context rather than paired evidence.

{{table:dense}}

The convolutional comparison in Table \ref{tab:conv} retains this distinction. Existing single-seed S/M tests improve by 3.26/2.08 points, respectively. Additional full-S seeds yield 61.05% mean U2 validation accuracy versus 59.82% for random, but U2 loses on seed 2. The paired gain is 1.23 points with interval $[-3.36,5.82]$. These new checkpoints have no held-out test results, so the single-seed test gains do not establish a replicated convolutional improvement.

{{table:conv}}

## Training-resource trade-offs

The matched dense models retain equal peak GPU allocations. For U2, construction takes approximately 14.33 seconds at M and 31.22 seconds at L, compared with 41.39 and 113.53 minutes of recorded training. Full-S training averages 4.97 hours for both methods at 1.83 GiB. Full-M U2 instead records 35.69 hours versus 25.47 for random. Different execution conditions prevent attributing that difference solely to topology, but the observed cost must remain visible.

Table \ref{tab:routing} compares U2 with learned Top-32 routing under the LILogic architectures and encoding. U2 uses 80% fewer training parameters and approximately 16-17 times less peak GPU memory, while losing 5.30/1.84 accuracy points at the two sizes. The Top-32 runs have one local seed. These descriptive trade-offs show when fixed connectivity is useful without claiming higher accuracy than learned routing. All memory values are peak PyTorch allocations, and wall times include evaluation overhead.

{{table:routing}}

## Where does structure help?

To examine the dense gain, we cross structured (S) and randomized (R) first/deeper layers at M with 20K updates. Randomization preserves each predecessor's degree in each input slot. Full U2 attains the highest mean in Table \ref{tab:factorial}. Structuring the first layer gains 4.49 points with structured deeper layers and 3.69 with randomized deeper layers; both intervals exclude zero. The additional deeper-layer gain is 0.52 points with interval $[-0.52,1.56]$. All five contrasts are exploratory and unadjusted.

{{table:factorial}}

The corresponding convolutional body/classifier study finds a 2.14-point body effect with a fixed reference-U2 classifier, interval $[0.67,3.61]$. Because that classifier uses reference-body ancestry, the effect is conditional. Separate controls do not establish an incremental benefit from ancestry-based selection over nominal stages. Adapted WARP preserves a dense U2 advantage but gives an inconclusive convolutional effect; both adapted variants trail the matched raw parameterization. These outcomes bound the mechanism and compatibility claims.
