# Supporting raw results

All values are percentages. Seeds are ordered 0, 1, 2. SD is the sample standard deviation.

## Dense held-out tests

| Model | Random | V3 | U2 | U2 gain, pp | Paired 95% CI | Wins |
|---|---|---|---|---:|---|---|
| MNIST | 91.190, 91.520, 91.110 | 91.570, 91.980, 92.170 | 91.790, 91.960, 92.060 | 0.663 | 0.015, 1.311 | 3/3 |
| Fashion-MNIST | 85.110, 85.490, 84.990 | 86.280, 85.890, 85.570 | 85.340, 85.590, 86.220 | 0.520 | -1.016, 2.056 | 3/3 |
| CIFAR-10 S | 48.840, 49.290, 48.610 | 52.400, 52.680, 52.090 | 52.070, 51.480, 52.740 | 3.183 | 0.772, 5.595 | 3/3 |
| CIFAR-10 M | 53.900, 54.210, 54.180 | 58.450, 58.070, 58.550 | 58.800, 58.690, 58.470 | 4.557 | 3.781, 5.332 | 3/3 |
| CIFAR-10 L | 55.820, 55.590, 56.200 | 60.540, 61.310, 61.370 | 60.080, 60.550, 60.760 | 4.593 | 3.721, 5.466 | 3/3 |

## Full-S validation

| Method | Best hard seeds | Final hard seeds |
|---|---|---|
| random | 58.680, 59.840, 60.940 | 58.500, 58.640, 60.340 |
| u2 | 61.000, 62.120, 60.040 | 60.280, 60.920, 59.340 |

## Dense factorial validation

| Arm | Best hard seeds | Mean | SD |
|---|---|---:|---:|
| SS | 58.860, 59.480, 59.060 | 59.133 | 0.316 |
| SR | 58.820, 58.780, 58.240 | 58.613 | 0.324 |
| RS | 54.260, 55.260, 54.400 | 54.640 | 0.541 |
| RR | 54.800, 54.480, 55.500 | 54.927 | 0.522 |

| Contrast | Paired seed differences, pp | Mean | Paired 95% CI | Wins |
|---|---|---:|---|---|
| deeper_given_randomized_first | -0.540, 0.780, -1.100 | -0.287 | -2.685, 2.111 | 1/3 |
| deeper_given_structured_first | 0.040, 0.700, 0.820 | 0.520 | -0.523, 1.563 | 3/3 |
| first_given_randomized_deeper | 4.020, 4.300, 2.740 | 3.687 | 1.621, 5.753 | 3/3 |
| first_given_structured_deeper | 4.600, 4.220, 4.660 | 4.493 | 3.901, 5.086 | 3/3 |
| interaction | 0.580, -0.080, 1.920 | 0.807 | -1.725, 3.338 | 2/3 |

All five factorial intervals are exploratory and unadjusted. Randomized factorial arms preserve exact per-node, per-slot degrees and are not native-random baselines.

Complete resource rows, supporting studies, saved training configurations, and source hashes are in [paper_evidence.json](paper_evidence.json). The source registry uses paths relative to the parent coverage_dlgn directory. No inference or training was performed to assemble these tables.
