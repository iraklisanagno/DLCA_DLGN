# Hardened dense circuit diagnostics (OUR, V-associated; no inference)

All 24 preregistered checkpoints analyzed. Local LUT relevance is exact;
dependency-pruned path support is only an upper bound on global dependence.

| Family | Best-checkpoint validation mean (%) | Final-layer structural sources | Final-layer path sources | Final-layer binary gates (%) |
|---|---:|---:|---:|---:|
| random | 54.820 | 15.961 | 6.627 | 56.22 |
| balanced_random | 58.647 | 15.964 | 6.684 | 54.97 |
| nominal | 59.220 | 15.964 | 6.763 | 55.94 |
| u2 | 59.133 | 15.969 | 6.764 | 55.90 |

All layers, raw seeds, best/final differences, unused predecessors and reachability
are retained in functional_support.json and functional_support.csv. Three training
replicates per family: gates/checkpoints are not additional independent replicates.
These are descriptive associations, not a causal claim, a routing objective, or
physical pruning/area evidence. Constants can still contribute to classifier sums.
