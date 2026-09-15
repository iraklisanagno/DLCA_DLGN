# Conclusion

LogicConnect treats fixed connectivity as an architectural design choice before gate-function training. Semantic input pairs organize access to image sources, while regular multiscale matchings extend the construction to deeper predecessors and convolutional channel groups. The graph retains the baseline gate budget and inference operators.

This design improves dense CIFAR-10 test accuracy by 3.18–4.59 percentage points across the three evaluated sizes. Against learned Top-32 routing, LogicConnect uses 93.7–94.2% less peak GPU memory at an accuracy cost of 1.84–5.30 points. This establishes an accuracy/resource choice for DLGN training.

Degree-preserving ablations locate the strongest dense effect in the first layer. Deeper structure, ancestry selection, and replicated convolutional gains remain unresolved. Broader claims require further held-out replications and physical implementation measurements.
