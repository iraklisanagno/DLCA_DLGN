# Published reference values

These numbers were checked against the supplied local PDFs. They are **REPORTED (R)**, not measurements produced by the local training harness. Page numbers below count PDF pages from one. The PDF hashes are in [literature/source_pdfs.json](literature/source_pdfs.json). A reported standard deviation is preserved as printed; it is not recomputed using the local three-seed protocol.

| Source | Coordinate | Reported test accuracy, % | Reported training time | Location |
|---|---|---:|---:|---|
| Deep Differentiable Logic Gate Networks | CIFAR-10 S, 48K | 51.27 ± 0.26 | 1.3 h | PDF p. 15, Table 9; mean also on p. 9 |
| Same | CIFAR-10 M, 512K | 57.39 ± 0.13 | 7.4 h | PDF p. 15, Table 9 |
| Same | CIFAR-10 L, 1.28M | 60.78 ± 0.12 | 24.2 h | PDF p. 15, Table 9 |
| Convolutional Differentiable Logic Gate Networks | LogicTreeNet-S | 60.38 | Not used in the manuscript | PDF p. 8 |
| Same | LogicTreeNet-M | 71.01 | Not used in the manuscript | PDF p. 8 |
| LILogic Net, archived arXiv v2 | 1F, 64K gates, seven thresholds | 49.17 ± 0.14 | 2.9 ± 0.0 min | PDF p. 14, seven-threshold columns |
| Same | 1Top32, 64K gates, seven thresholds | 57.28 ± 0.30 | 15.3 ± 0.0 min | PDF p. 14 |
| Same | 2F, 2 × 128K gates, seven thresholds | 54.76 ± 0.27 | 16.3 ± 0.0 min | PDF p. 14 |
| Same | 2Top32, 2 × 128K gates, seven thresholds | 60.98 ± 0.19 | 297.2 ± 6.6 min | PDF p. 14; accuracy also p. 7 |

The main manuscript includes the reported accuracy columns and the locally achieved accuracy, time, and memory. Published training times are retained here as contextual information; different hardware, implementations, and protocols prevent treating their ratio to local times as a measured speedup. No published peak GPU-memory number is inferred.

There is no exact published 8K six-layer MNIST or 16K six-layer Fashion-MNIST reference in the frozen comparison records. The paper leaves those R cells empty rather than substituting a larger architecture. Shared convolutional gate functions, spatial gate applications, and simplified logical operation counts are different quantities and must not be interchanged.

Primary references: [Deep DLGNs](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html), [Convolutional DLGNs](https://arxiv.org/abs/2411.04732), [LILogic Net v2](https://arxiv.org/abs/2511.12340v2). The manuscript cites the archived versions from which the numbers were extracted.

Dataset citations use [LeCun et al.](https://ieeexplore.ieee.org/document/726791), [Fashion-MNIST](https://arxiv.org/abs/1708.07747), [Krizhevsky's CIFAR technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), and [the CIFAR-10.1 release](https://github.com/modestyachts/CIFAR-10.1). These sources identify the datasets; all experimental results in the manuscript come from the repository records.
