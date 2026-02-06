# Differentiable Logic CA Experiments (Standalone Notebooks)

This folder contains standalone Jupyter notebooks split from `diffLogic_CA.ipynb`, each focused on a single experiment from the original Differentiable Logic Cellular Automata (DiffLogic CA) notebook.

## Notebooks

- `diffLogic_CA_game_of_life.ipynb`
  - Learns the classic Conway's Game of Life update rule using a DiffLogic CA and analyzes the learned dynamics.
- `diffLogic_CA_checkerboard.ipynb`
  - Learns a checkerboard pattern with synchronous updates; includes training, visualization, and gate analysis.
- `diffLogic_CA_checkerboard_async.ipynb`
  - Learns a checkerboard pattern with asynchronous updates; includes robustness and self-healing tests.
- `diffLogic_CA_lizard.ipynb`
  - Learns to grow a lizard silhouette from a single seed; includes generalization and hidden-state visualization.
- `diffLogic_CA_base64_G.ipynb`
  - Regenerates a multicolor "G" image from a base64 target; includes training and gate analysis.

## How to run

1) Open any notebook and run all cells from top to bottom.
2) Each notebook is standalone and includes all shared setup code from the original notebook.

Notes:
- The original notebook includes a reproducibility note and environment details near the top. Those cells are preserved in each standalone notebook.
- Training can be time- and memory-intensive depending on your hardware.

## Source

These notebooks are derived from the original DiffLogic CA notebook referenced here:
- `diffLogic_CA.ipynb`
- https://google-research.github.io/self-organising-systems/difflogic-ca/
