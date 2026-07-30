# Excluded: explicit classifier topology-RNG attempt

Date: 2026-07-29

These artifacts are retained for failure history and must not be included in
accuracy tables or method selection.

## What happened

The first V4 component configs used explicit
`conv_connections_init_method` and `classifier_connections_init_method`
overrides. With `topology_seed` set, the explicit random classifier uses an
independent topology generator. Historical frozen V4 instead uses the
component-wide method selector and lets its default dense classifier follow the
legacy global Torch RNG path.

Dense layers initialize weights before constructing connections. Advancing (or
not advancing) the global RNG during one dense layer therefore changes later
dense weight initialization. The attempted ablation consequently changed more
than the convolutional mechanism.

## Preserved artifacts

- completed no-swap seed 0: best hardened validation accuracy 57.18%;
- completed no-swap seed 1: best hardened validation accuracy 55.08%;
- interrupted no-swap seed 2;
- interrupted channel-spatial seed 0;
- the three no-swap training logs.

## Resolution

The corrected configs use the same historical component-wide selector as
frozen V4. This makes V4, no-swap, and channel-spatial constructions identical
in dense classifier indices, parameter initialization, and spatial coordinates
for a fixed seed. The only intended differences are the V4 coverage swaps
(component ablation) or the channel assignment at convolutional leaf gates
(channel-spatial adapter).
