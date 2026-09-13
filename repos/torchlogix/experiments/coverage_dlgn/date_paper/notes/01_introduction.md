# Introduction brief

- Open with a specific design constraint: two-input gates can only combine signals accessible through fixed predecessors. Learning gate functions does not change that graph.
- Motivate the design choice between random fixed connections and learned routing. The latter may improve accuracy, but introduces training state and candidate activations.
- State the central question: can an offline graph construction serve both dense and convolutional LGNs at a fixed declared architectural budget?
- Introduce the semantic first layer and multiscale deeper construction. Make the convolutional extension concrete: channel pairing, existing spatial sampler, same fixed classifier construction.
- State C1-C4 from `claims_and_budget.md` in three contribution bullets. Mention the three-seed convolutional qualification without turning the introduction into a status report.
- Bridge to related work through the distinction between connectivity, gate parameterization, and post-training simplification.

Sources: `../PAPER_DATE_SEPT_6_w_reported.md`, `../summary/third_round_results.json`, `../unified_refinement/summary/factorial_results.json`, and `../unified_refinement/summary/full_s_results.json`. Presentation tables are navigation aids; numbers must come from machine-readable summaries.
