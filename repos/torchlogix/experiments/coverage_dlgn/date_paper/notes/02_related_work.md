# Related-work brief

- Petersen et al. 2022: differentiable mixtures of the 16 Boolean two-input functions, then discretization. Connectivity is an architectural choice.
- Petersen et al. 2024: logic-tree convolutions, OR pooling, residual initialization. Do not conflate the number of shared gate functions with spatial gate applications.
- Mommen et al. 2025: optimize distributions over subsets of candidate predecessors. LILogic Net: scalable learnable connectivity with Top-K sparsity and basis projection. Describe training costs without claiming our method has higher accuracy.
- BitLogic: a different research contribution that factors the design space and supports LUT-native models. The repository's paper is the Bührer et al. TMLR paper, not the unrelated arXiv paper returned by a title-only web search.
- WARP and Light: gate parameterization axis. Compatibility is empirical; fixed routing alone does not guarantee compatibility across all training recipes.
- Unit tying is post-training simplification and changes the retained logical structure. It is conceptually complementary to construction, but stacking is not evaluated here.
- Close with the precise position: a fixed connectivity construction plus matched accuracy/resource comparisons and degree-preserving attribution.

Source PDFs are in repository `pdfs/`; extracted text resides in `evidence/literature/`. Reported numeric references use the exact local PDF versions, especially LILogic Net arXiv v2. They are contextual, not substitute local baselines. Bibliography entries must preserve authors and version/venue provenance.
