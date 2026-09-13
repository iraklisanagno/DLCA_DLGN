# Writing voice for the DATE paper

The shared voice of these papers builds an engineering argument: identify a concrete constraint, explain a mechanism that addresses it, and quantify the resulting benefit alongside its cost. Use that structure for the LGN paper while keeping claims within the completed evidence.

This summary accompanies [voice_profile.yaml](voice_profile.yaml). The YAML retains the writing-style skill's template fields and adds source provenance, measured cadence, and clearly separated project requirements.

## Corpus and confidence

Six user-provided PDFs were analyzed, totaling 60 pages. Page references below are PDF page numbers, starting at one.

| ID | Paper | Pages | Most useful voice evidence |
|---|---|---:|---|
| D1 | [Balancing Throughput and Fair Execution / FairBoost](/home/ianagno/.codex/skills/writing-style/assets/pdfs/Balancing_Throughput_and_Fair_Execution_of_Multi-DNN_Workloads_on_Heterogeneous_Embedded_Devices.pdf) | 14 | Motivating quantitative example; explicit definitions; throughput/fairness trade-offs and exceptions |
| D2 | [Ecomap](/home/ianagno/.codex/skills/writing-style/assets/pdfs/Ecomap.pdf) | 13 | Constraint-driven methodology; representation definitions; separate overhead analysis |
| D3 | [Less is More](/home/ianagno/.codex/skills/writing-style/assets/pdfs/Less_is_More_Optimizing_Function_Calling_for_LLM_Execution_on_Edge_Devices.pdf) | 7 | Compact DATE organization; concrete motivating query; offline/online distinction |
| D4 | [Pythia](/home/ianagno/.codex/skills/writing-style/assets/pdfs/Pythia_An_Edge_First_Agent_for_State_Prediction_in_High-Dimensional_Environments.pdf) | 4 | Concise component explanations; accuracy plus runtime and power; explicit limitations |
| D5 | [RankMap](/home/ianagno/.codex/skills/writing-style/assets/pdfs/RankMap_Priority_Aware_Multi_DNN_Manager_for_Heterogeneous_Embedded_Devices.pdf) | 8 | DATE contribution structure; worked reward example; metric-by-metric trade-offs |
| D6 | [ToolAssist](/home/ianagno/.codex/skills/writing-style/assets/pdfs/ToolAssist_An_Edge_First_Function_Calling_Framework_for_Large_Language_Models_with_Adaptive_Cloud_Collaboration.pdf) | 14 | Component ablations; interpretation of proxies; success/latency/cost analysis |

All six share Iraklis Anagnostopoulos as a coauthor. The profile captures the recurring voice of this author group; the PDFs cannot establish which individual wrote each passage. ToolAssist also discloses language-model assistance for grammar, language, and clarity on page 13. Recurrence across papers carries more weight than that manuscript's distinctive phrasing. The compact structures of the two PDFs labeled DATE 2025, D3 and D5, are the closest venue examples.

Text extraction was adequate for analysis. Two-column reading order, line-end hyphenation, equations, and figure labels introduce artifacts, so they are not treated as stylistic preferences. Source hashes and cadence-sample anchors are recorded in the YAML. These papers provide writing examples; their technical claims were not independently audited for this task.

## Rules to apply while drafting

1. **Lead with a specific technical problem.** Connect fixed connectivity to accuracy under a declared gate budget. Skip a general introduction about the growth of AI.
2. **Make the mechanism explicit.** Define the input, operation, and output. Explain why each choice addresses the preceding constraint.
3. **Use a direct author voice.** Write “we define” and “we evaluate” for research actions, then make the component the subject when describing its operation.
4. **Explain results through comparisons.** Identify the setting and baseline, state the measured difference, and then discuss the supported explanation and any cost or exception.
5. **Keep transitions purposeful.** “However,” “specifically,” and “for example” recur across the corpus. Use them for an actual logical relationship; do not insert one at every paragraph boundary.
6. **Vary sentence length.** Mix short statements with longer technical explanations. Review sentences above 35 words instead of enforcing a rigid limit.
7. **Preserve technical wording.** Correct grammar without replacing established nouns, component names, or the intended claim. Prefer precise metrics to general “performance” claims.
8. **Keep the requested constraints explicit.** No em dash or generic AI scaffolding. Avoid unsupported novelty, optimality, robustness, and universal-superiority claims, even where similar wording occurs in the sources.

## Evidence and examples

Short source excerpts illustrate the patterns; the LGN examples are newly written illustrations, not quotations or new experimental findings.

| Pattern | Source evidence | Application to the LGN paper |
|---|---|---|
| Define the object before discussing its benefit | D2, p4: “We define device-specific hardware operational modes to control power consumption.” | “We represent each logic gate by its two predecessor indices and its trainable gate function.” |
| State the motivating idea directly | D3, p2: “The core idea behind Less-is-More is to avoid presenting the LLM with all tools upfront.” | “We construct the fixed connectivity before training and optimize the gate functions on the resulting graph.” |
| Introduce a distinction before its implementation | D5, p3: “We employ two distinct approaches within the RankMap framework to prioritize DNNs” | “We apply the construction to dense layers and convolutional channel pairings.” |
| Make an exception visible | D1, p10: “A noticeable exception occurs in mix 7, where the GA achieves better fairness than FairBoost.” | “U2 has the higher mean validation accuracy, but the paired confidence interval includes zero.” |
| Bound the interpretation of a proxy | D6, p7: “Crossevaluation therefore measures alignment between local and cloud sequence predictions rather than absolute correctness.” | “Gate count describes the logical budget; physical area requires measurements from a synthesis flow.” |
| Name the limitation concretely | D4, p4 discusses inaccuracies during fast movements and the action representation | “The dense factorial study identifies first-layer structure as the dominant contributor under the evaluated configuration.” |

The corpus sometimes uses broad openings, repeated “novel” or “significant,” strong causal explanations, and repeated conclusion phrases. Those are observed habits, but they conflict with the requested precision and are excluded from the drafting defaults. Likewise, the five name-plus-subtitle titles do not require putting CoverageDLGN in this paper's title.

## Cadence and practical targets

A small, auditable cadence sample contains one complete abstract plus two selected body passages from each paper: **18 passages, 78 sentences, and 1,554 whitespace-delimited words**. Mean sentence length is **19.92 words**, the median is **18**, and individual sentences range from **7 to 41** words. Per-paper sample means range from **16.46 to 23.93**. These are approximate selected-passage statistics, not full-corpus estimates; PDF word joining affects counts.

Use a soft section-average target of **18-24 words per sentence** and usually **3-5 sentences per ordinary paragraph**. A short abstract can target **140-180 words** with a problem, mechanism, evaluation scope, and supported result. The six source abstracts range from approximately **100 to 201 words**. The abstract target is an editorial choice, not a verified DATE submission limit. Prefer **three contribution bullets**, adding a fourth only for a distinct contribution.

## Requirements specific to this paper

Keep these requirements separate from the inferred voice:

- Present U2 as the unified method and V3 as a dense specialization. Describe structured connectivity in both first and deeper layers.
- Limit the first-layer-dominance interpretation to the dense factorial setting. Do not attribute gains solely to degree balance, coverage, or ancestry novelty.
- Preserve achieved versus reported provenance, validation versus held-out test scope, paired seeds, and uncertainty. A positive mean alone does not establish a consistent gain.
- Distinguish offline construction, training time, allocated GPU memory, and inference cost. An unchanged gate budget does not establish physical area or energy savings.
- Recheck the current experiment artifacts before using numbers. The voice profile is not a frozen results ledger.

Explicit instructions and project evidence requirements take precedence over stylistic imitation; hard writing constraints take precedence over recurring corpus habits; numerical style targets remain flexible. Apply this profile before subsequent drafting with the paper-writing skill.
