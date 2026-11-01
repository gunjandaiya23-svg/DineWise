# Risks, validity and limitations

> Part of the DineWise project plan. Risk register, threats to validity, and ethical constraints.
> See [the plan index](README.md) for the full reading order.

---

## 20. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | 8 GB VRAM cannot co-load reranker, CLIP, LLM, and training | High | Medium | Model registry with explicit unload; sequential stages; 4-bit fallback; measure peak memory |
| 2 | Qwen 1.5B produces invalid JSON | High | Medium | Schema-constrained decoding; repair loop; format validity as a metric; consider 3B if needed |
| 3 | PDF extraction is slow or low quality | Medium | Medium | Cache extracted text; compact layout; per-city split if needed; record extraction time |
| 4 | Review/photo summaries contain unsupported claims | High | High | Synthetic evidence audit; prefer raw reviews for quotes; validator entailment checks |
| 5 | Gold-set labeling takes longer than planned | High | High | Prefill candidates; cap gold set at 10; label incrementally; freeze at 60 if necessary and report |
| 6 | Retrieval differences are not statistically significant | Medium | Medium | Report CIs; do not overclaim; larger gold set; analyze per-facet |
| 7 | LoRA degrades grounding | Medium | High | Include LoRA-only ablation; validate hard constraints; do not merge adapters before evaluation |
| 8 | Dataset download or license issue | Low | High | Legal use is non-commercial research; do not redistribute raw data; cite sources |
| 9 | Scope creep (20 notebook sections, many experiments) | High | High | Phase gates; MVP first; advanced experiments only after P6 passes |
| 10 | Reproducibility drift from library updates | Medium | Medium | Pin versions; lockfile; run manifest; content-hash caches |
| 11 | Ambience attributes too sparse (149 romantic=True) | High | Medium | Use review and visual summaries as soft evidence; state uncertainty; do not treat missing as false |
| 12 | CLIP cuisine mapping is unreliable | High | Medium | Soft signal only; confidence threshold; report top-1/top-5 accuracy and coverage |
| 13 | LLM judge bias in RAGAS | Medium | Medium | Triangulate with programmatic checks and human spot checks; name the judge |
| 14 | GPU thermal throttling on a laptop | Medium | Low | Warm-up runs; report variance; use median |
| 15 | Time lost to environment issues | Medium | Medium | P0 acceptance tests; freeze the environment before experiments |

---

---

## 21. Threats to validity and limitations

### Construct validity

- **Ratings and prices are dataset snapshots**, not live values. The system cannot answer "is it good now?" or "how much does it cost?".
- **Price bucket ≠ dollars.** Budget interpretation is approximate and documented.
- **Cuisine is derived** from category tokens, not an official label. Multi-label and ambiguous cases exist.
- **Dietary evidence is textual.** Absence of evidence is not evidence of absence, and medical guarantees are impossible.
- **Ambience attributes are sparse.** Using review and photo summaries introduces subjectivity.
- **"Personalization" is explicit profile matching**, not a learned user model.

### Internal validity

- Gold-set labels are subjective. Mitigate with a rubric, a second pass, and agreement reporting.
- LLM-generated summaries may bias both retrieval and evaluation. Mitigate with the synthetic evidence audit and raw-review support.
- Reranker and LLM changes could interact. Mitigate with one-variable-at-a-time experiments.
- Prompt changes can dominate effects. Freeze prompts before the main experiments and version them.
- Randomness in generation is controlled with greedy decoding; sampling runs are reported separately if used.

### External validity

- Five cities, US-centric, one platform (Yelp). Results may not transfer to other cities or countries.
- The held-out city experiment tests only two additional cities.
- The 1.5B model is a small model; conclusions may not hold for larger models.
- The 8 GB GPU constrains batch sizes and may affect latency conclusions.

### Conclusion validity

- Multiple comparisons increase false-positive risk. Report CIs and avoid claiming significance without them.
- Small gold sets produce wide intervals. Report them.
- Do not claim causality from observational retrieval metrics; the experiments are controlled but the data is not.

### Ethical and legal

- The dataset is for non-commercial research; do not redistribute raw data.
- Do not scrape Yelp or download restaurant photos.
- Do not present the system as a source of medical dietary advice.
- Do not collect or store user images without consent.
- Attribute LLM-generated summaries and state their provenance.

---
