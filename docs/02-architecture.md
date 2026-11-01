# System architecture

> Part of the DineWise project plan. Components, request lifecycle, and design principles.
> See [the plan index](README.md) for the full reading order.

---

## 4. System architecture

### 4.1 Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ Streamlit UI                                                         │
│  profile sidebar · chat · ranked cards · evidence panel · debug tab  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ request
┌───────────────────────────────▼──────────────────────────────────────┐
│ LangGraph workflow                                                   │
│                                                                      │
│ classify_intent → extract_preferences → rewrite_query                │
│        → retrieve → filter_candidates → rerank → generate_answer     │
│        → review_answer ──supported──→ END                            │
│              │                                                       │
│              └──insufficient──→ retry (max 1) → review_answer        │
│                                      │                               │
│                                      └──still insufficient──→ fallback│
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
┌───────▼────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│ Retrieval      │   │ Evidence layer     │   │ Generation         │
│ FAISS (BGE)    │   │ BM25 over candidate│   │ Qwen2.5-1.5B       │
│ BM25 (rank)    │   │ restaurants' raw   │   │ schema-constrained │
│ attribute      │   │ reviews → quotes   │   │ decoding + repair  │
│ filters        │   │ with review_id     │   │                    │
│ RRF + reranker │   │                    │   │                    │
└───────┬────────┘   └──────────┬─────────┘   └──────────┬─────────┘
        │                       │                        │
┌───────▼───────────────────────▼────────────────────────▼─────────┐
│ Data layer                                                        │
│ restaurants.parquet · reviews.parquet · doc chunks · FAISS index   │
│ · BM25 corpus · CLIP text index · photo metadata                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 Request lifecycle

1. The UI sends a query plus the current profile.
2. `classify_intent` labels the request as recommendation, comparison, visual, or out-of-scope.
3. `extract_preferences` produces a typed constraint object: city, cuisine families, max price bucket, min rating, required attributes, dietary terms, ambience terms, and soft preferences.
4. `rewrite_query` produces a retrieval-friendly query and a set of attribute filters.
5. `retrieve` runs dense search, BM25, and (where applicable) the visual index, then fuses results.
6. `filter_candidates` applies hard constraints against the structured catalogue. Hard constraints are never relaxed.
7. `rerank` orders candidates with a cross-encoder and applies personalization adjustments.
8. The evidence layer retrieves quotable review sentences for each of the top candidates.
9. `generate_answer` builds the prompt from facts, policy, and evidence, and calls the LLM with a JSON schema.
10. `review_answer` validates the output programmatically and with NLI entailment.
11. If validation fails, the workflow retries once with an expanded query and relaxed soft preferences only. If it fails again, `fallback` returns an explicit insufficiency message.

### 4.3 Design principles

1. **Facts come from structured data, not from the model.** Names, ratings, price buckets, and attributes are looked up by ID and injected into the prompt. The model is instructed not to generate them from memory.
2. **Rules are separate from facts.** The `dining_preferences.docx` policy document is retrieved and injected in a distinct prompt block. A policy such as "prefer quiet restaurants" can never be cited as evidence that a restaurant is quiet.
3. **Every claim must be traceable.** Each recommendation must cite a business ID and at least one evidence chunk or review ID, or declare that it is based only on structured attributes.
4. **Abstention is a feature.** Out-of-scope or unsupported queries must produce a clear refusal with an explanation, not a plausible invention.
5. **One variable at a time.** Every experiment changes one major component while holding the corpus, queries, and remaining settings fixed.
6. **Measure, do not assume.** Latency and memory claims are replaced with measured values from `results/`.

---
