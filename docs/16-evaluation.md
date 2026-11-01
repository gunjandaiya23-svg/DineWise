# Evaluation framework

> Part of the DineWise project plan. Gold set, annotation, metrics, validators, RAGAS, and statistics.
> See [the plan index](README.md) for the full reading order.

---

## 16. Evaluation framework

### 16.1 Gold set

**Size:** 120 queries for the full-semester plan (105 answerable + 15 out-of-scope).

**Composition of answerable queries (105):**

| Facet | Count | Example |
|---|---:|---|
| Cuisine | 20 | "Find Italian restaurants in Philadelphia." |
| Budget | 15 | "Recommend restaurants under $$." |
| Review aspect | 20 | "Which places are praised for desserts?" |
| Attribute / ambience | 15 | "Romantic restaurant with outdoor seating." |
| Comparison | 10 | "Compare Restaurant A and Restaurant B." |
| Multi-constraint | 15 | "Family-friendly Mexican under $$ in Tucson." |
| Visual | 10 | "Show restaurants with bright modern interiors." |

**Out-of-scope queries (15):**

- live availability or booking (5),
- city not in the dataset, e.g., Las Vegas (3),
- amenity absent from records (3),
- medical/allergy guarantees (2),
- requests for real-time prices or menus (2).

Each query record contains:

```json
{
  "query_id": "Q001",
  "query": "Find Italian restaurants with vegetarian options under $$ in Philadelphia.",
  "facet": "multi_constraint",
  "intent": "recommendation",
  "constraints": {
    "city": "Philadelphia",
    "cuisines": ["Italian"],
    "max_price_bucket": 2,
    "required_attributes": [],
    "dietary": ["vegetarian"],
    "min_rating": null
  },
  "gold_business_ids": [
    {"business_id": "B001", "grade": 2},
    {"business_id": "B042", "grade": 1}
  ],
  "expected_evidence_aspects": ["vegetarian", "food_quality"],
  "out_of_scope": false,
  "notes": "Gold set constructed from structured filters plus manual review."
}
```

### 16.2 Annotation protocol

1. **Prefill.** For each query, a script selects restaurants satisfying the hard constraints and ranks them by a transparent heuristic (attribute match, rating, review count). The top 20 are shown to the annotator.
2. **Grade.** The annotator assigns:
   - **2** — satisfies all hard constraints and has explicit evidence for the requested aspect;
   - **1** — satisfies all hard constraints but lacks explicit evidence for the aspect;
   - **0** — does not satisfy a hard constraint or is irrelevant.
3. **Bound the gold set.** Keep at most 10 graded-2/1 restaurants per query. If more than 10 qualify, prefer diversity across cuisine and price, and record the selection rule. This keeps evaluation tractable and makes Recall@k meaningful.
4. **Second pass.** Re-label 20 queries after at least one week; report intra-annotator agreement (Cohen's kappa). If a second annotator is available, report inter-annotator agreement on those 20.
5. **Freeze.** Once frozen, the gold set is versioned and never edited during experiments. Corrections require a new version and a rerun.

Estimated effort: 8–10 hours, spread across weeks 3–14.

### 16.3 Retrieval metrics

All metrics are computed at the **restaurant level**, not the chunk level, because the user sees restaurants.

Let `G` be the gold set for a query, `R_k` the top-k retrieved restaurants, and `rel(d)` the graded relevance.

**Success@k (hit rate)** — at least one relevant restaurant in the top k. This is the primary retrieval metric for recommendation, because a gold set of 5–10 items cannot support a meaningful Recall@5 over a corpus of 6,000 restaurants.

```
Success@k = 1[ |R_k ∩ G| > 0 ]
```

**Precision@k**

```
P@k = |R_k ∩ G| / k
```

**Mean Reciprocal Rank**

```
MRR = (1/|Q|) Σ_q 1 / rank_q(first relevant)
```

**nDCG@k** with graded gains:

```
DCG@k = Σ_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

**Recall@k** is reported only for the curated gold sets (≤10 items) and is labeled "Recall@k over curated gold set". It is not reported as if the gold set were complete.

Additional metrics:

- Attribute-filter compliance rate: percentage of returned restaurants that satisfy the hard constraints (should be 100%; any failure is a bug).
- Cuisine match rate.
- City match rate.
- Price compliance rate.
- Retrieval latency p50/p95.
- Index size.

### 16.4 Generation metrics

**Programmatic (all 120 queries):**

- constraint satisfaction rate (all hard constraints),
- rating accuracy (returned rating equals database),
- price accuracy,
- citation existence rate,
- citation support rate (quote is a substring of supplied evidence),
- unsupported-claim rate (claims not entailed by context),
- abstention correctness on out-of-scope queries,
- format validity,
- number of recommendations returned,
- fallback rate and retry rate.

**Semantic (RAGAS, stratified subset of 40 answers):**

- faithfulness,
- answer relevancy,
- context precision,
- context recall (requires reference answers or gold contexts; otherwise report as not applicable).

**Human spot check:** 30 answers, judged for usefulness, correctness, and explanation quality on a 1–5 scale. Report mean and agreement.

**Judge model:** RAGAS requires an LLM judge. Options, in order of preference:

1. A strong API model (Gemini Flash or GPT-class) if a key is available; record the exact model and date.
2. A local `Qwen2.5-7B-Instruct` in 4-bit, run as a separate offline process; record the model and acknowledge self-preference bias.
3. A local NLI model only, reporting entailment-based faithfulness without RAGAS.

The judge model is always named in the results table. The project does not present LLM-judge scores as objective truth; they are triangulated with programmatic checks and human spot checks.

### 16.5 Programmatic validators

Implemented as a reusable module with unit tests. Each validator returns `{passed, reason, evidence}`.

| Validator | Check | Severity |
|---|---|---|
| `check_business_exists` | ID in candidate set and catalogue | hard |
| `check_name` | Name matches catalogue | hard |
| `check_rating` | Rating equals catalogue to one decimal | hard |
| `check_price` | Price bucket equals catalogue | hard |
| `check_hard_constraints` | City, cuisine, price, attributes, dietary | hard |
| `check_citations_exist` | Cited review IDs exist for that restaurant | hard |
| `check_quote_substring` | Quote appears in supplied evidence | hard |
| `check_no_live_claims` | No booking/availability/current-price language | hard |
| `check_count` | 0 ≤ recommendations ≤ 5 | hard |
| `check_entailment` | Explanation sentences entailed by evidence | soft |
| `check_uncertainty_language` | Unknown attributes not stated as facts | soft |
| `check_acknowledgement` | Missing evidence acknowledged | soft |

The hard/soft distinction is fixed before the experiments and reported.

### 16.6 RAGAS configuration

- `ragas` version pinned in `requirements.lock`.
- Metrics: faithfulness, answer relevancy, context precision, context recall (when reference contexts exist).
- Embeddings for RAGAS: the same BGE model, to avoid introducing another variable.
- Judge: as above.
- Run on 40 answers stratified by facet, not on all 120, for cost and time reasons. The subset is fixed and reported.
- Report per-metric mean with bootstrap confidence intervals.

### 16.7 Statistical reporting

- Report mean, median, standard deviation, and 95% bootstrap confidence intervals (1,000 resamples) for every headline metric.
- For A/B comparisons, use a paired bootstrap over queries and report the mean difference, its 95% CI, and a two-sided p-value.
- Do not declare a winner when the confidence interval includes zero. State that the difference is inconclusive.
- Report the number of queries, the number of gold items, and any exclusions.
- Never compare runs with different gold-set versions or different query sets.

### 16.8 Synthetic evidence audit

Because review and photo summaries are LLM-generated, the project must estimate their reliability.

Protocol:

1. Sample 50 restaurants stratified by review count.
2. For each, sample 3 claims from `review_summary.csv`.
3. Check each claim against the raw reviews for that restaurant.
4. Label as supported, partially supported, or unsupported.
5. Report the support rate and examples of drift.

This audit is a validity check on the evidence layer, not a core experiment. It belongs in the results and limitations sections.

---
