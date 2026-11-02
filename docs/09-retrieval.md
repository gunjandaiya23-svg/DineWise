# Retrieval

> Part of the DineWise project plan. Dense, lexical, structured filters, RRF, reranking, and evidence.
> See [the plan index](README.md) for the full reading order.

---

## 9. Retrieval

### 9.1 Dense retrieval

- Embed the rewritten query with the BGE prefix.
- Retrieve top 30 chunks by cosine similarity.
- Collapse chunk hits to restaurants: a restaurant's score is the maximum chunk score, with a small bonus if multiple chunks from the same restaurant appear. The collapse rule is fixed before experiments.
- Apply metadata filters in Qdrant at query time (city, cuisine, price, attributes) and keep a client-side check as a safety net. Server-side pre-filtering avoids wasting the candidate budget.

### 9.2 BM25 retrieval

- Tokenize the rewritten query with the same tokenizer as the index.
- Retrieve top 30 chunks.
- Collapse to restaurants with the same rule as dense retrieval.
- BM25 is expected to help with dish names, proper nouns, and exact phrases; this is a hypothesis to test, not an assumption.

### 9.3 Structured filters

Hard constraints are applied against `restaurants.parquet` before or immediately after retrieval:

- city (exact match, case-insensitive),
- cuisine family membership,
- price bucket (`<=`, `>=`, or range),
- minimum rating,
- required attributes (e.g., `outdoor_seating is True`),
- open status (default: prefer open, do not hard-exclude unless the user asks),
- dietary evidence (only if explicit evidence exists; otherwise the constraint is soft and the answer must say so).

Tri-state logic: a required attribute is satisfied only by `True`; `False` and `None` both fail a hard requirement, but the answer distinguishes "known not to have it" from "unknown". This distinction matters for explanation quality.

### 9.4 Reciprocal Rank Fusion

Fuse the dense and BM25 restaurant rankings:

```
RRF(d) = Σ_i 1 / (k + rank_i(d)),   k = 60
```

- Only documents retrieved by at least one retriever are scored.
- Ties are broken by the better dense rank, then by rating.
- Attribute filters are applied before fusion, not after, so that filtered-out restaurants cannot influence ranks.
- `k=60` is the standard default and is not tuned on the test set. If tuned, it must be tuned on a validation split and reported.

### 9.5 Reranking

Model: `BAAI/bge-reranker-base` (cross-encoder).

- Rerank the top 20 fused candidates.
- Score query–document pairs at the restaurant level: use the best chunk per restaurant, or concatenate the top 2 chunks if they fit within the model's limit. The choice is fixed and reported.
- Final order: rerank score, then RRF score, then rating.
- Reranker latency is measured separately.
- The reranker is optional in the deployed app if latency is unacceptable; the evaluation reports both settings.

### 9.6 Evidence layer

Retrieval returns restaurants, not quotes. After reranking:

1. For each of the top 5 candidates, select up to 8 candidate evidence sentences from that restaurant's evidence pool.
2. Rank sentences by BM25 against the rewritten query plus aspect terms extracted from the query.
3. Deduplicate near-identical sentences.
4. Require the sentence to be from a review with a star rating that does not contradict the claim (e.g., do not use a 1★ complaint to support "great food").
5. Pass the top 2–3 sentences per restaurant to the generator, each with `review_id` and `stars`.

This keeps the prompt small and makes citations precise. It also prevents the generator from quoting a review that was never retrieved.

### 9.7 Ranking policy

Final ranking combines:

1. hard-constraint satisfaction (pass/fail; failures are removed),
2. rerank relevance,
3. evidence strength (number of supporting sentences, agreement with rating),
4. rating as a minor tie-breaker,
5. personalization adjustments (Section 12).

Rating alone must never determine the order. This is a stated design requirement because high-rated restaurants are not automatically relevant to a specific request.

### 9.8 Latency budget

Targets for the app (to be validated):

| Stage | Target p95 |
|---|---:|
| Intent + preference extraction (LLM or rules) | < 1.0 s |
| Dense retrieval | < 50 ms |
| BM25 retrieval | < 50 ms |
| Fusion + filtering | < 20 ms |
| Reranking (20 candidates) | < 500 ms |
| Evidence selection | < 100 ms |
| Generation (256 tokens) | < 8 s |
| Validation | < 300 ms |
| **End-to-end** | **< 10 s** |

Targets are engineering goals, not reported results. Actual values come from `results/tables/latency.csv`.

### 9.9 Retrieval metrics

Defined precisely in Section 16.3. At minimum: Success@5, Precision@5, MRR, nDCG@5, and latency, computed at the restaurant level against the gold set.

---
