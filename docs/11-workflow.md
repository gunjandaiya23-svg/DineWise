# LangGraph workflow

> Part of the DineWise project plan. State, nodes, edges, retry semantics, and workflow metrics.
> See [the plan index](README.md) for the full reading order.

---

## 11. LangGraph workflow

### 11.1 State

```python
class DineWiseState(TypedDict, total=False):
    query: str
    profile: dict
    intent: str                  # recommendation | comparison | visual | out_of_scope
    constraints: dict            # city, cuisines, max_price, min_rating, attributes, dietary
    soft_preferences: list[str]
    rewritten_query: str
    retrieved: list[dict]        # chunk-level hits
    candidates: list[dict]       # restaurant-level candidates
    evidence: dict[str, list[dict]]
    answer: dict                 # parsed JSON
    validation: dict             # hard_failures, soft_failures, scores
    retry_count: int
    retrieval_expanded: bool
    timings: dict
    errors: list[str]
```

### 11.2 Nodes

| Node | Function | Implementation notes |
|---|---|---|
| `classify_intent` | Recommendation, comparison, visual, or out-of-scope | Rule-based first; LLM fallback only for ambiguous cases. Comparison queries require extracting two restaurant names. |
| `extract_preferences` | Produce the typed constraint object | Hybrid: regex/lexicon for city, price words, cuisine, dietary terms; LLM for free-form phrasing. Validate against known values. |
| `rewrite_query` | Retrieval-friendly query | Remove conversational filler; add cuisine/city terms; keep dish and aspect terms. |
| `retrieve` | Dense + BM25 + optional visual | Returns restaurant-level candidates with scores. |
| `filter_candidates` | Apply hard constraints | Programmatic SQL/pandas filter against `restaurants.parquet`. |
| `rerank` | Cross-encoder ordering | Top 20 → top 5. |
| `generate_answer` | LLM call with schema | Facts and evidence injected; no free-form browsing. |
| `review_answer` | Programmatic + semantic validation | Hard failures trigger retry; soft failures are counted. |
| `retry` | Expand retrieval once | Relax soft preferences, broaden the query, keep hard constraints. |
| `fallback` | Safe insufficiency response | State what is missing; optionally offer constraint-satisfying alternatives with a caveat. |

### 11.3 Graph edges

```
START → classify_intent → extract_preferences → rewrite_query → retrieve
      → filter_candidates → rerank → generate_answer → review_answer
review_answer --supported--> END
review_answer --insufficient & retry_count == 0--> retry → retrieve
retry → ... → review_answer
review_answer --insufficient & retry_count == 1--> fallback → END
classify_intent --out_of_scope--> fallback → END
```

### 11.4 Retry semantics

- Maximum one retry per request. Enforced in state and in a test.
- The retry may:
  - broaden the lexical query,
  - add synonyms for cuisine or dish terms,
  - relax soft preferences (e.g., "romantic" if the attribute is unknown),
  - expand the candidate pool from 20 to 40 before reranking.
- The retry may **not**:
  - change the city,
  - exceed the budget bucket,
  - drop a dietary requirement,
  - invent evidence,
  - relax an explicit hard attribute the user requested.
- If the retry fails, the fallback states which constraint could not be satisfied.

### 11.5 Workflow metrics

- intent classification accuracy (macro-F1 over the gold test set),
- preference extraction precision, recall, and F1 per field,
- retrieval success percentage,
- retry rate,
- retry success rate,
- fallback accuracy on out-of-scope queries,
- end-to-end latency p50/p95,
- node-level latency breakdown.

---
