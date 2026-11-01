# Generation

> Part of the DineWise project plan. Model loading, prompts, JSON schema, constrained decoding, validators.
> See [the plan index](README.md) for the full reading order.

---

## 10. Generation

### 10.1 Model

Base generator: `Qwen/Qwen2.5-1.5B-Instruct`.

Rationale:

- small enough for 8 GB VRAM in FP16 and 4-bit,
- instruction-tuned,
- 32k context, sufficient for facts plus evidence,
- supports LoRA/QLoRA,
- permits a clean precision comparison.

Fallback if output quality is unacceptable after prompt engineering: `Qwen2.5-3B-Instruct` in 4-bit. The switch must be documented and the full experiment matrix rerun, not selectively reported.

Loading rules:

- FP16 baseline: `torch_dtype=torch.float16`, `device_map="auto"`.
- 8-bit: `load_in_8bit=True` via bitsandbytes.
- 4-bit: NF4 with double quantization, compute dtype bfloat16.
- Set a fixed seed and greedy decoding for evaluation runs.
- Use the model's chat template, not a hand-written format.
- Unload the reranker and CLIP before loading the LLM when VRAM is tight; the model registry handles this.

### 10.2 Prompt architecture

The prompt has five blocks, in this order:

1. **System / role**: DineWise is a restaurant recommendation assistant grounded in a static dataset.
2. **Policy**: the full contents of `dining_preferences.docx`, marked as rules.
3. **Facts**: structured records for the candidate restaurants, injected from the database.
4. **Evidence**: retrieved review sentences and visual summaries, each with an ID.
5. **Task**: the user query, the extracted constraints, and the required output format.

Grounding rules in the system prompt:

- use only the supplied records and evidence;
- never invent names, ratings, prices, addresses, dietary facts, or amenities;
- never present dataset ratings or prices as live;
- never claim live availability;
- if evidence is insufficient, say so and return fewer recommendations;
- cite a source ID for every factual sentence;
- keep explanations to 2–3 sentences per restaurant.

The policy block is injected in full rather than retrieved. This prevents a retrieved fragment such as "prefer quiet restaurants" from being misinterpreted as a fact about a restaurant.

A compact example of the fact block:

```
[CANDIDATE 1]
business_id: B001
name: Restaurant ABC
city: Philadelphia, PA
cuisine: Italian, Pizza, Vegetarian
price_bucket: 2 ($$)
rating: 4.5 (dataset average over 312 reviews)
attributes: outdoor_seating=yes; noise=average; romantic=unknown;
            good_for=dinner,dessert
review_summary: ...
visual_summary: ...
evidence:
  [E1] review_id=R123 stars=5 "The vegetarian lasagna was excellent..."
  [E2] review_id=R456 stars=4 "Service was slow on a Saturday..."
```

### 10.3 Output schema

The model must return JSON matching this schema (see Appendix B for the full version):

```json
{
  "recommendations": [
    {
      "business_id": "B001",
      "name": "Restaurant ABC",
      "cuisine": "Italian",
      "price_bucket": 2,
      "rating": 4.5,
      "evidence": [
        {"review_id": "R123", "quote": "The vegetarian lasagna was excellent..."}
      ],
      "explanation": "Matches your vegetarian and budget constraints; reviewers praise the vegetarian options.",
      "sources": ["B001", "R123"]
    }
  ],
  "insufficient_evidence": false,
  "notes": "Ratings and prices are from the dataset, not live."
}
```

Constraints:

- `business_id` must be one of the supplied candidates.
- `rating` and `price_bucket` must match the database exactly.
- `evidence[].quote` must be a substring of a supplied evidence sentence, up to whitespace normalization.
- `sources` must contain the business ID and any cited review IDs.
- If no candidate satisfies the request, return an empty list and `insufficient_evidence: true`.

### 10.4 Constrained decoding and repair

1. Prefer schema-constrained generation. Options, in order:
   - `outlines` or `lm-format-enforcer` with a JSON schema, if compatible with the installed transformers version;
   - a grammar-constrained backend;
   - a strict JSON-only prompt with a repair loop.
2. If parsing fails, attempt one repair: strip code fences, fix trailing commas, and retry with the parse error appended.
3. If repair fails, log the raw output and return a structured error to the graph, which triggers the retry/fallback path.
4. Record `format_valid` for every generation. Format validity is a reported metric, not a hidden retry.

### 10.5 Validators

Validation is programmatic first and semantic second.

**Programmatic checks (hard failures):**

1. Every `business_id` exists in the candidate set.
2. Every name matches the database for that ID.
3. Every rating equals the database value (exact, one decimal).
4. Every price bucket equals the database value.
5. Every hard constraint is satisfied by the returned restaurant.
6. Every cited review ID exists and is associated with that restaurant.
7. Every quote is a substring of the supplied evidence.
8. No claim of live availability, current price, or booking.
9. The number of recommendations is between 0 and 5.

**Semantic checks (soft failures):**

10. Each explanation sentence is entailed by the cited evidence or structured facts (NLI model).
11. The answer addresses the user's actual request (relevance).
12. The answer acknowledges missing information when applicable.
13. The explanation does not overstate uncertain attributes (e.g., "romantic" when the attribute is unknown).

Soft failures do not automatically discard the answer. They are counted and reported, and they trigger a retry only when they exceed a configured threshold or when a hard constraint is violated.

**NLI model:** `MoritzLaurer/DeBERTa-v3-base-mnli` or an equivalent three-class NLI model. Premise = evidence + fact block; hypothesis = explanation sentence. Entailment probability below 0.5 flags the sentence. The model choice and threshold are recorded; the threshold is calibrated on a small manually labeled set, not guessed.

### 10.6 Failure handling

| Failure | Action |
|---|---|
| Invalid JSON | One repair attempt, then retry at graph level |
| Hallucinated restaurant | Drop the item; if all items invalid, retry |
| Rating/price mismatch | Drop the item; log a hard-failure count |
| Hard constraint violated | Retry once with a clarifying retrieval; never silently relax |
| Insufficient evidence | Return fewer results or fallback |
| Model OOM | Unload other models, retry in 4-bit, log the downgrade |
| Timeout | Fallback with a clear message |

Every failure is logged with a `run_id`, query ID, and stage.

---
