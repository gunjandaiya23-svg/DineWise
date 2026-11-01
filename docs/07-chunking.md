# Chunking

> Part of the DineWise project plan. Restaurant-aware chunking, configurations, metadata, and metrics.
> See [the plan index](README.md) for the full reading order.

---

## 7. Chunking

### 7.1 Requirements

1. A chunk must never contain content from two restaurants.
2. Each chunk must carry `business_id`, `restaurant_name`, `city`, `cuisine_primary`, `price_range`, `source_file`, `document_type`, `section`, and `page`.
3. Chunk boundaries should follow section boundaries where possible.
4. Chunk length is measured in **tokens**, not characters.
5. The configuration must be a single parameter set so that C1–C4 differ only in size and overlap.

### 7.2 Algorithm

1. Build one `Document` per restaurant section:
   - profile section,
   - review-insight section,
   - visual section.
2. Split each section with a token-based recursive splitter:
   - separators: `["\n\n", "\n", ". ", " "]`,
   - length function: tokenizer of the embedding model (`bge-small-en-v1.5`),
   - `chunk_size` and `chunk_overlap` from configuration,
   - `keep_separator=True`.
3. Enforce a minimum chunk size of 80 tokens by merging a short tail into the previous chunk of the same restaurant.
4. Attach metadata after splitting, from the parent document. No metadata inference from text.
5. Assert that every chunk's `business_id` is unique within the chunk and equals the parent's.
6. Write `doc_chunks.parquet` plus a per-configuration manifest.

Implementation note: LangChain's `RecursiveCharacterTextSplitter` counts characters by default. Use `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` or a custom length function based on the BGE tokenizer. The plan standardizes on the BGE tokenizer so that chunk sizes match the embedding model's truncation limit.

### 7.3 Configurations

| Config | Chunk size (tokens) | Overlap | Overlap ratio | Purpose |
|---|---:|---:|---:|---|
| C1 | 200 | 30 | 15% | High precision, fragmented |
| C2 | 400 | 60 | 15% | Default hypothesis |
| C3 | 600 | 90 | 15% | More context |
| C4 | 800 | 120 | 15% | Long context, more noise |

All other settings are identical: same documents, same embedding model, same retrieval parameters, same queries.

Expected chunk counts for the study corpus, to be replaced by measurement:

- C1: ~28k–35k chunks,
- C2: ~18k–24k chunks,
- C3: ~12k–16k chunks,
- C4: ~9k–12k chunks.

### 7.4 Chunk metadata schema

```json
{
  "chunk_id": "B001::review::003",
  "business_id": "B001",
  "restaurant_name": "Restaurant ABC",
  "city": "Philadelphia",
  "state": "PA",
  "cuisine_primary": "Italian",
  "price_range": 2,
  "document_type": "review_insight",
  "section": "food_quality",
  "source_file": "customer_review_insights.pdf",
  "page": 142,
  "token_count": 387,
  "char_count": 1842,
  "chunk_config": "C2",
  "text": "..."
}
```

`document_type` ∈ {`profile`, `review_insight`, `visual`, `policy`}. Policy chunks have `business_id = null`.

### 7.5 Contamination guarantee

Because each restaurant is split independently, a chunk cannot cross a restaurant boundary. This is a structural property, not a hope. The verification is still performed:

- For every chunk, count distinct `[RESTAURANT ...]` markers in the text. Target: 1 for profile chunks, 0 or 1 for review/visual chunks that do not repeat the marker.
- Count chunks whose text contains a restaurant name other than their own, excluding policy chunks. Target: 0.
- Report the contamination rate in the chunking table. Any nonzero value is a bug.

### 7.6 Chunking metrics

- number of chunks per configuration,
- mean, median, min, and max token length,
- percentage below the 80-token floor after merging,
- percentage at or above the embedding model's truncation limit,
- cross-restaurant contamination rate,
- metadata completeness,
- index build time,
- retrieval metrics for each configuration (Section 16.3),
- end-to-end answer metrics for each configuration (Section 16.4).

The winning configuration is selected by nDCG@5 with faithfulness as a tie-breaker, not by a single metric. If differences are within the confidence interval, the plan reports that the configurations are indistinguishable and chooses the one with lower index cost.

---
