# Embeddings and indexes

> Part of the DineWise project plan. BGE, Qdrant, BM25, CLIP indexing, caching, and metrics.
> See [the plan index](README.md) for the full reading order.

---

## 8. Embeddings and indexes

### 8.1 Text embeddings

Model: `BAAI/bge-small-en-v1.5` (384 dimensions, ~33M parameters).

Rules:

- L2-normalize embeddings; Qdrant uses cosine distance, so the score is cosine similarity.
- Add the BGE retrieval instruction to **queries only**: `Represent this sentence for searching relevant passages: `. Documents are embedded without the prefix.
- Maximum sequence length: 512 tokens. Chunks longer than this are truncated by the model; the chunking experiment must not exceed 512 tokens without documenting truncation.
- Batch size 64 during indexing, adjusted downward if VRAM is tight. Embedding runs on GPU in FP16.
- Cache embeddings to `data/indexes/embeddings/<model>/<sha256(chunk_texts)>.npy` so that repeated experiments do not recompute them.
- Record the exact model revision/hash in the run manifest.

### 8.2 Qdrant collections (implemented)

Qdrant runs as a local Docker service with a persistent volume at `data/qdrant`.

- Start it with `make db-up`; check with `make db-status`.
- One collection per chunk configuration: `dinewise_chunks_c1` … `dinewise_chunks_c4`.
- One collection for visual descriptions: `dinewise_visual`.
- Vector config: 384 dimensions, cosine distance.
- Point IDs are deterministic UUIDv5 values derived from `chunk_id`; `chunk_id` is preserved in the payload.
- Payloads carry the chunk text plus filterable metadata: `business_id`, `city`, `cuisine_primary`, `cuisine_tags`, `price_range`, `stars`, `review_count`, `document_type`, and the restaurant attributes used by filters.
- Payload indexes are created for `city`, `cuisine_primary`, `cuisine_tags`, `document_type`, `business_id`, `price_range`, and `stars`.
- Experiments default to `vector_store.exact: true` so results are reproducible. The app may use HNSW (`exact: false`, `hnsw_ef`).
- Ingest with `scripts/05_build_index.py`, which verifies `count() == len(chunks)` and writes a manifest.
- A derived `data/indexes/chunks/<config>.parquet` cache is kept only for BM25; vectors and payloads are canonical in Qdrant.

### 8.3 BM25 index

- Tokenization: lowercase, Unicode-aware word tokens, stopword removal using a fixed English list, no stemming by default (record the choice).
- Use `rank_bm25.BM25Okapi` with `k1=1.5`, `b=0.75`.
- Build one BM25 index over the same chunk texts as the vector collection, so that hybrid fusion compares the same units.
- Persist the tokenized corpus and document IDs with `pickle` or `joblib`.
- BM25 is lexical retrieval and is not stored in Qdrant.

### 8.4 Visual index

Two separate artifacts:

1. **Text-to-visual retrieval:** BGE embeddings of the `photo_summay.csv` fields (`summary`, `venue_style`, `indoor_color_tone`, `special_features`, `target_audience`), stored in `dinewise_visual` with one point per business. This is text retrieval over visual descriptions, not image retrieval, and is labeled as such.
2. **CLIP text embeddings:** held in memory for the E6 ablation. They are model parameters, not dataset vectors, and are not mixed into the BGE collection.

### 8.5 Index metrics

- embedding generation time,
- throughput (chunks/second),
- embedding dimension,
- collection point count,
- ingest time,
- retrieval latency p50/p95 over the gold queries,
- exact vs HNSW recall and latency (E12),
- peak GPU memory during embedding.

---
