# Embeddings and indexes

> Part of the DineWise project plan. BGE, FAISS, BM25, CLIP indexing, caching, and metrics.
> See [the plan index](README.md) for the full reading order.

---

## 8. Embeddings and indexes

### 8.1 Text embeddings

Model: `BAAI/bge-small-en-v1.5` (384 dimensions, ~33M parameters).

Rules:

- L2-normalize embeddings and use inner product (`IndexFlatIP`) so that the score is cosine similarity.
- Add the BGE retrieval instruction to **queries only**: `Represent this sentence for searching relevant passages: `. Documents are embedded without the prefix.
- Maximum sequence length: 512 tokens. Chunks longer than this are truncated by the model; the chunking experiment must not exceed 512 tokens without documenting truncation.
- Batch size 64 during indexing, adjusted downward if VRAM is tight. Embedding runs on GPU in FP16.
- Cache embeddings to `data/indexes/embeddings/<model>/<sha256(chunk_texts)>.npy` so that repeated experiments do not recompute them.
- Record the exact model revision/hash in the run manifest.

### 8.2 FAISS index

- Use `faiss.IndexFlatIP` for exact search. The corpus is small enough (tens of thousands of vectors) that approximate indexes are unnecessary and would confound the retrieval experiment.
- Persist as `data/indexes/faiss/<config>/index.faiss` plus `chunk_ids.json`.
- Expected size: ~25k vectors × 384 dimensions × 4 bytes ≈ 38 MB.
- Build time and query latency are measured, not assumed.
- Do not use LangChain's FAISS wrapper for the evaluation path; use `faiss` directly and keep a thin adapter. LangChain's wrapper is acceptable for the tutorial notebook but obscures score semantics.

### 8.3 BM25 index

- Tokenization: lowercase, Unicode-aware word tokens, stopword removal using a fixed English list, no stemming by default (record the choice).
- Use `rank_bm25.BM25Okapi` with `k1=1.5`, `b=0.75`.
- Build one BM25 index over the same chunk texts as the dense index, so that hybrid fusion compares the same units.
- Persist the tokenized corpus and document IDs with `pickle` or `joblib`.
- BM25 is also built at the **restaurant level** for the evidence layer, but that index is per-candidate and constructed on the fly.

### 8.4 Visual index

Two separate artifacts:

1. **Text-to-visual retrieval:** BGE embeddings of the `photo_summay.csv` fields (`summary`, `venue_style`, `indoor_color_tone`, `special_features`, `target_audience`). One document per business, metadata `document_type=visual`. This is text retrieval over visual descriptions, not image retrieval, and is labeled as such.
2. **CLIP text embeddings:** optional, for comparing short text queries with CLIP text embeddings of photo captions. Used only in the E6 ablation. CLIP text embeddings are not mixed into the BGE index; scores are not comparable across spaces.

### 8.5 Index metrics

- embedding generation time,
- throughput (chunks/second),
- embedding dimension,
- index build time,
- index size on disk,
- retrieval latency p50/p95 over the gold queries,
- peak GPU memory during indexing.

---
