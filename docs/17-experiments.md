# Experiment matrix

> Part of the DineWise project plan. E0-E11, dependencies, ordering, and observation templates.
> See [the plan index](README.md) for the full reading order.

---

## 17. Experiment matrix

Every experiment changes one major component. The corpus, gold set, and remaining settings stay fixed. Each row produces one record in `results/runs/` and one row in `results/tables/experiments.csv`.

| ID | Variable | Fixed | Metrics | What it proves |
|---|---|---|---|---|
| E0 | Data pipeline (no model) | — | audit table, parse rate, coverage | Data quality baseline |
| E1 | BM25 only | Corpus, queries | Success@5, P@5, MRR, nDCG@5, latency | Lexical baseline |
| E2 | Dense only (BGE + Qdrant) | Same | Same | Semantic retrieval benefit |
| E3 | Hybrid RRF (dense + BM25) | Same | Same | Fusion benefit |
| E4 | Hybrid + cross-encoder rerank | Same | Same + rerank latency | Ranking improvement vs cost |
| E5 | Chunking C1–C4 (on the best retriever from E1–E4) | Same queries, same embedding model | Chunk stats + retrieval + faithfulness | Chunking effect |
| E6 | Visual retrieval: off / text-to-visual / + CLIP zero-shot | Same | Visual Success@5, cuisine accuracy, end-to-end nDCG | Value and limits of visual context |
| E7 | Base vs RAG vs RAG+LoRA vs LoRA-only | Same | Faithfulness, constraint satisfaction, format validity, unsupported claims | Adaptation effect |
| E8 | FP16 / INT8 / INT4 / LoRA-4bit | Same prompts | TTFT, TPOT, tok/s, peak VRAM, disk, quality | Efficiency trade-off |
| E9 | Linear pipeline vs full LangGraph (validator + retry + fallback) | Same | Unsupported-claim rate, fallback accuracy, retry rate, latency | Value of orchestration |
| E10 | Main corpus vs held-out cities (Nashville, Reno) | Same method | Metric degradation | Generalization |
| E11 | Profile off vs on | Same | Preference satisfaction, nDCG@5 | Personalization value |
| E12 | Qdrant exact vs HNSW | Same collection, queries | Recall@10, latency, nDCG@5 | Storage backend trade-off |

Experiment dependencies:

- E1–E5 require the gold set.
- E6 requires the visual gold subset.
- E7 requires the LoRA adapter and the 120-query set.
- E8 requires a fixed prompt set and a stable measurement harness.
- E9 requires both a linear runner and the graph runner.
- E10 requires a second index built over the held-out cities.
- E11 requires the profile definitions.
- E12 requires the chunk collection and the gold queries.

Ordering: E0 → E1–E4 → E5 → E9 (with the best retriever) → E6 → E7 → E8 → E10 → E11. This front-loads the retrieval contribution, which is the strongest part of the project.

Observation table template (to be filled with measured values only):

| Exp | Config | Success@5 | P@5 | MRR | nDCG@5 | Faithfulness | Latency p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | BM25 | | | | | | |
| E2 | Dense | | | | | | |
| E3 | Hybrid | | | | | | |
| E4 | Hybrid + rerank | | | | | | |
| E5 | Best chunking | | | | | | |
| E6 | + visual | | | | | | |
| E7 | + LoRA | | | | | | |
| E9 | Full graph | | | | | | |

---
