# Appendix E — Result table templates

> Part of the DineWise project plan. Operational reference for implementation.
> See [the plan index](../README.md) for the full reading order.

---


### E.1 Data audit

| Step | Restaurants | Reviews | Duplicates | Missing rating | Missing price | Notes |
|---|---:|---:|---:|---:|---:|---|
| Raw | 16,702 | 837,559 | | | | |
| Five-city filter | 6,946 | 414,952 | | | | |
| Food-service filter | 6,033 | 385,065 | | | | |
| Final cleaned | | | | | | |

### E.2 Chunking

| Config | Size | Overlap | Chunks | Mean tokens | Min | Max | Cross-restaurant | Metadata complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 200 | 30 | | | | | | |
| C2 | 400 | 60 | | | | | | |
| C3 | 600 | 90 | | | | | | |
| C4 | 800 | 120 | | | | | | |

### E.3 Retrieval

| Exp | Config | Success@5 | P@5 | MRR | nDCG@5 | Latency p50 | Latency p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | BM25 | | | | | | |
| E2 | Dense | | | | | | |
| E3 | Hybrid | | | | | | |
| E4 | + rerank | | | | | | |

### E.4 Generation

| System | Format valid | Constraint satisfaction | Faithfulness | Unsupported claims | Fallback correctness |
|---|---:|---:|---:|---:|---:|
| Base, no RAG | | | | | |
| + RAG | | | | | |
| + RAG + LoRA | | | | | |
| + LoRA, no RAG | | | | | |

### E.5 Efficiency

| Config | Disk | Peak VRAM | TTFT | TPOT | Tokens/s | nDCG@5 | Faithfulness |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | | | | | | | |
| INT8 | | | | | | | |
| INT4 | | | | | | | |
| LoRA-4bit | | | | | | | |

---
