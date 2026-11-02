# Appendix F — Glossary

> Part of the DineWise project plan. Operational reference for implementation.
> See [the plan index](../README.md) for the full reading order.

---


| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: retrieve context, then generate an answer grounded in it. |
| **Chunk** | A segment of a document small enough to embed and retrieve. |
| **Embedding** | A dense vector representation of text or an image. |
| **Qdrant** | A vector database with collections, payload filters, and exact/HNSW search. |
| **BM25** | A lexical ranking function based on term frequency and inverse document frequency. |
| **RRF** | Reciprocal Rank Fusion: merge ranked lists by summing reciprocal ranks. |
| **Cross-encoder** | A reranker that scores a query–document pair jointly. |
| **nDCG@k** | Normalized discounted cumulative gain at k, supporting graded relevance. |
| **MRR** | Mean reciprocal rank of the first relevant result. |
| **Success@k** | Whether at least one relevant item appears in the top k. |
| **Faithfulness** | The degree to which an answer is supported by the retrieved context. |
| **Hallucination** | A claim not supported by the supplied evidence. |
| **Abstention** | Refusing to answer when evidence is insufficient. |
| **LoRA** | Low-Rank Adaptation: train small adapter matrices instead of the full model. |
| **QLoRA** | LoRA on a quantized base model. |
| **INT8 / INT4** | Reduced-precision model weights. |
| **TTFT** | Time to first generated token. |
| **TPOT** | Time per output token. |
| **CLIP** | A model that maps images and text into a shared embedding space. |
| **Zero-shot classification** | Predicting classes without task-specific training, using text prompts. |
| **Tri-state attribute** | A value that is true, false, or unknown; unknown is not false. |
| **Gold set** | A manually judged query set used for evaluation. |
| **Ablation** | Removing or changing one component to measure its contribution. |

---

*End of plan.*
