# Deliverables and report

> Part of the DineWise project plan. Deliverables, report structure, and figures.
> See [the plan index](README.md) for the full reading order.

---

## 22. Deliverables and report structure

### 22.1 Deliverables

1. **Code repository** with `src/`, `scripts/`, `tests/`, `config/`, and a working `Makefile`.
2. **Notebooks** for the five experiment narratives.
3. **Streamlit app** demonstrating recommendation, comparison, and visual search.
4. **Results** in `results/runs/` and `results/tables/`, with manifests.
5. **Report** (~25–35 pages) following the structure below.
6. **Data card** describing the derived dataset and its limitations.
7. **Demo** (short screen recording or live walkthrough).

### 22.2 Report structure

1. Abstract
2. Problem statement
3. Objectives
4. Background theory: RAG, embeddings, BM25, RRF, cross-encoders, LoRA, quantization
5. Dataset description and audit
6. System architecture
7. Tools and libraries (with versions)
8. Data preprocessing
9. Document generation, loading, and chunking
10. Embedding and vector indexing
11. Retrieval strategy
12. LLM integration and prompt design
13. LangGraph workflow
14. Personalization
15. Visual retrieval
16. Fine-tuning and quantization
17. Evaluation methodology
18. Experimental results (E0–E11)
19. Observation tables
20. Discussion
21. Limitations and threats to validity
22. Conclusion and future work
23. References
24. Appendix: prompts, schemas, additional tables

### 22.3 Figures to produce

- dataset profile and cleaning funnel,
- architecture diagram,
- chunk length distributions for C1–C4,
- retrieval metric comparison across E1–E5,
- latency breakdown by node,
- faithfulness vs constraint satisfaction scatter,
- VRAM and tokens/second across precision configurations,
- example answer with citations.

---
