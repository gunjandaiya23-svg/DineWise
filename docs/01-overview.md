# Overview, scope and research questions

> Part of the DineWise project plan. How to use this document, the executive summary, and the research questions.
> See [the plan index](README.md) for the full reading order.

---

## Project metadata

| Field | Value |
|---|---|
| Project | DineWise: A Retrieval-Augmented Language Model Framework for Personalized and Explainable Restaurant Recommendations |
| Document version | 1.0 |
| Date | 2 November 2026 |
| Status | Implementation started |
| Environment | Local workstation, RTX 4060 Laptop (8 GB VRAM), Linux |
| Duration | One semester (~16 weeks) |

---


## 0. How to use this document

This document is the single source of truth for the project. It contains:

- the verified facts about the environment and dataset (Section 2),
- the scope and research questions (Section 3),
- the architecture and every component specification (Sections 4–15),
- the evaluation protocol (Section 16) and experiment matrix (Section 17),
- the engineering standards and repository layout (Section 18),
- the phased schedule with acceptance criteria (Section 19),
- risks, limitations, deliverables, and references (Sections 20–24),
- operational appendices: metadata schema, output schema, prompt templates, test queries, result-table templates, glossary.

Rules for keeping it useful:

1. Every number that appears in the final report must be produced by a script and stored under `results/`. This document contains only verified dataset facts and planned targets.
2. When a decision changes, update this document in the same commit as the code change.
3. Distinguish clearly between **measured**, **planned**, and **assumed**. The dataset statistics in Section 2 are measured. Timings and memory estimates are explicitly marked as estimates to be replaced by measurements.
4. Do not describe the system as a booking platform, a live availability service, or a production recommender. It is a research prototype over a static dataset.

A checklist of phase-level acceptance criteria is given in Section 19. Each phase is complete only when its acceptance tests pass.

---

---

## 1. Executive summary

DineWise is a restaurant discovery and recommendation assistant built on a retrieval-augmented generation (RAG) pipeline. A user states preferences in natural language — cuisine, city, budget, dietary needs, ambience, or an uploaded food image — and the system returns a ranked list of restaurants with:

- name, city, cuisine, price bucket, and dataset rating,
- supporting quotes from customer reviews with source identifiers,
- a short explanation of why each result matches the request,
- visual-context information derived from photo descriptions,
- an explicit statement when the dataset does not contain enough evidence.

The system is evaluated as a laboratory experiment, not demonstrated as a product. The central claim under test is:

> Document chunking strategy, hybrid retrieval, and lightweight LLM adaptation each measurably affect the relevance, factual grounding, and computational efficiency of restaurant recommendations, and those effects can be isolated with a controlled ablation study.

The project contributes four things:

1. A reproducible data-engineering pipeline that turns a messy multi-file Yelp-derived dataset into a validated restaurant catalogue and a multi-format document corpus (TXT, PDF, DOCX, CSV).
2. A retrieval pipeline combining dense embeddings, BM25, structured attribute filters, reciprocal rank fusion, and cross-encoder reranking, with restaurant-level evaluation.
3. A LangGraph workflow with intent classification, preference extraction, evidence validation, bounded retry, and safe fallback.
4. A controlled experiment matrix covering chunking, retrieval strategy, visual retrieval, LoRA adaptation, and FP16/INT8/INT4 efficiency, with programmatic grounding checks rather than LLM-as-judge alone.

The project explicitly does **not** claim to perform pruning or knowledge distillation, and does not claim that quantization is uniformly faster. Those are separate techniques and separate experiments.

---

---

## 3. Scope and research questions

### 3.1 In scope

- Restaurant discovery by cuisine, city, budget bucket, rating, and structured attributes.
- Review-grounded recommendations: retrieving and quoting review evidence.
- Ambience and occasion matching using structured attributes and review/photo summaries.
- Comparison of two named restaurants on food, service, ambience, and value, grounded in evidence.
- Visual-context search via textual photo summaries ("romantic interior", "outdoor patio").
- Uploaded food image → CLIP zero-shot food class → cuisine hint → restaurant filtering.
- Explicit personalization through a user profile and in-session feedback.
- Abstention when the dataset lacks evidence.
- Evaluation of retrieval, generation, recommendation ranking, and efficiency.
- A Streamlit interface for demonstration.

### 3.2 Out of scope

- Live availability, reservations, wait times, or booking.
- Real-time price or menu data.
- Scraping Yelp or downloading copyrighted restaurant photos.
- Pruning and knowledge distillation (mentioned only as future work).
- Collaborative filtering or training a recommender from user–item interactions.
- Multi-turn conversational memory beyond the current session.
- Languages other than English.
- Mobile deployment or authentication.

### 3.3 Research questions and hypotheses

**RQ1 — Chunking.** How does chunk size and overlap affect retrieval quality and answer grounding for restaurant documents?

- H1a: Restaurant-aware chunking with 400 tokens and 15% overlap outperforms 200, 600, and 800 tokens on nDCG@5.
- H1b: Smaller chunks increase retrieval precision but fragment review evidence, reducing faithfulness.
- H1c: Chunk size has a larger effect on retrieval metrics than on end-to-end answer quality, because reranking and generation absorb some retrieval noise.

**RQ2 — Retrieval.** Does hybrid retrieval with reranking outperform single-mode retrieval?

- H2a: Dense retrieval outperforms BM25 on natural-language preference queries.
- H2b: Hybrid RRF outperforms both dense and BM25 alone on nDCG@5.
- H2c: Cross-encoder reranking improves precision at the top of the list but adds latency that may not be justified.

**RQ3 — Grounding and adaptation.** Does lightweight adaptation improve recommendation quality without degrading factual grounding?

- H3a: RAG improves faithfulness over the base LLM without retrieval.
- H3b: LoRA on grounded instruction data improves format compliance and explanation quality, but does not improve factual accuracy beyond what retrieval provides.
- H3c: LoRA without retrieval increases unsupported claims relative to RAG.

**RQ4 — Efficiency.** What is the quality–efficiency trade-off between FP16, INT8, INT4, and LoRA-4bit?

- H4a: 4-bit quantization reduces VRAM and disk size substantially with a small quality drop.
- H4b: 4-bit does not necessarily reduce latency at batch size 1; the effect depends on dequantization overhead and memory bandwidth.
- H4c: LoRA adapters add negligible inference cost when merged.

**RQ5 — Personalization and visual context.** Do the profile and visual channels add measurable value?

- H5a: Applying an explicit user profile improves preference satisfaction without reducing relevance.
- H5b: Textual visual summaries improve ambience-related queries where structured attributes are missing or sparse.
- H5c: CLIP zero-shot cuisine inference is useful as a soft signal but not reliable enough to be a hard filter.

### 3.4 Contributions

1. A validated, reproducible data-engineering pipeline for a messy multi-file restaurant dataset.
2. A restaurant-aware chunking scheme with a structural guarantee against cross-restaurant contamination.
3. A hybrid retrieval and validation architecture with a bounded retry policy and explicit abstention.
4. A controlled ablation study across chunking, retrieval, visual context, adaptation, and precision.
5. A fully programmatic grounding checker that does not rely solely on LLM-as-judge.

---
