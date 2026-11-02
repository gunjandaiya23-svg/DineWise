# DineWise: A Retrieval-Augmented Framework for Personalized and Explainable Restaurant Recommendations

**Report generated:** 2026-11-02 11:03 UTC
**Code state:** see the Git commit referenced in `results/runs/`
**Label status:** retrieval labels are **silver** (rubric-based), not human judgments.
A manual review sheet is provided at `data/gold/review_sheet.csv`.

---

## Abstract

DineWise is a retrieval-augmented recommendation assistant over a static Yelp-derived
restaurant dataset. It combines a structured catalogue, a multi-format document
corpus, dense and lexical retrieval, reciprocal rank fusion, optional cross-encoder
reranking, and a LangGraph workflow with programmatic answer validation and bounded
retry. The project evaluates chunking, retrieval, visual matching, lightweight
adaptation, and precision trade-offs on a fixed 15-query suite and a 12-query
silver gold set. The strongest measured results are CLIP zero-shot food
classification (top-1 82.2%) and hard-constraint compliance (100% of answered
queries). Dense retrieval alone does not align with a quality-oriented gold set;
BM25 and hybrid fusion perform better, and cross-encoder reranking did not improve
nDCG@5 under this labeling scheme.

## 1. Problem statement

Restaurant recommendation requires combining subjective review evidence, structured
attributes, budget, location, and user preferences. A plain language model cannot be
trusted to recall restaurant facts, ratings, or availability. A plain vector search
cannot enforce hard constraints or explain its choices. DineWise addresses both by
separating four concerns: structured facts, unstructured evidence, retrieval, and
generation with validation.

## 2. Objectives

1. Build a reproducible data pipeline from a messy multi-file dataset.
2. Demonstrate TXT, PDF, and DOCX document handling while preserving structured data.
3. Compare chunking strategies, retrieval modes, and reranking on one fixed query set.
4. Ground generated answers in retrieved evidence and refuse unsupported questions.
5. Measure the quality and efficiency effects of LoRA and reduced precision.
6. Report negative results honestly.

## 3. Dataset and corpus

The study uses the Yelp Multimodal Recommendation dataset (16,702 businesses,
837,559 reviews, 118,951 photo records). The pilot corpus is a five-city subset
filtered to food-service businesses; the current build uses 1 city
and 500 restaurants sorted by review count.

| Item | Value |
|---|---|
| Restaurants in corpus | 500 |
| Reviews joined | 75323 |
| With review summary | 500 |
| With photo summary | 490 |
| With price bucket | 500 |
| Cuisine coverage | 99.8% |
| Attribute parse rate | 1.0 |
| Venue records excluded | 17 |

Known dataset limits: no image files, no review dates, no dollar prices, no
structured dietary fields, and LLM-generated review/photo summaries.

## 4. Environment

| Item | Value |
|---|---|
| Python | 3.12.15 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8.19 GB |
| CUDA / torch | 12.8 / 2.11.0+cu128 |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Vector store | Qdrant at http://localhost:6333 |
| Generator | Qwen/Qwen2.5-1.5B-Instruct |
| Reranker | BAAI/bge-reranker-base |
| CLIP | openai/clip-vit-base-patch32 |

## 5. System architecture

Streamlit UI -> LangGraph workflow -> (BM25 + Qdrant + structured filters +
RRF) -> optional cross-encoder rerank -> Qwen generation -> programmatic
validation -> answer, retry, or fallback.

Design rules: facts come from structured data; policy text is never cited as
restaurant evidence; every claim must cite a business or review ID; abstention is
a valid outcome.

## 6. Documents and chunking

Three knowledge-base documents are generated: `restaurant_profiles.txt`,
`customer_review_insights.pdf`, and `dining_preferences.docx`. Each restaurant
profile includes structured fields, the review summary, the visual summary, and up
to eight raw review excerpts. Chunking is restaurant-aware and token-based; a chunk
cannot cross a restaurant boundary by construction.

### E5: chunk-size comparison (hybrid retrieval, silver labels)

| config | size | overlap | chunks | mean_tokens | chunks_with_multiple_markers | strict_success@k | strict_precision@k | strict_mrr | ndcg@k | points | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 200 | 30 | 4132 | 211.600 | 0 | 0.333 | 0.117 | 0.229 | 0.417 | 4132 | 19.530 |
| C2 | 400 | 60 | 2187 | 379.500 | 0 | 0.500 | 0.100 | 0.350 | 0.421 | 2187 | 19.930 |
| C3 | 600 | 90 | 1511 | 534.200 | 0 | 0.500 | 0.167 | 0.306 | 0.444 | 1511 | 18.000 |
| C4 | 800 | 120 | 1116 | 701.300 | 0 | 0.583 | 0.150 | 0.271 | 0.427 | 1116 | 14.750 |

Selected: **C3** by nDCG@5.

## 7. Retrieval

Modes compared: E1 BM25, E2 dense BGE + Qdrant, E3 hybrid RRF, E4 hybrid +
cross-encoder, and E4 plus a quality prior. Metrics are restaurant-level.
`strict` means the rubric's top-10; `any` means any constraint-satisfying result.

| Mode | Strict Success@5 | Strict P@5 | Strict MRR | nDCG@5 | Any Success@5 |
|---|---|---|---|---|---|
| bm25 | 0.250 | 0.100 | 0.183 | 0.406 | 1.000 |
| dense | 0.333 | 0.117 | 0.292 | 0.428 | 1.000 |
| hybrid | 0.500 | 0.167 | 0.306 | 0.444 | 1.000 |
| hybrid_rerank | 0.333 | 0.083 | 0.114 | 0.380 | 1.000 |
| hybrid_rerank_quality | 0.333 | 0.083 | 0.174 | 0.390 | 1.000 |

Interpretation: hybrid retrieval satisfies every hard constraint and improves
MRR over dense alone. The cross-encoder did not improve nDCG@5 against a
quality-oriented gold set; this is reported as a negative result rather than tuned
away. The labels are silver and require human review before strong claims.

### E12: exact vs HNSW storage backend

| Metric | Value |
|---|---|
| Collection | dinewise_chunks_C3 |
| Points | 1511 |
| Mean Recall@10 vs exact | 1.000 |
| Mean nDCG@5 (HNSW) | 0.423 |
| Mean exact latency | 4.51 ms |
| Mean HNSW latency | 4.08 ms |

## 8. Generation and workflow

The LangGraph workflow classifies scope, extracts constraints, retrieves, filters,
generates, validates, retries once if needed, and falls back otherwise.

| ID | Facet | Outcome | Passed | Retries | Hits | Seconds |
|---|---|---|---|---|---|---|
| Q01 | cuisine | answered | True | 0 | 5 | 9.18 |
| Q02 | budget | answered | True | 0 | 5 | 5.43 |
| Q03 | review_aspect | answered | True | 0 | 5 | 5.35 |
| Q04 | ambience | answered | True | 0 | 5 | 5.07 |
| Q05 | personalization | answered | True | 0 | 5 | 5.17 |
| Q06 | comparison | answered | True | 0 | 5 | 5.24 |
| Q07 | dessert | answered | True | 0 | 5 | 5.06 |
| Q08 | dietary | answered | True | 0 | 5 | 4.99 |
| Q09 | image | answered | True | 0 | 5 | 4.88 |
| Q10 | visual | answered | True | 0 | 5 | 5.03 |
| Q11 | ranking | answered | True | 0 | 5 | 4.99 |
| Q12 | multi_constraint | answered | True | 0 | 5 | 5.05 |
| Q13 | out_of_scope | refused | False | 0 | 0 | 0.0 |
| Q14 | out_of_scope | fallback | False | 0 | 0 | 0.0 |
| Q15 | unsupported | fallback | False | 0 | 0 | 0.0 |

{
  "mode": "llm",
  "queries": 15,
  "answered": 12,
  "refused": 1,
  "fallbacks": 3,
  "validation_passed": 12,
  "retries": 0,
  "mean_seconds": 4.36
}

## 9. Visual retrieval

The dataset has no image files. Component A retrieves restaurants from GPT-4o
visual summaries using BGE. Component B uses CLIP zero-shot Food-101
classification on streamed images and maps predicted classes to cuisine families
as a soft signal.

| Metric | Value |
|---|---|
| Visual documents | 490 |
| Strict Success@5 | 0.400 |
| Strict Precision@5 | 0.080 |
| Strict MRR | 0.125 |
| nDCG@5 | 0.377 |
| CLIP images | 202 |
| CLIP top-1 | 0.8218 |
| CLIP top-5 | 0.9505 |
| Mapped cuisine accuracy | 0.8409 |

CLIP results are class-level on a stratified subset; they do not establish
restaurant identity, authenticity, or food quality.

## 10. LoRA fine-tuning

Data: 400 grounded instruction examples, split 320/40/40 by queried restaurant.
Training: QLoRA, rank 8, alpha 16, dropout 0.05, learning rate 2e-4, 2 epochs,
batch 1 with gradient accumulation 4, max length 1024.

| System | Format valid | Grounded | Hard pass | Live claims | Abstention | Seconds |
|---|---|---|---|---|---|---|
| base_no_rag | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.79 |
| base_rag | 1.0 | 0.8 | 1.0 | 0.0 | 0.0 | 4.92 |
| lora_rag | 1.0 | 0.8 | 1.0 | 0.0 | 0.0 | 4.94 |
| lora_no_rag | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.86 |

## 11. Efficiency

TTFT is measured as prefill latency; TPOT is (total generation time - prefill) /
new tokens. Batch size is 1 and decoding is greedy.

| config | precision | adapter | load_seconds | ttft_ms | tpot_ms | tokens_per_second | peak_vram_gb | weights_gb |
|---|---|---|---|---|---|---|---|---|
| M1_fp16 | fp16 | False | 0.86 | 458.1 | 16.4 | 43.17 | 4.0 | 3.09 |
| M2_int8 | int8 | False | 2.67 | 467.5 | 72.6 | 13.38 | 2.69 | 1.78 |
| M3_int4 | int4 | False | 0.78 | 479.1 | 17.2 | 42.98 | 2.05 | 1.12 |
| M4_lora_fp16 | fp16 | True | 1.2 | 673.6 | 25.9 | 29.67 | 4.03 | 3.12 |
| M5_lora_int4 | int4 | True | 1.09 | 683.7 | 40.3 | 20.78 | 2.09 | 1.16 |

## 12. Behaviour on the 15-query suite

| ID | Facet | Constraints | Candidates | Top hit | Compliance |
|---|---|---|---|---|---|
| Q01 | cuisine | Philadelphia,Italian | 71 | L'Angolo Ristorante | ok |
| Q02 | budget | Philadelphia,$$ max | 439 | Kensington Quarters | ok |
| Q03 | review_aspect | - | 500 | a.kitchen | ok |
| Q04 | ambience | Philadelphia | 500 | The Good King Tavern | ok |
| Q05 | personalization | Philadelphia | 202 | City Tavern Restaurant | ok |
| Q06 | comparison | Philadelphia,Italian | 71 | L'Angolo Ristorante | ok |
| Q07 | dessert | Philadelphia | 500 | The Sweet Life Bakeshop | ok |
| Q08 | dietary | Philadelphia | 500 | Vedge | ok |
| Q09 | image | Philadelphia,Italian | 71 | Pizzeria Vetri | ok |
| Q10 | visual | Philadelphia | 500 | Fond | ok |
| Q11 | ranking | Philadelphia | 500 | Angelo's Pizzeria | ok |
| Q12 | multi_constraint | Philadelphia,Mexican,$$ max | 20 | Los Gallos Mexican Taqueria | ok |
| Q13 | out_of_scope | out-of-scope: live availability | - | - | refused |
| Q14 | out_of_scope | out-of-scope: 'Las Vegas' is not in the dataset, which covers Philadelphia. | - | - | refused |
| Q15 | unsupported | out-of-scope: The dataset does not record information about rooftop areas, pools, so this cannot be answered. | - | - | refused |

## 13. Limitations

1. **No human relevance judgments.** All retrieval metrics use a rubric-based
   silver set. The review sheet exists but is ungraded.
2. **One city, 500 restaurants.** The full study corpus is larger; results may not
   transfer.
3. **No images in the dataset.** Visual retrieval is over generated descriptions;
   CLIP is evaluated on external Food-101 images, not restaurant photos.
4. **LLM-generated summaries.** Review and photo summaries were produced upstream
   and may contain unsupported claims. Raw reviews are the higher-authority source.
5. **No dates, no dollar prices, no live data.** Recency, exact budgets, and
   availability cannot be evaluated.
6. **Dietary constraints are soft.** The system surfaces textual evidence but
   cannot guarantee safety.
7. **Small model.** Qwen2.5-1.5B limits answer quality; larger models may behave
   differently.
8. **Silver chunking result.** C3 won by nDCG@5, but differences are small and
   should be rechecked with human labels.
9. **Reranker negative result.** Under a quality-oriented gold set, reranking did
   not help; a relevance-oriented human judgment may reverse this.
10. **No pruning or distillation.** These were not performed and are not claimed.

## 14. Conclusion

The project delivers a working, reproducible RAG system with explicit refusal
behaviour, a graph workflow, a visual channel, a fine-tuning path, and an
efficiency comparison. The most defensible claims are about engineering and
constraint compliance, not about beating a baseline on human-judged relevance.
The next step is to replace silver labels with human judgments and re-run E1-E5.

## References

1. Yelp Open Dataset. https://www.yelp.com/dataset
2. Yelp-Multimodal-Recommendation. https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation
3. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
4. Cormack et al. (2009). Reciprocal Rank Fusion. SIGIR.
5. Järvelin & Kekäläinen (2002). Cumulated Gain-based Evaluation. ACM TOIS.
6. Robertson & Zaragoza (2009). BM25 and Beyond.
7. Hu et al. (2021). LoRA. ICLR.
8. Dettmers et al. (2023). QLoRA. NeurIPS.
9. Radford et al. (2021). CLIP. ICML.
10. Bossard et al. (2014). Food-101. ECCV.
11. Es et al. (2023). RAGAS. arXiv:2309.15217.
12. Qwen Team (2024). Qwen2.5 Technical Report.

## Reproducibility

```bash
make setup && make db-up && make check
make download && make prep && make docs
make index && make bm25
make queries && make experiments
.venv/bin/python scripts/11_build_goldset.py
.venv/bin/python scripts/12_retrieval_metrics.py
.venv/bin/python scripts/13_chunking_experiments.py
.venv/bin/python scripts/21_backend_experiment.py
.venv/bin/python scripts/14_run_graph.py
.venv/bin/python scripts/15_visual_experiments.py
.venv/bin/python scripts/16_build_lora_data.py
.venv/bin/python scripts/17_train_lora.py
.venv/bin/python scripts/18_eval_models.py
.venv/bin/python scripts/19_efficiency.py
.venv/bin/python scripts/20_build_report.py
```
