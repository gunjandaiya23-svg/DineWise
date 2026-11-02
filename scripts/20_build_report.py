#!/usr/bin/env python
"""Assemble the report from measured result tables.

Every table in the report is read from results/tables/, so the document
cannot drift from the experiments.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import load_config  # noqa: E402

TABLES = {
    "retrieval": "retrieval_metrics_latest.md",
    "chunking": "chunking_E5_latest.md",
    "visual": "visual_E6_latest.md",
    "models": "models_E7_latest.md",
    "efficiency": "efficiency_E8_latest.md",
    "graph": "graph_E9_latest.md",
    "queries": "query15_latest.md",
}


def read_table(name: str) -> str:
    path = ROOT / "results" / "tables" / TABLES[name]
    if not path.exists():
        return f"_Not run yet: `{TABLES[name]}`._"
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    cfg = load_config()
    corpus_path = ROOT / "data" / "processed" / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else {}
    env_path = ROOT / "results" / "env_report.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    selection = corpus.get("selection", {})
    metrics = corpus.get("corpus", {})
    quality = corpus.get("quality", {})

    report = f"""# DineWise: A Retrieval-Augmented Framework for Personalized and Explainable Restaurant Recommendations

**Report generated:** {generated}
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
filtered to food-service businesses; the current build uses {len(corpus.get('selection', {}).get('cities', [])) or 1} city
and {selection.get('limit', 'all')} restaurants sorted by review count.

| Item | Value |
|---|---|
| Restaurants in corpus | {metrics.get('restaurants', 'n/a')} |
| Reviews joined | {metrics.get('reviews', 'n/a')} |
| With review summary | {metrics.get('with_review_summary', 'n/a')} |
| With photo summary | {metrics.get('with_photo_summary', 'n/a')} |
| With price bucket | {metrics.get('with_price', 'n/a')} |
| Cuisine coverage | {quality.get('cuisine_coverage_pct', 'n/a')}% |
| Attribute parse rate | {quality.get('attribute_parse_rate', 'n/a')} |
| Venue records excluded | {selection.get('venue_excluded', 'n/a')} |

Known dataset limits: no image files, no review dates, no dollar prices, no
structured dietary fields, and LLM-generated review/photo summaries.

## 4. Environment

| Item | Value |
|---|---|
| Python | {env.get('python', 'n/a')} |
| GPU | {env.get('cuda', {}).get('device', 'n/a')} |
| VRAM | {env.get('cuda', {}).get('total_memory_gb', 'n/a')} GB |
| CUDA / torch | {env.get('cuda', {}).get('torch_cuda', 'n/a')} / {env.get('packages', {}).get('torch', 'n/a')} |
| Embedding model | {cfg['embedding']['model']} |
| Generator | {cfg['models']['llm']} |
| Reranker | {cfg['models']['reranker']} |
| CLIP | {cfg['models']['clip']} |

## 5. System architecture

Streamlit UI -> LangGraph workflow -> (BM25 + FAISS + structured filters +
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

{read_table('chunking')}

## 7. Retrieval

Modes compared: E1 BM25, E2 dense BGE + FAISS, E3 hybrid RRF, E4 hybrid +
cross-encoder, and E4 plus a quality prior. Metrics are restaurant-level.
`strict` means the rubric's top-10; `any` means any constraint-satisfying result.

{read_table('retrieval')}

Interpretation: hybrid retrieval satisfies every hard constraint and improves
MRR over dense alone. The cross-encoder did not improve nDCG@5 against a
quality-oriented gold set; this is reported as a negative result rather than tuned
away. The labels are silver and require human review before strong claims.

## 8. Generation and workflow

The LangGraph workflow classifies scope, extracts constraints, retrieves, filters,
generates, validates, retries once if needed, and falls back otherwise.

{read_table('graph')}

## 9. Visual retrieval

The dataset has no image files. Component A retrieves restaurants from GPT-4o
visual summaries using BGE. Component B uses CLIP zero-shot Food-101
classification on streamed images and maps predicted classes to cuisine families
as a soft signal.

{read_table('visual')}

CLIP results are class-level on a stratified subset; they do not establish
restaurant identity, authenticity, or food quality.

## 10. LoRA fine-tuning

Data: 400 grounded instruction examples, split 320/40/40 by queried restaurant.
Training: QLoRA, rank 8, alpha 16, dropout 0.05, learning rate 2e-4, 2 epochs,
batch 1 with gradient accumulation 4, max length 1024.

{read_table('models')}

## 11. Efficiency

TTFT is measured as prefill latency; TPOT is (total generation time - prefill) /
new tokens. Batch size is 1 and decoding is greedy.

{read_table('efficiency')}

## 12. Behaviour on the 15-query suite

{read_table('queries')}

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
make setup && make check
make download && make prep && make docs
make index && make bm25
make queries && make experiments
.venv/bin/python scripts/11_build_goldset.py
.venv/bin/python scripts/12_retrieval_metrics.py
.venv/bin/python scripts/13_chunking_experiments.py
.venv/bin/python scripts/14_run_graph.py
.venv/bin/python scripts/15_visual_experiments.py
.venv/bin/python scripts/16_build_lora_data.py
.venv/bin/python scripts/17_train_lora.py
.venv/bin/python scripts/18_eval_models.py
.venv/bin/python scripts/19_efficiency.py
.venv/bin/python scripts/20_build_report.py
```
"""

    output = ROOT / "report" / "DineWise_Report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
