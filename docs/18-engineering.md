# Engineering plan

> Part of the DineWise project plan. Repository layout, configuration, manifests, tests, notebooks, app.
> See [the plan index](README.md) for the full reading order.

---

## 18. Engineering plan

### 18.1 Repository layout

```
DineWise/
├── PLAN.md
├── README.md
├── pyproject.toml
├── requirements.lock
├── Makefile
├── config/
│   ├── config.yaml
│   ├── cuisine_taxonomy.yaml
│   └── prompts/
│       ├── system.txt
│       ├── extract_preferences.txt
│       ├── rewrite_query.txt
│       └── generate_answer.txt
├── data/
│   ├── raw/                  # downloaded CSVs, gitignored
│   ├── processed/            # parquet tables, gitignored
│   ├── docs/                 # TXT, PDF, DOCX
│   └── indexes/              # Qdrant manifests, BM25, chunk cache
├── src/dinewise/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── registry.py           # model loading/unloading
│   ├── data/
│   │   ├── download.py
│   │   ├── attributes.py
│   │   ├── cuisine.py
│   │   ├── clean.py
│   │   ├── build_tables.py
│   │   └── build_docs.py
│   ├── rag/
│   │   ├── loaders.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── bm25_index.py
│   │   ├── collapse.py
│   │   ├── fusion.py
│   │   ├── rerank.py
│   │   └── evidence.py
│   ├── llm/
│   │   ├── loader.py
│   │   ├── prompts.py
│   │   ├── generate.py
│   │   └── validate.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── image/
│   │   ├── clip.py
│   │   └── visual_index.py
│   ├── eval/
│   │   ├── goldset.py
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── stats.py
│   │   └── run_experiments.py
│   └── app/
│       └── components.py
├── scripts/
│   ├── 00_setup_check.py
│   ├── 01_prepare_data.py
│   ├── 02_build_docs.py
│   ├── 03_build_indexes.py
│   ├── 04_run_chunking_experiments.py
│   ├── 05_run_retrieval_experiments.py
│   ├── 06_run_generation.py
│   ├── 07_run_lora.py
│   ├── 08_run_efficiency.py
│   └── 09_build_tables.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_chunking_experiment.ipynb
│   ├── 03_retrieval_experiment.ipynb
│   ├── 04_generation_eval.ipynb
│   └── 05_efficiency.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_attributes.py
│   ├── test_cuisine.py
│   ├── test_chunking.py
│   ├── test_filters.py
│   ├── test_fusion.py
│   └── test_validators.py
├── results/
│   ├── runs/
│   ├── tables/
│   └── figures/
├── report/
│   └── DineWise_Report.md
└── docs/
    └── data_card.md
```

Notebooks are thin: they import `dinewise`, call functions, display tables, and save figures. No business logic lives only in a notebook. This keeps the code testable and reproducible.

### 18.2 Configuration

A single `config/config.yaml` holds all experiment parameters. It is loaded into a dataclass and hashed for the run manifest. Example:

```yaml
project:
  seed: 42
  results_dir: results

data:
  raw_dir: data/raw
  processed_dir: data/processed
  cities: [Philadelphia, Indianapolis, Tampa, New Orleans, Tucson]
  heldout_cities: [Nashville, Reno]
  min_reviews_for_corpus: 10
  max_evidence_per_restaurant: 120

docs:
  profile_file: restaurant_profiles.txt
  review_pdf: customer_review_insights.pdf
  policy_docx: dining_preferences.docx
  pdf_cache: true

chunking:
  active: C2
  configs:
    C1: {size: 200, overlap: 30, min_size: 80}
    C2: {size: 400, overlap: 60, min_size: 80}
    C3: {size: 600, overlap: 90, min_size: 80}
    C4: {size: 800, overlap: 120, min_size: 80}

embedding:
  model: BAAI/bge-small-en-v1.5
  query_prefix: "Represent this sentence for searching relevant passages: "
  batch_size: 64
  device: cuda

retrieval:
  dense_top_k: 30
  bm25_top_k: 30
  rrf_k: 60
  rerank_top_k: 20
  final_k: 5
  collapse: max_chunk

models:
  llm: Qwen/Qwen2.5-1.5B-Instruct
  reranker: BAAI/bge-reranker-base
  clip: openai/clip-vit-base-patch32
  nli: MoritzLaurer/DeBERTa-v3-base-mnli

generation:
  max_new_tokens: 256
  temperature: 0.0
  format: json
  repair_attempts: 1

workflow:
  max_retries: 1
  validation_threshold: 0

evaluation:
  goldset: data/gold/goldset_v1.jsonl
  bootstrap_samples: 1000
  ragas_subset_size: 40

personalization:
  cuisine_bonus: 0.05
  ambience_bonus: 0.05
  attribute_bonus: 0.03
  like_bonus: 0.05
  dislike_penalty: 0.10
```

### 18.3 Run manifest

Every script that produces results writes `results/runs/<run_id>/manifest.json`:

```json
{
  "run_id": "20261102T141500Z_e3_hybrid",
  "experiment": "E3",
  "config_hash": "sha256:...",
  "git_commit": "...",
  "python": "3.12.x",
  "packages": {"torch": "...", "transformers": "...", "qdrant-client": "..."},
  "gpu": "NVIDIA GeForce RTX 4060 Laptop",
  "seed": 42,
  "dataset_hash": "sha256:...",
  "goldset_version": "v1",
  "started_at": "...",
  "finished_at": "...",
  "metrics": {},
  "notes": ""
}
```

No result is considered valid without a manifest.

### 18.4 Reproducibility

- `uv lock` and commit `requirements.lock`.
- Set `PYTHONHASHSEED=0`, `seed_everything(42)`, and `torch.use_deterministic_algorithms(True)` where feasible.
- Cache embeddings and extracted PDF text keyed by content hashes.
- Provide `make setup`, `make data`, `make docs`, `make index`, `make eval`, `make app`.
- Provide a small `--smoke` mode that runs on 100 restaurants for CI and demos.
- Do not commit raw data or generated indexes; provide a download script with checksums.
- Document the exact model revisions.

### 18.5 Tests

| Test file | Coverage |
|---|---|
| `test_attributes.py` | Malformed attribute strings, nested dicts, `u''` prefixes, tri-state logic |
| `test_cuisine.py` | Token matching, false positives (`Bar` vs `Barbecue`), multi-label, primary selection |
| `test_chunking.py` | No cross-restaurant chunks, metadata completeness, token limits, minimum size |
| `test_filters.py` | Price bucket mapping, unknown values, hard vs soft constraints |
| `test_fusion.py` | RRF correctness, tie-breaking, filter-before-fusion |
| `test_validators.py` | Hallucinated IDs, wrong ratings, fake quotes, live-availability language |
| `test_metrics.py` | nDCG, MRR, Success@k on hand-computed examples |

Target: ≥80% coverage on data and validation modules. This is not a generic software project; tests protect the claims.

### 18.6 Notebooks

1. **`01_data_audit.ipynb`** — dataset profile, cleaning before/after, attribute parsing examples, cuisine coverage, price distribution.
2. **`02_chunking_experiment.ipynb`** — C1–C4 chunk statistics, contamination check, retrieval comparison.
3. **`03_retrieval_experiment.ipynb`** — E1–E4 metrics, per-query examples, error analysis.
4. **`04_generation_eval.ipynb`** — E7/E9, validator failures, sample answers, RAGAS subset.
5. **`05_efficiency.ipynb`** — E8, memory and latency tables, quality deltas.

Notebook outputs are cleared before commit. Figures are saved as PNG/PDF to `results/figures/`.

### 18.7 Streamlit app

Pages:

1. **Chat / search** — natural-language query, ranked cards, evidence quotes, source badges.
2. **Profile sidebar** — city, budget, cuisines, dietary, ambience, accessibility.
3. **Visual search** — text query over visual summaries, and image upload with CLIP predictions.
4. **Comparison** — select two restaurants, side-by-side evidence.
5. **Debug tab** — intent, constraints, retrieved IDs, scores, validation results, timings. Disabled by default in screenshots but available for the demo.

UX requirements:

- Always label ratings and prices as dataset values.
- Show "insufficient evidence" clearly, not as an error.
- Show which constraints were relaxed on retry.
- Show citations inline, not only in a hidden panel.
- Never show a restaurant that is not in the retrieved candidate set.

The app is a demonstration layer. Evaluation does not run through Streamlit; it calls the Python API directly.

---
