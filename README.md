# DineWise

A retrieval-augmented restaurant recommendation assistant, built as a controlled laboratory project.

The system answers questions such as *"Find Italian restaurants with vegetarian options under $$ in Philadelphia"* by retrieving restaurant records and review evidence, then generating a grounded recommendation with sources.

This is a research prototype over a static dataset. It is **not** a booking system and does not provide live availability or prices.

## Status

| Phase | State |
|---|---|
| Documentation | Complete, split under [`docs/`](docs/README.md) |
| P0 Environment | Done: Python 3.12, torch 2.11 + CUDA 12.8, GPU verified |
| P1 Data layer | Done for the 500-restaurant pilot corpus |
| P2 Documents | Done: TXT, PDF, DOCX generated |
| P3 Chunking + index | Done: C2 chunks, BGE embeddings, FAISS |
| P4 Retrieval + generation | Done: filtered retrieval, Qwen answer, validators, 15-query suite |
| P5 BM25 + hybrid + reranker | BM25 + RRF done (E1–E3 harness); reranker not started |
| P6 Gold set + metrics | Harness ready; relevance judgments not started |
| P7 LangGraph + Streamlit | Not started |

### Pilot results (500 Philadelphia restaurants, filtered retrieval)

| Metric | Value |
|---|---:|
| Restaurants | 500 (17 venues excluded, 99.8% cuisine coverage) |
| Reviews | 75,323 |
| Chunks (C2, 400/60) | 500, 0 cross-restaurant |
| 15-query suite | 12 answered, 3 correctly refused (live availability, Las Vegas, rooftop pool) |
| E1/E2/E3 comparison | BM25 vs dense Jaccard 0.128; dense vs hybrid 0.33; compliance 12/12 all modes |
| Retrieval compliance | All answered queries satisfy hard constraints |
| Generation check | Sampled answer: validation PASSED |

### Known gaps

- Philadelphia Museum of Art still surfaces (venue with a cafe, kept deliberately).
- Dense retrieval only; no BM25, RRF, or reranker.
- Dietary preferences are soft, not evidence-checked.
- No relevance judgments yet, so no Precision/nDCG numbers.
- 500 restaurants, not the full study corpus.
- No LangGraph, no Streamlit.

## Quickstart

```bash
make setup        # create .venv (Python 3.12) and install dependencies
make check        # verify GPU, packages, and paths
make download     # fetch the dataset into data/raw with checksum verification
make prep         # clean the data and build data/processed/restaurants.parquet
make docs         # generate TXT, PDF, and DOCX knowledge-base documents
make chunks       # restaurant-aware token chunking
make index        # BGE embeddings + FAISS index
make bm25         # BM25 lexical index over the same chunks
make queries      # 15-query suite (12 answered + 3 refusals)
make experiments  # E1/E2/E3: BM25 vs dense vs hybrid comparison
```

Then:

```bash
make test         # run unit tests
make lint         # ruff
```

## Repository layout

```
config/          configuration and taxonomy
data/            raw downloads, processed tables, documents, indexes (gitignored)
docs/            the project plan, one file per concern
scripts/         numbered pipeline steps
src/dinewise/    library code
tests/           unit tests for the data and validation layers
app/             Streamlit interface (later)
results/         run manifests and result tables
```

## Documentation

Start at [`docs/README.md`](docs/README.md). The plan is intentionally split by concern:

- [Overview and research questions](docs/01-overview.md)
- [Dataset and study corpus](docs/04-dataset.md)
- [Data engineering](docs/05-data-engineering.md)
- [Evaluation framework](docs/16-evaluation.md)
- [Roadmap](docs/19-roadmap.md)

## Data

The project uses the [Yelp Multimodal Recommendation dataset](https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation), derived from the [Yelp Open Dataset](https://www.yelp.com/dataset). It is intended for educational, non-commercial research only. Raw data is not redistributed with this repository.

## License

Code: to be decided. Data: Yelp Open Dataset terms, non-commercial research use only.
