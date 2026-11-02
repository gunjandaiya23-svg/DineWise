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
| P3 Chunking + index | Done: C1–C4 evaluated; C3 active (1,511 chunks), BGE + Qdrant |
| P4 Retrieval + generation | Done: filtered retrieval, Qwen answer, validators, 15-query suite |
| P5 BM25 + hybrid + reranker | Done: BM25, RRF, cross-encoder, quality prior (E1–E4) |
| P6 Gold set + metrics | Done: silver gold set + Success/P@5/MRR/nDCG (manual review pending) |
| P7 LangGraph | Done: classify → retrieve → generate → validate → retry/fallback (E9) |
| P8 Visual | Done: visual-summary retrieval + CLIP zero-shot (E6) |
| P9 LoRA + efficiency | Done: QLoRA trained, E7 + E8 measured |
| P10 Streamlit app | Done: working local app |

### Final results

| Experiment | Result |
|---|---|
| Data | 500 restaurants, 75,323 reviews, 99.8% cuisine coverage, 100% attribute parse |
| E5 chunking | C3 (600/90) selected: 1,511 chunks, nDCG@5 0.444, 0 marker contamination |
| E1 BM25 | nDCG@5 0.406, strict Success@5 0.250, any-success 1.000 |
| E2 dense (Qdrant) | nDCG@5 0.428, strict Success@5 0.333, any-success 1.000 |
| E3 hybrid RRF | nDCG@5 0.444, strict Success@5 0.500, MRR 0.306 |
| E4 rerank | nDCG@5 0.380 (negative result), quality prior 0.390 |
| E6 visual | 490 docs; strict Success@5 0.400; CLIP top-1 82.2%, top-5 95.0%, mapped cuisine 84.1% |
| E7 LoRA | 400 examples, eval loss 0.122; RAG grounded 80%, hard-pass 100%; no-RAG grounded 0% |
| E8 efficiency | FP16 3.09 GB / 43.2 tok/s; INT4 1.12 GB / 43.0 tok/s; INT8 1.78 GB / 13.4 tok/s; LoRA-INT4 1.16 GB / 20.8 tok/s |
| E9 graph | 12/15 answered, 3 correctly refused, 12/12 validation passed, 4.36 s mean |
| E12 storage | 1,511 points; exact 4.51 ms, HNSW 4.08 ms, recall@10 1.000 |
| Tests | 73 passing, ruff clean |

### Known gaps

- Retrieval labels are **silver** (rubric), not human judgments. `data/gold/review_sheet.csv` is the review sheet.
- One city, 500 restaurants; E10 held-out-city and E11 personalization not run.
- Reranking did not improve nDCG against the quality-oriented silver set.
- Dietary constraints are soft, not safety guarantees.
- Philadelphia Museum of Art still surfaces (venue with a cafe).
- CLIP is evaluated on external Food-101 images, not restaurant photos.

## Quickstart

```bash
make setup        # create .venv (Python 3.12) and install dependencies
make db-up        # start Qdrant (Docker, persistent volume)
make check        # verify GPU, packages, Qdrant, and paths
make download     # fetch the dataset into data/raw with checksum verification
make prep         # clean the data and build data/processed/restaurants.parquet
make docs         # generate TXT, PDF, and DOCX knowledge-base documents
make chunks       # restaurant-aware token chunking
make index        # BGE embeddings -> Qdrant collections
make bm25         # BM25 lexical index over the same chunks
make queries      # 15-query suite (12 answered + 3 refusals)
make experiments  # E1/E2/E3: BM25 vs dense vs hybrid comparison
make backend-eval # E12: Qdrant exact vs HNSW
```

Then:

```bash
make test         # run unit tests
make lint         # ruff
```

## Repository layout

```
config/          configuration and taxonomy
data/            raw downloads, processed tables, documents, Qdrant volume (gitignored)
docs/            the project plan, one file per concern
scripts/         numbered pipeline steps
src/dinewise/    library code
tests/           unit tests for the data, storage, and validation layers
app/             Streamlit interface
results/         run manifests and result tables
docker-compose.yml  Qdrant service
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
