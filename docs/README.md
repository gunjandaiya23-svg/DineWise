# DineWise — Project Documentation

The project plan is split into one file per concern. Start here, then read only what you need.

## Status

Documentation is complete. Implementation has started with the minimal end-to-end path.

| Status | Meaning |
|---|---|
| `draft` | Written, not yet reviewed against code |
| `active` | Being implemented now |
| `done` | Implemented and verified |
| `later` | Planned, not started |

## Chapters

| # | Document | What it covers | Status |
|---:|---|---|---|
| 01 | [Overview, scope and research questions](01-overview.md) | Executive summary, in/out of scope, RQ1–RQ5, hypotheses, contributions | draft |
| 02 | [System architecture](02-architecture.md) | Component diagram, request lifecycle, design principles | draft |
| 03 | [Environment](03-environment.md) | Hardware, Python 3.12, dependencies, runtime decisions | active |
| 04 | [Dataset and study corpus](04-dataset.md) | Verified file schemas, counts, study corpus, dataset limits | active |
| 05 | [Data engineering](05-data-engineering.md) | Attribute parsing, cuisine taxonomy, price rules, cleaning, derived tables | active |
| 06 | [Document layer](06-documents.md) | TXT/PDF/DOCX specifications, generation, loaders | active |
| 07 | [Chunking](07-chunking.md) | Restaurant-aware chunking, C1–C4, metadata, contamination checks | done |
| 08 | [Embeddings and indexes](08-embeddings-indexing.md) | BGE, Qdrant, BM25, CLIP, caching, index metrics | active |
| 09 | [Retrieval](09-retrieval.md) | Dense, lexical, filters, RRF, reranking, evidence layer | active |
| 10 | [Generation](10-generation.md) | Qwen loading, prompts, JSON schema, constrained decoding, validators | active |
| 11 | [LangGraph workflow](11-workflow.md) | State, nodes, edges, retry, fallback | active |
| 12 | [Personalization](12-personalization.md) | Profiles, hard/soft constraints, feedback, evaluation | later |
| 13 | [Visual pipeline](13-visual.md) | Visual-summary retrieval, CLIP zero-shot, fusion, limits | active |
| 14 | [LoRA fine-tuning](14-finetuning.md) | Data construction, splits, hyperparameters, training | done |
| 15 | [Efficiency experiments](15-efficiency.md) | FP16/INT8/INT4, TTFT, TPOT, memory, interpretation | done |
| 16 | [Evaluation framework](16-evaluation.md) | Gold set, annotation, metrics, validators, RAGAS, statistics | active |
| 17 | [Experiment matrix](17-experiments.md) | E0–E11, dependencies, ordering | active |
| 18 | [Engineering plan](18-engineering.md) | Repo layout, config, run manifests, tests, notebooks, app | active |
| 19 | [Phases and schedule](19-roadmap.md) | P0–P11 acceptance criteria, 16-week plan | active |
| 20 | [Risks, validity and limitations](20-risks-limitations.md) | Risk register, threats to validity, ethics | draft |
| 21 | [Deliverables and report](21-report-deliverables.md) | Deliverables, report structure, figures | draft |
| 22 | [Decisions and references](22-decisions-references.md) | Decision log, bibliography | draft |

## Appendices

| Document | Contents |
|---|---|
| [A — Metadata schema](appendices/a-metadata-schema.md) | Chunk and document metadata fields |
| [B — Output schema](appendices/b-output-schema.md) | JSON schema for generated answers |
| [C — Prompt templates](appendices/c-prompts.md) | System, extraction, generation, retry, fallback prompts |
| [D — Test queries](appendices/d-test-queries.md) | The adapted 15-query test set |
| [E — Result tables](appendices/e-result-tables.md) | Templates for data, chunking, retrieval, generation, efficiency |
| [F — Glossary](appendices/f-glossary.md) | Terms used across the project |

## Reading paths

**New to the project:** 01 → 02 → 04 → 19
**Implementing the data layer:** 03 → 04 → 05 → 18
**Implementing retrieval:** 07 → 08 → 09 → 16 → 17
**Implementing generation:** 10 → 11 → 12 → 13
**Running experiments:** 16 → 17 → 19 → 20
**Writing the report:** 01 → 15 → 16 → 20 → 21 → 22

## Documentation rules

1. One concern per file. Do not merge chapters back into one document.
2. Every measured number in the report must come from `results/`, not from these files.
3. When a decision changes, update `22-decisions-references.md` and the relevant chapter in the same commit.
4. Keep the `Status` column honest. A chapter is `done` only when its code and tests exist.
5. Cross-link with relative paths so the docs work on GitHub and locally.
