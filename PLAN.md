# DineWise — Project Plan

**Project:** DineWise: A Retrieval-Augmented Language Model Framework for Personalized and Explainable Restaurant Recommendations
**Document version:** 1.0
**Date:** 2 November 2026
**Status:** Planning complete; implementation not started
**Target environment:** Local workstation, NVIDIA RTX 4060 Laptop (8 GB VRAM), Linux
**Target duration:** One semester (~16 weeks)

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

## 2. Verified starting point

All figures in this section were measured directly against the downloaded dataset on 2 November 2026. They supersede any numbers in earlier drafts or in the dataset's own README.

### 2.1 Environment

| Item | Value | Consequence |
|---|---|---|
| Working directory | `/home/naimish08/Desktop/DineWise` (empty at planning time) | Greenfield repository |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB VRAM, driver CUDA 13.2 | Sufficient for 1.5B FP16 inference, QLoRA, INT8/INT4 |
| System RAM | 22 GiB total, ~15 GiB available | Comfortable for pandas over the full review file; avoid loading multiple large models simultaneously |
| CPU | 20 logical cores | PDF generation and tokenization are not bottlenecks |
| Disk | 179 GiB free on `/` | Dataset (~1.3 GB), models (~6 GB), indexes (<1 GB) all fit comfortably |
| Default Python | 3.14.4 | **Unusable** for the ML stack; torch, faiss, and bitsandbytes do not publish 3.14 wheels |
| Available Python | 3.12 via `python3.12`, plus `uv` | Build the project environment with Python 3.12 |

**Decision:** create a project-local virtual environment with `uv venv --python 3.12`, install CUDA-enabled PyTorch from the PyTorch wheel index, then freeze exact versions into `requirements.lock`. Never install into the system interpreter. Set `HF_HOME` and the model cache to a persistent project or home directory rather than `/tmp`.

**Decision:** develop locally on the RTX 4060. A Colab notebook may be generated later as a submission artifact, but it is not the primary runtime. Local execution avoids session limits and keeps the dataset and indexes persistent.

### 2.2 Dataset files

Source: `wzehui/Yelp-Multimodal-Recommendation` on Hugging Face. Total size 743.1 MB. License: `other`; derived from the Yelp Open Dataset and restricted to educational, non-commercial research. The dataset README states that it accompanies a RecSys 2025 paper.

| File | Size | Rows | Purpose in DineWise |
|---|---:|---:|---|
| `business.csv` | 22.9 MB | 16,702 | Restaurant catalogue, attributes, categories, location, rating |
| `review.csv` | 648.0 MB | 837,559 | Raw review evidence and review-derived statistics |
| `review_summary.csv` | 10.0 MB | 16,702 | Per-restaurant LLM-generated review summary |
| `photo.csv` | 8.3 MB | 118,951 | Photo metadata: business mapping, label, caption |
| `photo_summay.csv` | 6.5 MB | 13,289 | Per-business LLM-generated visual description |
| `checkin.csv` | 47.4 MB | not inspected | Optional; check-in counts, not used in the core system |

Notes and traps:

- The photo-summary filename is **`photo_summay.csv`** (missing "r") in the actual repository. Download scripts must use the real name and then copy/alias it to `photo_summary.csv` locally.
- Several CSV files begin with a UTF-8 BOM. Read with `encoding="utf-8-sig"`.
- The dataset README claims `review.csv` contains a date. It does not: the header is `review_id,user_id,business_id,text,stars,reactions`. **No temporal analysis is possible.**
- There are no image files and no image URLs. Only photo metadata, captions, labels, and LLM-generated visual summaries are present.
- The dataset README refers to `photo_summary.csv` even though the file is misspelled; document this in the download script and in the report.

### 2.3 Dataset profile (measured)

Overall dataset:

| Metric | Value |
|---|---:|
| Businesses | 16,702 |
| Unique business IDs | 16,702 (no duplicates) |
| Businesses with a review summary | 16,702 (100%) |
| Businesses with a photo summary | 13,289 (79.6%) |
| Reviews | 837,559 |
| Unique review IDs | 837,559 (no duplicate IDs) |
| Review text, total | ≈582.6 million characters |
| Review length, median / p95 / max | 537 / 1,787 / 5,000 characters |
| Mean business rating | 3.86 (SD 0.60) |
| Mean review count per business | 223.7 (median 146) |
| Open businesses | 83% |
| States represented | 14 |
| Cities represented | 507 |
| Price range present | 91.8% |
| Photo records | 118,951 across 13,289 businesses |
| Photo labels | food 61,778; inside 36,197; outside 11,813; drink 9,163 |
| Photo captions present | 53.3% (46.7% missing) |
| Review star distribution | 1★ 37,847; 2★ 55,767; 3★ 123,986; 4★ 276,266; 5★ 343,693 |

Largest cities: Philadelphia 2,126; Indianapolis 1,363; Tampa 1,271; New Orleans 1,175; Tucson 1,011; Reno 976; Nashville 911; Saint Louis 522; Santa Barbara 331; Edmonton 268 (Canada); Boise 249; Saint Petersburg 243.

**Las Vegas is not in the dataset.** Nevada is represented only by Reno, Sparks, Verdi, Virginia City, and Spanish Springs. This is useful: "recommend a restaurant in Las Vegas" is a genuine out-of-scope test rather than an artificial one.

### 2.4 Study corpus (measured)

To keep the document corpus, PDF, index, annotation effort, and app tractable, the default corpus is five cities: **Philadelphia, Indianapolis, Tampa, New Orleans, Tucson**. These span five states (PA, IN, FL, LA, AZ) and provide variety in cuisine, price, and ambience.

| Metric | Value |
|---|---:|
| All businesses in the five cities | 6,946 |
| Businesses after food-service filtering (Section 5.3) | **6,033** |
| Reviews for the filtered corpus | **385,065** |
| Mean reviews per restaurant | 63.8 |
| Mean / median review length | 694 / 542 characters |
| Restaurants with a photo summary | 5,381 (89.2%) |
| Restaurants with ≥50 reviews | 5,578 (92.5%) |
| Open restaurants | 79.6% |
| Price range present | 96.4% |
| City distribution | Philadelphia 1,894; Indianapolis 1,147; Tampa 1,093; New Orleans 1,025; Tucson 874 |

Held-out cities for the generalization experiment (Section 17, E10): **Nashville and Reno**. They are not in the main corpus and are used only to test whether conclusions transfer.

### 2.5 Attribute availability in the study corpus

| Attribute | Present | Parse success |
|---|---:|---:|
| Any attributes | 99.5% of rows parse | 99.5% |
| `RestaurantsPriceRange2` | 90.2% | 1–4 bucket |
| `OutdoorSeating` | 79.5% | True/False/unknown |
| `NoiseLevel` | 75.7% | average/quiet/loud/very_loud/unknown |
| `Ambience` (nested dict) | 76.3% | 149 explicitly romantic, 4,726 false, 2,071 unknown |
| `GoodForMeal` (nested dict) | 70.5% | 415 explicitly dessert=true |
| `WiFi` | 83.7% | free/paid/no/unknown |
| `RestaurantsReservations` | 75.1% | True/False/unknown |
| `GoodForKids` | 78.3% | True/False/unknown |
| `BYOB` | 12.5% | True/False/unknown |

Cuisine family counts in the study corpus, derived from category tokens (approximate, multi-label, mean 1.52 families per restaurant):

| Family | Count | Family | Count |
|---|---:|---|---:|
| American | 2,395 | Chinese | 282 |
| Bar / Nightlife | 1,898 | Steakhouse | 219 |
| Breakfast / Brunch | 1,468 | Mediterranean | 212 |
| Bakery / Dessert | 734 | BBQ | 182 |
| Italian | 661 | Vietnamese | 136 |
| Mexican | 579 | Thai | 135 |
| Seafood | 545 | French | 114 |
| Japanese / Sushi | 348 | Indian | 87 |
| Vegetarian / Vegan | 317 | Greek | 87 |
| | | Spanish | 76 |
| | | Korean | 74 |

82.5% of restaurants match at least one cuisine family. The remainder are retail or service businesses (shopping, salons, hotels) or generic "Restaurants"/"Food" entries that must be filtered or labeled "Unspecified".

### 2.6 What the dataset cannot support

These constraints are binding on the project and must appear in the report's Limitations section:

1. **No images.** CLIP cannot match a user photo against restaurant photos. The visual experiment must use textual visual summaries plus CLIP zero-shot classification of food images.
2. **No dates.** No recency, trend, or seasonality analysis.
3. **No dollar prices.** Only a 1–4 bucket. Budget queries require a documented mapping rule.
4. **No menus, dish prices, or structured dietary fields.** Dietary and dish claims must come from review text or summaries, with uncertainty.
5. **No live availability, hours reliability, or booking data.** `hours` exists but is static and incomplete; the system must not present it as live.
6. **Summaries are LLM-generated.** `review_summary.csv` was generated with DeepSeek-R1 and `photo_summay.csv` with GPT-4o. They may contain unsupported claims. Raw reviews are the higher-authority source; summaries are an index and navigation aid.
7. **Ratings are static dataset averages**, not current live ratings.
8. **Coverage is US-centric and non-uniform.** Five cities dominate the study corpus.

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

## 4. System architecture

### 4.1 Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ Streamlit UI                                                         │
│  profile sidebar · chat · ranked cards · evidence panel · debug tab  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ request
┌───────────────────────────────▼──────────────────────────────────────┐
│ LangGraph workflow                                                   │
│                                                                      │
│ classify_intent → extract_preferences → rewrite_query                │
│        → retrieve → filter_candidates → rerank → generate_answer     │
│        → review_answer ──supported──→ END                            │
│              │                                                       │
│              └──insufficient──→ retry (max 1) → review_answer        │
│                                      │                               │
│                                      └──still insufficient──→ fallback│
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
┌───────▼────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│ Retrieval      │   │ Evidence layer     │   │ Generation         │
│ FAISS (BGE)    │   │ BM25 over candidate│   │ Qwen2.5-1.5B       │
│ BM25 (rank)    │   │ restaurants' raw   │   │ schema-constrained │
│ attribute      │   │ reviews → quotes   │   │ decoding + repair  │
│ filters        │   │ with review_id     │   │                    │
│ RRF + reranker │   │                    │   │                    │
└───────┬────────┘   └──────────┬─────────┘   └──────────┬─────────┘
        │                       │                        │
┌───────▼───────────────────────▼────────────────────────▼─────────┐
│ Data layer                                                        │
│ restaurants.parquet · reviews.parquet · doc chunks · FAISS index   │
│ · BM25 corpus · CLIP text index · photo metadata                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 Request lifecycle

1. The UI sends a query plus the current profile.
2. `classify_intent` labels the request as recommendation, comparison, visual, or out-of-scope.
3. `extract_preferences` produces a typed constraint object: city, cuisine families, max price bucket, min rating, required attributes, dietary terms, ambience terms, and soft preferences.
4. `rewrite_query` produces a retrieval-friendly query and a set of attribute filters.
5. `retrieve` runs dense search, BM25, and (where applicable) the visual index, then fuses results.
6. `filter_candidates` applies hard constraints against the structured catalogue. Hard constraints are never relaxed.
7. `rerank` orders candidates with a cross-encoder and applies personalization adjustments.
8. The evidence layer retrieves quotable review sentences for each of the top candidates.
9. `generate_answer` builds the prompt from facts, policy, and evidence, and calls the LLM with a JSON schema.
10. `review_answer` validates the output programmatically and with NLI entailment.
11. If validation fails, the workflow retries once with an expanded query and relaxed soft preferences only. If it fails again, `fallback` returns an explicit insufficiency message.

### 4.3 Design principles

1. **Facts come from structured data, not from the model.** Names, ratings, price buckets, and attributes are looked up by ID and injected into the prompt. The model is instructed not to generate them from memory.
2. **Rules are separate from facts.** The `dining_preferences.docx` policy document is retrieved and injected in a distinct prompt block. A policy such as "prefer quiet restaurants" can never be cited as evidence that a restaurant is quiet.
3. **Every claim must be traceable.** Each recommendation must cite a business ID and at least one evidence chunk or review ID, or declare that it is based only on structured attributes.
4. **Abstention is a feature.** Out-of-scope or unsupported queries must produce a clear refusal with an explanation, not a plausible invention.
5. **One variable at a time.** Every experiment changes one major component while holding the corpus, queries, and remaining settings fixed.
6. **Measure, do not assume.** Latency and memory claims are replaced with measured values from `results/`.

---

## 5. Data engineering

### 5.1 Source schemas (verified)

`business.csv`:

```
business_id, name, address, city, state, postal_code, latitude, longitude,
stars, review_count, is_open, attributes, categories, hours, photo_id
```

- `attributes` is a Python-dict string with nested dict strings and Python-2 `u'...'` prefixes.
- `categories` is a comma-separated string.
- `hours` is a dict string, frequently missing.
- `photo_id` is a single photo, not the full photo set; use `photo.csv` for the mapping.

`review.csv`:

```
review_id, user_id, business_id, text, stars, reactions
```

- No date, no useful/funny/cool counts. `reactions` is an integer.
- No duplicate review IDs, no empty text.

`review_summary.csv`:

```
business_id, sentiment_score, sentiment_confidence, summary, keywords, themes
```

- One row per business, 100% coverage.
- `themes` is a string containing aspect labels and phrases, not a structured list.

`photo.csv`:

```
photo_id, business_id, caption, label
```

- `label` ∈ {food, inside, outside, drink}.
- 46.7% of captions are missing.

`photo_summay.csv`:

```
business_id, summary, keywords, indoor_color_tone, venue_style,
food_style, drink_style, target_audience, special_features
```

- One row per business for 79.6% of businesses.
- Text is descriptive and subjective; treat it as weak evidence, not ground truth.

### 5.2 Study corpus definition

The corpus is defined in `config/config.yaml` and is reproducible:

```yaml
data:
  cities:
    - Philadelphia
    - Indianapolis
    - Tampa
    - New Orleans
    - Tucson
  heldout_cities:
    - Nashville
    - Reno
  min_reviews_for_corpus: 10
  require_food_service: true
```

The number of restaurants and reviews after filtering is logged to `results/tables/data_audit.csv`. A command-line flag `--all-cities` builds the full 16,702-business corpus for sensitivity analysis, but the main experiments use the five-city corpus.

### 5.3 Restaurant filter

The dataset contains non-restaurant businesses (salons, hotels, automotive, retail). The filter keeps a business if any category token matches the food-service vocabulary:

```
Restaurants, Food, Cafes, Coffee & Tea, Bakeries, Desserts, Bars,
Breweries, Wineries, Distilleries, Gastropubs, Food Trucks, Caterers,
Specialty Food, Ice Cream & Frozen Yogurt, Juice Bars & Smoothies,
Tea Rooms, Sandwiches, Pizza, Sushi Bars, Barbeque
```

Additionally, any category containing the substrings `Restaurant`, `Food`, `Cafe`, `Bakery`, `Pizza`, or `Sushi` qualifies. The filter is applied to the tokenized category set, not to raw substrings, to avoid false positives such as `Bar` matching `Barbecue` or `Barber`.

Effect: 6,946 five-city businesses → 6,033 food-service businesses. The filter rule and the excluded counts are reported in the data audit.

### 5.4 Attribute parsing specification

Parsing must be deterministic, lossless, and tested. The algorithm:

1. Read the CSV with `encoding="utf-8-sig"`.
2. Parse the outer string with `ast.literal_eval`. On failure, fall back to a tolerant regex parser and record a parse failure.
3. Recursively parse any value that is a string beginning with `{` into a dict.
4. Normalize values:
   - `"True"`/`"False"` → `True`/`False`;
   - `"None"`, `""`, and missing → `None` (unknown);
   - numeric strings for price range → `int`.
5. Normalize keys to `snake_case`.
6. Preserve the raw string in `attributes_raw` for audit.
7. Represent every boolean attribute as a **tri-state**: `True`, `False`, or `None`. Missing is never treated as `False`.
8. Store nested dictionaries (`Ambience`, `GoodForMeal`, `BusinessParking`, `Music`, `BestNights`) as flattened columns, for example `ambience_romantic`, `good_for_meal_dessert`, `parking_street`.

Known dirty values that must be covered by tests:

- `""u'no'""` → `no`
- `"{'touristy': False, 'romantic': True, ...}"` nested inside the outer dict
- `NoiseLevel` values `u'average'`, `'average'`, `u'quiet'`, `u'very_loud'`
- `RestaurantsAttire` values `u'casual'`, `'casual'`, `u'dressy'`
- String `'None'` alongside actual missing values
- Price range `'1'`–`'4'`

Output columns include at minimum:

```
price_range, outdoor_seating, noise_level, wifi, alcohol,
reservations, good_for_kids, attire, takeout, delivery, caters,
has_tv, accepts_credit_cards, wheelchair_accessible, dogs_allowed,
byob, corkage, good_for_groups, good_for_dancing,
ambience_romantic, ambience_intimate, ambience_classy, ambience_casual,
ambience_trendy, ambience_upscale, ambience_divey, ambience_hipster,
ambience_touristy,
good_for_meal_breakfast, good_for_meal_brunch, good_for_meal_lunch,
good_for_meal_dinner, good_for_meal_dessert, good_for_meal_latenight,
parking_street, parking_lot, parking_valet, parking_garage,
parking_validated
```

Acceptance: parse success ≥98% of non-null attribute strings; all malformed test fixtures handled; no silent conversion of unknown to false.

### 5.5 Cuisine taxonomy

`categories` is multi-label. DineWise derives a **cuisine family** mapping with explicit token lists (Section 2.5 counts). Rules:

- Match on exact category tokens after splitting on commas and stripping whitespace.
- A restaurant may belong to several families; `cuisine_primary` is chosen by specificity order (e.g., `Italian` before `American`, `Sushi Bars` before `Japanese` before `American`).
- Keep the full `cuisine_tags` list for filtering and explanation.
- Unmatched restaurants receive `cuisine_primary = "Unspecified"` and are excluded only from cuisine-constrained queries; they remain retrievable for open-ended queries.
- The taxonomy and its coverage are versioned in `config/cuisine_taxonomy.yaml`.
- The mapping is a derived annotation, not an official Yelp label. The report must say so.

### 5.6 Price model

Only a 1–4 bucket exists. The documented mapping:

| Bucket | Label in UI | Approximate interpretation |
|---|---|---|
| 1 | `$` | inexpensive |
| 2 | `$$` | moderate |
| 3 | `$$$` | expensive |
| 4 | `$$$$` | very expensive |
| unknown | `Price not listed` | unknown |

Query interpretation rules:

- "cheap", "budget", "under $30" → bucket ≤ 2.
- "moderate", "mid-range" → bucket 2–3.
- "fine dining", "upscale", "expensive" → bucket ≥ 3.
- If a numeric budget is supplied, it is converted with the documented rule and the UI displays a caveat: "Budget is approximated from the dataset's price bucket; exact prices are not available."
- A budget-constrained query excludes unknown-price restaurants from the hard-filtered set and states how many were excluded.

This rule is a limitation, not a hidden assumption. It appears in the report and in the UI.

### 5.7 Cleaning rules

| Rule | Action | Rationale |
|---|---|---|
| Duplicate business IDs | Keep first, assert count unchanged | Integrity |
| Duplicate review IDs | Drop duplicates | Measured zero, but enforce |
| Duplicate review text within a business | Keep the longest copy, record count | 2,867 exact duplicate texts overall |
| Empty or whitespace-only review text | Drop | Measured zero, but enforce |
| Reviews shorter than 20 characters | Exclude from evidence pool, keep in aggregates | Avoid meaningless quotes |
| Missing rating | Exclude from rating filters, keep with unknown | Do not impute |
| Missing price | Label unknown; exclude only from budget-constrained queries | Do not impute |
| Missing city/state | Exclude from location-filtered queries | Cannot be used reliably |
| Closed businesses (`is_open=0`) | Keep in catalogue, down-rank by default, expose toggle | Dataset is historical |
| Non-food-service businesses | Exclude from the recommendation corpus | Scope |
| Review summary text empty | Fall back to raw review excerpts | Robustness |
| Photo summary missing | Omit visual section; do not invent | Grounding |

No imputation of ratings, prices, or attributes is allowed except for explicitly labeled derived statistics. Any imputed value in an experiment must be reported.

### 5.8 Derived tables

**`data/processed/restaurants.parquet`** — one row per business:

```
business_id, name, address, city, state, postal_code, latitude, longitude,
stars, review_count, is_open, price_range, price_label,
cuisine_primary, cuisine_tags, food_service,
attributes_raw, attribute columns..., hours_raw,
review_summary, review_keywords, review_themes,
sentiment_score, sentiment_confidence,
photo_summary, venue_style, indoor_color_tone, food_style, drink_style,
target_audience, special_features,
n_photos, n_photos_by_label, has_photo_summary, has_review_summary,
mean_review_stars, star_5_ratio, star_1_ratio
```

**`data/processed/reviews.parquet`** — one row per review:

```
review_id, business_id, text, stars, reactions, char_len, word_count,
aspect_hits (JSON: food, service, ambience, price, dessert, vegetarian, ...)
```

**`data/processed/evidence.parquet`** — quotable review sentences:

```
evidence_id, review_id, business_id, sentence, stars, aspect_tags, char_len
```

Sentence-level evidence improves citation precision and keeps prompts short. Sentence splitting is rule-based; sentences between 30 and 400 characters are kept. A maximum of 120 evidence sentences per restaurant is stored to bound index size, selected to cover aspects first and then by review length.

**`data/processed/doc_chunks.parquet`** — see Section 7.4.

**`data/processed/image_assets.parquet`** — `photo_id, business_id, label, caption, caption_len`.

### 5.9 Data audit metrics

Produced by `scripts/01_prepare_data.py` into `results/tables/data_audit.csv`:

- total records before and after each cleaning step,
- duplicate percentage by table,
- missing-value percentage for every critical column,
- attribute parse success rate,
- cuisine mapping coverage,
- price availability,
- review-to-business join coverage,
- photo-to-business join coverage,
- review summary join coverage,
- number of businesses excluded by the food-service filter,
- number of restaurants with zero usable evidence,
- review length distribution,
- star distribution.

The audit table is included in the report as a dataset-quality table.

### 5.10 Known data issues to document

1. `review.csv` has no dates despite the README.
2. `photo_summay.csv` is misspelled.
3. Nested dict strings require double parsing.
4. `Ambience` is sparse: only 149 of 6,033 study restaurants are explicitly `romantic=True`; 2,071 are unknown. Absence of `romantic=True` is not evidence that a restaurant is not romantic.
5. `GoodForMeal.dessert=True` covers only 415 restaurants.
6. `BYOB` is present for 12.5% of restaurants; absence is not evidence of no BYOB.
7. Photo captions are missing for 46.7% of photos.
8. Review and photo summaries are LLM-generated and may contain unsupported claims.
9. The dataset is a subset of Yelp; it is not a representative sample of US restaurants.
10. Business hours are static and may be stale; never presented as live.

---

## 6. Document layer

### 6.1 Why multiple formats

The laboratory exercise requires demonstrating multiple document loaders and a realistic heterogeneous knowledge base. DineWise therefore builds three human-readable documents from the structured data, while keeping the structured tables for exact filtering.

The separation is deliberate:

- **Unstructured documents** support semantic retrieval and explanation.
- **Structured tables** support exact constraint checking, rating lookup, and evaluation.
- **Neither replaces the other.** Ratings and prices are never parsed back out of the documents for validation.

### 6.2 `restaurant_profiles.txt`

One block per restaurant, generated deterministically from `restaurants.parquet`. Template:

```
[RESTAURANT B001]
Name: Restaurant ABC
City: Philadelphia, PA
Cuisine: Italian; Pizza; Vegetarian
Price: $$ (dataset price bucket 2 of 4)
Rating: 4.5 out of 5 (dataset average, 312 reviews)
Status: Open in dataset
Address: 123 Example Street, Philadelphia, PA 19103
Outdoor seating: yes
Noise level: average
Romantic ambience: unknown
Good for: dinner, dessert
Dietary evidence: review mentions vegetarian options (see review insights)
Photo labels: 4 inside, 3 food, 1 outside
Source: business.csv
```

Rules:

- Always start with the `[RESTAURANT <id>]` marker.
- Never include a field whose value is unknown; use "unknown" explicitly only for fields central to the query.
- Never fabricate a price in dollars.
- Keep each block under ~180 tokens.
- Write with `utf-8`, one blank line between blocks.

Expected size: 6,033 blocks, roughly 1.0–1.3 million characters, ~250k tokens.

### 6.3 `customer_review_insights.pdf`

Generated with ReportLab. One entry per restaurant, in the same order as the profile file. Template:

```
Restaurant ABC (B001)
Cuisine: Italian · City: Philadelphia
Overall sentiment: 0.78 (confidence 0.81)
Summary: <review_summary.summary>
Recurring themes: <themes>
Keywords: <keywords>

What reviewers praise:
- <aspect: food quality> <short supporting phrase>
- <aspect: service> <short supporting phrase>

What reviewers criticize:
- <aspect: wait time> <short supporting phrase>

Popular dishes mentioned: <list, only if present in reviews>
Ambience observations: <from review summary and review sentences>
Source: review_summary.csv; review.csv (aggregated)
```

Layout decisions:

- Two columns, 9-point body text, compact spacing, to keep the page count near 2,000–2,500 pages.
- Each entry starts with a bold restaurant header so extraction remains human-checkable.
- The PDF is generated once and cached; regeneration is controlled by a hash of the input table.
- Because PyPDF extraction of 2,000+ pages is slow, the loader result is cached as JSONL with page numbers. The cache is invalidated by the PDF file hash.
- If the PDF exceeds 30 MB or extraction exceeds 5 minutes, fall back to one PDF per city (still exercising `PyPDFLoader`) and document the change.

The "praise" and "criticize" bullets are extracted from the review summary and from aspect-tagged evidence sentences. They are not generated by a new LLM call at document-build time. This preserves traceability.

### 6.4 `dining_preferences.docx`

A policy document written by hand, stored as `data/docs/dining_preferences.docx` and generated from a Markdown source with `python-docx` so it is version-controlled. Content:

```
DineWise Recommendation Policy

1. Budget
   Filter by the dataset's price bucket. Never invent a dollar price.
   Treat unknown price as unknown, not as cheap.

2. Cuisine
   Prefer restaurants whose cuisine tags match the request.
   If no exact match exists, say so before offering alternatives.

3. Dietary preferences
   Use explicit evidence only. A review mentioning "vegetarian options"
   is evidence; the absence of such a mention is not evidence of absence.
   Medical allergies must never be inferred.

4. Ambience
   Use structured attributes first (romantic, intimate, classy, noise level,
   outdoor seating). Use review and photo summaries only as supporting evidence.
   A policy preference for quiet restaurants does not establish that a
   particular restaurant is quiet.

5. Ranking
   Rank by constraint satisfaction, then relevance, then review evidence,
   then rating. Do not rank by rating alone.

6. Explanation
   Explain why each restaurant matches. Cite the source of every factual claim.

7. Uncertainty
   If the dataset lacks information, say so. Do not guess.

8. Out of scope
   Do not answer questions about live availability, reservations, current
   prices, or real-time wait times.
```

This document is indexed with `document_type=policy`. Retrieval for it is optional: the policy block is small enough to inject in full into every prompt. The safer design is to inject it in full and mark it as policy, which avoids retrieving a rule fragment out of context. The document loader is still exercised.

### 6.5 Loader pipeline

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
)

documents = []
documents += TextLoader("data/docs/restaurant_profiles.txt", encoding="utf-8").load()
documents += PyPDFLoader("data/docs/customer_review_insights.pdf").load()
documents += Docx2txtLoader("data/docs/dining_preferences.docx").load()
```

CSV files are loaded with pandas, not with `CSVLoader`, because exact typing and ID preservation matter. This is a deliberate deviation from a pure-LangChain pipeline and is documented.

After loading:

- attach `source_file`, `document_type`, and `page` to each document;
- for the profile TXT, split by `[RESTAURANT ...]` markers into one document per restaurant before chunking;
- for the PDF, associate each page with the restaurants whose headers appear on it; if a page spans restaurants, split at headers and assign metadata accordingly;
- reject documents with empty content and log the count.

### 6.6 Document metrics

Recorded in `results/tables/doc_stats.csv`:

- files loaded,
- documents produced,
- pages extracted,
- total characters,
- empty document percentage,
- mean characters per document,
- extraction time per loader,
- metadata completeness percentage,
- duplicate document percentage.

Acceptance: zero empty documents; metadata completeness ≥99% for `source_file`, `document_type`, and `page` (page is `null` for TXT/DOCX); extraction time recorded.

---

## 7. Chunking

### 7.1 Requirements

1. A chunk must never contain content from two restaurants.
2. Each chunk must carry `business_id`, `restaurant_name`, `city`, `cuisine_primary`, `price_range`, `source_file`, `document_type`, `section`, and `page`.
3. Chunk boundaries should follow section boundaries where possible.
4. Chunk length is measured in **tokens**, not characters.
5. The configuration must be a single parameter set so that C1–C4 differ only in size and overlap.

### 7.2 Algorithm

1. Build one `Document` per restaurant section:
   - profile section,
   - review-insight section,
   - visual section.
2. Split each section with a token-based recursive splitter:
   - separators: `["\n\n", "\n", ". ", " "]`,
   - length function: tokenizer of the embedding model (`bge-small-en-v1.5`),
   - `chunk_size` and `chunk_overlap` from configuration,
   - `keep_separator=True`.
3. Enforce a minimum chunk size of 80 tokens by merging a short tail into the previous chunk of the same restaurant.
4. Attach metadata after splitting, from the parent document. No metadata inference from text.
5. Assert that every chunk's `business_id` is unique within the chunk and equals the parent's.
6. Write `doc_chunks.parquet` plus a per-configuration manifest.

Implementation note: LangChain's `RecursiveCharacterTextSplitter` counts characters by default. Use `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` or a custom length function based on the BGE tokenizer. The plan standardizes on the BGE tokenizer so that chunk sizes match the embedding model's truncation limit.

### 7.3 Configurations

| Config | Chunk size (tokens) | Overlap | Overlap ratio | Purpose |
|---|---:|---:|---:|---|
| C1 | 200 | 30 | 15% | High precision, fragmented |
| C2 | 400 | 60 | 15% | Default hypothesis |
| C3 | 600 | 90 | 15% | More context |
| C4 | 800 | 120 | 15% | Long context, more noise |

All other settings are identical: same documents, same embedding model, same retrieval parameters, same queries.

Expected chunk counts for the study corpus, to be replaced by measurement:

- C1: ~28k–35k chunks,
- C2: ~18k–24k chunks,
- C3: ~12k–16k chunks,
- C4: ~9k–12k chunks.

### 7.4 Chunk metadata schema

```json
{
  "chunk_id": "B001::review::003",
  "business_id": "B001",
  "restaurant_name": "Restaurant ABC",
  "city": "Philadelphia",
  "state": "PA",
  "cuisine_primary": "Italian",
  "price_range": 2,
  "document_type": "review_insight",
  "section": "food_quality",
  "source_file": "customer_review_insights.pdf",
  "page": 142,
  "token_count": 387,
  "char_count": 1842,
  "chunk_config": "C2",
  "text": "..."
}
```

`document_type` ∈ {`profile`, `review_insight`, `visual`, `policy`}. Policy chunks have `business_id = null`.

### 7.5 Contamination guarantee

Because each restaurant is split independently, a chunk cannot cross a restaurant boundary. This is a structural property, not a hope. The verification is still performed:

- For every chunk, count distinct `[RESTAURANT ...]` markers in the text. Target: 1 for profile chunks, 0 or 1 for review/visual chunks that do not repeat the marker.
- Count chunks whose text contains a restaurant name other than their own, excluding policy chunks. Target: 0.
- Report the contamination rate in the chunking table. Any nonzero value is a bug.

### 7.6 Chunking metrics

- number of chunks per configuration,
- mean, median, min, and max token length,
- percentage below the 80-token floor after merging,
- percentage at or above the embedding model's truncation limit,
- cross-restaurant contamination rate,
- metadata completeness,
- index build time,
- retrieval metrics for each configuration (Section 16.3),
- end-to-end answer metrics for each configuration (Section 16.4).

The winning configuration is selected by nDCG@5 with faithfulness as a tie-breaker, not by a single metric. If differences are within the confidence interval, the plan reports that the configurations are indistinguishable and chooses the one with lower index cost.

---

## 8. Embeddings and indexes

### 8.1 Text embeddings

Model: `BAAI/bge-small-en-v1.5` (384 dimensions, ~33M parameters).

Rules:

- L2-normalize embeddings and use inner product (`IndexFlatIP`) so that the score is cosine similarity.
- Add the BGE retrieval instruction to **queries only**: `Represent this sentence for searching relevant passages: `. Documents are embedded without the prefix.
- Maximum sequence length: 512 tokens. Chunks longer than this are truncated by the model; the chunking experiment must not exceed 512 tokens without documenting truncation.
- Batch size 64 during indexing, adjusted downward if VRAM is tight. Embedding runs on GPU in FP16.
- Cache embeddings to `data/indexes/embeddings/<model>/<sha256(chunk_texts)>.npy` so that repeated experiments do not recompute them.
- Record the exact model revision/hash in the run manifest.

### 8.2 FAISS index

- Use `faiss.IndexFlatIP` for exact search. The corpus is small enough (tens of thousands of vectors) that approximate indexes are unnecessary and would confound the retrieval experiment.
- Persist as `data/indexes/faiss/<config>/index.faiss` plus `chunk_ids.json`.
- Expected size: ~25k vectors × 384 dimensions × 4 bytes ≈ 38 MB.
- Build time and query latency are measured, not assumed.
- Do not use LangChain's FAISS wrapper for the evaluation path; use `faiss` directly and keep a thin adapter. LangChain's wrapper is acceptable for the tutorial notebook but obscures score semantics.

### 8.3 BM25 index

- Tokenization: lowercase, Unicode-aware word tokens, stopword removal using a fixed English list, no stemming by default (record the choice).
- Use `rank_bm25.BM25Okapi` with `k1=1.5`, `b=0.75`.
- Build one BM25 index over the same chunk texts as the dense index, so that hybrid fusion compares the same units.
- Persist the tokenized corpus and document IDs with `pickle` or `joblib`.
- BM25 is also built at the **restaurant level** for the evidence layer, but that index is per-candidate and constructed on the fly.

### 8.4 Visual index

Two separate artifacts:

1. **Text-to-visual retrieval:** BGE embeddings of the `photo_summay.csv` fields (`summary`, `venue_style`, `indoor_color_tone`, `special_features`, `target_audience`). One document per business, metadata `document_type=visual`. This is text retrieval over visual descriptions, not image retrieval, and is labeled as such.
2. **CLIP text embeddings:** optional, for comparing short text queries with CLIP text embeddings of photo captions. Used only in the E6 ablation. CLIP text embeddings are not mixed into the BGE index; scores are not comparable across spaces.

### 8.5 Index metrics

- embedding generation time,
- throughput (chunks/second),
- embedding dimension,
- index build time,
- index size on disk,
- retrieval latency p50/p95 over the gold queries,
- peak GPU memory during indexing.

---

## 9. Retrieval

### 9.1 Dense retrieval

- Embed the rewritten query with the BGE prefix.
- Retrieve top 30 chunks by cosine similarity.
- Collapse chunk hits to restaurants: a restaurant's score is the maximum chunk score, with a small bonus if multiple chunks from the same restaurant appear. The collapse rule is fixed before experiments.
- Apply metadata filters either pre-search (FAISS `IDSelector`) or post-search. Pre-search is preferred for hard constraints to avoid wasting the candidate budget; the implementation must document which path is used.

### 9.2 BM25 retrieval

- Tokenize the rewritten query with the same tokenizer as the index.
- Retrieve top 30 chunks.
- Collapse to restaurants with the same rule as dense retrieval.
- BM25 is expected to help with dish names, proper nouns, and exact phrases; this is a hypothesis to test, not an assumption.

### 9.3 Structured filters

Hard constraints are applied against `restaurants.parquet` before or immediately after retrieval:

- city (exact match, case-insensitive),
- cuisine family membership,
- price bucket (`<=`, `>=`, or range),
- minimum rating,
- required attributes (e.g., `outdoor_seating is True`),
- open status (default: prefer open, do not hard-exclude unless the user asks),
- dietary evidence (only if explicit evidence exists; otherwise the constraint is soft and the answer must say so).

Tri-state logic: a required attribute is satisfied only by `True`; `False` and `None` both fail a hard requirement, but the answer distinguishes "known not to have it" from "unknown". This distinction matters for explanation quality.

### 9.4 Reciprocal Rank Fusion

Fuse the dense and BM25 restaurant rankings:

```
RRF(d) = Σ_i 1 / (k + rank_i(d)),   k = 60
```

- Only documents retrieved by at least one retriever are scored.
- Ties are broken by the better dense rank, then by rating.
- Attribute filters are applied before fusion, not after, so that filtered-out restaurants cannot influence ranks.
- `k=60` is the standard default and is not tuned on the test set. If tuned, it must be tuned on a validation split and reported.

### 9.5 Reranking

Model: `BAAI/bge-reranker-base` (cross-encoder).

- Rerank the top 20 fused candidates.
- Score query–document pairs at the restaurant level: use the best chunk per restaurant, or concatenate the top 2 chunks if they fit within the model's limit. The choice is fixed and reported.
- Final order: rerank score, then RRF score, then rating.
- Reranker latency is measured separately.
- The reranker is optional in the deployed app if latency is unacceptable; the evaluation reports both settings.

### 9.6 Evidence layer

Retrieval returns restaurants, not quotes. After reranking:

1. For each of the top 5 candidates, select up to 8 candidate evidence sentences from that restaurant's evidence pool.
2. Rank sentences by BM25 against the rewritten query plus aspect terms extracted from the query.
3. Deduplicate near-identical sentences.
4. Require the sentence to be from a review with a star rating that does not contradict the claim (e.g., do not use a 1★ complaint to support "great food").
5. Pass the top 2–3 sentences per restaurant to the generator, each with `review_id` and `stars`.

This keeps the prompt small and makes citations precise. It also prevents the generator from quoting a review that was never retrieved.

### 9.7 Ranking policy

Final ranking combines:

1. hard-constraint satisfaction (pass/fail; failures are removed),
2. rerank relevance,
3. evidence strength (number of supporting sentences, agreement with rating),
4. rating as a minor tie-breaker,
5. personalization adjustments (Section 12).

Rating alone must never determine the order. This is a stated design requirement because high-rated restaurants are not automatically relevant to a specific request.

### 9.8 Latency budget

Targets for the app (to be validated):

| Stage | Target p95 |
|---|---:|
| Intent + preference extraction (LLM or rules) | < 1.0 s |
| Dense retrieval | < 50 ms |
| BM25 retrieval | < 50 ms |
| Fusion + filtering | < 20 ms |
| Reranking (20 candidates) | < 500 ms |
| Evidence selection | < 100 ms |
| Generation (256 tokens) | < 8 s |
| Validation | < 300 ms |
| **End-to-end** | **< 10 s** |

Targets are engineering goals, not reported results. Actual values come from `results/tables/latency.csv`.

### 9.9 Retrieval metrics

Defined precisely in Section 16.3. At minimum: Success@5, Precision@5, MRR, nDCG@5, and latency, computed at the restaurant level against the gold set.

---

## 10. Generation

### 10.1 Model

Base generator: `Qwen/Qwen2.5-1.5B-Instruct`.

Rationale:

- small enough for 8 GB VRAM in FP16 and 4-bit,
- instruction-tuned,
- 32k context, sufficient for facts plus evidence,
- supports LoRA/QLoRA,
- permits a clean precision comparison.

Fallback if output quality is unacceptable after prompt engineering: `Qwen2.5-3B-Instruct` in 4-bit. The switch must be documented and the full experiment matrix rerun, not selectively reported.

Loading rules:

- FP16 baseline: `torch_dtype=torch.float16`, `device_map="auto"`.
- 8-bit: `load_in_8bit=True` via bitsandbytes.
- 4-bit: NF4 with double quantization, compute dtype bfloat16.
- Set a fixed seed and greedy decoding for evaluation runs.
- Use the model's chat template, not a hand-written format.
- Unload the reranker and CLIP before loading the LLM when VRAM is tight; the model registry handles this.

### 10.2 Prompt architecture

The prompt has five blocks, in this order:

1. **System / role**: DineWise is a restaurant recommendation assistant grounded in a static dataset.
2. **Policy**: the full contents of `dining_preferences.docx`, marked as rules.
3. **Facts**: structured records for the candidate restaurants, injected from the database.
4. **Evidence**: retrieved review sentences and visual summaries, each with an ID.
5. **Task**: the user query, the extracted constraints, and the required output format.

Grounding rules in the system prompt:

- use only the supplied records and evidence;
- never invent names, ratings, prices, addresses, dietary facts, or amenities;
- never present dataset ratings or prices as live;
- never claim live availability;
- if evidence is insufficient, say so and return fewer recommendations;
- cite a source ID for every factual sentence;
- keep explanations to 2–3 sentences per restaurant.

The policy block is injected in full rather than retrieved. This prevents a retrieved fragment such as "prefer quiet restaurants" from being misinterpreted as a fact about a restaurant.

A compact example of the fact block:

```
[CANDIDATE 1]
business_id: B001
name: Restaurant ABC
city: Philadelphia, PA
cuisine: Italian, Pizza, Vegetarian
price_bucket: 2 ($$)
rating: 4.5 (dataset average over 312 reviews)
attributes: outdoor_seating=yes; noise=average; romantic=unknown;
            good_for=dinner,dessert
review_summary: ...
visual_summary: ...
evidence:
  [E1] review_id=R123 stars=5 "The vegetarian lasagna was excellent..."
  [E2] review_id=R456 stars=4 "Service was slow on a Saturday..."
```

### 10.3 Output schema

The model must return JSON matching this schema (see Appendix B for the full version):

```json
{
  "recommendations": [
    {
      "business_id": "B001",
      "name": "Restaurant ABC",
      "cuisine": "Italian",
      "price_bucket": 2,
      "rating": 4.5,
      "evidence": [
        {"review_id": "R123", "quote": "The vegetarian lasagna was excellent..."}
      ],
      "explanation": "Matches your vegetarian and budget constraints; reviewers praise the vegetarian options.",
      "sources": ["B001", "R123"]
    }
  ],
  "insufficient_evidence": false,
  "notes": "Ratings and prices are from the dataset, not live."
}
```

Constraints:

- `business_id` must be one of the supplied candidates.
- `rating` and `price_bucket` must match the database exactly.
- `evidence[].quote` must be a substring of a supplied evidence sentence, up to whitespace normalization.
- `sources` must contain the business ID and any cited review IDs.
- If no candidate satisfies the request, return an empty list and `insufficient_evidence: true`.

### 10.4 Constrained decoding and repair

1. Prefer schema-constrained generation. Options, in order:
   - `outlines` or `lm-format-enforcer` with a JSON schema, if compatible with the installed transformers version;
   - a grammar-constrained backend;
   - a strict JSON-only prompt with a repair loop.
2. If parsing fails, attempt one repair: strip code fences, fix trailing commas, and retry with the parse error appended.
3. If repair fails, log the raw output and return a structured error to the graph, which triggers the retry/fallback path.
4. Record `format_valid` for every generation. Format validity is a reported metric, not a hidden retry.

### 10.5 Validators

Validation is programmatic first and semantic second.

**Programmatic checks (hard failures):**

1. Every `business_id` exists in the candidate set.
2. Every name matches the database for that ID.
3. Every rating equals the database value (exact, one decimal).
4. Every price bucket equals the database value.
5. Every hard constraint is satisfied by the returned restaurant.
6. Every cited review ID exists and is associated with that restaurant.
7. Every quote is a substring of the supplied evidence.
8. No claim of live availability, current price, or booking.
9. The number of recommendations is between 0 and 5.

**Semantic checks (soft failures):**

10. Each explanation sentence is entailed by the cited evidence or structured facts (NLI model).
11. The answer addresses the user's actual request (relevance).
12. The answer acknowledges missing information when applicable.
13. The explanation does not overstate uncertain attributes (e.g., "romantic" when the attribute is unknown).

Soft failures do not automatically discard the answer. They are counted and reported, and they trigger a retry only when they exceed a configured threshold or when a hard constraint is violated.

**NLI model:** `MoritzLaurer/DeBERTa-v3-base-mnli` or an equivalent three-class NLI model. Premise = evidence + fact block; hypothesis = explanation sentence. Entailment probability below 0.5 flags the sentence. The model choice and threshold are recorded; the threshold is calibrated on a small manually labeled set, not guessed.

### 10.6 Failure handling

| Failure | Action |
|---|---|
| Invalid JSON | One repair attempt, then retry at graph level |
| Hallucinated restaurant | Drop the item; if all items invalid, retry |
| Rating/price mismatch | Drop the item; log a hard-failure count |
| Hard constraint violated | Retry once with a clarifying retrieval; never silently relax |
| Insufficient evidence | Return fewer results or fallback |
| Model OOM | Unload other models, retry in 4-bit, log the downgrade |
| Timeout | Fallback with a clear message |

Every failure is logged with a `run_id`, query ID, and stage.

---

## 11. LangGraph workflow

### 11.1 State

```python
class DineWiseState(TypedDict, total=False):
    query: str
    profile: dict
    intent: str                  # recommendation | comparison | visual | out_of_scope
    constraints: dict            # city, cuisines, max_price, min_rating, attributes, dietary
    soft_preferences: list[str]
    rewritten_query: str
    retrieved: list[dict]        # chunk-level hits
    candidates: list[dict]       # restaurant-level candidates
    evidence: dict[str, list[dict]]
    answer: dict                 # parsed JSON
    validation: dict             # hard_failures, soft_failures, scores
    retry_count: int
    retrieval_expanded: bool
    timings: dict
    errors: list[str]
```

### 11.2 Nodes

| Node | Function | Implementation notes |
|---|---|---|
| `classify_intent` | Recommendation, comparison, visual, or out-of-scope | Rule-based first; LLM fallback only for ambiguous cases. Comparison queries require extracting two restaurant names. |
| `extract_preferences` | Produce the typed constraint object | Hybrid: regex/lexicon for city, price words, cuisine, dietary terms; LLM for free-form phrasing. Validate against known values. |
| `rewrite_query` | Retrieval-friendly query | Remove conversational filler; add cuisine/city terms; keep dish and aspect terms. |
| `retrieve` | Dense + BM25 + optional visual | Returns restaurant-level candidates with scores. |
| `filter_candidates` | Apply hard constraints | Programmatic SQL/pandas filter against `restaurants.parquet`. |
| `rerank` | Cross-encoder ordering | Top 20 → top 5. |
| `generate_answer` | LLM call with schema | Facts and evidence injected; no free-form browsing. |
| `review_answer` | Programmatic + semantic validation | Hard failures trigger retry; soft failures are counted. |
| `retry` | Expand retrieval once | Relax soft preferences, broaden the query, keep hard constraints. |
| `fallback` | Safe insufficiency response | State what is missing; optionally offer constraint-satisfying alternatives with a caveat. |

### 11.3 Graph edges

```
START → classify_intent → extract_preferences → rewrite_query → retrieve
      → filter_candidates → rerank → generate_answer → review_answer
review_answer --supported--> END
review_answer --insufficient & retry_count == 0--> retry → retrieve
retry → ... → review_answer
review_answer --insufficient & retry_count == 1--> fallback → END
classify_intent --out_of_scope--> fallback → END
```

### 11.4 Retry semantics

- Maximum one retry per request. Enforced in state and in a test.
- The retry may:
  - broaden the lexical query,
  - add synonyms for cuisine or dish terms,
  - relax soft preferences (e.g., "romantic" if the attribute is unknown),
  - expand the candidate pool from 20 to 40 before reranking.
- The retry may **not**:
  - change the city,
  - exceed the budget bucket,
  - drop a dietary requirement,
  - invent evidence,
  - relax an explicit hard attribute the user requested.
- If the retry fails, the fallback states which constraint could not be satisfied.

### 11.5 Workflow metrics

- intent classification accuracy (macro-F1 over the gold test set),
- preference extraction precision, recall, and F1 per field,
- retrieval success percentage,
- retry rate,
- retry success rate,
- fallback accuracy on out-of-scope queries,
- end-to-end latency p50/p95,
- node-level latency breakdown.

---

## 12. Personalization

Personalization is explicit and inspectable, not learned from interactions.

**Profile fields:**

- home city or preferred cities,
- budget bucket,
- preferred cuisines,
- dietary requirements (vegetarian, vegan, gluten-free, halal, etc.),
- preferred ambience (romantic, quiet, family-friendly, lively),
- accessibility needs,
- dislikes (e.g., no seafood).

**Constraint classification:**

- Hard: city, budget ceiling, dietary requirement explicitly marked as required, accessibility.
- Soft: cuisine preference, ambience preference, rating preference.

**Application:**

1. Hard constraints are applied in `filter_candidates`.
2. Soft preferences contribute a small additive score after reranking:
   - +0.05 for a cuisine match,
   - +0.05 for an ambience match,
   - +0.03 for an attribute match,
   - weights are fixed before evaluation and reported.
3. In-session feedback:
   - "like" adds +0.05 to similar restaurants,
   - "dislike" subtracts 0.10,
   - similarity is defined by cuisine family and price bucket, not by embeddings, to keep the mechanism explainable.
4. The UI shows which preferences influenced the ranking.

**Evaluation (E11):**

- Run the gold queries with and without the profile.
- Measure preference satisfaction with a rubric: does the returned set respect cuisine, budget, ambience, and dietary constraints?
- Report both relevance (nDCG@5) and satisfaction. A personalization feature that improves satisfaction but harms relevance is not automatically a win.

**What this is not:** collaborative filtering, implicit learning from other users, or a trained recommender. The report must state this plainly.

---

## 13. Visual pipeline

Because the dataset contains no image files, the visual experiment is split into two honest components.

### 13.1 Component A — text-to-visual retrieval

- Index the `photo_summay.csv` text with BGE.
- Queries such as "romantic interior", "outdoor patio", "bright and modern" retrieve visual descriptions.
- Results map to restaurants via `business_id`.
- Metadata: `document_type=visual`, plus `venue_style`, `indoor_color_tone`, `target_audience`.
- Evaluation: 20 visual queries with manually identified gold restaurants, using the photo summary text as the relevance source. Metrics: Success@5, Precision@5, MRR.
- Claim: this improves retrieval for ambience queries where structured attributes are sparse. It is **not** image understanding.

### 13.2 Component B — image-to-cuisine via CLIP zero-shot

Model: `openai/clip-vit-base-patch32`.

Process:

1. Build 101 text prompts from Food-101 class names: `"a photo of {class}, a type of food"`.
2. Compute CLIP text embeddings for all prompts.
3. Encode the uploaded image with the CLIP image encoder.
4. Compute cosine similarity; take the top-5 food classes and their probabilities.
5. Map food classes to cuisine families with a hand-built table (e.g., pizza → Italian, sushi → Japanese, tacos → Mexican, paella → Spanish, dumplings → Chinese).
6. Use the mapped cuisine as a **soft** filter or query expansion term.
7. Show the top-3 predicted classes in the UI with confidence, so the user can see the uncertainty.

Rules:

- Never claim that the image is from a specific restaurant.
- Never use the image to assert food quality or authenticity.
- If the top-1 probability is below a threshold (calibrated on a validation subset), do not apply a cuisine filter.
- Images are processed in memory and not stored unless the user opts in.

Evaluation:

- Download a stratified subset of Food-101 validation images via streaming (10 per class = 1,010 images) rather than the full 5 GB dataset.
- Measure top-1 and top-5 **class** accuracy.
- Measure mapped-cuisine accuracy on classes with a confident mapping.
- Report coverage: how many of the 101 classes have an unambiguous cuisine mapping.

### 13.3 Fusion

For a visual query:

- Component A contributes restaurant candidates directly.
- Component B contributes a cuisine hint only.
- Fuse with the text retrieval results using the same RRF procedure, with the visual list treated as another ranked list.
- The final answer must state that visual matching is based on descriptions or predicted food class, not on actual restaurant photos.

### 13.4 Metrics and limitations

Metrics: visual Success@5, visual Precision@5, visual MRR, CLIP top-1/top-5 class accuracy, mapped-cuisine accuracy, visual retrieval latency, and the effect on end-to-end nDCG.

Limitations to report:

- no ground-truth restaurant images,
- photo summaries are GPT-4o generated,
- Food-101 classes do not cover all cuisines,
- cuisine inference from a single dish is weak evidence,
- visual similarity does not establish restaurant identity, authenticity, or quality.

---

## 14. LoRA fine-tuning

### 14.1 Objective

Improve **response behavior** — format compliance, explanation structure, constraint acknowledgement, abstention — not factual knowledge. Restaurant facts must continue to come from retrieval.

Training a 1.5B model to memorize 6,033 restaurants is both infeasible and undesirable: it would increase hallucination and make the system stale.

### 14.2 Data construction

Target: 400 instruction–response examples (range 300–500).

Task types and approximate counts:

| Type | Count | Example instruction |
|---|---:|---|
| Budget recommendation | 60 | "Recommend a cheap Italian place in Tampa." |
| Cuisine recommendation | 60 | "Find good Vietnamese food in Philadelphia." |
| Dietary recommendation | 60 | "I need vegetarian options under $$." |
| Ambience recommendation | 50 | "Suggest a quiet restaurant for a date." |
| Comparison | 50 | "Compare Restaurant A and Restaurant B on food and service." |
| Review-grounded recommendation | 50 | "Which restaurants are praised for desserts?" |
| Out-of-scope / abstention | 40 | "Is Restaurant A available tonight at 8?" |
| Multi-constraint | 30 | "Family-friendly Mexican place in Tucson under $$." |

Construction method:

1. Sample restaurants and constraints from the training split only.
2. Build the context block from the actual structured record and retrieved evidence.
3. Generate the target answer with a deterministic template that:
   - cites the business ID and evidence IDs,
   - states the matched constraints,
   - uses 2–3 sentences,
   - abstains when constraints cannot be met.
4. Optionally paraphrase instructions with a teacher LLM if an API key is available. Every synthetic example must be manually inspected in a sample; synthetic data is not ground truth.
5. Manually review a random 15% (60 examples). Record the error rate and fix the templates. Do not ship examples with incorrect facts.

Leakage control:

- Split by restaurant, not by example, so the same restaurant never appears in both training and test.
- Split by instruction template family as a secondary check.
- Verify zero business-ID overlap between splits.

### 14.3 Splits

- 80% train (320),
- 10% validation (40),
- 10% test (40).

The 40 test examples are for format/behavior evaluation only. The main RAG evaluation uses the 120-query gold set, which is separate and never used for training.

### 14.4 Hyperparameters (starting point)

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | QLoRA (4-bit NF4, double quant) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | all linear projections (q, k, v, o, gate, up, down) |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Batch size | 1 per device, gradient accumulation 4 |
| Max sequence length | 1024 |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Precision | bfloat16 compute |
| Seed | 42 |
| Early stopping | validation loss, patience 1 epoch |

These are experimental settings, not claimed optima. Any tuning is done on validation data and reported.

### 14.5 Training procedure

1. Freeze the base model; train adapters only.
2. Log loss, learning rate, gradient norm, and VRAM every 10 steps.
3. Save the adapter, tokenizer, and training config.
4. Evaluate on the validation split; select the best checkpoint by validation loss and format validity.
5. Merge the adapter for the latency comparison (E8, M5); keep the unmerged adapter for the quality comparison (E7).

Expected training time on the RTX 4060: 30–90 minutes for 400 examples and 2 epochs. This is an estimate; measure and report.

### 14.6 Evaluation

Compare four systems on the same 120-query gold set:

| System | Purpose |
|---|---|
| Qwen base, no retrieval | Lower bound; shows hallucination |
| Qwen + RAG (no LoRA) | Grounded baseline |
| Qwen + RAG + LoRA | Adaptation effect |
| Qwen + LoRA, no retrieval | Tests whether tuning substitutes for retrieval |

Metrics: format validity, constraint satisfaction, faithfulness, unsupported-claim rate, answer relevance, abstention correctness, and nDCG@5.

Hypothesis: LoRA improves format and explanation quality; it does not improve factual accuracy beyond RAG, and without RAG it increases unsupported claims. If the data contradicts this, report the contradiction.

---

## 15. Efficiency experiments

### 15.1 Configurations

| ID | Configuration | Purpose |
|---|---|---|
| M1 | FP16 | Baseline quality and speed |
| M2 | INT8 (LLM.int8) | Memory reduction with minimal quality loss |
| M3 | INT4 (NF4, double quant) | Maximum memory reduction |
| M4 | LoRA adapter (FP16 base) | Adaptation cost |
| M5 | LoRA + INT4 | Deployed configuration |

### 15.2 Measurement protocol

- Same prompt set: 20 fixed prompts (queries plus context), same token budget.
- Warm-up: 3 runs discarded.
- Measurement: 5 runs per prompt; report median and IQR.
- Decoding: greedy, `max_new_tokens=256`, fixed stop tokens.
- Batch size 1, because the application is single-user.
- Record:
  - model disk size,
  - peak GPU memory (`torch.cuda.max_memory_allocated` and `nvidia-smi`),
  - TTFT (time to first token),
  - TPOT (time per output token),
  - tokens/second,
  - end-to-end latency,
  - output quality on the gold set.

### 15.3 Interpretation rules

- Do not claim that quantization is uniformly faster. At batch size 1, 4-bit may be slower because of dequantization overhead.
- Do not compare configurations measured under different prompt lengths or output budgets.
- Report variance, not just means.
- If a configuration fails to load on 8 GB, report that as a result, not as a failure to hide.

### 15.4 Pruning and distillation

Explicitly out of scope. If mentioned in the report, they belong in Future Work with a statement that they were not performed. Do not describe INT8/INT4 as "pruning".

---

## 16. Evaluation framework

### 16.1 Gold set

**Size:** 120 queries for the full-semester plan (105 answerable + 15 out-of-scope).

**Composition of answerable queries (105):**

| Facet | Count | Example |
|---|---:|---|
| Cuisine | 20 | "Find Italian restaurants in Philadelphia." |
| Budget | 15 | "Recommend restaurants under $$." |
| Review aspect | 20 | "Which places are praised for desserts?" |
| Attribute / ambience | 15 | "Romantic restaurant with outdoor seating." |
| Comparison | 10 | "Compare Restaurant A and Restaurant B." |
| Multi-constraint | 15 | "Family-friendly Mexican under $$ in Tucson." |
| Visual | 10 | "Show restaurants with bright modern interiors." |

**Out-of-scope queries (15):**

- live availability or booking (5),
- city not in the dataset, e.g., Las Vegas (3),
- amenity absent from records (3),
- medical/allergy guarantees (2),
- requests for real-time prices or menus (2).

Each query record contains:

```json
{
  "query_id": "Q001",
  "query": "Find Italian restaurants with vegetarian options under $$ in Philadelphia.",
  "facet": "multi_constraint",
  "intent": "recommendation",
  "constraints": {
    "city": "Philadelphia",
    "cuisines": ["Italian"],
    "max_price_bucket": 2,
    "required_attributes": [],
    "dietary": ["vegetarian"],
    "min_rating": null
  },
  "gold_business_ids": [
    {"business_id": "B001", "grade": 2},
    {"business_id": "B042", "grade": 1}
  ],
  "expected_evidence_aspects": ["vegetarian", "food_quality"],
  "out_of_scope": false,
  "notes": "Gold set constructed from structured filters plus manual review."
}
```

### 16.2 Annotation protocol

1. **Prefill.** For each query, a script selects restaurants satisfying the hard constraints and ranks them by a transparent heuristic (attribute match, rating, review count). The top 20 are shown to the annotator.
2. **Grade.** The annotator assigns:
   - **2** — satisfies all hard constraints and has explicit evidence for the requested aspect;
   - **1** — satisfies all hard constraints but lacks explicit evidence for the aspect;
   - **0** — does not satisfy a hard constraint or is irrelevant.
3. **Bound the gold set.** Keep at most 10 graded-2/1 restaurants per query. If more than 10 qualify, prefer diversity across cuisine and price, and record the selection rule. This keeps evaluation tractable and makes Recall@k meaningful.
4. **Second pass.** Re-label 20 queries after at least one week; report intra-annotator agreement (Cohen's kappa). If a second annotator is available, report inter-annotator agreement on those 20.
5. **Freeze.** Once frozen, the gold set is versioned and never edited during experiments. Corrections require a new version and a rerun.

Estimated effort: 8–10 hours, spread across weeks 3–14.

### 16.3 Retrieval metrics

All metrics are computed at the **restaurant level**, not the chunk level, because the user sees restaurants.

Let `G` be the gold set for a query, `R_k` the top-k retrieved restaurants, and `rel(d)` the graded relevance.

**Success@k (hit rate)** — at least one relevant restaurant in the top k. This is the primary retrieval metric for recommendation, because a gold set of 5–10 items cannot support a meaningful Recall@5 over a corpus of 6,000 restaurants.

```
Success@k = 1[ |R_k ∩ G| > 0 ]
```

**Precision@k**

```
P@k = |R_k ∩ G| / k
```

**Mean Reciprocal Rank**

```
MRR = (1/|Q|) Σ_q 1 / rank_q(first relevant)
```

**nDCG@k** with graded gains:

```
DCG@k = Σ_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

**Recall@k** is reported only for the curated gold sets (≤10 items) and is labeled "Recall@k over curated gold set". It is not reported as if the gold set were complete.

Additional metrics:

- Attribute-filter compliance rate: percentage of returned restaurants that satisfy the hard constraints (should be 100%; any failure is a bug).
- Cuisine match rate.
- City match rate.
- Price compliance rate.
- Retrieval latency p50/p95.
- Index size.

### 16.4 Generation metrics

**Programmatic (all 120 queries):**

- constraint satisfaction rate (all hard constraints),
- rating accuracy (returned rating equals database),
- price accuracy,
- citation existence rate,
- citation support rate (quote is a substring of supplied evidence),
- unsupported-claim rate (claims not entailed by context),
- abstention correctness on out-of-scope queries,
- format validity,
- number of recommendations returned,
- fallback rate and retry rate.

**Semantic (RAGAS, stratified subset of 40 answers):**

- faithfulness,
- answer relevancy,
- context precision,
- context recall (requires reference answers or gold contexts; otherwise report as not applicable).

**Human spot check:** 30 answers, judged for usefulness, correctness, and explanation quality on a 1–5 scale. Report mean and agreement.

**Judge model:** RAGAS requires an LLM judge. Options, in order of preference:

1. A strong API model (Gemini Flash or GPT-class) if a key is available; record the exact model and date.
2. A local `Qwen2.5-7B-Instruct` in 4-bit, run as a separate offline process; record the model and acknowledge self-preference bias.
3. A local NLI model only, reporting entailment-based faithfulness without RAGAS.

The judge model is always named in the results table. The project does not present LLM-judge scores as objective truth; they are triangulated with programmatic checks and human spot checks.

### 16.5 Programmatic validators

Implemented as a reusable module with unit tests. Each validator returns `{passed, reason, evidence}`.

| Validator | Check | Severity |
|---|---|---|
| `check_business_exists` | ID in candidate set and catalogue | hard |
| `check_name` | Name matches catalogue | hard |
| `check_rating` | Rating equals catalogue to one decimal | hard |
| `check_price` | Price bucket equals catalogue | hard |
| `check_hard_constraints` | City, cuisine, price, attributes, dietary | hard |
| `check_citations_exist` | Cited review IDs exist for that restaurant | hard |
| `check_quote_substring` | Quote appears in supplied evidence | hard |
| `check_no_live_claims` | No booking/availability/current-price language | hard |
| `check_count` | 0 ≤ recommendations ≤ 5 | hard |
| `check_entailment` | Explanation sentences entailed by evidence | soft |
| `check_uncertainty_language` | Unknown attributes not stated as facts | soft |
| `check_acknowledgement` | Missing evidence acknowledged | soft |

The hard/soft distinction is fixed before the experiments and reported.

### 16.6 RAGAS configuration

- `ragas` version pinned in `requirements.lock`.
- Metrics: faithfulness, answer relevancy, context precision, context recall (when reference contexts exist).
- Embeddings for RAGAS: the same BGE model, to avoid introducing another variable.
- Judge: as above.
- Run on 40 answers stratified by facet, not on all 120, for cost and time reasons. The subset is fixed and reported.
- Report per-metric mean with bootstrap confidence intervals.

### 16.7 Statistical reporting

- Report mean, median, standard deviation, and 95% bootstrap confidence intervals (1,000 resamples) for every headline metric.
- For A/B comparisons, use a paired bootstrap over queries and report the mean difference, its 95% CI, and a two-sided p-value.
- Do not declare a winner when the confidence interval includes zero. State that the difference is inconclusive.
- Report the number of queries, the number of gold items, and any exclusions.
- Never compare runs with different gold-set versions or different query sets.

### 16.8 Synthetic evidence audit

Because review and photo summaries are LLM-generated, the project must estimate their reliability.

Protocol:

1. Sample 50 restaurants stratified by review count.
2. For each, sample 3 claims from `review_summary.csv`.
3. Check each claim against the raw reviews for that restaurant.
4. Label as supported, partially supported, or unsupported.
5. Report the support rate and examples of drift.

This audit is a validity check on the evidence layer, not a core experiment. It belongs in the results and limitations sections.

---

## 17. Experiment matrix

Every experiment changes one major component. The corpus, gold set, and remaining settings stay fixed. Each row produces one record in `results/runs/` and one row in `results/tables/experiments.csv`.

| ID | Variable | Fixed | Metrics | What it proves |
|---|---|---|---|---|
| E0 | Data pipeline (no model) | — | audit table, parse rate, coverage | Data quality baseline |
| E1 | BM25 only | Corpus, queries | Success@5, P@5, MRR, nDCG@5, latency | Lexical baseline |
| E2 | Dense only (BGE + FAISS) | Same | Same | Semantic retrieval benefit |
| E3 | Hybrid RRF (dense + BM25) | Same | Same | Fusion benefit |
| E4 | Hybrid + cross-encoder rerank | Same | Same + rerank latency | Ranking improvement vs cost |
| E5 | Chunking C1–C4 (on the best retriever from E1–E4) | Same queries, same embedding model | Chunk stats + retrieval + faithfulness | Chunking effect |
| E6 | Visual retrieval: off / text-to-visual / + CLIP zero-shot | Same | Visual Success@5, cuisine accuracy, end-to-end nDCG | Value and limits of visual context |
| E7 | Base vs RAG vs RAG+LoRA vs LoRA-only | Same | Faithfulness, constraint satisfaction, format validity, unsupported claims | Adaptation effect |
| E8 | FP16 / INT8 / INT4 / LoRA-4bit | Same prompts | TTFT, TPOT, tok/s, peak VRAM, disk, quality | Efficiency trade-off |
| E9 | Linear pipeline vs full LangGraph (validator + retry + fallback) | Same | Unsupported-claim rate, fallback accuracy, retry rate, latency | Value of orchestration |
| E10 | Main corpus vs held-out cities (Nashville, Reno) | Same method | Metric degradation | Generalization |
| E11 | Profile off vs on | Same | Preference satisfaction, nDCG@5 | Personalization value |

Experiment dependencies:

- E1–E5 require the gold set.
- E6 requires the visual gold subset.
- E7 requires the LoRA adapter and the 120-query set.
- E8 requires a fixed prompt set and a stable measurement harness.
- E9 requires both a linear runner and the graph runner.
- E10 requires a second index built over the held-out cities.
- E11 requires the profile definitions.

Ordering: E0 → E1–E4 → E5 → E9 (with the best retriever) → E6 → E7 → E8 → E10 → E11. This front-loads the retrieval contribution, which is the strongest part of the project.

Observation table template (to be filled with measured values only):

| Exp | Config | Success@5 | P@5 | MRR | nDCG@5 | Faithfulness | Latency p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | BM25 | | | | | | |
| E2 | Dense | | | | | | |
| E3 | Hybrid | | | | | | |
| E4 | Hybrid + rerank | | | | | | |
| E5 | Best chunking | | | | | | |
| E6 | + visual | | | | | | |
| E7 | + LoRA | | | | | | |
| E9 | Full graph | | | | | | |

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
│   └── indexes/              # FAISS, BM25, embeddings cache
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
│   │   ├── faiss_index.py
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
  "packages": {"torch": "...", "transformers": "...", "faiss-cpu": "..."},
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

## 19. Phases and schedule

### 19.1 Phase details and acceptance criteria

#### P0 — Environment and skeleton (week 1, ~0.5 day)

Work:

- create the 3.12 venv with `uv`,
- install CUDA PyTorch, transformers, sentence-transformers, faiss-cpu, rank-bm25, langchain, langgraph, pypdf, docx2txt, python-docx, reportlab, streamlit, ragas, peft, bitsandbytes,
- configure `HF_HOME`,
- create the repo skeleton and config,
- write the download script with SHA-256 checksums,
- run a GPU and model smoke test.

Artifacts: `results/env_report.json`, `requirements.lock`, `README.md`.

Acceptance:

- `torch.cuda.is_available()` is true and reports the RTX 4060,
- all packages import,
- the download script verifies all six files,
- a 10-token generation runs on the GPU.

#### P1 — Data layer (week 1, ~2 days)

Work:

- implement the attribute parser with tests,
- implement the cuisine taxonomy,
- implement the food-service filter,
- clean and join the five tables,
- build `restaurants.parquet`, `reviews.parquet`, and `evidence.parquet`,
- produce the audit table.

Acceptance:

- 6,033 restaurants in the study corpus (or a documented deviation),
- 385,065 reviews joined (or a documented deviation),
- attribute parse success ≥98%,
- zero duplicate business IDs,
- 100% referential integrity for review→business and photo→business,
- audit table generated and summarized in the report.

#### P2 — Document layer (week 2, ~1.5 days)

Work:

- generate the TXT, PDF, and DOCX documents,
- implement the loaders and metadata attachment,
- cache extracted PDF text,
- produce document statistics.

Acceptance:

- all three formats load,
- zero empty documents,
- metadata completeness ≥99%,
- PDF page count and extraction time recorded,
- documents regenerate deterministically from the same input hash.

#### P3 — Chunking (week 2, ~1.5 days)

Work:

- implement restaurant-aware token chunking,
- build C1–C4,
- implement contamination checks,
- produce chunk statistics.

Acceptance:

- 0% cross-restaurant contamination,
- metadata completeness 100% for required fields,
- token length distributions within ±10% of target medians,
- chunk counts and index sizes recorded for each configuration.

#### P4 — Retrieval and gold set (weeks 3–4, ~4 days)

Work:

- build the FAISS and BM25 indexes,
- implement collapse, RRF, filters, and reranker,
- build the prefilled gold-set tool,
- label 60 queries (v1) and extend to 120 later,
- run E1–E4.

Acceptance:

- every returned restaurant satisfies hard constraints (100%),
- E1–E4 table produced by one command,
- gold-set v1 frozen and versioned,
- retrieval latency recorded.

#### P5 — Generation and workflow (weeks 5–6, ~4 days)

Work:

- load Qwen2.5-1.5B,
- implement prompts and JSON schema,
- implement constrained decoding and repair,
- implement validators and NLI checks,
- implement the LangGraph nodes and retry/fallback,
- run the 120-query generation evaluation.

Acceptance:

- format validity ≥95% after repair,
- hard-constraint violations = 0,
- hallucinated restaurants = 0,
- out-of-scope queries produce correct abstention,
- retry and fallback paths covered by tests,
- E7 baseline and E9 results available.

#### P6 — App (week 7, ~2 days)

Work:

- Streamlit chat, profile, visual tab, comparison, debug tab,
- connect to the Python API.

Acceptance:

- cold start and p95 latency measured,
- no fabricated fields displayed,
- sources visible for every recommendation,
- abstention path demonstrable.

#### P7 — Visual pipeline (weeks 9–10, ~3 days)

Work:

- build the visual text index,
- implement CLIP zero-shot classification,
- create the visual gold subset,
- run E6.

Acceptance:

- visual Success@5 and CLIP top-1/top-5 reported,
- limitations stated in the UI and report,
- no claim of image-to-restaurant matching.

#### P8 — LoRA (weeks 11–12, ~4 days)

Work:

- construct 400 examples,
- review 15% manually,
- train QLoRA,
- run E7 and the LoRA-only ablation.

Acceptance:

- zero business-ID overlap between splits,
- training log and adapter saved,
- format validity and faithfulness compared across four systems,
- no claim that LoRA adds factual knowledge.

#### P9 — Efficiency (week 13, ~2 days)

Work:

- run FP16/INT8/INT4/LoRA-4bit,
- measure TTFT, TPOT, tokens/s, peak VRAM, disk,
- run the quality comparison on a fixed subset.

Acceptance:

- measurement protocol followed (warm-up, 5 runs, fixed prompts),
- variance reported,
- any load failure reported as a result.

#### P10 — Ablations and generalization (week 14, ~3 days)

Work:

- E9 graph ablation,
- E10 held-out cities,
- E11 personalization,
- final statistical tests.

Acceptance:

- paired bootstrap CIs for every comparison,
- no winner declared when the CI includes zero,
- all tables generated by script.

#### P11 — Report and demo (weeks 15–16, ~4 days)

Work:

- write the report,
- generate figures,
- record a short demo,
- final reproducibility check.

Acceptance:

- a fresh clone can run `make setup && make data && make index && make eval` to reproduce the tables,
- every reported number traces to a results file,
- limitations section covers all items in Section 21.

### 19.2 Sixteen-week schedule

| Week | Focus | Milestone |
|---:|---|---|
| 1 | P0 + P1 | Environment verified; clean catalogue |
| 2 | P2 + P3 | Documents and chunks built; contamination = 0 |
| 3 | P4 start | Indexes built; gold-set tool ready |
| 4 | P4 | E1–E4 results; gold-set v1 frozen |
| 5 | P5 | Qwen loaded; prompts and validators working |
| 6 | P5 | Full 120-query generation run; E9 baseline |
| 7 | P6 | Streamlit demo working |
| 8 | Buffer + mid-project checkpoint | First complete report draft with E0–E5 |
| 9 | P7 | Visual text index + CLIP zero-shot |
| 10 | P7 | E6 results; visual gold subset |
| 11 | P8 | LoRA data + training |
| 12 | P8 | E7 results; adapter evaluated |
| 13 | P9 | E8 efficiency results |
| 14 | P10 | E9–E11, held-out city, statistics |
| 15 | Report | Full draft, figures, tables |
| 16 | Buffer | Reproducibility check, demo, submission |

Buffer is deliberate. Dataset download, PDF extraction, and model installation are the most likely sources of schedule slip.

---

## 20. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | 8 GB VRAM cannot co-load reranker, CLIP, LLM, and training | High | Medium | Model registry with explicit unload; sequential stages; 4-bit fallback; measure peak memory |
| 2 | Qwen 1.5B produces invalid JSON | High | Medium | Schema-constrained decoding; repair loop; format validity as a metric; consider 3B if needed |
| 3 | PDF extraction is slow or low quality | Medium | Medium | Cache extracted text; compact layout; per-city split if needed; record extraction time |
| 4 | Review/photo summaries contain unsupported claims | High | High | Synthetic evidence audit; prefer raw reviews for quotes; validator entailment checks |
| 5 | Gold-set labeling takes longer than planned | High | High | Prefill candidates; cap gold set at 10; label incrementally; freeze at 60 if necessary and report |
| 6 | Retrieval differences are not statistically significant | Medium | Medium | Report CIs; do not overclaim; larger gold set; analyze per-facet |
| 7 | LoRA degrades grounding | Medium | High | Include LoRA-only ablation; validate hard constraints; do not merge adapters before evaluation |
| 8 | Dataset download or license issue | Low | High | Legal use is non-commercial research; do not redistribute raw data; cite sources |
| 9 | Scope creep (20 notebook sections, many experiments) | High | High | Phase gates; MVP first; advanced experiments only after P6 passes |
| 10 | Reproducibility drift from library updates | Medium | Medium | Pin versions; lockfile; run manifest; content-hash caches |
| 11 | Ambience attributes too sparse (149 romantic=True) | High | Medium | Use review and visual summaries as soft evidence; state uncertainty; do not treat missing as false |
| 12 | CLIP cuisine mapping is unreliable | High | Medium | Soft signal only; confidence threshold; report top-1/top-5 accuracy and coverage |
| 13 | LLM judge bias in RAGAS | Medium | Medium | Triangulate with programmatic checks and human spot checks; name the judge |
| 14 | GPU thermal throttling on a laptop | Medium | Low | Warm-up runs; report variance; use median |
| 15 | Time lost to environment issues | Medium | Medium | P0 acceptance tests; freeze the environment before experiments |

---

## 21. Threats to validity and limitations

### Construct validity

- **Ratings and prices are dataset snapshots**, not live values. The system cannot answer "is it good now?" or "how much does it cost?".
- **Price bucket ≠ dollars.** Budget interpretation is approximate and documented.
- **Cuisine is derived** from category tokens, not an official label. Multi-label and ambiguous cases exist.
- **Dietary evidence is textual.** Absence of evidence is not evidence of absence, and medical guarantees are impossible.
- **Ambience attributes are sparse.** Using review and photo summaries introduces subjectivity.
- **"Personalization" is explicit profile matching**, not a learned user model.

### Internal validity

- Gold-set labels are subjective. Mitigate with a rubric, a second pass, and agreement reporting.
- LLM-generated summaries may bias both retrieval and evaluation. Mitigate with the synthetic evidence audit and raw-review support.
- Reranker and LLM changes could interact. Mitigate with one-variable-at-a-time experiments.
- Prompt changes can dominate effects. Freeze prompts before the main experiments and version them.
- Randomness in generation is controlled with greedy decoding; sampling runs are reported separately if used.

### External validity

- Five cities, US-centric, one platform (Yelp). Results may not transfer to other cities or countries.
- The held-out city experiment tests only two additional cities.
- The 1.5B model is a small model; conclusions may not hold for larger models.
- The 8 GB GPU constrains batch sizes and may affect latency conclusions.

### Conclusion validity

- Multiple comparisons increase false-positive risk. Report CIs and avoid claiming significance without them.
- Small gold sets produce wide intervals. Report them.
- Do not claim causality from observational retrieval metrics; the experiments are controlled but the data is not.

### Ethical and legal

- The dataset is for non-commercial research; do not redistribute raw data.
- Do not scrape Yelp or download restaurant photos.
- Do not present the system as a source of medical dietary advice.
- Do not collect or store user images without consent.
- Attribute LLM-generated summaries and state their provenance.

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

## 23. Assumptions and decisions log

| # | Decision | Rationale | Revisit condition |
|---|---|---|---|
| D1 | Python 3.12 venv via `uv` | 3.14 lacks ML wheels | If a dependency requires another version |
| D2 | Local RTX 4060, not Colab | Persistence, no session limits | If the GPU fails or VRAM is insufficient |
| D3 | Five-city study corpus | Tractable PDF, index, and annotation | If results are too narrow; run `--all-cities` sensitivity |
| D4 | BGE-small for text embeddings | Strong small model, fast on 8 GB | If retrieval quality is poor |
| D5 | Qwen2.5-1.5B-Instruct | Fits FP16 and 4-bit; supports LoRA | If format validity or quality is unacceptable; try 3B |
| D6 | FAISS `IndexFlatIP` | Exact search; avoids confounding | If corpus grows beyond ~1M vectors |
| D7 | BM25 + dense + RRF | Standard hybrid baseline | If hybrid does not help |
| D8 | Cross-encoder reranker optional | Quality vs latency trade-off | If latency budget is exceeded |
| D9 | No image-to-restaurant matching | No images exist | If licensed images become available |
| D10 | Explicit profile personalization | Explainable, no interaction data | If implicit feedback data becomes available |
| D11 | Programmatic validators primary | Avoid LLM-judge circularity | Never |
| D12 | LoRA after RAG baseline | Isolate adaptation effect | Never |
| D13 | No pruning or distillation | Out of scope; separate techniques | If the lab explicitly requires them |
| D14 | Gold set capped at 10 per query | Makes Recall@k meaningful and labeling tractable | If a larger labeled set is feasible |

---

## 24. References

1. Yelp Open Dataset. https://www.yelp.com/dataset
2. Yelp-Multimodal-Recommendation dataset. https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation
3. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
4. Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.
5. Xiao, S., et al. (2023). C-Pack: Packaged Resources to Advance General Chinese Embedding. (BGE models) arXiv:2309.07597.
6. BAAI. bge-small-en-v1.5 model card. https://huggingface.co/BAAI/bge-small-en-v1.5
7. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.
8. Johnson, J., Douze, M., Jégou, H. (2019). Billion-scale Similarity Search with GPUs (FAISS). IEEE Transactions on Big Data.
9. Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
10. Järvelin, K., Kekäläinen, J. (2002). Cumulated Gain-based Evaluation of IR Techniques. ACM TOIS.
11. Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR.
12. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
13. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
14. Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS.
15. Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
16. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
17. LangGraph documentation. https://langchain-ai.github.io/langgraph/
18. LangChain documentation. https://python.langchain.com/
19. Food-101 dataset. https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
20. Bossard, L., Guillaumin, M., Van Gool, L. (2014). Food-101: Mining Discriminative Components with Random Forests. ECCV.
21. MoritzLaurer. DeBERTa-v3-base-mnli model card. https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli
22. Streamlit documentation. https://docs.streamlit.io/

---

## Appendix A — Chunk and document metadata

Required metadata on every chunk:

| Field | Type | Nullable | Source |
|---|---|---|---|
| `chunk_id` | string | no | generated |
| `business_id` | string | yes (policy) | parent document |
| `restaurant_name` | string | yes | catalogue |
| `city` | string | yes | catalogue |
| `state` | string | yes | catalogue |
| `cuisine_primary` | string | yes | catalogue |
| `price_range` | int | yes | catalogue |
| `document_type` | enum | no | parent document |
| `section` | string | yes | parent section |
| `source_file` | string | no | loader |
| `page` | int | yes | loader |
| `token_count` | int | no | chunker |
| `char_count` | int | no | chunker |
| `chunk_config` | string | no | experiment config |
| `text` | string | no | chunker |

Validation rules:

- `document_type=policy` ⇒ `business_id` is null and `section` is a policy heading.
- `document_type in {profile, review_insight, visual}` ⇒ `business_id` is non-null.
- `token_count` ≤ 512 for the BGE index (or truncation is explicitly recorded).

---

## Appendix B — JSON output schema (full)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["recommendations", "insufficient_evidence"],
  "additionalProperties": false,
  "properties": {
    "recommendations": {
      "type": "array",
      "maxItems": 5,
      "items": {
        "type": "object",
        "required": [
          "business_id", "name", "cuisine", "price_bucket",
          "rating", "evidence", "explanation", "sources"
        ],
        "additionalProperties": false,
        "properties": {
          "business_id": {"type": "string"},
          "name": {"type": "string"},
          "cuisine": {"type": "string"},
          "price_bucket": {"type": "integer", "minimum": 1, "maximum": 4},
          "rating": {"type": "number", "minimum": 1, "maximum": 5},
          "evidence": {
            "type": "array",
            "maxItems": 3,
            "items": {
              "type": "object",
              "required": ["review_id", "quote"],
              "additionalProperties": false,
              "properties": {
                "review_id": {"type": "string"},
                "quote": {"type": "string", "minLength": 10, "maxLength": 400}
              }
            }
          },
          "explanation": {"type": "string", "minLength": 20, "maxLength": 600},
          "sources": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
          }
        }
      }
    },
    "insufficient_evidence": {"type": "boolean"},
    "notes": {"type": "string", "maxLength": 400}
  }
}
```

---

## Appendix C — Prompt templates

### C.1 System prompt

```
You are DineWise, a restaurant recommendation assistant.

You answer only from the RESTAURANT RECORDS and REVIEW EVIDENCE supplied
in this prompt. You do not have access to the internet, live availability,
current prices, or reservations.

Rules:
1. Never invent a restaurant, address, rating, price, dish, dietary fact,
   or amenity.
2. Ratings and prices come from a static dataset. Describe them as dataset
   values, never as live facts.
3. Never claim that a restaurant is currently open, has a table available,
   or offers a reservation.
4. Every factual sentence must be supported by the supplied records or
   evidence, and must cite a business ID or review ID.
5. If the records do not contain enough evidence, set
   "insufficient_evidence": true and explain what is missing.
6. Prefer fewer, well-supported recommendations over more, weakly supported
   ones.
7. Keep each explanation to 2-3 sentences.
8. Return valid JSON only, matching the provided schema.
```

### C.2 Preference extraction prompt

```
Extract the user's dining constraints.

Return JSON with:
- city (string or null)
- cuisines (array of strings)
- max_price_bucket (1-4 or null)
- min_rating (number or null)
- required_attributes (array: outdoor_seating, romantic, quiet,
  reservations, good_for_kids, vegetarian, vegan, gluten_free, ...)
- dietary_terms (array of strings)
- ambience_terms (array of strings)
- soft_preferences (array of strings)
- intent (recommendation | comparison | visual | out_of_scope)

Do not invent constraints that the user did not state.
If a term is ambiguous, put it in soft_preferences.
```

### C.3 Generation prompt (user block)

```
USER QUERY:
{query}

EXTRACTED CONSTRAINTS:
{constraints_json}

USER PROFILE:
{profile_json}

POLICY (rules, not facts):
{policy_text}

RESTAURANT RECORDS:
{facts_block}

REVIEW EVIDENCE:
{evidence_block}

TASK:
Recommend up to 5 restaurants that satisfy the hard constraints.
For each, give the business ID, name, cuisine, price bucket, dataset rating,
up to 3 evidence quotes with review IDs, a short explanation, and sources.
If no restaurant satisfies the constraints, return an empty list and set
insufficient_evidence to true.
Return JSON only.
```

### C.4 Retry prompt addition

```
The previous answer failed validation for: {failure_reasons}.

Retry using only the same restaurant records. You may:
- broaden the search terms,
- use alternative names for the requested cuisine or dish.

You may not:
- change the city,
- exceed the budget,
- drop a dietary requirement,
- invent evidence.

If the constraints cannot be met, return an empty list with
insufficient_evidence = true.
```

### C.5 Fallback template

```
I could not find a restaurant in the dataset that satisfies all of these
constraints: {constraints}.

What is missing:
{missing}

I can offer the closest alternatives only if you relax one of these
requirements: {relaxable}.

Note: ratings and prices are from a static dataset and are not live.
```

---

## Appendix D — Adapted 15 test queries

The original test list assumed dollar prices and Las Vegas. Adaptations are explicit.

| # | Query | Facet | Adaptation |
|---:|---|---|---|
| 1 | Find Italian restaurants with good reviews in Philadelphia. | Cuisine | Added a city from the corpus |
| 2 | Recommend restaurants under $$. | Budget | Bucket instead of dollars |
| 3 | Which restaurants have the best food quality according to reviews? | Review aspect | City-scoped or global |
| 4 | Find restaurants with positive ambience reviews. | Aspect | Unchanged |
| 5 | Recommend a restaurant for a family dinner. | Personalization | Use `good_for_kids` |
| 6 | Compare two restaurants based on reviews. | Comparison | Pick two IDs from the corpus |
| 7 | Which restaurants are praised for their desserts? | Review understanding | Use review and `good_for_meal_dessert` |
| 8 | Find restaurants with vegetarian options. | Dietary | Textual evidence only |
| 9 | Show restaurants similar to this uploaded food image. | Image | CLIP zero-shot → cuisine hint; no restaurant image match |
| 10 | Find restaurants with attractive interiors. | Visual | Text over photo summaries |
| 11 | Which restaurant offers the best value for money? | Ranking | Price bucket + rating + review evidence |
| 12 | Recommend a restaurant matching multiple preferences. | Multi-constraint | Explicit constraints |
| 13 | Give me the live availability of a restaurant. | Out of scope | Must refuse |
| 14 | Recommend a restaurant in Las Vegas. | Out of scope | City genuinely absent |
| 15 | Which restaurant has a rooftop pool? | Unsupported | Must say the dataset lacks this |

---

## Appendix E — Result table templates

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

## Appendix F — Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: retrieve context, then generate an answer grounded in it. |
| **Chunk** | A segment of a document small enough to embed and retrieve. |
| **Embedding** | A dense vector representation of text or an image. |
| **FAISS** | A library for efficient similarity search over dense vectors. |
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
