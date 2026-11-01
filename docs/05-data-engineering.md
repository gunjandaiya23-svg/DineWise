# Data engineering

> Part of the DineWise project plan. Parsing, cleaning, taxonomy, price rules, derived tables, and the audit.
> See [the plan index](README.md) for the full reading order.

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
