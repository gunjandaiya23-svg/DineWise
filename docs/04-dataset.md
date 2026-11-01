# Dataset and study corpus

> Part of the DineWise project plan. Verified dataset facts, schemas, the study corpus, and its limits.
> See [the plan index](README.md) for the full reading order.

---

## 2.2 Dataset files

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

---

## 2.3 Dataset profile (measured)

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

---

## 2.4 Study corpus (measured)

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

---

## 2.5 Attribute availability in the study corpus

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

---

## 2.6 What the dataset cannot support

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
