"""Silver-standard relevance labels.

The project plan calls for manually judged relevance. Human judgments are
not available yet, so this module builds a *silver* gold set with a
transparent rubric and marks every record ``label_source = silver_rubric_v1``.
The output CSV is the review sheet that converts it to a manual standard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..rag.constraints import Constraints, extract_constraints
from ..rag.filters import apply_filters

FACET_TERMS: dict[str, list[str]] = {
    "review_aspect": ["food", "quality", "delicious", "tasty", "flavor", "fresh", "excellent"],
    "ambience": ["ambience", "atmosphere", "romantic", "cozy", "intimate", "decor", "vibe"],
    "dessert": ["dessert", "cake", "sweet", "pastry", "ice cream", "chocolate", "bakery"],
    "dietary": ["vegetarian", "vegan", "plant-based", "gluten-free", "veggie"],
    "visual": ["interior", "decor", "atmosphere", "design", "bright", "cozy", "beautiful"],
    "image": ["pizza", "italian", "pasta"],
    "ranking": [],
    "cuisine": [],
    "budget": [],
    "personalization": [],
    "comparison": [],
}

MAX_GOLD = 10


def _text(row: pd.Series, *columns: str) -> str:
    return " ".join(str(row.get(column) or "") for column in columns).lower()


def facet_score(row: pd.Series, facet: str) -> tuple[int, str]:
    """Return (0/1/2, reason) for one candidate under one facet."""
    summary = _text(row, "review_summary", "review_keywords", "review_themes")
    visual = _text(row, "photo_summary", "venue_style", "special_features")

    if facet in {"review_aspect", "ambience", "dessert", "dietary", "visual", "image"}:
        terms = FACET_TERMS[facet]
        hits = [term for term in terms if term in summary or term in visual]
        if hits:
            return 2, f"evidence: {', '.join(hits[:4])}"
        if facet == "ambience" and row.get("ambience_romantic") is True:
            return 2, "evidence: ambience_romantic"
        if facet == "dessert" and row.get("good_for_meal_dessert") is True:
            return 2, "evidence: good_for_meal_dessert"
        if facet == "dietary" and row.get("cuisine_primary") == "Vegetarian/Vegan":
            return 2, "evidence: vegetarian cuisine"
        if facet == "visual" and visual.strip():
            return 2, "evidence: visual summary present"
        return 1, "constraints only"

    if facet == "personalization":
        if row.get("good_for_kids") is True:
            return 2, "evidence: good_for_kids"
        return 1, "constraints only"

    if facet == "ranking":
        if row.get("price_range") is not None and int(row["price_range"]) <= 2 and float(row["stars"]) >= 4.0:
            return 2, "evidence: cheap and highly rated"
        return 1, "constraints only"

    return 1, "constraints only"


def _sort_key(row: pd.Series, facet: str) -> tuple:
    score, _ = facet_score(row, facet)
    rating = float(row.get("stars") or 0)
    reviews = int(row.get("review_count") or 0)
    price = row.get("price_range")
    cheap_bonus = 1 if (price is not None and int(price) <= 2) else 0
    return (-score, -rating, -reviews, -cheap_bonus, str(row.get("business_id")))


def build_goldset(
    restaurants: pd.DataFrame,
    queries: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    known_cities = sorted(restaurants["city"].dropna().unique().tolist())
    records: list[dict[str, Any]] = []

    for query_id, facet, query in queries:
        constraints = extract_constraints(query, known_cities=known_cities)
        pool, _ = apply_filters(restaurants, constraints)
        if pool.empty:
            records.append(
                {
                    "query_id": query_id,
                    "facet": facet,
                    "query": query,
                    "constraints": _constraint_dict(constraints),
                    "gold": [],
                    "label_source": "silver_rubric_v1",
                    "needs_manual_review": True,
                }
            )
            continue

        rows = [row for _, row in pool.iterrows()]
        rows.sort(key=lambda row: _sort_key(row, facet))
        top_ids = {str(row["business_id"]) for row in rows[:MAX_GOLD]}

        # All constraint-satisfying restaurants are relevant (grade 1); the
        # rubric's top-10 are strictly relevant (grade 2). This separates
        # constraint compliance from ranking quality.
        gold = []
        for row in rows:
            business_id = str(row["business_id"])
            strict = business_id in top_ids
            score, reason = facet_score(row, facet)
            gold.append(
                {
                    "business_id": business_id,
                    "name": str(row["name"]),
                    "grade": 2 if strict else 1,
                    "reason": reason if strict else "constraint-satisfying",
                }
            )
        records.append(
            {
                "query_id": query_id,
                "facet": facet,
                "query": query,
                "constraints": _constraint_dict(constraints),
                "candidates_in_pool": int(len(pool)),
                "gold": gold,
                "label_source": "silver_rubric_v1",
                "needs_manual_review": True,
            }
        )
    return records


def _constraint_dict(constraints: Constraints) -> dict[str, Any]:
    return {
        "city": constraints.city,
        "cuisines": constraints.cuisines,
        "max_price_bucket": constraints.max_price_bucket,
        "min_rating": constraints.min_rating,
        "required_attributes": constraints.required_attributes,
        "dietary": constraints.dietary,
    }


def write_goldset(records: list[dict[str, Any]], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / "goldset_v1.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    rows = []
    for record in records:
        strict = [item for item in record["gold"] if item["grade"] == 2]
        relaxed = [item for item in record["gold"] if item["grade"] == 1][:10]
        for item in strict + relaxed:
            rows.append(
                {
                    "query_id": record["query_id"],
                    "facet": record["facet"],
                    "query": record["query"],
                    "business_id": item["business_id"],
                    "name": item["name"],
                    "silver_grade": item["grade"],
                    "rubric_reason": item["reason"],
                    "manual_grade": "",
                    "reviewer_note": "",
                }
            )
    review = directory / "review_sheet.csv"
    pd.DataFrame(rows).to_csv(review, index=False)
    return jsonl, review
