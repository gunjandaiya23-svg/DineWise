"""Loading, cleaning, and assembling the restaurant corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import data_path, ensure_dirs
from ..logging import get_logger
from .attributes import canonical_attributes, parse_ok
from .cuisine import annotate_cuisines, load_taxonomy

LOG = get_logger(__name__)

FOOD_TOKENS = {
    "Restaurants",
    "Food",
    "Cafes",
    "Coffee & Tea",
    "Bakeries",
    "Desserts",
    "Bars",
    "Breweries",
    "Wineries",
    "Distilleries",
    "Gastropubs",
    "Food Trucks",
    "Caterers",
    "Specialty Food",
    "Ice Cream & Frozen Yogurt",
    "Juice Bars & Smoothies",
    "Tea Rooms",
    "Sandwiches",
    "Pizza",
    "Sushi Bars",
    "Barbeque",
    "Barbecue",
    "Smokehouse",
}

FOOD_TOKENS_LOWER = {token.lower() for token in FOOD_TOKENS}
FOOD_SUBSTRINGS = ("restaurant", "food", "cafe", "bakery", "pizza", "sushi")

# Venues that operate a cafe or catering service but are not restaurants.
# Excluded from the recommendation corpus and reported in the audit.
VENUE_TOKENS = {"Hotels", "Furniture Stores"}
PRICE_LABELS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def food_service_mask(categories: pd.Series) -> pd.Series:
    """Keep food-service businesses, matched on exact category tokens."""

    def is_food(value: Any) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        tokens = {token.strip().lower() for token in value.split(",")}
        if tokens & FOOD_TOKENS_LOWER:
            return True
        lowered = value.lower()
        return any(needle in lowered for needle in FOOD_SUBSTRINGS)

    return categories.map(is_food)


def venue_mask(categories: pd.Series) -> pd.Series:
    """True for hotels and furniture stores, even when they list a cafe."""

    def is_venue(value: Any) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        tokens = {token.strip() for token in value.split(",")}
        return bool(tokens & VENUE_TOKENS)

    return categories.map(is_venue)


def select_corpus(
    business: pd.DataFrame,
    cities: list[str] | None,
    limit: int | None = None,
    sort_by: str = "review_count",
    sort_desc: bool = True,
    require_food_service: bool = True,
) -> pd.DataFrame:
    """Apply city, food-service, and size filters in a reproducible order."""
    frame = business.copy()
    if cities:
        frame = frame[frame["city"].isin(cities)]
    if require_food_service:
        frame = frame[food_service_mask(frame["categories"])]
        frame = frame[~venue_mask(frame["categories"])]
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=not sort_desc, kind="mergesort")
    if limit:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def _photo_summary_path(raw_dir: Path) -> Path:
    for name in ("photo_summary.csv", "photo_summay.csv"):
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("photo summary CSV not found in data/raw")


def load_raw(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    raw_dir = data_path(cfg, "raw_dir")
    return {
        "business": read_csv(raw_dir / "business.csv"),
        "reviews": read_csv(raw_dir / "review.csv"),
        "review_summary": read_csv(raw_dir / "review_summary.csv"),
        "photo_summary": read_csv(_photo_summary_path(raw_dir)),
        "photos": read_csv(raw_dir / "photo.csv"),
    }


def _coerce_attribute_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Give boolean and integer attribute columns nullable dtypes for parquet."""
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        if values.map(lambda value: isinstance(value, bool)).all():
            frame[column] = frame[column].astype("boolean")
        elif values.map(
            lambda value: isinstance(value, int) and not isinstance(value, bool)
        ).all():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def _sanitize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast genuinely mixed-type columns to nullable strings for parquet."""
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        if len({type(value) for value in values}) > 1:
            frame[column] = (
                frame[column]
                .map(lambda value: None if value is None else str(value))
                .astype("string")
            )
    return frame


def build_reviews(review: pd.DataFrame, business_ids: set[str], min_chars: int = 20) -> pd.DataFrame:
    frame = review[review["business_id"].isin(business_ids)].copy()
    frame = frame.drop_duplicates(subset=["review_id"])
    frame = frame[frame["text"].notna()]
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() >= min_chars]
    frame["char_len"] = frame["text"].str.len()
    frame["word_count"] = frame["text"].str.split().str.len()
    return frame.reset_index(drop=True)


def build_restaurant_table(
    business: pd.DataFrame,
    review_summary: pd.DataFrame,
    photo_summary: pd.DataFrame,
    photos: pd.DataFrame,
    taxonomy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    taxonomy = taxonomy or load_taxonomy()
    frame = business.copy().reset_index(drop=True)
    frame["attributes_raw"] = frame["attributes"]
    frame = frame.rename(columns={"hours": "hours_raw"})

    attributes = frame["attributes_raw"].map(canonical_attributes)
    attribute_frame = pd.json_normalize(attributes)
    frame = pd.concat([frame.drop(columns=["attributes"]), attribute_frame], axis=1)
    for column in ("price_range", "restaurants_price_range2"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame = _coerce_attribute_columns(frame)

    summaries = review_summary.rename(
        columns={
            "summary": "review_summary",
            "keywords": "review_keywords",
            "themes": "review_themes",
        }
    )
    frame = frame.merge(summaries, on="business_id", how="left", validate="many_to_one")

    visuals = photo_summary.rename(
        columns={"summary": "photo_summary", "keywords": "photo_keywords"}
    )
    frame = frame.merge(visuals, on="business_id", how="left", validate="many_to_one")

    photo_counts = photos.groupby("business_id").size().rename("n_photos")
    photo_labels = (
        photos.pivot_table(
            index="business_id",
            columns="label",
            values="photo_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda label: f"n_photos_{label}")
        .astype(int)
    )
    frame = frame.merge(photo_counts, left_on="business_id", right_index=True, how="left")
    frame = frame.merge(photo_labels, left_on="business_id", right_index=True, how="left")
    frame["n_photos"] = frame["n_photos"].fillna(0).astype(int)
    for column in [c for c in frame.columns if c.startswith("n_photos_")]:
        frame[column] = frame[column].fillna(0).astype(int)

    annotated = frame["categories"].map(lambda value: annotate_cuisines(value, taxonomy))
    frame["cuisine_tags"] = annotated.map(lambda pair: pair[0])
    frame["cuisine_primary"] = annotated.map(lambda pair: pair[1])
    frame["price_label"] = frame["price_range"].map(PRICE_LABELS)
    frame["has_review_summary"] = frame["review_summary"].notna()
    frame["has_photo_summary"] = frame["photo_summary"].notna()

    return _sanitize_object_columns(frame)


def attach_review_stats(restaurants: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    stats = reviews.groupby("business_id").agg(
        n_reviews_used=("review_id", "count"),
        mean_review_stars=("stars", "mean"),
    )
    five_star = reviews[reviews["stars"] == 5].groupby("business_id").size()
    stats["star_5_ratio"] = (five_star / stats["n_reviews_used"]).fillna(0.0)
    return restaurants.merge(stats, left_on="business_id", right_index=True, how="left")


def prepare_corpus(
    cfg: dict[str, Any],
    cities: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the whole P1 data step and return the tables plus an audit dict."""
    ensure_dirs(cfg)
    raw = load_raw(cfg)
    business = raw["business"]
    raw_counts = {
        "businesses": int(len(business)),
        "reviews": int(len(raw["reviews"])),
        "review_summaries": int(len(raw["review_summary"])),
        "photo_summaries": int(len(raw["photo_summary"])),
        "photos": int(len(raw["photos"])),
    }

    city_filter = cities if cities is not None else cfg["corpus"]["cities"]
    city_business = (
        business[business["city"].isin(city_filter)] if city_filter else business.copy()
    )
    after_city = len(city_business)
    food_business = city_business[food_service_mask(city_business["categories"])]
    after_food = len(food_business)
    venue_dropped = int(venue_mask(food_business["categories"]).sum())

    selected = select_corpus(
        city_business,
        cities=None,
        limit=limit if limit is not None else cfg["corpus"].get("limit"),
        sort_by=cfg["corpus"].get("sort_by", "review_count"),
        sort_desc=cfg["corpus"].get("sort_desc", True),
        require_food_service=True,
    )
    business_ids = set(selected["business_id"])

    restaurants = build_restaurant_table(
        selected,
        raw["review_summary"],
        raw["photo_summary"],
        raw["photos"],
    )
    reviews = build_reviews(raw["reviews"], business_ids)
    restaurants = attach_review_stats(restaurants, reviews)

    parse_rate = (
        float(selected["attributes"].map(parse_ok).mean()) if len(selected) else 0.0
    )
    cuisine_covered = float((restaurants["cuisine_primary"] != "Unspecified").mean())
    audit = {
        "raw": raw_counts,
        "selection": {
            "cities": city_filter,
            "limit": limit if limit is not None else cfg["corpus"].get("limit"),
            "after_city_filter": int(after_city),
            "after_food_service_filter": int(after_food),
            "venue_excluded": venue_dropped,
            "selected": int(len(restaurants)),
        },
        "corpus": {
            "restaurants": int(len(restaurants)),
            "reviews": int(len(reviews)),
            "with_review_summary": int(restaurants["has_review_summary"].sum()),
            "with_photo_summary": int(restaurants["has_photo_summary"].sum()),
            "with_price": int(restaurants["price_range"].notna().sum()),
            "cuisine_covered": int((restaurants["cuisine_primary"] != "Unspecified").sum()),
        },
        "quality": {
            "duplicate_business_ids": int(business["business_id"].duplicated().sum()),
            "duplicate_review_ids": int(raw["reviews"]["review_id"].duplicated().sum()),
            "attribute_parse_rate": round(parse_rate, 4),
            "missing_price_pct": round(
                100 * float(restaurants["price_range"].isna().mean()), 2
            ),
            "cuisine_coverage_pct": round(100 * cuisine_covered, 2),
            "review_join_coverage_pct": round(
                100 * float(restaurants["business_id"].isin(reviews["business_id"]).mean()), 2
            ),
        },
    }

    restaurants.to_parquet(
        data_path(cfg, "processed_dir", "restaurants.parquet"), index=False
    )
    reviews.to_parquet(data_path(cfg, "processed_dir", "reviews.parquet"), index=False)
    (data_path(cfg, "processed_dir", "corpus.json")).write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (data_path(cfg, "audit_dir") / "data_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    LOG.info(
        "corpus ready: %d restaurants, %d reviews", len(restaurants), len(reviews)
    )
    return {"restaurants": restaurants, "reviews": reviews, "audit": audit}
