"""Apply hard constraints against the restaurant catalogue."""

from __future__ import annotations

import pandas as pd

from .constraints import Constraints


def _tri_true(series: pd.Series) -> pd.Series:
    """True only where a nullable-boolean value is explicitly True."""
    return series.fillna(False).astype(bool)


def apply_filters(
    restaurants: pd.DataFrame, constraints: Constraints
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the filtered catalogue plus per-filter survival counts."""
    counts = {"start": len(restaurants)}
    frame = restaurants

    if constraints.city:
        frame = frame[frame["city"].str.lower() == constraints.city.lower()]
        counts["city"] = len(frame)

    if constraints.cuisines:
        wanted = set(constraints.cuisines)

        def matches(tags) -> bool:
            if isinstance(tags, (list, tuple)):
                return bool(wanted & set(tags))
            return str(tags) in wanted

        primary = frame["cuisine_primary"].isin(constraints.cuisines)
        frame = frame[primary | frame["cuisine_tags"].map(matches)]
        counts["cuisine"] = len(frame)

    if constraints.max_price_bucket is not None:
        known = frame[frame["price_range"].notna()]
        frame = known[known["price_range"] <= constraints.max_price_bucket]
        counts["price_max"] = len(frame)

    if constraints.min_price_bucket is not None:
        known = frame[frame["price_range"].notna()]
        frame = known[known["price_range"] >= constraints.min_price_bucket]
        counts["price_min"] = len(frame)

    if constraints.min_rating is not None:
        frame = frame[frame["stars"] >= constraints.min_rating]
        counts["rating"] = len(frame)

    for attribute in constraints.required_attributes:
        if attribute == "noise_level_quiet":
            frame = frame[frame["noise_level"] == "quiet"]
        elif attribute == "noise_level_loud":
            frame = frame[frame["noise_level"] == "loud"]
        elif attribute in frame.columns:
            frame = frame[_tri_true(frame[attribute])]
        counts[f"attr_{attribute}"] = len(frame)

    return frame.reset_index(drop=True), counts
