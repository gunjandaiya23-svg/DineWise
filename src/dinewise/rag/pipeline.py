"""Filtered retrieval: dense top-N, then restrict to constraint-satisfying IDs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constraints import Constraints, extract_constraints
from .filters import apply_filters
from .search import Retriever


def retrieve_filtered(
    retriever: Retriever,
    catalogue: pd.DataFrame,
    query: str,
    constraints: Constraints | None = None,
    pool: int = 30,
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve, filter, and rank. Returns hits plus diagnostics."""
    constraints = constraints or extract_constraints(
        query, known_cities=sorted(catalogue["city"].dropna().unique().tolist())
    )
    if constraints.unknown_city:
        return {
            "constraints": constraints,
            "refused": True,
            "reason": (
                f"'{constraints.unknown_city}' is not in the dataset, which covers "
                f"{', '.join(sorted(catalogue['city'].dropna().unique().tolist()))}."
            ),
            "filter_counts": {"start": len(catalogue)},
            "candidates_after_filter": 0,
            "skipped_by_filter": 0,
            "hits": [],
        }
    if constraints.unsupported_terms:
        return {
            "constraints": constraints,
            "refused": True,
            "reason": (
                "The dataset does not record information about "
                f"{', '.join(constraints.unsupported_terms)}, so this cannot be answered."
            ),
            "filter_counts": {"start": len(catalogue)},
            "candidates_after_filter": 0,
            "skipped_by_filter": 0,
            "hits": [],
        }
    allowed, filter_counts = apply_filters(catalogue, constraints)
    allowed_ids = set(allowed["business_id"])

    hits = retriever.search(query, k=pool)
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for hit in hits:
        business_id = str(hit["chunk"]["business_id"])
        if business_id in seen:
            continue
        if business_id not in allowed_ids:
            skipped += 1
            continue
        seen.add(business_id)
        ranked.append(hit)
        if len(ranked) >= k:
            break

    return {
        "constraints": constraints,
        "refused": False,
        "filter_counts": filter_counts,
        "candidates_after_filter": len(allowed),
        "skipped_by_filter": skipped,
        "hits": ranked,
    }
