"""Reciprocal Rank Fusion and chunk-to-restaurant collapse.

Fusion happens at the chunk level: each retriever contributes an ordered
list of chunk ids, and RRF assigns every chunk a fused score. Restaurant
candidates are then formed by collapsing chunk hits, keeping the best
chunk score per restaurant. Attribute filters are applied before fusion
in the pipeline, so filtered-out restaurants never influence ranks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ordered id lists. Returns (id, score) sorted by score desc."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def collapse_to_restaurants(
    fused: list[tuple[str, float]],
    chunks: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Map fused chunk hits to unique restaurants by best chunk score.

    Ties are broken by higher rating, then by review count, so that the
    order is fully deterministic for a fixed index.
    """
    table = chunks.set_index("chunk_id")
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for chunk_id, score in fused:
        if chunk_id not in table.index:
            continue
        row = table.loc[chunk_id]
        business_id = str(row["business_id"])
        if business_id in seen:
            if score > seen[business_id]["score"]:
                seen[business_id] = {"score": score, "chunk": row.to_dict()}
            continue
        seen[business_id] = {"score": score, "chunk": row.to_dict()}
        order.append(business_id)

    def sort_key(business_id: str) -> tuple:
        record = seen[business_id]
        chunk = record["chunk"]
        rating = chunk.get("stars")
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = -1.0
        reviews = chunk.get("review_count", 0) or 0
        return (-record["score"], -rating, -reviews, business_id)

    ranked = sorted(order, key=sort_key)[:top_k]
    return [seen[business_id] for business_id in ranked]
