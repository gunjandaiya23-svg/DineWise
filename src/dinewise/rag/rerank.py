"""Cross-encoder reranking and an explicit quality prior.

The reranker orders candidates by query-document relevance. The quality
prior then blends in dataset rating and review volume, because a relevance
model alone does not know which restaurant is well reviewed. The blend is
a fixed, reported policy, not a learned model.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str = "cuda"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=16,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def quality_prior(
    hits: list[dict[str, Any]],
    catalogue: pd.DataFrame,
    relevance_weight: float = 0.7,
    rating_weight: float = 0.2,
    review_weight: float = 0.1,
) -> list[dict[str, Any]]:
    """Blend normalized relevance with rating and log review volume."""
    if not hits:
        return []
    table = catalogue.set_index("business_id")
    relevance = normalize([hit["score"] for hit in hits])
    blended = []

    for hit, rel in zip(hits, relevance, strict=False):
        business_id = str(hit["chunk"]["business_id"])
        rating = 0.0
        reviews = 0.0
        if business_id in table.index:
            row = table.loc[business_id]
            rating = float(row.get("stars") or 0.0) / 5.0
            reviews = math.log1p(float(row.get("review_count") or 0)) / math.log1p(5000)
        score = (
            relevance_weight * rel
            + rating_weight * rating
            + review_weight * min(reviews, 1.0)
        )
        blended.append({**hit, "score": score, "relevance_score": hit["score"]})

    blended.sort(key=lambda hit: (-hit["score"], str(hit["chunk"]["business_id"])))
    return blended
