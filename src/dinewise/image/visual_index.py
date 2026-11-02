"""Text-to-visual retrieval stored in Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..rag.vectorstore import QdrantStore, jsonable

VISUAL_COLLECTION = "visual"

VISUAL_PAYLOAD_FIELDS = [
    "business_id",
    "name",
    "city",
    "cuisine_primary",
    "price_range",
    "stars",
    "review_count",
    "venue_style",
    "indoor_color_tone",
    "target_audience",
    "special_features",
    "food_style",
    "drink_style",
    "photo_summary",
]


def visual_document(row: pd.Series) -> str:
    parts = [
        f"[RESTAURANT {row['business_id']}]",
        f"Name: {row.get('name')}",
        f"Venue style: {row.get('venue_style') or 'unknown'}",
        f"Colour tone: {row.get('indoor_color_tone') or 'unknown'}",
        f"Target audience: {row.get('target_audience') or 'unknown'}",
        f"Special features: {row.get('special_features') or 'unknown'}",
        f"Food style: {row.get('food_style') or 'unknown'}",
        f"Visual summary: {row.get('photo_summary') or ''}",
    ]
    return "\n".join(str(part) for part in parts)


def visual_payload(row: pd.Series) -> dict[str, Any]:
    return {field: jsonable(row.get(field)) for field in VISUAL_PAYLOAD_FIELDS}


@dataclass
class VisualRetriever:
    model: SentenceTransformer
    store: QdrantStore
    rows: pd.DataFrame
    prefix: str = ""

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [self.prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return [
            {"score": hit.score, "row": hit.payload}
            for hit in self.store.search(VISUAL_COLLECTION, vector, k)
        ]


def build_visual_index(
    restaurants: pd.DataFrame,
    model: SentenceTransformer,
    store: QdrantStore,
    recreate: bool = True,
) -> VisualRetriever:
    subset = restaurants[restaurants["photo_summary"].notna()].reset_index(drop=True)
    documents = [visual_document(row) for _, row in subset.iterrows()]
    embeddings = model.encode(
        documents,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    store.ensure_collection(VISUAL_COLLECTION, embeddings.shape[1], recreate=recreate)
    payloads = [visual_payload(row) for _, row in subset.iterrows()]
    store.upsert(
        VISUAL_COLLECTION,
        subset["business_id"].astype(str).tolist(),
        embeddings,
        payloads,
    )
    count = store.count(VISUAL_COLLECTION)
    if count != len(subset):
        raise RuntimeError(
            f"visual collection has {count} points, expected {len(subset)}"
        )
    return VisualRetriever(model=model, store=store, rows=subset)
