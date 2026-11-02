"""Text-to-visual retrieval over photo summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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


@dataclass
class VisualRetriever:
    model: SentenceTransformer
    index: Any
    rows: pd.DataFrame
    prefix: str

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [self.prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, ids = self.index.search(vector, min(k, len(self.rows)))
        return [
            {"score": float(score), "row": self.rows.iloc[int(index)].to_dict()}
            for score, index in zip(scores[0], ids[0], strict=False)
            if int(index) >= 0
        ]


def build_visual_index(
    restaurants: pd.DataFrame, model: SentenceTransformer
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
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return VisualRetriever(model=model, index=index, rows=subset, prefix="")
