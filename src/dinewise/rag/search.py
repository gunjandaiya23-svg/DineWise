"""Retriever wrapper around the BGE embeddings and FAISS index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config import data_path


@dataclass
class Retriever:
    model: SentenceTransformer
    index: Any
    chunks: pd.DataFrame
    prefix: str

    def search(self, query: str, k: int = 30) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [self.prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [
            {"score": float(score), "chunk": self.chunks.iloc[int(index)].to_dict()}
            for score, index in zip(scores[0], ids[0], strict=False)
            if int(index) >= 0
        ]

    def unique_restaurants(self, query: str, k: int = 5, pool: int = 30) -> list[dict[str, Any]]:
        hits = self.search(query, k=pool)
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for hit in hits:
            business_id = str(hit["chunk"]["business_id"])
            if business_id in seen:
                continue
            seen.add(business_id)
            selected.append(hit)
            if len(selected) >= k:
                break
        return selected


def load_retriever(cfg: dict[str, Any], name: str | None = None) -> Retriever:
    name = name or cfg["chunking"]["active"]
    index_dir = data_path(cfg, "index_dir", "faiss", name)
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.parquet"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"missing index in {index_dir}")

    model = SentenceTransformer(
        cfg["embedding"]["model"], device=cfg["embedding"]["device"]
    )
    model.max_seq_length = int(cfg["embedding"]["max_seq_length"])
    return Retriever(
        model=model,
        index=faiss.read_index(str(index_path)),
        chunks=pd.read_parquet(chunks_path),
        prefix=cfg["embedding"]["query_prefix"],
    )
