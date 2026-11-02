"""Retriever backed by Qdrant, with BGE query embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer

from .vectorstore import QdrantStore, load_chunks, store_from_config


@dataclass
class Retriever:
    model: SentenceTransformer
    store: QdrantStore
    chunks: pd.DataFrame
    prefix: str
    collection: str

    def search(
        self,
        query: str,
        k: int = 30,
        query_filter: Filter | None = None,
    ) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [self.prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        hits = self.store.search(self.collection, vector, k, query_filter)
        table = self.chunks.set_index("chunk_id")
        results: list[dict[str, Any]] = []
        for hit in hits:
            chunk_id = hit.chunk_id
            if chunk_id not in table.index:
                continue
            row = table.loc[chunk_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            chunk = row.to_dict()
            chunk["chunk_id"] = chunk_id
            results.append({"score": hit.score, "chunk": chunk})
        return results

    def unique_restaurants(
        self, query: str, k: int = 5, pool: int = 30
    ) -> list[dict[str, Any]]:
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
    store = store_from_config(cfg)
    collection = f"chunks_{name}"
    if store.count(collection) == 0:
        raise FileNotFoundError(
            f"Qdrant collection {store.collection(collection)} is empty; "
            "run scripts/05_build_index.py"
        )
    chunks = load_chunks(cfg, name, store)
    model = SentenceTransformer(
        cfg["embedding"]["model"], device=cfg["embedding"]["device"]
    )
    model.max_seq_length = int(cfg["embedding"]["max_seq_length"])
    return Retriever(
        model=model,
        store=store,
        chunks=chunks,
        prefix=cfg["embedding"]["query_prefix"],
        collection=collection,
    )
