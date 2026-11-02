"""Qdrant-backed vector storage for chunks and visual documents.

Qdrant is the source of truth for vectors and payloads. A small parquet
cache under ``data/indexes/chunks`` is kept only so BM25 can build its
lexical corpus without scrolling the database on every run.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    SearchParams,
    VectorParams,
)

from ..config import data_path
from ..logging import get_logger
from .constraints import Constraints

LOG = get_logger(__name__)

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

CHUNK_PAYLOAD_FIELDS = [
    "chunk_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "cuisine_primary",
    "cuisine_tags",
    "price_range",
    "stars",
    "review_count",
    "document_type",
    "section",
    "source_file",
    "page",
    "token_count",
    "char_count",
    "chunk_config",
    "text",
    "outdoor_seating",
    "noise_level",
    "ambience_romantic",
    "ambience_intimate",
    "ambience_classy",
    "ambience_trendy",
    "ambience_casual",
    "good_for_kids",
    "good_for_meal_breakfast",
    "good_for_meal_brunch",
    "good_for_meal_lunch",
    "good_for_meal_dinner",
    "good_for_meal_dessert",
    "good_for_meal_latenight",
    "reservations",
    "byob",
    "delivery",
    "takeout",
    "wheelchair_accessible",
    "dogs_allowed",
]


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and missing values to Qdrant-safe types."""
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def chunk_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    record = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {field: jsonable(record.get(field)) for field in CHUNK_PAYLOAD_FIELDS}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]

    @property
    def chunk_id(self) -> str:
        return str(self.payload.get("chunk_id") or self.id)


class QdrantStore:
    """Thin wrapper around a Qdrant collection set."""

    def __init__(
        self,
        url: str | None = "http://localhost:6333",
        prefix: str = "dinewise",
        path: str | Path | None = None,
        exact: bool = True,
        hnsw_ef: int = 128,
        batch_size: int = 256,
    ):
        self.prefix = prefix
        self.exact = exact
        self.hnsw_ef = hnsw_ef
        self.batch_size = batch_size
        if path is not None:
            self.client = QdrantClient(path=str(path))
            self.url = f"local:{path}"
        else:
            self.client = QdrantClient(url=url, timeout=30)
            self.url = url or ""

    def collection(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    def ensure_collection(
        self, name: str, dim: int, recreate: bool = False
    ) -> str:
        full = self.collection(name)
        if recreate and self.client.collection_exists(full):
            LOG.info("recreating collection %s", full)
            self.client.delete_collection(full)

        if not self.client.collection_exists(full):
            self.client.create_collection(
                collection_name=full,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            LOG.info("created collection %s (%d dims)", full, dim)

        for field, schema in (
            ("city", PayloadSchemaType.KEYWORD),
            ("cuisine_primary", PayloadSchemaType.KEYWORD),
            ("cuisine_tags", PayloadSchemaType.KEYWORD),
            ("document_type", PayloadSchemaType.KEYWORD),
            ("business_id", PayloadSchemaType.KEYWORD),
            ("price_range", PayloadSchemaType.INTEGER),
            ("stars", PayloadSchemaType.FLOAT),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=full, field_name=field, field_schema=schema
                )
            except Exception:  # noqa: BLE001 - index may already exist
                pass
        return full

    def upsert(
        self,
        name: str,
        ids: Iterable[str],
        vectors: np.ndarray,
        payloads: Iterable[dict[str, Any]],
    ) -> int:
        full = self.collection(name)
        ids = list(ids)
        payloads = list(payloads)
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors, and payloads must have the same length")

        points = [
            PointStruct(id=point_id(item_id), vector=vector.tolist(), payload=payload)
            for item_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        for start in range(0, len(points), self.batch_size):
            self.client.upsert(
                collection_name=full,
                points=points[start : start + self.batch_size],
                wait=True,
            )
        return len(points)

    def search(
        self,
        name: str,
        vector: np.ndarray,
        k: int = 30,
        query_filter: Filter | None = None,
    ) -> list[VectorHit]:
        query_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        response = self.client.query_points(
            collection_name=self.collection(name),
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                exact=self.exact,
                hnsw_ef=None if self.exact else self.hnsw_ef,
            ),
        )
        return [
            VectorHit(id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

    def scroll(self, name: str) -> list[dict[str, Any]]:
        full = self.collection(name)
        records: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=full,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(point.payload or {} for point in points)
            if offset is None:
                break
        return records

    def count(self, name: str) -> int:
        return int(self.client.count(self.collection(name), exact=True).count)

    def collections(self) -> list[str]:
        return [item.name for item in self.client.get_collections().collections]

    def delete(self, name: str) -> None:
        full = self.collection(name)
        if self.client.collection_exists(full):
            self.client.delete_collection(full)


def store_from_config(cfg: dict[str, Any], path: str | Path | None = None) -> QdrantStore:
    settings = cfg["vector_store"]
    return QdrantStore(
        url=settings.get("url", "http://localhost:6333"),
        prefix=settings.get("collection_prefix", "dinewise"),
        path=path,
        exact=bool(settings.get("exact", True)),
        hnsw_ef=int(settings.get("hnsw_ef", 128)),
        batch_size=int(settings.get("batch_size", 256)),
    )


def to_qdrant_filter(constraints: Constraints) -> Filter | None:
    must: list[FieldCondition] = []
    if constraints.city:
        must.append(
            FieldCondition(key="city", match=MatchValue(value=constraints.city))
        )
    if constraints.cuisines:
        must.append(
            FieldCondition(
                key="cuisine_tags", match=MatchAny(any=list(constraints.cuisines))
            )
        )
    if constraints.max_price_bucket is not None:
        must.append(
            FieldCondition(
                key="price_range", range=Range(lte=constraints.max_price_bucket)
            )
        )
    if constraints.min_price_bucket is not None:
        must.append(
            FieldCondition(
                key="price_range", range=Range(gte=constraints.min_price_bucket)
            )
        )
    if constraints.min_rating is not None:
        must.append(
            FieldCondition(key="stars", range=Range(gte=constraints.min_rating))
        )
    for attribute in constraints.required_attributes:
        if attribute == "noise_level_quiet":
            must.append(
                FieldCondition(key="noise_level", match=MatchValue(value="quiet"))
            )
        elif attribute == "noise_level_loud":
            must.append(
                FieldCondition(key="noise_level", match=MatchValue(value="loud"))
            )
        else:
            must.append(
                FieldCondition(key=attribute, match=MatchValue(value=True))
            )
    return Filter(must=must) if must else None


def ingest_chunks(
    store: QdrantStore,
    collection: str,
    chunks: pd.DataFrame,
    vectors: np.ndarray,
    recreate: bool = False,
) -> int:
    store.ensure_collection(collection, vectors.shape[1], recreate=recreate)
    payloads = [chunk_payload(row) for _, row in chunks.iterrows()]
    written = store.upsert(collection, chunks["chunk_id"].tolist(), vectors, payloads)
    count = store.count(collection)
    if count != len(chunks):
        raise RuntimeError(
            f"collection {collection} has {count} points, expected {len(chunks)}"
        )
    return written


def chunks_cache_path(cfg: dict[str, Any], name: str) -> Path:
    return data_path(cfg, "index_dir", "chunks", f"{name}.parquet")


def write_chunks_cache(cfg: dict[str, Any], name: str, chunks: pd.DataFrame) -> Path:
    path = chunks_cache_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(path, index=False)
    return path


def load_chunks(
    cfg: dict[str, Any], name: str, store: QdrantStore | None = None
) -> pd.DataFrame:
    """Load chunk metadata from the cache, falling back to Qdrant scroll."""
    path = chunks_cache_path(cfg, name)
    if path.exists():
        return pd.read_parquet(path)
    store = store or store_from_config(cfg)
    records = store.scroll(f"chunks_{name}")
    if not records:
        raise FileNotFoundError(
            f"no chunk cache at {path} and collection chunks_{name} is empty"
        )
    return pd.DataFrame.from_records(records)
