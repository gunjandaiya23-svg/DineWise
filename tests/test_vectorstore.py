from __future__ import annotations

import numpy as np
import pytest

from dinewise.rag.constraints import Constraints
from dinewise.rag.vectorstore import QdrantStore, to_qdrant_filter


@pytest.fixture()
def store(tmp_path):
    instance = QdrantStore(
        path=tmp_path / "qdrant", prefix="test", exact=True
    )
    instance.ensure_collection("chunks", 3)
    vectors = np.array(
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype="float32"
    )
    payloads = [
        {
            "chunk_id": "a::1",
            "business_id": "a",
            "city": "Philadelphia",
            "cuisine_tags": ["Italian"],
            "price_range": 2,
            "text": "italian pasta",
            "stars": 4.5,
        },
        {
            "chunk_id": "b::1",
            "business_id": "b",
            "city": "Philadelphia",
            "cuisine_tags": ["Mexican"],
            "price_range": 1,
            "text": "mexican tacos",
            "stars": 4.0,
        },
        {
            "chunk_id": "c::1",
            "business_id": "c",
            "city": "Tucson",
            "cuisine_tags": ["Italian"],
            "price_range": 3,
            "text": "italian wine",
            "stars": 5.0,
        },
    ]
    instance.upsert("chunks", ["a::1", "b::1", "c::1"], vectors, payloads)
    yield instance
    instance.client.close()


def test_count_and_scroll(store: QdrantStore) -> None:
    assert store.count("chunks") == 3
    records = store.scroll("chunks")
    assert {record["chunk_id"] for record in records} == {"a::1", "b::1", "c::1"}


def test_search_returns_scores_and_payloads(store: QdrantStore) -> None:
    hits = store.search("chunks", np.array([1, 0, 0], dtype="float32"), 2)
    assert hits[0].chunk_id == "a::1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].payload["city"] == "Philadelphia"


def test_server_side_filters(store: QdrantStore) -> None:
    query_filter = to_qdrant_filter(
        Constraints(city="Philadelphia", cuisines=["Italian"], max_price_bucket=2)
    )
    hits = store.search(
        "chunks", np.array([1, 0, 0], dtype="float32"), 5, query_filter
    )
    assert [hit.chunk_id for hit in hits] == ["a::1"]


def test_city_filter_excludes_other_cities(store: QdrantStore) -> None:
    query_filter = to_qdrant_filter(Constraints(city="Tucson"))
    hits = store.search(
        "chunks", np.array([1, 0, 0], dtype="float32"), 5, query_filter
    )
    assert [hit.chunk_id for hit in hits] == ["c::1"]


def test_recreate_replaces_points(store: QdrantStore) -> None:
    store.ensure_collection("chunks", 3, recreate=True)
    assert store.count("chunks") == 0


def test_delete_collection(store: QdrantStore) -> None:
    store.delete("chunks")
    assert store.collection("chunks") not in store.collections()
