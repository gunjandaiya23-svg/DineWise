from __future__ import annotations

import pandas as pd

from dinewise.rag.bm25_index import build_bm25, tokenize


def _chunks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": ["c1", "c2", "c3"],
            "text": [
                "The vegetarian lasagna was excellent and fresh.",
                "Loud bar with great cocktails and late night music.",
                "Quiet romantic Italian restaurant with outdoor seating.",
            ],
        }
    )


def test_tokenize_lowercases_and_removes_stopwords() -> None:
    tokens = tokenize("The QUICK Brown Fox, with Cocktails!")
    assert tokens == ["quick", "brown", "fox", "cocktails"]


def test_bm25_finds_dish_terms() -> None:
    index = build_bm25(_chunks())
    hits = index.search("vegetarian lasagna", k=3)
    assert hits and hits[0]["chunk_id"] == "c1"


def test_bm25_finds_exact_phrases() -> None:
    index = build_bm25(_chunks())
    hits = index.search("outdoor seating romantic", k=3)
    assert hits and hits[0]["chunk_id"] == "c3"


def test_bm25_empty_query_returns_nothing() -> None:
    index = build_bm25(_chunks())
    assert index.search("the and of", k=3) == []


def test_bm25_meta_records_parameters() -> None:
    index = build_bm25(_chunks())
    assert index.meta["k1"] == 1.5
    assert index.meta["b"] == 0.75
    assert index.meta["chunks"] == 3
