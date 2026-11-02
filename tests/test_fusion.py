from __future__ import annotations

import pytest

from dinewise.rag.fusion import collapse_to_restaurants, rrf_fuse

RANK_A = ["c1", "c2", "c3"]
RANK_B = ["c2", "c4"]


def test_rrf_single_ranking_matches_formula() -> None:
    fused = dict(rrf_fuse([RANK_A], k=60))
    assert fused["c1"] == pytest.approx(1 / 61)
    assert fused["c2"] == pytest.approx(1 / 62)
    assert fused["c3"] == pytest.approx(1 / 63)


def test_rrf_rewards_appearance_in_multiple_rankings() -> None:
    fused = dict(rrf_fuse([RANK_A, RANK_B], k=60))
    assert fused["c2"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c2"] > fused["c1"] > fused["c3"]
    assert fused["c4"] == pytest.approx(1 / 62)


def test_rrf_ignores_duplicate_ids_within_one_ranking() -> None:
    fused = dict(rrf_fuse([["c1", "c1", "c2"]], k=60))
    assert fused["c1"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_empty_input_returns_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def _chunks() -> object:
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "business_id": "A",
                "restaurant_name": "Alpha",
                "stars": 4.5,
                "review_count": 100,
            },
            {
                "chunk_id": "c2",
                "business_id": "A",
                "restaurant_name": "Alpha",
                "stars": 4.5,
                "review_count": 100,
            },
            {
                "chunk_id": "c3",
                "business_id": "B",
                "restaurant_name": "Beta",
                "stars": 5.0,
                "review_count": 10,
            },
        ]
    )


def test_collapse_keeps_best_chunk_per_restaurant() -> None:
    fused = [("c2", 0.9), ("c1", 0.5), ("c3", 0.4)]
    collapsed = collapse_to_restaurants(fused, _chunks(), top_k=5)
    assert [hit["chunk"]["business_id"] for hit in collapsed] == ["A", "B"]
    assert collapsed[0]["score"] == pytest.approx(0.9)


def test_collapse_breaks_ties_by_rating() -> None:
    fused = [("c1", 0.5), ("c3", 0.5)]
    collapsed = collapse_to_restaurants(fused, _chunks(), top_k=5)
    assert collapsed[0]["chunk"]["business_id"] == "B"


def test_collapse_skips_unknown_chunk_ids() -> None:
    collapsed = collapse_to_restaurants([("zzz", 1.0)], _chunks(), top_k=5)
    assert collapsed == []
