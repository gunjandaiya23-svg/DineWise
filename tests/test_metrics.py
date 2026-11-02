from __future__ import annotations

import pytest

from dinewise.eval.metrics import (
    evaluate_ranking,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    success_at_k,
)

GOLD = [
    {"business_id": "A", "grade": 2},
    {"business_id": "B", "grade": 1},
    {"business_id": "C", "grade": 2},
]


def test_success_and_precision() -> None:
    assert success_at_k(["X", "A"], {"A"}, k=5) == 1
    assert success_at_k(["X", "Y"], {"A"}, k=5) == 0
    assert precision_at_k(["A", "B", "X", "Y", "Z"], {"A", "B"}, k=5) == pytest.approx(0.4)


def test_evaluate_ranking_separates_strict_and_any() -> None:
    result = evaluate_ranking(["X", "B", "Y", "Z", "W"], GOLD, k=5)
    assert result["strict_success@k"] == 0.0
    assert result["any_success@k"] == 1.0
    assert result["strict_precision@k"] == 0.0
    assert result["any_precision@k"] == pytest.approx(0.2)
    assert result["strict_mrr"] == 0.0


def test_evaluate_ranking_strict_hit() -> None:
    result = evaluate_ranking(["A", "X", "B"], GOLD, k=5)
    assert result["strict_success@k"] == 1.0
    assert result["strict_mrr"] == 1.0


def test_reciprocal_rank_uses_first_relevant() -> None:
    assert reciprocal_rank(["X", "A", "Y"], {"A"}) == pytest.approx(0.5)
    assert reciprocal_rank(["X", "Y"], {"A"}) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    grades = {"A": 2, "B": 1, "C": 2}
    # DCG = (2^2-1)/log2(2) + (2^1-1)/log2(3) + 0
    dcg = 3 / 1 + 1 / 1.584962500721156
    # Ideal order: 2, 2, 1
    idcg = 3 / 1 + 3 / 1.584962500721156 + 1 / 2
    assert ndcg_at_k(["A", "B", "X"], grades, k=5) == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_ideal_order() -> None:
    grades = {"A": 2, "B": 1}
    assert ndcg_at_k(["A", "B"], grades, k=5) == pytest.approx(1.0)


def test_evaluate_ranking_shape() -> None:
    result = evaluate_ranking(["A", "X", "B"], GOLD, k=5)
    assert set(result) == {
        "strict_success@k",
        "strict_precision@k",
        "strict_mrr",
        "ndcg@k",
        "any_success@k",
        "any_precision@k",
    }
    assert result["strict_success@k"] == 1.0
    assert result["strict_mrr"] == 1.0


def test_mean_metrics_rounds() -> None:
    mean = mean_metrics(
        [{"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 0.0}]
    )
    assert mean == {"a": 0.75, "b": 0.25}
