"""Retrieval metrics at the restaurant level.

Graded relevance from the silver rubric:

* grade 2 = strictly relevant (constraint-satisfying and in the rubric's top 10)
* grade 1 = relevant (satisfies the stated constraints)
* grade 0 = not in the candidate pool

Strict metrics use grade 2 only and therefore measure ranking quality.
``*_any`` metrics use grade >= 1 and therefore measure constraint compliance.
"""

from __future__ import annotations

import math
from typing import Any


def success_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> int:
    return int(bool(set(ranked[:k]) & relevant))


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    return len(set(ranked[:k]) & relevant) / k if k else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, business_id in enumerate(ranked, start=1):
        if business_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int = 5) -> float:
    def gain(grade: int) -> float:
        return 2**grade - 1

    dcg = 0.0
    for rank, business_id in enumerate(ranked[:k], start=1):
        dcg += gain(grades.get(business_id, 0)) / math.log2(rank + 1)

    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(
    ranked: list[str], gold: list[dict[str, Any]], k: int = 5
) -> dict[str, float]:
    strict = {item["business_id"] for item in gold if int(item["grade"]) == 2}
    relevant = {item["business_id"] for item in gold if int(item["grade"]) >= 1}
    grades = {item["business_id"]: int(item["grade"]) for item in gold}
    return {
        "strict_success@k": float(success_at_k(ranked, strict, k)),
        "strict_precision@k": precision_at_k(ranked, strict, k),
        "strict_mrr": reciprocal_rank(ranked, strict),
        "ndcg@k": ndcg_at_k(ranked, grades, k),
        "any_success@k": float(success_at_k(ranked, relevant, k)),
        "any_precision@k": precision_at_k(ranked, relevant, k),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(sum(row[key] for row in rows) / len(rows), 4) for key in keys}
