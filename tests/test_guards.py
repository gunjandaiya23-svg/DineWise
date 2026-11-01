from __future__ import annotations

import pytest

from dinewise.llm.guards import REFUSAL, check_scope


@pytest.mark.parametrize(
    "query",
    [
        "Is &pizza - Walnut available tonight at 8pm?",
        "Can I book a table for four?",
        "What is the wait time right now?",
        "Do you have availability this evening?",
        "What is the current price of the tasting menu?",
        "Is it open now?",
    ],
)
def test_live_queries_are_rejected(query: str) -> None:
    result = check_scope(query)
    assert not result.allowed
    assert result.reason
    assert "no live availability" in REFUSAL


@pytest.mark.parametrize(
    "query",
    [
        "Find Italian restaurants with vegetarian options in Philadelphia.",
        "Which restaurants are praised for their desserts?",
        "Recommend a cheap Mexican place for dinner.",
        "Find a restaurant for tonight.",
        "Compare Restaurant A and Restaurant B.",
    ],
)
def test_recommendation_queries_are_allowed(query: str) -> None:
    assert check_scope(query).allowed
