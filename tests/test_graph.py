from __future__ import annotations

from dinewise.graph.nodes import (
    route_after_classify,
    route_after_retrieve,
    route_after_review,
)


def test_route_out_of_scope_to_fallback() -> None:
    assert route_after_classify({"out_of_scope": True}) == "fallback"
    assert route_after_classify({"out_of_scope": False}) == "retrieve"


def test_route_empty_retrieval_to_fallback() -> None:
    assert route_after_retrieve({"hits": []}) == "fallback"
    assert route_after_retrieve({"refused": True, "hits": [{"x": 1}]}) == "fallback"
    assert route_after_retrieve({"hits": [{"x": 1}]}) == "generate"


def test_route_review_to_end_retry_or_fallback() -> None:
    assert route_after_review({"passed": True}, max_retries=1) == "end"
    assert route_after_review({"passed": False, "retry_count": 0}, max_retries=1) == "retry"
    assert route_after_review({"passed": False, "retry_count": 1}, max_retries=1) == "fallback"
