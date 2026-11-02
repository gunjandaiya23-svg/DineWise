"""Workflow nodes: classify, retrieve, generate, review, retry, fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pandas as pd

from ..llm.guards import REFUSAL, check_scope
from ..llm.validators import validate_answer
from ..rag.constraints import Constraints, extract_constraints
from ..rag.pipeline import retrieve_with_config

INSUFFICIENT = (
    "I could not produce an answer that satisfies all of your constraints using "
    "the restaurant records available in this dataset. Try relaxing the budget, "
    "cuisine, or ambience requirement."
)


def _trace(state: dict[str, Any], message: str) -> list[str]:
    return [*state.get("trace", []), message]


def make_nodes(
    cfg: dict[str, Any],
    retriever: Any,
    bm25: Any,
    chunks: pd.DataFrame,
    restaurants: pd.DataFrame,
    generator: Any,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    known_cities = sorted(restaurants["city"].dropna().unique().tolist())
    max_retries = int(cfg["workflow"]["max_retries"])

    def classify(state: dict[str, Any]) -> dict[str, Any]:
        scope = check_scope(state["query"])
        if not scope.allowed:
            return {
                "out_of_scope": True,
                "reason": scope.reason,
                "trace": _trace(state, f"out_of_scope:{scope.reason}"),
            }
        constraints = extract_constraints(
            state["query"], known_cities=known_cities
        )
        return {
            "out_of_scope": False,
            "constraints": constraints,
            "intent": constraints.intent,
            "retry_count": 0,
            "trace": _trace(
                state,
                f"constraints:city={constraints.city},cuisines={constraints.cuisines},"
                f"max_price={constraints.max_price_bucket}",
            ),
        }

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        outcome = retrieve_with_config(
            retriever,
            bm25,
            chunks,
            restaurants,
            state["query"],
            mode="hybrid",
            constraints=state["constraints"],
            k=5,
        )
        if outcome.get("refused"):
            return {
                "refused": True,
                "reason": outcome["reason"],
                "hits": [],
                "trace": _trace(state, f"refused:{outcome['reason']}"),
            }
        return {
            "refused": False,
            "hits": outcome["hits"],
            "trace": _trace(state, f"retrieved:{len(outcome['hits'])}"),
        }

    def generate(state: dict[str, Any]) -> dict[str, Any]:
        answer = generator.generate(state["query"], state.get("hits", []))
        return {"answer": answer, "trace": _trace(state, "generated")}

    def review(state: dict[str, Any]) -> dict[str, Any]:
        report = validate_answer(
            state.get("answer", ""),
            state.get("hits", []),
            restaurants,
            state.get("constraints"),
        )
        validation = {
            "passed": report.passed,
            "hard_failures": [
                {"name": check.name, "detail": check.detail}
                for check in report.hard_failures
            ],
            "soft_failures": [
                {"name": check.name, "detail": check.detail}
                for check in report.soft_failures
            ],
        }
        return {
            "passed": report.passed,
            "validation": validation,
            "trace": _trace(
                state,
                "validated:" + ("pass" if report.passed else "fail"),
            ),
        }

    def retry(state: dict[str, Any]) -> dict[str, Any]:
        constraints: Constraints = state["constraints"]
        relaxed = replace(
            constraints,
            max_price_bucket=None,
            min_price_bucket=None,
            required_attributes=[],
            soft_preferences=constraints.soft_preferences + ["relaxed_on_retry"],
        )
        return {
            "constraints": relaxed,
            "retry_count": int(state.get("retry_count", 0)) + 1,
            "trace": _trace(state, "retry_relaxed_soft_constraints"),
        }

    def fallback(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("out_of_scope"):
            message = REFUSAL
        elif state.get("refused"):
            message = (
                f"I can't answer that from this dataset. {state.get('reason', '')} "
                "I can still help with cuisine, budget bucket, rating, or review evidence."
            )
        elif not state.get("hits"):
            message = (
                "No restaurant in the dataset satisfies all of your constraints. "
                "Try relaxing one requirement."
            )
        else:
            message = INSUFFICIENT
        return {
            "answer": message,
            "fallback_used": True,
            "trace": _trace(state, "fallback"),
        }

    _ = max_retries
    return {
        "classify": classify,
        "retrieve": retrieve,
        "generate": generate,
        "review": review,
        "retry": retry,
        "fallback": fallback,
    }


def route_after_classify(state: dict[str, Any]) -> str:
    return "fallback" if state.get("out_of_scope") else "retrieve"


def route_after_retrieve(state: dict[str, Any]) -> str:
    if state.get("refused") or not state.get("hits"):
        return "fallback"
    return "generate"


def route_after_review(state: dict[str, Any], max_retries: int = 1) -> str:
    if state.get("passed"):
        return "end"
    if int(state.get("retry_count", 0)) < max_retries:
        return "retry"
    return "fallback"
