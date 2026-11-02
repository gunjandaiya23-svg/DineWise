"""Compile the DineWise LangGraph workflow."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from .nodes import (
    make_nodes,
    route_after_classify,
    route_after_retrieve,
    route_after_review,
)
from .state import DineWiseState


def build_graph(
    cfg: dict[str, Any],
    retriever: Any,
    bm25: Any,
    chunks: Any,
    restaurants: Any,
    generator: Any,
):
    nodes = make_nodes(cfg, retriever, bm25, chunks, restaurants, generator)
    max_retries = int(cfg["workflow"]["max_retries"])

    builder = StateGraph(DineWiseState)
    for name, function in nodes.items():
        builder.add_node(name, function)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "fallback": "fallback"},
    )
    builder.add_edge("generate", "review")
    builder.add_conditional_edges(
        "review",
        partial(route_after_review, max_retries=max_retries),
        {"end": END, "retry": "retry", "fallback": "fallback"},
    )
    builder.add_edge("retry", "retrieve")
    builder.add_edge("fallback", END)
    return builder.compile()


def run_query(graph, query: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    return graph.invoke({"query": query, "profile": profile or {}})
