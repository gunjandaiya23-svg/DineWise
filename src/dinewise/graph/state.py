"""Graph state shared by all workflow nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class DineWiseState(TypedDict, total=False):
    query: str
    profile: dict[str, Any]
    constraints: Any
    intent: str
    hits: list[dict[str, Any]]
    answer: str
    passed: bool
    validation: dict[str, Any]
    retry_count: int
    out_of_scope: bool
    refused: bool
    reason: str | None
    fallback_used: bool
    trace: list[str]
