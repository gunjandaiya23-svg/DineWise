"""Scope guards for questions the dataset cannot answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

OUT_OF_SCOPE_PATTERNS: list[tuple[str, str]] = [
    (
        r"\b(available|availability)\b.*\b(tonight|today|now|tomorrow|at\s+\d)",
        "live availability",
    ),
    (r"\b(open|opening)\b.*\b(now|tonight|today|at\s+\d)", "live opening status"),
    (r"\b(book|booking|reserve|reservation|table\s+for)\b", "booking or reservation"),
    (r"\b(wait\s*time|waitlist|wait\s*list)\b", "live wait time"),
    (r"\b(current|today'?s|live)\s+(price|prices|menu|availability)\b", "live price or menu"),
    (r"\b(do you have|is there)\b.*\b(table|availability)\b", "live availability"),
]

REFUSAL = (
    "I can't answer that from this dataset. DineWise uses static restaurant "
    "records and reviews, and it has no live availability, booking, wait-time, "
    "or current-price data. I can still help you find restaurants by cuisine, "
    "budget bucket, location, rating, or review evidence."
)


@dataclass(frozen=True)
class ScopeCheck:
    allowed: bool
    reason: str | None = None


def check_scope(query: str) -> ScopeCheck:
    """Return a refusal reason when the query needs live data."""
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return ScopeCheck(False, reason)
    return ScopeCheck(True)
