"""Cuisine family mapping from the free-text ``categories`` column."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT

DEFAULT_TAXONOMY = ROOT / "config" / "cuisine_taxonomy.yaml"


def load_taxonomy(path: str | Path | None = None) -> dict[str, Any]:
    """Load the taxonomy and build a lower-case token lookup."""
    taxonomy_path = Path(path) if path is not None else DEFAULT_TAXONOMY
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    families: dict[str, list[str]] = data.get("families", {})
    lookup: dict[str, str] = {}
    for family, tokens in families.items():
        for token in tokens:
            lookup[str(token).strip().lower()] = family

    priority = list(data.get("priority") or families.keys())
    return {
        "priority": priority,
        "families": families,
        "lookup": lookup,
        "path": str(taxonomy_path),
    }


def category_tokens(categories: Any) -> list[str]:
    """Split a comma-separated category string into lower-case tokens."""
    if not isinstance(categories, str):
        return []
    return [token.strip().lower() for token in categories.split(",") if token.strip()]


def match_cuisines(categories: Any, taxonomy: dict[str, Any]) -> list[str]:
    """Return matching cuisine families ordered by taxonomy priority."""
    tokens = set(category_tokens(categories))
    lookup = taxonomy["lookup"]
    found = {lookup[token] for token in tokens if token in lookup}
    order = {family: index for index, family in enumerate(taxonomy["priority"])}
    return sorted(found, key=lambda family: order.get(family, len(order)))


def primary_cuisine(matches: list[str]) -> str:
    return matches[0] if matches else "Unspecified"


def annotate_cuisines(categories: Any, taxonomy: dict[str, Any]) -> tuple[list[str], str]:
    matches = match_cuisines(categories, taxonomy)
    return matches, primary_cuisine(matches)
