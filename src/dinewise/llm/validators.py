"""Programmatic checks for generated answers and retrieved sets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..rag.constraints import Constraints

LIVE_LANGUAGE = [
    r"\bavailable\s+(tonight|today|now|tomorrow|this\s+evening)\b",
    r"\bopen\s+now\b",
    r"\btable\s+available\b",
    r"\bbook\s+(a\s+table|now|online)\b",
    r"\breserv(e|ation|e\s+a\s+table)\b",
    r"\bcurrent\s+price\b",
    r"\breal-?time\b",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    hard_failures: list[Check] = field(default_factory=list)
    soft_failures: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.hard_failures


def _ids_in_text(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_\-]{22}", text))


def validate_answer(
    answer: str,
    candidates: list[dict],
    restaurants: pd.DataFrame,
    constraints: Constraints | None = None,
) -> ValidationReport:
    """Check the free-text answer against the supplied candidates."""
    report = ValidationReport()
    catalogue = restaurants.set_index("business_id")

    mentioned = _ids_in_text(answer)
    known_ids = {str(hit["chunk"]["business_id"]) for hit in candidates}
    unknown = mentioned - known_ids
    if unknown:
        report.hard_failures.append(
            Check("ids_exist", False, f"unknown business ids: {sorted(unknown)[:3]}")
        )

    if not mentioned:
        report.soft_failures.append(
            Check("ids_cited", False, "answer cites no business ids")
        )

    mentioned_names = set(re.findall(r"\*\*(.+?)\*\*", answer))
    for name in mentioned_names:
        clean = name.strip().lower()
        if clean and not any(
            clean in str(hit["chunk"].get("restaurant_name", "")).lower()
            for hit in candidates
        ):
            # Names could be paraphrased; record as soft only when no match at all.
            report.soft_failures.append(
                Check("names_match", False, f"name not in candidates: {name[:40]}")
            )

    leaked = set(catalogue.index).intersection(mentioned) - known_ids
    if leaked:
        report.hard_failures.append(
            Check("no_leakage", False, f"mentions non-candidates: {sorted(leaked)[:3]}")
        )

    for pattern in LIVE_LANGUAGE:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            report.hard_failures.append(
                Check("no_live_claims", False, f"live-availability language: {pattern}")
            )
            break

    if constraints and mentioned:
        for business_id in sorted(mentioned & set(catalogue.index)):
            row = catalogue.loc[business_id]
            if constraints.max_price_bucket is not None and pd.notna(row.get("price_range")):
                if int(row["price_range"]) > constraints.max_price_bucket:
                    report.hard_failures.append(
                        Check(
                            "price_satisfied",
                            False,
                            f"{business_id} bucket {row['price_range']} exceeds max",
                        )
                    )
                    break

    return report


def check_retrieved_compliance(
    hits: list[dict], restaurants: pd.DataFrame, constraints: Constraints
) -> list[Check]:
    """Measure whether the retrieved set honours the hard constraints."""
    catalogue = restaurants.set_index("business_id")
    checks: list[Check] = []
    if not hits:
        return [Check("non_empty", False, "no candidates retrieved")]

    for hit in hits:
        business_id = str(hit["chunk"]["business_id"])
        if business_id not in catalogue.index:
            checks.append(Check("ids_in_catalogue", False, business_id))
            return checks
    checks.append(Check("ids_in_catalogue", True, f"{len(hits)} ids"))

    if constraints.cuisines:
        wanted = set(constraints.cuisines)
        bad = [
            str(hit["chunk"]["business_id"])
            for hit in hits
            if str(hit["chunk"].get("cuisine_primary")) not in wanted
        ]
        checks.append(
            Check(
                "cuisine_satisfied",
                not bad,
                f"{len(hits) - len(bad)}/{len(hits)} match" if bad else "all match",
            )
        )

    if constraints.max_price_bucket is not None:
        bad = [
            str(hit["chunk"]["business_id"])
            for hit in hits
            if pd.notna(hit["chunk"].get("price_range"))
            and int(hit["chunk"]["price_range"]) > constraints.max_price_bucket
        ]
        checks.append(
            Check(
                "price_satisfied",
                not bad,
                f"{len(hits) - len(bad)}/{len(hits)} within budget" if bad else "all match",
            )
        )

    if constraints.city:
        bad = [
            str(hit["chunk"]["business_id"])
            for hit in hits
            if str(hit["chunk"].get("city", "")).lower() != constraints.city.lower()
        ]
        checks.append(
            Check("city_satisfied", not bad, "all match" if not bad else f"{bad[:3]}")
        )

    return checks
