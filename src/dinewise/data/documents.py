"""Generate the human-readable knowledge-base documents (TXT, PDF, DOCX)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from ..logging import get_logger

LOG = get_logger(__name__)

POLICY: list[tuple[str, str]] = [
    (
        "Budget",
        "Filter by the dataset's price bucket. Never invent a dollar price. "
        "Treat an unknown price as unknown, not as cheap.",
    ),
    (
        "Cuisine",
        "Prefer restaurants whose cuisine tags match the request. If no exact match "
        "exists, say so before offering alternatives.",
    ),
    (
        "Dietary preferences",
        "Use explicit evidence only. A review mentioning vegetarian options is evidence; "
        "the absence of such a mention is not evidence of absence. Medical allergies must "
        "never be inferred.",
    ),
    (
        "Ambience",
        "Use structured attributes first (romantic, intimate, classy, noise level, outdoor "
        "seating). Use review and photo summaries only as supporting evidence. A policy "
        "preference for quiet restaurants does not establish that a particular restaurant "
        "is quiet.",
    ),
    (
        "Ranking",
        "Rank by constraint satisfaction, then relevance, then review evidence, then rating. "
        "Do not rank by rating alone.",
    ),
    (
        "Explanation",
        "Explain why each restaurant matches. Cite the source of every factual claim.",
    ),
    (
        "Uncertainty",
        "If the dataset lacks information, say so. Do not guess.",
    ),
    (
        "Out of scope",
        "Do not answer questions about live availability, reservations, current prices, or "
        "real-time wait times.",
    ),
]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any, default: str = "") -> str:
    return default if _missing(value) else str(value).strip()


def _yes_no(value: Any) -> str:
    if _missing(value):
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _join(values: Any, separator: str = "; ") -> str:
    if _missing(values):
        return ""
    if isinstance(values, str):
        return values
    try:
        return separator.join(str(value) for value in values)
    except TypeError:
        return str(values)


def _boolean_list(row: pd.Series, prefix: str) -> list[str]:
    labels = []
    for column in row.index:
        if column.startswith(prefix) and row[column] is True:
            labels.append(column[len(prefix) :].replace("_", " "))
    return labels


def build_profile_block(row: pd.Series) -> str:
    name = _text(row.get("name"), "Unknown")
    location = ", ".join(
        part for part in (_text(row.get("city")), _text(row.get("state"))) if part
    )
    cuisines = _join(row.get("cuisine_tags"))
    price = _text(row.get("price_label"), "unknown")
    bucket = row.get("price_range")
    bucket_text = "unknown" if _missing(bucket) else str(bucket)
    good_for = _boolean_list(row, "good_for_meal_")
    summary = _text(row.get("review_summary"))[:500]
    visual = _text(row.get("photo_summary"))[:300]

    lines = [
        f"[RESTAURANT {row['business_id']}]",
        f"Name: {name}",
        f"City: {location or 'unknown'}",
        f"Cuisine: {cuisines or 'Unspecified'}",
        f"Price: {price} (dataset bucket {bucket_text} of 4)",
        f"Rating: {row.get('stars')} out of 5 (dataset average, "
        f"{int(row.get('review_count') or 0)} reviews)",
        f"Status: {'Open' if row.get('is_open') == 1 else 'Closed'} in dataset",
        f"Outdoor seating: {_yes_no(row.get('outdoor_seating'))}",
        f"Noise level: {_text(row.get('noise_level'), 'unknown')}",
        f"Romantic ambience: {_yes_no(row.get('ambience_romantic'))}",
        f"Good for: {', '.join(good_for) if good_for else 'unknown'}",
    ]
    if summary:
        lines.append(f"Review summary: {summary}")
    if visual:
        lines.append(f"Visual summary: {visual}")
    lines.append("Source: business.csv; review_summary.csv; photo_summay.csv")
    return "\n".join(lines)


def write_profiles(restaurants: pd.DataFrame, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    blocks = [build_profile_block(row) for _, row in restaurants.iterrows()]
    content = "\n\n".join(blocks) + "\n"
    path.write_text(content, encoding="utf-8")
    elapsed = time.perf_counter() - started
    LOG.info("wrote %s (%d blocks, %.1f KB)", path.name, len(blocks), len(content) / 1024)
    return {
        "path": str(path),
        "blocks": len(blocks),
        "chars": len(content),
        "seconds": round(elapsed, 2),
    }


def _pdf_body(value: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(value), style)


def write_review_pdf(restaurants: pd.DataFrame, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "DineWiseHeading",
        parent=styles["Heading2"],
        fontSize=10.5,
        leading=13,
        spaceBefore=4,
        spaceAfter=2,
    )
    body = ParagraphStyle(
        "DineWiseBody", parent=styles["BodyText"], fontSize=8.2, leading=10.5
    )

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="DineWise Customer Review Insights",
        author="DineWise",
    )

    story: list[Any] = []
    for _, row in restaurants.iterrows():
        name = _text(row.get("name"), "Unknown")
        story.append(
            _pdf_body(f"{name} ({row['business_id']})", heading)
        )
        meta = (
            f"Cuisine: {_text(row.get('cuisine_tags'), 'Unspecified')} | "
            f"City: {_text(row.get('city'), 'unknown')} | "
            f"Price: {_text(row.get('price_label'), 'unknown')} | "
            f"Rating: {row.get('stars')} ({int(row.get('review_count') or 0)} reviews)"
        )
        story.append(_pdf_body(meta, body))

        summary = _text(row.get("review_summary"))
        if summary:
            story.append(_pdf_body(f"Review summary: {summary}", body))
        themes = _text(row.get("review_themes"))
        if themes:
            story.append(_pdf_body(f"Recurring themes: {themes}", body))
        keywords = _text(row.get("review_keywords"))
        if keywords:
            story.append(_pdf_body(f"Keywords: {keywords}", body))
        visual = _text(row.get("photo_summary"))
        if visual:
            story.append(_pdf_body(f"Visual description: {visual}", body))
        story.append(
            _pdf_body(
                "Source: review_summary.csv and review.csv (aggregated); "
                "summary generated upstream and not independently verified.",
                body,
            )
        )
        story.append(Spacer(1, 4 * mm))

    document.build(story)
    elapsed = time.perf_counter() - started
    LOG.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    return {
        "path": str(path),
        "entries": len(restaurants),
        "bytes": path.stat().st_size,
        "seconds": round(elapsed, 2),
    }


def write_policy_docx(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    document = Document()
    document.add_heading("DineWise Recommendation Policy", level=1)
    document.add_paragraph(
        "These are rules for the assistant, not facts about any restaurant. "
        "Policy text must never be cited as evidence about a specific business."
    )
    for title, text in POLICY:
        document.add_heading(title, level=2)
        document.add_paragraph(text)
    document.save(path)
    elapsed = time.perf_counter() - started
    LOG.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    return {
        "path": str(path),
        "sections": len(POLICY),
        "bytes": path.stat().st_size,
        "seconds": round(elapsed, 2),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_documents(restaurants: pd.DataFrame, docs_dir: Path) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "restaurants": int(len(restaurants)),
        "files": {},
    }
    manifest["files"]["profiles"] = write_profiles(
        restaurants, docs_dir / "restaurant_profiles.txt"
    )
    manifest["files"]["review_insights"] = write_review_pdf(
        restaurants, docs_dir / "customer_review_insights.pdf"
    )
    manifest["files"]["policy"] = write_policy_docx(
        docs_dir / "dining_preferences.docx"
    )
    for info in manifest["files"].values():
        info["sha256"] = sha256_file(Path(info["path"]))
    (docs_dir / "doc_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
