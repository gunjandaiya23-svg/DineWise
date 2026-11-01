"""Prompt construction for grounded generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

SYSTEM_PROMPT = """You are DineWise, a restaurant recommendation assistant.

You answer only from the RESTAURANT RECORDS supplied in this prompt. You do
not have access to the internet, live availability, current prices, or
reservations.

Rules:
1. Never invent a restaurant, address, rating, price, dish, dietary fact, or amenity.
2. Ratings and prices come from a static dataset. Describe them as dataset values,
   never as live facts.
3. Never claim that a restaurant is currently open, has a table available, or
   offers a reservation.
4. Every factual sentence must be supported by the supplied records, and must
   mention the restaurant name or ID.
5. If the records do not contain enough evidence, say so explicitly.
6. Prefer fewer, well-supported recommendations over more, weakly supported ones.
7. Keep the answer under 250 words. Use a short numbered list."""


def _value(record: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = record.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def build_context(hits: list[dict[str, Any]], max_restaurants: int = 5) -> str:
    """Format unique restaurants and their best chunk into a context block."""
    blocks: list[str] = []
    seen: set[str] = set()

    for hit in hits:
        record = hit["chunk"]
        business_id = str(record["business_id"])
        if business_id in seen:
            continue
        seen.add(business_id)

        block = [
            f"[RESTAURANT {business_id}]",
            f"Name: {_value(record, 'restaurant_name')}",
            f"City: {_value(record, 'city')}, {_value(record, 'state')}",
            f"Cuisine: {_value(record, 'cuisine_primary')}",
            f"Price bucket: {_value(record, 'price_range')} of 4",
            "Record excerpt:",
            str(record["text"]),
        ]
        blocks.append("\n".join(block))
        if len(blocks) >= max_restaurants:
            break

    return "\n\n".join(blocks) if blocks else "No restaurant records were retrieved."


def build_messages(query: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = build_context(hits)
    user = f"""USER QUESTION:
{query}

RESTAURANT RECORDS:
{context}

TASK:
Recommend up to 3 restaurants from the records above. For each one, give the
name, the business ID, why it matches the question, and note that ratings and
prices are dataset values. If the records do not support an answer, say so.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
