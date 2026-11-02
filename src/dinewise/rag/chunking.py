"""Restaurant-aware, token-based chunking.

Every chunk is produced from a single restaurant block. The block is split
independently, so a chunk can never cross a restaurant boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

PROFILE_MARKER = re.compile(r"^\[RESTAURANT ([^\]]+)\]$", re.MULTILINE)
ANY_MARKER = re.compile(r"\[RESTAURANT ([^\]]+)\]")


def parse_profile_blocks(text: str) -> list[tuple[str, str]]:
    """Split a profiles file into (business_id, block) pairs."""
    matches = list(PROFILE_MARKER.finditer(text))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1).strip(), text[match.start() : end].strip()))
    return blocks


def windows(
    token_ids: list[int],
    size: int,
    overlap: int,
    min_size: int,
) -> list[list[int]]:
    """Split token ids into overlapping windows with a minimum tail size."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if min_size <= 0:
        raise ValueError("min_size must be positive")
    if not token_ids:
        return []

    step = size - overlap
    result: list[list[int]] = []
    for start in range(0, len(token_ids), step):
        window = list(token_ids[start : start + size])
        if len(window) < min_size and result:
            result[-1].extend(window)
            break
        result.append(window)
        if start + size >= len(token_ids):
            break
    return result


@dataclass
class TokenChunker:
    tokenizer: Any
    size: int
    overlap: int
    min_size: int = 80

    @classmethod
    def from_pretrained(cls, model_name: str, size: int, overlap: int, min_size: int = 80):
        from transformers import AutoTokenizer

        return cls(
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            size=size,
            overlap=overlap,
            min_size=min_size,
        )

    def chunk(self, text: str) -> list[str]:
        """Split text while preserving its original characters and casing.

        BGE's tokenizer is uncased, so decoding token ids would lower-case and
        space out punctuation. Offset mapping lets us slice the original text.
        """
        encoded = None
        if callable(self.tokenizer):
            try:
                encoded = self.tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True
                )
            except (TypeError, ValueError):
                encoded = None

        if encoded is None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            offsets = None
        else:
            ids = encoded["input_ids"]
            offsets = encoded.get("offset_mapping")

        if offsets is None:
            chunks = []
            for window in windows(ids, self.size, self.overlap, self.min_size):
                decoded = self.tokenizer.decode(
                    window, skip_special_tokens=True
                ).strip()
                if decoded:
                    chunks.append(decoded)
            return chunks

        chunks = []
        for window in windows(
            list(range(len(offsets))), self.size, self.overlap, self.min_size
        ):
            start = offsets[window[0]][0]
            end = offsets[window[-1]][1]
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
        return chunks

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def build_chunks(
    restaurants: pd.DataFrame,
    profiles_text: str,
    chunker: TokenChunker,
    config_name: str,
    source_file: str = "restaurant_profiles.txt",
) -> pd.DataFrame:
    """Chunk every restaurant block and attach catalogue metadata."""
    catalogue = restaurants.set_index("business_id")
    records: list[dict[str, Any]] = []

    for business_id, block in parse_profile_blocks(profiles_text):
        if business_id not in catalogue.index:
            continue
        row = catalogue.loc[business_id]
        header = f"[RESTAURANT {business_id}]"
        for index, text in enumerate(chunker.chunk(block)):
            if not text.lstrip().startswith(header):
                text = f"{header}\n{text}"
            records.append(
                {
                    "chunk_id": f"{business_id}::profile::{index:03d}",
                    "business_id": business_id,
                    "restaurant_name": row.get("name"),
                    "city": row.get("city"),
                    "state": row.get("state"),
                    "cuisine_primary": row.get("cuisine_primary"),
                    "cuisine_tags": row.get("cuisine_tags"),
                    "price_range": row.get("price_range"),
                    "stars": row.get("stars"),
                    "review_count": row.get("review_count"),
                    "document_type": "profile",
                    "section": "profile",
                    "source_file": source_file,
                    "page": None,
                    "token_count": chunker.count(text),
                    "char_count": len(text),
                    "chunk_config": config_name,
                    "outdoor_seating": row.get("outdoor_seating"),
                    "noise_level": row.get("noise_level"),
                    "ambience_romantic": row.get("ambience_romantic"),
                    "ambience_intimate": row.get("ambience_intimate"),
                    "ambience_classy": row.get("ambience_classy"),
                    "ambience_trendy": row.get("ambience_trendy"),
                    "ambience_casual": row.get("ambience_casual"),
                    "good_for_kids": row.get("good_for_kids"),
                    "good_for_meal_breakfast": row.get("good_for_meal_breakfast"),
                    "good_for_meal_brunch": row.get("good_for_meal_brunch"),
                    "good_for_meal_lunch": row.get("good_for_meal_lunch"),
                    "good_for_meal_dinner": row.get("good_for_meal_dinner"),
                    "good_for_meal_dessert": row.get("good_for_meal_dessert"),
                    "good_for_meal_latenight": row.get("good_for_meal_latenight"),
                    "reservations": row.get("reservations"),
                    "byob": row.get("byob"),
                    "delivery": row.get("delivery"),
                    "takeout": row.get("takeout"),
                    "wheelchair_accessible": row.get("wheelchair_accessible"),
                    "dogs_allowed": row.get("dogs_allowed"),
                    "text": text,
                }
            )
    return pd.DataFrame.from_records(records)


def contamination_report(chunks: pd.DataFrame) -> dict[str, Any]:
    """Count chunks containing more than one restaurant marker or name."""
    marker_counts = chunks["text"].map(lambda text: len(ANY_MARKER.findall(text)))
    return {
        "chunks": int(len(chunks)),
        "chunks_with_multiple_markers": int((marker_counts > 1).sum()),
        "max_markers_in_a_chunk": int(marker_counts.max()) if len(chunks) else 0,
    }
