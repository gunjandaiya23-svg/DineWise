from __future__ import annotations

from dinewise.rag.chunking import (
    TokenChunker,
    build_chunks,
    contamination_report,
    parse_profile_blocks,
    windows,
)


class StubTokenizer:
    """Whitespace tokenizer so chunking tests need no model download."""

    def __init__(self) -> None:
        self.words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.words = text.split()
        return list(range(len(self.words)))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.words[index] for index in ids)


def test_windows_are_overlapping_and_cover_the_input() -> None:
    result = windows(list(range(10)), size=4, overlap=2, min_size=2)
    assert result[0] == [0, 1, 2, 3]
    assert result[1] == [2, 3, 4, 5]
    assert result[-1][-1] == 9


def test_short_tail_is_merged_into_previous_window() -> None:
    result = windows(list(range(9)), size=4, overlap=0, min_size=4)
    assert len(result) == 2
    assert result[-1] == [4, 5, 6, 7, 8]


def test_invalid_overlap_is_rejected() -> None:
    for overlap, size in ((4, 4), (-1, 4), (5, 4)):
        try:
            windows([1, 2, 3], size=size, overlap=overlap, min_size=1)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_parse_profile_blocks() -> None:
    text = "[RESTAURANT A]\nName: One\n\n[RESTAURANT B]\nName: Two\n"
    blocks = parse_profile_blocks(text)
    assert [block[0] for block in blocks] == ["A", "B"]
    assert "Name: One" in blocks[0][1]
    assert "Name: Two" in blocks[1][1]


def test_chunks_never_cross_restaurant_boundaries() -> None:
    import pandas as pd

    restaurants = pd.DataFrame(
        [
            {
                "business_id": "A",
                "name": "One",
                "city": "X",
                "state": "PA",
                "cuisine_primary": "Italian",
                "price_range": 2,
            },
            {
                "business_id": "B",
                "name": "Two",
                "city": "X",
                "state": "PA",
                "cuisine_primary": "Mexican",
                "price_range": 1,
            },
        ]
    )
    text = (
        "[RESTAURANT A]\n" + "alpha " * 200 + "\n\n"
        "[RESTAURANT B]\n" + "beta " * 200 + "\n"
    )
    chunker = TokenChunker(StubTokenizer(), size=40, overlap=10, min_size=10)
    chunks = build_chunks(restaurants, text, chunker, config_name="test")
    assert len(chunks) > 2
    report = contamination_report(chunks)
    assert report["chunks_with_multiple_markers"] == 0
    assert report["max_markers_in_a_chunk"] == 1
    for _, row in chunks.iterrows():
        own = row["business_id"]
        other = "B" if own == "A" else "A"
        assert f"[RESTAURANT {other}]" not in row["text"]
