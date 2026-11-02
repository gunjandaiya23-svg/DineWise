"""BM25 lexical index over the same chunks as the dense index.

Tokenization is deliberately simple and dependency-free: lowercase,
alphanumeric word tokens, fixed English stopword removal, no stemming.
The choice is recorded in the index metadata so retrieval experiments
compare identical chunk units across dense and lexical paths.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from ..config import data_path

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Fixed stopword list (Lucene-style subset). Versioned here; any change
# requires rebuilding the index and is recorded in the metadata.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no nor
    not of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why with would you your yours yourself yourselves
    """.split()
)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    ]


@dataclass
class BM25Index:
    model: BM25Okapi
    chunk_ids: list[str]
    meta: dict[str, Any]

    def search(self, query: str, k: int = 30) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"chunk_id": self.chunk_ids[i], "score": float(scores[i])}
            for i in order
            if scores[i] > 0
        ]


def build_bm25(chunks: pd.DataFrame) -> BM25Index:
    tokenized = [tokenize(text) for text in chunks["text"].tolist()]
    model = BM25Okapi(tokenized, k1=K1, b=B)
    return BM25Index(
        model=model,
        chunk_ids=chunks["chunk_id"].tolist(),
        meta={
            "built_at": datetime.now(UTC).isoformat(),
            "chunks": int(len(chunks)),
            "k1": K1,
            "b": B,
            "tokenizer": "lowercase_alnum_stopwords_v1",
        },
    )


def save_bm25(index: BM25Index, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "bm25.pkl").open("wb") as handle:
        pickle.dump(
            {"model": index.model, "chunk_ids": index.chunk_ids}, handle
        )
    meta = dict(index.meta)
    meta["path"] = str(directory / "bm25.pkl")
    (directory / "bm25_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return directory / "bm25.pkl"


def load_bm25(cfg: dict[str, Any], name: str | None = None) -> BM25Index:
    name = name or cfg["chunking"]["active"]
    directory = data_path(cfg, "index_dir", "bm25", name)
    path = directory / "bm25.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing BM25 index at {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - own artifact
    meta = json.loads((directory / "bm25_meta.json").read_text(encoding="utf-8"))
    return BM25Index(model=payload["model"], chunk_ids=payload["chunk_ids"], meta=meta)
