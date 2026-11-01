#!/usr/bin/env python
"""Embed chunks with BGE and build a FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--config-name", default=None, help="chunking config, e.g. C2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = args.config_name or cfg["chunking"]["active"]
    chunks_path = data_path(cfg, "processed_dir", f"chunks_{name}.parquet")
    if not chunks_path.exists():
        raise SystemExit(f"missing {chunks_path}; run scripts/04_build_chunks.py first")

    chunks = pd.read_parquet(chunks_path)
    texts = chunks["text"].tolist()

    model = SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    model.max_seq_length = int(cfg["embedding"]["max_seq_length"])

    started = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=int(cfg["embedding"]["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    embed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_seconds = time.perf_counter() - started

    index_dir = data_path(cfg, "index_dir", "faiss", name)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    chunks.to_parquet(index_dir / "chunks.parquet", index=False)

    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": cfg["embedding"]["model"],
        "chunk_config": name,
        "chunks": int(len(chunks)),
        "dimension": int(embeddings.shape[1]),
        "embed_seconds": round(embed_seconds, 2),
        "chunks_per_second": round(len(chunks) / embed_seconds, 2),
        "index_seconds": round(index_seconds, 3),
        "index_bytes": (index_dir / "index.faiss").stat().st_size,
    }
    (index_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
