#!/usr/bin/env python
"""Embed chunks and ingest them into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.rag.vectorstore import (  # noqa: E402
    ingest_chunks,
    store_from_config,
    write_chunks_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--config-name", default=None, help="chunking config, e.g. C3")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild collection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = args.config_name or cfg["chunking"]["active"]
    chunks_path = data_path(cfg, "processed_dir", f"chunks_{name}.parquet")
    if not chunks_path.exists():
        raise SystemExit(f"missing {chunks_path}; run scripts/04_build_chunks.py first")

    chunks = pd.read_parquet(chunks_path)
    model = SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    model.max_seq_length = int(cfg["embedding"]["max_seq_length"])

    started = time.perf_counter()
    vectors = model.encode(
        chunks["text"].tolist(),
        batch_size=int(cfg["embedding"]["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    embed_seconds = time.perf_counter() - started

    store = store_from_config(cfg)
    started = time.perf_counter()
    written = ingest_chunks(
        store,
        f"chunks_{name}",
        chunks,
        vectors,
        recreate=args.recreate or bool(cfg["vector_store"].get("recreate", False)),
    )
    ingest_seconds = time.perf_counter() - started
    cache = write_chunks_cache(cfg, name, chunks)
    points = store.count(f"chunks_{name}")

    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "qdrant",
        "url": store.url,
        "collection": store.collection(f"chunks_{name}"),
        "model": cfg["embedding"]["model"],
        "chunk_config": name,
        "chunks": int(len(chunks)),
        "points": points,
        "written": written,
        "dimension": int(vectors.shape[1]),
        "exact": store.exact,
        "embed_seconds": round(embed_seconds, 2),
        "chunks_per_second": round(len(chunks) / embed_seconds, 2),
        "ingest_seconds": round(ingest_seconds, 2),
        "cache": str(cache),
    }
    run_dir = ROOT / "results" / "runs" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_ingest_qdrant"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    (data_path(cfg, "index_dir", "qdrant", name)).mkdir(parents=True, exist_ok=True)
    (data_path(cfg, "index_dir", "qdrant", name) / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
