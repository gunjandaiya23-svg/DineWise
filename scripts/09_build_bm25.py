#!/usr/bin/env python
"""Build the BM25 lexical index over the chunked documents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.rag.bm25_index import build_bm25, save_bm25  # noqa: E402


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

    started = time.perf_counter()
    chunks = pd.read_parquet(chunks_path)
    index = build_bm25(chunks)
    path = save_bm25(index, data_path(cfg, "index_dir", "bm25", name))
    elapsed = time.perf_counter() - started

    meta = dict(index.meta)
    meta.update({"build_seconds": round(elapsed, 2), "bytes": path.stat().st_size})
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
