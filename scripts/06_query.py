#!/usr/bin/env python
"""Retrieve restaurants for a query (retrieval only, no LLM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import load_config  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--config-name", default=None, help="chunking config, e.g. C2")
    parser.add_argument("--k", type=int, default=5, help="number of restaurants to show")
    args = parser.parse_args()

    cfg = load_config(args.config)
    retriever = load_retriever(cfg, args.config_name)
    hits = retriever.unique_restaurants(args.query, k=args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        row = hit["chunk"]
        print(
            f"{rank}. {row['restaurant_name']} ({row['business_id']})  "
            f"score={hit['score']:.4f}  {row['cuisine_primary']}  {row['city']}"
        )
        print(f"   {str(row['text']).replace(chr(10), ' ')[:240]}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
