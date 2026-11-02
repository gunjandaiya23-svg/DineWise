#!/usr/bin/env python
"""Build the TXT, PDF, and DOCX knowledge-base documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.data.documents import build_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    restaurants_path = data_path(cfg, "processed_dir", "restaurants.parquet")
    if not restaurants_path.exists():
        raise SystemExit(f"missing {restaurants_path}; run scripts/02_prepare_data.py first")

    restaurants = pd.read_parquet(restaurants_path)
    reviews_path = data_path(cfg, "processed_dir", "reviews.parquet")
    reviews = pd.read_parquet(reviews_path) if reviews_path.exists() else None
    manifest = build_documents(restaurants, reviews, data_path(cfg, "docs_dir"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
