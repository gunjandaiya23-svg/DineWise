#!/usr/bin/env python
"""Chunk the profile documents with the active configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.rag.chunking import (  # noqa: E402
    TokenChunker,
    build_chunks,
    contamination_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--config-name", default=None, help="chunking config, e.g. C2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = args.config_name or cfg["chunking"]["active"]
    settings = cfg["chunking"]["configs"][name]

    profiles = data_path(cfg, "docs_dir", cfg["docs"]["profile_file"])
    if not profiles.exists():
        raise SystemExit(f"missing {profiles}; run scripts/03_build_documents.py first")

    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunker = TokenChunker.from_pretrained(
        cfg["embedding"]["model"],
        size=settings["size"],
        overlap=settings["overlap"],
        min_size=settings["min_size"],
    )
    chunks = build_chunks(
        restaurants,
        profiles.read_text(encoding="utf-8"),
        chunker,
        config_name=name,
    )

    output = data_path(cfg, "processed_dir", f"chunks_{name}.parquet")
    chunks.to_parquet(output, index=False)
    report = contamination_report(chunks)
    report.update(
        {
            "config": name,
            "size": settings["size"],
            "overlap": settings["overlap"],
            "mean_tokens": round(float(chunks["token_count"].mean()), 1),
            "min_tokens": int(chunks["token_count"].min()),
            "max_tokens": int(chunks["token_count"].max()),
            "output": str(output),
        }
    )
    (data_path(cfg, "processed_dir", f"chunks_{name}.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
