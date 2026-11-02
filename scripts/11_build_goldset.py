#!/usr/bin/env python
"""Build the silver gold set and its manual review sheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval import ANSWERABLE_QUERIES  # noqa: E402
from dinewise.eval.goldset import build_goldset, write_goldset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    records = build_goldset(restaurants, ANSWERABLE_QUERIES)
    jsonl, review = write_goldset(records, ROOT / "data" / "gold")

    summary = []
    for record in records:
        grades = [item["grade"] for item in record["gold"]]
        summary.append(
            {
                "query_id": record["query_id"],
                "facet": record["facet"],
                "pool": record.get("candidates_in_pool", 0),
                "gold": len(grades),
                "grade2": grades.count(2),
                "grade1": grades.count(1),
            }
        )
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {jsonl} and {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
