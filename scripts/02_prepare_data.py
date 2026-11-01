#!/usr/bin/env python
"""Clean the raw data and build the restaurant and review tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import load_config  # noqa: E402
from dinewise.data.clean import prepare_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="city to include; repeat for several. Default: config corpus.cities",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum restaurants")
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="ignore config corpus.limit and keep every selected restaurant",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_limit:
        cfg["corpus"]["limit"] = None
    cities = args.city if args.city else cfg["corpus"]["cities"]
    limit = args.limit if args.limit is not None else cfg["corpus"].get("limit")

    result = prepare_corpus(cfg, cities=cities, limit=limit)
    audit = result["audit"]
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
