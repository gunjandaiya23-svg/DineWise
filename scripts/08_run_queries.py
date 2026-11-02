#!/usr/bin/env python
"""Run the adapted 15-query test set through extract + filter + retrieve.

Saves machine-readable results and a markdown table under results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval import ALL_QUERIES  # noqa: E402
from dinewise.llm.guards import check_scope  # noqa: E402
from dinewise.llm.validators import check_retrieved_compliance  # noqa: E402
from dinewise.rag.pipeline import retrieve_filtered  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

QUERIES = ALL_QUERIES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    retriever = load_retriever(cfg)

    results: list[dict] = []
    for query_id, facet, query in QUERIES:
        started = time.perf_counter()
        scope = check_scope(query)
        entry: dict = {
            "query_id": query_id,
            "facet": facet,
            "query": query,
            "in_scope": scope.allowed,
            "scope_reason": scope.reason,
        }
        if scope.allowed:
            outcome = retrieve_filtered(retriever, restaurants, query, k=args.k)
            if outcome.get("refused"):
                entry["in_scope"] = False
                entry["scope_reason"] = outcome["reason"]
                entry["seconds"] = round(time.perf_counter() - started, 2)
                results.append(entry)
                print(f"{query_id} [{facet}] refused: {outcome['reason']}")
                continue
            constraints = outcome["constraints"]
            entry.update(
                {
                    "constraints": {
                        "city": constraints.city,
                        "cuisines": constraints.cuisines,
                        "max_price_bucket": constraints.max_price_bucket,
                        "min_rating": constraints.min_rating,
                        "required_attributes": constraints.required_attributes,
                        "dietary": constraints.dietary,
                        "intent": constraints.intent,
                    },
                    "filter_counts": outcome["filter_counts"],
                    "candidates_after_filter": outcome["candidates_after_filter"],
                    "hits": [
                        {
                            "business_id": str(hit["chunk"]["business_id"]),
                            "name": str(hit["chunk"].get("restaurant_name")),
                            "cuisine": str(hit["chunk"].get("cuisine_primary")),
                            "city": str(hit["chunk"].get("city")),
                            "score": round(hit["score"], 4),
                        }
                        for hit in outcome["hits"]
                    ],
                    "compliance": [
                        {"name": check.name, "passed": check.passed, "detail": check.detail}
                        for check in check_retrieved_compliance(
                            outcome["hits"], restaurants, constraints
                        )
                    ],
                }
            )
        entry["seconds"] = round(time.perf_counter() - started, 2)
        results.append(entry)
        names = ", ".join(hit["name"] for hit in entry.get("hits", [])[:3]) or entry.get(
            "scope_reason", "no hits"
        )
        print(f"{query_id} [{facet}] {names}")

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_query15"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["| ID | Facet | Constraints | Candidates | Top hit | Compliance |", "|---|---|---|---|---|---|"]
    for entry in results:
        if not entry["in_scope"]:
            lines.append(
                f"| {entry['query_id']} | {entry['facet']} | out-of-scope: "
                f"{entry['scope_reason']} | - | - | refused |"
            )
            continue
        constraint = entry["constraints"]
        short = ",".join(
            part
            for part in [
                constraint["city"] or "",
                "/".join(constraint["cuisines"]),
                f"{'$' * (constraint['max_price_bucket'] or 0)} max"
                if constraint["max_price_bucket"]
                else "",
            ]
            if part
        )
        failed = [
            check["name"]
            for check in entry["compliance"]
            if not check["passed"] and check["name"] != "non_empty"
        ]
        top = entry["hits"][0]["name"] if entry["hits"] else "none"
        lines.append(
            f"| {entry['query_id']} | {entry['facet']} | {short or '-'} | "
            f"{entry['candidates_after_filter']} | {top} | "
            f"{'FAIL:' + ','.join(failed) if failed else 'ok'} |"
        )
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table, encoding="utf-8")
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "query15_latest.md").write_text(table, encoding="utf-8")

    print(f"\nsaved to {run_dir}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
