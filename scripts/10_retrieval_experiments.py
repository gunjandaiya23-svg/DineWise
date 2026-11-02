#!/usr/bin/env python
"""E1/E2/E3: compare BM25-only, dense-only, and hybrid RRF retrieval.

Relevance judgments do not exist yet, so this run measures what can be
measured honestly: hard-constraint compliance, latency, candidate pools,
and agreement between configurations. Precision/nDCG columns are added
by a later script once the gold set is judged.
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
from dinewise.eval import ANSWERABLE_QUERIES  # noqa: E402
from dinewise.llm.validators import check_retrieved_compliance  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

MODES = ["bm25", "dense", "hybrid"]


def _ids(hits: list[dict]) -> list[str]:
    return [str(hit["chunk"]["business_id"]) for hit in hits]


def _jaccard(first: list[str], second: list[str]) -> float:
    left, right = set(first), set(second)
    return len(left & right) / len(left | right) if left | right else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = pd.read_parquet(data_path(cfg, "index_dir", "faiss", name, "chunks.parquet"))
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)

    per_query: list[dict] = []
    for query_id, facet, query in ANSWERABLE_QUERIES:
        entry: dict = {"query_id": query_id, "facet": facet, "query": query, "modes": {}}
        for mode in MODES:
            started = time.perf_counter()
            outcome = retrieve_with_config(
                retriever, bm25, chunks, restaurants, query,
                mode=mode, pool=args.pool, k=args.k,
            )
            wall = time.perf_counter() - started
            compliance = check_retrieved_compliance(
                outcome["hits"], restaurants, outcome["constraints"]
            )
            failed = [c.name for c in compliance if not c.passed]
            entry["modes"][mode] = {
                "ids": _ids(outcome["hits"]),
                "names": [str(h["chunk"].get("restaurant_name")) for h in outcome["hits"]],
                "timings": outcome["timings"],
                "wall_seconds": round(wall, 3),
                "candidates_after_filter": outcome["candidates_after_filter"],
                "compliance_failed": failed,
            }
        ids = {mode: entry["modes"][mode]["ids"] for mode in MODES}
        entry["agreement"] = {
            "bm25_dense_jaccard": round(_jaccard(ids["bm25"], ids["dense"]), 3),
            "dense_hybrid_jaccard": round(_jaccard(ids["dense"], ids["hybrid"]), 3),
            "bm25_hybrid_jaccard": round(_jaccard(ids["bm25"], ids["hybrid"]), 3),
            "same_top1_dense_hybrid": bool(ids["dense"][:1] == ids["hybrid"][:1]),
            "same_top1_bm25_dense": bool(ids["bm25"][:1] == ids["dense"][:1]),
        }
        per_query.append(entry)
        tops = " | ".join(
            f"{mode}={entry['modes'][mode]['names'][0] if entry['modes'][mode]['names'] else 'none'}"
            for mode in MODES
        )
        print(f"{query_id} [{facet}] {tops}")

    summary: dict = {"modes": {}, "agreement_means": {}}
    for mode in MODES:
        walls = [q["modes"][mode]["wall_seconds"] for q in per_query]
        summary["modes"][mode] = {
            "mean_wall_seconds": round(sum(walls) / len(walls), 3),
            "max_wall_seconds": round(max(walls), 3),
            "queries_with_compliance_failure": sum(
                1 for q in per_query if q["modes"][mode]["compliance_failed"]
            ),
            "queries_with_empty_result": sum(
                1 for q in per_query if not q["modes"][mode]["ids"]
            ),
        }
    for key in per_query[0]["agreement"]:
        values = [q["agreement"][key] for q in per_query]
        summary["agreement_means"][key] = (
            round(sum(values) / len(values), 3)
            if isinstance(values[0], float)
            else sum(values)
        )

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_retrieval_E1E2E3"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E1/E2/E3",
        "chunk_config": name,
        "pool": args.pool,
        "k": args.k,
        "queries": per_query,
        "summary": summary,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "| Mode | Mean wall (s) | Max wall (s) | Compliance failures | Empty |",
        "|---|---|---|---|---|",
    ]
    labels = {"bm25": "E1 BM25", "dense": "E2 dense", "hybrid": "E3 hybrid"}
    for mode in MODES:
        row = summary["modes"][mode]
        lines.append(
            f"| {labels[mode]} | {row['mean_wall_seconds']} | {row['max_wall_seconds']} | "
            f"{row['queries_with_compliance_failure']}/12 | {row['queries_with_empty_result']}/12 |"
        )
    lines += [
        "",
        "| Agreement | Mean |",
        "|---|---|",
        f"| BM25 vs dense Jaccard | {summary['agreement_means']['bm25_dense_jaccard']} |",
        f"| Dense vs hybrid Jaccard | {summary['agreement_means']['dense_hybrid_jaccard']} |",
        f"| Same top-1 dense/hybrid | {summary['agreement_means']['same_top1_dense_hybrid']}/12 |",
        f"| Same top-1 BM25/dense | {summary['agreement_means']['same_top1_bm25_dense']}/12 |",
    ]
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "retrieval_E1E2E3_latest.md").write_text(table)

    print(f"\nsaved to {run_dir}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
