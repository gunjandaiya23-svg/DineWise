#!/usr/bin/env python
"""E12: Qdrant exact search vs HNSW approximate search.

Both modes query the same collection and the same vectors. Recall@10 is
measured against exact search; latency is measured per query.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import load_config  # noqa: E402
from dinewise.eval.metrics import evaluate_ranking  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402
from dinewise.rag.vectorstore import store_from_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["chunking"]["active"]
    retriever = load_retriever(cfg, name)
    store_exact = store_from_config(cfg)
    store_exact.exact = True
    store_hnsw = store_from_config(cfg)
    store_hnsw.exact = False
    store_hnsw.hnsw_ef = int(cfg["vector_store"].get("hnsw_ef", 128))

    gold = [
        json.loads(line)
        for line in (ROOT / "data" / "gold" / "goldset_v1.jsonl").read_text().splitlines()
        if line
    ]

    rows = []
    for record in gold:
        vector = retriever.model.encode(
            [retriever.prefix + record["query"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        exact_times, hnsw_times = [], []
        exact_hits = hnsw_hits = []
        for _ in range(args.runs):
            started = time.perf_counter()
            exact_hits = store_exact.search(f"chunks_{name}", vector, args.k)
            exact_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            hnsw_hits = store_hnsw.search(f"chunks_{name}", vector, args.k)
            hnsw_times.append(time.perf_counter() - started)

        exact_ids = [hit.chunk_id for hit in exact_hits]
        hnsw_ids = [hit.chunk_id for hit in hnsw_hits]
        recall = (
            len(set(exact_ids) & set(hnsw_ids)) / len(exact_ids) if exact_ids else 0.0
        )
        ranked_business = [
            str(hit.payload.get("business_id")) for hit in hnsw_hits[:5]
        ]
        metrics = evaluate_ranking(ranked_business, record["gold"], k=5)
        rows.append(
            {
                "query_id": record["query_id"],
                "recall@10": round(recall, 4),
                "ndcg@5": round(metrics["ndcg@k"], 4),
                "exact_ms": round(sorted(exact_times)[len(exact_times) // 2] * 1000, 2),
                "hnsw_ms": round(sorted(hnsw_times)[len(hnsw_times) // 2] * 1000, 2),
            }
        )
        print(
            f"{record['query_id']} recall={recall:.2f} "
            f"exact={rows[-1]['exact_ms']}ms hnsw={rows[-1]['hnsw_ms']}ms"
        )

    summary = {
        "collection": f"dinewise_chunks_{name}",
        "points": store_exact.count(f"chunks_{name}"),
        "k": args.k,
        "runs": args.runs,
        "hnsw_ef": store_hnsw.hnsw_ef,
        "mean_recall@10": round(sum(r["recall@10"] for r in rows) / len(rows), 4),
        "mean_ndcg@5": round(sum(r["ndcg@5"] for r in rows) / len(rows), 4),
        "mean_exact_ms": round(sum(r["exact_ms"] for r in rows) / len(rows), 2),
        "mean_hnsw_ms": round(sum(r["hnsw_ms"] for r in rows) / len(rows), 2),
    }

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_backend_E12"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps({"summary": summary, "queries": rows}, indent=2) + "\n"
    )

    table = "\n".join(
        [
            "| Metric | Value |",
            "|---|---|",
            f"| Collection | {summary['collection']} |",
            f"| Points | {summary['points']} |",
            f"| Mean Recall@10 vs exact | {summary['mean_recall@10']:.3f} |",
            f"| Mean nDCG@5 (HNSW) | {summary['mean_ndcg@5']:.3f} |",
            f"| Mean exact latency | {summary['mean_exact_ms']:.2f} ms |",
            f"| Mean HNSW latency | {summary['mean_hnsw_ms']:.2f} ms |",
        ]
    ) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables" / "backend_E12_latest.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
