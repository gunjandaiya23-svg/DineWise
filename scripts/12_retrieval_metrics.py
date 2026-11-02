#!/usr/bin/env python
"""Compute Success@5, Precision@5, MRR, and nDCG@5 for retrieval modes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval.metrics import evaluate_ranking, mean_metrics  # noqa: E402
from dinewise.llm.validators import check_retrieved_compliance  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.constraints import Constraints  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.rerank import Reranker  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_quality"]


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
    print("Loading cross-encoder reranker ...")
    reranker = Reranker(cfg["models"]["reranker"], device=cfg["embedding"]["device"])

    gold_path = ROOT / "data" / "gold" / "goldset_v1.jsonl"
    if not gold_path.exists():
        raise SystemExit("missing gold set; run scripts/11_build_goldset.py first")
    gold_records = [json.loads(line) for line in gold_path.read_text().splitlines() if line]

    per_query: list[dict] = []
    mode_rows: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for record in gold_records:
        constraints = Constraints(**record["constraints"])
        entry = {
            "query_id": record["query_id"],
            "facet": record["facet"],
            "gold": record["gold"],
            "modes": {},
        }
        for mode in MODES:
            outcome = retrieve_with_config(
                retriever, bm25, chunks, restaurants, record["query"],
                mode=mode, constraints=constraints, pool=args.pool, k=args.k,
                reranker=reranker if mode.startswith("hybrid_rerank") else None,
            )
            ranked = [str(hit["chunk"]["business_id"]) for hit in outcome["hits"]]
            metrics = evaluate_ranking(ranked, record["gold"], k=args.k)
            compliance = check_retrieved_compliance(
                outcome["hits"], restaurants, constraints
            )
            entry["modes"][mode] = {
                "ranked": ranked,
                "names": [str(h.get("restaurant_name")) for h in outcome["hits"]],
                **{key: round(value, 4) for key, value in metrics.items()},
                "compliance_failed": [c.name for c in compliance if not c.passed],
            }
            mode_rows[mode].append(metrics)
        per_query.append(entry)
        print(
            f"{record['query_id']} "
            + " ".join(
                f"{mode}:ndcg={entry['modes'][mode]['ndcg@k']:.2f}"
                for mode in MODES
            )
        )

    summary = {
        "label_source": "silver_rubric_v1",
        "needs_manual_review": True,
        "chunk_config": name,
        "k": args.k,
        "modes": {mode: mean_metrics(rows) for mode, rows in mode_rows.items()},
    }

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_retrieval_metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps({"summary": summary, "queries": per_query}, indent=2) + "\n"
    )

    lines = [
        "| Mode | Strict Success@5 | Strict P@5 | Strict MRR | nDCG@5 | Any Success@5 |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        row = summary["modes"][mode]
        lines.append(
            f"| {mode} | {row['strict_success@k']:.3f} | {row['strict_precision@k']:.3f} | "
            f"{row['strict_mrr']:.3f} | {row['ndcg@k']:.3f} | {row['any_success@k']:.3f} |"
        )
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "retrieval_metrics_latest.md").write_text(table)

    print(f"\nsaved to {run_dir}")
    print(table)
    print("NOTE: labels are silver (rubric), not human judgments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
