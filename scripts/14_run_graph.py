#!/usr/bin/env python
"""E9: run the full LangGraph workflow over the 15-query suite."""

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
from dinewise.graph.workflow import build_graph, run_query  # noqa: E402
from dinewise.llm.generator import Generator  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402
from dinewise.rag.vectorstore import load_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--llm", action="store_true", help="use Qwen instead of the template")
    parser.add_argument("--four-bit", dest="four_bit", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = load_chunks(cfg, name)
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)
    generator = Generator(cfg, use_llm=args.llm, four_bit=args.four_bit)
    graph = build_graph(cfg, retriever, bm25, chunks, restaurants, generator)

    results = []
    for query_id, facet, query in ALL_QUERIES:
        started = time.perf_counter()
        state = run_query(graph, query, profile={"city": "Philadelphia"})
        elapsed = time.perf_counter() - started
        results.append(
            {
                "query_id": query_id,
                "facet": facet,
                "query": query,
                "answer": state.get("answer", ""),
                "passed": bool(state.get("passed", False)),
                "fallback": bool(state.get("fallback_used", False)),
                "retry_count": int(state.get("retry_count", 0)),
                "out_of_scope": bool(state.get("out_of_scope", False)),
                "refused": bool(state.get("refused", False)),
                "n_hits": len(state.get("hits", [])),
                "validation": state.get("validation", {}),
                "trace": state.get("trace", []),
                "seconds": round(elapsed, 2),
            }
        )
        label = (
            "refused"
            if results[-1]["out_of_scope"]
            else ("fallback" if results[-1]["fallback"] else "answered")
        )
        print(
            f"{query_id} [{facet}] {label} "
            f"passed={results[-1]['passed']} retries={results[-1]['retry_count']} "
            f"hits={results[-1]['n_hits']} {elapsed:.2f}s"
        )

    summary = {
        "mode": "llm" if args.llm else "template",
        "queries": len(results),
        "answered": sum(
            1 for r in results if not r["fallback"] and not r["out_of_scope"]
        ),
        "refused": sum(1 for r in results if r["out_of_scope"]),
        "fallbacks": sum(1 for r in results if r["fallback"]),
        "validation_passed": sum(1 for r in results if r["passed"]),
        "retries": sum(1 for r in results if r["retry_count"] > 0),
        "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2),
    }

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_graph_E9"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n"
    )

    lines = [
        "| ID | Facet | Outcome | Passed | Retries | Hits | Seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        outcome = (
            "refused"
            if result["out_of_scope"]
            else ("fallback" if result["fallback"] else "answered")
        )
        lines.append(
            f"| {result['query_id']} | {result['facet']} | {outcome} | "
            f"{result['passed']} | {result['retry_count']} | {result['n_hits']} | "
            f"{result['seconds']} |"
        )
    table = "\n".join(lines) + "\n\n" + json.dumps(summary, indent=2) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables" / "graph_E9_latest.md").write_text(table)
    print(f"\nsaved to {run_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
