#!/usr/bin/env python
"""E5: chunk-size comparison (C1-C4) with hybrid retrieval.

For each configuration this rebuilds chunks, FAISS, and BM25, then scores
the same 12 gold queries. Only the chunk configuration changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval.metrics import evaluate_ranking, mean_metrics  # noqa: E402
from dinewise.rag.bm25_index import build_bm25, save_bm25  # noqa: E402
from dinewise.rag.chunking import (  # noqa: E402
    TokenChunker,
    build_chunks,
    contamination_report,
)
from dinewise.rag.constraints import Constraints  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.search import Retriever  # noqa: E402


def build_config(
    cfg: dict,
    name: str,
    settings: dict,
    restaurants: pd.DataFrame,
    profiles_text: str,
    tokenizer,
    encoder: SentenceTransformer,
) -> dict:
    started = time.perf_counter()
    chunker = TokenChunker(
        tokenizer=tokenizer,
        size=int(settings["size"]),
        overlap=int(settings["overlap"]),
        min_size=int(settings["min_size"]),
    )
    chunks = build_chunks(restaurants, profiles_text, chunker, config_name=name)
    chunks.to_parquet(data_path(cfg, "processed_dir", f"chunks_{name}.parquet"), index=False)

    index_dir = data_path(cfg, "index_dir", "faiss", name)
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = encoder.encode(
        chunks["text"].tolist(),
        batch_size=int(cfg["embedding"]["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    chunks.to_parquet(index_dir / "chunks.parquet", index=False)
    (index_dir / "index_meta.json").write_text(
        json.dumps(
            {"model": cfg["embedding"]["model"], "chunk_config": name,
             "chunks": len(chunks), "dimension": int(embeddings.shape[1])},
            indent=2,
        )
        + "\n"
    )

    bm25 = build_bm25(chunks)
    save_bm25(bm25, data_path(cfg, "index_dir", "bm25", name))
    report = contamination_report(chunks)
    report.update(
        {
            "config": name,
            "size": settings["size"],
            "overlap": settings["overlap"],
            "mean_tokens": round(float(chunks["token_count"].mean()), 1),
            "index_bytes": (index_dir / "index.faiss").stat().st_size,
            "seconds": round(time.perf_counter() - started, 2),
        }
    )
    return {"chunks": chunks, "index": index, "bm25": bm25, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    profiles = data_path(cfg, "docs_dir", cfg["docs"]["profile_file"])
    profiles_text = profiles.read_text(encoding="utf-8")

    gold = [
        json.loads(line)
        for line in (ROOT / "data" / "gold" / "goldset_v1.jsonl").read_text().splitlines()
        if line
    ]

    encoder = SentenceTransformer(
        cfg["embedding"]["model"], device=cfg["embedding"]["device"]
    )
    encoder.max_seq_length = int(cfg["embedding"]["max_seq_length"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["embedding"]["model"])

    rows = []
    details: dict[str, list[dict]] = {}
    for name, settings in cfg["chunking"]["configs"].items():
        print(f"building {name} ({settings['size']}/{settings['overlap']}) ...")
        built = build_config(
            cfg, name, settings, restaurants, profiles_text, tokenizer, encoder
        )
        chunks = built["chunks"]
        retriever = Retriever(
            model=encoder,
            index=built["index"],
            chunks=chunks,
            prefix=cfg["embedding"]["query_prefix"],
        )
        metrics = []
        for record in gold:
            constraints = Constraints(**record["constraints"])
            outcome = retrieve_with_config(
                retriever, built["bm25"], chunks, restaurants, record["query"],
                mode="hybrid", constraints=constraints, k=5,
            )
            ranked = [str(hit["chunk"]["business_id"]) for hit in outcome["hits"]]
            metrics.append(evaluate_ranking(ranked, record["gold"], k=5))
        summary = mean_metrics(metrics)
        details[name] = metrics
        row = {**built["report"], **summary}
        rows.append(row)
        print(
            f"  chunks={row['chunks']} mean_tokens={row['mean_tokens']} "
            f"ndcg={row['ndcg@k']:.3f} strict_success={row['strict_success@k']:.3f}"
        )

    table = pd.DataFrame(rows)
    best = table.sort_values("ndcg@k", ascending=False).iloc[0]

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_chunking_E5"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E5",
        "retrieval_mode": "hybrid",
        "label_source": "silver_rubric_v1",
        "rows": rows,
        "best_config": str(best["config"]),
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    columns = [
        "config", "size", "overlap", "chunks", "mean_tokens",
        "chunks_with_multiple_markers", "strict_success@k", "strict_precision@k",
        "strict_mrr", "ndcg@k", "index_bytes", "seconds",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                f"{row[column]:.3f}" if isinstance(row[column], float) else str(row[column])
                for column in columns
            ) + " |"
        )
    table_md = "\n".join(lines) + f"\n\nSelected: **{best['config']}** by nDCG@5.\n"
    (run_dir / "table.md").write_text(table_md)
    (ROOT / "results" / "tables" / "chunking_E5_latest.md").write_text(table_md)
    print(f"\nsaved to {run_dir}")
    print(table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
