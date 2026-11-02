"""Filtered retrieval: dense top-N, then restrict to constraint-satisfying IDs.

Also provides BM25-only and hybrid (RRF) variants so the experiment
scripts can compare E1 (BM25), E2 (dense), and E3 (hybrid) on one path.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .bm25_index import BM25Index
from .constraints import Constraints, extract_constraints
from .filters import apply_filters
from .fusion import collapse_to_restaurants, rrf_fuse
from .rerank import quality_prior
from .search import Retriever
from .vectorstore import to_qdrant_filter


def retrieve_filtered(
    retriever: Retriever,
    catalogue: pd.DataFrame,
    query: str,
    constraints: Constraints | None = None,
    pool: int = 30,
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve, filter, and rank. Returns hits plus diagnostics."""
    constraints = constraints or extract_constraints(
        query, known_cities=sorted(catalogue["city"].dropna().unique().tolist())
    )
    if constraints.unknown_city:
        return {
            "constraints": constraints,
            "refused": True,
            "reason": (
                f"'{constraints.unknown_city}' is not in the dataset, which covers "
                f"{', '.join(sorted(catalogue['city'].dropna().unique().tolist()))}."
            ),
            "filter_counts": {"start": len(catalogue)},
            "candidates_after_filter": 0,
            "skipped_by_filter": 0,
            "hits": [],
        }
    if constraints.unsupported_terms:
        return {
            "constraints": constraints,
            "refused": True,
            "reason": (
                "The dataset does not record information about "
                f"{', '.join(constraints.unsupported_terms)}, so this cannot be answered."
            ),
            "filter_counts": {"start": len(catalogue)},
            "candidates_after_filter": 0,
            "skipped_by_filter": 0,
            "hits": [],
        }
    allowed, filter_counts = apply_filters(catalogue, constraints)
    allowed_ids = set(allowed["business_id"])

    hits = retriever.search(query, k=pool, query_filter=to_qdrant_filter(constraints))
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for hit in hits:
        business_id = str(hit["chunk"]["business_id"])
        if business_id in seen:
            continue
        if business_id not in allowed_ids:
            skipped += 1
            continue
        seen.add(business_id)
        ranked.append(hit)
        if len(ranked) >= k:
            break

    return {
        "constraints": constraints,
        "refused": False,
        "filter_counts": filter_counts,
        "candidates_after_filter": len(allowed),
        "skipped_by_filter": skipped,
        "hits": ranked,
    }


def scope_refusal(constraints: Constraints, catalogue: pd.DataFrame) -> str | None:
    """Return a refusal reason for unknown places or unrecorded amenities."""
    if constraints.unknown_city:
        return (
            f"'{constraints.unknown_city}' is not in the dataset, which covers "
            f"{', '.join(sorted(catalogue['city'].dropna().unique().tolist()))}."
        )
    if constraints.unsupported_terms:
        return (
            "The dataset does not record information about "
            f"{', '.join(constraints.unsupported_terms)}, so this cannot be answered."
        )
    return None


def _restrict_to_allowed(
    chunk_ranking: list[str],
    chunk_table: pd.DataFrame,
    allowed_ids: set[str],
    k: int,
) -> list[dict[str, Any]]:
    """Collapse a chunk ranking to allowed restaurants, keeping order."""
    table = chunk_table.set_index("chunk_id")
    seen: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for chunk_id in chunk_ranking:
        if chunk_id not in table.index:
            continue
        row = table.loc[chunk_id]
        business_id = str(row["business_id"])
        if business_id in seen or business_id not in allowed_ids:
            continue
        seen.add(business_id)
        chunk = row.to_dict()
        chunk["chunk_id"] = chunk_id
        ranked.append({"score": 0.0, "chunk": chunk})
        if len(ranked) >= k:
            break
    return ranked


def retrieve_with_config(
    retriever: Retriever,
    bm25: BM25Index,
    chunks: pd.DataFrame,
    catalogue: pd.DataFrame,
    query: str,
    mode: str,
    constraints: Constraints | None = None,
    pool: int = 30,
    k: int = 5,
    rrf_k: int = 60,
    reranker: Any | None = None,
    rerank_pool: int = 20,
) -> dict[str, Any]:
    """Run one retrieval configuration with per-stage timings.

    mode is one of "dense" (E2), "bm25" (E1), "hybrid" (E3, RRF fusion),
    "hybrid_rerank" (E4), or "hybrid_rerank_quality" (E4 + quality prior).
    Returns the same diagnostics shape as retrieve_filtered plus timings.
    """
    constraints = constraints or extract_constraints(
        query, known_cities=sorted(catalogue["city"].dropna().unique().tolist())
    )
    refusal = scope_refusal(constraints, catalogue)
    if refusal:
        return {
            "mode": mode,
            "constraints": constraints,
            "refused": True,
            "reason": refusal,
            "filter_counts": {"start": len(catalogue)},
            "candidates_after_filter": 0,
            "timings": {},
            "hits": [],
        }
    allowed, filter_counts = apply_filters(catalogue, constraints)
    allowed_ids = set(allowed["business_id"])
    timings: dict[str, float] = {}

    if mode == "dense":
        started = time.perf_counter()
        chunk_hits = retriever.search(query, k=pool, query_filter=to_qdrant_filter(constraints))
        timings["search"] = time.perf_counter() - started
        ranking = [str(hit["chunk"]["chunk_id"]) for hit in chunk_hits]
        hits = _restrict_to_allowed(ranking, chunks, allowed_ids, k)
    elif mode == "bm25":
        started = time.perf_counter()
        chunk_hits = bm25.search(query, k=pool)
        timings["search"] = time.perf_counter() - started
        ranking = [hit["chunk_id"] for hit in chunk_hits]
        hits = _restrict_to_allowed(ranking, chunks, allowed_ids, k)
    elif mode == "hybrid":
        started = time.perf_counter()
        dense_hits = retriever.search(query, k=pool, query_filter=to_qdrant_filter(constraints))
        timings["dense"] = time.perf_counter() - started
        started = time.perf_counter()
        lexical_hits = bm25.search(query, k=pool)
        timings["bm25"] = time.perf_counter() - started
        started = time.perf_counter()
        fused = rrf_fuse(
            [
                [str(hit["chunk"]["chunk_id"]) for hit in dense_hits],
                [hit["chunk_id"] for hit in lexical_hits],
            ],
            k=rrf_k,
        )
        # Drop fused chunks from filtered-out restaurants before collapsing.
        table = chunks.set_index("chunk_id")
        allowed_chunks = [
            (chunk_id, score)
            for chunk_id, score in fused
            if chunk_id in table.index
            and str(table.loc[chunk_id]["business_id"]) in allowed_ids
        ]
        hits = collapse_to_restaurants(allowed_chunks, chunks, top_k=k)
        timings["fusion"] = time.perf_counter() - started
    elif mode in {"hybrid_rerank", "hybrid_rerank_quality"}:
        started = time.perf_counter()
        dense_hits = retriever.search(query, k=pool, query_filter=to_qdrant_filter(constraints))
        timings["dense"] = time.perf_counter() - started
        started = time.perf_counter()
        lexical_hits = bm25.search(query, k=pool)
        timings["bm25"] = time.perf_counter() - started
        started = time.perf_counter()
        fused = rrf_fuse(
            [
                [str(hit["chunk"]["chunk_id"]) for hit in dense_hits],
                [hit["chunk_id"] for hit in lexical_hits],
            ],
            k=rrf_k,
        )
        table = chunks.set_index("chunk_id")
        allowed_chunks = [
            (chunk_id, score)
            for chunk_id, score in fused
            if chunk_id in table.index
            and str(table.loc[chunk_id]["business_id"]) in allowed_ids
        ]
        timings["fusion"] = time.perf_counter() - started

        if reranker is None:
            raise ValueError(f"mode {mode} requires a reranker")

        started = time.perf_counter()
        shortlist = allowed_chunks[:rerank_pool]
        texts = [str(table.loc[chunk_id]["text"]) for chunk_id, _ in shortlist]
        scores = reranker.score(query, texts)
        reranked = sorted(
            zip([chunk_id for chunk_id, _ in shortlist], scores, strict=False),
            key=lambda item: (-item[1], item[0]),
        )
        hits = collapse_to_restaurants(
            reranked, chunks, top_k=k * 2 if mode.endswith("quality") else k
        )
        timings["rerank"] = time.perf_counter() - started

        if mode.endswith("quality"):
            started = time.perf_counter()
            hits = quality_prior(hits, catalogue)[:k]
            timings["quality_prior"] = time.perf_counter() - started
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")

    return {
        "mode": mode,
        "constraints": constraints,
        "refused": False,
        "filter_counts": filter_counts,
        "candidates_after_filter": len(allowed),
        "timings": timings,
        "hits": hits,
    }
