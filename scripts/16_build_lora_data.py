#!/usr/bin/env python
"""Build grounded instruction data for LoRA fine-tuning.

Target answers are deterministic templates over retrieved records, not
teacher-LLM output. This teaches response behaviour and citation format,
not restaurant facts. Splits are by primary restaurant to prevent leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.llm.guards import REFUSAL  # noqa: E402
from dinewise.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.constraints import Constraints  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402
from dinewise.rag.vectorstore import load_chunks  # noqa: E402

CUISINES = [
    "Italian", "Mexican", "Chinese", "Japanese", "Thai", "Indian",
    "Mediterranean", "American", "Seafood", "BBQ", "French", "Vietnamese",
    "Greek", "Korean", "Spanish", "Steakhouse", "Bakery/Dessert",
]

TASKS = ["cuisine", "budget", "dietary", "ambience", "review", "comparison", "multi"]


def _price(value) -> str:
    try:
        bucket = int(value)
    except (TypeError, ValueError):
        return "price unknown"
    return "$" * bucket


def _record(hit: dict) -> str:
    row = hit["chunk"]
    return (
        f"[RESTAURANT {row['business_id']}]\n"
        f"Name: {row.get('restaurant_name')}\n"
        f"City: {row.get('city')}\n"
        f"Cuisine: {row.get('cuisine_primary')}\n"
        f"Price: {_price(row.get('price_range'))}\n"
        f"Rating: {row.get('stars')} (dataset value)\n"
        f"Review summary: {str(row.get('review_summary') or '')[:300]}"
    )


def _answer(hits: list[dict], reason: str) -> str:
    if not hits:
        return REFUSAL
    lines = []
    for index, hit in enumerate(hits[:3], start=1):
        row = hit["chunk"]
        lines.append(
            f"{index}. **{row.get('restaurant_name')}** (ID: {row['business_id']}) — "
            f"{row.get('cuisine_primary')}, {_price(row.get('price_range'))}, "
            f"rating {row.get('stars')} (dataset value). {reason}"
        )
    lines.append("Ratings and prices are from a static dataset, not live information.")
    return "\n".join(lines)


def _task_query(task: str, row: pd.Series, city: str, rng: random.Random) -> tuple[str, Constraints, str]:
    cuisine = row.get("cuisine_primary")
    if not isinstance(cuisine, str) or cuisine == "Unspecified":
        cuisine = rng.choice(CUISINES)
    price = row.get("price_range")
    bucket = int(price) if pd.notna(price) else 2

    if task == "cuisine":
        return (
            f"Find good {cuisine} restaurants in {city}.",
            Constraints(city=city, cuisines=[cuisine]),
            f"It matches the requested {cuisine} cuisine.",
        )
    if task == "budget":
        return (
            f"Recommend a cheap restaurant in {city} under {_price(bucket)}.",
            Constraints(city=city, max_price_bucket=max(bucket, 1)),
            "It fits the budget bucket from the dataset.",
        )
    if task == "dietary":
        return (
            f"I need vegetarian options in {city} under {_price(bucket)}.",
            Constraints(city=city, max_price_bucket=max(bucket, 1), dietary=["vegetarian"]),
            "Review evidence mentions vegetarian options; confirm before relying on it.",
        )
    if task == "ambience":
        return (
            f"Suggest a romantic restaurant for a date in {city}.",
            Constraints(city=city, required_attributes=["ambience_romantic"]),
            "Its ambience attributes support a date-night request.",
        )
    if task == "review":
        return (
            f"Which restaurants are praised for desserts in {city}?",
            Constraints(city=city),
            "Review summaries mention desserts or sweets.",
        )
    if task == "comparison":
        return (
            f"Compare {row.get('name')} with another {cuisine} restaurant in {city}.",
            Constraints(city=city, cuisines=[cuisine]),
            "It is one of the compared restaurants; evidence is limited to the summaries.",
        )
    return (
        f"Family-friendly {cuisine} restaurant in {city} under {_price(bucket)}.",
        Constraints(city=city, cuisines=[cuisine], max_price_bucket=max(bucket, 1)),
        "It satisfies the cuisine, city, and budget constraints; family suitability may be unrecorded.",
    )


def build_examples(n: int = 400, seed: int = 42) -> list[dict]:
    cfg = load_config(None)
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = load_chunks(cfg, name)
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)
    rng = random.Random(seed)

    cities = sorted(restaurants["city"].dropna().unique().tolist())
    n_scope = 40
    n_regular = max(n - n_scope, 1)
    sample = restaurants.sample(frac=1.0, random_state=seed).head(n_regular * 3)

    examples: list[dict] = []
    for index, (_, row) in enumerate(sample.iterrows()):
        if len(examples) >= n_regular:
            break
        task = TASKS[index % len(TASKS)]
        city = str(row.get("city") or rng.choice(cities))
        query, constraints, reason = _task_query(task, row, city, rng)

        outcome = retrieve_with_config(
            retriever, bm25, chunks, restaurants, query,
            mode="hybrid", constraints=constraints, k=3,
        )
        hits = outcome["hits"]
        if not hits:
            continue
        # Group by the restaurant the query was built from, so no queried
        # restaurant appears in more than one split.
        group = str(row["business_id"])

        context = "\n\n".join(_record(hit) for hit in hits)
        user = f"USER QUESTION:\n{query}\n\nRESTAURANT RECORDS:\n{context}"
        examples.append(
            {
                "task": task,
                "group_id": group,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": _answer(hits, reason)},
                ],
            }
        )

    for index in range(n_scope):
        city = rng.choice(cities)
        query = f"Is a restaurant in {city} available tonight at 8pm?"
        examples.append(
            {
                "task": "out_of_scope",
                "group_id": f"scope-{index}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": REFUSAL},
                ],
            }
        )
    return examples


def split_by_group(examples: list[dict], seed: int = 42) -> tuple[list, list, list]:
    groups = sorted({example["group_id"] for example in examples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = int(0.8 * len(groups))
    n_val = int(0.1 * len(groups))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    train = [e for e in examples if e["group_id"] in train_groups]
    val = [e for e in examples if e["group_id"] in val_groups]
    test = [e for e in examples if e["group_id"] not in train_groups | val_groups]
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--out", default="data/tuning")
    args = parser.parse_args()

    examples = build_examples(args.n)
    train, val, test = split_by_group(examples)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    summary = {
        "total": len(examples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "unique_groups": len({e["group_id"] for e in examples}),
        "tasks": pd.Series([e["task"] for e in examples]).value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
