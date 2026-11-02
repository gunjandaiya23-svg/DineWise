#!/usr/bin/env python
"""E6: text-to-visual retrieval plus CLIP zero-shot food classification.

The dataset has no image files, so this experiment does two honest things:
1. retrieves restaurants from GPT-4o visual summaries using BGE text search;
2. evaluates CLIP zero-shot Food-101 classification on streamed images and
   maps predicted food classes to cuisine families as a soft signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval.metrics import evaluate_ranking, mean_metrics  # noqa: E402
from dinewise.image.clip import CUISINE_MAP, CLIPZeroShot  # noqa: E402
from dinewise.image.visual_index import build_visual_index  # noqa: E402

VISUAL_QUERIES: list[tuple[str, str, list[str]]] = [
    ("V01", "romantic interior for a date", ["romantic", "intimate", "candle", "cozy", "dim"]),
    ("V02", "outdoor patio seating", ["outdoor", "patio", "terrace", "sidewalk", "garden"]),
    ("V03", "bright modern decor", ["bright", "modern", "contemporary", "minimalist"]),
    ("V04", "cozy atmosphere", ["cozy", "warm", "intimate", "rustic"]),
    ("V05", "industrial style interior", ["industrial", "exposed", "brick", "warehouse"]),
    ("V06", "elegant fine dining room", ["elegant", "upscale", "fine dining", "classy"]),
    ("V07", "casual family friendly space", ["casual", "family", "relaxed", "kid"]),
    ("V08", "lively bar area", ["bar", "lively", "energetic", "cocktail"]),
    ("V09", "colorful interior", ["colorful", "vibrant", "bold", "bright"]),
    ("V10", "rustic decor", ["rustic", "wood", "farmhouse", "vintage"]),
]


def visual_text(row: pd.Series) -> str:
    columns = [
        "photo_summary", "venue_style", "indoor_color_tone",
        "special_features", "target_audience", "food_style",
    ]
    return " ".join(str(row.get(column) or "") for column in columns).lower()


def build_visual_gold(rows: pd.DataFrame, terms: list[str]) -> list[dict]:
    scored = []
    for _, row in rows.iterrows():
        text = visual_text(row)
        hits = sum(1 for term in terms if term in text)
        scored.append((hits, float(row.get("stars") or 0), str(row["business_id"]), row["name"]))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    top_ids = {item[2] for item in scored[:10]}
    return [
        {
            "business_id": item[2],
            "name": item[3],
            "grade": 2 if item[2] in top_ids else 1,
            "reason": "term match" if item[0] else "constraint only",
        }
        for item in scored
    ]


def run_clip_eval(cfg: dict, per_class: int = 3) -> dict:
    try:
        from datasets import load_dataset, load_dataset_builder
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "error": str(exc)}

    builder = load_dataset_builder("ethz/food101")
    classes = list(builder.info.features["label"].names)

    stream = load_dataset("ethz/food101", split="validation", streaming=True)
    counts: dict[int, int] = {}
    images = []
    labels = []
    for example in stream:
        label = int(example["label"])
        if counts.get(label, 0) >= per_class:
            if len(counts) == len(classes) or len(images) >= per_class * len(classes):
                break
            continue
        counts[label] = counts.get(label, 0) + 1
        images.append(example["image"].convert("RGB"))
        labels.append(label)
        if len(images) >= per_class * len(classes):
            break

    model = CLIPZeroShot(cfg["models"]["clip"], classes, device=cfg["embedding"]["device"])
    top1 = top5 = 0
    mapped = mapped_correct = 0
    for start in range(0, len(images), 16):
        batch = images[start : start + 16]
        batch_labels = labels[start : start + 16]
        predictions = model.classify(batch, top_k=5)
        for label, preds in zip(batch_labels, predictions, strict=False):
            names = [pred["class"] for pred in preds]
            true_name = classes[label]
            top1 += int(names[0] == true_name)
            top5 += int(true_name in names)
            true_cuisine = CUISINE_MAP.get(true_name)
            if true_cuisine:
                mapped += 1
                mapped_correct += int(CUISINE_MAP.get(names[0]) == true_cuisine)

    n = len(images)
    return {
        "status": "ok",
        "images": n,
        "classes_seen": len(counts),
        "top1_accuracy": round(top1 / n, 4) if n else 0.0,
        "top5_accuracy": round(top5 / n, 4) if n else 0.0,
        "mapped_classes": len(CUISINE_MAP),
        "mapped_samples": mapped,
        "mapped_cuisine_accuracy": round(mapped_correct / mapped, 4) if mapped else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-per-class", type=int, default=3)
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    model = SentenceTransformer(
        cfg["embedding"]["model"], device=cfg["embedding"]["device"]
    )
    model.max_seq_length = int(cfg["embedding"]["max_seq_length"])

    started = time.perf_counter()
    retriever = build_visual_index(restaurants, model)
    build_seconds = time.perf_counter() - started
    rows = retriever.rows

    per_query = []
    metric_rows = []
    for query_id, query, terms in VISUAL_QUERIES:
        gold = build_visual_gold(rows, terms)
        hits = retriever.search(query, k=5)
        ranked = [str(hit["row"]["business_id"]) for hit in hits]
        metrics = evaluate_ranking(ranked, gold, k=5)
        metric_rows.append(metrics)
        per_query.append(
            {
                "query_id": query_id,
                "query": query,
                "terms": terms,
                "ranked": ranked,
                "names": [hit["row"]["name"] for hit in hits],
                **{key: round(value, 4) for key, value in metrics.items()},
            }
        )
        print(f"{query_id} {query} -> {per_query[-1]['names']}")

    summary = {
        "visual_documents": int(len(rows)),
        "build_seconds": round(build_seconds, 2),
        "metrics": mean_metrics(metric_rows),
        "label_source": "silver_visual_terms_v1",
    }

    clip_result = {"status": "skipped"}
    if not args.skip_clip:
        print("Running CLIP zero-shot evaluation ...")
        clip_result = run_clip_eval(cfg, per_class=args.clip_per_class)
        print(json.dumps(clip_result, indent=2))

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_visual_E6"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {"summary": summary, "clip": clip_result, "queries": per_query}, indent=2
        )
        + "\n"
    )

    metrics = summary["metrics"]
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Visual documents | {summary['visual_documents']} |",
        f"| Strict Success@5 | {metrics['strict_success@k']:.3f} |",
        f"| Strict Precision@5 | {metrics['strict_precision@k']:.3f} |",
        f"| Strict MRR | {metrics['strict_mrr']:.3f} |",
        f"| nDCG@5 | {metrics['ndcg@k']:.3f} |",
        f"| CLIP images | {clip_result.get('images', 0)} |",
        f"| CLIP top-1 | {clip_result.get('top1_accuracy')} |",
        f"| CLIP top-5 | {clip_result.get('top5_accuracy')} |",
        f"| Mapped cuisine accuracy | {clip_result.get('mapped_cuisine_accuracy')} |",
    ]
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables" / "visual_E6_latest.md").write_text(table)
    print(f"\nsaved to {run_dir}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
