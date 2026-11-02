#!/usr/bin/env python
"""E7: base vs RAG vs RAG+LoRA vs LoRA-only on the 15-query suite."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.eval import ALL_QUERIES  # noqa: E402
from dinewise.llm.guards import check_scope  # noqa: E402
from dinewise.llm.prompts import build_messages  # noqa: E402
from dinewise.llm.validators import validate_answer  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.constraints import extract_constraints  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

ID_PATTERN = re.compile(r"[A-Za-z0-9_\-]{22}")
LIVE_PATTERN = re.compile(
    r"\b(available tonight|open now|book a table|reservation|current price|real-?time)\b",
    re.IGNORECASE,
)

SYSTEMS = ["base_no_rag", "base_rag", "lora_rag", "lora_no_rag"]


def generate(model, tokenizer, messages, max_new_tokens: int = 220) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def score_answer(
    answer: str,
    hits: list[dict],
    restaurants: pd.DataFrame,
    constraints,
    out_of_scope: bool,
    used_rag: bool,
) -> dict:
    catalogue_ids = set(restaurants["business_id"])
    context_ids = {str(hit["chunk"]["business_id"]) for hit in hits}
    mentioned = set(ID_PATTERN.findall(answer))
    valid_ids = mentioned & catalogue_ids
    grounded = used_rag and mentioned.issubset(context_ids) and bool(mentioned)
    live = bool(LIVE_PATTERN.search(answer))
    refusal = any(
        phrase in answer.lower()
        for phrase in ("can't answer", "cannot answer", "no live", "not in the dataset", "does not record")
    )
    report = validate_answer(answer, hits, restaurants, constraints) if used_rag else None
    return {
        "format_valid": 40 <= len(answer) <= 4000,
        "ids_cited": len(mentioned),
        "ids_valid": len(valid_ids),
        "grounded": grounded,
        "live_claim": live,
        "refusal": refusal,
        "hard_pass": bool(report.passed) if report else False,
        "abstained_correctly": bool(refusal) if out_of_scope else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter", default="models/lora-qwen")
    parser.add_argument("--four-bit", dest="four_bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = pd.read_parquet(data_path(cfg, "index_dir", "faiss", name, "chunks.parquet"))
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)

    model_name = cfg["models"]["llm"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto"}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
    else:
        load_kwargs["dtype"] = torch.float16
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()
    load_seconds = time.perf_counter() - started

    adapter_path = ROOT / args.adapter
    lora_model = None
    if adapter_path.exists():
        lora_model = PeftModel.from_pretrained(base, str(adapter_path)).eval()
    else:
        print(f"WARNING: adapter not found at {adapter_path}; LoRA systems skipped")

    rows = []
    for query_id, facet, query in ALL_QUERIES:
        scope = check_scope(query)
        constraints = extract_constraints(
            query, known_cities=sorted(restaurants["city"].dropna().unique().tolist())
        )
        hits = []
        if scope.allowed:
            outcome = retrieve_with_config(
                retriever, bm25, chunks, restaurants, query,
                mode="hybrid", constraints=constraints, k=3,
            )
            hits = outcome["hits"]

        for system in SYSTEMS:
            if system.startswith("lora") and lora_model is None:
                continue
            used_rag = system in {"base_rag", "lora_rag"}
            if used_rag:
                messages = build_messages(query, hits)
            else:
                messages = [
                    {"role": "system", "content": "You are a restaurant recommendation assistant."},
                    {"role": "user", "content": query},
                ]

            model = lora_model if system.startswith("lora") else base
            started = time.perf_counter()
            answer = generate(model, tokenizer, messages, args.max_new_tokens)
            elapsed = time.perf_counter() - started
            scores = score_answer(
                answer, hits, restaurants, constraints, not scope.allowed, used_rag
            )
            rows.append(
                {
                    "query_id": query_id,
                    "facet": facet,
                    "system": system,
                    **scores,
                    "seconds": round(elapsed, 2),
                    "answer": answer,
                }
            )
            print(f"{query_id} {system} grounded={scores['grounded']} live={scores['live_claim']}")

    summary = {}
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        if not subset:
            continue
        scope_rows = [row for row in subset if row["abstained_correctly"] is not None]
        summary[system] = {
            "queries": len(subset),
            "format_valid": round(sum(r["format_valid"] for r in subset) / len(subset), 3),
            "grounded_rate": round(sum(r["grounded"] for r in subset) / len(subset), 3),
            "hard_pass_rate": round(sum(r["hard_pass"] for r in subset) / len(subset), 3),
            "live_claim_rate": round(sum(r["live_claim"] for r in subset) / len(subset), 3),
            "abstention_accuracy": round(
                sum(r["abstained_correctly"] for r in scope_rows) / len(scope_rows), 3
            ) if scope_rows else None,
            "mean_seconds": round(sum(r["seconds"] for r in subset) / len(subset), 2),
        }

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_models_E7"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E7",
        "load_seconds": round(load_seconds, 2),
        "four_bit": args.four_bit,
        "summary": summary,
        "results": rows,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "| System | Format valid | Grounded | Hard pass | Live claims | Abstention | Seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for system, row in summary.items():
        lines.append(
            f"| {system} | {row['format_valid']} | {row['grounded_rate']} | "
            f"{row['hard_pass_rate']} | {row['live_claim_rate']} | "
            f"{row['abstention_accuracy']} | {row['mean_seconds']} |"
        )
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table)
    (ROOT / "results" / "tables" / "models_E7_latest.md").write_text(table)
    print(json.dumps(summary, indent=2))
    print(table)

    del lora_model, base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
