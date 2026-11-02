#!/usr/bin/env python
"""E8: FP16, INT8, INT4, and LoRA precision comparisons."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.llm.prompts import build_messages  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.constraints import Constraints  # noqa: E402
from dinewise.rag.pipeline import retrieve_with_config  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

CONFIGS = [
    ("M1_fp16", "fp16", False),
    ("M2_int8", "int8", False),
    ("M3_int4", "int4", False),
    ("M4_lora_fp16", "fp16", True),
    ("M5_lora_int4", "int4", True),
]


def load_model(model_name: str, precision: str, adapter: Path | None):
    kwargs: dict = {"device_map": "auto"}
    if precision == "fp16":
        kwargs["dtype"] = torch.float16
    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter", default="models/lora-qwen")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = pd.read_parquet(data_path(cfg, "index_dir", "faiss", name, "chunks.parquet"))
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)

    query = "Find Italian restaurants with vegetarian options in Philadelphia."
    constraints = Constraints(city="Philadelphia", cuisines=["Italian"])
    outcome = retrieve_with_config(
        retriever, bm25, chunks, restaurants, query,
        mode="hybrid", constraints=constraints, k=3,
    )
    messages = build_messages(query, outcome["hits"])
    adapter = ROOT / args.adapter if (ROOT / args.adapter).exists() else None

    rows = []
    for label, precision, use_adapter in CONFIGS:
        if use_adapter and adapter is None:
            print(f"skipping {label}: adapter missing")
            continue
        print(f"measuring {label} ...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["llm"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        started = time.perf_counter()
        model = load_model(cfg["models"]["llm"], precision, adapter if use_adapter else None)
        load_seconds = time.perf_counter() - started

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warm-up.
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

        timings = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                model(**inputs)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - prefill_start

            generate_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            torch.cuda.synchronize()
            total = time.perf_counter() - generate_start
            new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            timings.append(
                {
                    "ttft": prefill,
                    "total": total,
                    "tpot": (total - prefill) / max(new_tokens, 1),
                    "tokens_per_second": new_tokens / max(total, 1e-9),
                }
            )

        median = {
            key: sorted(t[key] for t in timings)[len(timings) // 2] for key in timings[0]
        }
        rows.append(
            {
                "config": label,
                "precision": precision,
                "adapter": bool(use_adapter),
                "load_seconds": round(load_seconds, 2),
                "ttft_ms": round(median["ttft"] * 1000, 1),
                "tpot_ms": round(median["tpot"] * 1000, 1),
                "tokens_per_second": round(median["tokens_per_second"], 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "weights_gb": round(model.get_memory_footprint() / 1e9, 2),
            }
        )
        print(rows[-1])
        del model
        gc.collect()
        torch.cuda.empty_cache()

    stamped = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "runs" / f"{stamped}_efficiency_E8"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps({"runs": args.runs, "rows": rows}, indent=2) + "\n"
    )
    columns = [
        "config", "precision", "adapter", "load_seconds", "ttft_ms",
        "tpot_ms", "tokens_per_second", "peak_vram_gb", "weights_gb",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table + "\n")
    (ROOT / "results" / "tables" / "efficiency_E8_latest.md").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
