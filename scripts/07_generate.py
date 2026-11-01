#!/usr/bin/env python
"""Retrieve context and generate a grounded recommendation with Qwen."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.llm.guards import REFUSAL, check_scope  # noqa: E402
from dinewise.llm.prompts import build_context, build_messages  # noqa: E402
from dinewise.llm.validators import validate_answer  # noqa: E402
from dinewise.rag.pipeline import retrieve_filtered  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--config-name", default=None, help="chunking config, e.g. C2")
    parser.add_argument("--k", type=int, default=5, help="restaurants in context")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--four-bit", dest="four_bit", action="store_true", help="load the model in 4-bit"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_new_tokens = args.max_new_tokens or int(cfg["generation"]["max_new_tokens"])

    scope = check_scope(args.query)
    if not scope.allowed:
        print(f"\nOut of scope: {scope.reason}\n")
        print(REFUSAL)
        return 0

    started = time.perf_counter()
    retriever = load_retriever(cfg, args.config_name)
    restaurants = pd.read_parquet(
        data_path(cfg, "processed_dir", "restaurants.parquet")
    )
    outcome = retrieve_filtered(retriever, restaurants, args.query, k=args.k)
    retrieval_seconds = time.perf_counter() - started
    constraints = outcome["constraints"]

    if outcome.get("refused"):
        print(f"\nCannot answer: {outcome['reason']}\n")
        print(
            "I can still help with restaurants in the covered cities by cuisine, "
            "budget bucket, rating, or review evidence."
        )
        return 0

    hits = outcome["hits"]
    if not hits:
        print("No restaurant records satisfy all constraints. Refusing to answer.")
        print(f"Filter trace: {outcome['filter_counts']}")
        return 0

    print(f"Constraints: city={constraints.city} cuisines={constraints.cuisines} "
          f"max_price={constraints.max_price_bucket} attrs={constraints.required_attributes}")
    print(f"Candidates after filter: {outcome['candidates_after_filter']} "
          f"(skipped {outcome['skipped_by_filter']} non-matching)")

    model_name = cfg["models"]["llm"]
    print(f"Loading {model_name} (four_bit={args.four_bit}) ...")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {"device_map": "auto"}
    if args.four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    load_seconds = time.perf_counter() - started

    messages = build_messages(args.query, hits)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generation_seconds = time.perf_counter() - started
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    print("\n" + "=" * 72)
    print(answer)
    print("=" * 72)
    print("\nSources:")
    for hit in hits:
        row = hit["chunk"]
        print(f"  - {row['restaurant_name']} ({row['business_id']})")

    validation = validate_answer(answer, hits, restaurants, constraints)
    if validation.passed:
        print("\nValidation: PASSED (all hard checks)")
    else:
        print("\nValidation: FAILED")
        for check in validation.hard_failures:
            print(f"  - {check.name}: {check.detail}")
    for check in validation.soft_failures:
        print(f"  (soft) {check.name}: {check.detail}")

    print("\nTimings:")
    print(f"  retrieval: {retrieval_seconds:.2f}s")
    print(f"  model load: {load_seconds:.2f}s")
    print(f"  generation: {generation_seconds:.2f}s")
    print(f"  context chars: {len(build_context(hits))}")
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
