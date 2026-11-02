"""Answer synthesis: Qwen generation or a deterministic grounded template."""

from __future__ import annotations

from typing import Any

import torch

from .prompts import build_messages


class Generator:
    """Lazy Qwen generator with a no-LLM template for graph testing."""

    def __init__(self, cfg: dict[str, Any], use_llm: bool = True, four_bit: bool = False):
        self.cfg = cfg
        self.use_llm = use_llm
        self.four_bit = four_bit
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = self.cfg["models"]["llm"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.four_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["dtype"] = torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self._model.eval()

    def generate(self, query: str, hits: list[dict[str, Any]]) -> str:
        if not self.use_llm:
            return self._template(query, hits)
        self._load()
        assert self._tokenizer is not None and self._model is not None
        messages = build_messages(query, hits)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
                do_sample=False,
            )
        return self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    @staticmethod
    def _template(query: str, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "No restaurant records satisfy the request."
        lines = [
            "Based on the dataset records, these restaurants match your request:",
            "",
        ]
        for index, hit in enumerate(hits[:3], start=1):
            row = hit["chunk"]
            price = row.get("price_range")
            price_text = f"{'$' * int(price)}" if price is not None else "price unknown"
            lines.append(
                f"{index}. **{row.get('restaurant_name')}** (ID: {row['business_id']}) — "
                f"{row.get('cuisine_primary')}, {price_text}, "
                f"rating {row.get('stars')} (dataset value)."
            )
            summary = str(row.get("review_summary") or "").strip()
            if summary:
                lines.append(f"   Evidence: {summary[:220]}")
        lines.append("")
        lines.append("Ratings and prices are dataset values, not live information.")
        return "\n".join(lines)
