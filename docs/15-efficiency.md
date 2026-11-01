# Efficiency experiments

> Part of the DineWise project plan. FP16, INT8, INT4, LoRA-4bit, measurement protocol, interpretation.
> See [the plan index](README.md) for the full reading order.

---

## 15. Efficiency experiments

### 15.1 Configurations

| ID | Configuration | Purpose |
|---|---|---|
| M1 | FP16 | Baseline quality and speed |
| M2 | INT8 (LLM.int8) | Memory reduction with minimal quality loss |
| M3 | INT4 (NF4, double quant) | Maximum memory reduction |
| M4 | LoRA adapter (FP16 base) | Adaptation cost |
| M5 | LoRA + INT4 | Deployed configuration |

### 15.2 Measurement protocol

- Same prompt set: 20 fixed prompts (queries plus context), same token budget.
- Warm-up: 3 runs discarded.
- Measurement: 5 runs per prompt; report median and IQR.
- Decoding: greedy, `max_new_tokens=256`, fixed stop tokens.
- Batch size 1, because the application is single-user.
- Record:
  - model disk size,
  - peak GPU memory (`torch.cuda.max_memory_allocated` and `nvidia-smi`),
  - TTFT (time to first token),
  - TPOT (time per output token),
  - tokens/second,
  - end-to-end latency,
  - output quality on the gold set.

### 15.3 Interpretation rules

- Do not claim that quantization is uniformly faster. At batch size 1, 4-bit may be slower because of dequantization overhead.
- Do not compare configurations measured under different prompt lengths or output budgets.
- Report variance, not just means.
- If a configuration fails to load on 8 GB, report that as a result, not as a failure to hide.

### 15.4 Pruning and distillation

Explicitly out of scope. If mentioned in the report, they belong in Future Work with a statement that they were not performed. Do not describe INT8/INT4 as "pruning".

---
