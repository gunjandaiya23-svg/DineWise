# LoRA fine-tuning

> Part of the DineWise project plan. Data construction, splits, hyperparameters, training, and evaluation.
> See [the plan index](README.md) for the full reading order.

---

## 14. LoRA fine-tuning

### 14.1 Objective

Improve **response behavior** — format compliance, explanation structure, constraint acknowledgement, abstention — not factual knowledge. Restaurant facts must continue to come from retrieval.

Training a 1.5B model to memorize 6,033 restaurants is both infeasible and undesirable: it would increase hallucination and make the system stale.

### 14.2 Data construction

Target: 400 instruction–response examples (range 300–500).

Task types and approximate counts:

| Type | Count | Example instruction |
|---|---:|---|
| Budget recommendation | 60 | "Recommend a cheap Italian place in Tampa." |
| Cuisine recommendation | 60 | "Find good Vietnamese food in Philadelphia." |
| Dietary recommendation | 60 | "I need vegetarian options under $$." |
| Ambience recommendation | 50 | "Suggest a quiet restaurant for a date." |
| Comparison | 50 | "Compare Restaurant A and Restaurant B on food and service." |
| Review-grounded recommendation | 50 | "Which restaurants are praised for desserts?" |
| Out-of-scope / abstention | 40 | "Is Restaurant A available tonight at 8?" |
| Multi-constraint | 30 | "Family-friendly Mexican place in Tucson under $$." |

Construction method:

1. Sample restaurants and constraints from the training split only.
2. Build the context block from the actual structured record and retrieved evidence.
3. Generate the target answer with a deterministic template that:
   - cites the business ID and evidence IDs,
   - states the matched constraints,
   - uses 2–3 sentences,
   - abstains when constraints cannot be met.
4. Optionally paraphrase instructions with a teacher LLM if an API key is available. Every synthetic example must be manually inspected in a sample; synthetic data is not ground truth.
5. Manually review a random 15% (60 examples). Record the error rate and fix the templates. Do not ship examples with incorrect facts.

Leakage control:

- Split by restaurant, not by example, so the same restaurant never appears in both training and test.
- Split by instruction template family as a secondary check.
- Verify zero business-ID overlap between splits.

### 14.3 Splits

- 80% train (320),
- 10% validation (40),
- 10% test (40).

The 40 test examples are for format/behavior evaluation only. The main RAG evaluation uses the 120-query gold set, which is separate and never used for training.

### 14.4 Hyperparameters (starting point)

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | QLoRA (4-bit NF4, double quant) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | all linear projections (q, k, v, o, gate, up, down) |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Batch size | 1 per device, gradient accumulation 4 |
| Max sequence length | 1024 |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Precision | bfloat16 compute |
| Seed | 42 |
| Early stopping | validation loss, patience 1 epoch |

These are experimental settings, not claimed optima. Any tuning is done on validation data and reported.

### 14.5 Training procedure

1. Freeze the base model; train adapters only.
2. Log loss, learning rate, gradient norm, and VRAM every 10 steps.
3. Save the adapter, tokenizer, and training config.
4. Evaluate on the validation split; select the best checkpoint by validation loss and format validity.
5. Merge the adapter for the latency comparison (E8, M5); keep the unmerged adapter for the quality comparison (E7).

Expected training time on the RTX 4060: 30–90 minutes for 400 examples and 2 epochs. This is an estimate; measure and report.

### 14.6 Evaluation

Compare four systems on the same 120-query gold set:

| System | Purpose |
|---|---|
| Qwen base, no retrieval | Lower bound; shows hallucination |
| Qwen + RAG (no LoRA) | Grounded baseline |
| Qwen + RAG + LoRA | Adaptation effect |
| Qwen + LoRA, no retrieval | Tests whether tuning substitutes for retrieval |

Metrics: format validity, constraint satisfaction, faithfulness, unsupported-claim rate, answer relevance, abstention correctness, and nDCG@5.

Hypothesis: LoRA improves format and explanation quality; it does not improve factual accuracy beyond RAG, and without RAG it increases unsupported claims. If the data contradicts this, report the contradiction.

---
