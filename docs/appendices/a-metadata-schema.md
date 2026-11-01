# Appendix A — Metadata schema

> Part of the DineWise project plan. Operational reference for implementation.
> See [the plan index](../README.md) for the full reading order.

---


Required metadata on every chunk:

| Field | Type | Nullable | Source |
|---|---|---|---|
| `chunk_id` | string | no | generated |
| `business_id` | string | yes (policy) | parent document |
| `restaurant_name` | string | yes | catalogue |
| `city` | string | yes | catalogue |
| `state` | string | yes | catalogue |
| `cuisine_primary` | string | yes | catalogue |
| `price_range` | int | yes | catalogue |
| `document_type` | enum | no | parent document |
| `section` | string | yes | parent section |
| `source_file` | string | no | loader |
| `page` | int | yes | loader |
| `token_count` | int | no | chunker |
| `char_count` | int | no | chunker |
| `chunk_config` | string | no | experiment config |
| `text` | string | no | chunker |

Validation rules:

- `document_type=policy` ⇒ `business_id` is null and `section` is a policy heading.
- `document_type in {profile, review_insight, visual}` ⇒ `business_id` is non-null.
- `token_count` ≤ 512 for the BGE index (or truncation is explicitly recorded).

---
