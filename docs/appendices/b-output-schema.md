# Appendix B — Output schema

> Part of the DineWise project plan. Operational reference for implementation.
> See [the plan index](../README.md) for the full reading order.

---


```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["recommendations", "insufficient_evidence"],
  "additionalProperties": false,
  "properties": {
    "recommendations": {
      "type": "array",
      "maxItems": 5,
      "items": {
        "type": "object",
        "required": [
          "business_id", "name", "cuisine", "price_bucket",
          "rating", "evidence", "explanation", "sources"
        ],
        "additionalProperties": false,
        "properties": {
          "business_id": {"type": "string"},
          "name": {"type": "string"},
          "cuisine": {"type": "string"},
          "price_bucket": {"type": "integer", "minimum": 1, "maximum": 4},
          "rating": {"type": "number", "minimum": 1, "maximum": 5},
          "evidence": {
            "type": "array",
            "maxItems": 3,
            "items": {
              "type": "object",
              "required": ["review_id", "quote"],
              "additionalProperties": false,
              "properties": {
                "review_id": {"type": "string"},
                "quote": {"type": "string", "minLength": 10, "maxLength": 400}
              }
            }
          },
          "explanation": {"type": "string", "minLength": 20, "maxLength": 600},
          "sources": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
          }
        }
      }
    },
    "insufficient_evidence": {"type": "boolean"},
    "notes": {"type": "string", "maxLength": 400}
  }
}
```

---
