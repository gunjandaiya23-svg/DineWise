# Appendix D — Test queries

> Part of the DineWise project plan. Operational reference for implementation.
> See [the plan index](../README.md) for the full reading order.

---


The original test list assumed dollar prices and Las Vegas. Adaptations are explicit.

| # | Query | Facet | Adaptation |
|---:|---|---|---|
| 1 | Find Italian restaurants with good reviews in Philadelphia. | Cuisine | Added a city from the corpus |
| 2 | Recommend restaurants under $$. | Budget | Bucket instead of dollars |
| 3 | Which restaurants have the best food quality according to reviews? | Review aspect | City-scoped or global |
| 4 | Find restaurants with positive ambience reviews. | Aspect | Unchanged |
| 5 | Recommend a restaurant for a family dinner. | Personalization | Use `good_for_kids` |
| 6 | Compare two restaurants based on reviews. | Comparison | Pick two IDs from the corpus |
| 7 | Which restaurants are praised for their desserts? | Review understanding | Use review and `good_for_meal_dessert` |
| 8 | Find restaurants with vegetarian options. | Dietary | Textual evidence only |
| 9 | Show restaurants similar to this uploaded food image. | Image | CLIP zero-shot → cuisine hint; no restaurant image match |
| 10 | Find restaurants with attractive interiors. | Visual | Text over photo summaries |
| 11 | Which restaurant offers the best value for money? | Ranking | Price bucket + rating + review evidence |
| 12 | Recommend a restaurant matching multiple preferences. | Multi-constraint | Explicit constraints |
| 13 | Give me the live availability of a restaurant. | Out of scope | Must refuse |
| 14 | Recommend a restaurant in Las Vegas. | Out of scope | City genuinely absent |
| 15 | Which restaurant has a rooftop pool? | Unsupported | Must say the dataset lacks this |

---
