# Personalization

> Part of the DineWise project plan. Explicit profiles, constraint classes, feedback, and evaluation.
> See [the plan index](README.md) for the full reading order.

---

## 12. Personalization

Personalization is explicit and inspectable, not learned from interactions.

**Profile fields:**

- home city or preferred cities,
- budget bucket,
- preferred cuisines,
- dietary requirements (vegetarian, vegan, gluten-free, halal, etc.),
- preferred ambience (romantic, quiet, family-friendly, lively),
- accessibility needs,
- dislikes (e.g., no seafood).

**Constraint classification:**

- Hard: city, budget ceiling, dietary requirement explicitly marked as required, accessibility.
- Soft: cuisine preference, ambience preference, rating preference.

**Application:**

1. Hard constraints are applied in `filter_candidates`.
2. Soft preferences contribute a small additive score after reranking:
   - +0.05 for a cuisine match,
   - +0.05 for an ambience match,
   - +0.03 for an attribute match,
   - weights are fixed before evaluation and reported.
3. In-session feedback:
   - "like" adds +0.05 to similar restaurants,
   - "dislike" subtracts 0.10,
   - similarity is defined by cuisine family and price bucket, not by embeddings, to keep the mechanism explainable.
4. The UI shows which preferences influenced the ranking.

**Evaluation (E11):**

- Run the gold queries with and without the profile.
- Measure preference satisfaction with a rubric: does the returned set respect cuisine, budget, ambience, and dietary constraints?
- Report both relevance (nDCG@5) and satisfaction. A personalization feature that improves satisfaction but harms relevance is not automatically a win.

**What this is not:** collaborative filtering, implicit learning from other users, or a trained recommender. The report must state this plainly.

---
