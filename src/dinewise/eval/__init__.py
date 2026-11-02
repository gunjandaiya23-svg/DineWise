"""Fixed test queries. Imported by the query runner and experiment scripts."""

from __future__ import annotations

ANSWERABLE_QUERIES: list[tuple[str, str, str]] = [
    ("Q01", "cuisine", "Find Italian restaurants with good reviews in Philadelphia."),
    ("Q02", "budget", "Recommend restaurants under $$ in Philadelphia."),
    ("Q03", "review_aspect", "Which restaurants have the best food quality according to reviews?"),
    ("Q04", "ambience", "Find restaurants with positive ambience reviews in Philadelphia."),
    ("Q05", "personalization", "Recommend a restaurant for a family dinner in Philadelphia."),
    ("Q06", "comparison", "Compare two Italian restaurants in Philadelphia based on reviews."),
    ("Q07", "dessert", "Which restaurants in Philadelphia are praised for their desserts?"),
    ("Q08", "dietary", "Find restaurants with vegetarian options in Philadelphia."),
    ("Q09", "image", "Show restaurants similar to this uploaded pizza photo in Philadelphia."),
    ("Q10", "visual", "Find restaurants with attractive interiors in Philadelphia."),
    ("Q11", "ranking", "Which restaurant in Philadelphia offers the best value for money?"),
    ("Q12", "multi_constraint", "Family-friendly Mexican restaurant in Philadelphia under $$."),
]

OUT_OF_SCOPE_QUERIES: list[tuple[str, str, str]] = [
    ("Q13", "out_of_scope", "Give me the live availability of Reading Terminal Market tonight."),
    ("Q14", "out_of_scope", "Recommend a restaurant in Las Vegas."),
    ("Q15", "unsupported", "Which restaurant in Philadelphia has a rooftop pool?"),
]

ALL_QUERIES = ANSWERABLE_QUERIES + OUT_OF_SCOPE_QUERIES
