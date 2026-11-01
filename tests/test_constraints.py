from __future__ import annotations

import pandas as pd

from dinewise.llm.validators import check_retrieved_compliance, validate_answer
from dinewise.rag.constraints import Constraints, extract_constraints
from dinewise.rag.filters import apply_filters


def _catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "business_id": "A" * 22,
                "name": "Alpha",
                "city": "Philadelphia",
                "cuisine_primary": "Italian",
                "cuisine_tags": ["Italian"],
                "price_range": 2,
                "stars": 4.5,
                "outdoor_seating": True,
            },
            {
                "business_id": "B" * 22,
                "name": "Beta",
                "city": "Philadelphia",
                "cuisine_primary": "Chinese",
                "cuisine_tags": ["Chinese"],
                "price_range": 1,
                "stars": 4.0,
                "outdoor_seating": False,
            },
        ]
    )


def _hits(ids: list[str]) -> list[dict]:
    catalogue = _catalogue().set_index("business_id")
    out = []
    for business_id in ids:
        row = catalogue.loc[business_id].to_dict()
        row["business_id"] = business_id
        out.append({"score": 0.7, "chunk": row})
    return out


def test_extract_cuisine_city_and_budget() -> None:
    result = extract_constraints("Find cheap Italian restaurants in Philly")
    assert result.city == "Philadelphia"
    assert result.cuisines == ["Italian"]
    assert result.max_price_bucket == 2


def test_dollar_sign_budget_maps_to_bucket() -> None:
    assert extract_constraints("Recommend restaurants under $$.").max_price_bucket == 2
    assert extract_constraints("Recommend restaurants under $.").max_price_bucket == 1
    assert extract_constraints("Upscale places under $$$ in Philly.").max_price_bucket == 3


def test_dietary_terms_stay_soft() -> None:
    result = extract_constraints("Find restaurants with vegetarian options in Philadelphia.")
    assert result.dietary == ["vegetarian"]
    assert "Vegetarian/Vegan" not in result.cuisines


def test_unknown_city_is_flagged() -> None:
    result = extract_constraints(
        "Recommend a restaurant in Las Vegas.", known_cities=["Philadelphia"]
    )
    assert result.unknown_city == "Las Vegas"
    assert result.city is None


def test_known_city_is_not_flagged() -> None:
    result = extract_constraints(
        "Find Italian restaurants in Philadelphia.", known_cities=["Philadelphia"]
    )
    assert result.city == "Philadelphia"
    assert result.unknown_city is None


def test_unsupported_amenity_is_flagged() -> None:
    result = extract_constraints("Which restaurant has a rooftop pool?")
    assert "rooftop areas" in result.unsupported_terms
    assert "pools" in result.unsupported_terms


def test_supported_attributes_are_not_flagged() -> None:
    result = extract_constraints("Romantic restaurant with outdoor seating.")
    assert result.unsupported_terms == []
    assert "ambience_romantic" in result.required_attributes
    assert "outdoor_seating" in result.required_attributes


def test_extract_attributes_and_dietary() -> None:
    result = extract_constraints("romantic vegetarian place with outdoor seating")
    assert "ambience_romantic" in result.required_attributes
    assert "outdoor_seating" in result.required_attributes
    assert result.dietary == ["vegetarian"]


def test_extract_rating_and_comparison_intent() -> None:
    result = extract_constraints("Compare restaurants rated 4.5 stars or higher")
    assert result.intent == "comparison"
    assert result.min_rating == 4.5


def test_filters_apply_cuisine_and_price() -> None:
    catalogue = _catalogue()
    constraints = Constraints(cuisines=["Italian"], max_price_bucket=2)
    filtered, counts = apply_filters(catalogue, constraints)
    assert list(filtered["business_id"]) == ["A" * 22]
    assert counts["cuisine"] == 1


def test_filters_respect_tri_state_unknown() -> None:
    catalogue = _catalogue()
    catalogue["outdoor_seating"] = pd.Series(
        [True, pd.NA], dtype="boolean", index=catalogue.index
    )
    constraints = Constraints(required_attributes=["outdoor_seating"])
    filtered, _ = apply_filters(catalogue, constraints)
    assert list(filtered["business_id"]) == ["A" * 22]


def test_validate_answer_rejects_unknown_ids() -> None:
    catalogue = _catalogue()
    hits = _hits(["A" * 22])
    report = validate_answer(f"Try {('C' * 22)} tonight", hits, catalogue)
    assert not report.passed
    assert any(check.name == "ids_exist" for check in report.hard_failures)


def test_validate_answer_rejects_live_language() -> None:
    catalogue = _catalogue()
    hits = _hits(["A" * 22])
    report = validate_answer("Alpha is available tonight at 8pm", hits, catalogue)
    assert not report.passed
    assert any(check.name == "no_live_claims" for check in report.hard_failures)


def test_retrieved_compliance_flags_cuisine_mismatch() -> None:
    catalogue = _catalogue()
    hits = _hits(["A" * 22, "B" * 22])
    checks = check_retrieved_compliance(
        hits, catalogue, Constraints(cuisines=["Italian"])
    )
    assert any(
        check.name == "cuisine_satisfied" and not check.passed for check in checks
    )
