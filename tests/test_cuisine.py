from __future__ import annotations

from dinewise.data.cuisine import load_taxonomy, match_cuisines, primary_cuisine

TAXONOMY = load_taxonomy()


def test_exact_token_matching_avoids_barber_and_barbecue_confusion() -> None:
    assert match_cuisines("Barber, Beauty & Spas", TAXONOMY) == []
    matches = match_cuisines("Barbeque, Bars", TAXONOMY)
    assert "BBQ" in matches
    assert "Bar/Nightlife" in matches


def test_multi_label_and_primary_priority() -> None:
    matches = match_cuisines("Pizza, Italian, Vegetarian", TAXONOMY)
    assert matches == ["Italian", "Vegetarian/Vegan"]
    assert primary_cuisine(matches) == "Italian"


def test_american_and_pizza_prefers_italian() -> None:
    matches = match_cuisines("American (New), Pizza", TAXONOMY)
    assert primary_cuisine(matches) == "Italian"


def test_unspecified() -> None:
    assert match_cuisines("Shopping, Home Services", TAXONOMY) == []
    assert primary_cuisine([]) == "Unspecified"


def test_case_insensitive() -> None:
    assert match_cuisines("mexican, TACOS", TAXONOMY)[0] == "Mexican"


def test_new_families_cover_previous_gaps() -> None:
    assert match_cuisines("Burmese, Restaurants", TAXONOMY) == ["Southeast Asian"]
    assert match_cuisines("Soul Food, Restaurants", TAXONOMY) == ["Cajun/Southern"]
    assert match_cuisines("Restaurants, Cajun/Creole", TAXONOMY) == ["Cajun/Southern"]
    assert match_cuisines("Indonesian, Restaurants", TAXONOMY) == ["Southeast Asian"]


def test_latin_american_is_not_mexican() -> None:
    assert match_cuisines("Cuban, Latin American, Restaurants", TAXONOMY) == [
        "Latin/Caribbean"
    ]
