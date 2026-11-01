from __future__ import annotations

from dinewise.data.attributes import (
    canonical_attributes,
    parse_attributes,
    snake_case,
    to_price_bucket,
    to_tristate,
)


def test_snake_case_handles_yelp_keys() -> None:
    assert snake_case("RestaurantsPriceRange2") == "restaurants_price_range2"
    assert snake_case("HasTV") == "has_tv"
    assert snake_case("BYOB") == "byob"
    assert snake_case("WiFi") == "wi_fi"
    assert snake_case("RestaurantsTakeOut") == "restaurants_take_out"


def test_outer_dict_parses() -> None:
    raw = "{'OutdoorSeating': 'True', 'GoodForKids': 'False'}"
    parsed = parse_attributes(raw)
    assert parsed["outdoor_seating"] is True
    assert parsed["good_for_kids"] is False


def test_nested_dict_is_parsed() -> None:
    raw = "{'Ambience': \"{'romantic': True, 'classy': None}\"}"
    parsed = parse_attributes(raw)
    assert parsed["ambience_romantic"] is True
    assert parsed["ambience_classy"] is None


def test_malformed_and_missing_values() -> None:
    assert parse_attributes(None) == {}
    assert parse_attributes("") == {}
    assert parse_attributes("not a dict") == {}
    assert parse_attributes("{broken") == {}


def test_python2_prefix_and_wrapping_quotes() -> None:
    assert to_tristate("u'no'") is False
    assert to_tristate('"u\'no\'"') is False
    assert to_tristate("'yes'") is True
    assert to_tristate("None") is None
    assert to_tristate(None) is None


def test_price_bucket() -> None:
    assert to_price_bucket("1") == 1
    assert to_price_bucket("4") == 4
    assert to_price_bucket("None") is None
    assert to_price_bucket("0") is None
    assert to_price_bucket(None) is None


def test_canonical_unknown_is_not_false() -> None:
    raw = "{'RestaurantsPriceRange2': '2', 'NoiseLevel': \"u'average'\"}"
    canonical = canonical_attributes(raw)
    assert canonical["price_range"] == 2
    assert canonical["noise_level"] == "average"
    assert canonical["outdoor_seating"] is None
    assert canonical["byob"] is None
