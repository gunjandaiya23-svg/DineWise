"""Parser for the Yelp ``attributes`` column.

The column is a Python-dict string with two quirks:

* nested values are sometimes themselves dict strings, and
* values sometimes carry Python-2 ``u'...'`` prefixes and wrapping quotes.

The parser is lossless for values, normalises keys to snake_case, flattens
nested dicts, and exposes canonical tri-state fields. Missing values become
``None`` and are never silently turned into ``False``.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

_TRUE = {"true", "yes"}
_FALSE = {"false", "no"}


def snake_case(name: str) -> str:
    """Convert a Yelp attribute key to snake_case."""
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    second = second.replace("-", "_").replace(" ", "_").replace("/", "_")
    second = re.sub(r"_+", "_", second).strip("_")
    return second.lower()


def _maybe_dict(value: Any) -> Any:
    """Parse a string that looks like a dict; return it unchanged otherwise."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def _clean_scalar(value: Any) -> Any:
    """Remove one layer of wrapping quotes and Python-2 ``u'...'`` prefixes."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    match = re.fullmatch(r"u'([^']*)'", text)
    if match:
        text = match.group(1)
    match = re.fullmatch(r"'([^']*)'", text)
    if match:
        text = match.group(1)
    return text


def _normalise(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    value = _clean_scalar(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "" or text == "None":
        return None
    lowered = text.lower()
    if lowered in {"null", "nan"}:
        return None
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def parse_ok(raw: Any) -> bool:
    """True when the outer dict string parsed successfully."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    return isinstance(_maybe_dict(raw), dict)


def parse_attributes(raw: Any) -> dict[str, Any]:
    """Return a flat dict of normalised attributes."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    outer = _maybe_dict(raw)
    if not isinstance(outer, dict):
        return {}

    flat: dict[str, Any] = {}
    for key, value in outer.items():
        value = _maybe_dict(value)
        key_snake = snake_case(key)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key_snake}_{snake_case(sub_key)}"] = _normalise(sub_value)
        else:
            flat[key_snake] = _normalise(value)
    return flat


def to_tristate(value: Any) -> bool | None:
    """Convert a value to True / False / None. Unknown never becomes False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = _clean_scalar(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return None


def to_price_bucket(value: Any) -> int | None:
    """Return a 1-4 price bucket, or None when unknown."""
    if value is None or isinstance(value, bool):
        return None
    value = _clean_scalar(value)
    if isinstance(value, (int, float)):
        bucket = int(value)
    else:
        match = re.search(r"[1-4]", str(value))
        if not match:
            return None
        bucket = int(match.group())
    return bucket if 1 <= bucket <= 4 else None


def to_enum(value: Any) -> str | None:
    """Normalise an enum-like value to a lower-case string or None."""
    if value is None:
        return None
    value = _clean_scalar(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "null"}:
            return None
        return text.lower()
    return str(value)


# name -> (flattened source key, conversion)
_CANONICAL: dict[str, tuple[str, str]] = {
    "price_range": ("restaurants_price_range2", "price"),
    "outdoor_seating": ("outdoor_seating", "tri"),
    "noise_level": ("noise_level", "enum"),
    "wifi": ("wi_fi", "enum"),
    "alcohol": ("alcohol", "enum"),
    "reservations": ("restaurants_reservations", "tri"),
    "good_for_kids": ("good_for_kids", "tri"),
    "attire": ("restaurants_attire", "enum"),
    "takeout": ("restaurants_take_out", "tri"),
    "delivery": ("restaurants_delivery", "tri"),
    "caters": ("caters", "tri"),
    "has_tv": ("has_tv", "tri"),
    "accepts_credit_cards": ("business_accepts_credit_cards", "tri"),
    "wheelchair_accessible": ("wheelchair_accessible", "tri"),
    "dogs_allowed": ("dogs_allowed", "tri"),
    "byob": ("byob", "tri"),
    "corkage": ("corkage", "tri"),
    "good_for_groups": ("restaurants_good_for_groups", "tri"),
    "good_for_dancing": ("good_for_dancing", "tri"),
}


def canonical_attributes(raw: Any) -> dict[str, Any]:
    """Return the flattened attributes plus canonical typed columns."""
    flat = parse_attributes(raw)
    out = dict(flat)
    for name, (source, kind) in _CANONICAL.items():
        value = flat.get(source)
        if kind == "price":
            out[name] = to_price_bucket(value)
        elif kind == "tri":
            out[name] = to_tristate(value)
        else:
            out[name] = to_enum(value)
    return out
