"""Rule-based extraction of structured constraints from a query.

No LLM is involved at this stage: city, cuisine, budget, rating, attributes,
and dietary terms are matched with lexicons against the known catalogue.
Ambiguous terms go into ``soft_preferences``; nothing is ever invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Amenity words with no corresponding dataset field. Matching queries are
# refused with an explanation instead of answered from unrelated records.
_UNSUPPORTED_AMENITIES = {
    "rooftop": "rooftop areas",
    "pool": "pools",
    "swimming": "pools",
    "spa": "spas",
    "gym": "gyms",
    "playground": "playgrounds",
    "karaoke": "karaoke",
}


@dataclass
class Constraints:
    city: str | None = None
    unknown_city: str | None = None
    unsupported_terms: list[str] = field(default_factory=list)
    cuisines: list[str] = field(default_factory=list)
    max_price_bucket: int | None = None
    min_price_bucket: int | None = None
    min_rating: float | None = None
    required_attributes: list[str] = field(default_factory=list)
    dietary: list[str] = field(default_factory=list)
    ambience: list[str] = field(default_factory=list)
    soft_preferences: list[str] = field(default_factory=list)
    intent: str = "recommendation"


_CITY_ALIASES = {
    "philly": "Philadelphia",
    "phl": "Philadelphia",
    "philadelphia": "Philadelphia",
}

_CUISINE_WORDS = {
    "italian": "Italian",
    "pizza": "Italian",
    "mexican": "Mexican",
    "tacos": "Mexican",
    "tex-mex": "Mexican",
    "chinese": "Chinese",
    "dim sum": "Chinese",
    "japanese": "Japanese",
    "sushi": "Japanese",
    "ramen": "Japanese",
    "indian": "Indian",
    "thai": "Thai",
    "korean": "Korean",
    "vietnamese": "Vietnamese",
    "french": "French",
    "greek": "Greek",
    "mediterranean": "Mediterranean",
    "spanish": "Spanish",
    "tapas": "Spanish",
    "steakhouse": "Steakhouse",
    "steak": "Steakhouse",
    "seafood": "Seafood",
    "bbq": "BBQ",
    "barbecue": "BBQ",
    "bakery": "Bakery/Dessert",
    "dessert": "Bakery/Dessert",
    "breakfast": "Breakfast/Brunch",
    "brunch": "Breakfast/Brunch",
    "cafe": "Breakfast/Brunch",
    "coffee": "Breakfast/Brunch",
    "burger": "American",
    "american": "American",
    "sandwich": "American",
    "diner": "American",
    "soul food": "Cajun/Southern",
    "southern": "Cajun/Southern",
    "cajun": "Cajun/Southern",
    "creole": "Cajun/Southern",
    "cuban": "Latin/Caribbean",
    "caribbean": "Latin/Caribbean",
    "peruvian": "Latin/Caribbean",
    "burmese": "Southeast Asian",
    "indonesian": "Southeast Asian",
    "cambodian": "Southeast Asian",
    "filipino": "Southeast Asian",
    "bar": "Bar/Nightlife",
    "pub": "Bar/Nightlife",
    "brewery": "Bar/Nightlife",
    "cocktail": "Bar/Nightlife",
}

# word -> attribute column in restaurants.parquet
_ATTRIBUTE_WORDS = {
    "romantic": "ambience_romantic",
    "date": "ambience_romantic",
    "intimate": "ambience_intimate",
    "classy": "ambience_classy",
    "upscale": "ambience_classy",
    "trendy": "ambience_trendy",
    "casual": "ambience_casual",
    "quiet": "noise_level_quiet",
    "loud": "noise_level_loud",
    "outdoor": "outdoor_seating",
    "patio": "outdoor_seating",
    "outside seating": "outdoor_seating",
    "reservation": "reservations",
    "reserve": "reservations",
    "kid": "good_for_kids",
    "family": "good_for_kids",
    "child": "good_for_kids",
    "dessert": "good_for_meal_dessert",
    "dinner": "good_for_meal_dinner",
    "lunch": "good_for_meal_lunch",
    "brunch": "good_for_meal_brunch",
    "breakfast": "good_for_meal_breakfast",
    "late night": "good_for_meal_latenight",
    "takeout": "takeout",
    "take-out": "takeout",
    "delivery": "delivery",
    "byob": "byob",
    "wheelchair": "wheelchair_accessible",
    "dog": "dogs_allowed",
}

_DIETARY_WORDS = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "gluten-free": "gluten_free",
    "gluten free": "gluten_free",
    "halal": "halal",
    "kosher": "kosher",
    "dairy-free": "dairy_free",
    "nut-free": "nut_free",
}

_CHEAP_WORDS = {"cheap", "budget", "affordable", "inexpensive", "under $30", "under $25"}

# Capitalized words that are not places, to avoid false unknown-city flags.
_NON_PLACES = {
    "General", "Town", "City", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
_MODERATE_WORDS = {"moderate", "mid-range", "mid range", "reasonable"}
_EXPENSIVE_WORDS = {"fine dining", "upscale", "expensive", "fancy", "high-end"}


def _find_all(query: str, words: dict[str, str]) -> list[str]:
    lowered = query.lower()
    found: list[str] = []
    for word, value in words.items():
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            if value not in found:
                found.append(value)
    return found


def extract_constraints(query: str, known_cities: list[str] | None = None) -> Constraints:
    """Extract hard constraints and soft preferences from a query."""
    lowered = query.lower()
    result = Constraints()

    cities = known_cities or ["Philadelphia"]
    alias_hits = [name for alias, name in _CITY_ALIASES.items() if alias in lowered]
    city_hits = [city for city in cities if city.lower() in lowered]
    if city_hits:
        result.city = city_hits[0]
    elif alias_hits:
        result.city = alias_hits[0]

    result.cuisines = _find_all(query, _CUISINE_WORDS)

    cheap = any(word in lowered for word in _CHEAP_WORDS)
    moderate = any(word in lowered for word in _MODERATE_WORDS)
    expensive = any(word in lowered for word in _EXPENSIVE_WORDS)
    if cheap and not expensive:
        result.max_price_bucket = 2
    elif moderate and not expensive:
        result.min_price_bucket, result.max_price_bucket = 2, 3
    elif expensive:
        result.min_price_bucket = 3

    dollar = re.search(r"\$\s*(\d+)", query)
    if dollar and result.max_price_bucket is None:
        amount = int(dollar.group(1))
        result.max_price_bucket = 1 if amount < 15 else (2 if amount <= 30 else 4)

    buckets = re.search(r"under\s*(\${1,4})", query)
    if buckets and result.max_price_bucket is None:
        result.max_price_bucket = len(buckets.group(1))

    rating = re.search(r"(\d(?:\.\d)?)\s*(?:\+|plus)?\s*(?:stars?|rating)", lowered)
    if rating:
        result.min_rating = float(rating.group(1))
    elif "highly rated" in lowered or "best" in lowered:
        result.soft_preferences.append("highly_rated")

    result.required_attributes = _find_all(query, _ATTRIBUTE_WORDS)
    result.dietary = _find_all(query, _DIETARY_WORDS)
    result.unsupported_terms = _find_all(query, _UNSUPPORTED_AMENITIES)
    ambience_words = {"romantic", "intimate", "classy", "trendy", "casual", "quiet"}
    result.ambience = [
        word for word in ambience_words if re.search(r"\b" + word + r"\b", lowered)
    ]

    if re.search(r"\bcompare\b|versus|\bvs\b|\bX and Y\b", lowered):
        result.intent = "comparison"
    elif re.search(r"interior|photo|picture|image|upload|look like|show me.*inside", lowered):
        result.intent = "visual"

    for match in re.finditer(
        r"\b(?:in|near|around|across from)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)",
        query,
    ):
        place = match.group(1)
        if place in _NON_PLACES or len(place) < 3:
            continue
        known = [city.lower() for city in (known_cities or [])]
        if result.city and place.lower() == result.city.lower():
            continue
        if place.lower() not in known:
            result.unknown_city = place
            break

    return result
