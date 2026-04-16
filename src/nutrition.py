"""
nutrition.py

Optional enrichment layer: queries the USDA FoodData Central API to retrieve
a reference kcal value for a given food description (typically the BLIP-2 caption).

Used in app.py after the classifier runs:
  - If a confident USDA match is found  → show DB kcal + source label
  - Otherwise                           → fall back to weighted class-probability estimate

API: https://api.nal.usda.gov/fdc/v1/foods/search
  - Free, no account needed with DEMO_KEY (30 req/hour)
  - Register at https://fdc.nal.usda.gov/api-guide.html for a personal key (1000 req/hour)

Set env var USDA_API_KEY to override the default DEMO_KEY.
"""

import json
import os
import urllib.parse
import urllib.request
from typing import Optional, Tuple

USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Minimum relevance score to trust a USDA match.
# From testing: scores above 700 correspond to food-name matches;
# below that the API tends to return unrelated items.
SCORE_THRESHOLD = 700

# Branded foods report kcal per serving; SR Legacy per 100g.
# For SR Legacy we apply a 300g default meal portion to convert to per-meal kcal.
SR_LEGACY_PORTION_G = 300

# Sanity clamp: discard implausible kcal values
KCAL_MIN = 50
KCAL_MAX = 2500


def _extract_kcal(food: dict) -> Optional[float]:
    """Pull energy (kcal) from a USDA food item, scaling SR Legacy by portion size."""
    kcal = None
    for n in food.get("foodNutrients", []):
        if "Energy" in n.get("nutrientName", "") and n.get("unitName") == "KCAL":
            kcal = n.get("value")
            break

    if kcal is None:
        return None

    # SR Legacy values are per 100g — scale to a typical meal portion
    if food.get("dataType") == "SR Legacy":
        kcal = kcal * SR_LEGACY_PORTION_G / 100

    return kcal


def lookup_kcal(
    query: str,
    score_threshold: float = SCORE_THRESHOLD,
    timeout: int = 6,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Query USDA FoodData Central for the best matching food and return its kcal.

    Returns:
        (kcal, description)  if a confident match is found and kcal is plausible
        (None, None)         if no match, low confidence, or request fails
    """
    if not query or not query.strip():
        return None, None

    params = urllib.parse.urlencode({
        "query": query.strip(),
        "pageSize": 3,
        "api_key": USDA_API_KEY,
    })
    url = f"{USDA_SEARCH_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None, None

    foods = data.get("foods", [])
    if not foods:
        return None, None

    best = foods[0]
    score = best.get("score", 0)

    if score < score_threshold:
        return None, None

    kcal = _extract_kcal(best)
    if kcal is None:
        return None, None

    kcal = round(kcal)
    if not (KCAL_MIN <= kcal <= KCAL_MAX):
        return None, None

    description = best.get("description", "").title()
    return kcal, description
