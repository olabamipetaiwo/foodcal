"""
label_mapping.py

Assigns Low / Medium / High calorie range labels to each Food-101 category
using USDA FoodData Central reference values (kcal per typical serving).

Thresholds:
    Low    < 400 kcal
    Medium 400 – 700 kcal
    High   > 700 kcal

Run this script directly to write data/labels.json.
"""

import json
import os

# ---------------------------------------------------------------------------
# USDA-referenced approximate kcal per typical serving for each Food-101 class
# Sources: USDA FoodData Central (fdc.nal.usda.gov) and standard portion sizes
# ---------------------------------------------------------------------------
FOOD101_KCAL = {
    "apple_pie": 296,
    "baby_back_ribs": 700,
    "baklava": 334,
    "beef_carpaccio": 210,
    "beef_tartare": 220,
    "beet_salad": 150,
    "beignets": 420,
    "bibimbap": 560,
    "bread_pudding": 480,
    "breakfast_burrito": 610,
    "bruschetta": 200,
    "caesar_salad": 360,
    "cannoli": 220,
    "caprese_salad": 200,
    "carrot_cake": 415,
    "ceviche": 150,
    "cheese_plate": 440,
    "cheesecake": 401,
    "chicken_curry": 450,
    "chicken_quesadilla": 530,
    "chicken_wings": 430,
    "chocolate_cake": 537,
    "chocolate_mousse": 380,
    "churros": 400,
    "clam_chowder": 190,
    "club_sandwich": 590,
    "crab_cakes": 260,
    "creme_brulee": 330,
    "croque_madame": 470,
    "cup_cakes": 290,
    "deviled_eggs": 140,
    "donuts": 452,
    "dumplings": 330,
    "edamame": 120,
    "eggs_benedict": 520,
    "escargots": 140,
    "falafel": 333,
    "filet_mignon": 350,
    "fish_and_chips": 600,
    "foie_gras": 462,
    "french_fries": 365,
    "french_onion_soup": 190,
    "french_toast": 450,
    "fried_calamari": 325,
    "fried_rice": 333,
    "frozen_yogurt": 160,
    "garlic_bread": 300,
    "gnocchi": 250,
    "greek_salad": 180,
    "grilled_cheese_sandwich": 520,
    "grilled_salmon": 280,
    "guacamole": 240,
    "gyoza": 280,
    "hamburger": 540,
    "hot_and_sour_soup": 95,
    "hot_dog": 290,
    "huevos_rancheros": 420,
    "hummus": 170,
    "ice_cream": 273,
    "lasagna": 336,
    "lobster_bisque": 280,
    "lobster_roll_sandwich": 580,
    "macaroni_and_cheese": 489,
    "macarons": 390,
    "miso_soup": 40,
    "mussels": 200,
    "nachos": 740,
    "omelette": 300,
    "onion_rings": 480,
    "oysters": 80,
    "pad_thai": 430,
    "paella": 380,
    "pancakes": 520,
    "panna_cotta": 280,
    "peking_duck": 490,
    "pho": 350,
    "pizza": 570,
    "pork_chop": 350,
    "poutine": 700,
    "prime_rib": 660,
    "pulled_pork_sandwich": 620,
    "ramen": 470,
    "ravioli": 310,
    "red_velvet_cake": 498,
    "risotto": 340,
    "samosa": 260,
    "sashimi": 130,
    "scallops": 140,
    "seaweed_salad": 70,
    "shrimp_and_grits": 480,
    "spaghetti_bolognese": 430,
    "spaghetti_carbonara": 540,
    "spring_rolls": 220,
    "steak": 540,
    "strawberry_shortcake": 360,
    "sushi": 350,
    "tacos": 430,
    "takoyaki": 200,
    "tiramisu": 470,
    "tuna_tartare": 185,
    "waffles": 540,
}

LOW_MAX = 400
HIGH_MIN = 700


def kcal_to_label(kcal: int) -> str:
    if kcal < LOW_MAX:
        return "Low"
    elif kcal <= HIGH_MIN:
        return "Medium"
    else:
        return "High"


def build_label_map() -> dict:
    """Return {food_class: label} for all 101 Food-101 categories."""
    return {cls: kcal_to_label(kcal) for cls, kcal in FOOD101_KCAL.items()}


def save_labels(out_path: str = None) -> dict:
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "..", "data", "labels.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    label_map = build_label_map()
    with open(out_path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"Saved {len(label_map)} labels to {out_path}")
    # Print distribution
    from collections import Counter
    dist = Counter(label_map.values())
    print(f"Distribution: {dict(dist)}")
    return label_map


if __name__ == "__main__":
    save_labels()
