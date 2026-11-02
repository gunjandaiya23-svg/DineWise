"""CLIP zero-shot classification over Food-101 classes."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

# Only classes with an unambiguous cuisine family are used for the cuisine
# accuracy metric. The rest still participate in class-level accuracy.
CUISINE_MAP: dict[str, str] = {
    "pizza": "Italian",
    "spaghetti_bolognese": "Italian",
    "spaghetti_carbonara": "Italian",
    "lasagna": "Italian",
    "risotto": "Italian",
    "gnocchi": "Italian",
    "sushi": "Japanese",
    "ramen": "Japanese",
    "tempura": "Japanese",
    "edamame": "Japanese",
    "miso_soup": "Japanese",
    "tacos": "Mexican",
    "burrito": "Mexican",
    "nachos": "Mexican",
    "guacamole": "Mexican",
    "ceviche": "Latin/Caribbean",
    "pad_thai": "Thai",
    "spring_rolls": "Southeast Asian",
    "dumplings": "Chinese",
    "fried_rice": "Chinese",
    "peking_duck": "Chinese",
    "hummus": "Mediterranean",
    "falafel": "Mediterranean",
    "greek_salad": "Greek",
    "baklava": "Mediterranean",
    "steak": "Steakhouse",
    "baby_back_ribs": "BBQ",
    "pulled_pork_sandwich": "BBQ",
    "hamburger": "American",
    "hot_dog": "American",
    "macaroni_and_cheese": "American",
    "fried_chicken": "Cajun/Southern",
    "croissant": "French",
    "creme_brulee": "French",
    "escargots": "French",
    "french_onion_soup": "French",
    "cheesecake": "Bakery/Dessert",
    "chocolate_cake": "Bakery/Dessert",
    "ice_cream": "Bakery/Dessert",
    "tiramisu": "Bakery/Dessert",
    "pancakes": "Breakfast/Brunch",
    "waffles": "Breakfast/Brunch",
    "omelette": "Breakfast/Brunch",
    "fish_and_chips": "Seafood",
    "grilled_salmon": "Seafood",
    "oysters": "Seafood",
    "chicken_curry": "Indian",
    "naan": "Indian",
    "samosa": "Indian",
}


def _features(output: Any, preferred: str) -> Any:
    """Normalise transformers 4.x tensors and 5.x output objects."""
    if torch.is_tensor(output):
        return output
    for name in (preferred, "text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"cannot extract features from {type(output)!r}")


class CLIPZeroShot:
    def __init__(self, model_name: str, classes: list[str], device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.classes = classes
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        prompts = [f"a photo of {name.replace('_', ' ')}, a type of food" for name in classes]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            text_features = _features(
                self.model.get_text_features(**inputs), "text_embeds"
            )
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, images: list[Image.Image], top_k: int = 5) -> list[list[dict[str, Any]]]:
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = _features(
                self.model.get_image_features(**inputs), "image_embeds"
            )
        features = features / features.norm(dim=-1, keepdim=True)
        logits = (features @ self.text_features.T).softmax(dim=-1).float().cpu()

        results = []
        for row in logits:
            scores, indices = row.topk(min(top_k, len(self.classes)))
            results.append(
                [
                    {"class": self.classes[int(index)], "probability": float(score)}
                    for score, index in zip(scores, indices, strict=False)
                ]
            )
        return results
