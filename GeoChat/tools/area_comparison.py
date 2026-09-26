"""Visual area comparison for natural-color optical satellite images.

This module estimates the visible pixel area of a few broad classes:
- water / water body
- vegetation / greenery / forest / trees / agriculture
- land

The result is an approximate visual estimate, not a calibrated
remote-sensing land-cover classification.
"""

import numpy as np
from PIL import Image


def _water_mask(rgb):
    """Detect likely water pixels using color and saturation cues."""

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)

    saturation = max_rgb - min_rgb

    # Blue/cyan dominance.
    blue_dominant = (
        (b > r * 1.03)
        & (b > g * 1.00)
    )

    # Dark blue/cyan water.
    dark_blue = (
        blue_dominant
        & (b < 180)
        & (saturation > 8)
    )

    # Brighter blue/cyan water.
    bright_blue = (
        (b > r * 1.05)
        & (b >= g * 0.98)
        & (saturation > 10)
    )

    return dark_blue | bright_blue


def _vegetation_mask(rgb):
    """Detect likely green vegetation pixels."""

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    return (
        (g > r * 1.04)
        & (g > b * 1.02)
        & ((g - r) > 5)
    )


def _supported_class(entity):
    """Convert natural-language class names to supported classes."""

    entity = (entity or "").lower().strip()

    if "water" in entity:
        return "water"

    if any(
        word in entity
        for word in (
            "vegetation",
            "greenery",
            "forest",
            "trees",
            "tree",
            "agriculture",
            "cropland",
            "grass",
        )
    ):
        return "vegetation"

    if any(
        word in entity
        for word in (
            "land",
            "ground",
            "terrain",
        )
    ):
        return "land"

    return None


def compare_visible_area(image_path, entity_a, entity_b):
    """
    Compare approximate visible area of two supported classes.

    Important:
    'land' means pixels not classified as water or vegetation.

    This avoids the previous bug where land was simply defined as
    '~water', causing land to become nearly 100% whenever water
    detection was weak.
    """

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)

    class_a = _supported_class(entity_a)
    class_b = _supported_class(entity_b)

    if class_a is None or class_b is None:
        return None

    water = _water_mask(rgb)
    vegetation = _vegetation_mask(rgb)

    # Prevent vegetation from being counted as water.
    vegetation = vegetation & ~water

    # Remaining broad non-water/non-vegetation pixels.
    land = ~(water | vegetation)

    masks = {
        "water": water,
        "vegetation": vegetation,
        "land": land,
    }

    mask_a = masks[class_a]
    mask_b = masks[class_b]

    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())

    total_pixels = rgb.shape[0] * rgb.shape[1]

    if total_pixels == 0:
        return None

    pct_a = (area_a / total_pixels) * 100.0
    pct_b = (area_b / total_pixels) * 100.0

    # Determine winner.
    difference = abs(area_a - area_b)

    # Treat very small differences as approximately equal.
    if difference <= max(1, int(total_pixels * 0.01)):
        winner = "approximately equal"
    elif area_a > area_b:
        winner = entity_a
    else:
        winner = entity_b

    # Confidence is based on separation AND whether both classes
    # were actually detected.
    larger = max(area_a, area_b)
    smaller = min(area_a, area_b)

    if larger == 0:
        confidence = 0.0

    else:
        margin = (larger - smaller) / larger

        confidence = 0.55 + (margin * 0.30)

        # If one class was completely absent, do not claim extremely
        # high confidence because the color heuristic may have missed it.
        if smaller == 0:
            confidence = min(confidence, 0.70)

        confidence = min(0.85, confidence)

    return {
        "winner": winner,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "area_a_percent": round(pct_a, 1),
        "area_b_percent": round(pct_b, 1),
        "confidence": round(confidence, 2),
        "method": "pixel-based visual area estimate",
        "pixel_counts": {
            "entity_a": area_a,
            "entity_b": area_b,
            "total": total_pixels,
        },
    }