from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# Sector centers and widths are fractions of the hue wheel. The sizes follow
# Cohen-Or et al.'s appendix: i/L/Y small=5%, L large=22%, V/Y/X large=26%,
# T=50%. L_mirror is included because the paper discusses mirror-L examples.
HUE_TEMPLATES: dict[str, list[tuple[float, float]]] = {
    "i": [(0.0, 0.05)],
    "V": [(0.0, 0.26)],
    "L": [(0.0, 0.05), (0.25, 0.22)],
    "L_mirror": [(0.0, 0.05), (-0.25, 0.22)],
    "I": [(0.0, 0.05), (0.50, 0.05)],
    "T": [(0.0, 0.50)],
    "Y": [(0.0, 0.26), (0.50, 0.05)],
    "X": [(0.0, 0.26), (0.50, 0.26)],
}

# Empirical AADB validation-set distance thresholds, in degrees, for the
# current implementation and 256px images. Lower distance means a stronger
# template fit. These thresholds make the structured output easier to compare
# without pretending the convenience 0-1 score has an absolute meaning.
DEFAULT_DISTANCE_THRESHOLDS = {
    "very_high": 0.04,  # about top quartile
    "high": 0.26,      # about better than median
    "medium": 0.91,    # about better than 80th percentile
}


def angular_distance_degrees(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def distance_to_template(
    hues: np.ndarray,
    template: list[tuple[float, float]],
    rotation_degrees: float,
) -> np.ndarray:
    distances = np.full_like(hues, fill_value=np.inf, dtype=np.float32)
    for center_fraction, width_fraction in template:
        center = (center_fraction * 360.0 + rotation_degrees) % 360.0
        half_width = width_fraction * 360.0 / 2.0
        outside_distance = np.maximum(angular_distance_degrees(hues, center) - half_width, 0.0)
        distances = np.minimum(distances, outside_distance)
    return distances


def _dominant_hues(hist: np.ndarray, min_separation: int = 15, limit: int = 3) -> list[int]:
    candidates = np.argsort(hist)[::-1]
    selected: list[int] = []
    for hue in candidates:
        if hist[hue] <= 0:
            break
        if all(min(abs(int(hue) - prev), 360 - abs(int(hue) - prev)) >= min_separation for prev in selected):
            selected.append(int(hue))
        if len(selected) == limit:
            break
    return selected


def harmony_level(distance_degrees: float, colored_pixel_ratio: float) -> str:
    if colored_pixel_ratio < 0.05:
        return "mostly neutral or grayscale"
    if distance_degrees <= DEFAULT_DISTANCE_THRESHOLDS["very_high"]:
        return "very high"
    if distance_degrees <= DEFAULT_DISTANCE_THRESHOLDS["high"]:
        return "high"
    if distance_degrees <= DEFAULT_DISTANCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def harmony_percentile(distance_degrees: float) -> float:
    """Approximate validation-set percentile where higher is better."""
    # Piecewise interpolation from the validation distribution measured on AADB.
    percentiles = np.asarray([0, 10, 20, 25, 50, 75, 80, 90, 100], dtype=np.float32)
    distances = np.asarray([0.0, 0.0011, 0.0187, 0.0405, 0.2564, 0.7898, 0.9125, 1.552, 7.7407], dtype=np.float32)
    worse_or_equal_percentile = float(np.interp(distance_degrees, distances, percentiles))
    return float(np.clip(100.0 - worse_or_equal_percentile, 0.0, 100.0))


def analyze_image(
    image_path: str | Path,
    bins: int = 360,
    rotation_step: int = 2,
    min_saturation: float = 0.12,
    min_value: float = 0.08,
    max_size: int = 256,
    score_scale_degrees: float = 10.0,
) -> dict[str, object]:
    path = Path(image_path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_size, max_size))
        rgb = np.asarray(image, dtype=np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    mask = (saturation >= min_saturation) & (value >= min_value)
    colored_ratio = float(mask.mean())
    if mask.sum() == 0:
        return {
            "best_template": "N",
            "best_rotation_degrees": 0,
            "harmony_distance": 0.0,
            "harmony_score": 1.0,
            "harmony_distance_radians": 0.0,
            "harmony_level": "mostly neutral or grayscale",
            "harmony_percentile": 100.0,
            "colored_pixel_ratio": colored_ratio,
            "dominant_hues": [],
            "hue_histogram": [0.0] * bins,
        }

    hue_values = hue[mask]
    weights = saturation[mask]
    hist, _ = np.histogram(hue_values, bins=bins, range=(0.0, 360.0), weights=weights)
    hist = hist.astype(np.float32)
    hist_sum = float(hist.sum())
    hue_centers = np.linspace(0.5 * 360.0 / bins, 360.0 - 0.5 * 360.0 / bins, bins).astype(np.float32)

    best_template = ""
    best_rotation = 0
    best_distance = float("inf")
    for name, template in HUE_TEMPLATES.items():
        for rotation in range(0, 360, rotation_step):
            distances = distance_to_template(hue_centers, template, float(rotation))
            weighted_distance = float(np.sum(hist * distances) / hist_sum)
            if weighted_distance < best_distance:
                best_distance = weighted_distance
                best_template = name
                best_rotation = rotation

    # The paper defines lower F / distance as more harmonic. The score below
    # keeps that monotonic relation while avoiding the near-constant values
    # produced by a linear 1 - distance/180 mapping on natural images.
    harmony_score = float(np.exp(-best_distance / score_scale_degrees))
    return {
        "best_template": best_template,
        "best_rotation_degrees": int(best_rotation),
        "harmony_distance": best_distance,
        "harmony_distance_radians": float(np.deg2rad(best_distance)),
        "harmony_score": harmony_score,
        "harmony_level": harmony_level(best_distance, colored_ratio),
        "harmony_percentile": harmony_percentile(best_distance),
        "colored_pixel_ratio": colored_ratio,
        "dominant_hues": _dominant_hues(hist),
        "hue_histogram": (hist / hist_sum).round(6).tolist(),
    }
