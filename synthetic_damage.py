"""
Seeded synthetic damage for the compositing experiment.

The experiment needs damage masks with a controlled area and a known damaged
image D to repair. The existing mask code does not fit: mask_generator.py
*detects* damage from image content (edges, thresholds) with no seed or
coverage control, and masking.py only converts Gradio brush strokes.

``synthesize_mask`` draws a mix of thin scratch-like strokes (random-walk
polylines) and blob-like defects (irregular filled polygons) until the mask
covers a target fraction of the image, landing within ``COVERAGE_TOLERANCE``
(relative) of it. The output is fully determined by (width, height, coverage,
image_index, seed). ``damage_fill`` / ``apply_damage`` overwrite the masked
pixels with a flat near-white value plus seeded Gaussian noise.
"""

import math
from typing import Dict, Tuple

import cv2
import numpy as np

# Relative tolerance on achieved coverage (|actual / target - 1|).
COVERAGE_TOLERANCE = 0.10
# Internal acceptance band the synthesiser aims for (well inside the tolerance).
_INNER_BAND = 0.02

# Damage fill for D: flat near-white with Gaussian noise, clipped to [0, 255].
FILL_VALUE = 235
FILL_SIGMA = 10.0

# Shape parameters (fractions of the image's short / long edge). Recorded in the
# experiment metadata so a run can be reproduced exactly.
MASK_PARAMS: Dict[str, object] = {
    "stroke_share": 0.4,                 # strokes until 40% of target area, then blobs
    "stroke_segments": [3, 8],           # inclusive range of polyline segments
    "stroke_segment_len_long_edge": [0.03, 0.10],
    "stroke_turn_sigma_rad": 0.35,
    "stroke_thickness_short_edge": [0.002, 0.005],
    "stroke_min_thickness_px": 2,
    "blob_expected_count": 6,
    "blob_radius_jitter": [0.6, 1.4],
    "blob_vertices": 16,
    "blob_radial_noise": 0.35,
    "overshoot_shrink": 0.6,
    "inner_band": _INNER_BAND,
    "coverage_tolerance": COVERAGE_TOLERANCE,
    "line_type": "cv2.LINE_8 (binary, no anti-aliasing)",
}

_MAX_CANDIDATES = 20000
# Stream ids keep the mask and fill RNGs independent for the same key.
_STREAM_MASK = 0
_STREAM_FILL = 1


def _rng(seed: int, image_index: int, coverage: float, stream: int) -> np.random.Generator:
    key = [int(seed), int(image_index), int(round(coverage * 1_000_000)), int(stream)]
    return np.random.default_rng(np.random.SeedSequence(key))


def _draw_stroke(canvas: np.ndarray, rng: np.random.Generator, scale: float) -> None:
    h, w = canvas.shape
    short, long_ = min(w, h), max(w, h)
    p = MASK_PARAMS
    n_seg = int(rng.integers(p["stroke_segments"][0], p["stroke_segments"][1] + 1))
    seg_len = long_ * rng.uniform(*p["stroke_segment_len_long_edge"]) * scale
    thickness = max(p["stroke_min_thickness_px"],
                    int(round(short * rng.uniform(*p["stroke_thickness_short_edge"]))))
    x, y = rng.uniform(0, w), rng.uniform(0, h)
    angle = rng.uniform(0, 2 * math.pi)
    pts = [(x, y)]
    for _ in range(n_seg):
        angle += rng.normal(0.0, p["stroke_turn_sigma_rad"])
        x += seg_len * math.cos(angle)
        y += seg_len * math.sin(angle)
        pts.append((x, y))
    cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 255, thickness, cv2.LINE_8)


def _draw_blob(canvas: np.ndarray, rng: np.random.Generator, base_radius: float,
               scale: float) -> None:
    h, w = canvas.shape
    p = MASK_PARAMS
    k = p["blob_vertices"]
    r0 = base_radius * scale * rng.uniform(*p["blob_radius_jitter"])
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    # Smooth radial noise: circularly average white noise so the outline is
    # irregular but not spiky.
    noise = rng.normal(0.0, 1.0, k)
    noise = (np.roll(noise, 1) + noise + np.roll(noise, -1)) / 3.0
    radii = r0 * np.clip(1.0 + p["blob_radial_noise"] * noise, 0.3, None)
    angles = np.linspace(0, 2 * math.pi, k, endpoint=False) + rng.uniform(0, 2 * math.pi / k)
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 255, cv2.LINE_8)


def synthesize_mask(width: int, height: int, coverage: float, image_index: int,
                    seed: int = 0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Synthesize a binary damage mask covering ``coverage`` of the image.

    Args:
        width, height: Mask size in pixels.
        coverage: Target fraction of the image area, e.g. 0.01, 0.05, 0.15.
        image_index: Index of the image in the experiment (part of the RNG key).
        seed: Experiment seed (part of the RNG key).

    Returns:
        ``(mask, info)``: ``mask`` is uint8 (height, width) with values 0/255;
        ``info`` has the achieved ``coverage``, stroke/blob counts and the
        coverage reached by strokes alone.

    Raises:
        RuntimeError: if the target band cannot be reached.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    rng = _rng(seed, image_index, coverage, _STREAM_MASK)
    total = width * height
    target_px = coverage * total
    lo, hi = target_px * (1 - _INNER_BAND), target_px * (1 + _INNER_BAND)
    stroke_goal = MASK_PARAMS["stroke_share"] * target_px
    blob_area = (1 - MASK_PARAMS["stroke_share"]) * target_px / MASK_PARAMS["blob_expected_count"]
    base_radius = math.sqrt(blob_area / math.pi)

    mask = np.zeros((height, width), dtype=np.uint8)
    area = 0
    n_strokes = n_blobs = 0
    stroke_px = None
    scale = 1.0
    for _ in range(_MAX_CANDIDATES):
        if area >= lo:
            break
        doing_strokes = area < stroke_goal and stroke_px is None
        if not doing_strokes and stroke_px is None:
            stroke_px = area
        cand = mask.copy()
        if doing_strokes:
            _draw_stroke(cand, rng, scale)
        else:
            _draw_blob(cand, rng, base_radius, scale)
        new_area = int(np.count_nonzero(cand))
        if new_area > hi:
            scale *= MASK_PARAMS["overshoot_shrink"]
            continue
        if new_area > area:
            mask, area = cand, new_area
            if doing_strokes:
                n_strokes += 1
            else:
                n_blobs += 1
        scale = 1.0
    else:
        raise RuntimeError(
            f"could not reach coverage {coverage} on {width}x{height} "
            f"(got {area / total:.4f}) after {_MAX_CANDIDATES} candidates")
    if stroke_px is None:
        stroke_px = area
    if n_strokes == 0 or n_blobs == 0:
        raise RuntimeError(f"mask lacks a shape type: {n_strokes} strokes, {n_blobs} blobs")

    actual = area / total
    if abs(actual / coverage - 1) > COVERAGE_TOLERANCE:
        raise RuntimeError(f"coverage {actual:.5f} outside tolerance of target {coverage}")
    info = {
        "coverage": actual,
        "n_strokes": n_strokes,
        "n_blobs": n_blobs,
        "stroke_coverage": stroke_px / total,
    }
    return mask, info


def damage_fill(height: int, width: int, image_index: int, coverage: float,
                seed: int = 0) -> np.ndarray:
    """Seeded fill: ``FILL_VALUE`` + N(0, ``FILL_SIGMA``), rounded, clipped, uint8 RGB."""
    rng = _rng(seed, image_index, coverage, _STREAM_FILL)
    noise = rng.normal(0.0, FILL_SIGMA, (height, width, 3))
    return np.clip(np.rint(FILL_VALUE + noise), 0, 255).astype(np.uint8)


def apply_damage(image: np.ndarray, mask: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` (uint8 RGB) with pixels inside ``mask`` taken from ``fill``."""
    if image.shape != fill.shape or image.shape[:2] != mask.shape:
        raise ValueError("image, mask and fill shapes do not match")
    out = image.copy()
    inside = mask > 0
    out[inside] = fill[inside]
    return out
