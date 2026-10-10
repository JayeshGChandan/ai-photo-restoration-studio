"""
Identity tests for native-resolution feathered compositing.

``composite()`` blends ``restored`` into ``original`` with
alpha = feather_mask(mask) / 255. These tests pin down, with zero tolerance,
which pixels are allowed to change:

  1. where the feathered alpha is exactly 0, the output equals the original;
  2. where it is exactly 255, the output equals the restored image;
  3. when restored == original, the output equals the original everywhere;
  4. every changed pixel lies inside the support (alpha > 0) of the feather.

Inputs are random uint8 images with fixed seeds at two sizes: one at or below
LaMa's 1536 px long-edge cap and one above it. For each case the test for (4)
also prints how far, in Euclidean pixels, changes reach beyond the binary mask.

Run with:  ./venv/bin/python -m pytest -s
"""

from functools import lru_cache

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from compositing import composite, feather_mask

# (width, height, seed): long edge 1536 (at the cap) and 2400 (above it).
SIZES = [(1536, 1024, 0), (2400, 1600, 1)]
RADII = [0, 3, 15]
CASES = [(w, h, seed, r) for (w, h, seed) in SIZES for r in RADII]


def _case_id(case):
    w, h, _, r = case
    return f"{w}x{h}-r{r}"


@lru_cache(maxsize=None)
def _inputs(w, h, seed):
    """Random original / restored images and a binary mask, kept off the borders."""
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    restored = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    # A block large enough that its interior stays at alpha 255 at r=15,
    # a thin scratch-like line, and a disk.
    mask[int(0.20 * h):int(0.45 * h), int(0.10 * w):int(0.35 * w)] = 255
    cv2.line(mask, (int(0.45 * w), int(0.15 * h)), (int(0.85 * w), int(0.80 * h)), 255, 9)
    cv2.circle(mask, (int(0.25 * w), int(0.70 * h)), int(0.08 * h), 255, -1)
    return original, restored, mask


@lru_cache(maxsize=None)
def _run(w, h, seed, radius):
    original, restored, mask = _inputs(w, h, seed)
    mask_img = Image.fromarray(mask, "L")
    alpha = np.asarray(feather_mask(mask_img, (w, h), radius))
    out = np.asarray(composite(Image.fromarray(original), Image.fromarray(restored),
                               mask_img, feather_radius=radius))
    same = np.asarray(composite(Image.fromarray(original), Image.fromarray(original.copy()),
                                mask_img, feather_radius=radius))
    return original, restored, mask > 0, alpha, out, same


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_alpha_zero_pixels_equal_original(case):
    original, _, _, alpha, out, _ = _run(*case)
    untouched = alpha == 0
    assert untouched.any()
    assert np.array_equal(out[untouched], original[untouched])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_alpha_full_pixels_equal_restored(case):
    _, restored, _, alpha, out, _ = _run(*case)
    full = alpha == 255
    assert full.any()
    assert np.array_equal(out[full], restored[full])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_restored_equal_to_original_is_a_no_op(case):
    original, _, _, _, _, same = _run(*case)
    assert np.array_equal(same, original)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_changes_stay_inside_feather_support(case, capsys):
    original, _, mask, alpha, out, _ = _run(*case)
    changed = (out != original).any(axis=-1)
    assert changed.any()
    assert not (changed & (alpha == 0)).any()

    # Distance from every pixel to the nearest pixel of the binary mask.
    dist = ndimage.distance_transform_edt(~mask)
    outside = changed & ~mask
    extension = float(dist[outside].max()) if outside.any() else 0.0
    w, h, _, radius = case
    with capsys.disabled():
        print(f"\n[{w}x{h}] feather radius {radius:2d}: changes extend up to "
              f"{extension:.2f} px beyond the binary mask "
              f"({int(outside.sum())} changed px outside it)")
