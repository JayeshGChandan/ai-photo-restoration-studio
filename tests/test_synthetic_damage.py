"""Tests for the seeded mask synthesiser and damage fill (synthetic_damage.py)."""

import numpy as np
import pytest

from synthetic_damage import (
    COVERAGE_TOLERANCE, FILL_SIGMA, FILL_VALUE, apply_damage, damage_fill, synthesize_mask,
)

# Landscape and portrait, from the smallest CLIC image (512x384) to the largest.
SIZES = [(512, 384), (1536, 1024), (2048, 1536), (1152, 2048)]
COVERAGES = [0.01, 0.05, 0.15]
SEEDS = [0, 1, 2]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("coverage", COVERAGES)
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_coverage_within_tolerance(size, coverage, seed):
    w, h = size
    mask, info = synthesize_mask(w, h, coverage, image_index=5, seed=seed)
    assert mask.shape == (h, w) and mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0, 255}
    actual = np.count_nonzero(mask) / (w * h)
    assert actual == info["coverage"]
    assert abs(actual / coverage - 1) <= COVERAGE_TOLERANCE
    # A mix of thin strokes and blob-like defects in every mask.
    assert info["n_strokes"] >= 1 and info["n_blobs"] >= 1


@pytest.mark.parametrize("coverage", COVERAGES)
def test_identical_inputs_give_identical_masks(coverage):
    a, info_a = synthesize_mask(2048, 1536, coverage, image_index=3, seed=0)
    b, info_b = synthesize_mask(2048, 1536, coverage, image_index=3, seed=0)
    assert np.array_equal(a, b)
    assert info_a == info_b


def test_mask_depends_on_index_seed_and_coverage():
    base, _ = synthesize_mask(1536, 1024, 0.05, image_index=3, seed=0)
    for kwargs in ({"image_index": 4, "seed": 0}, {"image_index": 3, "seed": 1}):
        other, _ = synthesize_mask(1536, 1024, 0.05, **kwargs)
        assert not np.array_equal(base, other)
    other, _ = synthesize_mask(1536, 1024, 0.15, image_index=3, seed=0)
    assert not np.array_equal(base, other)


def test_fill_is_deterministic_and_seeded():
    a = damage_fill(300, 400, image_index=2, coverage=0.05, seed=0)
    b = damage_fill(300, 400, image_index=2, coverage=0.05, seed=0)
    c = damage_fill(300, 400, image_index=2, coverage=0.05, seed=1)
    assert a.dtype == np.uint8 and a.shape == (300, 400, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fill_statistics():
    # 3M samples: the standard error of the mean is ~0.006, so these bounds
    # are many SEs wide. Clipping at 255 (2 sigma) barely moves mean or std.
    fill = damage_fill(1000, 1000, image_index=0, coverage=0.05, seed=0).astype(np.float64)
    assert abs(fill.mean() - FILL_VALUE) < 0.1
    assert abs(fill.std() - FILL_SIGMA) < 0.2


def test_apply_damage_only_touches_masked_pixels():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (384, 512, 3), dtype=np.uint8)
    mask, _ = synthesize_mask(512, 384, 0.05, image_index=0, seed=0)
    fill = damage_fill(384, 512, image_index=0, coverage=0.05, seed=0)
    damaged = apply_damage(image, mask, fill)
    inside = mask > 0
    assert np.array_equal(damaged[~inside], image[~inside])
    assert np.array_equal(damaged[inside], fill[inside])
    assert not np.shares_memory(damaged, image)
