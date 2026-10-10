"""Tests for region-restricted scoring (region_metrics.py) on arrays with known differences."""

import math

import numpy as np
import pytest
from PIL import Image

import region_metrics
from region_metrics import (
    lpips_inside, lpips_outside, luma_bt601, psnr_y_region, region_diff_stats, region_diff_tail,
    splice,
)


def _pair():
    """4x5 image of value 100 and a copy with two known pixel changes."""
    ref = np.full((4, 5, 3), 100, dtype=np.uint8)
    x = ref.copy()
    x[0, 0] = [103, 100, 100]   # +3 in R
    x[1, 1] = [100, 95, 101]    # -5 in G, +1 in B
    return x, ref


def test_diff_stats_full_region_exact():
    x, ref = _pair()
    s = region_diff_stats(x, ref, np.ones((4, 5), bool))
    assert s == {"n_pixels": 20, "pct_changed": 10.0, "max_abs": 5,
                 "mean_abs": 9 / 60, "mse": 35 / 60}


def test_diff_stats_subregion_exact():
    x, ref = _pair()
    region = np.zeros((4, 5), bool)
    region[0, :3] = True  # pixel (0,0) changed, (0,1) and (0,2) unchanged
    s = region_diff_stats(x, ref, region)
    assert s == {"n_pixels": 3, "pct_changed": 100 / 3, "max_abs": 3,
                 "mean_abs": 3 / 9, "mse": 9 / 9}


def test_diff_stats_unchanged_region_is_zero():
    x, ref = _pair()
    region = np.zeros((4, 5), bool)
    region[2:, :] = True
    s = region_diff_stats(x, ref, region)
    assert s == {"n_pixels": 10, "pct_changed": 0.0, "max_abs": 0, "mean_abs": 0.0, "mse": 0.0}


@pytest.mark.parametrize("x_val,ref_val", [(255, 0), (0, 255)])
def test_diff_stats_cannot_overflow(x_val, ref_val):
    # uint8 arithmetic would wrap (0 - 255 == 1); int64 must give the true values.
    x = np.full((1000, 1000, 3), x_val, dtype=np.uint8)
    ref = np.full((1000, 1000, 3), ref_val, dtype=np.uint8)
    s = region_diff_stats(x, ref, np.ones((1000, 1000), bool))
    assert s == {"n_pixels": 1_000_000, "pct_changed": 100.0, "max_abs": 255,
                 "mean_abs": 255.0, "mse": 65025.0}


def test_empty_region_raises():
    x, ref = _pair()
    with pytest.raises(ValueError):
        region_diff_stats(x, ref, np.zeros((4, 5), bool))


def test_luma_matches_bt601_definition():
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]]],
                  dtype=np.uint8)
    y = luma_bt601(px)[0] * 255.0
    expected = [16 + 65.481, 16 + 128.553, 16 + 24.966, 16.0, 235.0]
    assert y == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("k", [1, 4, 37])
def test_psnr_y_known_value(k):
    # Grey images differing by k in every channel: Y differs by k * 219 / 255^2
    # everywhere (65.481 + 128.553 + 24.966 = 219), so PSNR = -20 log10(219k / 65025).
    ref = np.full((6, 7, 3), 90, dtype=np.uint8)
    x = ref + np.uint8(k)
    expected = -20 * math.log10(219 * k / 65025)
    # Closed form vs float64 luma arithmetic: equal up to float rounding.
    assert psnr_y_region(x, ref, np.ones((6, 7), bool)) == pytest.approx(expected, rel=1e-12, abs=0)


def test_psnr_y_is_restricted_to_region():
    ref = np.full((6, 7, 3), 90, dtype=np.uint8)
    x = ref.copy()
    region = np.zeros((6, 7), bool)
    region[:3] = True
    x[3:] = 0                       # large change outside the region only
    assert psnr_y_region(x, ref, region) == math.inf
    x[:3] = 91                      # +1 everywhere inside the region
    expected = -20 * math.log10(219 / 65025)
    assert psnr_y_region(x, ref, region) == pytest.approx(expected, rel=1e-12, abs=0)


def test_psnr_y_agrees_with_benchmark_metric():
    # Cross-implementation check against pyiqa's psnr_y (float32, eps 1e-8).
    from metrics import get_scorer

    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    x = np.clip(ref.astype(int) + rng.integers(-20, 21, ref.shape), 0, 255).astype(np.uint8)
    ours = psnr_y_region(x, ref, np.ones((64, 80), bool))
    pyiqa_value = get_scorer().score_reference(Image.fromarray(x), Image.fromarray(ref))["psnr_y"]
    assert ours == pytest.approx(pyiqa_value, abs=1e-3)


def test_splice_exact():
    a = np.zeros((3, 4, 3), np.uint8)
    b = np.full((3, 4, 3), 200, np.uint8)
    m = np.zeros((3, 4), bool)
    m[1, 2] = True
    out = splice(a, b, m)
    expected = b.copy()
    expected[1, 2] = 0
    assert np.array_equal(out, expected)


def test_lpips_receives_the_spliced_images(monkeypatch):
    seen = []
    monkeypatch.setattr(region_metrics, "_lpips", lambda c, r: seen.append((c, r)) or 0.5)
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
    x = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
    m = np.zeros((8, 9), bool)
    m[2:5, 3:7] = True
    assert lpips_inside(x, ref, m) == 0.5
    assert lpips_outside(x, ref, m) == 0.5
    (in_c, in_r), (out_c, out_r) = seen
    assert np.array_equal(in_c[m], x[m]) and np.array_equal(in_c[~m], ref[~m])
    assert np.array_equal(out_c[~m], x[~m]) and np.array_equal(out_c[m], ref[m])
    assert in_r is ref and out_r is ref


def test_lpips_ignores_changes_on_the_other_side():
    # Real LPIPS: changes confined to one side leave the other score at exactly 0.
    rng = np.random.default_rng(2)
    ref = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    m = np.zeros((96, 128), bool)
    m[30:60, 40:90] = True
    outside_only = ref.copy()
    outside_only[~m] = 255 - outside_only[~m]
    inside_only = ref.copy()
    inside_only[m] = 255 - inside_only[m]
    assert lpips_inside(outside_only, ref, m) == 0.0
    assert lpips_outside(inside_only, ref, m) == 0.0
    assert lpips_inside(inside_only, ref, m) > 0.0
    assert lpips_outside(outside_only, ref, m) > 0.0


def test_diff_tail_exact():
    x, ref = _pair()   # pixel (0,0): +3 R; pixel (1,1): -5 G, +1 B; 18 pixels unchanged
    # 60 channel values: 57 zeros, then 1, 3, 5. The 99th percentile is the
    # ceil(0.99 * 60) = 60th smallest value, 5.
    assert region_diff_tail(x, ref, np.ones((4, 5), bool)) == {"pct_changed_gt1": 10.0,
                                                                "p99_abs": 5}


def test_diff_tail_ramp():
    # 100 pixels with R differing by i (i = 0..99), G and B unchanged: 300 values,
    # 201 zeros then 1..99 once each. ceil(0.99 * 300) = 297th smallest = 96.
    ref = np.zeros((10, 10, 3), dtype=np.uint8)
    x = ref.copy()
    x[..., 0] = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert region_diff_tail(x, ref, np.ones((10, 10), bool)) == {"pct_changed_gt1": 98.0,
                                                                  "p99_abs": 96}


def test_diff_tail_counts_only_changes_above_one_level():
    ref = np.full((2, 3, 3), 50, dtype=np.uint8)
    x = ref.copy()
    x[0, 0] = [51, 49, 51]      # every channel off by exactly 1: not counted
    x[0, 1] = [50, 52, 50]      # off by 2: counted
    out = region_diff_tail(x, ref, np.ones((2, 3), bool))
    assert out["pct_changed_gt1"] == 100 / 6
    assert region_diff_stats(x, ref, np.ones((2, 3), bool))["pct_changed"] == 200 / 6


@pytest.mark.parametrize("x_val,ref_val", [(255, 0), (0, 255)])
def test_diff_tail_cannot_overflow(x_val, ref_val):
    x = np.full((300, 300, 3), x_val, dtype=np.uint8)
    ref = np.full((300, 300, 3), ref_val, dtype=np.uint8)
    assert region_diff_tail(x, ref, np.ones((300, 300), bool)) == {"pct_changed_gt1": 100.0,
                                                                    "p99_abs": 255}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diff_tail_p99_matches_numpy_inverted_cdf(seed):
    rng = np.random.default_rng(seed)
    ref = rng.integers(0, 256, (123, 157, 3), dtype=np.uint8)
    noise = np.rint(rng.laplace(0, 4, ref.shape)).astype(int)
    x = np.clip(ref.astype(int) + noise, 0, 255).astype(np.uint8)
    region = rng.random((123, 157)) > 0.3
    abs_d = np.abs(x[region].astype(int) - ref[region].astype(int)).ravel()
    expected = int(np.percentile(abs_d, 99, method="inverted_cdf"))
    assert region_diff_tail(x, ref, region)["p99_abs"] == expected
