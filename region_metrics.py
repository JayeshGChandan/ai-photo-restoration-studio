"""
Region-restricted scoring for the compositing experiment.

Every function scores a candidate X against the clean image I, both uint8 RGB
arrays of shape (H, W, 3), over a boolean region of shape (H, W):

  * ``region_diff_stats`` -- how much X departs from I on a region: share of
    pixels changed in any channel, max / mean absolute difference and MSE
    (0-255 scale, averaged over pixels x channels). Integer arithmetic in
    int64, so nothing can overflow.
  * ``psnr_y_region`` -- PSNR on BT.601 luma over a region, the same luma
    (Y in [16, 235] / 255, data range 1.0) as the benchmark's psnr_y metric.
  * ``lpips_inside`` / ``lpips_outside`` -- LPIPS of X spliced into I: only
    the repair (X inside M, I elsewhere) or only the rest (X outside M, I
    inside), using the same LPIPS model as the benchmark.
"""

from typing import Dict

import numpy as np
from PIL import Image

# ITU-R BT.601 RGB -> Y for RGB in [0, 1], matching pyiqa's rgb2ycbcr.
_BT601_WEIGHTS = np.array([65.481, 128.553, 24.966], dtype=np.float64)
_BT601_OFFSET = 16.0


def _check(x: np.ndarray, ref: np.ndarray, region: np.ndarray) -> None:
    if x.dtype != np.uint8 or ref.dtype != np.uint8:
        raise TypeError("x and ref must be uint8")
    if x.shape != ref.shape or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"x {x.shape} and ref {ref.shape} must both be (H, W, 3)")
    if region.dtype != bool or region.shape != x.shape[:2]:
        raise ValueError(f"region must be a bool array of shape {x.shape[:2]}")
    if not region.any():
        raise ValueError("region is empty")


def region_diff_stats(x: np.ndarray, ref: np.ndarray, region: np.ndarray) -> Dict[str, float]:
    """Difference statistics of ``x`` vs ``ref`` over ``region``.

    Returns ``n_pixels``, ``pct_changed`` (percent of region pixels where any
    channel differs), ``max_abs``, ``mean_abs`` and ``mse``; the last two are
    averaged over region pixels x 3 channels on the 0-255 scale.
    """
    _check(x, ref, region)
    d = x[region].astype(np.int64) - ref[region].astype(np.int64)  # (n, 3)
    n = d.shape[0]
    abs_d = np.abs(d)
    return {
        "n_pixels": int(n),
        "pct_changed": 100.0 * int(np.count_nonzero(abs_d.max(axis=1))) / n,
        "max_abs": int(abs_d.max()),
        "mean_abs": int(abs_d.sum()) / (3 * n),
        "mse": int((d * d).sum()) / (3 * n),
    }


def luma_bt601(x: np.ndarray) -> np.ndarray:
    """BT.601 luma of a uint8 RGB array, in [16/255, 235/255] (float64)."""
    rgb = x.astype(np.float64) / 255.0
    return (_BT601_OFFSET + rgb @ _BT601_WEIGHTS) / 255.0


def psnr_y_region(x: np.ndarray, ref: np.ndarray, region: np.ndarray) -> float:
    """PSNR (dB, data range 1.0) on BT.601 luma over ``region``.

    Returns ``inf`` when the region's luma is identical.
    """
    _check(x, ref, region)
    diff = luma_bt601(x)[region] - luma_bt601(ref)[region]
    mse = float(np.mean(diff * diff))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def splice(inside: np.ndarray, outside: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``inside`` where ``mask`` is True, ``outside`` elsewhere (new array)."""
    if inside.shape != outside.shape or mask.shape != inside.shape[:2]:
        raise ValueError("splice: shape mismatch")
    return np.where(mask[..., None], inside, outside)


def _lpips(candidate: np.ndarray, ref: np.ndarray) -> float:
    from metrics import get_scorer

    # Same LPIPS instance and preprocessing as the benchmark's full-reference
    # scores; returns None on failure, which callers must treat as an error.
    value = get_scorer()._score_ref_one("lpips", Image.fromarray(candidate), Image.fromarray(ref))
    if value is None:
        raise RuntimeError("LPIPS returned None")
    return float(value)


def lpips_inside(x: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS of X_in vs I, where X_in = X inside ``mask`` and I elsewhere."""
    _check(x, ref, mask)
    return _lpips(splice(x, ref, mask), ref)


def lpips_outside(x: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS of X_out vs I, where X_out = X outside ``mask`` and I inside it."""
    _check(x, ref, mask)
    return _lpips(splice(ref, x, mask), ref)


def region_diff_tail(x: np.ndarray, ref: np.ndarray, region: np.ndarray) -> Dict[str, float]:
    """Tail of the difference distribution of ``x`` vs ``ref`` over ``region``.

    Returns ``pct_changed_gt1``, the percent of region pixels where any channel
    differs by more than 1 level, and ``p99_abs``, the 99th percentile of the
    per-channel absolute difference over region pixels x 3 channels (the same
    population as ``mean_abs``), taken as the smallest value v with at least 99%
    of values <= v (numpy's ``inverted_cdf`` method). Exact integer arithmetic.
    """
    _check(x, ref, region)
    abs_d = np.abs(x[region].astype(np.int16) - ref[region].astype(np.int16))  # (n, 3), 0..255
    n = abs_d.shape[0]
    cum = np.cumsum(np.bincount(abs_d.ravel(), minlength=256))
    total = int(cum[-1])
    p99 = int(np.argmax(cum * 100 >= 99 * total))
    return {
        "pct_changed_gt1": 100.0 * int(np.count_nonzero(abs_d.max(axis=1) > 1)) / n,
        "p99_abs": p99,
    }
