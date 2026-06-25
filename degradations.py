"""
Synthetic degradations for the benchmark.

To measure restoration quality objectively we need a ground truth, which real
damaged photos don't have. So we take a *clean* image, apply known, controllable
degradations to it, restore the result, and compare back to the original with
full-reference metrics (PSNR / SSIM / LPIPS).

The degradations mimic how real photos go bad, applied in a plausible order:
optical blur -> resolution loss -> sensor noise -> JPEG compression.
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from PIL import Image


@dataclass
class DegradationResult:
    image: Image.Image          # the degraded image (may be smaller than input)
    downscale: int              # downscale factor applied (1 = none)
    applied: List[str] = field(default_factory=list)


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    return cv2.GaussianBlur(arr, (0, 0), float(sigma))


def downscale(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    return cv2.resize(arr, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)


def add_gaussian_noise(arr: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    if sigma <= 0:
        return arr
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(arr: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return arr
    ok, enc = cv2.imencode(".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return arr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec[:, :, ::-1]  # BGR -> RGB


def degrade(
    image: Image.Image,
    downscale_factor: int = 2,
    blur_sigma: float = 1.0,
    noise_sigma: float = 8.0,
    jpeg_quality: int = 60,
    seed: int = 0,
) -> DegradationResult:
    """Apply the configured degradations to a clean image."""
    arr = np.asarray(image.convert("RGB"))
    applied: List[str] = []

    if blur_sigma > 0:
        arr = gaussian_blur(arr, blur_sigma)
        applied.append(f"blur sigma {blur_sigma:g}")
    if downscale_factor > 1:
        arr = downscale(arr, downscale_factor)
        applied.append(f"downscale {downscale_factor}x")
    if noise_sigma > 0:
        arr = add_gaussian_noise(arr, noise_sigma, seed=seed)
        applied.append(f"noise sigma {noise_sigma:g}")
    if jpeg_quality < 100:
        arr = jpeg_compress(arr, jpeg_quality)
        applied.append(f"JPEG q{jpeg_quality}")

    if not applied:
        applied.append("none")
    return DegradationResult(
        image=Image.fromarray(arr, "RGB"),
        downscale=int(downscale_factor),
        applied=applied,
    )
