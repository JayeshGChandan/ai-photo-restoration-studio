"""
Native-resolution compositing.

Blends a restored image back into the original so that ONLY the repaired
(masked) region changes. The mask edges are feathered so the boundary between
repaired and untouched pixels is invisible, and everything outside the mask
keeps its original, full-resolution pixels.

This replaces the old behaviour of downscaling the whole image to 512px and
upscaling it back -- which degraded the entire photo to fix a small defect.
"""

import numpy as np
from PIL import Image, ImageFilter


def feather_mask(mask: Image.Image, size: tuple, radius: int = 3) -> Image.Image:
    """Resize a mask to ``size`` and soften its edges with a Gaussian blur.

    Args:
        mask: Binary PIL mask (white = repair).
        size: Target (width, height) -- should match the original image.
        radius: Feather radius in pixels (0 disables feathering).

    Returns:
        A grayscale (L) mask aligned to ``size`` with soft edges.
    """
    mask = mask.convert("L")
    if mask.size != size:
        # Nearest keeps the hard region intact before we deliberately soften it.
        mask = mask.resize(size, Image.NEAREST)
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    return mask


def composite(
    original: Image.Image,
    restored: Image.Image,
    mask: Image.Image,
    feather_radius: int = 3,
) -> Image.Image:
    """Composite ``restored`` into ``original`` within the feathered ``mask``.

    Args:
        original: Original full-resolution PIL image.
        restored: Restored image (any size; resized to match original).
        mask: Binary PIL mask, white = use restored pixels.
        feather_radius: Edge feathering in pixels for a seamless blend.

    Returns:
        The blended PIL image at the original resolution.
    """
    original = original.convert("RGB")
    restored = restored.convert("RGB")
    if restored.size != original.size:
        restored = restored.resize(original.size, Image.LANCZOS)

    soft = feather_mask(mask, original.size, feather_radius)

    alpha = np.asarray(soft, dtype=np.float32)[..., None] / 255.0
    orig_arr = np.asarray(original, dtype=np.float32)
    rep_arr = np.asarray(restored, dtype=np.float32)

    blended = rep_arr * alpha + orig_arr * (1.0 - alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
