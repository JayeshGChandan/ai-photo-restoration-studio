"""
Native-resolution compositing.

Blends a restored image back into the original. Every masked pixel takes the
restored value in full; the feather (a soft transition from restored to
original) lies entirely outside the mask, so the boundary is invisible, and
everything beyond the feather keeps its original, full-resolution pixels.

This replaces the old behaviour of downscaling the whole image to 512px and
upscaling it back -- which degraded the entire photo to fix a small defect.
"""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageFilter


def _blur(mask: Image.Image, radius) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius))


@lru_cache(maxsize=None)
def feather_reach(radius) -> int:
    """Dilation (px, Chebyshev) that keeps the feather wholly outside the mask.

    Derivation: ``feather_mask`` blurs with Pillow's GaussianBlur, which is three
    box-filter passes per axis (extended box blur, Gwosdek et al. 2011), so its
    kernel is a separable, compactly supported, non-negative filter normalised
    to 1. The output at pixel p is therefore 255 exactly when every input pixel
    in the kernel's footprint around p, the square of Chebyshev radius R
    centred on p, is 255. (Edges are padded by repeating the border pixel, so a
    border pixel's footprint still lies inside the dilated region.) Dilating
    the binary mask by a (2R+1)x(2R+1) square before blurring then guarantees
    alpha == 255 on every original mask pixel, and all of the soft edge falls
    outside the mask.

    R is not hand-picked: it is the kernel's *effective* reach, measured by
    applying the same blur to a white square of half-width d on black and
    taking the smallest d whose centre comes out as exactly 255 (the
    furthest kernel taps carry weight too small to move the 8-bit output, so
    the effective reach is a few px below Pillow's nominal 3*ceil(box radius)
    footprint). That is also the minimal dilation: with R - 1, the centre of an
    isolated mask pixel comes out below 255. Measured with Pillow 9.5.0:
    r=3 -> 8 px, r=15 -> 40 px.
    """
    if radius <= 0:
        return 0
    d = 0
    while True:
        side = 2 * d + 3                   # square of half-width d plus a 1 px black margin
        probe = np.zeros((side, side), dtype=np.uint8)
        probe[1:-1, 1:-1] = 255
        if np.asarray(_blur(Image.fromarray(probe, "L"), radius))[d + 1, d + 1] == 255:
            return d
        d += 1


def feather_mask(mask: Image.Image, size: tuple, radius: int = 3) -> Image.Image:
    """Resize a mask to ``size`` and feather it outward with a Gaussian blur.

    The mask is first dilated by ``feather_reach(radius)`` so that after the
    blur the alpha is exactly 255 on every original mask pixel and the soft
    edge lies entirely outside the mask.

    Args:
        mask: Binary PIL mask (white = repair).
        size: Target (width, height) -- should match the original image.
        radius: Feather radius in pixels (0 disables feathering).

    Returns:
        A grayscale (L) mask aligned to ``size`` with soft outer edges.
    """
    mask = mask.convert("L")
    if mask.size != size:
        # Nearest keeps the hard region intact before we deliberately soften it.
        mask = mask.resize(size, Image.NEAREST)
    if radius > 0:
        reach = feather_reach(radius)
        if reach > 0:
            # Square (Chebyshev) dilation by `reach` px; pixels beyond the
            # border are ignored, so the result stays within the image.
            kernel = np.ones((2 * reach + 1, 2 * reach + 1), dtype=np.uint8)
            mask = Image.fromarray(cv2.dilate(np.asarray(mask), kernel), "L")
        mask = _blur(mask, radius)
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
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8), "RGB")
