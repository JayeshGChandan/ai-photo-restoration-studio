"""
Frozen copy of the pre-fix compositing logic, for the compositing experiment only.

This is compositing.py as of commit 3bece00a3bbd02b2e4bd062249b54dbcb2dc3760
(blob bdd95c117cbddf7e269dcf524c9c8042db1a21a3), before the feather was moved
outside the mask. Its Gaussian feather is centred on the mask edge, so inside
the mask the alpha is below 255 near the boundary and the original (damaged)
image bleeds back into the repair.

It lives in scripts/ so the app cannot pick it up: app.py and pipeline.py run
with the repo root on sys.path, not scripts/. tests/test_legacy_compositing.py
checks it byte for byte against the frozen blob and that neither app.py nor
pipeline.py imports it. Do not edit the two functions below.
"""

import numpy as np
from PIL import Image, ImageFilter


def legacy_feather_mask(mask: Image.Image, size: tuple, radius: int = 3) -> Image.Image:
    """Pre-fix ``feather_mask``: resize to ``size`` (nearest) and Gaussian-blur the edge."""
    mask = mask.convert("L")
    if mask.size != size:
        # Nearest keeps the hard region intact before we deliberately soften it.
        mask = mask.resize(size, Image.NEAREST)
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    return mask


def legacy_composite(
    original: Image.Image,
    restored: Image.Image,
    mask: Image.Image,
    feather_radius: int = 3,
) -> Image.Image:
    """Pre-fix ``composite``: blend ``restored`` into ``original`` with the centred feather."""
    original = original.convert("RGB")
    restored = restored.convert("RGB")
    if restored.size != original.size:
        restored = restored.resize(original.size, Image.LANCZOS)

    soft = legacy_feather_mask(mask, original.size, feather_radius)

    alpha = np.asarray(soft, dtype=np.float32)[..., None] / 255.0
    orig_arr = np.asarray(original, dtype=np.float32)
    rep_arr = np.asarray(restored, dtype=np.float32)

    blended = rep_arr * alpha + orig_arr * (1.0 - alpha)
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8), "RGB")
