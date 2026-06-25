"""
Masking utilities.

Phase 1 makes interactive brushing the primary way to mark damage: the user
paints over scratches/tears/spots and that brush stroke becomes the repair mask.
The classical auto-detector (mask_generator.py) is kept only as an optional
*suggestion* overlay the user can reference while brushing -- it is no longer
trusted as ground truth, because edge/threshold heuristics flag far too much
healthy content as "damage".
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from mask_generator import mask_from_pil, create_visualization


def mask_from_editor(editor_value: Optional[dict]) -> Optional[Image.Image]:
    """Extract a binary repair mask from a Gradio ImageEditor value.

    The editor value is a dict with ``background`` (the image) and ``layers``
    (brush strokes as RGBA images). Any painted pixel becomes part of the mask.

    Args:
        editor_value: The dict returned by gr.ImageEditor(type="pil").

    Returns:
        A binary PIL mask (white = repair), or None if nothing was painted.
    """
    if not editor_value:
        return None

    background = editor_value.get("background")
    layers = editor_value.get("layers") or []
    if background is None:
        return None

    width, height = background.size
    mask = np.zeros((height, width), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue
        arr = np.array(layer.convert("RGBA"))
        if arr.shape[:2] != (height, width):
            layer = layer.resize((width, height), Image.NEAREST)
            arr = np.array(layer.convert("RGBA"))
        # A painted pixel is any with non-trivial alpha.
        painted = arr[..., 3] > 10
        mask = np.maximum(mask, painted.astype(np.uint8) * 255)

    if mask.max() == 0:
        return None
    return Image.fromarray(mask, "L")


def suggest_mask(
    image: Image.Image,
    sensitivity: float = 1.0,
    detect_bright: bool = True,
    detect_dark: bool = True,
    min_area: int = 50,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """Run the classical detector to *suggest* damaged regions.

    Returns (mask, overlay_visualization, status). The overlay is meant purely
    as a visual reference to guide manual brushing.
    """
    if image is None:
        return None, None, "Upload an image first."

    try:
        mask_pil, stats = mask_from_pil(
            image.convert("RGB"),
            scratch_sensitivity=sensitivity,
            detect_bright=detect_bright,
            detect_dark=detect_dark,
            min_damage_area=min_area,
        )
    except Exception as e:
        return None, None, f"Suggestion failed: {e}"

    overlay = create_visualization(
        np.array(image.convert("RGB")), np.array(mask_pil)
    )
    overlay_pil = Image.fromarray(overlay)

    pct = stats.get("damage_percentage", 0.0)
    status = (
        f"Suggested {pct:.1f}% of the image as possible damage (red overlay). "
        "This is only a hint -- brush the real damage in the editor."
    )
    return mask_pil, overlay_pil, status
