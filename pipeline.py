"""
Restoration pipeline orchestrator.

Given an image and a set of (possibly auto-recommended) settings, this runs the
"route -> restore" stage: it upscales the background with Real-ESRGAN and/or
restores faces with CodeFormer, ordering the steps so faces are pasted in at the
final resolution. The localized brush/LaMa path lives in the Brush Repair tab.
"""

from typing import Optional, Tuple

from PIL import Image

from restorers.upscale import get_upscaler
from restorers.face import get_face_restorer


def restore(
    image: Image.Image,
    do_face: bool = True,
    fidelity: float = 0.7,
    do_upscale: bool = True,
    outscale: int = 2,
) -> Tuple[Image.Image, str]:
    """Run the routed restoration pipeline and return (result, status).

    Combinations:
      * face + upscale: Real-ESRGAN produces the high-res background, then the
        restored faces are pasted in at the upscaled resolution (best quality).
      * upscale only: Real-ESRGAN on the whole image.
      * face only: CodeFormer at native resolution.
      * neither: returns the input unchanged.
    """
    if image is None:
        raise ValueError("No image provided.")
    image = image.convert("RGB")
    outscale = int(outscale)
    steps = []

    if do_face and do_upscale and outscale > 1:
        background = get_upscaler().enhance(image, outscale=outscale)
        result, info = get_face_restorer().restore(
            image, fidelity=fidelity, upscale=outscale, background_image=background
        )
        steps.append(f"Real-ESRGAN x{outscale}")
        if info["num_faces"]:
            steps.append(f"CodeFormer on {info['num_faces']} face(s) @ fidelity {fidelity:.2f}")
        else:
            steps.append("no faces found (upscale only)")

    elif do_upscale and outscale > 1:
        result = get_upscaler().enhance(image, outscale=outscale)
        steps.append(f"Real-ESRGAN x{outscale}")

    elif do_face:
        result, info = get_face_restorer().restore(image, fidelity=fidelity, upscale=1)
        if info["num_faces"]:
            steps.append(f"CodeFormer on {info['num_faces']} face(s) @ fidelity {fidelity:.2f}")
        else:
            steps.append("no faces found (image unchanged)")

    else:
        result = image
        steps.append("no operation selected")

    status = " -> ".join(steps)
    return result, status
