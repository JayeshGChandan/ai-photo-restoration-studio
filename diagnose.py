"""
Image diagnosis and restoration routing.

Before restoring anything, we look at the image and decide *what kind* of damage
it has, then route to the right tool:

    * faces present            -> CodeFormer face restoration  (restorers/face.py)
    * low-res / soft / noisy   -> Real-ESRGAN upscaling        (restorers/upscale.py)
    * localized scratch/spot   -> LaMa inpainting via brushing (restorers/inpaint.py)

The measurements are deliberately simple, fast, and explainable (sharpness,
noise, resolution, face coverage) so the UI can show *why* a given pipeline was
recommended rather than presenting a black box.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from restorers.common import resolve_device


# ----------------------------------------------------------------------------
# Lightweight face detector (facexlib RetinaFace) -- no CodeFormer needed here.
# ----------------------------------------------------------------------------

class _FaceDetector:
    def __init__(self, device: Optional[str] = None):
        self._device = resolve_device(device)
        self._net = None

    def load(self):
        if self._net is not None:
            return
        from facexlib.detection import init_detection_model
        try:
            self._net = init_detection_model("retinaface_resnet50", half=False, device=self._device)
        except Exception:
            self._device = torch.device("cpu")
            self._net = init_detection_model("retinaface_resnet50", half=False, device=self._device)

    def detect(self, image: Image.Image, conf: float = 0.9) -> np.ndarray:
        """Return an (N, 5) array of [x1, y1, x2, y2, score] face boxes."""
        self.load()
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
        with torch.no_grad():
            bboxes = self._net.detect_faces(bgr, conf)
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(bboxes, dtype=np.float32)[:, :5]


_detector: Optional[_FaceDetector] = None


def _get_detector() -> _FaceDetector:
    global _detector
    if _detector is None:
        _detector = _FaceDetector()
    return _detector


# ----------------------------------------------------------------------------
# Diagnosis
# ----------------------------------------------------------------------------

@dataclass
class Diagnosis:
    width: int
    height: int
    num_faces: int
    face_area_pct: float       # % of the frame covered by faces
    sharpness: float           # variance of Laplacian (higher = sharper)
    noise_sigma: float         # estimated gaussian noise std (0-255 scale)
    is_low_res: bool
    is_blurry: bool
    is_noisy: bool
    findings: List[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


# Thresholds tuned for typical consumer / scanned old photos.
_BLUR_THRESHOLD = 120.0       # variance-of-Laplacian below this looks soft
_NOISE_THRESHOLD = 6.0        # estimated sigma above this looks grainy
_LOWRES_MAX_EDGE = 900        # longest edge below this benefits from upscaling


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise estimation: convolve with a Laplacian-like kernel."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def diagnose(image: Image.Image) -> Diagnosis:
    """Analyze an image and return a structured, explainable Diagnosis."""
    image = image.convert("RGB")
    w, h = image.size
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise_sigma = _estimate_noise_sigma(gray)

    boxes = _get_detector().detect(image)
    num_faces = int(len(boxes))
    face_area = 0.0
    for x1, y1, x2, y2, _ in boxes:
        face_area += max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    face_area_pct = 100.0 * face_area / float(w * h) if w and h else 0.0

    is_low_res = max(w, h) < _LOWRES_MAX_EDGE
    is_blurry = sharpness < _BLUR_THRESHOLD
    is_noisy = noise_sigma > _NOISE_THRESHOLD

    findings: List[str] = []
    if num_faces:
        findings.append(
            f"{num_faces} face{'s' if num_faces > 1 else ''} detected "
            f"({face_area_pct:.0f}% of the frame)"
        )
    else:
        findings.append("No faces detected")
    findings.append(f"Resolution {w}x{h} ({w * h / 1e6:.1f} MP)"
                    + (" - low, would benefit from upscaling" if is_low_res else ""))
    findings.append(("Soft / blurry" if is_blurry else "Reasonably sharp")
                    + f" (sharpness {sharpness:.0f})")
    findings.append(("Noticeable noise/grain" if is_noisy else "Low noise")
                    + f" (sigma {noise_sigma:.1f})")

    return Diagnosis(
        width=w, height=h, num_faces=num_faces, face_area_pct=face_area_pct,
        sharpness=sharpness, noise_sigma=noise_sigma,
        is_low_res=is_low_res, is_blurry=is_blurry, is_noisy=is_noisy,
        findings=findings,
    )


# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------

@dataclass
class Recommendation:
    do_face: bool
    fidelity: float
    do_upscale: bool
    outscale: int
    summary: str               # one-line plain-English recommended pipeline
    reasons: List[str] = field(default_factory=list)


def recommend(diag: Diagnosis) -> Recommendation:
    """Turn a Diagnosis into a concrete, explainable pipeline recommendation."""
    reasons: List[str] = []

    do_face = diag.num_faces > 0
    if do_face:
        reasons.append(
            f"Faces present -> CodeFormer face restoration "
            f"({diag.num_faces} face{'s' if diag.num_faces > 1 else ''})."
        )

    # Upscale when the image is small, or soft/noisy enough that Real-ESRGAN's
    # detail reconstruction and denoising will visibly help.
    do_upscale = diag.is_low_res or diag.is_blurry or diag.is_noisy
    if do_upscale:
        outscale = 4 if diag.is_low_res and max(diag.width, diag.height) < 500 else 2
        why = []
        if diag.is_low_res:
            why.append("low resolution")
        if diag.is_blurry:
            why.append("softness")
        if diag.is_noisy:
            why.append("noise")
        reasons.append(f"Real-ESRGAN x{outscale} -> addresses {', '.join(why)}.")
    else:
        outscale = 1

    # Slightly higher fidelity (closer to the original) for already-decent
    # photos; lean a bit more on the prior for heavily degraded ones.
    fidelity = 0.5 if (diag.is_blurry or diag.is_noisy) else 0.7

    if not do_face and not do_upscale:
        reasons.append(
            "Image looks clean and high-resolution. For a localized scratch, "
            "tear, or spot, use the Brush Repair tab instead."
        )

    parts = []
    if do_face:
        parts.append("face restoration")
    if do_upscale:
        parts.append(f"x{outscale} enhance")
    summary = " + ".join(parts) if parts else "no automatic restoration needed"

    return Recommendation(
        do_face=do_face, fidelity=fidelity, do_upscale=do_upscale,
        outscale=outscale, summary=summary, reasons=reasons,
    )
