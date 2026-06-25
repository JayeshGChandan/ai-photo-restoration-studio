"""
Face Restoration Backend (CodeFormer)
Faces are where the human eye is least forgiving, so they get a dedicated,
face-aware model. The pipeline is the standard, high-quality one:

    detect faces -> align each to a 512x512 frontal template -> restore with
    CodeFormer -> paste back into the original using a parsing mask for a
    seamless blend.

CodeFormer exposes a *fidelity weight*: 0.0 leans on the generative prior for
maximum quality (can subtly alter identity), 1.0 stays faithful to the input
(but preserves more of its degradation). ~0.7 is a good default for old photos.

Networks load through ``spandrel`` (CodeFormer) and ``facexlib`` (detection,
alignment, parsing) -- no basicsr -- so everything runs on modern torch.
Optionally, the non-face background is upscaled with Real-ESRGAN and the
restored faces are pasted in at the higher resolution.
"""

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch

from .common import resolve_device, WEIGHTS_DIR

DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "codeformer.pth")


class FaceRestorer:
    """Lazy-loaded CodeFormer face restorer with detect/align/paste-back."""

    def __init__(
        self,
        device: Optional[str] = None,
        weights_path: str = DEFAULT_WEIGHTS,
        face_size: int = 512,
        detection_resize: int = 640,
    ):
        """
        Args:
            device: Force a torch device; None auto-selects (see resolve_device).
            weights_path: Path to the CodeFormer ``.pth`` weights.
            face_size: Aligned face crop size the model operates on (512).
            detection_resize: Long edge the detector runs at (speed/recall
                trade-off); landmarks are mapped back to full resolution.
        """
        self.weights_path = weights_path
        self.face_size = face_size
        self.detection_resize = detection_resize
        self._device = resolve_device(device)
        self._model = None        # spandrel descriptor
        self._raw = None          # underlying nn.Module (lets us pass fidelity)
        self._helper = None       # facexlib FaceRestoreHelper

    def load(self) -> None:
        """Lazily load CodeFormer + the facexlib face helper."""
        if self._model is not None:
            return
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"CodeFormer weights not found at {self.weights_path}. "
                "Run the model download step (see README) first."
            )

        import spandrel_extra_arches
        spandrel_extra_arches.install(ignore_duplicates=True)
        from spandrel import ModelLoader
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        def _build():
            model = ModelLoader(device=self._device).load_from_file(self.weights_path).eval()
            helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=self.face_size,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self._device,
            )
            return model, helper

        try:
            print(f"Loading CodeFormer + face helper on {self._device}...")
            self._model, self._helper = _build()
        except Exception as e:
            if self._device.type != "cpu":
                print(f"Face restorer failed on {self._device} ({e}); falling back to CPU.")
                self._device = torch.device("cpu")
                self._model, self._helper = _build()
            else:
                raise
        self._raw = self._model.model
        print("Face restorer loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)

    def _restore_crop(self, cropped_bgr: np.ndarray, fidelity: float) -> np.ndarray:
        """Restore a single aligned BGR face crop; returns a BGR uint8 array."""
        rgb = cropped_bgr[:, :, ::-1].astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._raw(t, weight=float(fidelity))[0].clamp(0, 1)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        out = out.round().astype(np.uint8)
        return out[:, :, ::-1]  # RGB -> BGR for the helper

    def restore(
        self,
        image: Image.Image,
        fidelity: float = 0.7,
        upscale: int = 1,
        background_image: Optional[Image.Image] = None,
        only_center_face: bool = False,
    ) -> Tuple[Image.Image, dict]:
        """Restore every detected face and paste the results back.

        Args:
            image: Source PIL image.
            fidelity: CodeFormer fidelity weight in [0, 1].
            upscale: Final whole-image scale factor (1, 2, or 4). When > 1 you
                should pass ``background_image`` (a Real-ESRGAN upscaled copy) so
                the non-face background matches; otherwise it is naively resized.
            background_image: Pre-upscaled full image to paste faces onto. Must
                be ``upscale`` x the input size.
            only_center_face: Restore only the largest/central face if True.

        Returns:
            (result_image, info) where info has ``num_faces`` and ``upscale``.
            If no face is detected, returns the (optionally upscaled) input
            unchanged with ``num_faces == 0``.
        """
        if image is None:
            raise ValueError("No image provided for face restoration.")
        self.load()

        image = image.convert("RGB")
        bgr = np.asarray(image)[:, :, ::-1]  # RGB -> BGR

        helper = self._helper
        helper.clean_all()
        helper.upscale_factor = int(upscale)
        helper.read_image(bgr)
        num = helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=self.detection_resize
        )

        if num == 0:
            # Nothing face-specific to do; return the requested-scale input.
            if upscale > 1:
                if background_image is not None:
                    return background_image.convert("RGB"), {"num_faces": 0, "upscale": upscale}
                w, h = image.size
                return image.resize((w * upscale, h * upscale), Image.LANCZOS), {
                    "num_faces": 0, "upscale": upscale,
                }
            return image, {"num_faces": 0, "upscale": 1}

        helper.align_warp_face()
        for cropped in helper.cropped_faces:
            helper.add_restored_face(self._restore_crop(cropped, fidelity))

        helper.get_inverse_affine(None)

        upsample_img = None
        if background_image is not None:
            bg = background_image.convert("RGB")
            target = (image.size[0] * int(upscale), image.size[1] * int(upscale))
            if bg.size != target:
                bg = bg.resize(target, Image.LANCZOS)
            upsample_img = np.asarray(bg)[:, :, ::-1]  # RGB -> BGR

        result_bgr = helper.paste_faces_to_input_image(upsample_img=upsample_img)
        result = Image.fromarray(result_bgr[:, :, ::-1].astype(np.uint8), "RGB")
        return result, {"num_faces": int(num), "upscale": int(upscale)}

    def detect_face_count(self, image: Image.Image) -> int:
        """Lightweight face count used by the diagnosis step."""
        self.load()
        image = image.convert("RGB")
        bgr = np.asarray(image)[:, :, ::-1]
        helper = self._helper
        helper.clean_all()
        helper.upscale_factor = 1
        helper.read_image(bgr)
        return int(helper.get_face_landmarks_5(only_center_face=False, resize=self.detection_resize))


# Global singleton so weights load once and are reused across requests.
_face_restorer_instance: Optional[FaceRestorer] = None


def get_face_restorer() -> FaceRestorer:
    """Get or create the global CodeFormer face restorer instance."""
    global _face_restorer_instance
    if _face_restorer_instance is None:
        _face_restorer_instance = FaceRestorer()
    return _face_restorer_instance
