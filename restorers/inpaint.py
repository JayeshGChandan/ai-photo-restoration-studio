"""
LaMa Inpainting Backend
Resolution-robust large-mask inpainting via the LaMa model (Fast Fourier
Convolutions). Unlike diffusion inpainting, LaMa *continues* existing structure
and texture into the masked region instead of hallucinating new content. That is
what makes repairs of scratches, tears, spots, and small object removals look
natural and faithful rather than artificially generated.

The model is loaded lazily and runs at native resolution (with an optional cap
for very large images). The caller is expected to composite the result back into
the original with a feathered mask (see compositing.py), so only the repaired
region ever changes.
"""

import os
import gc
from typing import Optional

import numpy as np
from PIL import Image
import torch


class LamaInpainter:
    """Lazy-loaded LaMa inpainting model with sane device selection."""

    def __init__(self, device: Optional[str] = None, max_dimension: int = 1536):
        """
        Args:
            device: Force a torch device ("cpu", "mps", "cuda"). If None, an
                appropriate device is chosen automatically (see _resolve_device).
            max_dimension: Longest-edge cap fed to the model. Images larger than
                this are downscaled for inference and the result is upscaled back;
                native fidelity outside the repaired region is preserved by the
                compositing step.
        """
        self.max_dimension = max_dimension
        self._model = None
        self._device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """Pick a compute device.

        LaMa relies on FFT ops that are not reliably supported on Apple's MPS
        backend, so we default to CPU on Apple Silicon (CUDA when available).
        MPS can still be forced via the device argument or LAMA_DEVICE env var.
        """
        if device:
            return torch.device(device)
        env = os.environ.get("LAMA_DEVICE")
        if env:
            return torch.device(env)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> None:
        """Lazily load the LaMa model (downloads ~200MB on first run)."""
        if self._model is not None:
            return

        from simple_lama_inpainting import SimpleLama

        try:
            print(f"Loading LaMa inpainting model on {self._device}...")
            self._model = SimpleLama(device=self._device)
            print("LaMa model loaded.")
        except Exception as e:
            # Fall back to CPU if the requested device cannot run the graph.
            if self._device.type != "cpu":
                print(f"LaMa failed on {self._device} ({e}); falling back to CPU.")
                self._device = torch.device("cpu")
                self._model = SimpleLama(device=self._device)
            else:
                raise

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)

    def _maybe_downscale(self, image: Image.Image, mask: Image.Image):
        """Downscale to the configured cap for inference; returns work copies."""
        w, h = image.size
        longest = max(w, h)
        if self.max_dimension and longest > self.max_dimension:
            scale = self.max_dimension / float(longest)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            return (
                image.resize((nw, nh), Image.LANCZOS),
                mask.resize((nw, nh), Image.NEAREST),
            )
        return image, mask

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint the masked (white) region of ``image``.

        Args:
            image: Source PIL image.
            mask: Binary PIL mask, white (255) = repair, black (0) = keep.

        Returns:
            A restored PIL image at the original resolution. (Composite this
            back onto the original with a feathered mask for a seamless result.)
        """
        if image is None:
            raise ValueError("No image provided for inpainting.")
        if mask is None:
            raise ValueError("No mask provided for inpainting.")

        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        if int(np.asarray(mask).max()) == 0:
            raise ValueError(
                "The mask is empty. Brush over the damaged areas before repairing."
            )

        orig_size = image.size  # (w, h)
        work_image, work_mask = self._maybe_downscale(image, mask)

        self.load()
        try:
            result = self._model(work_image, work_mask)
        except Exception as e:
            raise RuntimeError(f"LaMa inpainting failed: {e}")

        # SimpleLama symmetrically pads to a multiple of 8 (bottom/right) and does
        # not crop back -- so the output can be a few pixels larger. Crop to the
        # pre-pad working size, then restore native resolution if we downscaled.
        ww, wh = work_image.size
        result = result.crop((0, 0, ww, wh))
        if result.size != orig_size:
            result = result.resize(orig_size, Image.LANCZOS)
        return result

    def unload(self) -> None:
        """Free the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global singleton so the model is loaded once and reused across requests.
_inpainter_instance: Optional[LamaInpainter] = None


def get_inpainter() -> LamaInpainter:
    """Get or create the global LaMa inpainter instance."""
    global _inpainter_instance
    if _inpainter_instance is None:
        _inpainter_instance = LamaInpainter()
    return _inpainter_instance
