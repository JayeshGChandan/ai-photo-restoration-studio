"""
Real-ESRGAN Super-Resolution Backend
Global-degradation restoration: upscales and cleans up images that are low
resolution, soft/blurry, noisy, or JPEG-compressed. Real-ESRGAN reconstructs
plausible high-frequency detail (texture, edges) across the whole image, which
is the right tool when the damage is *global* rather than a localized defect.

The network is loaded through ``spandrel`` (a maintained, basicsr-free model
loader), so it runs cleanly on modern torch/torchvision. Inference is tiled with
overlap so large images stay within memory on CPU/MPS, and the tile context
prevents visible seams.
"""

import gc
import os
from typing import Optional

import numpy as np
from PIL import Image
import torch

from .common import resolve_device, WEIGHTS_DIR

DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")


class RealESRGANUpscaler:
    """Lazy-loaded Real-ESRGAN upscaler with tiled, seam-free inference."""

    def __init__(
        self,
        device: Optional[str] = None,
        weights_path: str = DEFAULT_WEIGHTS,
        tile: int = 256,
        tile_pad: int = 16,
        max_input_dim: int = 2048,
    ):
        """
        Args:
            device: Force a torch device; None auto-selects (see resolve_device).
            weights_path: Path to the Real-ESRGAN ``.pth`` weights.
            tile: Input tile size in pixels. Tiles larger than this are split.
                Set 0 to disable tiling.
            tile_pad: Context padding (input px) added around each tile before
                inference and cropped off after, to avoid seams.
            max_input_dim: Longest input edge fed to the model. Larger inputs are
                downscaled first so a 4x upscale stays tractable on CPU.
        """
        self.weights_path = weights_path
        self.tile = tile
        self.tile_pad = tile_pad
        self.max_input_dim = max_input_dim
        self._device = resolve_device(device)
        self._model = None
        self._native_scale = 4

    def load(self) -> None:
        """Lazily load the Real-ESRGAN weights via spandrel."""
        if self._model is not None:
            return
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found at {self.weights_path}. "
                "Run the model download step (see README) first."
            )
        from spandrel import ModelLoader

        try:
            print(f"Loading Real-ESRGAN on {self._device}...")
            self._model = ModelLoader(device=self._device).load_from_file(self.weights_path).eval()
        except Exception as e:
            if self._device.type != "cpu":
                print(f"Real-ESRGAN failed on {self._device} ({e}); falling back to CPU.")
                self._device = torch.device("cpu")
                self._model = ModelLoader(device=self._device).load_from_file(self.weights_path).eval()
            else:
                raise
        self._native_scale = int(self._model.scale)
        print(f"Real-ESRGAN loaded (native x{self._native_scale}).")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def native_scale(self) -> int:
        return self._native_scale

    def _infer(self, tile: Image.Image) -> Image.Image:
        """Run the model on a single PIL tile and return the upscaled tile."""
        self.load()
        arr = np.asarray(tile.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(t)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
        return Image.fromarray(out.round().astype(np.uint8), "RGB")

    def _tiled_upscale(self, image: Image.Image) -> Image.Image:
        """Upscale by the native scale, tiling with padded context for seamlessness."""
        scale = self._native_scale
        w, h = image.size
        if self.tile <= 0 or (w <= self.tile and h <= self.tile):
            return self._infer(image)

        arr = np.asarray(image.convert("RGB"))
        out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        pad = self.tile_pad

        for y in range(0, h, self.tile):
            for x in range(0, w, self.tile):
                x0, y0 = x, y
                x1, y1 = min(x + self.tile, w), min(y + self.tile, h)
                # Read a padded region for context, clamped to the image.
                px0, py0 = max(x0 - pad, 0), max(y0 - pad, 0)
                px1, py1 = min(x1 + pad, w), min(y1 + pad, h)

                sub = Image.fromarray(arr[py0:py1, px0:px1])
                up = np.asarray(self._infer(sub), dtype=np.uint8)

                # Crop the padded border back off in output (scaled) coordinates.
                cx0, cy0 = (x0 - px0) * scale, (y0 - py0) * scale
                cx1, cy1 = cx0 + (x1 - x0) * scale, cy0 + (y1 - y0) * scale
                out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = up[cy0:cy1, cx0:cx1]

        return Image.fromarray(out, "RGB")

    def enhance(self, image: Image.Image, outscale: float = 4.0) -> Image.Image:
        """Upscale/clean ``image`` and return it at ``outscale`` x the input size.

        Args:
            image: Source PIL image.
            outscale: Final size multiplier relative to the (capped) input,
                1.0-4.0. The model always runs at its native 4x; the result is
                resampled to the requested multiplier.

        Returns:
            The enhanced PIL image.
        """
        if image is None:
            raise ValueError("No image provided for upscaling.")
        image = image.convert("RGB")
        w, h = image.size

        # Cap the working size so a 4x upscale stays tractable on CPU/MPS.
        if self.max_input_dim and max(w, h) > self.max_input_dim:
            s = self.max_input_dim / float(max(w, h))
            image = image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
            w, h = image.size

        self.load()
        scale = self._native_scale
        target = (max(1, round(w * outscale)), max(1, round(h * outscale)))

        # The model upscales by a fixed native factor (e.g. 4x). When the caller
        # wants less than that (e.g. 2x), pre-shrink the input so native
        # upscaling lands on the target -- running the full 4x and discarding
        # half the pixels would cost ~4x the compute and memory for nothing.
        if outscale < scale:
            pre = outscale / float(scale)
            work = image.resize((max(1, round(w * pre)), max(1, round(h * pre))), Image.LANCZOS)
        else:
            work = image

        up = self._tiled_upscale(work)
        if up.size != target:
            up = up.resize(target, Image.LANCZOS)
        return up

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global singleton so weights load once and are reused across requests.
_upscaler_instance: Optional[RealESRGANUpscaler] = None


def get_upscaler() -> RealESRGANUpscaler:
    """Get or create the global Real-ESRGAN upscaler instance."""
    global _upscaler_instance
    if _upscaler_instance is None:
        _upscaler_instance = RealESRGANUpscaler()
    return _upscaler_instance
