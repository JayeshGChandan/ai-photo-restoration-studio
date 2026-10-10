"""
No-reference image-quality assessment (IQA).

A real restoration has no ground-truth "correct" image, so full-reference
metrics (PSNR / SSIM / LPIPS) don't apply to the live before/after -- there is
nothing to compare against. Instead we use *no-reference* metrics, which predict
perceptual quality from a single image, and report them for the original and the
restored output so the improvement is quantified:

    * MUSIQ     -- transformer trained to predict human MOS  (higher = better)
    * CLIP-IQA  -- CLIP-based semantic quality               (higher = better)
    * NIQE      -- classical naturalness, no training labels  (lower  = better)

Each metric is a lazily-created singleton (weights load once) and is independent:
if one fails to load (e.g. a weight download is unavailable), the others still
report. Full-reference metrics live in the synthetic benchmark, not here.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch

from restorers.common import resolve_device


# No-reference metrics: (pyiqa name, display label, lower_is_better, description)
_METRIC_SPECS = [
    ("musiq",   "MUSIQ",    False, "Predicted human opinion score (0-100)"),
    ("clipiqa", "CLIP-IQA", False, "CLIP-based perceptual quality (0-1)"),
    ("niqe",    "NIQE",     True,  "Naturalness deviation (lower is better)"),
]

# Full-reference metrics (need ground truth) used by the synthetic benchmark.
# PSNR and SSIM are both computed on the luma channel of ITU-R BT.601 YCbCr
# (Y in [16, 235]), the super-resolution convention (BasicSR / MATLAB
# rgb2ycbcr); the "_y" keys make the colour space explicit.
_FR_METRIC_SPECS = [
    ("psnr_y", "PSNR-Y", False, "dB on BT.601 luma vs ground truth (higher is better)"),
    ("ssim_y", "SSIM-Y", False, "Structural similarity on BT.601 luma, 1 = identical"),
    ("lpips",  "LPIPS",  True,  "Perceptual distance (lower is better)"),
]

# Result key -> (pyiqa metric name, options overriding pyiqa's defaults).
# pyiqa's test_y_channel converts with color_space='yiq' unless told otherwise,
# and its PSNR defaults to all RGB channels, so both options are set explicitly.
_PYIQA_METRICS = {
    "psnr_y": ("psnr", {"test_y_channel": True, "color_space": "ycbcr"}),
    "ssim_y": ("ssim", {"test_y_channel": True, "color_space": "ycbcr"}),
}

# Exposed so the UI can label benchmark columns without importing pyiqa.
FR_METRICS = [(k, label, lower) for k, label, lower, _ in _FR_METRIC_SPECS]


@dataclass
class MetricResult:
    key: str
    label: str
    lower_better: bool
    description: str
    before: Optional[float]
    after: Optional[float]

    @property
    def available(self) -> bool:
        return self.before is not None and self.after is not None

    @property
    def delta(self) -> Optional[float]:
        if not self.available:
            return None
        return self.after - self.before

    @property
    def improved(self) -> Optional[bool]:
        """True if ``after`` is better than ``before`` for this metric."""
        d = self.delta
        if d is None or abs(d) < 1e-6:
            return None
        return (d < 0) if self.lower_better else (d > 0)


class IQAScorer:
    """Lazily-loaded collection of no-reference IQA metrics."""

    def __init__(self, device: Optional[str] = None):
        self._device = resolve_device(device)
        self._metrics: Dict[str, object] = {}
        self._failed: set = set()

    def _get(self, name: str):
        """Get (or lazily create) one pyiqa metric; cache failures."""
        if name in self._metrics:
            return self._metrics[name]
        if name in self._failed:
            return None
        try:
            import pyiqa
            pyiqa_name, opts = _PYIQA_METRICS.get(name, (name, {}))
            metric = pyiqa.create_metric(pyiqa_name, device=self._device, **opts)
            metric.eval()
            self._metrics[name] = metric
            return metric
        except Exception as e:
            print(f"IQA metric '{name}' unavailable: {e}")
            self._failed.add(name)
            return None

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _score_one(self, name: str, image: Image.Image) -> Optional[float]:
        metric = self._get(name)
        if metric is None:
            return None
        try:
            with torch.no_grad():
                val = metric(self._to_tensor(image).to(self._device))
            return float(val.item() if hasattr(val, "item") else float(val))
        except Exception as e:
            print(f"IQA metric '{name}' failed on an image: {e}")
            return None

    def score_pair(self, before: Image.Image, after: Image.Image) -> List[MetricResult]:
        """Score ``before`` and ``after`` with every metric.

        Images are scored at their own resolution (no-reference metrics are
        designed for arbitrary sizes), so an upscaled result is judged on its
        own merits.
        """
        results: List[MetricResult] = []
        for key, label, lower, desc in _METRIC_SPECS:
            b = self._score_one(key, before)
            a = self._score_one(key, after)
            results.append(MetricResult(key, label, lower, desc, b, a))
        return results

    def _score_ref_one(self, name: str, candidate: Image.Image,
                       reference: Image.Image) -> Optional[float]:
        """Score one full-reference metric of ``candidate`` against ``reference``."""
        metric = self._get(name)
        if metric is None:
            return None
        # Full-reference metrics need matching dimensions.
        if candidate.size != reference.size:
            candidate = candidate.resize(reference.size, Image.LANCZOS)
        try:
            with torch.no_grad():
                val = metric(
                    self._to_tensor(candidate).to(self._device),
                    self._to_tensor(reference).to(self._device),
                )
            return float(val.item() if hasattr(val, "item") else float(val))
        except Exception as e:
            print(f"FR metric '{name}' failed: {e}")
            return None

    def score_reference(self, candidate: Image.Image,
                        reference: Image.Image) -> Dict[str, Optional[float]]:
        """Full-reference scores for ``candidate`` vs a ground-truth ``reference``.

        Returns a dict keyed by metric name (psnr_y/ssim_y/lpips). Used by the
        synthetic benchmark, where a real ground truth exists.
        """
        ref = reference.convert("RGB")
        cand = candidate.convert("RGB")
        return {key: self._score_ref_one(key, cand, ref) for key, *_ in _FR_METRIC_SPECS}

    @property
    def device(self) -> str:
        return str(self._device)


# Global singleton so metric weights load once and are reused.
_scorer_instance: Optional[IQAScorer] = None


def get_scorer() -> IQAScorer:
    """Get or create the global IQA scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = IQAScorer()
    return _scorer_instance
