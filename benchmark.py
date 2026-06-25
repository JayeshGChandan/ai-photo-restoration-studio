"""
Synthetic-degradation benchmark.

Pipeline: take a clean image -> apply known degradations -> restore -> measure
how close the restoration gets to the original with full-reference metrics. We
also compute the same metrics for a naive bicubic upscale of the degraded input,
so the table shows what the learned restoration adds over plain interpolation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

from degradations import degrade
from pipeline import restore as run_pipeline
from metrics import get_scorer


@dataclass
class BenchmarkResult:
    clean: Image.Image
    degraded: Image.Image          # actual degraded input (may be small)
    baseline: Image.Image          # naive bicubic upscale of degraded -> clean size
    restored: Image.Image          # restoration output at clean size
    applied: List[str]
    steps: str
    baseline_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    restored_scores: Dict[str, Optional[float]] = field(default_factory=dict)


def run_benchmark(
    clean: Image.Image,
    downscale_factor: int = 2,
    blur_sigma: float = 1.0,
    noise_sigma: float = 8.0,
    jpeg_quality: int = 60,
    do_face: bool = True,
    fidelity: float = 0.7,
    score: bool = True,
) -> BenchmarkResult:
    """Degrade ``clean``, restore it, and score both restoration and a baseline."""
    if clean is None:
        raise ValueError("No clean image provided.")
    clean = clean.convert("RGB")
    size = clean.size

    deg = degrade(clean, downscale_factor, blur_sigma, noise_sigma, jpeg_quality)
    degraded = deg.image
    baseline = degraded.resize(size, Image.BICUBIC)

    # Recover resolution: upscale by the factor we lost (or 2x as a denoise/deblur
    # pass when there was no resolution loss), then snap back to the clean size.
    outscale = downscale_factor if downscale_factor > 1 else 2
    restored_raw, steps = run_pipeline(
        degraded, do_face=do_face, fidelity=fidelity, do_upscale=True, outscale=outscale
    )
    restored = restored_raw if restored_raw.size == size else restored_raw.resize(size, Image.LANCZOS)

    baseline_scores: Dict[str, Optional[float]] = {}
    restored_scores: Dict[str, Optional[float]] = {}
    if score:
        scorer = get_scorer()
        baseline_scores = scorer.score_reference(baseline, clean)
        restored_scores = scorer.score_reference(restored, clean)

    return BenchmarkResult(
        clean=clean, degraded=degraded, baseline=baseline, restored=restored,
        applied=deg.applied, steps=steps,
        baseline_scores=baseline_scores, restored_scores=restored_scores,
    )
