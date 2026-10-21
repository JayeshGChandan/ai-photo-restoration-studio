# AI Photo Restoration Studio

A photo-restoration workbench built around a simple principle: **diagnose the
damage, route it to the right model, restore it, then measure the result.**
Different kinds of damage need different tools, so instead of one model for
everything, the app picks the right specialist and checks the outcome with
image-quality metrics.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.x-green.svg)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-supported-black.svg)

---

## Why this design

Old "AI photo repair" demos tend to push every image through a single Stable
Diffusion inpainting pass at 512px. That destroys resolution and *hallucinates*
new content where it should be continuing what's already there. This project
takes the opposite approach:

```
            ┌─────────────┐
  image ──▶ │  DIAGNOSE   │  faces?  blur / noise / low-res?  localized defect?
            └──────┬──────┘
                   ▼
            ┌─────────────┐   faces        → CodeFormer        (face restoration)
            │    ROUTE    │   global decay → Real-ESRGAN       (upscale / denoise)
            └──────┬──────┘   localized    → LaMa (brush)      (structure-aware fill)
                   ▼
            ┌─────────────┐
            │   RESTORE   │  run the chosen specialist(s)
            └──────┬──────┘
                   ▼
            ┌─────────────┐  brush repair only: paste the fill back at native res;
            │  COMPOSITE  │  pixels beyond the feathered edge stay exact
            └──────┬──────┘
                   ▼
            ┌─────────────┐  no-reference IQA live (MUSIQ / CLIP-IQA / NIQE)
            │    SCORE    │  full-reference in the benchmark (PSNR-Y/SSIM-Y/LPIPS)
            └─────────────┘
```

---

## The three tabs

### 1. Auto Restore
Click **Diagnose** and the app analyses the photo (face count and coverage,
sharpness as the variance of the Laplacian, noise as an Immerkaer estimate, and
resolution), then recommends a pipeline with the reasons shown. **Restore** runs
it:

- Faces → CodeFormer (detect → align to a 512 template → restore → paste back
  with a parsing mask). A *fidelity* slider trades generative quality against
  faithfulness to the original.
- Low-res / soft / noisy → Real-ESRGAN x2/x4 with tiled, seam-free inference.
- When both apply, the background is upscaled first and the restored faces are
  pasted in at the final resolution.

A no-reference quality scorecard (MUSIQ, CLIP-IQA, NIQE) reports the perceptual
quality of the input vs the result.

### 2. Brush Repair
For a localized scratch, tear, spot, or an object to remove: brush over it and
LaMa fills only that region by *continuing surrounding structure* (Fast Fourier
Convolutions) rather than inventing new content. LaMa works on images up to
1536 px on the long edge, so larger photos are inpainted on a reduced copy. The
fill is composited back with a feathered mask at full resolution, so the brushed
area is exactly LaMa's output and every pixel beyond the feathered edge (about
19 px out at the default feather of 3, wider at higher settings) stays
byte-for-byte identical.

### 3. Benchmark
The honest part. Restoration has no ground truth on a real photo, so this tab
makes one: take a clean image, apply known degradations (downscale, blur, noise,
JPEG), restore it, and score against the original with full-reference metrics
(PSNR-Y / SSIM-Y on BT.601 luma, LPIPS), next to a naive bicubic baseline.

> This surfaces the perception–distortion tradeoff: GAN-based restorers improve
> LPIPS (how good it looks to a person), often while giving up a little
> PSNR-Y/SSIM-Y. LPIPS tracks human judgement more closely than PSNR or SSIM.

---

## Evaluation

Run the tests:

```bash
./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest
```

Fetch the data:

```bash
./venv/bin/python -I scripts/fetch_data.py
```

Run the compositing experiment:

```bash
./venv/bin/python scripts/run_compositing_experiment.py --preflight-only
./venv/bin/python scripts/run_compositing_experiment.py --out results/compositing_repro
```

Summarise it:

```bash
./venv/bin/python scripts/summarise_compositing.py --dir results/compositing_repro
```

Make the figures and tables:

```bash
./venv/bin/python scripts/make_figures.py
./venv/bin/python scripts/make_tables.py
```

Committed results of the full run: `results/compositing/`

Figures and charts: `docs/figures/`

---

## Architecture

```
app.py               Gradio UI — three tabs, handlers, scorecards
diagnose.py          image analysis + explainable routing (Diagnosis, recommend)
pipeline.py          route → restore orchestration (face + upscale)
compositing.py       native-resolution feathered blend (only mask + feathered edge change)
masking.py           brush-stroke → mask; classical auto-suggest (hint only)
degradations.py      synthetic damage for the benchmark
benchmark.py         clean → degrade → restore → full-reference scoring
metrics.py           IQA: no-reference (MUSIQ/CLIP-IQA/NIQE) + full-reference (PSNR-Y/SSIM-Y/LPIPS)
synthetic_damage.py  seeded damage masks + fill for the compositing experiment
region_metrics.py    region-restricted scoring (pixel differences, PSNR-Y, LPIPS)
restorers/
  inpaint.py         LaMa backend (simple-lama-inpainting)
  face.py            CodeFormer via spandrel + facexlib (detect/align/paste-back)
  upscale.py         Real-ESRGAN via spandrel, tiled
  common.py          shared device selection
scripts/             data fetch, compositing experiment + summary, figure/table builders,
                     Real-ESRGAN fix measurement, frozen legacy composite
tests/               pytest suite: compositing identity, legacy parity, region metrics,
                     damage synthesis, experiment empty-U handling
data/                CLIC 2020 manifest + licence record (images fetched, not committed)
results/             committed outputs (compositing experiment, Real-ESRGAN fix comparison)
docs/                GitHub Pages source (index.md, figures/)
```

Model loading without the dependency hell: the usual CodeFormer / GFPGAN /
Real-ESRGAN packages are built on `basicsr`, which fails to import on modern
torchvision (`functional_tensor` was removed). This project loads those networks
through [spandrel](https://github.com/chaiNNer-org/spandrel) (a maintained,
basicsr-free loader) plus facexlib for face geometry, so everything runs on
current PyTorch with no monkey-patching.

---

## Install

Tested on Python 3.11, Apple Silicon (macOS). CUDA is used automatically if
present; otherwise it runs on CPU. CPU is the default on Apple Silicon, with MPS
as an opt-in: `LAMA_DEVICE=mps` moves LaMa, and `RESTORE_DEVICE=mps` moves
Real-ESRGAN, CodeFormer, the Diagnose face detector and the quality metrics.
LaMa uses FFT operations that are unreliable on MPS and is untested there.
Real-ESRGAN ran correctly on MPS in testing, within one level of the CPU output
and about 7-10x faster. CodeFormer, the face detector and the metrics are
untested on MPS. A model that fails to load on MPS falls back to CPU; a quality
metric that fails to load is shown as unavailable instead.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# No-reference / full-reference IQA (pyiqa) — installed lean to keep our pins:
./venv/bin/pip install --no-deps pyiqa==0.1.15.post2
./venv/bin/pip install timm==1.0.27 openai-clip==1.0.1 ftfy==6.3.1 regex==2026.5.9 requests==2.34.2
```

> `pyiqa` otherwise resolves numpy≥2 / Pillow≥10, which breaks the LaMa stack
> (`simple-lama-inpainting` pins Pillow<10). Installing it `--no-deps` plus the
> few libs its metrics actually import keeps the environment consistent.

### Model weights

Most weights auto-download on first use (LaMa, the facexlib detector/parser, and
the MUSIQ / CLIP-IQA / LPIPS metric weights). Two go in `./weights/`:

```bash
mkdir -p weights
curl -L -o weights/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
curl -L -o weights/codeformer.pth \
  https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
```

---

## Run

```bash
./venv/bin/python app.py     # → http://127.0.0.1:7860
```

The first restoration is slow (one-time model downloads + load); afterwards each
runs in a few seconds to a minute or more on CPU, depending on image size. Models
are cached as singletons and reused.

---

## Tech & credits

LaMa · CodeFormer · Real-ESRGAN · facexlib · spandrel · pyiqa · PyTorch · Gradio.
Restoration models are used under their respective licenses (CodeFormer and
GFPGAN are research / non-commercial).

Brush Repair's paste-back approach is an established technique. IOPaint
(https://github.com/Sanster/IOPaint, formerly Lama Cleaner) offers similar Resize
and Crop strategies.

Evaluation images: CLIC 2020 validation set, fetched by scripts/fetch_data.py and
not redistributed here. Licences are recorded in data/LICENSES.md.
