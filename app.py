"""
AI Photo Restoration Studio
A diagnose -> route -> restore workflow with two complementary tools:

  * Auto Restore -- analyzes the photo, recommends a pipeline, and runs
    CodeFormer (faces) and/or Real-ESRGAN (global upscale/denoise).
  * Brush Repair -- paint over a localized scratch/tear/spot and LaMa fills it
    by continuing surrounding structure, composited back at native resolution.
"""

import gradio as gr

from masking import mask_from_editor, suggest_mask
from compositing import composite
from restorers.inpaint import get_inpainter
from diagnose import diagnose, recommend
from pipeline import restore as run_pipeline
from metrics import get_scorer, FR_METRICS
from benchmark import run_benchmark


# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    background: #141B2D !important;
    color: #F1F3F5 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.header-container { padding: 1.5rem 0 0.5rem; text-align: center; }
.header-title {
    font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em;
    background: linear-gradient(135deg, #8B5CF6, #22D3EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-subtitle { color: #9CA3AF; font-size: 0.95rem; margin-top: 0.25rem; }
.section-label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9CA3AF; margin: 0.5rem 0;
}
.primary-btn {
    background: linear-gradient(135deg, #8B5CF6, #22D3EE) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.25) !important;
}
.diag-card {
    background: #1B2238; border: 1px solid #2A3450; border-radius: 12px;
    padding: 1rem 1.1rem; margin: 0.25rem 0;
}
.diag-reco {
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(34,211,238,0.12));
    border: 1px solid rgba(139,92,246,0.45); border-radius: 10px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.75rem;
}
.diag-reco b { color: #C4B5FD; }
.diag-chip {
    display: inline-block; background: #232C46; border: 1px solid #313C5C;
    color: #CBD5E1; border-radius: 999px; padding: 0.25rem 0.7rem;
    margin: 0.2rem 0.25rem 0.2rem 0; font-size: 0.82rem;
}
.diag-chip.warn { border-color: #B45309; color: #FCD34D; background: #2A2410; }
.score-card {
    background: #1B2238; border: 1px solid #2A3450; border-radius: 12px;
    padding: 0.9rem 1.0rem; margin: 0.25rem 0;
}
.score-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.4rem 0; border-bottom: 1px solid #232C46;
}
.score-row:last-child { border-bottom: none; }
.score-name { color: #E2E8F0; font-weight: 600; font-size: 0.9rem; }
.score-desc { color: #7C879E; font-size: 0.72rem; }
.score-vals { font-variant-numeric: tabular-nums; color: #CBD5E1; font-size: 0.9rem; }
.score-vals .arrow { color: #64748B; margin: 0 0.35rem; }
.score-delta { font-weight: 700; font-size: 0.85rem; margin-left: 0.5rem; }
.score-delta.up { color: #34D399; }
.score-delta.down { color: #F87171; }
.score-delta.flat { color: #94A3B8; }
.score-note { color: #7C879E; font-size: 0.72rem; margin-top: 0.5rem; }
.fr-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.fr-table th, .fr-table td { padding: 0.5rem 0.6rem; text-align: right; border-bottom: 1px solid #232C46; }
.fr-table th { color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.fr-table td.metric, .fr-table th.metric { text-align: left; color: #E2E8F0; font-weight: 600; }
.fr-table td.win { color: #34D399; font-weight: 700; }
.fr-table td.lose { color: #94A3B8; }
.fr-desc { color: #7C879E; font-size: 0.72rem; }
footer { display: none !important; }
"""


# ----------------------------------------------------------------------------
# Auto Restore tab
# ----------------------------------------------------------------------------

def _render_card(diag, rec) -> str:
    chips = ""
    # Findings as chips; flag the degraded ones.
    warn_terms = ("low", "Soft", "blurry", "noise", "grain")
    for f in diag.findings:
        cls = "diag-chip warn" if any(t in f for t in warn_terms) else "diag-chip"
        chips += f'<span class="{cls}">{f}</span>'
    reasons = "".join(f"<div>&bull; {r}</div>" for r in rec.reasons)
    return f"""
    <div class="diag-card">
      <div class="diag-reco">
        <b>Recommended pipeline:</b> {rec.summary}
        <div style="margin-top:0.4rem; color:#CBD5E1; font-size:0.86rem;">{reasons}</div>
      </div>
      <div>{chips}</div>
    </div>
    """


_SCORE_PLACEHOLDER = "<div class='score-card'>Quality scores appear here after restoring.</div>"


def _fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def _render_scorecard(results) -> str:
    """Render before/after no-reference IQA scores with improvement arrows."""
    if not any(r.available for r in results):
        return ("<div class='score-card'>Quality scoring unavailable "
                "(metric weights could not be loaded).</div>")
    rows = ""
    for r in results:
        if r.available:
            if r.improved is True:
                cls, sym = "up", "&#9650;"      # ▲
            elif r.improved is False:
                cls, sym = "down", "&#9660;"     # ▼
            else:
                cls, sym = "flat", "&#8211;"     # –
            delta = f'<span class="score-delta {cls}">{sym} {abs(r.delta):.2f}</span>'
            vals = (f'<span class="score-vals">{_fmt(r.before)}'
                    f'<span class="arrow">&rarr;</span>{_fmt(r.after)}{delta}</span>')
        else:
            vals = '<span class="score-vals">n/a</span>'
        rows += (
            f'<div class="score-row"><div><div class="score-name">{r.label}</div>'
            f'<div class="score-desc">{r.description}</div></div>{vals}</div>'
        )
    note = ("Higher MUSIQ / CLIP-IQA is better; lower NIQE is better. "
            "No-reference metrics judge each image on its own.")
    return f'<div class="score-card">{rows}<div class="score-note">{note}</div></div>'


def compute_scorecard(before, after, enabled) -> str:
    """Score a before/after pair, returning scorecard HTML (resilient)."""
    if not enabled:
        return _SCORE_PLACEHOLDER
    if before is None or after is None:
        return _SCORE_PLACEHOLDER
    try:
        results = get_scorer().score_pair(before.convert("RGB"), after.convert("RGB"))
        return _render_scorecard(results)
    except Exception as e:
        return f"<div class='score-card'>Quality scoring failed: {e}</div>"


_BENCH_PLACEHOLDER = "<div class='score-card'>Run the benchmark to see full-reference scores.</div>"


def _render_fr_table(applied, baseline_scores, restored_scores) -> str:
    """Render a full-reference metrics table (naive baseline vs restoration)."""
    rows = ""
    for key, label, lower in FR_METRICS:
        b = baseline_scores.get(key)
        a = restored_scores.get(key)
        if b is None or a is None:
            rows += (f'<tr><td class="metric">{label}</td>'
                     f'<td colspan="2">n/a</td></tr>')
            continue
        restored_wins = (a < b) if lower else (a > b)
        b_cls = "" if restored_wins else "win"
        a_cls = "win" if restored_wins else "lose"
        rows += (
            f'<tr><td class="metric">{label}</td>'
            f'<td class="{b_cls}">{b:.3f}</td>'
            f'<td class="{a_cls}">{a:.3f}</td></tr>'
        )
    note = (
        "Measured against the original (ground truth). Higher PSNR/SSIM and "
        "lower LPIPS are better. GAN restorers optimise <b>perceptual</b> quality "
        "(LPIPS) and often trade a little PSNR/SSIM for it &mdash; LPIPS is the "
        "metric that tracks how good the result looks to a person."
    )
    return f"""
    <div class="score-card">
      <div class="score-desc" style="margin-bottom:0.5rem;">Degraded with: {', '.join(applied)}</div>
      <table class="fr-table">
        <tr><th class="metric">Metric</th><th>Naive bicubic</th><th>Restored</th></tr>
        {rows}
      </table>
      <div class="score-note">{note}</div>
    </div>
    """


def run_benchmark_handler(clean, downscale_label, blur, noise, jpeg, do_face, fidelity):
    """Degrade -> restore -> score against ground truth."""
    if clean is None:
        return None, None, None, _BENCH_PLACEHOLDER, "Upload a clean image first."
    factor = int(str(downscale_label).replace("x", "")) if downscale_label else 1
    try:
        r = run_benchmark(
            clean, downscale_factor=factor, blur_sigma=float(blur),
            noise_sigma=float(noise), jpeg_quality=int(jpeg),
            do_face=bool(do_face), fidelity=float(fidelity), score=True,
        )
    except Exception as e:
        return None, None, None, _BENCH_PLACEHOLDER, f"Benchmark failed: {e}"
    table = _render_fr_table(r.applied, r.baseline_scores, r.restored_scores)
    status = f"Done: {r.steps}. Degraded with {', '.join(r.applied)}."
    return r.clean, r.baseline, r.restored, table, status


def clear_benchmark():
    return None, None, None, None, _BENCH_PLACEHOLDER, "Cleared. Upload a clean image to begin."


def analyze(image):
    """Diagnose the image and pre-fill the recommended controls."""
    if image is None:
        return "<div class='diag-card'>Upload an image to analyze.</div>", \
               gr.update(), gr.update(), gr.update(), "Upload an image first."
    diag = diagnose(image)
    rec = recommend(diag)
    card = _render_card(diag, rec)
    outscale_label = f"{rec.outscale}x"
    status = f"Diagnosed. Recommended: {rec.summary}."
    return (
        card,
        gr.update(value=rec.do_face),
        gr.update(value=round(rec.fidelity, 2)),
        gr.update(value=outscale_label),
        status,
    )


def restore_auto(image, do_face, fidelity, outscale_label, show_scores):
    """Run the routed restoration pipeline and score the result."""
    if image is None:
        return None, None, _SCORE_PLACEHOLDER, "Upload an image first."
    outscale = int(str(outscale_label).replace("x", "")) if outscale_label else 1
    do_upscale = outscale > 1
    try:
        result, steps = run_pipeline(
            image, do_face=bool(do_face), fidelity=float(fidelity),
            do_upscale=do_upscale, outscale=outscale,
        )
    except Exception as e:
        return None, None, _SCORE_PLACEHOLDER, f"Restoration failed: {e}"
    before = image.convert("RGB")
    scorecard = compute_scorecard(before, result, show_scores)
    return result, (before, result), scorecard, f"Done: {steps}."


def clear_auto():
    return None, None, None, "<div class='diag-card'>Upload an image to analyze.</div>", \
           _SCORE_PLACEHOLDER, "Cleared. Upload an image to begin."


# ----------------------------------------------------------------------------
# Brush Repair tab (Phase 1)
# ----------------------------------------------------------------------------

def suggest(editor_value, sensitivity, detect_bright, detect_dark, min_area):
    if not editor_value or editor_value.get("background") is None:
        return None, "Upload an image first."
    _, overlay, status = suggest_mask(
        editor_value["background"], sensitivity, detect_bright, detect_dark, min_area
    )
    return overlay, status


def repair(editor_value, feather_radius, show_scores):
    if not editor_value or editor_value.get("background") is None:
        return None, None, _SCORE_PLACEHOLDER, "Upload an image and brush over the damage first."
    image = editor_value["background"].convert("RGB")
    mask = mask_from_editor(editor_value)
    if mask is None:
        return None, None, _SCORE_PLACEHOLDER, "No brush strokes found -- paint over the damaged areas."
    try:
        inpainter = get_inpainter()
        restored = inpainter.inpaint(image, mask)
        result = composite(image, restored, mask, feather_radius=int(feather_radius))
    except ValueError as e:
        return None, None, _SCORE_PLACEHOLDER, f"{e}"
    except Exception as e:
        return None, None, _SCORE_PLACEHOLDER, f"Repair failed: {e}"
    scorecard = compute_scorecard(image, result, show_scores)
    return result, (image, result), scorecard, f"Repair complete (LaMa on {inpainter.device})."


def clear_brush():
    return None, None, None, None, _SCORE_PLACEHOLDER, "Cleared. Upload an image to begin."


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

with gr.Blocks(title="AI Photo Restoration Studio") as app:
    gr.HTML(
        """
        <div class="header-container">
            <div class="header-title">AI Photo Restoration Studio</div>
            <div class="header-subtitle">Diagnose &middot; route &middot; restore &mdash; faces, upscaling, and seamless brush repair</div>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------------------------------------------------------- Auto
        with gr.Tab("Auto Restore"):
            gr.Markdown(
                "Upload a photo and click **Diagnose** to get a recommended "
                "pipeline, then **Restore**. Faces are handled by CodeFormer; "
                "low-res / soft / noisy images by Real-ESRGAN."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Input</div>')
                    auto_input = gr.Image(type="pil", label="", show_label=False, height=360)
                    diagnose_btn = gr.Button("Diagnose", elem_classes=["primary-btn"])
                    diag_card = gr.HTML("<div class='diag-card'>Upload an image to analyze.</div>")

                    gr.HTML('<div class="section-label">Settings</div>')
                    do_face = gr.Checkbox(label="Restore faces (CodeFormer)", value=True)
                    fidelity = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.05, label="Face fidelity",
                        info="0 = max quality (more generative) - 1 = faithful to the original",
                    )
                    outscale = gr.Radio(
                        ["1x", "2x", "4x"], value="2x", label="Upscale (Real-ESRGAN)",
                    )
                    auto_scores = gr.Checkbox(
                        label="Compute quality scores (IQA)", value=True,
                        info="No-reference before/after metrics. First run downloads weights.",
                    )
                    with gr.Row():
                        restore_btn = gr.Button("Restore", variant="primary", elem_classes=["primary-btn"], size="lg")
                        auto_clear_btn = gr.Button("Clear")
                    auto_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Result</div>')
                    auto_result = gr.Image(type="pil", label="", show_label=False, height=360)
                    gr.HTML('<div class="section-label">Before / After</div>')
                    auto_compare = gr.ImageSlider(type="pil", label="", show_label=False, height=360)
                    gr.HTML('<div class="section-label">Quality scores</div>')
                    auto_scorecard = gr.HTML(_SCORE_PLACEHOLDER)

            diagnose_btn.click(
                fn=analyze, inputs=[auto_input],
                outputs=[diag_card, do_face, fidelity, outscale, auto_status],
            )
            restore_btn.click(
                fn=restore_auto, inputs=[auto_input, do_face, fidelity, outscale, auto_scores],
                outputs=[auto_result, auto_compare, auto_scorecard, auto_status],
            )
            auto_clear_btn.click(
                fn=clear_auto, inputs=[],
                outputs=[auto_input, auto_result, auto_compare, diag_card, auto_scorecard, auto_status],
            )

        # --------------------------------------------------------------- Brush
        with gr.Tab("Brush Repair"):
            gr.Markdown(
                "Brush over a localized defect (scratch, tear, spot, or an object "
                "to remove). LaMa fills only the brushed region by continuing the "
                "surrounding structure; the result is composited back at full "
                "resolution so nothing else changes."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Input &mdash; brush the damage</div>')
                    editor = gr.ImageEditor(
                        type="pil", label="", show_label=False, height=420,
                        sources=["upload", "clipboard"], transforms=(), layers=False,
                        brush=gr.Brush(default_size=12, colors=["#FF3B3B"],
                                       default_color="#FF3B3B", color_mode="fixed"),
                        eraser=gr.Eraser(default_size=14),
                    )
                    feather = gr.Slider(
                        minimum=0, maximum=15, value=3, step=1, label="Edge feather (px)",
                        info="Softens the repaired boundary for a seamless blend.",
                    )
                    brush_scores = gr.Checkbox(
                        label="Compute quality scores (IQA)", value=True,
                        info="No-reference before/after metrics on the full image.",
                    )
                    with gr.Accordion("Auto-suggest damage (optional hint)", open=False):
                        sensitivity = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Sensitivity")
                        with gr.Row():
                            detect_bright = gr.Checkbox(label="Bright spots", value=True)
                            detect_dark = gr.Checkbox(label="Dark spots", value=True)
                        min_area = gr.Slider(10, 500, value=50, step=10, label="Min. region area")
                        suggest_btn = gr.Button("Suggest damage")
                        suggestion = gr.Image(label="Suggested damage (reference only)", height=200)
                    with gr.Row():
                        repair_btn = gr.Button("Repair", variant="primary", elem_classes=["primary-btn"], size="lg")
                        clear_btn = gr.Button("Clear")
                    status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Result</div>')
                    result_image = gr.Image(type="pil", label="", show_label=False, height=420)
                    gr.HTML('<div class="section-label">Before / After</div>')
                    comparison = gr.ImageSlider(type="pil", label="", show_label=False, height=420)
                    gr.HTML('<div class="section-label">Quality scores</div>')
                    brush_scorecard = gr.HTML(_SCORE_PLACEHOLDER)

            suggest_btn.click(
                fn=suggest, inputs=[editor, sensitivity, detect_bright, detect_dark, min_area],
                outputs=[suggestion, status],
            )
            repair_btn.click(
                fn=repair, inputs=[editor, feather, brush_scores],
                outputs=[result_image, comparison, brush_scorecard, status],
            )
            clear_btn.click(
                fn=clear_brush, inputs=[],
                outputs=[editor, suggestion, result_image, comparison, brush_scorecard, status],
            )

        # ----------------------------------------------------------- Benchmark
        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Upload a **clean** image. We apply known degradations, restore "
                "the result, and measure how close it gets to the original with "
                "full-reference metrics (PSNR / SSIM / LPIPS) -- something only "
                "possible when the ground truth is known."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Clean image (ground truth)</div>')
                    bench_input = gr.Image(type="pil", label="", show_label=False, height=300)

                    gr.HTML('<div class="section-label">Degradations to apply</div>')
                    bench_downscale = gr.Radio(["1x", "2x", "4x"], value="2x", label="Downscale")
                    bench_blur = gr.Slider(0.0, 3.0, value=1.0, step=0.25, label="Blur sigma")
                    bench_noise = gr.Slider(0, 25, value=8, step=1, label="Noise sigma")
                    bench_jpeg = gr.Slider(10, 100, value=60, step=5, label="JPEG quality (100 = off)")

                    gr.HTML('<div class="section-label">Restoration</div>')
                    bench_face = gr.Checkbox(label="Restore faces (CodeFormer)", value=True)
                    bench_fidelity = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Face fidelity")
                    with gr.Row():
                        bench_btn = gr.Button("Run benchmark", variant="primary", elem_classes=["primary-btn"], size="lg")
                        bench_clear_btn = gr.Button("Clear")
                    bench_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Column(scale=1):
                    with gr.Row():
                        bench_clean = gr.Image(type="pil", label="Ground truth", height=220)
                        bench_degraded = gr.Image(type="pil", label="Degraded (bicubic)", height=220)
                        bench_restored = gr.Image(type="pil", label="Restored", height=220)
                    gr.HTML('<div class="section-label">Full-reference scores</div>')
                    bench_table = gr.HTML(_BENCH_PLACEHOLDER)

            bench_btn.click(
                fn=run_benchmark_handler,
                inputs=[bench_input, bench_downscale, bench_blur, bench_noise,
                        bench_jpeg, bench_face, bench_fidelity],
                outputs=[bench_clean, bench_degraded, bench_restored, bench_table, bench_status],
            )
            bench_clear_btn.click(
                fn=clear_benchmark, inputs=[],
                outputs=[bench_input, bench_clean, bench_degraded, bench_restored, bench_table, bench_status],
            )


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
