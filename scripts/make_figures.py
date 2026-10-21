"""
Regenerate every figure and chart for the compositing write-up into docs/figures/.

Inputs, and nothing else:
  * data/manifest.csv and the images it lists (sha256-checked),
  * results/compositing/raw.csv and summary.csv (the committed full run),
  * the experiment seed, read from raw.csv,
  * the selection rules in docs/figures/manifest.json ("rules"; the script
    seeds the file with DEFAULT_RULES when it does not exist yet).

Example images are recomputed with the experiment's own code: the same mask
(synthesize_mask), damage fill, LaMa at the 1536 px cap and at 512 px (CPU),
and the composite / legacy_composite definitions from the runner. Every
recomputed condition's U pct_changed and M PSNR-Y must equal its raw.csv row
exactly, or the script stops before writing anything.

Charts 1-3 read results/compositing/summary.csv (means and 95% bootstrap
CIs) and raw.csv (per-image values); chart 4 recomputes the halo reach on the
identity-test inputs of tests/test_compositing.py with the fixed composite
and legacy_composite.

Every example image, crop window and boundary point is chosen by a fixed
rule; each rule and its outcome is written to docs/figures/manifest.json
("outcomes"), together with the sha256 of every output file. Outputs are
lossless PNG (images and charts) and SVG (charts and the schematic).

Usage (from the repo root):
  ./venv/bin/python scripts/make_figures.py
"""

import csv
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage  # noqa: E402

from compositing import feather_mask  # noqa: E402
from legacy_compositing import legacy_feather_mask  # noqa: E402
from region_metrics import luma_bt601, psnr_y_region, region_diff_stats  # noqa: E402
from synthetic_damage import apply_damage, damage_fill, synthesize_mask  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")
MANIFEST_JSON = os.path.join(OUT_DIR, "manifest.json")
DATA_MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.csv")
RAW_CSV = os.path.join(REPO_ROOT, "results", "compositing", "raw.csv")
SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "compositing", "summary.csv")
FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The runner's condition table (feather version, radius, fill) and U definition.
runner = _load_module("run_compositing_experiment",
                      os.path.join(REPO_ROOT, "scripts", "run_compositing_experiment.py"))

# ---------------------------------------------------------------------------
# Palette (dataviz reference palette, light mode), checked with
# validate_palette.js. Entity hues use only the first three categorical slots,
# which pass every check with all pairs compared (needed for the chart 3
# scatter): C_3 blue, C_3_legacy orange, N1 aqua (contrast WARN 2.74:1,
# relieved by direct labels and the tables on the page). N2 is a neutral
# grey; C_3_N2fill reuses C_3's blue as a hollow square. The coverage ramp is
# blue steps 250/450/650, which pass the ordinal checks.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"
COLOR = {"C_3": "#2a78d6", "C_3_legacy": "#eb6834", "N1": "#1baf7a", "N2": "#77766f",
         "C_3_N2fill": "#2a78d6"}
LABEL = {"C_3": "Composite (C_3)", "C_3_legacy": "Legacy composite (C_3_legacy)",
         "N1": "Raw LaMa (N1)", "N2": "512 px LaMa (N2)",
         "C_3_N2fill": "512 px fill, composited (C_3_N2fill)"}
OUTLINE = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COVERAGE_RAMP = {"1": "#86b6ef", "5": "#2a78d6", "15": "#104281"}  # ramp steps 250/450/650
COVERAGE_MARKER = {"1": "o", "5": "s", "15": "^"}

DEFAULT_RULES = {
    "seed": "taken from raw.csv (every row must share it)",
    "representative_image": {
        "candidates": "split == professional, size_class == above_cap, coverage_target == 5",
        "statistic": "N1 u_pct_changed",
        "median_over": "every above_cap pair at coverage 5 (both splits): the size-class median",
        "select": "candidate with the smallest |N1 u_pct_changed - median|",
        "tie_break": "filename ascending",
        "used_for": ["A", "B", "C", "D", "E"],
    },
    "verify": {
        "conditions": ["C_0", "C_3", "C_15", "C_3_legacy", "C_15_legacy", "C_3_N2fill",
                       "N1", "N2"],
        "fields": ["u_pct_changed", "m_psnr_y", "coverage_actual"],
        "comparison": "exact float equality with the raw.csv row",
    },
    "A": {
        "panels": ["N1", "C_3"],
        "value": "per-pixel max over RGB of |X - I|",
        "clip_levels": 10,
        "colour_scale": "11 discrete levels 0..10: 0 = surface, 1..10 = blue ramp 100..700",
        "outline": "pixels of M with a 4-neighbour outside M, 1 px",
        "display_long_edge": 1600,
        "display_resample": "clipped values box-averaged to the display panel size, then the "
                            "same scale (linear between levels); outline = 4-boundary of "
                            "pixels whose box-averaged M is > 0",
    },
    "B": {
        "panels": ["I", "N1", "N2", "C_3"],
        "window": 128,
        "inside": "U (fixed alpha == 0 at r=15), every pixel of the window",
        "energy": "sum over the window of gx^2 + gy^2, np.gradient of BT.601 luma Y (0-255) "
                  "of I over the full image",
        "percentile": 90,
        "percentile_method": "inverted_cdf",
        "percentile_over": "every window lying entirely inside U",
        "select": "window whose energy is closest to the percentile; tie: smallest (y, x)",
        "upscale": 4,
        "upscale_method": "nearest",
    },
    "C": {
        "panels": ["D", "C_3_legacy", "C_3"],
        "crop": 96,
        "boundary_points": "pixels of M with a 4-neighbour outside M, whose 96x96 crop "
                           "centred on them lies inside the image",
        "score": "sum over a 9x9 window centred on the point of sum over RGB of "
                 "|C_3_legacy - C_3|",
        "score_window": 9,
        "percentile": 90,
        "percentile_method": "inverted_cdf",
        "percentile_over": "every boundary point",
        "select": "point whose score is closest to the percentile; tie: smallest (y, x)",
        "crop_box": "rows y-48..y+47, columns x-48..x+47",
        "upscale": 4,
        "upscale_method": "nearest",
        "alpha_line": {
            "radius": 3,
            "direction": "of the 4-neighbour steps from the point to a pixel outside M, the "
                         "one best aligned with the outward normal -grad(G_2 * M) (Gaussian "
                         "sigma 2); ties in listed order (0,1),(0,-1),(1,0),(-1,0) as (dy,dx)",
            "samples": "pixels p + j*d for j = -23..24; x = j - 0.5 px from the mask edge "
                       "(negative = towards M)",
            "values": "alpha / 255 of feather_mask (fixed) and legacy_feather_mask (legacy)",
        },
    },
    "D": {"panels": ["C_0", "C_3", "I"], "location": "same crop as C",
          "upscale": 4, "upscale_method": "nearest"},
    "E": {"panels": ["C_3", "C_3_N2fill", "I"], "location": "same crop as C",
          "upscale": 4, "upscale_method": "nearest"},
    "F": {"form": "SVG schematic: boxes, arrows, short labels"},
    "charts": {
        "1": {"source": "summary.csv section metric, above_cap, u_pct_changed",
              "conditions": ["N2", "N1", "C_3"], "error_bars": "95% bootstrap CI"},
        "2": {"source": "raw.csv m_psnr_y per (image, coverage)",
              "x": "C_3_legacy", "y": "C_3", "panels": ["above_cap", "at_or_below_cap"],
              "reference": "y = x"},
        "3": {"source": "summary.csv section metric, above_cap, coverage 5",
              "x": "off_lpips", "y": "m_lpips",
              "conditions": ["C_3", "C_3_legacy", "C_3_N2fill", "N1", "N2"],
              "error_bars": "95% bootstrap CI on both axes",
              "x_scale": "log when the largest x mean exceeds 4x the second largest"},
        "4": {"source": "tests/test_compositing.py identity-test inputs (both sizes), "
                        "max Euclidean distance from the binary mask of any changed pixel",
              "radii": [0, 3, 15], "versions": ["legacy", "fixed"]},
    },
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def load_rules():
    """Rules recorded in manifest.json win; rule keys it lacks come from DEFAULT_RULES."""
    rules = dict(DEFAULT_RULES)
    if os.path.exists(MANIFEST_JSON):
        with open(MANIFEST_JSON) as f:
            rules.update(json.load(f).get("rules") or {})
    return rules


def read_raw():
    with open(RAW_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    seeds = {r["seed"] for r in rows}
    if len(seeds) != 1:
        raise SystemExit(f"raw.csv mixes seeds {seeds}")
    return rows, int(seeds.pop())


def raw_index(rows):
    return {(r["split"], r["image"], r["coverage_target"], r["condition"]): r for r in rows}


def select_representative(rows, rule):
    n1 = [r for r in rows if r["condition"] == "N1" and r["size_class"] == "above_cap"
          and r["coverage_target"] == "5"]
    median = statistics.median(float(r["u_pct_changed"]) for r in n1)
    cands = [r for r in n1 if r["split"] == "professional"]
    ranked = sorted(cands, key=lambda r: (abs(float(r["u_pct_changed"]) - median), r["image"]))
    best = ranked[0]
    prof_median = statistics.median(float(r["u_pct_changed"]) for r in cands)
    alt = sorted(cands, key=lambda r: (abs(float(r["u_pct_changed"]) - prof_median),
                                       r["image"]))[0]
    return {
        "split": best["split"], "filename": best["image"],
        "image_index": int(best["image_index"]), "coverage_target": best["coverage_target"],
        "width": int(best["width"]), "height": int(best["height"]),
        "n1_u_pct_changed": float(best["u_pct_changed"]),
        "class_median_n1_u_pct_changed": median, "n_class_pairs": len(n1),
        "n_candidates": len(cands),
        "distance_to_median": abs(float(best["u_pct_changed"]) - median),
        "runner_up": {"filename": ranked[1]["image"],
                      "distance_to_median": abs(float(ranked[1]["u_pct_changed"]) - median)},
        "alternative_if_professional_only_median": {
            "median": prof_median, "filename": alt["image"]},
    }


def recompute(rep, seed, index):
    """Recompute I, M, D and every condition for the representative pair; verify."""
    from restorers.inpaint import LamaInpainter

    with open(DATA_MANIFEST, newline="") as f:
        mrow = list(csv.DictReader(f))[rep["image_index"]]
    if (mrow["split"], mrow["filename"]) != (rep["split"], rep["filename"]):
        raise SystemExit("image_index does not point at the representative image")
    if mrow["split"] != "professional":
        raise SystemExit("figures may only use Professional images")
    path = os.path.join(REPO_ROOT, "data", "clic2020", mrow["split"], mrow["filename"])
    if sha256_file(path) != mrow["sha256"]:
        raise SystemExit(f"{path}: sha256 does not match data/manifest.csv")

    I = np.asarray(Image.open(path).convert("RGB"))
    h, w = I.shape[:2]
    coverage = float(rep["coverage_target"]) / 100.0
    mask, info = synthesize_mask(w, h, coverage, rep["image_index"], seed)
    D = apply_damage(I, mask, damage_fill(h, w, rep["image_index"], coverage, seed))
    m_img, d_img = Image.fromarray(mask, "L"), Image.fromarray(D, "RGB")

    lama = LamaInpainter(device="cpu")
    lama_n2 = LamaInpainter(device="cpu", max_dimension=runner.N2_MAX_DIMENSION)
    L = np.asarray(lama.inpaint(d_img, m_img))
    N2 = np.asarray(lama_n2.inpaint(d_img, m_img))
    if lama.device != "cpu" or lama_n2.device != "cpu":
        raise SystemExit("LaMa did not run on CPU")
    fills = {"L": Image.fromarray(L, "RGB"), "N2": Image.fromarray(N2, "RGB")}

    out = {"N1": L, "N2": N2}
    alphas = {}
    for cond, (version, r, source) in runner.COMPOSITES.items():
        composite_fn, feather_fn = runner.FEATHERS[version]
        alphas[cond] = np.asarray(feather_fn(m_img, (w, h), r))
        out[cond] = np.asarray(composite_fn(d_img, fills[source], m_img, feather_radius=r))

    U = runner.u_region(mask)
    in_m = mask > 0
    checks = []
    for cond in runner.CONDITIONS:
        row = index[(rep["split"], rep["filename"], rep["coverage_target"], cond)]
        got = {"u_pct_changed": region_diff_stats(out[cond], I, U)["pct_changed"],
               "m_psnr_y": psnr_y_region(out[cond], I, in_m),
               "coverage_actual": 100.0 * info["coverage"]}
        for field, value in got.items():
            expected = float(row[field])
            checks.append({"condition": cond, "field": field, "recomputed": value,
                           "raw_csv": expected, "equal": value == expected})
    bad = [c for c in checks if not c["equal"]]
    if bad:
        raise SystemExit("Recomputed metrics differ from raw.csv:\n  " + "\n  ".join(
            f"{c['condition']} {c['field']}: recomputed {c['recomputed']!r}, raw.csv "
            f"{c['raw_csv']!r}" for c in bad))
    return {"I": I, "M": mask, "D": D, "U": U, "out": out, "alphas": alphas,
            "lama_devices": [lama.device, lama_n2.device]}, checks


# ---------------------------------------------------------------------------
# Image composition (exact pixels: panels are pasted, never resampled)
# ---------------------------------------------------------------------------
def font(px):
    return ImageFont.truetype(FONT_PATH, px)


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def text_width(f, s):
    left, _, right, _ = f.getbbox(s)
    return right - left


def compose(panels, labels, font_px, gap, margin, right=None):
    """Paste equal-height panels in a row with a label above each.

    A 1 px hairline is drawn just outside each panel; the panel pixels are
    pasted unchanged. ``right`` is an optional extra image (e.g. a colourbar)
    placed after the last panel, top-aligned with the panels.
    """
    f = font(font_px)
    ascent, descent = f.getmetrics()
    label_h = ascent + descent + font_px // 2
    ph = panels[0].shape[0]
    if any(p.shape[0] != ph for p in panels):
        raise ValueError("panels differ in height")
    for p, lab in zip(panels, labels):
        if text_width(f, lab) > p.shape[1]:
            raise ValueError(f"label {lab!r} is wider than its panel")
    width = 2 * margin + sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    if right is not None:
        width += gap + right.width
    height = 2 * margin + label_h + max(ph, right.height if right is not None else 0)
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    x, y = margin, margin + label_h
    for p, lab in zip(panels, labels):
        draw.text((x, margin), lab, font=f, fill=INK)
        draw.rectangle([x - 1, y - 1, x + p.shape[1], y + ph], outline=GRID)
        canvas.paste(Image.fromarray(p), (x, y))
        x += p.shape[1] + gap
    if right is not None:
        canvas.paste(right, (x, y))
    return canvas


def upscale(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def crop(a, box):
    x0, y0, w, h = box
    return a[y0:y0 + h, x0:x0 + w]


def save_png(img, name, outputs):
    path = os.path.join(OUT_DIR, name)
    img.save(path, format="PNG", optimize=False)
    outputs.append(name)
    return path


def inner_boundary(m):
    """Pixels of boolean m with a 4-neighbour outside m (image edge counts as outside)."""
    padded = np.pad(m, 1, constant_values=False)
    nb_out = (~padded[:-2, 1:-1]) | (~padded[2:, 1:-1]) | (~padded[1:-1, :-2]) | (~padded[1:-1, 2:])
    return m & nb_out


# ---------------------------------------------------------------------------
# Figure A: untouched-pixel change maps
# ---------------------------------------------------------------------------
def level_lut(clip):
    ramp = np.array([hex_rgb(c) for c in BLUE_RAMP], dtype=np.float64)
    pos = np.linspace(0, len(ramp) - 1, clip)          # levels 1..clip along the ramp
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    t = (pos - lo)[:, None]
    levels = (1 - t) * ramp[lo] + t * ramp[hi]
    return np.vstack([np.array(hex_rgb(SURFACE), float), levels])   # (clip + 1, 3)


def colourise(values, lut):
    """values in [0, clip] (float); linear between integer levels."""
    lo = np.floor(values).astype(int)
    hi = np.minimum(lo + 1, len(lut) - 1)
    t = (values - lo)[..., None]
    return np.rint((1 - t) * lut[lo] + t * lut[hi]).astype(np.uint8)


def colourbar(height, lut, font_px, heading):
    f = font(font_px)
    ascent, descent = f.getmetrics()
    line_h = ascent + descent
    bar_w = int(round(font_px * 1.2))
    n = len(lut)
    head_h = line_h * len(heading) + font_px // 2
    bar_h = height - head_h - line_h // 2
    labels = {0: "0", 2: "2", 4: "4", 6: "6", 8: "8", n - 1: f"≥{n - 1}"}
    width = max(bar_w + font_px // 2 + max(text_width(f, s) for s in labels.values()),
                max(text_width(f, s) for s in heading))
    img = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(heading):
        draw.text((0, i * line_h), s, font=f, fill=INK2)
    top = head_h
    edges = np.linspace(top + bar_h, top, n + 1)        # level 0 at the bottom
    for lvl in range(n):
        y1, y0 = int(round(edges[lvl])), int(round(edges[lvl + 1]))
        draw.rectangle([0, y0, bar_w - 1, y1 - 1], fill=tuple(int(v) for v in lut[lvl]))
        if lvl in labels:
            yc = (y0 + y1) / 2
            draw.text((bar_w + font_px // 2, yc), labels[lvl], font=f, fill=INK2, anchor="lm")
    draw.rectangle([0, top, bar_w - 1, top + bar_h - 1], outline=GRID)
    return img


def figure_a(ex, rules, outputs):
    rule = rules["A"]
    clip = rule["clip_levels"]
    lut = level_lut(clip)
    I, M = ex["I"], ex["M"] > 0
    h, w = M.shape
    vals = {c: np.minimum(np.abs(ex["out"][c].astype(np.int16) - I.astype(np.int16)).max(-1),
                          clip) for c in rule["panels"]}
    labels = ["Raw LaMa: |N1 − I|", "Composite: |C_3 − I|"]
    heading = ["max over RGB", "(levels)"]
    outline_rgb = np.array(hex_rgb(OUTLINE), np.uint8)

    # Full resolution: one image pixel per panel pixel.
    edge = inner_boundary(M)
    panels = []
    for c in rule["panels"]:
        rgb = colourise(vals[c].astype(np.float64), lut)
        rgb[edge] = outline_rgb
        panels.append(rgb)
    fpx = max(16, int(round(w * 0.022)))
    full = compose(panels, labels, fpx, gap=fpx, margin=fpx,
                   right=colourbar(h, lut, fpx, heading))
    save_png(full, "fig_a_change_maps_full.png", outputs)

    # Display version: the whole figure has a 1600 px long edge.
    target = rule["display_long_edge"]
    fpx_d, gap_d, margin_d = 16, 16, 16
    cbar_probe = colourbar(400, lut, fpx_d, heading)
    pw = (target - 2 * margin_d - 2 * gap_d - cbar_probe.width) // 2
    ph = int(round(pw * h / w))
    m_box = np.asarray(Image.fromarray(M.astype(np.float32), "F").resize((pw, ph), Image.BOX))
    m_disp = m_box > 0
    edge_d = inner_boundary(m_disp)
    panels_d = []
    for c in rule["panels"]:
        v = np.asarray(Image.fromarray(vals[c].astype(np.float32), "F").resize((pw, ph), Image.BOX))
        rgb = colourise(np.clip(v.astype(np.float64), 0, clip), lut)
        rgb[edge_d] = outline_rgb
        panels_d.append(rgb)
    disp = compose(panels_d, labels, fpx_d, gap=gap_d, margin=margin_d,
                   right=colourbar(ph, lut, fpx_d, heading))
    if disp.width < target:          # pad the right margin to land exactly on target
        padded = Image.new("RGB", (target, disp.height), SURFACE)
        padded.paste(disp, (0, 0))
        disp = padded
    if max(disp.size) != target:
        raise RuntimeError(f"display version long edge {max(disp.size)} != {target}")
    save_png(disp, "fig_a_change_maps.png", outputs)
    return {"panel_size_full": [w, h], "panel_size_display": [pw, ph],
            "full_size": list(full.size), "display_size": list(disp.size),
            "max_clipped_value": {c: int(vals[c].max()) for c in rule["panels"]},
            "pixels_at_or_above_clip": {c: int((vals[c] >= clip).sum()) for c in rule["panels"]}}


# ---------------------------------------------------------------------------
# Figure B: detail crops inside U
# ---------------------------------------------------------------------------
def window_sums(a, k):
    """Sum of a over every k x k window; result[y, x] is the window with top-left (y, x)."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


def figure_b(ex, rules, outputs):
    rule = rules["B"]
    k = rule["window"]
    I, U = ex["I"], ex["U"]
    Y = luma_bt601(I) * 255.0
    gy, gx = np.gradient(Y)
    energy = window_sums(gx * gx + gy * gy, k)
    inside = window_sums(U.astype(np.float64), k) == k * k
    vals = energy[inside]
    if vals.size == 0:
        raise SystemExit("no 128x128 window lies entirely inside U")
    pct = float(np.percentile(vals, rule["percentile"], method=rule["percentile_method"]))
    dist = np.where(inside, np.abs(energy - pct), np.inf)
    y0, x0 = np.unravel_index(int(np.argmin(dist)), dist.shape)
    box = (int(x0), int(y0), k, k)
    if not U[y0:y0 + k, x0:x0 + k].all():
        raise RuntimeError("selected window leaves U")
    names = rule["panels"]
    arrays = {"I": I, **ex["out"]}
    panels = [upscale(crop(arrays[n], box), rule["upscale"]) for n in names]
    labels = ["Original (I)", "Raw LaMa (N1)", "512 px LaMa (N2)", "Composite (C_3)"]
    save_png(compose(panels, labels, 22, gap=24, margin=24), "fig_b_detail_crops.png", outputs)
    return {"crop_box_xywh": list(box), "energy": float(energy[y0, x0]),
            "percentile_value": pct, "n_valid_windows": int(vals.size),
            "pixels_changed_in_crop": {n: int((crop(arrays[n], box) != crop(I, box))
                                              .any(-1).sum()) for n in names}}


# ---------------------------------------------------------------------------
# Figures C, D, E: the feather defect, seams, and the 512 px fill
# ---------------------------------------------------------------------------
def select_boundary_point(ex, rule):
    M = ex["M"] > 0
    h, w = M.shape
    half = rule["crop"] // 2
    diff = np.abs(ex["out"]["C_3_legacy"].astype(np.int32)
                  - ex["out"]["C_3"].astype(np.int32)).sum(-1).astype(np.float64)
    sw = rule["score_window"]
    sums = window_sums(diff, sw)                  # top-left indexed
    r = sw // 2
    pts = np.argwhere(inner_boundary(M))
    ok = ((pts[:, 0] >= half) & (pts[:, 0] <= h - half) & (pts[:, 1] >= half)
          & (pts[:, 1] <= w - half))
    pts = pts[ok]
    if pts.size == 0:
        raise SystemExit("no boundary point admits a full crop")
    scores = sums[pts[:, 0] - r, pts[:, 1] - r]
    pct = float(np.percentile(scores, rule["percentile"], method=rule["percentile_method"]))
    order = np.lexsort((pts[:, 1], pts[:, 0], np.abs(scores - pct)))
    y, x = (int(v) for v in pts[order[0]])
    return {"point_yx": [y, x], "score": float(scores[order[0]]), "percentile_value": pct,
            "n_boundary_points": int(len(pts)),
            "crop_box_xywh": [x - half, y - half, rule["crop"], rule["crop"]]}


def alpha_profile(ex, point, rule):
    lrule = rule["alpha_line"]
    M = ex["M"] > 0
    y, x = point
    normal = -np.array(np.gradient(ndimage.gaussian_filter(M.astype(np.float64), 2.0)))[:, y, x]
    steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    ok = [d for d in steps if not M[y + d[0], x + d[1]]]
    d = max(ok, key=lambda s: (s[0] * normal[0] + s[1] * normal[1], -steps.index(s)))
    js = np.arange(-23, 25)
    ys, xs = y + js * d[0], x + js * d[1]
    r = lrule["radius"]
    m_img = Image.fromarray(ex["M"], "L")
    size = (M.shape[1], M.shape[0])
    fixed = np.asarray(feather_mask(m_img, size, r))[ys, xs] / 255.0
    legacy = np.asarray(legacy_feather_mask(m_img, size, r))[ys, xs] / 255.0
    return {"direction_dydx": list(d), "normal_dydx": [float(normal[0]), float(normal[1])],
            "x_px": (js - 0.5).tolist(), "in_M": M[ys, xs].tolist(),
            "alpha_fixed": fixed.tolist(), "alpha_legacy": legacy.tolist()}


def figure_cde(ex, rules, outputs):
    rule = rules["C"]
    sel = select_boundary_point(ex, rule)
    box = tuple(sel["crop_box_xywh"])
    arrays = {"I": ex["I"], "D": ex["D"], **ex["out"]}
    k = rule["upscale"]
    specs = {
        "C": (rule["panels"], ["Damaged input (D)", "Legacy composite", "Fixed composite (C_3)"],
              "fig_c_feather_defect.png"),
        "D": (rules["D"]["panels"], ["Hard edge (C_0)", "Feathered (C_3)", "Original (I)"],
              "fig_d_seams.png"),
        "E": (rules["E"]["panels"], ["1536 px fill (C_3)", "512 px fill (C_3_N2fill)",
                                     "Original (I)"], "fig_e_n2fill.png"),
    }
    for _, (names, labels, fname) in specs.items():
        panels = [upscale(crop(arrays[n], box), k) for n in names]
        save_png(compose(panels, labels, 22, gap=24, margin=24), fname, outputs)
    prof = alpha_profile(ex, sel["point_yx"], rule)
    alpha_chart(prof, outputs)
    return sel, prof


# ---------------------------------------------------------------------------
# Charts (matplotlib; 1 CSS px = 0.72 pt at 100 dpi, saved at 2x)
# ---------------------------------------------------------------------------
PX = 0.72
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13 * PX * 1.0,
    "svg.hashsalt": "make_figures", "svg.fonttype": "path",
    "axes.edgecolor": AXIS, "axes.linewidth": PX, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": PX, "ytick.major.width": PX,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=PX, linestyle="-")
    ax.set_axisbelow(True)


def save_chart(fig, stem, outputs):
    for ext, meta in (("png", {"Software": None}), ("svg", {"Date": None})):
        name = f"{stem}.{ext}"
        fig.savefig(os.path.join(OUT_DIR, name), dpi=200, metadata=meta)
        outputs.append(name)
    plt.close(fig)


def alpha_chart(prof, outputs):
    x = np.array(prof["x_px"])
    in_m = np.array(prof["in_M"])
    fig = plt.figure(figsize=(7.2, 3.6), dpi=100)
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.72])
    style_axes(ax, grid_axis="y")
    # Shade the samples that lie inside M.
    for xi, inside in zip(x, in_m):
        if inside:
            ax.axvspan(xi - 0.5, xi + 0.5, color=WASH, linewidth=0, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=PX, zorder=1)
    ax.plot(x, prof["alpha_legacy"], color=COLOR["C_3_legacy"], linewidth=2 * PX,
            solid_joinstyle="round", solid_capstyle="round", label="Legacy feather", zorder=3)
    ax.plot(x, prof["alpha_fixed"], color=COLOR["C_3"], linewidth=2 * PX,
            solid_joinstyle="round", solid_capstyle="round", label="Fixed feather", zorder=3)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("px from mask edge (negative: towards M)")
    ax.set_ylabel("alpha")
    # Label the shaded band (the samples inside M) once, centred over it.
    xt = ax.get_xaxis_transform()        # x in data, y in axes fraction
    ax.text(float(np.mean(x[in_m])), 1.02, "M", ha="center", va="bottom", color=INK2,
            transform=xt)
    ax.legend(loc="center right")
    save_chart(fig, "fig_c_alpha_profile", outputs)


def bar_path(x0, x1, h, rx, ry):
    """Bar from the baseline to h with rounded data-end corners (radius rx, ry in data units)."""
    from matplotlib.path import Path
    ry = min(ry, h)
    rx = min(rx, (x1 - x0) / 2)
    if h <= 0:
        return None
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h), (x1, h), (x1, h - ry),
             (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes)


def data_per_px(ax):
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return 1.0 / (x1 - x0), 1.0 / (y1 - y0)


def read_summary():
    with open(SUMMARY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["section"] == "metric" and r["mean"] != "":
            out[(r["size_class"], r["coverage_target"], r["condition"], r["metric"])] = (
                float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]), int(r["n"]))
    return out


def chart_1(summary, outputs):
    from matplotlib.patches import PathPatch
    conds = ["N2", "N1", "C_3"]
    covs = ["1", "5", "15"]
    fig = plt.figure(figsize=(7.2, 4.0), dpi=100)
    ax = fig.add_axes([0.09, 0.12, 0.88, 0.70])
    style_axes(ax, grid_axis="y")
    ax.set_xlim(-0.6, len(covs) - 0.4)
    ax.set_ylim(0, 105)
    dx, dy = data_per_px(ax)
    bw, gap = 22 * dx, 2 * dx
    values = {}
    for gi, cov in enumerate(covs):
        for bi, cond in enumerate(conds):
            mean, lo, hi, n = summary[("above_cap", cov, cond, "u_pct_changed")]
            values[f"{cond}@{cov}"] = [mean, lo, hi, n]
            xc = gi + (bi - 1) * (bw + gap)
            path = bar_path(xc - bw / 2, xc + bw / 2, mean, 4 * dx, 4 * dy)
            if path is not None:
                ax.add_patch(PathPatch(path, facecolor=COLOR[cond], edgecolor="none", zorder=2))
            ax.errorbar([xc], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                        ecolor=INK2, elinewidth=PX, capsize=3, capthick=PX, zorder=3)
            ax.text(xc, hi + 6 * dy, f"{mean:.1f}", ha="center", va="bottom", color=INK2,
                    fontsize=11 * PX)
    ax.axhline(0, color=AXIS, linewidth=PX, zorder=4)
    ax.set_xticks(range(len(covs)), [f"{c}% coverage" for c in covs])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("U pixels changed (%), above cap")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in conds]
    fig.legend(handles, [LABEL[c] for c in conds], loc="upper left", ncol=3,
               bbox_to_anchor=(0.08, 0.98), handlelength=1.0, columnspacing=1.6)
    save_chart(fig, "chart_1_u_changed", outputs)
    return values


def chart_2(rows, outputs):
    pts = {}
    for r in rows:
        if r["condition"] in ("C_3", "C_3_legacy"):
            key = (r["size_class"], r["split"], r["image"], r["coverage_target"])
            pts.setdefault(key, {})[r["condition"]] = float(r["m_psnr_y"])
    xs = [v["C_3_legacy"] for v in pts.values()]
    ys = [v["C_3"] for v in pts.values()]
    lo = np.floor(min(xs + ys) / 5) * 5
    hi = np.ceil(max(xs + ys) / 5) * 5
    fig = plt.figure(figsize=(9.6, 4.6), dpi=100)
    panels = [("above_cap", "Above the 1536 px cap"), ("at_or_below_cap", "At or below the cap")]
    counts = {}
    for pi, (sc, title) in enumerate(panels):
        ax = fig.add_axes([0.07 + pi * 0.48, 0.12, 0.40, 0.72])
        style_axes(ax, grid_axis="both")
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=PX, zorder=1)
        n = 0
        for cov in ("1", "5", "15"):
            sel = [v for k, v in pts.items() if k[0] == sc and k[3] == cov]
            n += len(sel)
            ax.scatter([v["C_3_legacy"] for v in sel], [v["C_3"] for v in sel],
                       s=(11 * PX) ** 2, marker=COVERAGE_MARKER[cov], color=COVERAGE_RAMP[cov],
                       edgecolors=SURFACE, linewidths=1.5 * PX, zorder=2,
                       label=f"{cov}% coverage")
        counts[sc] = n
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{title} · {n} pairs", loc="left", color=INK, fontsize=13 * PX)
        ax.set_xlabel("M PSNR-Y, legacy composite (dB)")
        if pi == 0:
            ax.set_ylabel("M PSNR-Y, fixed composite C_3 (dB)")
        ax.text(hi - 0.5, hi - 0.5, "y = x", ha="right", va="top", color=MUTED,
                fontsize=11 * PX, rotation=45, rotation_mode="anchor")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.99),
               markerscale=1.4)
    save_chart(fig, "chart_2_m_psnr_legacy_vs_fixed", outputs)
    above = sum(1 for k, v in pts.items() if v["C_3"] > v["C_3_legacy"])
    return {"pairs_per_panel": counts, "axis_limits_db": [float(lo), float(hi)],
            "pairs_above_y_equals_x": above, "pairs_total": len(pts)}


def chart_3(summary, outputs):
    conds = ["C_3", "C_3_legacy", "C_3_N2fill", "N1", "N2"]
    vals = {c: {"off_lpips": summary[("above_cap", "5", c, "off_lpips")],
                "m_lpips": summary[("above_cap", "5", c, "m_lpips")]} for c in conds}
    xm = sorted(v["off_lpips"][0] for v in vals.values())
    use_log = xm[-1] > 4 * xm[-2]
    fig = plt.figure(figsize=(7.2, 4.6), dpi=100)
    ax = fig.add_axes([0.11, 0.12, 0.85, 0.64])
    style_axes(ax, grid_axis="both")
    marker = {"C_3": "o", "C_3_legacy": "o", "C_3_N2fill": "s", "N1": "o", "N2": "o"}
    offset = {"C_3": (8, 8), "C_3_legacy": (8, 8), "C_3_N2fill": (8, -14), "N1": (8, 8),
              "N2": (-8, 8)}
    for c in conds:
        (x, xlo, xhi, _), (y, ylo, yhi, _) = vals[c]["off_lpips"], vals[c]["m_lpips"]
        ax.errorbar([x], [y], xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt="none", ecolor=INK2, elinewidth=PX, capsize=3, capthick=PX, zorder=2)
        hollow = c == "C_3_N2fill"
        ax.scatter([x], [y], s=(10 * PX) ** 2, marker=marker[c],
                   facecolors=SURFACE if hollow else COLOR[c], edgecolors=COLOR[c],
                   linewidths=2 * PX, zorder=3, label=LABEL[c])
        ax.annotate(c, (x, y), xytext=offset[c], textcoords="offset points",
                    ha="right" if offset[c][0] < 0 else "left", color=INK, fontsize=12 * PX)
    if use_log:
        ax.set_xscale("log")
    ax.set_xlabel("Off-repair LPIPS (lower is better)" + (", log scale" if use_log else ""))
    ax.set_ylabel("M LPIPS (lower is better)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=2,
               bbox_to_anchor=(0.10, 0.99), handletextpad=0.4, columnspacing=1.4)
    save_chart(fig, "chart_3_lpips_tradeoff", outputs)
    return {"x_scale": "log" if use_log else "linear",
            "values": {c: {m: list(v) for m, v in d.items()} for c, d in vals.items()}}


def halo_reach_values():
    """Max distance beyond the binary mask of any changed pixel, on the identity-test inputs."""
    from compositing import composite
    from legacy_compositing import legacy_composite
    tests = _load_module("test_compositing", os.path.join(REPO_ROOT, "tests",
                                                          "test_compositing.py"))
    out = {}
    for (w, h, seed) in tests.SIZES:
        original, restored, mask = tests._inputs(w, h, seed)
        dist = ndimage.distance_transform_edt(mask == 0)
        o, rimg, m = Image.fromarray(original), Image.fromarray(restored), Image.fromarray(mask)
        for version, fn in (("fixed", composite), ("legacy", legacy_composite)):
            for r in tests.RADII:
                res = np.asarray(fn(o, rimg, m, feather_radius=r))
                changed = (res != original).any(-1) & (mask == 0)
                out.setdefault(version, {}).setdefault(str(r), {})[f"{w}x{h}"] = (
                    float(dist[changed].max()) if changed.any() else 0.0)
    return out, tests.RADII


def chart_4(outputs):
    reach, radii = halo_reach_values()
    per_version = {}
    for version, by_r in reach.items():
        per_version[version] = []
        for r in radii:
            vals = set(by_r[str(r)].values())
            if len(vals) != 1:
                raise SystemExit(f"halo reach differs between test sizes: {version} r={r}")
            per_version[version].append(vals.pop())
    fig = plt.figure(figsize=(7.2, 4.0), dpi=100)
    ax = fig.add_axes([0.10, 0.13, 0.78, 0.72])
    style_axes(ax, grid_axis="y")
    for version, key in (("legacy", "C_3_legacy"), ("fixed", "C_3")):
        ys = per_version[version]
        ax.plot(radii, ys, color=COLOR[key], linewidth=2 * PX, solid_joinstyle="round",
                solid_capstyle="round", marker="o", markersize=8 * PX, markeredgecolor=SURFACE,
                markeredgewidth=2 * PX, label=f"{version.capitalize()} feather", zorder=3)
        for r, y in zip(radii, ys):
            if r == 0:
                continue
            ax.annotate(f"{y:.2f}", (r, y), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=11 * PX)
    ax.set_xticks(radii)
    ax.set_xlim(-0.8, max(radii) + 2.6)
    ax.set_ylim(0, max(max(v) for v in per_version.values()) * 1.12)
    ax.set_xlabel("Feather radius r (px)")
    ax.set_ylabel("Max reach of changes beyond mask (px)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=2,
               bbox_to_anchor=(0.09, 0.99))
    save_chart(fig, "chart_4_halo_reach", outputs)
    return {"radii": radii, "reach_px": per_version, "per_test_size": reach}


# ---------------------------------------------------------------------------
# Figure F: schematic
# ---------------------------------------------------------------------------
def figure_f(outputs):
    W, H = 1000, 330
    boxes = {  # name: (x, y, w, h, label lines, stroke)
        "D": (20, 120, 150, 60, ["Damaged image D", "+ mask M"], AXIS),
        "down": (215, 40, 150, 60, ["Downscale to", "1536 px"], AXIS),
        "lama": (410, 40, 150, 60, ["LaMa", "fills M"], AXIS),
        "up": (605, 40, 150, 60, ["Upscale to", "full size"], AXIS),
        "n1": (800, 40, 180, 60, ["N1", "raw LaMa output"], COLOR["N1"]),
        "comp": (605, 200, 150, 60, ["Composite into D", "feather outside M"], AXIS),
        "c3": (800, 200, 180, 60, ["C_3", "composite output"], COLOR["C_3"]),
    }
    arrows = [("D", "down"), ("down", "lama"), ("lama", "up"), ("up", "n1"), ("up", "comp"),
              ("comp", "c3"), ("D", "comp")]

    def anchor(name, side):
        x, y, w, h, _, _ = boxes[name]
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y + h),
                "t": (x + w / 2, y)}[side]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, \'Segoe UI\', '
             f'sans-serif" font-size="15">',
             '<title>Raw LaMa round trip vs composite</title>',
             f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
             f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             f'fill="{INK2}"/></marker></defs>',
             f'<text x="215" y="26" fill="{INK2}" font-size="14">Raw LaMa round trip</text>',
             f'<text x="605" y="290" fill="{INK2}" font-size="14">Composite</text>']
    for name, (x, y, w, h, lines, stroke) in boxes.items():
        sw = 2 if stroke != AXIS else 1
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{SURFACE}" '
                     f'stroke="{stroke}" stroke-width="{sw}"/>')
        for i, line in enumerate(lines):
            ty = y + h / 2 + (i - (len(lines) - 1) / 2) * 19 + 5
            weight = ' font-weight="600"' if i == 0 else ""
            parts.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" fill="{INK}"'
                         f'{weight}>{line}</text>')

    def path_for(a, b):
        if (a, b) == ("D", "down"):
            (x1, y1), (x2, y2) = anchor("D", "r"), anchor("down", "l")
            return f"M{x1},{y1} L{(x1 + x2) / 2},{y1} L{(x1 + x2) / 2},{y2} L{x2 - 2},{y2}"
        if (a, b) == ("D", "comp"):
            (x1, y1), (x2, y2) = anchor("D", "b"), anchor("comp", "l")
            return f"M{x1},{y1} L{x1},{y2} L{x2 - 2},{y2}"
        if (a, b) == ("up", "comp"):
            (x1, y1), (x2, y2) = anchor("up", "b"), anchor("comp", "t")
            return f"M{x1},{y1} L{x2},{y2 - 2}"
        (x1, y1), (x2, y2) = anchor(a, "r"), anchor(b, "l")
        return f"M{x1},{y1} L{x2 - 2},{y2}"

    for a, b in arrows:
        parts.append(f'<path d="{path_for(a, b)}" fill="none" stroke="{INK2}" '
                     f'stroke-width="1.5" marker-end="url(#arrow)"/>')
    parts.append(f'<text x="{(anchor("up", "b")[0]) + 8}" y="150" fill="{INK2}" '
                 f'font-size="13">L</text>')
    parts.append("</svg>")
    name = "fig_f_schematic.svg"
    with open(os.path.join(OUT_DIR, name), "w") as f:
        f.write("\n".join(parts) + "\n")
    outputs.append(name)


# ---------------------------------------------------------------------------
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rules = load_rules()
    rows, seed = read_raw()
    index = raw_index(rows)
    rep = select_representative(rows, rules["representative_image"])
    print(f"Representative image: {rep['split']}/{rep['filename']} (N1 U changed "
          f"{rep['n1_u_pct_changed']:.4f}%, class median {rep['class_median_n1_u_pct_changed']:.4f}%)",
          flush=True)
    ex, checks = recompute(rep, seed, index)
    print(f"Verified {len(checks)} recomputed values against raw.csv: all equal.", flush=True)

    outputs, figs = [], {}
    figs["A"] = {"files": [], "source": f"{rep['split']}/{rep['filename']}",
                 "location": "full image"}
    n0 = len(outputs)
    figs["A"]["outcome"] = figure_a(ex, rules, outputs)
    figs["A"]["files"] = outputs[n0:]

    n0 = len(outputs)
    out_b = figure_b(ex, rules, outputs)
    figs["B"] = {"files": outputs[n0:], "source": f"{rep['split']}/{rep['filename']}",
                 "location": {"crop_box_xywh": out_b["crop_box_xywh"]}, "outcome": out_b}

    n0 = len(outputs)
    sel, prof = figure_cde(ex, rules, outputs)
    files = outputs[n0:]
    loc = {"crop_box_xywh": sel["crop_box_xywh"], "boundary_point_yx": sel["point_yx"]}
    figs["C"] = {"files": [f for f in files if f.startswith("fig_c")],
                 "source": f"{rep['split']}/{rep['filename']}", "location": loc,
                 "outcome": {**sel, "alpha_line": prof}}
    for fid, prefix in (("D", "fig_d"), ("E", "fig_e")):
        figs[fid] = {"files": [f for f in files if f.startswith(prefix)],
                     "source": f"{rep['split']}/{rep['filename']}", "location": loc,
                     "outcome": "same crop as C"}

    n0 = len(outputs)
    figure_f(outputs)
    figs["F"] = {"files": outputs[n0:], "source": None, "location": None,
                 "outcome": "schematic"}

    summary = read_summary()
    charts = {}
    for cid, fn in (("1", lambda o: chart_1(summary, o)), ("2", lambda o: chart_2(rows, o)),
                    ("3", lambda o: chart_3(summary, o)), ("4", chart_4)):
        n0 = len(outputs)
        outcome = fn(outputs)
        charts[cid] = {"files": outputs[n0:], "outcome": outcome}

    manifest = {
        "generated_by": "scripts/make_figures.py",
        "code": {"git_head": git("rev-parse", "HEAD"),
                 "make_figures_py_sha256": sha256_file(os.path.abspath(__file__)),
                 "make_figures_py_modified_vs_head": bool(git("status", "--porcelain", "--",
                                                              "scripts/make_figures.py"))},
        "inputs": {"raw_csv_sha256": sha256_file(RAW_CSV),
                   "summary_csv_sha256": sha256_file(SUMMARY_CSV),
                   "data_manifest_sha256": sha256_file(DATA_MANIFEST), "seed": seed,
                   "lama_devices": ex["lama_devices"]},
        "rules": rules,
        "outcomes": {"representative_image": rep, "verification": checks, "figures": figs,
                     "charts": charts},
        "images_used": [{"split": rep["split"], "filename": rep["filename"]}],
        "files": {name: sha256_file(os.path.join(OUT_DIR, name)) for name in sorted(outputs)},
    }
    if any(i["split"] != "professional" for i in manifest["images_used"]):
        raise SystemExit("a non-Professional image is used in a figure")
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(outputs)} files and {os.path.relpath(MANIFEST_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
