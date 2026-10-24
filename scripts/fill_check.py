"""
Does LaMa's output depend on the colour written into the masked pixels?

LaMa is meant to ignore the pixels under its mask. Below the 1536 px cap the
image is passed to the model as is, so that should hold exactly. Above the cap,
LamaInpainter LANCZOS-downscales the damaged image before inference, which
mixes masked pixels into their unmasked neighbours, so the fill colour can leak
into the model's input and therefore into its output.

For one image above the cap and one at or below it, this runs LaMa (CPU, the
default 1536 px cap) twice on the same mask with two different damage fills,
and reports how many mask pixels differ between the two outputs and how far
outside the mask any difference reaches.

Selection rules (fixed, not by eye):
  above_cap        the representative image recorded in docs/figures/manifest.json
  at_or_below_cap  the Professional image with the largest long edge <= 1536 px
                   (ties: filename ascending)
Mask: synthetic_damage.synthesize_mask at 5% coverage, seed 0, keyed on the
image's manifest row, exactly as in the compositing experiment.
Fills: "experiment" = synthetic_damage.damage_fill (235 + N(0, 10), seed 0),
       "black"      = every masked pixel set to (0, 0, 0).

Writes results/fill_check.json. Usage (from the repo root):
  ./venv/bin/python scripts/fill_check.py
"""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from compositing import feather_mask  # noqa: E402
from synthetic_damage import apply_damage, damage_fill, synthesize_mask  # noqa: E402

MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.csv")
FIG_MANIFEST = os.path.join(REPO_ROOT, "docs", "figures", "manifest.json")
OUT = os.path.join(REPO_ROOT, "results", "fill_check.json")
CAP = 1536
COVERAGE = 0.05
SEED = 0
U_RADIUS = 15
FEATHER_RADIUS = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def select_images():
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(FIG_MANIFEST) as f:
        rep = json.load(f)["outcomes"]["representative_image"]
    above = rows[rep["image_index"]]
    if (above["split"], above["filename"]) != (rep["split"], rep["filename"]):
        raise SystemExit("figure manifest image_index does not match data/manifest.csv")
    below = [(i, r) for i, r in enumerate(rows) if r["split"] == "professional"
             and max(int(r["width"]), int(r["height"])) <= CAP]
    below.sort(key=lambda ir: (-max(int(ir[1]["width"]), int(ir[1]["height"])),
                               ir[1]["filename"]))
    return [("above_cap", rep["image_index"], above), ("at_or_below_cap", *below[0])]


def compare(a, b, mask, U, halo3):
    inside = mask > 0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    changed = diff > 0
    outside = changed & ~inside
    dist = ndimage.distance_transform_edt(~inside)
    return {
        "mask_pixels": int(inside.sum()),
        "mask_pixels_differing": int((changed & inside).sum()),
        "mask_pct_differing": 100.0 * int((changed & inside).sum()) / int(inside.sum()),
        "mask_max_abs_diff": int(diff[inside].max()),
        "outside_pixels_differing": int(outside.sum()),
        "outside_max_abs_diff": int(diff[~inside].max()),
        "outside_max_reach_px": float(dist[outside].max()) if outside.any() else 0.0,
        "differing_pixels_in_U": int((changed & U).sum()),
        "differing_outside_pixels_beyond_r3_feather": int((outside & ~halo3).sum()),
    }


def main() -> int:
    from restorers.inpaint import LamaInpainter

    lama = LamaInpainter(device="cpu")
    if lama.max_dimension != CAP:
        raise SystemExit(f"LaMa cap is {lama.max_dimension}, expected {CAP}")
    results = []
    for size_class, index, row in select_images():
        path = os.path.join(REPO_ROOT, "data", "clic2020", row["split"], row["filename"])
        if sha256_file(path) != row["sha256"]:
            raise SystemExit(f"{path}: sha256 does not match data/manifest.csv")
        I = np.asarray(Image.open(path).convert("RGB"))
        h, w = I.shape[:2]
        mask, info = synthesize_mask(w, h, COVERAGE, index, SEED)
        m_img = Image.fromarray(mask, "L")
        fills = {
            "experiment": damage_fill(h, w, index, COVERAGE, SEED),
            "black": np.zeros((h, w, 3), dtype=np.uint8),
        }
        outputs = {}
        for name, fill in fills.items():
            D = apply_damage(I, mask, fill)
            outputs[name] = np.asarray(lama.inpaint(Image.fromarray(D, "RGB"), m_img))
        if lama.device != "cpu":
            raise SystemExit(f"LaMa ran on {lama.device}, expected cpu")
        U = np.asarray(feather_mask(m_img, (w, h), U_RADIUS)) == 0
        halo3 = np.asarray(feather_mask(m_img, (w, h), FEATHER_RADIUS)) > 0
        stats = compare(outputs["experiment"], outputs["black"], mask, U, halo3)
        results.append({
            "size_class": size_class, "image": f"{row['split']}/{row['filename']}",
            "image_index": index, "width": w, "height": h,
            "lama_input_long_edge": min(max(w, h), CAP),
            "coverage_target": COVERAGE * 100, "coverage_actual": 100.0 * info["coverage"],
            "outputs_identical": bool(np.array_equal(outputs["experiment"], outputs["black"])),
            **stats,
        })
        print(f"{size_class}: {row['split']}/{row['filename']} {w}x{h} -> "
              f"identical={results[-1]['outputs_identical']}, "
              f"{stats['mask_pct_differing']:.2f}% of mask pixels differ, "
              f"outside reach {stats['outside_max_reach_px']:.2f} px", flush=True)

    import torch
    out = {
        "description": "LaMa output with two damage fills on the same mask; differences "
                       "between the two outputs",
        "fills": {"experiment": "synthetic_damage.damage_fill: 235 + N(0, 10), rounded, "
                                "clipped, seed 0",
                  "black": "(0, 0, 0) on every masked pixel"},
        "mask": f"synthetic_damage.synthesize_mask, coverage {COVERAGE * 100:g}%, seed {SEED}",
        "lama": {"device": lama.device, "max_dimension": lama.max_dimension},
        "regions": {"U": f"feather_mask(M, r={U_RADIUS}) == 0 (the experiment's U)",
                    "r3_feather": f"feather_mask(M, r={FEATHER_RADIUS}) > 0"},
        "selection": {"above_cap": "representative image in docs/figures/manifest.json",
                      "at_or_below_cap": "Professional image with the largest long edge "
                                         f"<= {CAP} px, ties by filename"},
        "results": results,
        "provenance": {"git_head": git("rev-parse", "HEAD"),
                       "git_dirty_tracked": bool(git("status", "--porcelain",
                                                     "--untracked-files=no")),
                       "torch": torch.__version__, "python": platform.python_version(),
                       "data_manifest_sha256": sha256_file(MANIFEST)},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
