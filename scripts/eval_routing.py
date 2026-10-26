"""
Evaluation 2: does Diagnose recommend the route its own design intends?

Every manifest image is resized to a 1024 px long edge (PIL LANCZOS; the few
images smaller than that are enlarged, and flagged), then six versions are made:

  clean    the 1024 px image
  blur     Gaussian blur, sigma 2            (degradations.gaussian_blur)
  noise    Gaussian noise, sigma 15, seed 0  (degradations.add_gaussian_noise)
  lowres   downscaled 4x and kept small      (degradations.downscale, INTER_AREA)
  soft     downscaled 4x, then back to the 1024 px size with PIL bicubic
  jpeg20   JPEG at quality 20                (degradations.jpeg_compress)

diagnose.diagnose() and diagnose.recommend() run on every version with the
app's defaults. The expected route for each condition (EXPECTED below) is
derived from diagnose.py and the README and written to meta.json before any
image is diagnosed. The route compared is the upscale decision (do_upscale,
outscale): the conditions only change blur, noise, resolution and compression,
so whether a face is present is not set by the condition. Face detections are
recorded and reported separately.

Writes results/routing/raw.csv and meta.json (run), then summary.csv
(summarise). Usage (from the repo root):
  ./venv/bin/python scripts/eval_routing.py run
  ./venv/bin/python scripts/eval_routing.py summarise
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (pins RESTORE_DEVICE=cpu first)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_OUT = os.path.join(ec.REPO_ROOT, "results", "routing")
OUT_DIR = DEFAULT_OUT
LONG_EDGE = 1024
BLUR_SIGMA = 2.0
NOISE_SIGMA = 15.0
NOISE_SEED = 0
LOWRES_FACTOR = 4
JPEG_QUALITY = 20
CONDITIONS = ["clean", "blur", "noise", "lowres", "soft", "jpeg20"]

# Route labels: the upscale part of recommend()'s summary.
ROUTE_NONE, ROUTE_X2, ROUTE_X4 = "no upscale", "Real-ESRGAN x2", "Real-ESRGAN x4"

# Written down from diagnose.py (thresholds: sharpness < 120 -> blurry, noise
# sigma > 6.0 -> noisy, long edge < 900 -> low-res; x4 only when low-res and the
# long edge is < 500, else x2) and the README ("global decay -> Real-ESRGAN",
# "Low-res / soft / noisy -> Real-ESRGAN x2/x4") before any image was diagnosed.
EXPECTED = {
    "clean": {"route": ROUTE_NONE, "flag": None,
              "why": "a sharp, low-noise 1024 px photo is above the 900 px low-res edge "
                     "and should trip no flag, so recommend() returns no upscale "
                     "('no automatic restoration needed' when no face is found)"},
    "blur": {"route": ROUTE_X2, "flag": "is_blurry",
             "why": "Gaussian blur lowers the variance of the Laplacian below 120 -> "
                    "softness -> Real-ESRGAN; 1024 px is not low-res, so x2"},
    "noise": {"route": ROUTE_X2, "flag": "is_noisy",
              "why": "sigma 15 noise raises the Immerkaer estimate above 6.0 -> noise -> "
                     "Real-ESRGAN x2"},
    "lowres": {"route": ROUTE_X4, "flag": "is_low_res",
               "why": "long edge 256 px < 900 -> low resolution, and < 500 -> x4"},
    "soft": {"route": ROUTE_X2, "flag": "is_blurry",
             "why": "4x down and bicubic up removes fine detail at 1024 px -> softness -> "
                    "Real-ESRGAN x2"},
    "jpeg20": {"route": ROUTE_X2, "flag": None,
               "why": "README routes global decay to Real-ESRGAN and restorers/upscale.py "
                      "lists JPEG-compressed images among its targets, so the design intent "
                      "is Real-ESRGAN x2. diagnose.py has no compression measure: this "
                      "route is taken only if JPEG artefacts move the sharpness or noise "
                      "estimate past a threshold"},
}
CODE_README_DISAGREEMENTS = [
    "README diagram: Diagnose asks 'localized defect?' and routes 'localized -> LaMa "
    "(brush)'. diagnose.py measures no localized defect; when no flag and no face is "
    "found, recommend() only adds a reason suggesting the Brush Repair tab.",
    "README routes 'global decay' to Real-ESRGAN, and restorers/upscale.py lists "
    "JPEG-compressed images among its targets, but diagnose.py has no compression or "
    "blocking measure; its only signals are sharpness, noise, resolution and faces.",
    "diagnose.py says its thresholds are 'tuned for typical consumer / scanned old "
    "photos'; neither the code nor the README records what they were tuned on.",
]

RAW_COLUMNS = ["image", "split", "image_index", "orig_width", "orig_height",
               "enlarged_to_1024", "condition", "width", "height", "sharpness", "noise_sigma",
               "num_faces", "face_area_pct", "is_low_res", "is_blurry", "is_noisy",
               "do_face", "do_upscale", "outscale", "fidelity", "summary", "route",
               "expected_route", "as_expected", "seconds"]
SUMMARY_COLUMNS = ["section", "condition", "recommendation", "n", "count", "share", "mean",
                   "ci_low", "ci_high", "min", "max"]


def base_image(img):
    w, h = img.size
    s = LONG_EDGE / max(w, h)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return img.resize(size, Image.LANCZOS), s > 1


def versions(base):
    from degradations import add_gaussian_noise, downscale, gaussian_blur, jpeg_compress

    arr = np.asarray(base)
    small = downscale(arr, LOWRES_FACTOR)
    return {
        "clean": base,
        "blur": Image.fromarray(gaussian_blur(arr, BLUR_SIGMA)),
        "noise": Image.fromarray(add_gaussian_noise(arr, NOISE_SIGMA, seed=NOISE_SEED)),
        "lowres": Image.fromarray(small),
        "soft": Image.fromarray(small).resize(base.size, Image.BICUBIC),
        "jpeg20": Image.fromarray(jpeg_compress(arr, JPEG_QUALITY)),
    }


def route_of(rec):
    if not rec.do_upscale:
        return ROUTE_NONE
    return f"Real-ESRGAN x{rec.outscale}"


def write_meta(meta):
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def run(limit=None) -> int:
    import diagnose as dg

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "evaluation": "routing",
        "description": "Diagnose + recommend on six synthetic versions of each CLIC 2020 "
                       "manifest image, against the route the design intends",
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "expected_mapping": EXPECTED,
        "expected_mapping_written": "before any image was diagnosed (this file is first "
                                    "written with it, then completed after the run)",
        "route_compared": "upscale decision of recommend(): 'no upscale', "
                          "'Real-ESRGAN x2' or 'Real-ESRGAN x4'; faces reported separately",
        "code_readme_disagreements": CODE_README_DISAGREEMENTS,
        "parameters": {
            "base": f"PIL LANCZOS resize to a {LONG_EDGE} px long edge (images smaller than "
                    "that are enlarged; flagged enlarged_to_1024)",
            "blur": f"degradations.gaussian_blur(sigma={BLUR_SIGMA}) (cv2.GaussianBlur, "
                    "kernel from sigma)",
            "noise": f"degradations.add_gaussian_noise(sigma={NOISE_SIGMA}, seed={NOISE_SEED})",
            "lowres": f"degradations.downscale(factor={LOWRES_FACTOR}) (cv2 INTER_AREA), "
                      "kept small",
            "soft": f"degradations.downscale(factor={LOWRES_FACTOR}), then PIL BICUBIC back "
                    "to the base size",
            "jpeg20": f"degradations.jpeg_compress(quality={JPEG_QUALITY}) (cv2.imencode)",
            "diagnose_thresholds": {"_BLUR_THRESHOLD": dg._BLUR_THRESHOLD,
                                    "_NOISE_THRESHOLD": dg._NOISE_THRESHOLD,
                                    "_LOWRES_MAX_EDGE": dg._LOWRES_MAX_EDGE},
            "face_detector": "diagnose._FaceDetector: facexlib retinaface_resnet50, "
                             "conf 0.9, full image, no resize",
            "recommend": "diagnose.recommend(diagnose(image)), app defaults",
        },
    }
    write_meta(meta)  # the expected mapping is on disk before the first diagnosis

    writer = ec.RowWriter(os.path.join(OUT_DIR, "raw.csv"), RAW_COLUMNS)
    rows = ec.manifest_rows()
    n_processed = 0
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        n_processed = index + 1
        img = ec.load_image(row)
        base, enlarged = base_image(img)
        t_img = time.perf_counter()
        for cond, im in versions(base).items():
            t0 = time.perf_counter()
            d = dg.diagnose(im)
            rec = dg.recommend(d)
            route = route_of(rec)
            writer.write({
                "image": row["filename"], "split": row["split"], "image_index": index,
                "orig_width": img.size[0], "orig_height": img.size[1],
                "enlarged_to_1024": str(enlarged).lower(), "condition": cond,
                "width": d.width, "height": d.height, "sharpness": repr(d.sharpness),
                "noise_sigma": repr(d.noise_sigma), "num_faces": d.num_faces,
                "face_area_pct": repr(d.face_area_pct), "is_low_res": str(d.is_low_res).lower(),
                "is_blurry": str(d.is_blurry).lower(), "is_noisy": str(d.is_noisy).lower(),
                "do_face": str(rec.do_face).lower(), "do_upscale": str(rec.do_upscale).lower(),
                "outscale": rec.outscale, "fidelity": rec.fidelity, "summary": rec.summary,
                "route": route, "expected_route": EXPECTED[cond]["route"],
                "as_expected": str(route == EXPECTED[cond]["route"]).lower(),
                "seconds": f"{time.perf_counter() - t0:.2f}",
            })
        print(f"[{index + 1}/{len(rows)}] {row['split']}/{row['filename']} "
              f"({time.perf_counter() - t_img:.1f}s)", flush=True)
    writer.close()

    detector = dg._get_detector()
    meta.update({
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": {"face_detector": str(detector._device)},
        "images": {"manifest_rows": len(rows), "processed": n_processed},
        "provenance": ec.provenance([os.path.abspath(__file__),
                                     os.path.join(ec.REPO_ROOT, "scripts", "eval_common.py")]),
    })
    if str(detector._device) != "cpu":
        raise SystemExit(f"face detector ran on {detector._device}, expected cpu")
    write_meta(meta)
    print(f"Wrote {OUT_DIR}/raw.csv and meta.json ({n_processed} images)")
    return 0


def summarise() -> int:
    with open(os.path.join(OUT_DIR, "raw.csv"), newline="") as f:
        raw = list(csv.DictReader(f))
    images = sorted({(r["split"], r["image"]) for r in raw})
    if OUT_DIR == DEFAULT_OUT and len(images) != len(ec.manifest_rows()):
        raise SystemExit(f"raw.csv covers {len(images)} images, manifest has "
                         f"{len(ec.manifest_rows())}")
    by = {}
    for r in raw:
        key = ((r["split"], r["image"]), r["condition"])
        if key in by:
            raise SystemExit(f"duplicate row {key}")
        by[key] = r
    for img in images:
        for cond in CONDITIONS:
            if (img, cond) not in by:
                raise SystemExit(f"{img} lacks condition {cond}")

    out = []
    routes = [ROUTE_NONE, ROUTE_X2, ROUTE_X4]
    summaries = sorted({r["summary"] for r in raw})
    for cond in CONDITIONS:
        rs = [by[(i, cond)] for i in images]
        n = len(rs)
        for rt in routes:
            c = sum(r["route"] == rt for r in rs)
            out.append({"section": "confusion_route", "condition": cond, "recommendation": rt,
                        "n": n, "count": c, "share": c / n})
        for s in summaries:
            c = sum(r["summary"] == s for r in rs)
            out.append({"section": "confusion_summary", "condition": cond,
                        "recommendation": s, "n": n, "count": c, "share": c / n})
        ok = [1.0 if r["as_expected"] == "true" else 0.0 for r in rs]
        out.append({**ec.mean_row(ok, section="as_expected", condition=cond,
                                  recommendation=EXPECTED[cond]["route"]),
                    "count": int(sum(ok)), "share": sum(ok) / n})
        for flag in ("is_low_res", "is_blurry", "is_noisy", "do_face"):
            v = [1.0 if r[flag] == "true" else 0.0 for r in rs]
            out.append({"section": f"flag_{flag}", "condition": cond, "n": n,
                        "count": int(sum(v)), "share": sum(v) / n})
        for m in ("sharpness", "noise_sigma"):
            out.append(ec.mean_row([r[m] for r in rs], section=f"measure_{m}", condition=cond))
        faces = [float(r["num_faces"]) for r in rs]
        out.append({"section": "faces_detected", "condition": cond, "n": n,
                    "count": int(sum(f > 0 for f in faces)), "share": sum(f > 0 for f in faces) / n})

    clean = [by[(i, "clean")] for i in images]
    fp = [1.0 if r["do_upscale"] == "true" else 0.0 for r in clean]
    out.append({**ec.mean_row(fp, section="false_positive_clean_upscale", condition="clean",
                              recommendation="Real-ESRGAN recommended"),
                "count": int(sum(fp)), "share": sum(fp) / len(fp)})
    anyact = [1.0 if (r["do_upscale"] == "true" or r["do_face"] == "true") else 0.0
              for r in clean]
    out.append({"section": "clean_any_action", "condition": "clean",
                "recommendation": "face restoration or Real-ESRGAN recommended",
                "n": len(anyact), "count": int(sum(anyact)), "share": sum(anyact) / len(anyact)})
    enlarged = sorted({i for i in images if by[(i, "clean")]["enlarged_to_1024"] == "true"})
    for cond in CONDITIONS:
        rs = [by[(i, cond)] for i in images if i not in enlarged]
        c = sum(r["as_expected"] == "true" for r in rs)
        out.append({"section": "as_expected_excluding_enlarged", "condition": cond,
                    "recommendation": EXPECTED[cond]["route"], "n": len(rs), "count": c,
                    "share": c / len(rs) if rs else None})
    path = os.path.join(OUT_DIR, "summary.csv")
    ec.write_csv(path, SUMMARY_COLUMNS, out)
    print(f"Wrote {path} ({len(out)} rows; {len(images)} images, {len(enlarged)} enlarged)")
    return 0


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--out", help="output folder (default results/routing)")
    ap.add_argument("--limit", type=int, help="first N manifest rows only (smoke test)")
    args = ap.parse_args()
    if args.out:
        OUT_DIR = os.path.abspath(args.out)
    if args.limit is not None and not args.out:
        raise SystemExit("--limit is for smoke tests; pass --out outside results/routing")
    return run(args.limit) if args.cmd == "run" else summarise()


if __name__ == "__main__":
    sys.exit(main())
