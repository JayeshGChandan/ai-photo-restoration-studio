"""
Evaluation 3: face restoration (Auto Restore, CodeFormer) on CLIC 2020 faces.

Step 1, detection: the app's face detector at its default settings (the one
Diagnose uses to set do_face: diagnose._FaceDetector, facexlib
retinaface_resnet50, confidence 0.9, full image) runs on all manifest images.
An image qualifies when its largest face box (by area) is at least 64 px
across, i.e. box width x2 - x1 >= 64. Every image's detections are written to
meta.json. With fewer than MIN_FACE_IMAGES qualifying images the evaluation
stops there and records the count; no other dataset is used.

Step 2, restoration, for each qualifying image:
  * a square crop of side 2 x the larger side of the largest face, centred on
    that face's box centre; a crop that would leave the image is shifted inside
    it (the shift is recorded), and a side longer than the image's short edge is
    clamped to it (recorded); the crop is resized to 512x512 with PIL LANCZOS;
  * benchmark.run_benchmark(crop, seed=0) with the default degradations, once
    with do_face=False and once with do_face=True (default fidelity 0.7); the
    bicubic baseline and the degraded input must be identical across the two;
  * three outputs, bicubic, restore_noface and restore_face, are scored with the
    benchmark's PSNR-Y, SSIM-Y and LPIPS on the face bounding box only (the
    detected box mapped into the 512x512 crop, rounded outwards), against the
    clean crop.
No image is written; no face appears in any figure.

Writes results/faces/raw.csv, summary.csv and meta.json. Usage (from the repo root):
  ./venv/bin/python scripts/eval_faces.py run
  ./venv/bin/python scripts/eval_faces.py summarise
"""

import argparse
import csv
import datetime
import inspect
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (pins RESTORE_DEVICE=cpu first)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_OUT = os.path.join(ec.REPO_ROOT, "results", "faces")
OUT_DIR = DEFAULT_OUT
MIN_FACE_PX = 64
MIN_FACE_IMAGES = 5
CROP_FACTOR = 2
CROP = 512
SEED = 0
CONDITIONS = ["bicubic", "restore_noface", "restore_face"]
METRICS = [("psnr_y", False), ("ssim_y", False), ("lpips", True)]
COMPARISONS = [("restore_face", "bicubic"), ("restore_noface", "bicubic"),
               ("restore_face", "restore_noface")]
RAW_COLUMNS = ["image", "split", "image_index", "width", "height", "face_box_xyxy",
               "face_score", "crop_box_xywh", "crop_shift_xy", "crop_side_clamped",
               "face_box_512_xyxy", "condition", "steps", "psnr_y", "ssim_y", "lpips",
               "seconds"]
SUMMARY_COLUMNS = ["section", "condition", "metric", "n", "mean", "ci_low", "ci_high", "min",
                   "max", "n_better", "n_equal", "n_worse"]


def write_meta(meta):
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def square_crop(box, w, h):
    """Square of side CROP_FACTOR x the face's larger side, centred on the box."""
    x1, y1, x2, y2 = box
    side = CROP_FACTOR * max(x2 - x1, y2 - y1)
    clamped = side > min(w, h)
    side = int(round(min(side, min(w, h))))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    sl, st = min(max(left, 0), w - side), min(max(top, 0), h - side)
    return (sl, st, side, side), (sl - left, st - top), clamped


def map_box(box, crop_box):
    """Face box in 512x512 crop coordinates, rounded outwards and clipped."""
    x0, y0, side, _ = crop_box
    s = CROP / side
    x1, y1, x2, y2 = box
    return (max(0, math.floor((x1 - x0) * s)), max(0, math.floor((y1 - y0) * s)),
            min(CROP, math.ceil((x2 - x0) * s)), min(CROP, math.ceil((y2 - y0) * s)))


def run(limit=None) -> int:
    import diagnose as dg

    os.makedirs(OUT_DIR, exist_ok=True)
    det = dg._get_detector()
    conf_default = inspect.signature(dg._FaceDetector.detect).parameters["conf"].default
    meta = {
        "evaluation": "faces",
        "description": "CodeFormer face restoration vs Real-ESRGAN only vs bicubic, scored "
                       "on the face box of CLIC 2020 images that contain a face",
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "parameters": {
            "detector": "diagnose._FaceDetector.detect (facexlib retinaface_resnet50) on the "
                        "full-resolution clean image, the detector Diagnose uses for do_face",
            "detector_conf": conf_default,
            "qualifying_rule": f"largest face box (by area) with width x2 - x1 >= {MIN_FACE_PX} px",
            "min_face_images": MIN_FACE_IMAGES,
            "crop": f"square, side {CROP_FACTOR} x the larger side of the largest face, "
                    "centred on its box centre; shifted inside the image if needed; side "
                    "clamped to the image's short edge if longer; PIL LANCZOS to "
                    f"{CROP}x{CROP}",
            "benchmark": f"run_benchmark(crop, do_face=False|True, seed={SEED}, score=False) "
                         "with its default degradations and fidelity",
            "scoring": "metrics.IQAScorer.score_reference on the face box in crop "
                       "coordinates (rounded outwards), output vs clean crop",
            "codeformer_detection_note": "CodeFormer's own face helper detects again on the "
                                         "degraded input (FaceRestoreHelper, resize 640, "
                                         "conf 0.97); 'steps' records whether it found a face",
        },
    }
    rows = ec.manifest_rows()
    detections = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        img = ec.load_image(row)
        t0 = time.perf_counter()
        boxes = det.detect(img)
        entry = {"image": row["filename"], "split": row["split"], "image_index": index,
                 "width": img.size[0], "height": img.size[1], "num_faces": int(len(boxes)),
                 "boxes_xyxy_score": [[float(v) for v in b] for b in boxes]}
        if len(boxes):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            big = boxes[int(np.argmax(areas))]
            entry["largest_box_xyxy"] = [float(v) for v in big[:4]]
            entry["largest_width"] = float(big[2] - big[0])
            entry["largest_height"] = float(big[3] - big[1])
            entry["qualifies"] = bool(big[2] - big[0] >= MIN_FACE_PX)
        else:
            entry["qualifies"] = False
        detections.append(entry)
        print(f"[detect {index + 1}/{len(rows)}] {row['split']}/{row['filename']}: "
              f"{len(boxes)} face(s)"
              + (f", largest {entry['largest_width']:.0f}x{entry['largest_height']:.0f} px"
                 if len(boxes) else "") + f" ({time.perf_counter() - t0:.1f}s)", flush=True)
    if str(det._device) != "cpu":
        raise SystemExit(f"face detector ran on {det._device}, expected cpu")

    qualifying = [d for d in detections if d["qualifies"]]
    larger_side = sum(1 for d in detections if d["num_faces"]
                      and max(d["largest_width"], d["largest_height"]) >= MIN_FACE_PX)
    meta["detection"] = {
        "images_scanned": len(detections),
        "images_with_any_face": sum(1 for d in detections if d["num_faces"]),
        "images_qualifying": len(qualifying),
        "images_qualifying_if_larger_side_used": larger_side,
        "qualifying": [f"{d['split']}/{d['image']}" for d in qualifying],
        "per_image": detections,
    }
    meta["device"] = {"face_detector": str(det._device)}
    if len(qualifying) < MIN_FACE_IMAGES:
        meta["stopped"] = (f"only {len(qualifying)} image(s) contain a face at least "
                           f"{MIN_FACE_PX} px across (< {MIN_FACE_IMAGES}); evaluation stopped "
                           "after detection, as specified")
        meta["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        meta["provenance"] = ec.provenance([os.path.abspath(__file__),
                                            os.path.join(ec.REPO_ROOT, "scripts",
                                                         "eval_common.py")])
        write_meta(meta)
        print(meta["stopped"])
        return 0
    write_meta(meta)  # detections are on disk before any restoration

    from benchmark import run_benchmark
    from metrics import get_scorer
    from restorers.face import get_face_restorer
    from restorers.upscale import get_upscaler

    scorer = get_scorer()
    writer = ec.RowWriter(os.path.join(OUT_DIR, "raw.csv"), RAW_COLUMNS)
    for d in qualifying:
        row = rows[d["image_index"]]
        img = ec.load_image(row)
        w, h = img.size
        box = d["largest_box_xyxy"]
        score = next(b[4] for b in d["boxes_xyxy_score"] if b[:4] == box)
        cbox, shift, clamped = square_crop(box, w, h)
        crop = img.crop((cbox[0], cbox[1], cbox[0] + cbox[2], cbox[1] + cbox[3]))
        clean = crop.resize((CROP, CROP), Image.LANCZOS)
        fb = map_box(box, cbox)
        if fb[2] - fb[0] < 7 or fb[3] - fb[1] < 7:
            raise SystemExit(f"{row['filename']}: face box {fb} too small to score")
        t0 = time.perf_counter()
        r0 = run_benchmark(clean, do_face=False, seed=SEED, score=False)
        r1 = run_benchmark(clean, do_face=True, seed=SEED, score=False)
        if np.asarray(r0.degraded).tobytes() != np.asarray(r1.degraded).tobytes() or \
                np.asarray(r0.baseline).tobytes() != np.asarray(r1.baseline).tobytes():
            raise SystemExit(f"{row['filename']}: degraded input differs between runs")
        outs = {"bicubic": (r0.baseline, "bicubic baseline"),
                "restore_noface": (r0.restored, r0.steps),
                "restore_face": (r1.restored, r1.steps)}
        ref = clean.crop(fb)
        for cond, (out, steps) in outs.items():
            if out.size != clean.size:
                raise SystemExit(f"{row['filename']} {cond}: output {out.size} != {clean.size}")
            s = scorer.score_reference(out.crop(fb), ref)
            missing = [m for m, _ in METRICS if s.get(m) is None]
            if missing:
                raise SystemExit(f"{row['filename']} {cond}: metric(s) {missing} returned None")
            writer.write({"image": row["filename"], "split": row["split"],
                          "image_index": d["image_index"], "width": w, "height": h,
                          "face_box_xyxy": " ".join(f"{v:.2f}" for v in box),
                          "face_score": f"{score:.4f}",
                          "crop_box_xywh": " ".join(map(str, cbox)),
                          "crop_shift_xy": f"{shift[0]} {shift[1]}",
                          "crop_side_clamped": str(clamped).lower(),
                          "face_box_512_xyxy": " ".join(map(str, fb)), "condition": cond,
                          "steps": steps, **{m: repr(float(s[m])) for m, _ in METRICS},
                          "seconds": f"{time.perf_counter() - t0:.2f}"})
        print(f"[restore] {row['split']}/{row['filename']}: {r1.steps} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    writer.close()

    up, fr = get_upscaler(), get_face_restorer()
    devices = {"face_detector": str(det._device), "upscaler": up.device,
               "face_restorer": fr.device, "metrics": scorer.device}
    if any(v != "cpu" for v in devices.values()):
        raise SystemExit(f"devices {devices}; expected cpu")
    meta["device"] = devices
    meta["parameters"]["run_benchmark_defaults"] = {
        k: v.default for k, v in inspect.signature(run_benchmark).parameters.items()
        if v.default is not inspect.Parameter.empty}
    meta["parameters"]["weights_sha256"] = {
        "codeformer": ec.sha256_file(fr.weights_path),
        "realesrgan": ec.sha256_file(up.weights_path)}
    meta["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    meta["provenance"] = ec.provenance([os.path.abspath(__file__),
                                        os.path.join(ec.REPO_ROOT, "scripts", "eval_common.py")])
    write_meta(meta)
    print(f"Wrote {OUT_DIR}/raw.csv and meta.json ({len(qualifying)} face images)")
    return 0


def summarise() -> int:
    raw_path = os.path.join(OUT_DIR, "raw.csv")
    with open(os.path.join(OUT_DIR, "meta.json")) as f:
        meta = json.load(f)
    if "stopped" in meta:
        print(f"Evaluation stopped: {meta['stopped']}; no summary written")
        return 0
    with open(raw_path, newline="") as f:
        raw = list(csv.DictReader(f))
    by = {}
    for r in raw:
        key = ((r["split"], r["image"]), r["condition"])
        if key in by:
            raise SystemExit(f"duplicate row {key}")
        by[key] = r
    images = sorted({k for k, _ in by})
    expected = sorted((q.split("/")[0], q.split("/")[1]) for q in meta["detection"]["qualifying"])
    if images != expected:
        raise SystemExit("raw.csv images do not match the qualifying list in meta.json")
    for i in images:
        for c in CONDITIONS:
            if (i, c) not in by:
                raise SystemExit(f"{i} lacks {c}")
    out = []
    for c in CONDITIONS:
        for m, _ in METRICS:
            out.append(ec.mean_row([by[(i, c)][m] for i in images], section="metric",
                                   condition=c, metric=m))
    for a, b in COMPARISONS:
        for m, lower in METRICS:
            out.append(ec.paired_row([by[(i, a)][m] for i in images],
                                     [by[(i, b)][m] for i in images], lower,
                                     section="paired_diff", condition=f"{a} - {b}", metric=m))
    ran = sum(1 for i in images if by[(i, "restore_face")]["steps"].startswith("Real-ESRGAN")
              and "CodeFormer" in by[(i, "restore_face")]["steps"])
    out.append({"section": "codeformer_ran", "condition": "restore_face", "n": len(images),
                "mean": ran})
    ec.write_csv(os.path.join(OUT_DIR, "summary.csv"), SUMMARY_COLUMNS, out)
    print(f"Wrote {OUT_DIR}/summary.csv ({len(images)} images; CodeFormer ran on {ran})")
    return 0


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--out", help="output folder (default results/faces)")
    ap.add_argument("--limit", type=int, help="first N manifest rows only (smoke test)")
    args = ap.parse_args()
    if args.out:
        OUT_DIR = os.path.abspath(args.out)
    if args.limit is not None and not args.out:
        raise SystemExit("--limit is for smoke tests; pass --out outside results/faces")
    return run(args.limit) if args.cmd == "run" else summarise()


if __name__ == "__main__":
    sys.exit(main())
