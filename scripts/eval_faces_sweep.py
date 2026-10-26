"""
Face follow-up: does the CodeFormer result depend on damage severity and on
the fidelity setting?

Images: the CLIC 2020 images whose largest detected face (results/faces/
meta.json, the app's Diagnose detector at its defaults) is at least 64 px on
its larger side. The detections are re-run here and must match meta.json.
Crop: the same rule as scripts/eval_faces.py (square, twice the face's larger
side, centred on the face, LANCZOS to 512x512).

Damage levels, both through benchmark.run_benchmark with seed 0:
  mild    the benchmark defaults: blur 1.0, 2x downscale, noise 8, JPEG 60
  severe  blur 2.0, 4x downscale, noise 15, JPEG 30
(run_benchmark applies blur, then downscale, then noise, then JPEG, and asks
restore() for outscale = the downscale factor.)

Outputs per image and level: bicubic (the benchmark baseline), Real-ESRGAN only
(do_face=False) and Real-ESRGAN + CodeFormer (do_face=True) at fidelity 0.3,
0.7 and 1.0. The degraded input and the bicubic baseline must be identical
across the four restore calls.

Scores, all on the face bounding box (the detected box mapped into the crop,
rounded outwards):
  psnr_y, ssim_y, lpips  metrics.IQAScorer.score_reference vs the clean crop
  musiq                  metrics.IQAScorer._score_one (no reference)
  identity_cos           cosine similarity of ArcFace embeddings, output vs
                         clean (see IDENTITY below)
The clean face box's MUSIQ is recorded as a reference row.

IDENTITY: facexlib ships an ArcFace IR-SE50 recognition model
(facexlib.recognition). Its init_recognition_model() moves the network to
'cuda' unconditionally, so it cannot run on this CPU-only machine; this script
builds the same network (Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se'),
the arguments init_recognition_model uses) and loads the same official weights
on CPU. Faces are aligned once, from the 5 landmarks the app's RetinaFace
detector finds on the clean crop, to the standard 112x112 ArcFace template
with a similarity transform; the same transform is applied to every output, so
only the content differs. Input is RGB scaled to [-1, 1].

The mild level must reproduce results/faces/raw.csv exactly for the six
images it shares (bicubic, Real-ESRGAN only, CodeFormer at 0.7).

Writes results/faces_sweep/raw.csv, summary.csv and meta.json. Usage:
  ./venv/bin/python scripts/eval_faces_sweep.py run
  ./venv/bin/python scripts/eval_faces_sweep.py summarise
"""

import argparse
import csv
import datetime
import inspect
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (pins RESTORE_DEVICE=cpu first)
import eval_faces as ef  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_OUT = os.path.join(ec.REPO_ROOT, "results", "faces_sweep")
OUT_DIR = DEFAULT_OUT
FACES_META = os.path.join(ec.REPO_ROOT, "results", "faces", "meta.json")
FACES_RAW = os.path.join(ec.REPO_ROOT, "results", "faces", "raw.csv")
MIN_LARGER_SIDE = 64
SEED = 0
FIDELITIES = [0.3, 0.7, 1.0]
LEVELS = {
    "mild": {"downscale_factor": 2, "blur_sigma": 1.0, "noise_sigma": 8.0, "jpeg_quality": 60},
    "severe": {"downscale_factor": 4, "blur_sigma": 2.0, "noise_sigma": 15.0, "jpeg_quality": 30},
}
CONDITIONS = ["bicubic", "realesrgan"] + [f"codeformer_f{f}" for f in FIDELITIES]
REFERENCE = "realesrgan"
# (metric, lower is better)
METRICS = [("psnr_y", False), ("ssim_y", False), ("lpips", True), ("musiq", False),
           ("identity_cos", False)]
ARCFACE_URL = ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
               "recognition_arcface_ir_se50.pth")
# Standard 112x112 ArcFace landmark template (left eye, right eye, nose, left
# and right mouth corner), as used by insightface and by facexlib's ArcFace.
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
RAW_COLUMNS = ["image", "split", "image_index", "width", "height", "face_box_xyxy",
               "crop_box_xywh", "face_box_512_xyxy", "level", "condition", "fidelity",
               "steps", "codeformer_ran", "psnr_y", "ssim_y", "lpips", "musiq",
               "identity_cos", "seconds"]
SUMMARY_COLUMNS = ["section", "level", "condition", "metric", "n", "mean", "ci_low",
                   "ci_high", "min", "max", "n_better", "n_equal", "n_worse"]


def write_meta(meta):
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def select_images():
    with open(FACES_META) as f:
        per_image = json.load(f)["detection"]["per_image"]
    return [d for d in per_image if d["num_faces"]
            and max(d["largest_width"], d["largest_height"]) >= MIN_LARGER_SIDE]


class Identity:
    """ArcFace IR-SE50 (facexlib) on CPU, plus 5-point alignment."""

    def __init__(self):
        import torch
        from facexlib.recognition.arcface_arch import Backbone
        from facexlib.utils import load_file_from_url

        self.torch = torch
        self.path = load_file_from_url(url=ARCFACE_URL, model_dir="facexlib/weights",
                                       progress=False)
        self.sha256 = ec.sha256_file(self.path)
        self.net = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
        self.net.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True),
                                 strict=True)
        self.net.eval()

    def transform(self, clean, face_box_512):
        """Similarity transform to the ArcFace template from the clean crop's landmarks."""
        import diagnose as dg

        det = dg._get_detector()
        det.load()
        bgr = np.ascontiguousarray(np.asarray(clean)[:, :, ::-1])
        with self.torch.no_grad():
            dets = det._net.detect_faces(bgr, 0.9)
        if dets is None or len(dets) == 0:
            raise SystemExit("identity: no face found on the clean crop")
        dets = np.asarray(dets)
        fx1, fy1, fx2, fy2 = face_box_512

        def iou(b):
            ix = max(0.0, min(b[2], fx2) - max(b[0], fx1))
            iy = max(0.0, min(b[3], fy2) - max(b[1], fy1))
            inter = ix * iy
            return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (fx2 - fx1) * (fy2 - fy1) - inter)

        ious = [iou(b) for b in dets]
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            raise SystemExit(f"identity: best clean-crop detection overlaps the face box "
                             f"with IoU {ious[best]:.2f} < 0.5")
        lm = dets[best, 5:15].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm, ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            raise SystemExit("identity: could not estimate the alignment transform")
        return M, float(ious[best]), lm.tolist()

    def embed(self, image, M):
        face = cv2.warpAffine(np.asarray(image), M, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t = self.torch.from_numpy(face.astype(np.float32) / 255.0)
        t = ((t - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0)
        with self.torch.no_grad():
            return self.net(t)[0].numpy()


def run(limit=None) -> int:
    import diagnose as dg
    from benchmark import run_benchmark
    from metrics import get_scorer
    from restorers.face import get_face_restorer
    from restorers.upscale import get_upscaler

    os.makedirs(OUT_DIR, exist_ok=True)
    defaults = {k: v.default for k, v in inspect.signature(run_benchmark).parameters.items()
                if v.default is not inspect.Parameter.empty}
    if any(defaults[k] != v for k, v in LEVELS["mild"].items()):
        raise SystemExit(f"mild level {LEVELS['mild']} differs from run_benchmark defaults")
    if defaults["seed"] != SEED or defaults["fidelity"] != 0.7:
        raise SystemExit(f"unexpected run_benchmark defaults {defaults}")

    chosen = select_images()
    with open(FACES_META) as f:
        before = json.load(f)["detection"]["qualifying"]
    names = sorted(f"{d['split']}/{d['image']}" for d in chosen)
    if names != sorted(before + ["mobile/IMG_1696.png"]):
        raise SystemExit(f"selection {names} is not the six earlier images plus IMG_1696.png")
    if limit is not None:
        chosen = chosen[:limit]

    identity = Identity()
    scorer = get_scorer()
    det = dg._get_detector()
    rows = ec.manifest_rows()
    meta = {
        "evaluation": "faces_sweep",
        "description": "CodeFormer at fidelity 0.3, 0.7 and 1.0 vs Real-ESRGAN only vs "
                       "bicubic, at two damage levels, on the face box of the CLIC 2020 "
                       "images with a face at least 64 px on its larger side",
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fidelity_direction": {
            "source": "restorers/face.py module docstring and _restore_crop()",
            "code": "self._raw(t, weight=float(fidelity)): the fidelity value is passed "
                    "unchanged as CodeFormer's weight w",
            "docstring": "0.0 leans on the generative prior for maximum quality (can subtly "
                         "alter identity), 1.0 stays faithful to the input (but preserves "
                         "more of its degradation). ~0.7 is a good default for old photos.",
            "reading": "1.0 = most faithful to the (degraded) input; 0.0 = most generative",
        },
        "selection": {
            "rule": f"largest face in results/faces/meta.json with max(width, height) >= "
                    f"{MIN_LARGER_SIDE} px",
            "images": names,
            "detections_rechecked": "Diagnose's detector is re-run on every full image and "
                                    "its largest box must equal meta.json",
        },
        "parameters": {
            "crop": "scripts/eval_faces.py square_crop + map_box: square of side 2 x the "
                    "larger side of the largest face, centred on it, LANCZOS to 512x512",
            "levels": LEVELS,
            "level_order": "run_benchmark/degrade: blur, then downscale (cv2 INTER_AREA), "
                           "then Gaussian noise (seed), then JPEG; restore outscale = "
                           "downscale factor",
            "seed": SEED,
            "fidelities": FIDELITIES,
            "run_benchmark_defaults": defaults,
            "calls": "run_benchmark(clean, **level, do_face=False, seed=0, score=False) and "
                     "run_benchmark(clean, **level, do_face=True, fidelity=f, seed=0, "
                     "score=False) for f in fidelities",
            "scoring": {"psnr_y, ssim_y, lpips": "IQAScorer.score_reference(out face box, "
                                                 "clean face box)",
                        "musiq": "IQAScorer._score_one('musiq', out face box), pyiqa default",
                        "identity_cos": "dot product of L2-normalised ArcFace embeddings, "
                                        "output vs clean, aligned with the clean-crop "
                                        "landmarks"},
            "identity_model": {
                "available_in_facexlib": True,
                "module": "facexlib.recognition (init_recognition_model('arcface'))",
                "architecture": "Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se'), "
                                "built directly because init_recognition_model hard-codes "
                                ".to('cuda')",
                "weights_url": ARCFACE_URL,
                "weights_source": "official facexlib release v0.1.0, downloaded with "
                                  "facexlib.utils.load_file_from_url to facexlib's default "
                                  "weights folder",
                "weights_path": os.path.relpath(identity.path, ec.REPO_ROOT),
                "weights_sha256": identity.sha256,
                "alignment": "RetinaFace (Diagnose's detector, conf 0.9) 5 landmarks on the "
                             "clean 512x512 crop, detection with the highest IoU against the "
                             "face box; cv2.estimateAffinePartial2D (LMEDS) to the 112x112 "
                             "ArcFace template; the same warp for every output",
                "input": "RGB, (x/255 - 0.5) / 0.5",
            },
        },
    }
    write_meta(meta)

    writer = ec.RowWriter(os.path.join(OUT_DIR, "raw.csv"), RAW_COLUMNS)
    clean_embeds = {}
    alignment = {}
    for d in chosen:
        row = rows[d["image_index"]]
        img = ec.load_image(row)
        w, h = img.size
        boxes = det.detect(img)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = [float(v) for v in boxes[int(np.argmax(areas))][:4]]
        if box != d["largest_box_xyxy"]:
            raise SystemExit(f"{row['filename']}: face box {box} differs from meta.json "
                             f"{d['largest_box_xyxy']}")
        cbox, shift, clamped = ef.square_crop(box, w, h)
        crop = img.crop((cbox[0], cbox[1], cbox[0] + cbox[2], cbox[1] + cbox[3]))
        clean = crop.resize((ef.CROP, ef.CROP), Image.LANCZOS)
        fb = ef.map_box(box, cbox)
        M, iou, lm = identity.transform(clean, fb)
        e_clean = identity.embed(clean, M)
        key = f"{row['split']}/{row['filename']}"
        clean_embeds[key] = e_clean
        alignment[key] = {"clean_detection_iou": iou, "landmarks_512": lm,
                          "crop_shift_xy": list(shift), "crop_side_clamped": clamped}
        base = {"image": row["filename"], "split": row["split"],
                "image_index": d["image_index"], "width": w, "height": h,
                "face_box_xyxy": " ".join(f"{v:.2f}" for v in box),
                "crop_box_xywh": " ".join(map(str, cbox)),
                "face_box_512_xyxy": " ".join(map(str, fb))}
        ref_face = clean.crop(fb)
        clean_musiq = scorer._score_one("musiq", ref_face)
        if clean_musiq is None:
            raise SystemExit(f"{key}: MUSIQ returned None on the clean face")
        writer.write({**base, "level": "none", "condition": "clean", "fidelity": "",
                      "steps": "clean reference", "codeformer_ran": "", "psnr_y": "",
                      "ssim_y": "", "lpips": "", "musiq": repr(float(clean_musiq)),
                      "identity_cos": repr(float(np.dot(e_clean, e_clean))), "seconds": ""})
        for level, params in LEVELS.items():
            t0 = time.perf_counter()
            r0 = run_benchmark(clean, **params, do_face=False, seed=SEED, score=False)
            outs = {"bicubic": (r0.baseline, "bicubic baseline", ""),
                    "realesrgan": (r0.restored, r0.steps, "")}
            for fid in FIDELITIES:
                rf = run_benchmark(clean, **params, do_face=True, fidelity=fid, seed=SEED,
                                   score=False)
                if np.asarray(rf.degraded).tobytes() != np.asarray(r0.degraded).tobytes() or \
                        np.asarray(rf.baseline).tobytes() != np.asarray(r0.baseline).tobytes():
                    raise SystemExit(f"{key} {level}: degraded input differs between calls")
                outs[f"codeformer_f{fid}"] = (rf.restored, rf.steps, fid)
            for cond in CONDITIONS:
                out, steps, fid = outs[cond]
                if out.size != clean.size:
                    raise SystemExit(f"{key} {level} {cond}: output {out.size} != {clean.size}")
                s = scorer.score_reference(out.crop(fb), ref_face)
                s["musiq"] = scorer._score_one("musiq", out.crop(fb))
                s["identity_cos"] = float(np.dot(identity.embed(out, M), e_clean))
                missing = [m for m, _ in METRICS if s.get(m) is None]
                if missing:
                    raise SystemExit(f"{key} {level} {cond}: metric(s) {missing} returned None")
                writer.write({**base, "level": level, "condition": cond, "fidelity": fid,
                              "steps": steps,
                              "codeformer_ran": str("CodeFormer on" in steps).lower()
                              if cond.startswith("codeformer") else "",
                              **{m: repr(float(s[m])) for m, _ in METRICS},
                              "seconds": f"{time.perf_counter() - t0:.2f}"})
            ran = [outs[f"codeformer_f{f}"][1] for f in FIDELITIES]
            print(f"[{key}] {level}: {ran[1]} ({time.perf_counter() - t0:.1f}s)", flush=True)
    writer.close()

    # Mild must reproduce the first face evaluation for the images it shares.
    with open(FACES_RAW, newline="") as f:
        first = {(r["split"], r["image"], r["condition"]): r for r in csv.DictReader(f)}
    with open(os.path.join(OUT_DIR, "raw.csv"), newline="") as f:
        now = {(r["split"], r["image"], r["condition"]): r for r in csv.DictReader(f)
               if r["level"] == "mild"}
    mapping = {"bicubic": "bicubic", "restore_noface": "realesrgan",
               "restore_face": "codeformer_f0.7"}
    checks = []
    for (sp, im, c_old), r_old in first.items():
        r_new = now.get((sp, im, mapping[c_old]))
        if r_new is None:
            continue
        for m in ("psnr_y", "ssim_y", "lpips"):
            checks.append({"image": f"{sp}/{im}", "condition": mapping[c_old], "metric": m,
                           "faces_raw": float(r_old[m]), "sweep_raw": float(r_new[m]),
                           "equal": float(r_old[m]) == float(r_new[m])})
    bad = [c for c in checks if not c["equal"]]
    if bad:
        raise SystemExit(f"mild level does not reproduce results/faces/raw.csv: {bad[:3]}")

    pairs = []
    for a, b in itertools.combinations(sorted(clean_embeds), 2):
        pairs.append({"a": a, "b": b, "cos": float(np.dot(clean_embeds[a], clean_embeds[b]))})
    up, fr = get_upscaler(), get_face_restorer()
    devices = {"face_detector": str(det._device), "upscaler": up.device,
               "face_restorer": fr.device, "metrics": scorer.device, "identity": "cpu"}
    if any(v != "cpu" for v in devices.values()):
        raise SystemExit(f"devices {devices}; expected cpu")
    meta.update({
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": devices,
        "alignment": alignment,
        "identity_reference_different_images": pairs,
        "reproduces_results_faces": {"n_values": len(checks), "all_equal": True},
        "weights_sha256": {"codeformer": ec.sha256_file(fr.weights_path),
                           "realesrgan": ec.sha256_file(up.weights_path),
                           "arcface": identity.sha256},
        "provenance": ec.provenance([os.path.abspath(__file__),
                                     os.path.join(ec.REPO_ROOT, "scripts", "eval_faces.py"),
                                     os.path.join(ec.REPO_ROOT, "scripts", "eval_common.py")]),
    })
    write_meta(meta)
    print(f"Wrote {OUT_DIR}/raw.csv and meta.json ({len(chosen)} images; mild reproduces "
          f"{len(checks)} values of results/faces/raw.csv)")
    return 0


def summarise() -> int:
    with open(os.path.join(OUT_DIR, "raw.csv"), newline="") as f:
        raw = list(csv.DictReader(f))
    with open(os.path.join(OUT_DIR, "meta.json")) as f:
        meta = json.load(f)
    by = {}
    for r in raw:
        k = ((r["split"], r["image"]), r["level"], r["condition"])
        if k in by:
            raise SystemExit(f"duplicate row {k}")
        by[k] = r
    images = sorted({k[0] for k in by})
    if OUT_DIR == DEFAULT_OUT and [f"{s}/{i}" for s, i in images] != meta["selection"]["images"]:
        raise SystemExit("raw.csv images do not match meta.json selection")
    for i in images:
        if (i, "none", "clean") not in by:
            raise SystemExit(f"{i} lacks the clean reference row")
        for lv in LEVELS:
            for c in CONDITIONS:
                if (i, lv, c) not in by:
                    raise SystemExit(f"{i} lacks {lv} {c}")
    out = [ec.mean_row([by[(i, "none", "clean")]["musiq"] for i in images],
                       section="reference_clean", level="none", condition="clean",
                       metric="musiq")]
    for lv in LEVELS:
        for c in CONDITIONS:
            for m, _ in METRICS:
                out.append(ec.mean_row([by[(i, lv, c)][m] for i in images], section="metric",
                                       level=lv, condition=c, metric=m))
        for c in CONDITIONS:
            if c == REFERENCE:
                continue
            for m, lower in METRICS:
                out.append(ec.paired_row([by[(i, lv, c)][m] for i in images],
                                         [by[(i, lv, REFERENCE)][m] for i in images], lower,
                                         section="paired_diff", level=lv,
                                         condition=f"{c} - {REFERENCE}", metric=m))
        for f in FIDELITIES:
            c = f"codeformer_f{f}"
            n_ran = sum(by[(i, lv, c)]["codeformer_ran"] == "true" for i in images)
            out.append({"section": "codeformer_ran", "level": lv, "condition": c,
                        "n": len(images), "mean": n_ran})
    pairs = meta.get("identity_reference_different_images") or []
    if pairs:
        out.append(ec.mean_row([p["cos"] for p in pairs],
                               section="identity_reference_different_images", level="none",
                               condition="clean vs clean, different images",
                               metric="identity_cos"))
    path = os.path.join(OUT_DIR, "summary.csv")
    ec.write_csv(path, SUMMARY_COLUMNS, out)
    print(f"Wrote {path} ({len(out)} rows; {len(images)} images)")
    return 0


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--out", help="output folder (default results/faces_sweep)")
    ap.add_argument("--limit", type=int, help="first N selected images only (smoke test)")
    args = ap.parse_args()
    if args.out:
        OUT_DIR = os.path.abspath(args.out)
    if args.limit is not None and not args.out:
        raise SystemExit("--limit is for smoke tests; pass --out outside results/faces_sweep")
    return run(args.limit) if args.cmd == "run" else summarise()


if __name__ == "__main__":
    sys.exit(main())
