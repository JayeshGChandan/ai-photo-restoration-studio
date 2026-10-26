"""
Evaluation 1: upscaling (Auto Restore, Real-ESRGAN only) on the CLIC 2020 images.

For every manifest image: take the 512x512 centre crop, run the app's own
benchmark (benchmark.run_benchmark) with its default degradations, seed 0 and
do_face=False, so restore() runs Real-ESRGAN only. Two outputs are compared
with the clean crop:

  bicubic     the benchmark's baseline: the degraded input resized to the clean
              size with PIL bicubic
  realesrgan  restore(degraded, do_face=False, do_upscale=True, outscale=2),
              resized to the clean size by the benchmark if needed

Each output gets the benchmark's full-reference scores (PSNR-Y, SSIM-Y, LPIPS)
and the app's no-reference MUSIQ and NIQE, all from metrics.get_scorer().

An image whose short edge is below 512 px has no 512x512 centre crop. It is
not dropped: it gets one row per condition with status
"excluded_short_edge_lt_512" and empty scores, and is listed in meta.json.
Any other failure (a metric returning None, a size mismatch) stops the run.

Writes results/upscale/raw.csv and meta.json (run), then summary.csv
(summarise). Usage (from the repo root):
  ./venv/bin/python scripts/eval_upscale.py run
  ./venv/bin/python scripts/eval_upscale.py summarise
"""

import argparse
import csv
import datetime
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec  # noqa: E402  (pins RESTORE_DEVICE=cpu first)

DEFAULT_OUT = os.path.join(ec.REPO_ROOT, "results", "upscale")
OUT_DIR = DEFAULT_OUT
CROP = 512
SEED = 0
CONDITIONS = ["bicubic", "realesrgan"]
# (metric, lower is better)
METRICS = [("psnr_y", False), ("ssim_y", False), ("lpips", True), ("musiq", False),
           ("niqe", True)]
SPLITS = ["all", "professional", "mobile"]
RAW_COLUMNS = ["image", "split", "image_index", "width", "height", "status", "condition",
               "crop_box_xywh", "degraded_size", "output_size", "steps"] + \
              [m for m, _ in METRICS] + ["seconds"]
SUMMARY_COLUMNS = ["section", "split", "condition", "metric", "n", "mean", "ci_low",
                   "ci_high", "min", "max", "n_better", "n_equal", "n_worse", "n_total",
                   "n_excluded"]


def centre_box(w, h):
    return ((w - CROP) // 2, (h - CROP) // 2, CROP, CROP)


def run(limit=None) -> int:
    from benchmark import run_benchmark
    from metrics import get_scorer
    from restorers.upscale import get_upscaler

    os.makedirs(OUT_DIR, exist_ok=True)
    scorer = get_scorer()
    up = get_upscaler()
    defaults = {k: v.default for k, v in inspect.signature(run_benchmark).parameters.items()
                if v.default is not inspect.Parameter.empty}
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    writer = ec.RowWriter(os.path.join(OUT_DIR, "raw.csv"), RAW_COLUMNS)
    excluded = []
    rows = ec.manifest_rows()
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        n_processed = index + 1
        name = f"{row['split']}/{row['filename']}"
        w, h = int(row["width"]), int(row["height"])
        base = {"image": row["filename"], "split": row["split"], "image_index": index,
                "width": w, "height": h}
        if min(w, h) < CROP:
            excluded.append({"image": name, "width": w, "height": h,
                             "reason": f"short edge {min(w, h)} px < {CROP} px crop"})
            for cond in CONDITIONS:
                writer.write({**base, "status": "excluded_short_edge_lt_512", "condition": cond,
                              **{c: "" for c in RAW_COLUMNS if c not in base and c not in
                                 ("status", "condition")}})
            print(f"[{index + 1}/{len(rows)}] {name}: excluded ({w}x{h})", flush=True)
            continue

        img = ec.load_image(row)
        box = centre_box(w, h)
        clean = img.crop((box[0], box[1], box[0] + CROP, box[1] + CROP))
        t0 = time.perf_counter()
        r = run_benchmark(clean, do_face=False, seed=SEED, score=True)
        outputs = {"bicubic": (r.baseline, r.baseline_scores, "bicubic baseline"),
                   "realesrgan": (r.restored, r.restored_scores, r.steps)}
        if r.steps != "Real-ESRGAN x2":
            raise SystemExit(f"{name}: unexpected pipeline steps {r.steps!r}")
        for cond, (out, fr, steps) in outputs.items():
            if out.size != clean.size:
                raise SystemExit(f"{name} {cond}: output {out.size} != clean {clean.size}")
            scores = dict(fr)
            for nr in ("musiq", "niqe"):
                scores[nr] = scorer._score_one(nr, out)
            missing = [m for m, _ in METRICS if scores.get(m) is None]
            if missing:
                raise SystemExit(f"{name} {cond}: metric(s) {missing} returned None")
            writer.write({**base, "status": "ok", "condition": cond,
                          "crop_box_xywh": " ".join(map(str, box)),
                          "degraded_size": f"{r.degraded.size[0]}x{r.degraded.size[1]}",
                          "output_size": f"{out.size[0]}x{out.size[1]}", "steps": steps,
                          **{m: repr(float(scores[m])) for m, _ in METRICS},
                          "seconds": f"{time.perf_counter() - t0:.2f}"})
        rb, re_ = r.baseline_scores, r.restored_scores
        print(f"[{index + 1}/{len(rows)}] {name}: LPIPS bicubic {rb['lpips']:.4f} -> "
              f"Real-ESRGAN {re_['lpips']:.4f}, PSNR-Y {rb['psnr_y']:.2f} -> "
              f"{re_['psnr_y']:.2f} ({time.perf_counter() - t0:.1f}s)", flush=True)
    writer.close()

    if up.device != "cpu" or scorer.device != "cpu":
        raise SystemExit(f"ran on upscaler {up.device}, metrics {scorer.device}; expected cpu")
    meta = {
        "evaluation": "upscale",
        "description": "Auto Restore upscaling path (Real-ESRGAN only) vs bicubic on 512x512 "
                       "centre crops of the CLIC 2020 manifest images",
        "started": started,
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": {"upscaler": up.device, "metrics": scorer.device},
        "images": {"manifest_rows": len(rows), "processed": n_processed,
                   "scored": n_processed - len(excluded), "excluded": excluded},
        "parameters": {
            "crop": f"{CROP}x{CROP} centre crop, box ((w-{CROP})//2, (h-{CROP})//2)",
            "seed": SEED,
            "run_benchmark_defaults": defaults,
            "run_benchmark_call": f"run_benchmark(clean, do_face=False, seed={SEED}, "
                                  "score=True)",
            "pipeline": "restore(degraded, do_face=False, fidelity=0.7, do_upscale=True, "
                        "outscale=downscale_factor) via run_benchmark",
            "bicubic": "degraded.resize(clean.size, Image.BICUBIC) (run_benchmark baseline)",
            "upscaler": {"tile": up.tile, "tile_pad": up.tile_pad,
                         "max_input_dim": up.max_input_dim, "native_scale": up.native_scale,
                         "weights": os.path.relpath(up.weights_path, ec.REPO_ROOT),
                         "weights_sha256": ec.sha256_file(up.weights_path)},
            "metrics": {"psnr_y, ssim_y, lpips": "metrics.IQAScorer.score_reference "
                                                 "(benchmark full-reference scores)",
                        "musiq, niqe": "metrics.IQAScorer._score_one on each output at "
                                       "512x512 (the app's no-reference scorer)",
                        "pyiqa_options": {"psnr_y": "psnr, test_y_channel=True, "
                                                    "color_space=ycbcr",
                                          "ssim_y": "ssim, test_y_channel=True, "
                                                    "color_space=ycbcr",
                                          "lpips": "lpips (pyiqa default)",
                                          "musiq": "musiq (pyiqa default)",
                                          "niqe": "niqe (pyiqa default)"}},
        },
        "provenance": ec.provenance([os.path.abspath(__file__),
                                     os.path.join(ec.REPO_ROOT, "scripts", "eval_common.py")]),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUT_DIR}/raw.csv and meta.json ({n_processed - len(excluded)} scored, "
          f"{len(excluded)} excluded)")
    return 0


def summarise() -> int:
    with open(os.path.join(OUT_DIR, "raw.csv"), newline="") as f:
        raw = list(csv.DictReader(f))
    images = {(r["split"], r["image"]) for r in raw}
    if OUT_DIR == DEFAULT_OUT and len(images) != len(ec.manifest_rows()):
        raise SystemExit(f"raw.csv covers {len(images)} images, manifest has "
                         f"{len(ec.manifest_rows())}")
    ok = [r for r in raw if r["status"] == "ok"]
    by = {}
    for r in ok:
        key = (r["split"], r["image"])
        if (key, r["condition"]) in by:
            raise SystemExit(f"duplicate row {key} {r['condition']}")
        by[(key, r["condition"])] = r
    scored = sorted({k for k, _ in by})
    for k in scored:
        for cond in CONDITIONS:
            if (k, cond) not in by:
                raise SystemExit(f"{k} lacks condition {cond}")

    out = []
    for split in SPLITS:
        all_imgs = sorted(i for i in images if split == "all" or i[0] == split)
        imgs = [k for k in scored if split == "all" or k[0] == split]
        out.append({"section": "images", "split": split, "n": len(imgs),
                    "n_total": len(all_imgs), "n_excluded": len(all_imgs) - len(imgs)})
        if not imgs:
            continue
        for cond in CONDITIONS:
            for m, _ in METRICS:
                out.append(ec.mean_row([by[(k, cond)][m] for k in imgs], section="metric",
                                       split=split, condition=cond, metric=m))
        for m, lower in METRICS:
            out.append(ec.paired_row([by[(k, "realesrgan")][m] for k in imgs],
                                     [by[(k, "bicubic")][m] for k in imgs], lower,
                                     section="paired_diff", split=split,
                                     condition="realesrgan - bicubic", metric=m))
    path = os.path.join(OUT_DIR, "summary.csv")
    ec.write_csv(path, SUMMARY_COLUMNS, out)
    print(f"Wrote {path} ({len(out)} rows; {len(scored)} images scored)")
    return 0


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--out", help="output folder (default results/upscale)")
    ap.add_argument("--limit", type=int, help="first N manifest rows only (smoke test)")
    args = ap.parse_args()
    if args.out:
        OUT_DIR = os.path.abspath(args.out)
    if args.limit is not None and not args.out:
        raise SystemExit("--limit is for smoke tests; pass --out outside results/upscale")
    return run(args.limit) if args.cmd == "run" else summarise()


if __name__ == "__main__":
    sys.exit(main())
