"""
Measure the Real-ESRGAN upscale path before and after the half-resolution fix.

Run against whatever restorers/upscale.py is checked out; ``--label`` names
that code state ("old" or "fixed") and the git SHA plus a hash of upscale.py
are recorded alongside the numbers.

  latency  Time RealESRGANUpscaler.enhance(img, outscale=2) at 256x256 and
           512x512 inputs (centre crops of the first manifest image): one
           warm-up run, then three timed runs; reports the median. The device
           comes from RESTORE_DEVICE (cpu / mps) and the device actually used
           is recorded.
  quality  First 10 manifest images, centre-cropped to 512x512, through
           run_benchmark(seed=0) with default degradations; records bicubic
           and restored full-reference scores per image.
  combine  Merge an "old" and a "fixed" quality run into
           results/sr_fix_comparison.csv (image, method, psnr_y, ssim_y,
           lpips) plus results/sr_fix_comparison_meta.json (devices, versions,
           code hashes, latency). Fails if the bicubic baseline differs
           between the two runs.

Usage (from the repo root):
  RESTORE_DEVICE=cpu ./venv/bin/python scripts/measure_sr_fix.py latency --label old --out old_lat_cpu.json
  RESTORE_DEVICE=mps ./venv/bin/python scripts/measure_sr_fix.py latency --label old --out old_lat_mps.json
  RESTORE_DEVICE=cpu ./venv/bin/python scripts/measure_sr_fix.py quality --label old --out old_q.json
  ... repeat with --label fixed after the fix, then:
  ./venv/bin/python scripts/measure_sr_fix.py combine --old old_q.json --fixed fixed_q.json \
      --latency old_lat_cpu.json old_lat_mps.json fixed_lat_cpu.json fixed_lat_mps.json
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from PIL import Image  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data", "clic2020")
MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FR_KEYS = ["psnr_y", "ssim_y", "lpips"]
LATENCY_SIZES = [256, 512]
OUTSCALE = 2
N_QUALITY_IMAGES = 10
CROP = 512


def manifest_rows():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def centre_crop(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < size or h < size:
        raise ValueError(f"{path} is {w}x{h}, smaller than a {size}px crop")
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def image_path(row) -> str:
    return os.path.join(DATA_DIR, row["split"], row["filename"])


def provenance(label: str) -> dict:
    import torch

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()

    with open(os.path.join(REPO_ROOT, "restorers", "upscale.py"), "rb") as f:
        upscale_sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "label": label,
        "git_head": git("rev-parse", "HEAD"),
        "upscale_py_dirty": bool(git("status", "--porcelain", "--", "restorers/upscale.py")),
        "upscale_py_sha256": upscale_sha,
        "restore_device_env": os.environ.get("RESTORE_DEVICE"),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_latency(label: str) -> dict:
    from restorers.upscale import get_upscaler

    up = get_upscaler()
    up.load()
    first = manifest_rows()[0]
    out = {"provenance": provenance(label), "device": up.device, "outscale": OUTSCALE,
           "source_image": f"{first['split']}/{first['filename']}", "runs": []}
    for size in LATENCY_SIZES:
        img = centre_crop(image_path(first), size)
        up.enhance(img, outscale=OUTSCALE)  # warm-up
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = up.enhance(img, outscale=OUTSCALE)
            times.append(time.perf_counter() - t0)
        out["runs"].append({
            "input": f"{size}x{size}", "output": f"{result.size[0]}x{result.size[1]}",
            "times_s": times, "median_s": statistics.median(times),
        })
        print(f"[{label}] {up.device} {size}x{size} -> {result.size}: "
              f"median {statistics.median(times):.3f}s  runs {[round(t, 3) for t in times]}")
    out["device_after"] = up.device  # catches a silent fallback during inference
    return out


def run_quality(label: str) -> dict:
    from benchmark import run_benchmark
    from restorers.upscale import get_upscaler
    from metrics import get_scorer

    rows = manifest_rows()[:N_QUALITY_IMAGES]
    out = {"provenance": provenance(label), "crop": CROP, "seed": 0,
           "benchmark_defaults": "downscale 2x, blur 1.0, noise 8.0, JPEG 60, do_face True, fidelity 0.7",
           "images": []}
    for row in rows:
        name = f"{row['split']}/{row['filename']}"
        clean = centre_crop(image_path(row), CROP)
        t0 = time.perf_counter()
        r = run_benchmark(clean, seed=0)
        dt = time.perf_counter() - t0
        for scores in (r.baseline_scores, r.restored_scores):
            missing = [k for k in FR_KEYS if scores.get(k) is None]
            if missing:
                raise RuntimeError(f"{name}: metric(s) {missing} returned None")
        out["images"].append({"image": name, "applied": r.applied, "steps": r.steps,
                              "seconds": dt, "bicubic": r.baseline_scores,
                              "restored": r.restored_scores})
        print(f"[{label}] {name}: {r.steps} ({dt:.1f}s) bicubic "
              f"{[round(r.baseline_scores[k], 4) for k in FR_KEYS]} restored "
              f"{[round(r.restored_scores[k], 4) for k in FR_KEYS]}")
    out["device"] = {"upscaler": get_upscaler().device, "metrics": get_scorer().device}
    return out


def combine(old_path: str, fixed_path: str, latency_paths) -> None:
    with open(old_path) as f:
        old = json.load(f)
    with open(fixed_path) as f:
        fixed = json.load(f)
    old_imgs = [i["image"] for i in old["images"]]
    if old_imgs != [i["image"] for i in fixed["images"]]:
        raise RuntimeError("old and fixed runs used different images")
    for o, x in zip(old["images"], fixed["images"]):
        if o["bicubic"] != x["bicubic"]:
            raise RuntimeError(f"bicubic baseline differs between runs for {o['image']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "sr_fix_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image", "method"] + FR_KEYS)
        for method, run, key in (("bicubic", old, "bicubic"),
                                 ("old_restored", old, "restored"),
                                 ("fixed_restored", fixed, "restored")):
            for img in run["images"]:
                w.writerow([img["image"], method] + [f"{img[key][k]:.6f}" for k in FR_KEYS])
    print(f"Wrote {csv_path}")

    latency = []
    for p in latency_paths or []:
        with open(p) as f:
            latency.append(json.load(f))
    meta = {
        "description": "Real-ESRGAN outscale=2 half-resolution fix: before/after measurements",
        "quality": {
            "images": old_imgs, "crop": old["crop"], "seed": old["seed"],
            "benchmark_defaults": old["benchmark_defaults"],
            "old": {"provenance": old["provenance"], "device": old["device"],
                    "steps": {i["image"]: i["steps"] for i in old["images"]}},
            "fixed": {"provenance": fixed["provenance"], "device": fixed["device"],
                      "steps": {i["image"]: i["steps"] for i in fixed["images"]}},
            "bicubic_identical_across_runs": True,
        },
        "latency": [{"label": l["provenance"]["label"], "device": l["device"],
                     "device_after": l["device_after"], "outscale": l["outscale"],
                     "source_image": l["source_image"], "runs": l["runs"],
                     "provenance": l["provenance"]} for l in latency],
    }
    meta_path = os.path.join(RESULTS_DIR, "sr_fix_comparison_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {meta_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("latency", "quality"):
        p = sub.add_parser(name)
        p.add_argument("--label", required=True, choices=["old", "fixed"])
        p.add_argument("--out", required=True)
    p = sub.add_parser("combine")
    p.add_argument("--old", required=True)
    p.add_argument("--fixed", required=True)
    p.add_argument("--latency", nargs="*")
    args = ap.parse_args()

    if args.cmd == "combine":
        combine(args.old, args.fixed, args.latency)
        return 0
    result = run_latency(args.label) if args.cmd == "latency" else run_quality(args.label)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
