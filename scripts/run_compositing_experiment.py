"""
Compositing experiment: what does a LaMa repair change outside the repair?

For each clean image I (from the manifest) and target coverage, with damage
mask M (synthetic_damage.synthesize_mask) and D = I with M overwritten by the
seeded damage fill:

  L         = LaMa(D, M) at the default 1536 px cap (raw output, CPU)
  C_r       = composite(D, L, M, r), r in {0, 3, 15} (same L): the fixed
              composite, whose feather lies wholly outside M
  C_r_legacy = legacy_composite(D, L, M, r), r in {3, 15}: the pre-fix
              composite, whose feather is centred on M's edge (C_0 is the
              same in both versions, so it is run once)
  N1        = L itself: the naive whole-image result at the 1536 px cap
  N2        = raw LaMa output with max_dimension=512 (the old 512 px pipeline)

Regions: M = the binary mask; U = pixels where the FIXED r=15 feathered alpha
is exactly 0 (the common untouched region; asserted to be untouched by every
legacy condition too); H_r = pixels outside M where the r-feathered alpha is
> 0 (the halo), computed separately for the fixed and legacy feathers.

raw.csv gets one row per condition per (image, coverage) pair with U
difference stats and their tail (% changed by more than 1 level, 99th
percentile |diff|) for all conditions, H_r stats (C_3, C_15 and their legacy
versions; C_0's halo is checked to be empty), the mean weight of D inside M
(composites), PSNR-Y inside M (all), LPIPS of the repair spliced into I and of
everything outside M spliced into I (C_3, C_3_legacy, N1, N2), and runtimes.

The run is resumable (complete pairs already in raw.csv are skipped, each
new pair is appended and fsynced), fails loudly on any missing / non-finite
metric, and writes only CSV and JSON (no images). meta.json records code
version, environment, devices, arguments, timestamps, and the damage fill
and mask parameters for every invocation.

Example (from the repo root):
  ./venv/bin/python scripts/run_compositing_experiment.py --limit 2 --coverages 5
"""

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# legacy_compositing lives next to this script, deliberately off the app's path.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from compositing import composite, feather_mask  # noqa: E402
from legacy_compositing import legacy_composite, legacy_feather_mask  # noqa: E402
from region_metrics import (  # noqa: E402
    lpips_inside, lpips_outside, psnr_y_region, region_diff_stats, region_diff_tail,
)
from synthetic_damage import (  # noqa: E402
    FILL_SIGMA, FILL_VALUE, MASK_PARAMS, apply_damage, damage_fill, synthesize_mask,
)

# Composite conditions: name -> (feather version, radius). N1 / N2 are raw LaMa.
COMPOSITES = {
    "C_0": ("fixed", 0),
    "C_3": ("fixed", 3),
    "C_15": ("fixed", 15),
    "C_3_legacy": ("legacy", 3),
    "C_15_legacy": ("legacy", 15),
}
FEATHERS = {"fixed": (composite, feather_mask), "legacy": (legacy_composite, legacy_feather_mask)}
CONDITIONS = list(COMPOSITES) + ["N1", "N2"]
U_RADIUS = 15
HALO_CONDITIONS = {"C_3", "C_15", "C_3_legacy", "C_15_legacy"}
LPIPS_CONDITIONS = {"C_3", "C_3_legacy", "N1", "N2"}
N2_MAX_DIMENSION = 512

DIFF_KEYS = ["n_pixels", "pct_changed", "max_abs", "mean_abs", "mse"]
U_KEYS = DIFF_KEYS + ["pct_changed_gt1", "p99_abs"]
COLUMNS = (
    ["image", "split", "width", "height", "size_class", "image_index", "seed",
     "coverage_target", "coverage_actual", "mask_n_strokes", "mask_n_blobs",
     "condition", "feather", "radius"]
    + [f"u_{k}" for k in U_KEYS]
    + [f"h_{k}" for k in DIFF_KEYS]
    + ["m_n_pixels", "m_d_weight", "m_psnr_y", "m_lpips", "off_lpips",
       "runtime_s", "pair_runtime_s"]
)
METRIC_COLUMNS = [c for c in COLUMNS if c.startswith(("u_", "h_", "m_", "off_"))]


def applicable_columns(condition: str) -> set:
    """Metric columns that must hold a finite number for ``condition``."""
    cols = {f"u_{k}" for k in U_KEYS} | {"m_n_pixels", "m_psnr_y"}
    if condition in COMPOSITES:
        cols |= {"m_d_weight"}
    if condition in HALO_CONDITIONS:
        cols |= {f"h_{k}" for k in DIFF_KEYS}
    if condition == "C_0":
        cols |= {"h_n_pixels"}  # recorded as 0: a binary mask has no halo
    if condition in LPIPS_CONDITIONS:
        cols |= {"m_lpips", "off_lpips"}
    return cols


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def environment() -> dict:
    import pyiqa
    import torch

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "pyiqa": getattr(pyiqa, "__version__", "unknown"),
        "env": {k: os.environ.get(k) for k in ("LAMA_DEVICE", "RESTORE_DEVICE")},
        "pip_freeze": freeze,
    }


def select_images(manifest_path: str, patterns, limit):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for index, row in enumerate(rows):  # index in the FULL manifest keys the RNG
        name = f"{row['split']}/{row['filename']}"
        if patterns and not any(fnmatch.fnmatch(row["filename"], p) or fnmatch.fnmatch(name, p)
                                for p in patterns):
            continue
        selected.append((index, row))
    if patterns and not selected:
        raise SystemExit(f"--images {patterns} matched nothing in {manifest_path}")
    return selected[:limit] if limit else selected


def load_existing(raw_path: str, seed: int) -> set:
    """Return completed (split, image, coverage_target) keys; raise on anything odd."""
    if not os.path.exists(raw_path):
        return set()
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{raw_path}: header does not match this script's columns")
        pairs = {}
        for i, row in enumerate(reader, start=2):
            if None in row.values() or None in row:
                raise SystemExit(f"{raw_path}:{i}: malformed row (interrupted write?); "
                                 "remove the partial pair to resume")
            if int(row["seed"]) != seed:
                raise SystemExit(f"{raw_path}:{i}: seed {row['seed']} != --seed {seed}")
            key = (row["split"], row["image"], row["coverage_target"])
            pairs.setdefault(key, []).append(row["condition"])
    for key, conds in pairs.items():
        if sorted(conds) != sorted(CONDITIONS):
            raise SystemExit(f"{raw_path}: pair {key} has conditions {conds}; expected "
                             f"{CONDITIONS}. Remove the partial pair to resume.")
    return set(pairs)


def preflight(todo, done, seed) -> None:
    """Fail before any LaMa work if a pair to run would have an empty U or M.

    Masks and the fixed r=15 feather depend only on the image size, so this
    checks every pending pair in seconds, using the manifest's dimensions.
    """
    problems = []
    for index, row, cov in todo:
        if (row["split"], row["filename"], f"{cov:g}") in done:
            continue
        w, h = int(row["width"]), int(row["height"])
        mask, _ = synthesize_mask(w, h, cov / 100.0, index, seed)
        n_u = int(np.count_nonzero(
            np.asarray(feather_mask(Image.fromarray(mask, "L"), (w, h), U_RADIUS)) == 0))
        if n_u == 0 or not mask.any():
            problems.append(f"{row['split']}/{row['filename']} ({w}x{h}) at {cov:g}%: "
                            f"U has {n_u} px, M has {int(np.count_nonzero(mask))} px")
    if problems:
        raise SystemExit("Preflight failed: U (fixed alpha == 0 at r=15) or M is empty for "
                         f"{len(problems)} pair(s); U metrics would be undefined:\n  "
                         + "\n  ".join(problems))


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def validate(row: dict) -> None:
    need = applicable_columns(row["condition"])
    for col in METRIC_COLUMNS:
        v = row.get(col)
        if col in need:
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise RuntimeError(f"{row['split']}/{row['image']} cov {row['coverage_target']} "
                                   f"{row['condition']}: {col} = {v!r}")
        elif v is not None:
            raise RuntimeError(f"{row['condition']}: {col} unexpectedly set to {v!r}")


def run_pair(index, row, coverage_pct, seed, lama, lama_n2, cap):
    t_pair = time.perf_counter()
    path = os.path.join(REPO_ROOT, "data", "clic2020", row["split"], row["filename"])
    digest = sha256_file(path)
    if digest != row["sha256"]:
        raise RuntimeError(f"{path}: sha256 {digest} does not match manifest {row['sha256']}")
    clean = Image.open(path).convert("RGB")
    w, h = clean.size
    if (w, h) != (int(row["width"]), int(row["height"])):
        raise RuntimeError(f"{path}: size {w}x{h} does not match manifest")
    I = np.asarray(clean)
    coverage = coverage_pct / 100.0

    mask, info = synthesize_mask(w, h, coverage, index, seed)
    D = apply_damage(I, mask, damage_fill(h, w, index, coverage, seed))
    m_img, d_img = Image.fromarray(mask, "L"), Image.fromarray(D, "RGB")
    in_m = mask > 0

    t = time.perf_counter()
    L_img = lama.inpaint(d_img, m_img)
    t_lama = time.perf_counter() - t
    t = time.perf_counter()
    N2_img = lama_n2.inpaint(d_img, m_img)
    t_n2 = time.perf_counter() - t
    for name, im in (("L", L_img), ("N2", N2_img)):
        if im.size != (w, h):
            raise RuntimeError(f"{name} size {im.size} != {(w, h)}")

    outputs, runtimes, alphas = {}, {}, {}
    for cond, (version, r) in COMPOSITES.items():
        composite_fn, feather_fn = FEATHERS[version]
        alphas[cond] = np.asarray(feather_fn(m_img, (w, h), r))
        t = time.perf_counter()
        outputs[cond] = np.asarray(composite_fn(d_img, L_img, m_img, feather_radius=r))
        runtimes[cond] = t_lama + (time.perf_counter() - t)
    outputs["N1"], runtimes["N1"] = np.asarray(L_img), t_lama
    outputs["N2"], runtimes["N2"] = np.asarray(N2_img), t_n2

    # U is defined by the FIXED feather at r=15; it must be disjoint from M and
    # untouched by every legacy composite (legacy alpha 0 there, output == I).
    U = alphas[f"C_{U_RADIUS}"] == 0
    if (U & in_m).any():
        raise RuntimeError("U overlaps the mask")
    for cond, (version, _) in COMPOSITES.items():
        if version == "legacy":
            if (alphas[cond][U] != 0).any():
                raise RuntimeError(f"{cond}: legacy feather reaches into U")
            if not np.array_equal(outputs[cond][U], I[U]):
                raise RuntimeError(f"{cond}: legacy composite changes pixels in U")
    halos = {cond: ~in_m & (alphas[cond] > 0) for cond in COMPOSITES}
    if halos["C_0"].any():
        raise RuntimeError("r=0 feather produced a halo outside the binary mask")
    n_m = int(in_m.sum())

    base = {
        "image": row["filename"], "split": row["split"], "width": w, "height": h,
        "size_class": "above_cap" if max(w, h) > cap else "at_or_below_cap",
        "image_index": index, "seed": seed, "coverage_target": f"{coverage_pct:g}",
        "coverage_actual": 100.0 * info["coverage"],
        "mask_n_strokes": info["n_strokes"], "mask_n_blobs": info["n_blobs"],
    }
    rows = []
    for cond in CONDITIONS:
        X = outputs[cond]
        version, radius = COMPOSITES.get(cond, (None, None))
        out = dict(base, condition=cond, feather=version, radius=radius)
        out.update({col: None for col in METRIC_COLUMNS})
        out.update({f"u_{k}": v for k, v in region_diff_stats(X, I, U).items()})
        out.update({f"u_{k}": v for k, v in region_diff_tail(X, I, U).items()})
        if cond in HALO_CONDITIONS:
            out.update({f"h_{k}": v for k, v in region_diff_stats(X, I, halos[cond]).items()})
        elif cond == "C_0":
            out["h_n_pixels"] = 0
        out["m_n_pixels"] = n_m
        if cond in COMPOSITES:
            # Mean of (1 - alpha/255) over M: the weight the composite gives D
            # inside the repair. Integer sum, so 0 means exactly 0.
            out["m_d_weight"] = int((255 - alphas[cond][in_m].astype(np.int64)).sum()) / (255 * n_m)
        out["m_psnr_y"] = psnr_y_region(X, I, in_m)
        if cond in LPIPS_CONDITIONS:
            out["m_lpips"] = lpips_inside(X, I, in_m)
            out["off_lpips"] = lpips_outside(X, I, in_m)
        out["runtime_s"] = runtimes[cond]
        rows.append(out)

    pair_time = time.perf_counter() - t_pair
    for out in rows:
        out["pair_runtime_s"] = pair_time
        validate(out)
    timing = {"lama_s": t_lama, "n2_s": t_n2, "pair_s": pair_time}
    return rows, timing


def append_rows(raw_path: str, rows) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if not os.path.exists(raw_path):
        writer.writerow(COLUMNS)
    for r in rows:
        writer.writerow([fmt(r[c]) for c in COLUMNS])
    with open(raw_path, "a", newline="") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the compositing experiment (see module docstring for definitions).")
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "manifest.csv"),
                    help="CSV of images to use (default: data/manifest.csv)")
    ap.add_argument("--coverages", type=float, nargs="+", default=[1, 5, 15],
                    help="target mask coverages in percent (default: 1 5 15)")
    ap.add_argument("--seed", type=int, default=0, help="experiment seed (default: 0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N selected images")
    ap.add_argument("--images", nargs="+", default=None,
                    help="filename filters (fnmatch on 'file.png' or 'split/file.png')")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "compositing"),
                    help="output directory (default: results/compositing/)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even if tracked files have uncommitted changes")
    args = ap.parse_args()

    from restorers.inpaint import LamaInpainter
    from metrics import get_scorer

    env = environment()
    if env["git_dirty"] and not args.allow_dirty:
        raise SystemExit("Tracked files have uncommitted changes; commit them or pass "
                         "--allow-dirty (recorded in meta.json).")

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.csv")
    meta_path = os.path.join(args.out, "meta.json")
    manifest_sha = sha256_file(args.manifest)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        first = meta["runs"][0]
        for what, old, new in (("git_head", first["git_head"], env["git_head"]),
                               ("seed", first["args"]["seed"], args.seed),
                               ("manifest_sha256", meta["manifest_sha256"], manifest_sha)):
            if old != new:
                raise SystemExit(f"{args.out} was produced with {what}={old}; now {new}. "
                                 "Use a new --out directory.")
        if meta["damage_fill"]["value"] != FILL_VALUE or meta["mask_params"] != MASK_PARAMS:
            raise SystemExit(f"{args.out}: damage fill / mask parameters changed since the "
                             "first run. Use a new --out directory.")
    else:
        meta = {
            "experiment": "compositing",
            "definitions": {
                "D": "I with pixels inside M overwritten by the damage fill",
                "L": "LamaInpainter(device='cpu').inpaint(D, M), default 1536 px cap, raw",
                "C_r": "compositing.composite(D, L, M, feather_radius=r), r in {0, 3, 15}, "
                       "same L (feather wholly outside M)",
                "C_r_legacy": "legacy_compositing.legacy_composite(D, L, M, feather_radius=r), "
                              "r in {3, 15}, same L (feather centred on M's edge)",
                "N1": "L itself",
                "N2": "LamaInpainter(device='cpu', max_dimension=512).inpaint(D, M), raw",
                "U": "pixels where the fixed feather_mask(M, r=15) == 0; asserted disjoint "
                     "from M and untouched by every legacy composite",
                "H_r": "pixels outside M where the condition's own feather (fixed or "
                       "legacy) at radius r is > 0",
                "size_class": "above_cap if long edge > LaMa cap (1536) else at_or_below_cap",
                "u_pct_changed_gt1": "% of U pixels where any channel differs by > 1 level",
                "u_p99_abs": "99th percentile (inverted CDF) of per-channel |X - I| over U",
                "m_d_weight": "mean of (1 - alpha/255) over M: the weight of D inside M",
                "m_lpips": "LPIPS(X inside M + I outside M, I)",
                "off_lpips": "LPIPS(I inside M + X outside M, I)",
                "runtime_s": "time to produce the condition's output (composites include L)",
            },
            "conditions": CONDITIONS,
            "composites": {k: {"feather": v, "radius": r} for k, (v, r) in COMPOSITES.items()},
            "u_radius": U_RADIUS,
            "code": {"compositing_py_blob": git("hash-object", "compositing.py"),
                     "legacy_compositing_py_blob": git("hash-object",
                                                       "scripts/legacy_compositing.py")},
            "manifest": os.path.relpath(os.path.abspath(args.manifest), REPO_ROOT),
            "manifest_sha256": manifest_sha,
            "damage_fill": {"value": FILL_VALUE, "sigma": FILL_SIGMA, "clip": [0, 255],
                            "rounding": "nearest (np.rint)",
                            "rng": "SeedSequence([seed, image_index, round(coverage*1e6), 1])"},
            "mask_params": MASK_PARAMS,
            "mask_rng": "SeedSequence([seed, image_index, round(coverage*1e6), 0])",
            "runs": [],
        }

    # Validate existing results and the image selection before registering this run.
    done = load_existing(raw_path, args.seed)
    images = select_images(args.manifest, args.images, args.limit)
    todo = [(i, r, c) for i, r in images for c in args.coverages]
    t = time.perf_counter()
    preflight(todo, done, args.seed)
    print(f"Preflight: U and M non-empty for all pending pairs ({time.perf_counter() - t:.1f}s).",
          flush=True)

    lama = LamaInpainter(device="cpu")
    lama_n2 = LamaInpainter(device="cpu", max_dimension=N2_MAX_DIMENSION)
    lama.load()
    lama_n2.load()
    scorer = get_scorer()
    if scorer._get("lpips") is None:
        raise SystemExit("LPIPS metric failed to load")

    run = dict(env, start=now(), end=None, status="running", args=vars(args),
               devices={"lama_n1": lama.device, "lama_n2": lama_n2.device,
                        "metrics": scorer.device},
               lama={"max_dimension_n1": lama.max_dimension,
                     "max_dimension_n2": lama_n2.max_dimension},
               pairs_completed=0, pairs_skipped=0)
    meta["runs"].append(run)
    write_json(meta_path, meta)

    print(f"{len(images)} images x {len(args.coverages)} coverages = {len(todo)} pairs; "
          f"{sum((r['split'], r['filename'], f'{c:g}') in done for _, r, c in todo)} already done. "
          f"LaMa on {lama.device}, metrics on {scorer.device}. Output: {args.out}", flush=True)

    try:
        for k, (index, row, cov) in enumerate(todo, start=1):
            key = (row["split"], row["filename"], f"{cov:g}")
            if key in done:
                run["pairs_skipped"] += 1
                continue
            rows, timing = run_pair(index, row, cov, args.seed, lama, lama_n2,
                                    cap=lama.max_dimension)
            if lama.device != "cpu" or lama_n2.device != "cpu":
                raise RuntimeError(f"LaMa left CPU: {lama.device}, {lama_n2.device}")
            append_rows(raw_path, rows)
            done.add(key)
            run["pairs_completed"] += 1
            by = {r["condition"]: r for r in rows}
            print(f"[{k}/{len(todo)}] {key[0]}/{key[1]} {key[2]}% ({rows[0]['size_class']}): "
                  f"LaMa {timing['lama_s']:.1f}s, N2 {timing['n2_s']:.1f}s, pair "
                  f"{timing['pair_s']:.1f}s | U changed: N1 {by['N1']['u_pct_changed']:.2f}% "
                  f"N2 {by['N2']['u_pct_changed']:.2f}% C_3 {by['C_3']['u_pct_changed']:.2f}% | "
                  f"D weight in M: C_3 {by['C_3']['m_d_weight']:.4f} "
                  f"C_3_legacy {by['C_3_legacy']['m_d_weight']:.4f}",
                  flush=True)
            write_json(meta_path, meta)
        run["status"] = "completed"
    except BaseException as e:
        run["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        run["end"] = now()
        write_json(meta_path, meta)
    print(f"Done: {run['pairs_completed']} new pairs, {run['pairs_skipped']} skipped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
