"""
Summarise the compositing experiment's raw.csv.

Writes summary.csv and prints a table with, for each condition x size class
x target coverage, the mean across images and a percentile-bootstrap 95% CI
(10,000 resamples of images, seed 0, fresh RNG per statistic) of:

  U pct_changed, U pct_changed_gt1 (> 1 level), U mean_abs, U p99_abs,
  U max_abs, H_r pct_changed (C_3, C_15 and legacy versions), the weight of D
  inside M (composites), M PSNR-Y, M LPIPS and off-repair LPIPS (C_3,
  C_3_legacy, N1, N2).

Then paired per-image differences, each with a bootstrap CI of the mean
difference and how many images C_3 is better / equal / worse on:
  C_3 - C_3_legacy on M PSNR-Y, M LPIPS, off-repair LPIPS
  C_3 - N1 and C_3 - N2 on M PSNR-Y, M LPIPS, U mean_abs
the null case (N1's U pct_changed for images at or below the LaMa cap), and
how many images are at most 512 px on the long edge, where N2 equals N1.

Pairs whose U is empty (u_empty = true: the fixed r=15 halo covers every
non-mask pixel) have blank U metrics. They are excluded from every U statistic
and U paired difference, and kept for everything else; each U cell prints its
own n. A U-size table gives, per size class x coverage, the mean and minimum
u_fraction (U as a share of the image) and the number of u_empty pairs.

Usage (from the repo root):
  ./venv/bin/python scripts/summarise_compositing.py                      # results/compositing/
  ./venv/bin/python scripts/summarise_compositing.py --dir results/compositing_smoke
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPOSITES = ["C_0", "C_3", "C_15", "C_3_legacy", "C_15_legacy"]
CONDITIONS = COMPOSITES + ["N1", "N2"]
HALO_CONDITIONS = ["C_3", "C_15", "C_3_legacy", "C_15_legacy"]
LPIPS_CONDITIONS = ["C_3", "C_3_legacy", "N1", "N2"]
SIZE_CLASSES = ["above_cap", "at_or_below_cap"]
N2_CAP = 512  # N2's LaMa max_dimension: at or below it, N2 runs at native size like N1
N_RESAMPLES = 10_000
BOOT_SEED = 0

# (column, label, conditions it applies to, format)
METRICS = [
    ("u_pct_changed", "U % changed", CONDITIONS, "{:.3f}"),
    ("u_pct_changed_gt1", "U % changed >1", CONDITIONS, "{:.3f}"),
    ("u_mean_abs", "U mean |diff|", CONDITIONS, "{:.4f}"),
    ("u_p99_abs", "U p99 |diff|", CONDITIONS, "{:.1f}"),
    ("u_max_abs", "U max |diff|", CONDITIONS, "{:.1f}"),
    ("h_pct_changed", "H_r % changed", HALO_CONDITIONS, "{:.2f}"),
    ("m_d_weight", "M weight of D", COMPOSITES, "{:.4f}"),
    ("m_psnr_y", "M PSNR-Y (dB)", CONDITIONS, "{:.2f}"),
    ("m_lpips", "M LPIPS", LPIPS_CONDITIONS, "{:.5f}"),
    ("off_lpips", "off-repair LPIPS", LPIPS_CONDITIONS, "{:.5f}"),
]
# Paired comparisons: (a, b, [(column, lower is better), ...]); differences are a - b.
COMPARISONS = [
    ("C_3", "C_3_legacy", [("m_psnr_y", False), ("m_lpips", True), ("off_lpips", True)]),
    ("C_3", "N1", [("m_psnr_y", False), ("m_lpips", True), ("u_mean_abs", True)]),
    ("C_3", "N2", [("m_psnr_y", False), ("m_lpips", True), ("u_mean_abs", True)]),
]

SUMMARY_COLUMNS = ["section", "size_class", "coverage_target", "condition", "metric", "n",
                   "mean", "ci_low", "ci_high", "max", "min", "n_better", "n_equal", "n_worse",
                   "n_images_changed", "n_total", "n_u_empty"]


def is_u_metric(col: str) -> bool:
    return col.startswith("u_")


def u_empty(row) -> bool:
    return row["u_empty"] == "true"


def bootstrap_ci(values):
    """Mean and percentile-bootstrap 95% CI of the mean."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    means = v[rng.integers(0, len(v), size=(N_RESAMPLES, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def load(raw_path):
    """Return {(size_class, coverage, condition): {(split, image): row}}; validate fully."""
    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{raw_path} has no rows")
    groups = defaultdict(dict)
    pairs = defaultdict(set)
    u_state = {}
    for i, row in enumerate(rows, start=2):
        key = (row["size_class"], row["coverage_target"], row["condition"])
        img = (row["split"], row["image"])
        if img in groups[key]:
            raise SystemExit(f"{raw_path}:{i}: duplicate row for {img} {key}")
        if row["u_empty"] not in ("true", "false"):
            raise SystemExit(f"{raw_path}:{i}: u_empty = {row['u_empty']!r}")
        if (int(row["u_pixels"]) == 0) != u_empty(row):
            raise SystemExit(f"{raw_path}:{i}: u_empty {row['u_empty']} with "
                             f"u_pixels {row['u_pixels']}")
        state = (row["u_pixels"], row["u_fraction"], row["u_empty"])
        if u_state.setdefault((img, row["coverage_target"]), state) != state:
            raise SystemExit(f"{raw_path}:{i}: U size differs between conditions of {img}")
        for col, _, conds, _ in METRICS:
            if row["condition"] in conds:
                if is_u_metric(col) and u_empty(row):
                    if row[col] != "":
                        raise SystemExit(f"{raw_path}:{i}: {col} = {row[col]!r} with empty U")
                    continue
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    raise SystemExit(f"{raw_path}:{i}: {col} = {row[col]!r} for {row['condition']}")
                if not math.isfinite(v):
                    raise SystemExit(f"{raw_path}:{i}: {col} = {v} for {row['condition']}")
        groups[key][img] = row
        pairs[(img, row["coverage_target"])].add(row["condition"])
    for pair, conds in pairs.items():
        if conds != set(CONDITIONS):
            raise SystemExit(f"{raw_path}: incomplete pair {pair}: {sorted(conds)}")
    return groups


def coverage_order(groups):
    return sorted({k[1] for k in groups}, key=float)


def summarise(groups):
    out, cells = [], {}
    for sc in SIZE_CLASSES:
        for cov in coverage_order(groups):
            for cond in CONDITIONS:
                rows = groups.get((sc, cov, cond))
                if not rows:
                    continue
                for col, _, conds, _ in METRICS:
                    if cond not in conds:
                        continue
                    use = [r for r in rows.values() if not (is_u_metric(col) and u_empty(r))]
                    n_empty = len(rows) - len(use)
                    vals = [float(r[col]) for r in use]
                    if not vals:  # every pair in the group has an empty U
                        cells[(sc, cov, cond, col)] = (None, None, None, 0)
                        out.append({"section": "metric", "size_class": sc,
                                    "coverage_target": cov, "condition": cond, "metric": col,
                                    "n": 0, "n_u_empty": n_empty})
                        continue
                    mean, lo, hi = bootstrap_ci(vals)
                    cells[(sc, cov, cond, col)] = (mean, lo, hi, len(vals))
                    out.append({"section": "metric", "size_class": sc, "coverage_target": cov,
                                "condition": cond, "metric": col, "n": len(vals), "mean": mean,
                                "ci_low": lo, "ci_high": hi, "max": max(vals),
                                "n_u_empty": n_empty if is_u_metric(col) else None})
    return out, cells


def paired(groups):
    out = []
    for sc in SIZE_CLASSES:
        for cov in coverage_order(groups):
            for a, b, metrics in COMPARISONS:
                ra, rb = groups.get((sc, cov, a)), groups.get((sc, cov, b))
                if not ra or not rb:
                    continue
                if set(ra) != set(rb):
                    raise SystemExit(f"{sc} {cov}%: {a} and {b} cover different images")
                for col, lower_better in metrics:
                    imgs = sorted(i for i in ra if not (is_u_metric(col) and u_empty(ra[i])))
                    n_empty = len(ra) - len(imgs)
                    if not imgs:
                        out.append({"section": "paired_diff", "size_class": sc,
                                    "coverage_target": cov, "condition": f"{a} - {b}",
                                    "metric": col, "n": 0, "n_u_empty": n_empty})
                        continue
                    d = np.array([float(ra[i][col]) - float(rb[i][col]) for i in imgs])
                    better = int(np.sum(d < 0) if lower_better else np.sum(d > 0))
                    equal = int(np.sum(d == 0))
                    mean, lo, hi = bootstrap_ci(d)
                    out.append({"section": "paired_diff", "size_class": sc,
                                "coverage_target": cov, "condition": f"{a} - {b}",
                                "metric": col, "n": len(d), "mean": mean, "ci_low": lo,
                                "ci_high": hi, "n_better": better, "n_equal": equal,
                                "n_worse": len(d) - better - equal,
                                "n_u_empty": n_empty if is_u_metric(col) else None})
    return out


def null_case(groups):
    out = []
    for cov in coverage_order(groups):
        rows = groups.get(("at_or_below_cap", cov, "N1"))
        if not rows:
            continue
        vals = [float(r["u_pct_changed"]) for r in rows.values() if not u_empty(r)]
        n_empty = len(rows) - len(vals)
        if not vals:
            out.append({"section": "null_case", "size_class": "at_or_below_cap",
                        "coverage_target": cov, "condition": "N1", "metric": "u_pct_changed",
                        "n": 0, "n_u_empty": n_empty})
            continue
        mean, lo, hi = bootstrap_ci(vals)
        out.append({"section": "null_case", "size_class": "at_or_below_cap",
                    "coverage_target": cov, "condition": "N1", "metric": "u_pct_changed",
                    "n": len(vals), "mean": mean, "ci_low": lo, "ci_high": hi,
                    "max": max(vals), "n_images_changed": sum(v > 0 for v in vals),
                    "n_u_empty": n_empty})
    return out


def u_size(groups):
    """Per size class x coverage: mean / min u_fraction and u_empty pair count."""
    out = []
    for sc in SIZE_CLASSES:
        for cov in coverage_order(groups):
            rows = groups.get((sc, cov, "N1"))  # U size is pair-level; N1 has one row per pair
            if not rows:
                continue
            fr = [float(r["u_fraction"]) for r in rows.values()]
            out.append({"section": "u_size", "size_class": sc, "coverage_target": cov,
                        "metric": "u_fraction", "n": len(fr), "mean": sum(fr) / len(fr),
                        "min": min(fr), "max": max(fr),
                        "n_u_empty": sum(u_empty(r) for r in rows.values())})
    return out


def n2_equals_n1(groups):
    """Images at most N2_CAP px on the long edge, where N2 runs exactly like N1."""
    images = {}
    for rows in groups.values():
        for img, row in rows.items():
            images[img] = max(int(row["width"]), int(row["height"]))
    small = sorted(img for img, long_edge in images.items() if long_edge <= N2_CAP)
    return [{"section": "n2_equals_n1", "size_class": "at_or_below_cap",
             "condition": "N2 = N1", "metric": f"long_edge_le_{N2_CAP}",
             "n": len(small), "n_total": len(images)}], small, len(images)


def ci_text(fmt, mean, lo, hi):
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def cell_text(cell, fmt, col):
    if cell is None:
        return "-"
    mean, lo, hi, n = cell
    if not is_u_metric(col):
        return ci_text(fmt, mean, lo, hi)
    return "n/a n=0" if n == 0 else f"{ci_text(fmt, mean, lo, hi)} n={n}"


def print_tables(groups, cells, paired_rows, null_rows, small_images, n_images, size_rows):
    width = 34
    sizes = {(r["size_class"], r["coverage_target"]): r for r in size_rows}
    for sc in SIZE_CLASSES:
        for cov in coverage_order(groups):
            ns = {len(groups[(sc, cov, c)]) for c in CONDITIONS if (sc, cov, c) in groups}
            if not ns:
                continue
            u = sizes[(sc, cov)]
            print(f"\n== {sc}, coverage {cov}% (n = {', '.join(map(str, sorted(ns)))} images; "
                  f"mean u_fraction {u['mean']:.4f}, u_empty pairs {u['n_u_empty']}) "
                  "-- mean [95% bootstrap CI]; U rows show their own n")
            print("metric".ljust(18) + "".join(c.ljust(width) for c in CONDITIONS))
            for col, label, _, fmt in METRICS:
                line = label.ljust(18)
                for cond in CONDITIONS:
                    line += cell_text(cells.get((sc, cov, cond, col)), fmt, col).ljust(width)
                print(line.rstrip())

    print("\n== Paired differences (per image), mean [95% bootstrap CI]; "
          "better/equal/worse counts are for C_3")
    print("size_class".ljust(17) + "cov".ljust(5) + "comparison".ljust(16) + "metric".ljust(12)
          + "mean diff [CI]".ljust(36) + "better/equal/worse")
    fmts = {"u_mean_abs": "{:+.4f}", "m_psnr_y": "{:+.3f}", "m_lpips": "{:+.5f}",
            "off_lpips": "{:+.5f}"}
    for r in paired_rows:
        head = (r["size_class"].ljust(17) + f"{r['coverage_target']}%".ljust(5)
                + r["condition"].replace(" ", "").ljust(16) + r["metric"].ljust(12))
        excl = (f" ({r['n_u_empty']} u_empty excluded)" if r.get("n_u_empty") else "")
        if r["n"] == 0:
            print(head + "n/a".ljust(36) + f"n = 0{excl}")
            continue
        print(head + ci_text(fmts[r["metric"]], r["mean"], r["ci_low"], r["ci_high"]).ljust(36)
              + f"{r['n_better']}/{r['n_equal']}/{r['n_worse']} of {r['n']}{excl}")

    print("\n== Null case: N1 U % changed, images at or below the 1536 px cap")
    if not null_rows:
        print("(no at_or_below_cap images in raw.csv)")
    for r in null_rows:
        excl = f" ({r['n_u_empty']} u_empty excluded)" if r.get("n_u_empty") else ""
        if r["n"] == 0:
            print(f"coverage {r['coverage_target']}%: n = 0{excl}")
            continue
        print(f"coverage {r['coverage_target']}%: n = {r['n']}, mean {r['mean']:.6f}%, "
              f"max {r['max']:.6f}%, images with any U change: {r['n_images_changed']}{excl}")

    print("\n== U size: share of the image untouched at r=15 (fixed feather), per pair")
    print("size_class".ljust(17) + "coverage".ljust(10) + "pairs".ljust(7)
          + "mean u_fraction".ljust(17) + "min u_fraction".ljust(16) + "u_empty pairs")
    for r in size_rows:
        print(r["size_class"].ljust(17) + f"{r['coverage_target']}%".ljust(10)
              + str(r["n"]).ljust(7) + f"{r['mean']:.4f}".ljust(17) + f"{r['min']:.4f}".ljust(16)
              + str(r["n_u_empty"]))

    print(f"\n== N2 equals N1 (long edge <= {N2_CAP} px, so N2's cap does not apply): "
          f"{len(small_images)} of {n_images} images"
          + (f": {', '.join('/'.join(i) for i in small_images)}" if small_images else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise compositing experiment results.")
    ap.add_argument("--dir", default=os.path.join(REPO_ROOT, "results", "compositing"),
                    help="directory holding raw.csv; summary.csv is written there "
                         "(default: results/compositing/)")
    args = ap.parse_args()

    raw_path = os.path.join(args.dir, "raw.csv")
    groups = load(raw_path)
    metric_rows, cells = summarise(groups)
    paired_rows = paired(groups)
    null_rows = null_case(groups)
    n2_rows, small_images, n_images = n2_equals_n1(groups)
    size_rows = u_size(groups)

    summary_path = os.path.join(args.dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in metric_rows + paired_rows + null_rows + size_rows + n2_rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()
                        if v is not None})

    print_tables(groups, cells, paired_rows, null_rows, small_images, n_images, size_rows)
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
