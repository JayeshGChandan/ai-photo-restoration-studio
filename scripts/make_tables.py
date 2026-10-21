"""
Write the write-up's Markdown tables into docs/index.md.

Each table lives between marker comments in docs/index.md:

    <!-- BEGIN name -->
    ...replaced on every run...
    <!-- END name -->

Tables and their sources:
  headline  results/compositing/summary.csv (metric rows): mean [95% bootstrap
            CI] per size class x coverage of U pct_changed and U pct_changed_gt1
            (N1, N2, C_3), M PSNR-Y (C_3_legacy, C_3, C_3_N2fill) and off-repair
            LPIPS (C_3_legacy, C_3, C_3_N2fill, N1, N2).
  paired    summary.csv (paired_diff rows): every paired comparison, with the
            mean difference, its CI and better / equal / worse counts.
  upscaler  results/sr_fix_comparison.csv: mean, min and max per method.
  credits   docs/figures/manifest.json (images_used) with the licence name and
            URLs recorded in data/LICENSES.md.

Fails if a marker pair is missing or duplicated, or if a value it needs is
absent. Usage (from the repo root):
  ./venv/bin/python scripts/make_tables.py
"""

import csv
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_MD = os.path.join(REPO_ROOT, "docs", "index.md")
SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "compositing", "summary.csv")
SR_CSV = os.path.join(REPO_ROOT, "results", "sr_fix_comparison.csv")
FIG_MANIFEST = os.path.join(REPO_ROOT, "docs", "figures", "manifest.json")
LICENSES_MD = os.path.join(REPO_ROOT, "data", "LICENSES.md")

SIZE_CLASSES = [("above_cap", "Above cap"), ("at_or_below_cap", "At or below cap")]
COVERAGES = ["1", "5", "15"]

# (metric, conditions, row label, number format)
HEADLINE = [
    ("u_pct_changed", ["N1", "N2", "C_3"], "U changed (%)", "{:.2f}"),
    ("u_pct_changed_gt1", ["N1", "N2", "C_3"], "U changed by >1 level (%)", "{:.2f}"),
    ("m_psnr_y", ["C_3_legacy", "C_3", "C_3_N2fill"], "M PSNR-Y (dB)", "{:.2f}"),
    ("off_lpips", ["C_3_legacy", "C_3", "C_3_N2fill", "N1", "N2"], "Off-repair LPIPS",
     "{:.4f}"),
]
PAIRED_FORMAT = {"m_psnr_y": "{:+.3f}", "m_lpips": "{:+.4f}", "off_lpips": "{:+.4f}",
                 "u_mean_abs": "{:+.3f}"}
METRIC_LABEL = {"m_psnr_y": "M PSNR-Y (dB)", "m_lpips": "M LPIPS",
                "off_lpips": "Off-repair LPIPS", "u_mean_abs": "U mean abs diff"}
SR_METHODS = [("bicubic", "Bicubic"), ("old_restored", "Real-ESRGAN, old path"),
              ("fixed_restored", "Real-ESRGAN, fixed path")]
SR_METRICS = [("psnr_y", "PSNR-Y (dB)", "{:.2f}"), ("ssim_y", "SSIM-Y", "{:.4f}"),
              ("lpips", "LPIPS", "{:.4f}")]

# Licence record for each split, as written in data/LICENSES.md (checked below).
LICENCES = {
    "professional": {
        "name": "Unsplash License",
        "licence_url": "https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_professional_2020.txt",
        "archive_url": "https://downloads.compression.cc/clic2020_professional_valid.zip",
    },
}


def read_summary():
    with open(SUMMARY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def ci(fmt, mean, lo, hi):
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def headline(rows):
    metric = {(r["size_class"], r["coverage_target"], r["condition"], r["metric"]): r
              for r in rows if r["section"] == "metric"}
    sizes = {(r["size_class"], r["coverage_target"]): r for r in rows
             if r["section"] == "u_size"}
    groups = [(sc, label, cov) for sc, label in SIZE_CLASSES for cov in COVERAGES]
    head = ["Metric", "Condition"] + [f"{label}, {cov}%" for _, label, cov in groups]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---:|" * len(groups)]
    pairs = ["**Pairs (u_empty)**", ""]
    for sc, _, cov in groups:
        s = sizes[(sc, cov)]
        pairs.append(f"{s['n']} ({s['n_u_empty']})")
    lines.append("| " + " | ".join(pairs) + " |")
    for col, conds, label, fmt in HEADLINE:
        for cond in conds:
            cells = [label, cond]
            for sc, _, cov in groups:
                r = metric.get((sc, cov, cond, col))
                if r is None:
                    raise SystemExit(f"summary.csv lacks {sc} {cov}% {cond} {col}")
                if r["mean"] == "":
                    cells.append(f"n/a (n = {r['n']})")
                    continue
                text = ci(fmt, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]))
                n_pairs = int(sizes[(sc, cov)]["n"])
                if int(r["n"]) != n_pairs:
                    text += f" (n = {r['n']})"
                cells.append(text)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(rows):
    head = ["Size class", "Coverage", "Comparison (a − b)", "Metric",
            "Mean difference [95% CI]", "a better / equal / worse", "n"]
    lines = ["| " + " | ".join(head) + " |", "|---|---:|---|---|---:|---:|---:|"]
    label = dict(SIZE_CLASSES)
    order = {sc: i for i, (sc, _) in enumerate(SIZE_CLASSES)}
    prs = [r for r in rows if r["section"] == "paired_diff"]
    if not prs:
        raise SystemExit("summary.csv has no paired_diff rows")
    for r in sorted(prs, key=lambda r: (order[r["size_class"]], float(r["coverage_target"]))):
        comp = r["condition"].replace(" - ", " − ")
        if r["n"] == "0":
            diff, counts = "n/a", "n/a"
        else:
            diff = ci(PAIRED_FORMAT[r["metric"]], float(r["mean"]), float(r["ci_low"]),
                      float(r["ci_high"]))
            counts = f"{r['n_better']} / {r['n_equal']} / {r['n_worse']}"
        n = r["n"] + (f" ({r['n_u_empty']} u_empty excluded)"
                      if r.get("n_u_empty") not in (None, "", "0") else "")
        lines.append("| " + " | ".join([label[r["size_class"]], f"{r['coverage_target']}%", comp,
                                         METRIC_LABEL[r["metric"]], diff, counts, n]) + " |")
    return "\n".join(lines)


def upscaler():
    with open(SR_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    images = sorted({r["image"] for r in rows})
    if len(images) != 10 or any(not i.startswith("mobile/") for i in images):
        raise SystemExit(f"{SR_CSV}: expected 10 Mobile crops, found {images}")
    head = ["Method (10 Mobile crops, 512×512)"]
    for _, label, _ in SR_METRICS:
        head += [f"{label} mean", "min", "max"]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (3 * len(SR_METRICS))]
    for key, label in SR_METHODS:
        sel = [r for r in rows if r["method"] == key]
        if len(sel) != 10:
            raise SystemExit(f"{SR_CSV}: {key} has {len(sel)} rows, expected 10")
        cells = [label]
        for col, _, fmt in SR_METRICS:
            v = [float(r[col]) for r in sel]
            cells += [fmt.format(sum(v) / len(v)), fmt.format(min(v)), fmt.format(max(v))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def credits():
    with open(FIG_MANIFEST) as f:
        used = json.load(f)["images_used"]
    with open(LICENSES_MD) as f:
        licences_text = f.read()
    lines = ["| Image | Split | Licence | Licence source | Dataset source |",
             "|---|---|---|---|---|"]
    for img in sorted(used, key=lambda i: (i["split"], i["filename"])):
        if img["split"] != "professional":
            raise SystemExit(f"non-Professional image in figures: {img}")
        lic = LICENCES[img["split"]]
        for value in lic.values():
            if value not in licences_text:
                raise SystemExit(f"{value!r} not found in data/LICENSES.md")
        lines.append(f"| {img['filename']} | {img['split']} | {lic['name']} | "
                     f"<{lic['licence_url']}> | <{lic['archive_url']}> |")
    return "\n".join(lines)


def replace_block(text, name, body):
    pattern = re.compile(rf"(<!-- BEGIN {name} -->\n)(.*?)(<!-- END {name} -->)", re.S)
    found = pattern.findall(text)
    if len(found) != 1:
        raise SystemExit(f"docs/index.md needs exactly one '{name}' marker pair "
                         f"(found {len(found)})")
    return pattern.sub(lambda m: m.group(1) + "\n" + body + "\n\n" + m.group(3), text)


def main() -> int:
    with open(INDEX_MD) as f:
        text = f.read()
    rows = read_summary()
    for name, body in (("headline", headline(rows)), ("paired", paired(rows)),
                       ("upscaler", upscaler()), ("credits", credits())):
        text = replace_block(text, name, body)
        print(f"{name}: {body.count(chr(10)) - 1} rows")
    with open(INDEX_MD, "w") as f:
        f.write(text)
    print(f"Wrote {os.path.relpath(INDEX_MD, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
