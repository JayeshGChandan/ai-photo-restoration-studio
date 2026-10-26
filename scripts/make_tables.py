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
  credits   docs/figures/manifest.json: the source image of every figure whose
            file docs/index.md embeds, with the licence name and URLs recorded
            in data/LICENSES.md.
  upscale   results/upscale/summary.csv: bicubic and Real-ESRGAN mean [95% CI]
            per split and metric, the paired difference and Real-ESRGAN's
            better / equal / worse count against bicubic.
  routing   results/routing/summary.csv and meta.json: condition x recommended
            route counts, the expected route and the share routed as expected.
  faces     results/faces/summary.csv and meta.json: mean [95% CI] per method
            on the face box, or the reason the evaluation stopped.
  faces_sweep  results/faces_sweep/summary.csv: one table per damage level,
            mean [95% CI] per method and metric on the face box, and each
            method's better / equal / worse count against Real-ESRGAN only.

headline, paired, upscaler and credits must each have exactly one marker
pair. upscale, routing, faces and faces_sweep are written only when their markers are
present (one pair each); without markers they are skipped. Fails if a value a
table needs is absent. Usage (from the repo root):
  ./venv/bin/python scripts/make_tables.py
  ./venv/bin/python scripts/make_tables.py --print upscale   # show, write nothing
"""

import argparse
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
UPSCALE_SUMMARY = os.path.join(REPO_ROOT, "results", "upscale", "summary.csv")
ROUTING_SUMMARY = os.path.join(REPO_ROOT, "results", "routing", "summary.csv")
ROUTING_META = os.path.join(REPO_ROOT, "results", "routing", "meta.json")
FACES_SUMMARY = os.path.join(REPO_ROOT, "results", "faces", "summary.csv")
FACES_META = os.path.join(REPO_ROOT, "results", "faces", "meta.json")
SWEEP_SUMMARY = os.path.join(REPO_ROOT, "results", "faces_sweep", "summary.csv")

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


def credits(index_text):
    with open(FIG_MANIFEST) as f:
        manifest = json.load(f)
    with open(LICENSES_MD) as f:
        licences_text = f.read()
    used = manifest["images_used"]
    # Credit the source image of each figure the page actually embeds.
    shown = []
    for fig in manifest["outcomes"]["figures"].values():
        if fig.get("source") and any(f"figures/{name}" in index_text for name in fig["files"]):
            split, filename = fig["source"].split("/", 1)
            entry = {"split": split, "filename": filename}
            if entry not in used:
                raise SystemExit(f"{fig['source']} is a figure source but not in images_used")
            if entry not in shown:
                shown.append(entry)
    if not shown:
        raise SystemExit("docs/index.md embeds no figure with a source image")
    lines = ["| Image | Split | Licence | Licence source | Dataset source |",
             "|---|---|---|---|---|"]
    for img in sorted(shown, key=lambda i: (i["split"], i["filename"])):
        if img["split"] != "professional":
            raise SystemExit(f"non-Professional image in figures: {img}")
        lic = LICENCES[img["split"]]
        for value in lic.values():
            if value not in licences_text:
                raise SystemExit(f"{value!r} not found in data/LICENSES.md")
        lines.append(f"| {img['filename']} | {img['split']} | {lic['name']} | "
                     f"<{lic['licence_url']}> | <{lic['archive_url']}> |")
    return "\n".join(lines)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


UP_SPLITS = [("all", "All"), ("professional", "Professional"), ("mobile", "Mobile")]
UP_METRICS = [("psnr_y", "PSNR-Y (dB)", "{:.2f}", "{:+.2f}"),
              ("ssim_y", "SSIM-Y", "{:.4f}", "{:+.4f}"),
              ("lpips", "LPIPS", "{:.4f}", "{:+.4f}"),
              ("musiq", "MUSIQ", "{:.2f}", "{:+.2f}"),
              ("niqe", "NIQE", "{:.3f}", "{:+.3f}")]


def upscale():
    rows = read_csv(UPSCALE_SUMMARY)
    metric = {(r["split"], r["condition"], r["metric"]): r for r in rows
              if r["section"] == "metric"}
    paired_ = {(r["split"], r["metric"]): r for r in rows if r["section"] == "paired_diff"}
    counts = {r["split"]: r for r in rows if r["section"] == "images"}
    head = ["Split (images)", "Metric", "Bicubic", "Real-ESRGAN",
            "Real-ESRGAN − bicubic", "Real-ESRGAN better / equal / worse"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---:|---:|---:|---:|"]
    for sp, label in UP_SPLITS:
        c = counts[sp]
        sp_label = f"{label} ({c['n']}" + (f", {c['n_excluded']} excluded)"
                                           if c["n_excluded"] not in ("", "0") else ")")
        for m, mlabel, fmt, dfmt in UP_METRICS:
            b, r_ = metric[(sp, "bicubic", m)], metric[(sp, "realesrgan", m)]
            p = paired_[(sp, m)]
            cells = [sp_label, mlabel,
                     ci(fmt, float(b["mean"]), float(b["ci_low"]), float(b["ci_high"])),
                     ci(fmt, float(r_["mean"]), float(r_["ci_low"]), float(r_["ci_high"])),
                     ci(dfmt, float(p["mean"]), float(p["ci_low"]), float(p["ci_high"])),
                     f"{p['n_better']} / {p['n_equal']} / {p['n_worse']}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


ROUTES = ["no upscale", "Real-ESRGAN x2", "Real-ESRGAN x4"]


def routing():
    rows = read_csv(ROUTING_SUMMARY)
    with open(ROUTING_META) as f:
        expected = {c: v["route"] for c, v in json.load(f)["expected_mapping"].items()}
    conf = {(r["condition"], r["recommendation"]): r for r in rows
            if r["section"] == "confusion_route"}
    ok = {r["condition"]: r for r in rows if r["section"] == "as_expected"}
    conds = list(expected)
    head = ["Condition", "Expected route"] + ROUTES + ["Routed as expected [95% CI]"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---:|---:|---:|---:|"]
    for c in conds:
        cells = [c, expected[c]]
        for rt in ROUTES:
            r = conf[(c, rt)]
            cells.append(f"{r['count']} ({100 * float(r['share']):.1f}%)")
        o = ok[c]
        cells.append(f"{o['count']} / {o['n']}: "
                     + ci("{:.1f}%", 100 * float(o["mean"]), 100 * float(o["ci_low"]),
                          100 * float(o["ci_high"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FACE_CONDS = [("bicubic", "Bicubic"), ("restore_noface", "Real-ESRGAN only"),
              ("restore_face", "Real-ESRGAN + CodeFormer")]
FACE_METRICS = [("psnr_y", "PSNR-Y (dB)", "{:.2f}"), ("ssim_y", "SSIM-Y", "{:.4f}"),
                ("lpips", "LPIPS", "{:.4f}")]


def faces():
    with open(FACES_META) as f:
        meta = json.load(f)
    det = meta["detection"]
    if "stopped" in meta:
        return ("| Images scanned | With any face | With a face at least 64 px across | "
                "Evaluation |\n|---:|---:|---:|---|\n"
                f"| {det['images_scanned']} | {det['images_with_any_face']} | "
                f"{det['images_qualifying']} | stopped: fewer than {meta['parameters']['min_face_images']} "
                "face images |")
    rows = read_csv(FACES_SUMMARY)
    metric = {(r["condition"], r["metric"]): r for r in rows if r["section"] == "metric"}
    paired_ = {(r["condition"], r["metric"]): r for r in rows if r["section"] == "paired_diff"}
    n = metric[("bicubic", "psnr_y")]["n"]
    head = [f"Method (face box, {n} images)"] + [m[1] for m in FACE_METRICS]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * len(FACE_METRICS)]
    for c, label in FACE_CONDS:
        cells = [label]
        for m, _, fmt in FACE_METRICS:
            r = metric[(c, m)]
            cells.append(ci(fmt, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])))
        lines.append("| " + " | ".join(cells) + " |")
    for comp, label in (("restore_face - restore_noface", "CodeFormer vs Real-ESRGAN only: "
                         "better / equal / worse"),
                        ("restore_face - bicubic", "CodeFormer vs bicubic: better / equal / worse")):
        cells = [label]
        for m, _, _ in FACE_METRICS:
            p = paired_[(comp, m)]
            cells.append(f"{p['n_better']} / {p['n_equal']} / {p['n_worse']}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SWEEP_CONDS = [("bicubic", "Bicubic"), ("realesrgan", "Real-ESRGAN only"),
               ("codeformer_f0.3", "Real-ESRGAN + CodeFormer, fidelity 0.3"),
               ("codeformer_f0.7", "Real-ESRGAN + CodeFormer, fidelity 0.7"),
               ("codeformer_f1.0", "Real-ESRGAN + CodeFormer, fidelity 1.0")]
SWEEP_METRICS = FACE_METRICS + [("musiq", "MUSIQ", "{:.2f}"),
                                ("identity_cos", "Identity cosine", "{:.4f}")]
SWEEP_LEVELS = [("mild", "Mild damage (benchmark defaults)"),
                ("severe", "Severe damage (4x down, blur 2, noise 15, JPEG 30)")]


def faces_sweep():
    rows = read_csv(SWEEP_SUMMARY)
    metric = {(r["level"], r["condition"], r["metric"]): r for r in rows
              if r["section"] == "metric"}
    paired_ = {(r["level"], r["condition"], r["metric"]): r for r in rows
               if r["section"] == "paired_diff"}
    ran = {(r["level"], r["condition"]): r for r in rows if r["section"] == "codeformer_ran"}
    out = []
    for lv, lv_label in SWEEP_LEVELS:
        n = metric[(lv, "bicubic", "psnr_y")]["n"]
        ran_txt = ", ".join(f"{c.split('_f')[1]}: {ran[(lv, c)]['mean'].split('.')[0]} of {n}"
                            for c, _ in SWEEP_CONDS if c.startswith("codeformer"))
        out.append(f"**{lv_label}**, face box, {n} images. CodeFormer found a face to "
                   f"restore ({ran_txt}).")
        out.append("")
        head = ["Method"] + [m[1] for m in SWEEP_METRICS]
        out.append("| " + " | ".join(head) + " |")
        out.append("|---|" + "---:|" * len(SWEEP_METRICS))
        for c, label in SWEEP_CONDS:
            cells = [label]
            for m, _, fmt in SWEEP_METRICS:
                r = metric[(lv, c, m)]
                cells.append(ci(fmt, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])))
            out.append("| " + " | ".join(cells) + " |")
        for c, label in SWEEP_CONDS:
            if c == "realesrgan":
                continue
            cells = [f"{label} vs Real-ESRGAN only: better / equal / worse"]
            for m, _, _ in SWEEP_METRICS:
                p = paired_[(lv, f"{c} - realesrgan", m)]
                cells.append(f"{p['n_better']} / {p['n_equal']} / {p['n_worse']}")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip("\n")


OPTIONAL = {"upscale": upscale, "routing": routing, "faces": faces, "faces_sweep": faces_sweep}


def replace_block(text, name, body):
    pattern = re.compile(rf"(<!-- BEGIN {name} -->\n)(.*?)(<!-- END {name} -->)", re.S)
    found = pattern.findall(text)
    if len(found) != 1:
        raise SystemExit(f"docs/index.md needs exactly one '{name}' marker pair "
                         f"(found {len(found)})")
    return pattern.sub(lambda m: m.group(1) + "\n" + body + "\n\n" + m.group(3), text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="show", choices=["headline", "paired", "upscaler",
                                                      "credits"] + list(OPTIONAL),
                    help="print one table to stdout and write nothing")
    args = ap.parse_args()
    with open(INDEX_MD) as f:
        text = f.read()
    if args.show:
        rows = read_summary()
        builders = {"headline": lambda: headline(rows), "paired": lambda: paired(rows),
                    "upscaler": upscaler, "credits": lambda: credits(text), **OPTIONAL}
        print(builders[args.show]())
        return 0
    rows = read_summary()
    for name, body in (("headline", headline(rows)), ("paired", paired(rows)),
                       ("upscaler", upscaler()), ("credits", credits(text))):
        text = replace_block(text, name, body)
        print(f"{name}: {body.count(chr(10)) - 1} rows")
    for name, build in OPTIONAL.items():
        begins = text.count(f"<!-- BEGIN {name} -->")
        ends = text.count(f"<!-- END {name} -->")
        if begins == 0 and ends == 0:
            print(f"{name}: no markers in docs/index.md, skipped")
            continue
        body = build()
        text = replace_block(text, name, body)
        print(f"{name}: {body.count(chr(10)) - 1} rows")
    with open(INDEX_MD, "w") as f:
        f.write(text)
    print(f"Wrote {os.path.relpath(INDEX_MD, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
