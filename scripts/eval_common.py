"""
Shared helpers for the Auto Restore evaluations (eval_upscale.py, eval_routing.py,
eval_faces.py).

  * every image comes from data/manifest.csv and is sha256-checked;
  * every run is forced onto the CPU (RESTORE_DEVICE=cpu before any model
    loads) and records the device each model actually used;
  * means use the compositing experiment's own percentile bootstrap
    (scripts/summarise_compositing.py: 10,000 resamples, seed 0);
  * paired better / equal / worse counts compare per-image values exactly.
"""

import csv
import hashlib
import os
import platform
import subprocess
import sys

# Pin every backend to the CPU before restorers/ or metrics.py read the
# environment. An explicit non-CPU request is refused rather than overridden.
if os.environ.get("RESTORE_DEVICE", "cpu") != "cpu":
    raise SystemExit("These evaluations run on CPU only; unset RESTORE_DEVICE or set it to cpu.")
os.environ["RESTORE_DEVICE"] = "cpu"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from summarise_compositing import BOOT_SEED, N_RESAMPLES, bootstrap_ci  # noqa: E402,F401

MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.csv")
DATA_DIR = os.path.join(REPO_ROOT, "data", "clic2020")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def manifest_rows():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def load_image(row):
    """Open a manifest image as RGB after checking its sha256."""
    path = os.path.join(DATA_DIR, row["split"], row["filename"])
    if sha256_file(path) != row["sha256"]:
        raise SystemExit(f"{path}: sha256 does not match data/manifest.csv")
    return Image.open(path).convert("RGB")


def provenance(script_paths):
    """Git state, environment and hashes of the scripts that produced a result."""
    import torch
    import pyiqa
    import spandrel
    import cv2
    import facexlib
    import PIL

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True).stdout.split("\n")
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no")
             .splitlines()]
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked_paths": dirty,
        "scripts_sha256": {os.path.relpath(p, REPO_ROOT): sha256_file(p) for p in script_paths},
        "scripts_committed": "in the same commit as these results",
        "restore_device_env": os.environ.get("RESTORE_DEVICE"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "pyiqa": pyiqa.__version__,
        "spandrel": spandrel.__version__,
        "opencv": cv2.__version__,
        "facexlib": getattr(facexlib, "__version__", "unknown"),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "data_manifest_sha256": sha256_file(MANIFEST),
        "pip_freeze": [x for x in freeze if x],
    }


class RowWriter:
    """CSV writer that flushes after every row, so a long run can be followed."""

    def __init__(self, path, columns):
        self.columns = columns
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=columns, lineterminator="\n")
        self.w.writeheader()
        self.f.flush()

    def write(self, row):
        missing = set(self.columns) - set(row)
        extra = set(row) - set(self.columns)
        if missing or extra:
            raise SystemExit(f"row fields do not match columns: missing {missing}, extra {extra}")
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})


def mean_row(values, **fields):
    """Mean and bootstrap 95% CI of ``values`` plus their range."""
    v = [float(x) for x in values]
    mean, lo, hi = bootstrap_ci(v)
    return {**fields, "n": len(v), "mean": mean, "ci_low": lo, "ci_high": hi,
            "min": min(v), "max": max(v)}


def paired_row(a_vals, b_vals, lower_better, **fields):
    """Per-image differences a - b: mean, bootstrap CI and better / equal / worse for a."""
    d = np.asarray([float(x) for x in a_vals]) - np.asarray([float(x) for x in b_vals])
    better = int(np.sum(d < 0) if lower_better else np.sum(d > 0))
    equal = int(np.sum(d == 0))
    mean, lo, hi = bootstrap_ci(d)
    return {**fields, "n": len(d), "mean": mean, "ci_low": lo, "ci_high": hi,
            "min": float(d.min()), "max": float(d.max()), "n_better": better,
            "n_equal": equal, "n_worse": len(d) - better - equal}
