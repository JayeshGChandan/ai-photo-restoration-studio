"""
legacy_composite (scripts/legacy_compositing.py) must reproduce the pre-fix
compositing.py byte for byte, and must stay out of the app's import graph.

The reference is the pre-fix compositing.py itself, read from git by blob hash
(it was compositing.py at commit 3bece00 and when this test was written), so
the comparison stays exact after compositing.py changes.
"""

import os
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from legacy_compositing import legacy_composite, legacy_feather_mask  # noqa: E402

PREFIX_BLOB = "bdd95c117cbddf7e269dcf524c9c8042db1a21a3"
SIZES = [(1536, 1024, 10), (2400, 1600, 11)]
RADII = [0, 3, 15]


def _prefix_module():
    src = subprocess.run(["git", "cat-file", "-p", PREFIX_BLOB], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    module = types.ModuleType("compositing_prefix")
    exec(compile(src, f"<git blob {PREFIX_BLOB}>", "exec"), module.__dict__)
    return module


PREFIX = _prefix_module()


def _inputs(w, h, seed):
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    restored = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(0.2 * h):int(0.45 * h), int(0.1 * w):int(0.35 * w)] = 255
    cv2.line(mask, (int(0.45 * w), int(0.15 * h)), (int(0.85 * w), int(0.8 * h)), 255, 9)
    cv2.circle(mask, (int(0.25 * w), int(0.7 * h)), int(0.08 * h), 255, -1)
    mask[:20, w - 60:] = 255        # touches the top and right borders
    mask[h - 7:, :35] = 255         # touches the bottom and left borders
    return (Image.fromarray(original), Image.fromarray(restored), Image.fromarray(mask, "L"))


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_legacy_matches_prefix_composite_byte_for_byte(size, radius):
    original, restored, mask = _inputs(*size)
    w, h, _ = size
    ours = np.asarray(legacy_composite(original, restored, mask, feather_radius=radius))
    ref = np.asarray(PREFIX.composite(original, restored, mask, feather_radius=radius))
    assert ours.dtype == ref.dtype and ours.shape == ref.shape
    assert ours.tobytes() == ref.tobytes()
    assert (np.asarray(legacy_feather_mask(mask, (w, h), radius)).tobytes()
            == np.asarray(PREFIX.feather_mask(mask, (w, h), radius)).tobytes())


def test_app_and_pipeline_do_not_import_legacy():
    for name in ("app.py", "pipeline.py"):
        with open(os.path.join(REPO_ROOT, name)) as f:
            assert "legacy_compositing" not in f.read(), f"{name} mentions legacy_compositing"
    # Import both the way the app runs (repo root on sys.path, scripts/ not) and
    # check the legacy module is neither loaded nor importable by its name.
    code = (
        "import importlib, sys\n"
        "import app, pipeline\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[-1] == 'legacy_compositing')\n"
        "try:\n"
        "    importlib.import_module('legacy_compositing')\n"
        "    importable = True\n"
        "except ModuleNotFoundError:\n"
        "    importable = False\n"
        "print(loaded, importable)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True, env=dict(os.environ, PYTHONPATH=""))
    assert out.stdout.strip().splitlines()[-1] == "[] False"
