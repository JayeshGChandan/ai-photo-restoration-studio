"""
Empty-U handling in the compositing experiment (runner + aggregation).

score_conditions() is driven with synthetic arrays standing in for the LaMa
outputs, so no model is loaded; LPIPS runs for real on small images. One pair
has a mask whose fixed r=15 halo covers every non-mask pixel (U empty), the
other two leave U non-empty.
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_compositing_experiment")
summ = _load("summarise_compositing")


def _pair(name, w, h, mask_box, seed):
    """Score one synthetic pair; mask_box = (y0, y1, x0, x1) in pixels."""
    rng = np.random.default_rng(seed)
    I = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1, x0, x1 = mask_box
    mask[y0:y1, x0:x1] = 255
    D = I.copy()
    D[mask > 0] = 235
    # Stand-ins for LaMa: L changes every pixel slightly (like the 1536 px
    # round trip), N2 changes every pixel more.
    L = np.clip(I.astype(int) + rng.integers(-2, 3, I.shape), 0, 255).astype(np.uint8)
    N2 = np.clip(I.astype(int) + rng.integers(-9, 10, I.shape), 0, 255).astype(np.uint8)
    base = {"image": name, "split": "synthetic", "width": w, "height": h,
            "size_class": "at_or_below_cap", "image_index": seed, "seed": 0,
            "coverage_target": "15", "coverage_actual": 100.0 * (mask > 0).mean(),
            "mask_n_strokes": 1, "mask_n_blobs": 1}
    rows = runner.score_conditions(I, D, mask, L, N2, base)
    for r in rows:
        r["pair_runtime_s"] = 0.0
    return rows


@pytest.fixture(scope="module")
def pairs():
    return {
        # 96x72 with a central 10x10 mask: the 40 px dilation plus blur cover it all.
        "empty": _pair("empty.png", 96, 72, (31, 41, 43, 53), seed=1),
        # 240x180 with a small corner mask: most of the image stays untouched.
        "full_a": _pair("full_a.png", 240, 180, (4, 14, 4, 14), seed=2),
        "full_b": _pair("full_b.png", 240, 180, (160, 172, 220, 232), seed=3),
    }


def test_empty_u_is_recorded_not_raised(pairs):
    rows = pairs["empty"]
    assert [r["condition"] for r in rows] == runner.CONDITIONS
    for r in rows:
        assert r["u_empty"] is True and r["u_pixels"] == 0 and r["u_fraction"] == 0.0
        for col in runner.U_METRIC_COLUMNS:
            assert r[col] is None, col
        # Every non-U metric is still computed and finite.
        assert r["m_n_pixels"] == 100
        assert np.isfinite(r["m_psnr_y"])
        if r["condition"] in runner.LPIPS_CONDITIONS:
            assert np.isfinite(r["m_lpips"]) and np.isfinite(r["off_lpips"])
        if r["condition"] in runner.COMPOSITES:
            assert np.isfinite(r["m_d_weight"])
        if r["condition"] in runner.HALO_CONDITIONS:
            assert r["h_n_pixels"] > 0 and np.isfinite(r["h_pct_changed"])


def test_non_empty_u_has_every_u_metric(pairs):
    for r in pairs["full_a"]:
        assert r["u_empty"] is False and r["u_pixels"] > 0
        assert 0.0 < r["u_fraction"] < 1.0
        assert r["u_pixels"] == round(r["u_fraction"] * 240 * 180)
        for col in runner.U_METRIC_COLUMNS:
            assert r[col] is not None and np.isfinite(r[col]), col


def test_validate_still_fails_loudly_on_other_metrics(pairs):
    row = dict(pairs["empty"][1])          # C_3
    row["m_psnr_y"] = float("nan")
    with pytest.raises(RuntimeError, match="m_psnr_y"):
        runner.validate(row)
    row = dict(pairs["empty"][1])
    row["u_mean_abs"] = 0.0                # a U metric must stay blank when U is empty
    with pytest.raises(RuntimeError, match="u_mean_abs"):
        runner.validate(row)
    row = dict(pairs["full_a"][1])
    row["u_mean_abs"] = None               # ...and must be set when U is not empty
    with pytest.raises(RuntimeError, match="u_mean_abs"):
        runner.validate(row)
    row = dict(pairs["full_a"][1])
    row["u_empty"] = True                  # u_empty must agree with u_pixels
    with pytest.raises(RuntimeError, match="u_empty"):
        runner.validate(row)


@pytest.fixture(scope="module")
def raw_csv(pairs, tmp_path_factory):
    path = tmp_path_factory.mktemp("exp") / "raw.csv"
    for key in ("empty", "full_a", "full_b"):
        runner.append_rows(str(path), pairs[key])
    return path


def test_raw_csv_writes_blank_u_metrics(raw_csv):
    with open(raw_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    empty = [r for r in rows if r["image"] == "empty.png"]
    assert len(empty) == len(runner.CONDITIONS)
    for r in empty:
        assert r["u_empty"] == "true" and r["u_pixels"] == "0"
        assert all(r[c] == "" for c in runner.U_METRIC_COLUMNS)
        assert r["m_psnr_y"] != ""


def test_aggregation_excludes_empty_u_from_u_statistics_only(raw_csv):
    groups = summ.load(str(raw_csv))
    metric_rows, cells = summ.summarise(groups)
    key = ("at_or_below_cap", "15")
    for cond in summ.CONDITIONS:
        for col, _, conds, _ in summ.METRICS:
            if cond not in conds:
                continue
            n = cells[key + (cond, col)][3]
            assert n == (2 if summ.is_u_metric(col) else 3), (cond, col, n)
    rec = [r for r in metric_rows if r["condition"] == "C_3" and r["metric"] == "u_mean_abs"]
    assert rec[0]["n"] == 2 and rec[0]["n_u_empty"] == 1

    paired = summ.paired(groups)
    for r in paired:
        expected = 2 if summ.is_u_metric(r["metric"]) else 3
        assert r["n"] == expected, r
        assert r["n_better"] + r["n_equal"] + r["n_worse"] == expected

    nulls = summ.null_case(groups)
    assert [(r["n"], r["n_u_empty"]) for r in nulls] == [(2, 1)]

    sizes = summ.u_size(groups)
    assert len(sizes) == 1 and sizes[0]["n"] == 3 and sizes[0]["n_u_empty"] == 1
    assert sizes[0]["min"] == 0.0 and 0.0 < sizes[0]["mean"] < 1.0


def test_aggregation_prints_reduced_n(raw_csv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["summarise_compositing.py", "--dir", str(raw_csv.parent)])
    assert summ.main() == 0
    out = capsys.readouterr().out
    assert "u_empty pairs 1" in out
    u_line = next(line for line in out.splitlines() if line.startswith("U mean |diff|"))
    assert u_line.count("n=2") == len(summ.CONDITIONS)
    m_line = next(line for line in out.splitlines() if line.startswith("M PSNR-Y"))
    assert "n=" not in m_line
    assert "(1 u_empty excluded)" in out


def test_aggregation_rejects_u_metric_on_empty_u(raw_csv, tmp_path):
    with open(raw_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["image"] == "empty.png" and r["condition"] == "N1":
            r["u_mean_abs"] = "0.5"
    bad = tmp_path / "raw.csv"
    with open(bad, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="empty U"):
        summ.load(str(bad))
