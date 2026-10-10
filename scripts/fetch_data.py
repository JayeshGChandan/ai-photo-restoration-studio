"""
Download the CLIC 2020 validation sets and verify them against the manifest.

Both splits come from the official CLIC download host (links on
https://archive.compression.cc/challenge/). Images are extracted flat into
``data/clic2020/<split>/`` (gitignored) and every file is checked against
``data/manifest.csv`` (filename, split, width, height, sha256). Any missing,
extra, or mismatching file makes the script exit non-zero.

Usage (from the repo root):
    ./venv/bin/python -I scripts/fetch_data.py                 # download + verify
    ./venv/bin/python -I scripts/fetch_data.py --verify-only   # verify existing files
    ./venv/bin/python -I scripts/fetch_data.py --write-manifest
        # one-off: (re)build the manifest from freshly downloaded files

Licence terms for each split are recorded in data/LICENSES.md.
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "clic2020")
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.csv")

SPLITS = {
    "mobile": "https://downloads.compression.cc/clic2020_mobile_valid.zip",
    "professional": "https://downloads.compression.cc/clic2020_professional_valid.zip",
}
MANIFEST_FIELDS = ["filename", "split", "width", "height", "sha256"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
        expected = resp.headers.get("Content-Length")
        shutil.copyfileobj(resp, out, length=1 << 20)
    size = os.path.getsize(dest)
    if expected is not None and size != int(expected):
        raise RuntimeError(f"Truncated download: got {size} bytes, expected {expected}")
    print(f"  {size:,} bytes")


def extract_pngs(zip_path: str, out_dir: str) -> int:
    """Extract only .png members, flattened to their basename.

    Taking the basename (and writing bytes ourselves) means no archive member
    can escape ``out_dir`` or create links, whatever its stored path.
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    seen = set()
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member in {zip_path}: {bad}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if not name.lower().endswith(".png") or name.startswith("."):
                continue
            if name in seen:
                raise RuntimeError(f"Duplicate basename in archive: {name}")
            seen.add(name)
            with zf.open(info) as src, open(os.path.join(out_dir, name), "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            count += 1
    return count


def describe(path: str, split: str) -> dict:
    with Image.open(path) as im:
        width, height = im.size
    return {
        "filename": os.path.basename(path),
        "split": split,
        "width": width,
        "height": height,
        "sha256": sha256_of(path),
    }


def read_manifest(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(data_dir: str, path: str) -> None:
    rows = []
    for split in sorted(SPLITS):
        split_dir = os.path.join(data_dir, split)
        for name in sorted(os.listdir(split_dir)):
            if name.lower().endswith(".png"):
                rows.append(describe(os.path.join(split_dir, name), split))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def verify(data_dir: str, manifest_path: str) -> bool:
    rows = read_manifest(manifest_path)
    errors = []
    expected = {(r["split"], r["filename"]) for r in rows}
    for r in rows:
        path = os.path.join(data_dir, r["split"], r["filename"])
        if not os.path.isfile(path):
            errors.append(f"missing: {r['split']}/{r['filename']}")
            continue
        got = describe(path, r["split"])
        for key in ("width", "height", "sha256"):
            if str(got[key]) != r[key]:
                errors.append(f"{key} mismatch: {r['split']}/{r['filename']} "
                              f"(manifest {r[key]}, file {got[key]})")
    for split in SPLITS:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            for name in os.listdir(split_dir):
                if name.lower().endswith(".png") and (split, name) not in expected:
                    errors.append(f"not in manifest: {split}/{name}")
    for e in errors:
        print(f"  FAIL {e}")
    print(f"Verified {len(rows) - len([e for e in errors if not e.startswith('not in')])}"
          f"/{len(rows)} files against {manifest_path}; {len(errors)} problem(s).")
    return not errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip downloading; verify files already on disk.")
    ap.add_argument("--write-manifest", action="store_true",
                    help="Download, then (re)write the manifest from the files.")
    args = ap.parse_args()

    if not args.verify_only:
        tmp = tempfile.mkdtemp(prefix="clic2020_")
        try:
            for split, url in SPLITS.items():
                zip_path = os.path.join(tmp, f"{split}.zip")
                download(url, zip_path)
                split_dir = os.path.join(args.data_dir, split)
                if os.path.isdir(split_dir):
                    shutil.rmtree(split_dir)
                n = extract_pngs(zip_path, split_dir)
                print(f"  extracted {n} PNGs to {split_dir}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if args.write_manifest:
        write_manifest(args.data_dir, args.manifest)

    return 0 if verify(args.data_dir, args.manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
