"""Download and extract ImageNette for mae_test.ipynb (train/ + val/)."""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "imagenette"
ARCHIVE = ROOT / "imagenette.tgz"


def _has_splits(folder: Path) -> bool:
    return (folder / "train").is_dir() and (folder / "val").is_dir()


def _find_extracted_root(extract_dir: Path) -> Path | None:
    for candidate in extract_dir.iterdir():
        if candidate.is_dir() and _has_splits(candidate):
            return candidate
    if _has_splits(extract_dir):
        return extract_dir
    return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        print(f"\r  {done / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def main() -> int:
    if _has_splits(TARGET):
        print(f"ImageNette already present at {TARGET}")
        return 0

    if not ARCHIVE.is_file():
        try:
            download(IMAGENETTE_URL, ARCHIVE)
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1

    print(f"Extracting {ARCHIVE.name} ...")
    extract_dir = ROOT / "_imagenette_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(extract_dir)

    extracted = _find_extracted_root(extract_dir)
    if extracted is None:
        print("Could not find train/ and val/ in archive.", file=sys.stderr)
        return 1

    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.move(str(extracted), str(TARGET))
    shutil.rmtree(extract_dir, ignore_errors=True)

    n_train = sum(1 for _ in (TARGET / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (TARGET / "val").rglob("*.JPEG"))
    print(f"Ready: {TARGET}")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
