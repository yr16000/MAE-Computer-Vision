"""Quick check: venv deps, MAE weights, ImageNette splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vit-mae-large"
DATA = ROOT / "imagenette"


def main() -> int:
    ok = True

    weights = MODEL / "pytorch_model.bin"
    for name in ("config.json", "preprocessor_config.json"):
        path = MODEL / name
        if path.is_file():
            print(f"OK  MAE  {path.relative_to(ROOT)}")
        else:
            print(f"MISSING  MAE  {path.relative_to(ROOT)}")
            ok = False

    if weights.is_file() and weights.stat().st_size >= 100_000_000:
        print(f"OK  MAE  {weights.relative_to(ROOT)}  ({weights.stat().st_size / 1e9:.2f} GB)")
    elif weights.is_file():
        print(f"LFS_POINTER  MAE  {weights.relative_to(ROOT)}  — run: python scripts/download_mae.py")
        ok = False
    else:
        print(f"MISSING  MAE  {weights.relative_to(ROOT)}")
        ok = False

    for split in ("train", "val"):
        d = DATA / split
        if d.is_dir() and any(d.rglob("*.JPEG")):
            n = sum(1 for _ in d.rglob("*.JPEG"))
            print(f"OK  ImageNette  {split}/  ({n} images)")
        else:
            print(f"MISSING  ImageNette  {split}/")
            ok = False

    try:
        import torch
        from transformers import ViTMAEModel

        print(f"OK  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        ViTMAEModel.from_pretrained(str(MODEL))
        print("OK  ViTMAEModel loads from vit-mae-large/")
    except Exception as exc:
        print(f"FAIL  imports/model: {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
