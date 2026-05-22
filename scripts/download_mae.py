"""Download vit-mae-large weights if Git LFS pointers were not pulled."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "vit-mae-large"
WEIGHTS = MODEL_DIR / "pytorch_model.bin"
MIN_BYTES = 100_000_000  # real checkpoint is ~1.3 GB


def weights_ok() -> bool:
    return WEIGHTS.is_file() and WEIGHTS.stat().st_size >= MIN_BYTES


def main() -> int:
    if weights_ok():
        print(f"MAE weights already present ({WEIGHTS.stat().st_size / 1e9:.2f} GB)")
        return 0

    if WEIGHTS.is_file() and WEIGHTS.stat().st_size < 1024:
        print("Detected Git LFS pointer — downloading real weights from Hugging Face ...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install dependencies first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id="facebook/vit-mae-large",
            filename="pytorch_model.bin",
            local_dir=str(MODEL_DIR),
            force_download=not weights_ok(),
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "\nRetry when online:\n"
            "  .venv\\Scripts\\python scripts\\download_mae.py\n"
            "The GitHub repo LFS quota is exceeded; Hugging Face is the supported source.",
            file=sys.stderr,
        )
        return 1

    if not weights_ok():
        print(
            "Download finished but pytorch_model.bin still looks invalid.",
            file=sys.stderr,
        )
        return 1

    print(f"Ready: {MODEL_DIR} ({WEIGHTS.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
