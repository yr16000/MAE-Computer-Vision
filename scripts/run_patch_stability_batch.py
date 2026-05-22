"""Batch patch-stability analysis (no notebook). Usage:

    .venv\\Scripts\\python scripts/run_patch_stability_batch.py --n-images 20 --k 24
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTMAEModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mae_patch_stability import (  # noqa: E402
    PATCH_CENTER,
    PATCH_PROBES,
    collect_representation_bank,
    save_umap_bundle,
    stability_across_images,
    stability_grid,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = ROOT / "outputs" / "patch_stability"
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    ds = ImageFolder(str(ROOT / "imagenette" / "val"), transform=transform)

    processor = ViTImageProcessor(
        size={"height": 224, "width": 224},
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    encoder = ViTMAEModel.from_pretrained(str(ROOT / "vit-mae-large")).to(args.device)
    encoder.eval()
    device = torch.device(args.device)

    rng = np.random.default_rng(42)
    idx = sorted(rng.choice(len(ds), size=min(args.n_images, len(ds)), replace=False))

    rows = stability_across_images(
        ds, encoder, processor, device, idx, patch_ids=PATCH_PROBES, K=args.k, class_names=ds.classes
    )
    csv_path = out / "stability_by_image_patch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    emb, meta = collect_representation_bank(
        ds, encoder, processor, device, idx[: min(15, len(idx))], patch_id=PATCH_CENTER, K=6
    )
    npz = out / "latent_bank_center_patch.npz"
    save_umap_bundle(str(npz), emb, meta)
    print(f"Wrote {npz} {emb.shape}")

    if args.full_grid:
        img, _ = ds[idx[0]]
        pmap, _ = stability_grid(img, encoder, processor, device, K=12)
        np.save(out / "stability_grid_patch.npy", pmap)
        print(f"Wrote grid {pmap.shape} mean={np.nanmean(pmap):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
