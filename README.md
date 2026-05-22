# MAE Computer Vision — SPEIT 2026

*Yann BUTEL, Ahmed EL KHOULY, Philippe LIN, Yanis RABIA*

---

## Setup

```bash
git lfs install   # only needed once, if not already installed
git clone https://github.com/yr16000/MAE-Computer-Vision.git
cd MAE-Computer-Vision
pip install -r requirements
```

| Resource | Size | Auto-download |
|---|---|---|
| `facebook/vit-mae-large` | ~1 GB | Yes |
| `openai/clip-vit-large-patch14` | ~1 GB | Yes |
| `llava-1.5-7b-hf` | ~14 GB | Yes |
| ImageNette (val set) | ~1.5 GB | Yes |
| ImageNet-100 (`ilee0022/ImageNet100`) | ~17 GB | Yes |

---

## Notebooks

### MAE Latent Space Analysis

| Notebook | Description |
|---|---|
| `wp1_mae_umap_visualization.ipynb` | UMAP visualization of MAE patch and CLS embeddings on ImageNette |
| `wp2_mae_patch_sampling_analysis.ipynb` | Latent stability analysis of patch representations under random masking |

### MAE → CLIP Projection (closed vocabulary)

| Notebook | Description |
|---|---|
| `wp3_6_cls_augmented.ipynb` | Training — CLS token + masking augmentation K=5 on ImageNet-100 |
| `wp3_cls_aug_test.ipynb` | Zero-shot evaluation of the two best models — loads checkpoints directly, no retraining needed |

Requires: `projection_cls_aug_mlp_best.pt`, `projection_cls_aug_mlpv2_best.pt`, `z_cls_aug_K5_norm_stats.pt` (included).

### MAE → LLaVA-1.5 Projection (open-ended generation)

| Notebook | Description |
|---|---|
| `wp4v5_generation_training.ipynb` | Generates 6.5M training pairs (100 masked MAE CLS per image, averaged into 50 groups) then trains the projection MLP targeting LLaVA's CLIP 336px vision tower |
| `wp4v5_4_test.ipynb` | Injects projected MAE embeddings into LLaVA-1.5 and generates image descriptions |

Run `wp4v5_generation_training.ipynb` first. `wp4v5_4_test.ipynb` requires `wp4v5_ftheta_best_24.pt` and `wp4v5_norm_stats.pt` (included via Git LFS).
