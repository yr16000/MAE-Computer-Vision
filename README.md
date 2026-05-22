# WP3 — MAE → CLIP Alignment via Projection MLP

Aligning visual representations from **MAE ViT-L/16** to the **CLIP ViT-L/14** space via a trained projection MLP on ImageNet-100 (117,000 images, 100 classes).

## Base Models

| Model | HuggingFace ID | Dimension |
|-------|---------------|-----------|
| MAE ViT-L/16 (Facebook) | `facebook/vit-mae-large` | 1024 |
| CLIP ViT-L/14 (OpenAI) | `openai/clip-vit-large-patch14` | 768 |

Downloaded automatically from HuggingFace on first run.

## Requirements

- Python 3.10+
- GPU 16 GB+ VRAM recommended (for pair generation and training)
- ~20 GB disk space (ImageNet-100 dataset)

```bash
git clone <repo>
cd Computer-Vision
pip install -r requirements
```

## Dataset — ImageNet-100

117,000 train / 13,000 val images, 100 classes, ~17 GB.
Downloaded automatically on first run into `./imagenet100-hf/`.

---

## Full Pipeline (WP3)

### Pair Generation

| Notebook | Feature | Output files |
|----------|---------|--------------|
| `wp3_1_generation.ipynb` | `z_full` — mean of all MAE patches (noise=0) | `pairs_train.pt`, `pairs_val.pt` |
| `wp3_4_cls_projection.ipynb` | `z_cls` — MAE CLS token (noise=0) | `pairs_cls_train.pt`, `pairs_cls_val.pt` |
| `wp3_5_augmented.ipynb` | `z_full` with 75% masking, K=5 masks/image | `pairs_aug_K5_train.pt`, `pairs_aug_K5_val.pt` |
| `wp3_6_cls_augmented.ipynb` | `z_cls` with 75% masking, K=5 masks/image | `pairs_cls_aug_K5_train.pt`, `pairs_cls_aug_K5_val.pt` |

> `pairs_*.pt` files are **not** in the repo (too large). Regenerate by running the corresponding notebooks.

### MLP Architectures

| Name | Layers |
|------|--------|
| **MLP** | Linear → GELU → LayerNorm → Linear |
| **MLPv2** | BatchNorm → Linear → GELU → Dropout(0.2) → LayerNorm → Linear |

Both project from MAE space (1024d) to CLIP space (768d) with L2 normalisation on the output.

### Training & Results

| Notebook | Model | Val cos sim | Top-1 (test 5k) |
|----------|-------|-------------|-----------------|
| `wp3_2_training_baseline.ipynb` | MLP   + z_full          | 0.8894 | 38.1% |
| `wp3_3_improved.ipynb`          | MLPv2 + z_full          | 0.8898 | 38.4% |
| `wp3_4_cls_projection.ipynb`    | MLP   + z_cls           | 0.8925 | 40.5% |
| `wp3_4_cls_projection.ipynb`    | MLPv2 + z_cls           | 0.8929 | 41.7% |
| `wp3_5_augmented.ipynb`         | MLP   + z_full + aug K=5 | **0.9319** | 58.8% |
| `wp3_5_augmented.ipynb`         | MLPv2 + z_full + aug K=5 | 0.9306 | 71.9% |
| `wp3_6_cls_augmented.ipynb`     | MLP   + z_cls  + aug K=5 | **0.9345** | 63.5% |
| `wp3_6_cls_augmented.ipynb`     | MLPv2 + z_cls  + aug K=5 | 0.9331 | **73.5%** |

### Analysis

| Notebook | Content |
|----------|---------|
| `wp3_7_patch_analysis.ipynb` | Patch-level semantic analysis: masked image + CLIP top-1 overlay per visible patch |

---

## Provided Checkpoints

All trained models are available directly in the repo:

| File | Model | Val cos |
|------|-------|---------|
| `projection_mlp_best.pt` + `z_full_norm_stats.pt` | MLP   + z_full          | 0.8894 |
| `projection_mlpv2_best.pt` + `z_full_norm_stats.pt` | MLPv2 + z_full          | 0.8898 |
| `projection_cls_mlp_best.pt` + `z_cls_norm_stats.pt` | MLP   + z_cls           | 0.8925 |
| `projection_cls_mlpv2_best.pt` + `z_cls_norm_stats.pt` | MLPv2 + z_cls           | 0.8929 |
| `projection_aug_mlp_best.pt` + `z_aug_K5_norm_stats.pt` | MLP   + z_full + aug K=5 | 0.9319 |
| `projection_aug_mlpv2_best.pt` + `z_aug_K5_norm_stats.pt` | MLPv2 + z_full + aug K=5 | 0.9306 |
| `projection_cls_aug_mlp_best.pt` + `z_cls_aug_K5_norm_stats.pt` | MLP   + z_cls  + aug K=5 | 0.9345 |
| `projection_cls_aug_mlpv2_best.pt` + `z_cls_aug_K5_norm_stats.pt` | MLPv2 + z_cls  + aug K=5 | 0.9331 |

---

## Evaluation & Testing

```
wp3_evaluation.ipynb
```

Standalone evaluation notebook — **no retraining needed**.

- Loads all available models automatically
- Computes cosine similarity on the val set
- Zero-shot top-k predictions per model
- Final Top-1 accuracy on the full test set (5,000 images)
- **Custom image test**:

```python
IMAGE_PATH = './my_image.jpg'   # any image
TRUE_LABEL = None               # optional
```

---

## Repository Structure

```
.
├── wp3_1_generation.ipynb           # generate z_full pairs
├── wp3_2_training_baseline.ipynb    # MLP   + z_full
├── wp3_3_improved.ipynb             # MLPv2 + z_full
├── wp3_4_cls_projection.ipynb       # MLP + MLPv2 + z_cls
├── wp3_5_augmented.ipynb            # MLP + MLPv2 + z_full + aug K=5
├── wp3_6_cls_augmented.ipynb        # MLP + MLPv2 + z_cls  + aug K=5
├── wp3_7_patch_analysis.ipynb       # patch-level analysis
├── wp3_evaluation.ipynb             # evaluate all models
│
├── projection_mlp_best.pt           # checkpoint MLP   + z_full
├── projection_mlpv2_best.pt         # checkpoint MLPv2 + z_full
├── projection_cls_mlp_best.pt       # checkpoint MLP   + z_cls
├── projection_cls_mlpv2_best.pt     # checkpoint MLPv2 + z_cls
├── projection_aug_mlp_best.pt       # checkpoint MLP   + z_full + aug K=5
├── projection_aug_mlpv2_best.pt     # checkpoint MLPv2 + z_full + aug K=5
├── projection_cls_aug_mlp_best.pt   # checkpoint MLP   + z_cls  + aug K=5
├── projection_cls_aug_mlpv2_best.pt # checkpoint MLPv2 + z_cls  + aug K=5
│
├── z_full_norm_stats.pt             # normalisation stats z_full
├── z_cls_norm_stats.pt              # normalisation stats z_cls
├── z_aug_K5_norm_stats.pt           # normalisation stats z_full + aug K=5
├── z_cls_aug_K5_norm_stats.pt       # normalisation stats z_cls  + aug K=5
```
