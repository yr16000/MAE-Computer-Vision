from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ── Grid constants ─────────────────────────────────────────────────────────
GRID = 14
N_PATCHES = GRID * GRID  # 196


def patch_rc(patch_id: int) -> Tuple[int, int]:
    return divmod(patch_id, GRID)


def rc_patch(r: int, c: int) -> int:
    return r * GRID + c


# ── Region definitions ─────────────────────────────────────────────────────
BORDER_PATCH_IDS: List[int] = [
    pid for pid in range(N_PATCHES)
    if patch_rc(pid)[0] in (0, GRID - 1) or patch_rc(pid)[1] in (0, GRID - 1)
]

PATCH_CENTER: int = rc_patch(GRID // 2, GRID // 2)          # 98
PATCH_CORNERS: Tuple[int, ...] = (0, GRID - 1, (GRID - 1) * GRID, N_PATCHES - 1)  # 0, 13, 182, 195
PATCH_BORDERS: List[List[int]] = [BORDER_PATCH_IDS]
PATCH_PROBES: Tuple[int, ...] = (0, 13, 182, 195, 6, 189, 93, 102, 98, 45, 150, 72, 123)


def patch_region(patch_id: int) -> str:
    r, c = patch_rc(patch_id)
    corner = r in (0, GRID - 1) and c in (0, GRID - 1)
    edge = r in (0, GRID - 1) or c in (0, GRID - 1)
    if corner:
        return 'corner'
    if edge:
        return 'border'
    return 'interior'


# ── Core math ──────────────────────────────────────────────────────────────
def _cosine_matrix(z: torch.Tensor) -> np.ndarray:
    z_norm = F.normalize(z.float(), dim=-1)
    return (z_norm @ z_norm.T).cpu().numpy()


def mean_offdiag_cos(mat: np.ndarray) -> float:
    k = mat.shape[0]
    mask = ~np.eye(k, dtype=bool)
    return float(mat[mask].mean())


def pca_coords(z: torch.Tensor, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(z.float().numpy())


# ── Result container ───────────────────────────────────────────────────────
@dataclass
class StabilityResult:
    patch_id: int
    fixed: bool
    z_patch: torch.Tensor
    z_cls: torch.Tensor
    z_mean: torch.Tensor
    masks: torch.Tensor
    pairwise_cos_patch: np.ndarray
    pairwise_cos_cls: np.ndarray
    pairwise_cos_mean: np.ndarray

    @property
    def mean_offdiag_cos_patch(self) -> float:
        return mean_offdiag_cos(self.pairwise_cos_patch)


# ── Noise builders ─────────────────────────────────────────────────────────
def build_noise(
    patch_id: int, seed: int, device: torch.device,
    *, fixed: bool = True, n_patches: int = N_PATCHES,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.rand(1, n_patches, generator=gen, device=device)
    if fixed:
        noise[0, patch_id] = -1.0  # lowest value → always kept visible
    return noise


def build_border_noise(
    border_ids: Sequence[int], seed: int, device: torch.device,
    *, n_patches: int = N_PATCHES,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.full((1, n_patches), float('inf'), device=device)
    border_noise = torch.rand(1, len(border_ids), generator=gen, device=device)
    for j, pid in enumerate(border_ids):
        noise[0, pid] = border_noise[0, j]
    return noise


# ── Encoder wrapper ────────────────────────────────────────────────────────
def encode_once(
    image, encoder, processor, device,
    *, noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = encoder(**inputs, noise=noise)
    hidden = out.last_hidden_state[0]       # (n_visible+1, D)
    mask = out.mask[0]                      # (N_PATCHES,)  1=masked 0=visible
    ids_restore = out.ids_restore[0]
    ids_keep = torch.where(mask == 0)[0]
    return hidden, mask, ids_restore, ids_keep


def latent_for_patch(
    hidden: torch.Tensor, ids_keep: torch.Tensor, patch_id: int,
) -> torch.Tensor:
    pos = (ids_keep == patch_id).nonzero(as_tuple=False)
    if len(pos) == 0:
        raise ValueError(f"Patch {patch_id} not visible")
    return hidden[pos[0].item() + 1].cpu().float()  # +1 skips CLS token


# ── Stability sampling ─────────────────────────────────────────────────────
def sample_stability(
    image, encoder, processor, device,
    *, patch_id: int, K: int, seed0: int = 0, fixed: bool = True,
) -> StabilityResult:
    z_patch_list, z_cls_list, z_mean_list, masks_list = [], [], [], []
    seed = seed0
    attempts = 0

    while len(z_patch_list) < K and attempts < K * 50:
        noise = build_noise(patch_id, seed, device, fixed=fixed)
        hidden, mask, _, ids_keep = encode_once(image, encoder, processor, device, noise=noise)
        seed += 1
        attempts += 1

        if mask[patch_id] != 0:
            continue

        try:
            zp = latent_for_patch(hidden, ids_keep, patch_id)
        except ValueError:
            continue

        z_patch_list.append(zp)
        z_cls_list.append(hidden[0].cpu().float())
        z_mean_list.append(hidden[1:].mean(dim=0).cpu().float())
        masks_list.append(mask.cpu())

    if len(z_patch_list) < 2:
        raise RuntimeError(f"Not enough visible draws for patch {patch_id} (got {len(z_patch_list)})")

    zp = torch.stack(z_patch_list)
    zc = torch.stack(z_cls_list)
    zm = torch.stack(z_mean_list)

    return StabilityResult(
        patch_id=patch_id, fixed=fixed,
        z_patch=zp, z_cls=zc, z_mean=zm,
        masks=torch.stack(masks_list),
        pairwise_cos_patch=_cosine_matrix(zp),
        pairwise_cos_cls=_cosine_matrix(zc),
        pairwise_cos_mean=_cosine_matrix(zm),
    )


def summarize(result: StabilityResult) -> Dict:
    k = result.z_patch.shape[0]
    off = result.pairwise_cos_patch[~np.eye(k, dtype=bool)]
    return {
        'patch_id': result.patch_id,
        'K': float(k),
        'mean_cos_patch': mean_offdiag_cos(result.pairwise_cos_patch),
        'mean_cos_cls':   mean_offdiag_cos(result.pairwise_cos_cls),
        'mean_cos_mean':  mean_offdiag_cos(result.pairwise_cos_mean),
        'std_cos_patch':  float(off.std()),
    }


# ── Grid-level analysis ────────────────────────────────────────────────────
def stability_grid(
    image, encoder, processor, device,
    *, K: int = 12, seed0: int = 0,
    visible_region: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    patch_map = np.full((GRID, GRID), np.nan, dtype=np.float32)
    cls_map   = np.full((GRID, GRID), np.nan, dtype=np.float32)

    probe_ids = BORDER_PATCH_IDS if visible_region == 'border' else list(range(N_PATCHES))

    for pid in probe_ids:
        try:
            res = sample_stability(
                image, encoder, processor, device,
                patch_id=pid, K=K, seed0=seed0 + pid, fixed=True,
            )
        except RuntimeError:
            continue
        r, c = patch_rc(pid)
        patch_map[r, c] = res.mean_offdiag_cos_patch
        cls_map[r, c]   = mean_offdiag_cos(res.pairwise_cos_cls)

    return patch_map, cls_map


def stability_across_images(
    dataset, encoder, processor, device,
    indices: Sequence[int],
    *, patch_ids: Sequence[int], K: int,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict]:
    rows = []
    for img_idx in indices:
        img_pil, label = dataset[img_idx]
        class_name = class_names[label] if class_names is not None else str(label)
        for pid in patch_ids:
            try:
                res = sample_stability(
                    img_pil, encoder, processor, device,
                    patch_id=pid, K=K, seed0=img_idx, fixed=True,
                )
            except RuntimeError:
                continue
            rows.append({
                'image_idx':      img_idx,
                'class_id':       label,
                'class_name':     class_name,
                'patch_id':       pid,
                'region':         patch_region(pid),
                'mean_cos_patch': res.mean_offdiag_cos_patch,
                'mean_cos_cls':   mean_offdiag_cos(res.pairwise_cos_cls),
                'mean_cos_mean':  mean_offdiag_cos(res.pairwise_cos_mean),
            })
    return rows


# ── UMAP bank ──────────────────────────────────────────────────────────────
def collect_representation_bank(
    dataset, encoder, processor, device,
    indices: Sequence[int],
    *, patch_id: int, K: int,
    rep_types: Tuple[str, ...] = ('patch', 'mean', 'cls'),
) -> Tuple[np.ndarray, Dict]:
    type_map = {'patch': 0, 'mean': 1, 'cls': 2}
    emb_list, class_ids, image_idxs, rep_type_ids = [], [], [], []

    for img_idx in indices:
        img_pil, label = dataset[img_idx]
        try:
            res = sample_stability(
                img_pil, encoder, processor, device,
                patch_id=patch_id, K=K, seed0=img_idx * 100, fixed=True,
            )
        except RuntimeError:
            continue

        for rep in rep_types:
            z = {'patch': res.z_patch, 'mean': res.z_mean, 'cls': res.z_cls}[rep]
            for vec in z:
                emb_list.append(vec.numpy())
                class_ids.append(label)
                image_idxs.append(img_idx)
                rep_type_ids.append(type_map[rep])

    emb = np.stack(emb_list)
    meta = {
        'class_id':  np.array(class_ids),
        'image_idx': np.array(image_idxs),
        'rep_type':  np.array(rep_type_ids),
    }
    return emb, meta


def save_umap_bundle(path: str, emb: np.ndarray, meta: Dict) -> None:
    np.savez(path, emb=emb, **meta)


# ── Convergence analysis ───────────────────────────────────────────────────
def convergence_vs_K(
    image, encoder, processor, device,
    *, patch_id: int,
    K_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> Tuple[List[int], List[float], List[float]]:
    K_max = max(K_values)
    res = sample_stability(
        image, encoder, processor, device,
        patch_id=patch_id, K=K_max, seed0=0, fixed=True,
    )
    zp = res.z_patch  # (K_max, D)

    K_list, cos_list, std_list = [], [], []
    n_boot = 20

    for K in K_values:
        cosines = []
        for _ in range(n_boot):
            idx_a = torch.randperm(K_max)[:K]
            idx_b = torch.randperm(K_max)[:K]
            mean_a = zp[idx_a].float().mean(dim=0)
            mean_b = zp[idx_b].float().mean(dim=0)
            cos = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()
            cosines.append(cos)
        K_list.append(K)
        cos_list.append(float(np.mean(cosines)))
        std_list.append(float(zp[torch.randperm(K_max)[:K]].float().numpy().std(axis=0).mean()))

    return K_list, cos_list, std_list
