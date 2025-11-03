# -*- coding: utf-8 -*-
"""
Panoptic visualization for CellViT (ViT-256) on arbitrary image folders.

- Loads a trained checkpoint
- Iterates over images from a directory (no GT required)
- Postprocess (NP/HV -> watershed) to get instances
- Assign a single class per instance (core soft-vote from type head)
- Save side-by-side: [Raw | Panoptic(typed instances + yellow boundaries)]

Requirements:
    pip install scikit-image scipy pillow tqdm

Example:
    python infer_viz_images_panoptic.py \
      --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
      --input_dir Dataset/1024_crop/Preprocessed \
      --out eval_out/1024_crop \
      --batch_size 8 --device cuda \
      --img_size 256 --num_classes 5 \
      --np_thresh 0.4 --min_size 10 \
      --bnd_thickness 1 --bnd_alpha 0.95 \
      --core_erode_px 2
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.special import expit as sigmoid
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, feature
from skimage.morphology import disk, square

# local imports — adjust if your paths differ
from Model.CellViT_ViT256_Custom import CellViTCustom

# -----------------------------
# Small image-folder dataset
# -----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ImageFolderInferenceDataset(Dataset):
    def __init__(self, root: Path, img_size: int = 256, exts: Tuple[str, ...] = IMG_EXTS):
        self.root = Path(root)
        files = []
        for ext in exts:
            files += list(self.root.rglob(f"*{ext}"))
        self.files = sorted(files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images with extensions {exts} were found under: {root}")
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        im = Image.open(p).convert("RGB")
        if self.img_size is not None:
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        chw = np.transpose(arr, (2, 0, 1))             # (3,H,W)
        return {
            "image": torch.from_numpy(chw),  # float32 [0,1]
            "path": str(p)
        }


# -----------------------------
# Viz helpers
# -----------------------------
def to_hwc01(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) or (H,W) tensor -> (H,W,3) in [0,1]."""
    if x.ndim == 3:
        arr = x.detach().cpu().float().numpy()
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        else:
            arr = np.transpose(arr, (1, 2, 0))
    elif x.ndim == 2:
        arr = x.detach().cpu().float().numpy()
        arr = np.stack([arr]*3, axis=-1)
    else:
        raise ValueError('unexpected image ndim')
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def colorize_indices(idx: np.ndarray, palette: Dict[int, Tuple[float, float, float]]) -> np.ndarray:
    h, w = idx.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for k, rgb in palette.items():
        mask = (idx == k)
        if mask.any():
            out[mask] = np.array(rgb, dtype=np.float32)
    return out


def blend_overlay(img01: np.ndarray, overlay01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = (1 - alpha) * img01 + alpha * overlay01
    return np.clip(out, 0.0, 1.0)


def draw_yellow_boundaries(overlay_rgb: np.ndarray,
                           inst_labels: np.ndarray,
                           alpha: float = 0.95,
                           thickness: int = 1) -> np.ndarray:
    """
    Draw ONLY boundaries in yellow on top of an already colorized overlay.
    Keeps the underlying class colors; adds thin yellow strokes on boundaries.
    """
    out = overlay_rgb.copy()
    bnd = segmentation.find_boundaries(inst_labels, mode='outer')
    if thickness > 1:
        bnd = ndi.binary_dilation(bnd, structure=square(thickness))
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    out[bnd] = (1 - alpha) * out[bnd] + alpha * yellow
    return np.clip(out, 0.0, 1.0)


def add_legend(fig, ax, class_names: Dict[int, str], palette: Dict[int, Tuple[float, float, float]]):
    import matplotlib.patches as mpatches
    patches = []
    for cid in sorted(class_names.keys()):
        name = class_names[cid]
        color = palette.get(cid, (0.3, 0.3, 0.3))
        patches.append(mpatches.Patch(color=color, label=f"{cid}: {name}"))
    ax.axis('off')
    ax.set_title("Legend")
    ax.legend(handles=patches, loc='center')


# -----------------------------
# Model output pickers
# -----------------------------
def _first_present(y: dict, keys: List[str]):
    for k in keys:
        if k in y and y[k] is not None:
            return y[k]
    return None


def extract_np_hv_nt(y: dict):
    """
    Returns:
      np_like:  (B,1,H,W) or (B,2,H,W) or (B,H,W)  (logits or prob)
      hv_like:  (B,2,H,W) or (B,H,W,2)
      nt_like:  (B,C,H,W) logits or (B,H,W) indices
    Any of them can be None if not present.
    """
    np_like = _first_present(y, [
        'nuclei_binary_map', 'bin_logits', 'np', 'np_map', 'binary_map', 'np_logits', 'nuclei_pred'
    ])
    hv_like = _first_present(y, ['hv_map', 'hv', 'hover', 'horizontal_vertical', 'dist_map'])
    nt_like = _first_present(y, ['nuclei_type_map', 'type_logits', 'type', 'np_type', 'type_map_pred'])
    return np_like, hv_like, nt_like


# -----------------------------
# HoVer-style postprocessing
# -----------------------------
def hovernet_postprocess(np_like: torch.Tensor,
                         hv_like: torch.Tensor,
                         np_thresh: float = 0.4,
                         min_size: int = 10) -> np.ndarray:
    """
    Args:
      np_like: (H,W) or (1,H,W) or (2,H,W) tensor (logits or prob). If 2ch, assumes bg/fg logits.
      hv_like: (2,H,W) or (H,W,2)
    Returns:
      inst_labels: (H,W) int32 labeled instances
    """
    # ---- NP to prob (H,W) ----
    np_arr = np_like.detach().cpu().float().numpy()
    if np_arr.ndim == 3 and np_arr.shape[0] == 1:
        np_arr = np_arr[0]
    elif np_arr.ndim == 3 and np_arr.shape[0] == 2:
        # bg/fg 2ch -> softmax -> fg prob
        sm = np.exp(np_arr - np_arr.max(axis=0, keepdims=True))
        sm = sm / np.clip(sm.sum(axis=0, keepdims=True), 1e-6, None)
        np_arr = sm[1]  # foreground prob
    elif np_arr.ndim != 2:
        raise ValueError(f"Unexpected NP shape: {np_arr.shape}")
    # logits vs prob
    if np_arr.max() > 1.0 or np_arr.min() < 0.0:
        np_prob = sigmoid(np_arr)
    else:
        np_prob = np_arr

    # ---- HV to (2,H,W) ----
    hv_arr = hv_like.detach().cpu().float().numpy()
    if hv_arr.ndim == 3 and hv_arr.shape[0] == 2:
        hv_x, hv_y = hv_arr[0], hv_arr[1]
    elif hv_arr.ndim == 3 and hv_arr.shape[2] == 2:
        hv_x, hv_y = hv_arr[..., 0], hv_arr[..., 1]
    else:
        raise ValueError(f"Unexpected HV shape: {hv_arr.shape}")

    H, W = np_prob.shape

    # ---- foreground mask from NP ----
    nuclei_bin = (np_prob > np_thresh)
    nuclei_bin = morphology.remove_small_holes(nuclei_bin, area_threshold=32)
    nuclei_bin = morphology.remove_small_objects(nuclei_bin, min_size=min_size)

    if not np.any(nuclei_bin):
        return np.zeros((H, W), np.int32)

    # ---- HV gradients -> edge-like energy ----
    gx = filters.sobel(hv_x)
    gy = filters.sobel(hv_y)
    hv_edge = np.hypot(gx, gy)
    energy = hv_edge + (1.0 - np_prob)

    # ---- markers via distance peaks ----
    dist = ndi.distance_transform_edt(nuclei_bin)
    dist_s = ndi.gaussian_filter(dist, sigma=1.0)
    peaks = feature.peak_local_max(dist_s, labels=nuclei_bin, footprint=np.ones((3, 3)), exclude_border=False)
    markers = np.zeros((H, W), dtype=np.int32)
    if peaks.size > 0:
        markers[tuple(peaks.T)] = 1
    markers = morphology.label(markers, connectivity=1)
    if markers.max() == 0:
        markers = measure.label(nuclei_bin, connectivity=1)

    # ---- watershed ----
    inst_labels = segmentation.watershed(energy, markers=markers, mask=nuclei_bin)
    inst_labels = inst_labels.astype(np.int32)

    # remove tiny segments
    lbl = inst_labels.copy()
    for r in measure.regionprops(lbl):
        if r.area < min_size:
            lbl[lbl == r.label] = 0
    inst_labels = measure.label(lbl > 0).astype(np.int32)
    return inst_labels


# -----------------------------
# Instance type assignment (soft vote on core)
# -----------------------------
def instance_types_softvote(inst_labels: np.ndarray,
                            type_logits_or_idx: np.ndarray,
                            num_classes: int,
                            bg_index: int = 0,
                            use_core: bool = True,
                            core_erode_px: int = 2) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Return:
      inst_id_to_type: {id: class}
      panoptic_type_map: (H,W) with a single class id per instance area
    """
    H, W = inst_labels.shape
    # to probs (C,H,W)
    if type_logits_or_idx.ndim == 3:  # (C,H,W) logits
        tl = type_logits_or_idx - type_logits_or_idx.max(axis=0, keepdims=True)
        exp = np.exp(tl)
        prob = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-6, None)
    elif type_logits_or_idx.ndim == 2:  # (H,W) indices
        idx = type_logits_or_idx
        prob = np.zeros((num_classes, H, W), dtype=np.float32)
        for c in range(num_classes):
            prob[c] = (idx == c).astype(np.float32)
    else:
        raise ValueError("type_logits_or_idx must be (C,H,W) or (H,W)")

    if 0 <= bg_index < num_classes:
        prob[bg_index] = 0.0  # ignore background in voting

    selem = disk(max(1, core_erode_px)) if use_core and core_erode_px > 0 else None

    inst_ids = [i for i in np.unique(inst_labels) if i != 0]
    panoptic = np.zeros((H, W), dtype=np.int32)
    inst2type = {}

    for i_id in inst_ids:
        mask = (inst_labels == i_id)
        if mask.sum() == 0:
            continue
        core = morphology.erosion(mask, selem) if selem is not None else mask
        if core.sum() == 0:
            core = mask
        cls_scores = prob[:, core].mean(axis=1)  # (C,)
        cls = int(np.argmax(cls_scores)) if cls_scores.sum() > 0 else bg_index
        inst2type[i_id] = cls
        panoptic[mask] = cls

    return inst2type, panoptic


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Panoptic visualization on an image folder (Raw | Panoptic overlay)")
    ap.add_argument('--ckpt', type=Path, required=True, help='trained checkpoint (.pth)')
    ap.add_argument('--input_dir', type=Path, required=True, help='directory containing images')
    ap.add_argument('--out', type=Path, default=Path('infer_out_panoptic'))
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)

    # Model hyperparams (must match training!)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--patch_size', type=int, default=16)
    ap.add_argument('--vit_embed_dim', type=int, default=384)
    ap.add_argument('--vit_depth', type=int, default=12)
    ap.add_argument('--vit_heads', type=int, default=6)
    ap.add_argument('--vit_mlp_ratio', type=float, default=4.0)
    ap.add_argument('--num_classes', type=int, default=5, help='incl. background (consep_merged=5)')

    # Postproc + viz
    ap.add_argument('--np_thresh', type=float, default=0.4, help='nuclei prob threshold for NP mask')
    ap.add_argument('--min_size', type=int, default=10, help='min instance area')
    ap.add_argument('--core_erode_px', type=int, default=2, help='pixels to erode for core voting')
    ap.add_argument('--alpha', type=float, default=0.45, help='type overlay alpha')
    ap.add_argument('--bnd_thickness', type=int, default=1, help='yellow boundary thickness (px)')
    ap.add_argument('--bnd_alpha', type=float, default=0.95, help='yellow boundary alpha [0..1]')
    ap.add_argument('--dpi', type=int, default=150)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset / loader
    ds = ImageFolderInferenceDataset(args.input_dir, img_size=args.img_size, exts=IMG_EXTS)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)

    # Model
    model = CellViTCustom(
        num_nuclei_classes=args.num_classes,
        num_tissue_classes=0,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_ratio=args.vit_mlp_ratio,
    )

    # Robust state_dict extraction
    blob = torch.load(str(args.ckpt), map_location='cpu')
    sd = blob
    if isinstance(blob, dict):
        for k in ['model', 'model_state_dict', 'state_dict']:
            if k in blob and isinstance(blob[k], dict):
                sd = blob[k]
                print(f"[init] picked state_dict from key: '{k}'")
                break
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        print("[init] stripped 'module.' prefix")
    _ = model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # Palette + class names (consep_merged)
    palette = {
        0: (0.50, 0.50, 0.50),   # bg (gray)
        1: (0.21, 0.49, 0.00),   # other
        2: (1.00, 0.55, 0.00),   # inflammatory
        3: (0.27, 0.56, 0.96),   # epithelial (3+4)
        4: (0.91, 0.00, 0.91),   # spindle (5+6+7)
    }
    class_names = {
        0: "bg",
        1: "other",
        2: "inflammatory",
        3: "epithelial (3+4)",
        4: "spindle (5+6+7)",
    }

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = torch.cuda.is_available()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        for batch in tqdm(dl, ncols=100, desc='Infer'):
            x = batch['image'].to(device, non_blocking=True)  # (B,3,H,W), [0,1]
            paths = batch['path']

            y = model(x)
            np_like, hv_like, nt_like = extract_np_hv_nt(y)
            if np_like is None or hv_like is None:
                warnings.warn("NP/HV heads not found in model output. Panoptic instances will be skipped.")
            if nt_like is None:
                warnings.warn("Type head not found; per-instance typing will be skipped.")

            # Prepare type field to (B,C,H,W) logits or (B,H,W) indices
            # We'll pass per-sample slice later.
            B = x.size(0)
            for b in range(B):
                img01 = to_hwc01(x[b])

                inst_labels = None
                if np_like is not None and hv_like is not None:
                    # shape normalization for per-sample tensors
                    np_b = np_like[b]
                    hv_b = hv_like[b]
                    # if np_b is (C,H,W) ensure it's 1 or 2ch; if (H,W) it's fine
                    inst_labels = hovernet_postprocess(np_b, hv_b,
                                                       np_thresh=args.np_thresh,
                                                       min_size=args.min_size)

                # Build panoptic type map (fallback to pixel argmax if inst unavailable)
                panoptic_type = None
                # extract per-sample type head
                t_b = None
                if nt_like is not None:
                    t_b = nt_like[b].detach().cpu().numpy()
                    # Expect (C,H,W) logits or (H,W) indices; keep as-is

                if inst_labels is not None and t_b is not None:
                    _, panoptic_type = instance_types_softvote(
                        inst_labels,
                        t_b,
                        num_classes=args.num_classes,
                        bg_index=0,
                        use_core=True,
                        core_erode_px=args.core_erode_px
                    )
                elif t_b is not None:
                    # fallback: pixel-wise classes only
                    if t_b.ndim == 3:
                        panoptic_type = t_b.argmax(axis=0).astype(np.int32)
                    else:
                        panoptic_type = t_b.astype(np.int32)

                # Colorize + overlay
                if panoptic_type is not None:
                    pan_rgb = colorize_indices(panoptic_type, palette)
                    overlay = blend_overlay(img01, pan_rgb, alpha=args.alpha)
                else:
                    # nothing to colorize
                    overlay = img01.copy()

                # Add yellow boundaries (if we have instances)
                if inst_labels is not None:
                    overlay = draw_yellow_boundaries(
                        overlay, inst_labels,
                        alpha=args.bnd_alpha,
                        thickness=args.bnd_thickness
                    )

                # Compose figure: Raw | Panoptic(+boundaries) | Legend
                fig = plt.figure(figsize=(15, 7.5))
                gs = fig.add_gridspec(1, 3, wspace=0.02, width_ratios=[1.0, 1.0, 0.6])
                ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img01);   ax0.set_title('Raw'); ax0.axis('off')
                ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(overlay); ax1.set_title('Panoptic (types + yellow boundaries)'); ax1.axis('off')
                ax2 = fig.add_subplot(gs[0, 2]); add_legend(fig, ax2, class_names, palette)

                stem = Path(paths[b]).stem
                out_path = out_dir / f"{stem}_panoptic.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
                plt.close(fig)

    print(f"[DONE] Saved visualizations to: {out_dir}")
    

if __name__ == '__main__':
    main()
