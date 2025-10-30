# -*- coding: utf-8 -*-
"""
Inference-only visualization for CellViT (ViT-256) on arbitrary image folders.

- Loads a trained checkpoint
- Iterates over images from a directory (no GT required)
- Saves a 3-panel figure: [Raw | Type overlay (+ yellow instance boundaries) | Legend]

Assumptions:
- Input is a directory containing .png/.jpg/.jpeg/.tif/.tiff/.bmp
- Each image will be resized to (--img_size, --img_size) before inference
- Model forward returns a dict that includes heads under common keys:
  * Type head: ['nuclei_type_map','type_logits','type','np_type','type_map_pred']
  * Binary/NP head (optional but needed for instances): 
      ['np','np_map','nuclei','nuclei_map','nuclei_binary_map','binary','np_logits','nuclei_pred']
  * HV head (optional but needed for instances):
      ['hv','hv_map','hover','horizontal_vertical','hv_logits','dist_map','distance_map']

Example:
    python infer_viz_images.py \
      --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
      --input_dir path/to/images \
      --out infer_out_images \
      --batch_size 8 --device cuda
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# postproc deps
from scipy.special import expit as sigmoid
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, feature

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


# -----------------------------
# Model-output pickers
# -----------------------------
def _first_present(y: dict, keys: List[str]):
    for k in keys:
        if k in y and y[k] is not None:
            return y[k]
    return None


def extract_type_like(y: dict):
    """Return (B,C,H,W) logits or (B,H,W) indices for nuclei type."""
    t = _first_present(y, ['nuclei_type_map', 'type_logits', 'type', 'np_type', 'type_map_pred'])
    if t is None:
        raise KeyError(f"Type head not found in model output. Got keys={list(y.keys())}")
    return t


def extract_np_hv_nt(y: dict):
    """
    Returns:
      np_like:  (B,1|2,H,W) or (B,H,W)
      hv_like:  (B,2,H,W) or (B,H,W,2)
      nt_like:  (B,C,H,W) or (B,H,W) or None
    """
    np_like = _first_present(y, [
        'np', 'np_map', 'nuclei', 'nuclei_map', 'nuclei_binary_map', 'binary', 'np_logits', 'nuclei_pred'
    ])
    hv_like = _first_present(y, [
        'hv', 'hv_map', 'hover', 'horizontal_vertical', 'hv_logits', 'dist_map', 'distance_map'
    ])
    nt_like = _first_present(y, [
        'nuclei_type_map', 'type_logits', 'type', 'np_type', 'type_map_pred'
    ])
    return np_like, hv_like, nt_like


# -----------------------------
# HoVer-style postprocessing
# -----------------------------
def hovernet_postprocess(np_like: torch.Tensor,
                         hv_like: torch.Tensor,
                         nt_like: torch.Tensor = None,
                         np_thresh: float = 0.4,
                         min_size: int = 10) -> np.ndarray:
    """
    Args:
      np_like:  (1,H,W) or (H,W) tensor (prob or logits)
      hv_like:  (2,H,W) or (H,W,2)
    Returns:
      inst_labels: (H,W) int32 labeled instances
    """
    # --- to numpy ---
    np_arr = np_like.detach().cpu().float().numpy()
    if np_arr.ndim == 3 and np_arr.shape[0] in (1, 2):
        # (1,H,W) or (2,H,W) -> take fg channel if 2
        if np_arr.shape[0] == 2:
            # assume [bg, fg]
            np_arr = np_arr[1]
        else:
            np_arr = np_arr[0]
    # logits -> prob
    np_prob = sigmoid(np_arr) if (np_arr.max() > 1.0 or np_arr.min() < 0.0) else np_arr

    hv_arr = hv_like.detach().cpu().float().numpy()
    if hv_arr.ndim == 3 and hv_arr.shape[0] == 2:
        hv_x, hv_y = hv_arr[0], hv_arr[1]
    elif hv_arr.ndim == 3 and hv_arr.shape[-1] == 2:
        hv_x, hv_y = hv_arr[..., 0], hv_arr[..., 1]
    else:
        raise ValueError("HV map must have 2 channels")

    H, W = np_prob.shape

    # --- binary mask from NP ---
    nuclei_bin = (np_prob > np_thresh)
    nuclei_bin = morphology.remove_small_objects(nuclei_bin, min_size=min_size)

    if not nuclei_bin.any():
        return np.zeros((H, W), np.int32)

    # --- edge from HV gradients (Sobel on HV) ---
    gx = filters.sobel(hv_x)
    gy = filters.sobel(hv_y)
    hv_edge = np.hypot(gx, gy)

    # 에너지 맵: 경계 강조 + 내부 억제
    energy = hv_edge + (1.0 - np_prob)

    # --- markers (distance peaks) ---
    dist = ndi.distance_transform_edt(nuclei_bin)
    local_max = feature.peak_local_max(
        dist, labels=nuclei_bin, footprint=np.ones((3,3)), exclude_border=False
    )
    markers = np.zeros((H, W), dtype=np.int32)
    if local_max.size > 0:
        markers[tuple(local_max.T)] = 1
    markers = morphology.label(markers, connectivity=1)
    if markers.max() == 0:
        markers = measure.label(nuclei_bin, connectivity=1)

    # --- watershed ---
    inst_labels = segmentation.watershed(energy, markers=markers, mask=nuclei_bin)
    inst_labels = inst_labels.astype(np.int32)

    # remove tiny
    for r in measure.regionprops(inst_labels):
        if r.area < min_size:
            inst_labels[inst_labels == r.label] = 0
    inst_labels = measure.label(inst_labels > 0)
    return inst_labels.astype(np.int32)




def overlay_boundaries(img01: np.ndarray,
                       inst_labels: np.ndarray,
                       color=(1.0, 1.0, 0.0),  # bright yellow
                       alpha: float = 1.0,     # 1.0이면 경계 픽셀을 완전히 노란색으로
                       thickness: int = 2) -> np.ndarray:
    """
    img01: (H,W,3) float [0,1]  ← '타입 오버레이'된 이미지를 넣으세요
    inst_labels: (H,W) int32     ← 인스턴스 라벨 맵
    color: 노란색 경계 색상 (RGB in [0,1])
    alpha: 경계 픽셀에서만 적용할 블렌드 비율 (1.0 = 완전 노란색)
    thickness: 경계 두께 (픽셀)
    """
    out = img01.copy()
    if inst_labels is None or inst_labels.max() == 0:
        return out

    # 경계 마스크 (outer)
    bnd = segmentation.find_boundaries(inst_labels, mode='outer')
    if thickness > 1:
        bnd = morphology.dilation(bnd, morphology.disk(max(1, thickness // 2)))

    # 경계 픽셀에만 색을 칠하거나(blend) 치환(replace)
    if alpha >= 1.0:
        out[bnd] = np.array(color, dtype=np.float32)
    else:
        out[bnd] = (1.0 - alpha) * out[bnd] + alpha * np.array(color, dtype=np.float32)

    return np.clip(out, 0.0, 1.0)



def build_legend_handles(palette: Dict[int, Tuple[float, float, float]],
                         class_names: Dict[int, str],
                         boundary_color=(1.0, 1.0, 0.0)):
    """Return legend handles: class color patches + instance boundary line."""
    patches = []
    for cid in sorted(class_names.keys()):
        if cid == 0:
            continue
        rgb = palette.get(cid, (0.6, 0.6, 0.6))
        patches.append(mpatches.Patch(facecolor=rgb, edgecolor='black', label=f"{cid}: {class_names[cid]}"))
    line = Line2D([0], [0], color=boundary_color, lw=3, label="Instance boundary")
    return patches + [line]


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Inference-only visualization on an image folder (Raw | Pred overlay | Legend)")
    ap.add_argument('--ckpt', type=Path, required=True, help='trained checkpoint (.pth)')
    ap.add_argument('--input_dir', type=Path, required=True, help='directory containing images')
    ap.add_argument('--out', type=Path, default=Path('infer_out_images'))
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

    # Postproc hyperparams for instances
    ap.add_argument('--np_thresh', type=float, default=0.4, help='threshold for nuclei prob to form FG')
    ap.add_argument('--min_size', type=int, default=10, help='min instance area')
    ap.add_argument('--bnd_thickness', type=int, default=1, help='boundary thickness (pixels)')
    ap.add_argument('--bnd_alpha', type=float, default=0.95, help='boundary overlay alpha')

    # Viz
    ap.add_argument('--alpha', type=float, default=0.45, help='type overlay alpha')
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--class_names', type=str,
                    default='bg,other,inflammatory,epithelial (3+4),spindle (5+6+7)',
                    help='comma-separated names; index 0 is background')

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
    if isinstance(sd, dict) and any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        print("[init] stripped 'module.' prefix")
    _ = model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # Palette (0..4) for consep_merged-like heads
    palette = {
        0: (0.50, 0.50, 0.50),   # bg (gray)
        1: (0.21, 0.49, 0.00),   # other
        2: (1.00, 0.55, 0.00),   # inflammatory
        3: (0.27, 0.56, 0.96),   # epithelial (3+4)
        4: (0.91, 0.00, 0.91),   # spindle (5+6+7)
    }
    # class names
    raw_names = [s.strip() for s in str(args.class_names).split(',')]
    class_names = {i: (raw_names[i] if i < len(raw_names) else f"class{i}") for i in range(args.num_classes)}

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = torch.cuda.is_available()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        for batch in tqdm(dl, ncols=100, desc='Infer'):
            x = batch['image'].to(device, non_blocking=True)  # (B,3,H,W), [0,1]
            paths = batch['path']

            y = model(x)
            # type logits/indices
            type_like = extract_type_like(y)
            # optional heads for instances
            np_like, hv_like, nt_like = extract_np_hv_nt(y)

            # Convert type to indices (B,H,W)
            if type_like.ndim == 4:      # (B,C,H,W) logits
                pred_idx_all = type_like.argmax(dim=1)
            elif type_like.ndim == 3:    # already indices
                pred_idx_all = type_like
            else:
                raise ValueError(f"Unexpected type_like shape: {tuple(type_like.shape)}")

            B = x.size(0)
            for b in range(B):
                img01 = to_hwc01(x[b])
                pred_idx = pred_idx_all[b].detach().cpu().numpy().astype(np.int32)
                pred_rgb = colorize_indices(pred_idx, palette)
                overlay_type = blend_overlay(img01, pred_rgb, alpha=args.alpha)

                # instances (if np/hv available)
                overlay_inst = overlay_type
                if (np_like is not None) and (hv_like is not None):
                    try:
                        np_b = np_like[b]
                        hv_b = hv_like[b]
                        inst_labels = hovernet_postprocess(
                            np_b, hv_b, nt_like=nt_like[b] if nt_like is not None else None,
                            np_thresh=args.np_thresh, min_size=args.min_size
                        )
                        overlay_inst = overlay_boundaries(
                            overlay_type,        # ← 타입 컬러가 입혀진 이미지를 넣음
                            inst_labels,
                            color=(1.0, 1.0, 0.0),   # 샛노랑
                            alpha=args.bnd_alpha,    # 1.0이면 경계 픽셀은 완전 노랑, 0.6이면 부분 블렌드
                            thickness=args.bnd_thickness
                        )
                    except Exception as e:
                        warnings.warn(f"[postproc] failed on {paths[b]}: {e}")

                # Compose figure: Raw | Overlay(+boundaries) | Legend
                fig = plt.figure(figsize=(15, 7.5))
                gs = fig.add_gridspec(1, 3, wspace=0.02, width_ratios=[1.0, 1.0, 0.6])

                ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img01);       ax0.set_title('Raw'); ax0.axis('off')
                ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(overlay_inst); ax1.set_title('Prediction'); ax1.axis('off')

                ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off'); ax2.set_title('Legend')
                handles = build_legend_handles(palette, class_names, boundary_color=(1.0,1.0,0.0))
                ax2.legend(handles=handles, loc='center')

                stem = Path(paths[b]).stem
                out_path = out_dir / f"{stem}_raw_pred.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
                plt.close(fig)

    print(f"[DONE] Saved visualizations to: {out_dir}")


if __name__ == '__main__':
    main()
