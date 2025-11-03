# -*- coding: utf-8 -*-
"""
Panoptic visualization + (optional) metrics for CellViT (ViT-256)
with HV gradient visualization and improved instance separation controls.

Panels:
[Raw | Pred Panoptic(+yellow)]
+ Optional: HV gradient / Energy / Distance / Markers debug maps.

Example:
python infer_viz_images_panoptic.py \
    --dataset consep \
    --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
    --consep_root /workspace/CellViT_Custom/Dataset/CoNSeP/Preprocessed/Test \
    --out eval_out/CoNSeP_panoptic \
    --device cuda \
    --save_hv_grad \
    --np_thresh 0.5 \
    --alpha 2.5 --beta 1.0 \
    --peak_footprint 2 --peak_thresh_rel 0.10 \
    --dist_sigma 0.6 \
    --pre_erode_px 0 \
    --save_debug
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import csv
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
from skimage.transform import resize as sk_resize

# === local import ===
from Model.CellViT_ViT256_Custom import CellViTCustom

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ============================ CoNSeP type merge ============================
def consep_merge_type_ids(type_map: np.ndarray) -> np.ndarray:
    remap = np.zeros(8, dtype=np.int32)
    remap[1] = 1
    remap[2] = 2
    remap[3] = 3; remap[4] = 3
    remap[5] = 4; remap[6] = 4; remap[7] = 4
    return remap.take(type_map.astype(np.int32), mode='clip')

def sanitize_type_map(idx: np.ndarray, num_classes: int, palette: Dict[int, Tuple[float,float,float]], try_merge=True) -> np.ndarray:
    out = idx.astype(np.int32, copy=False)
    if try_merge and out.max() > (num_classes - 1):
        out = consep_merge_type_ids(out)
    valid = np.array(sorted(palette.keys()), dtype=np.int32)
    mask = ~np.isin(out, valid)
    if mask.any():
        out = out.copy(); out[mask] = 0
    return out


# ============================ Dataset ============================
class ConsepNPZDataset(Dataset):
    def __init__(self, root: Path, img_size: int = 256):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.lab_dir = self.root / "labels"
        self.img_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if not self.img_files:
            raise FileNotFoundError(f"No images under {self.img_dir}")
        self.lab_files = [self.lab_dir / f"{ip.stem}.npz" for ip in self.img_files]
        for lp in self.lab_files:
            if not lp.exists():
                raise FileNotFoundError(f"Missing label npz: {lp}")
        self.img_size = img_size

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx: int):
        ip, lp = self.img_files[idx], self.lab_files[idx]
        im = Image.open(ip).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))

        data = np.load(lp)
        inst, typ, hv, bmap = data["inst_map"], data["type_map"], data["hv_map"], data["bin_map"]

        def _resize(x, order):
            return sk_resize(x, (self.img_size, self.img_size), order=order,
                             preserve_range=True, anti_aliasing=(order>0)).astype(x.dtype)

        inst_r = _resize(inst, 0); typ_r = _resize(typ, 0)
        if hv.ndim == 3 and hv.shape[-1] == 2:
            hv_r = np.stack([_resize(hv[...,0],1), _resize(hv[...,1],1)], axis=0)
        elif hv.ndim == 3 and hv.shape[0] == 2:
            hv_r = np.stack([_resize(hv[0],1), _resize(hv[1],1)], axis=0)
        else:
            raise ValueError(f"Unexpected hv_map shape: {hv.shape}")
        bin_r = _resize(bmap if bmap.ndim==2 else bmap[..., -1], 1)[None,...]

        return {
            "image": torch.from_numpy(chw),
            "path": str(ip),
            "gt_inst": torch.from_numpy(inst_r),
            "gt_type": torch.from_numpy(typ_r),
            "gt_hv": torch.from_numpy(hv_r),
            "gt_bin": torch.from_numpy(bin_r),
        }


# ============================ Utils ============================
def to_hwc01(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1,3):
        arr = np.transpose(arr, (1,2,0))
    return np.clip(arr, 0, 1)

def draw_boundaries(overlay_rgb: np.ndarray, inst_labels: np.ndarray,
                    color=(1,1,0), alpha: float = 0.95) -> np.ndarray:
    if inst_labels is None or inst_labels.max()==0: return overlay_rgb
    bnd = segmentation.find_boundaries(inst_labels, mode='outer')
    overlay_rgb[bnd] = (1-alpha)*overlay_rgb[bnd] + alpha*np.array(color)
    return overlay_rgb

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
    """
    np_like = _first_present(y, [
        'nuclei_binary_map','bin_logits','np','np_map','binary_map','np_logits','nuclei_pred'
    ])
    hv_like = _first_present(y, ['hv_map','hv','hover','horizontal_vertical','dist_map'])
    nt_like = _first_present(y, ['nuclei_type_map','type_logits','type','np_type','type_map_pred'])
    return np_like, hv_like, nt_like


# ============================ HV-based postprocess (Improved) ============================
def hovernet_postprocess(
    np_like: torch.Tensor,
    hv_like: torch.Tensor,
    np_thresh: float = 0.5,
    min_size: int = 10,
    alpha: float = 2.5,           # weight for hv_edge
    beta: float  = 1.0,           # weight for (1 - np_prob)
    dist_sigma: float = 0.6,      # smoothing on distance map
    peak_footprint: int = 2,      # seed footprint (NxN)
    peak_thresh_rel: Optional[float] = 0.10,  # relative threshold for peaks
    peak_thresh_abs: Optional[float] = None,  # absolute threshold (optional)
    pre_erode_px: int = 0,        # optional pre-erosion to break thin bridges
    return_grad: bool = False,
    return_debug: bool = False,
):
    """
    Returns:
      inst_label  (H,W) int32
      hv_grad01   (H,W) float32 in [0,1]                (if return_grad)
      debug_dict  {'hv_edge','energy','dist','markers'} (if return_debug)
    """
    # ---- NP ----
    np_arr = np_like.detach().cpu().float().numpy()
    if np_arr.ndim == 3 and np_arr.shape[0] in (1,2):
        np_arr = np_arr[-1]
    np_prob = sigmoid(np_arr) if (np_arr.max()>1 or np_arr.min()<0) else np_arr
    H, W = np_prob.shape

    # ---- HV ----
    hv_arr = hv_like.detach().cpu().float().numpy()
    if hv_arr.ndim == 3 and hv_arr.shape[0] == 2:
        hv_x, hv_y = hv_arr[0], hv_arr[1]
    elif hv_arr.ndim == 3 and hv_arr.shape[-1] == 2:
        hv_x, hv_y = hv_arr[...,0], hv_arr[...,1]
    else:
        raise ValueError(f"HV map must have 2 channels; got {hv_arr.shape}")

    # ---- Binary nuclei mask + optional erosion to split isthmuses ----
    nuclei_bin = (np_prob > np_thresh)
    nuclei_bin = morphology.remove_small_holes(nuclei_bin, area_threshold=32)
    nuclei_bin = morphology.remove_small_objects(nuclei_bin, min_size=min_size)
    if pre_erode_px > 0:
        se = morphology.disk(pre_erode_px)
        nuclei_bin = morphology.erosion(nuclei_bin, se)

    if not nuclei_bin.any():
        empty = np.zeros((H,W), np.int32)
        if return_grad and return_debug:
            return empty, np.zeros((H,W), np.float32), {'hv_edge':np.zeros((H,W),np.float32),
                                                        'energy':np.zeros((H,W),np.float32),
                                                        'dist':np.zeros((H,W),np.float32),
                                                        'markers':empty}
        if return_grad:  return empty, np.zeros((H,W), np.float32)
        if return_debug: return empty, {'hv_edge':np.zeros((H,W),np.float32),
                                        'energy':np.zeros((H,W),np.float32),
                                        'dist':np.zeros((H,W),np.float32),
                                        'markers':empty}
        return empty

    # ---- HV gradient and energy ----
    gx, gy = filters.sobel(hv_x), filters.sobel(hv_y)
    hv_edge = np.hypot(gx, gy)

    # normalize hv_edge for stability
    if hv_edge.max() > 0:
        hv_edge = (hv_edge - hv_edge.min()) / (hv_edge.ptp() + 1e-6)

    energy = alpha * hv_edge + beta * (1.0 - np_prob)

    # ---- Distance & seeds ----
    dist = ndi.distance_transform_edt(nuclei_bin)
    dist_s = ndi.gaussian_filter(dist, sigma=max(0.0, float(dist_sigma)))

    # peak_local_max controls
    fp = int(max(1, peak_footprint))
    kwargs = dict(labels=nuclei_bin, footprint=np.ones((fp, fp), np.uint8))
    if peak_thresh_rel is not None:
        kwargs['threshold_rel'] = float(peak_thresh_rel)
    if peak_thresh_abs is not None:
        kwargs['threshold_abs'] = float(peak_thresh_abs)

    peaks = feature.peak_local_max(dist_s, **kwargs)
    markers = np.zeros((H,W), np.int32)
    if peaks.size > 0:
        markers[tuple(peaks.T)] = 1
    markers = morphology.label(markers)
    if markers.max() == 0:
        # fallback: one marker per connected component
        markers = measure.label(nuclei_bin)

    # ---- Watershed ----
    inst = segmentation.watershed(energy, markers=markers, mask=nuclei_bin)
    inst = measure.label(inst > 0).astype(np.int32)

    # ---- Returns ----
    grad01 = (hv_edge - hv_edge.min()) / (hv_edge.ptp() + 1e-6)
    if return_grad and return_debug:
        dbg = {'hv_edge': grad01.astype(np.float32),
               'energy': (energy - energy.min())/ (energy.ptp()+1e-6),
               'dist': (dist_s - dist_s.min())/ (dist_s.ptp()+1e-6),
               'markers': markers.astype(np.int32)}
        return inst, grad01.astype(np.float32), dbg
    if return_grad:
        return inst, grad01.astype(np.float32)
    if return_debug:
        dbg = {'hv_edge': grad01.astype(np.float32),
               'energy': (energy - energy.min())/ (energy.ptp()+1e-6),
               'dist': (dist_s - dist_s.min())/ (dist_s.ptp()+1e-6),
               'markers': markers.astype(np.int32)}
        return inst, dbg
    return inst


# ============================ Main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['consep'], required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--consep_root', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('infer_out_panoptic'))
    ap.add_argument('--device', type=str, default='cuda')

    # Visualization/debug
    ap.add_argument('--save_hv_grad', action='store_true', help='save predicted HV gradient map')
    ap.add_argument('--save_debug', action='store_true', help='save extra debug maps (energy/dist/markers)')

    # Key knobs
    ap.add_argument('--np_thresh', type=float, default=0.5)
    ap.add_argument('--min_size', type=int, default=10)
    ap.add_argument('--alpha', type=float, default=2.5)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--dist_sigma', type=float, default=0.6)
    ap.add_argument('--peak_footprint', type=int, default=2)
    ap.add_argument('--peak_thresh_rel', type=float, default=0.10)
    ap.add_argument('--peak_thresh_abs', type=float, default=None)
    ap.add_argument('--pre_erode_px', type=int, default=0)

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ds = ConsepNPZDataset(args.consep_root)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = CellViTCustom(num_nuclei_classes=5, num_tissue_classes=0, img_size=256, patch_size=16)
    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('model', sd)
    if isinstance(sd, dict) and any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dl, ncols=100, desc='Infer'):
            img = batch['image'].to(device)
            y = model(img)
            np_like, hv_like, _ = extract_np_hv_nt(y)

            inst, hv_grad, dbg = hovernet_postprocess(
                np_like[0], hv_like[0],
                np_thresh=args.np_thresh,
                min_size=args.min_size,
                alpha=args.alpha,
                beta=args.beta,
                dist_sigma=args.dist_sigma,
                peak_footprint=args.peak_footprint,
                peak_thresh_rel=args.peak_thresh_rel,
                peak_thresh_abs=args.peak_thresh_abs,
                pre_erode_px=args.pre_erode_px,
                return_grad=True,
                return_debug=args.save_debug
            )

            stem = Path(batch['path'][0]).stem
            # === save instance overlay ===
            img01 = to_hwc01(img[0])
            overlay = draw_boundaries(img01.copy(), inst)
            out_path = args.out / f"{stem}_panoptic.png"
            plt.imsave(out_path, overlay)

            # === save HV gradient map ===
            if args.save_hv_grad:
                grad_path = args.out / f"{stem}_hv_grad.png"
                plt.figure(figsize=(4,4))
                plt.imshow(hv_grad, cmap='inferno', vmin=0, vmax=1)
                plt.axis('off'); plt.title("Predicted HV Gradient Magnitude")
                plt.tight_layout(); plt.savefig(grad_path, dpi=150); plt.close()

            # === debug maps ===
            if args.save_debug and isinstance(dbg, dict):
                # energy
                e_path = args.out / f"{stem}_energy.png"
                plt.figure(figsize=(4,4)); plt.imshow(dbg['energy'], cmap='magma', vmin=0, vmax=1)
                plt.axis('off'); plt.title("Energy (α*|∇HV| + β*(1-NP))")
                plt.tight_layout(); plt.savefig(e_path, dpi=150); plt.close()

                # distance
                d_path = args.out / f"{stem}_dist.png"
                plt.figure(figsize=(4,4)); plt.imshow(dbg['dist'], cmap='viridis', vmin=0, vmax=1)
                plt.axis('off'); plt.title("Distance (smoothed)")
                plt.tight_layout(); plt.savefig(d_path, dpi=150); plt.close()

                # markers (overlay)
                m_path = args.out / f"{stem}_markers.png"
                mk = dbg['markers']
                plt.figure(figsize=(4,4)); plt.imshow(img01)
                yy, xx = np.nonzero(mk)
                if len(xx):
                    plt.scatter(xx, yy, s=6, c='red', marker='o')
                plt.axis('off'); plt.title("Watershed Markers")
                plt.tight_layout(); plt.savefig(m_path, dpi=150); plt.close()

    print(f"[Done] Saved outputs to: {args.out}")


if __name__ == "__main__":
    main()
