# -*- coding: utf-8 -*-
"""
Evaluate CellViT (ViT-256) on CoNSeP (merged) Test set and compute
Hover-Net style metrics: DICE, AJI, DQ, SQ, PQ.

Requirements:
    pip install scikit-image scipy

Example:
    python eval_consep_metrics.py \
      --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
      --test_root Dataset/CoNSeP/Preprocessed/Test \
      --batch_size 8 --device cuda

Notes:
    - This script expects the dataset batch to include a ground-truth instance
      map under one of the keys: 'inst_map' or 'instance_map'. If not present,
      the script will exit with an instruction to expose it from the dataset.
    - The model output is expected to be a dict with keys similar to
      {'bin_logits', 'hv_map', 'type_logits'}; adjust key mapping below if your
      code uses different names.
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

# repo-local imports (adjust if your paths differ)
from Model.CellViT_ViT256_Custom import CellViTCustom
from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged

from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed


# -------------------------------
# Utilities
# -------------------------------

def to_hwc01(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) or (H,W) tensor -> (H,W,3) in [0,1]."""
    if x.ndim == 3:  # C,H,W
        arr = x.detach().cpu().float().numpy()
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        else:
            # unknown channel first, try as HWC already
            arr = np.transpose(arr, (1, 2, 0))
    elif x.ndim == 2:
        arr = x.detach().cpu().float().numpy()
        arr = np.stack([arr]*3, axis=-1)
    else:
        raise ValueError('unexpected image ndim')
    # normalize to [0,1]
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def make_type_overlay(img01: np.ndarray, type_map: np.ndarray, class_names: dict, alpha: float = 0.45) -> np.ndarray:
    """Overlay a discrete type_map onto RGB image in [0,1]."""
    # fixed palette matching example figure
    palette = {
        0: (0.5, 0.5, 0.5),   # bg (unused, appears via alpha on gray)
        1: (0.21, 0.49, 0.00),# other (greenish)
        2: (1.00, 0.55, 0.00),# inflammatory (orange)
        3: (0.27, 0.56, 0.96),# epithelial (blue)
        4: (0.91, 0.00, 0.91),# spindle (magenta)
    }
    h, w = type_map.shape
    color = np.zeros_like(img01)
    for k, rgb in palette.items():
        mask = (type_map == k)
        if not np.any(mask):
            continue
        color[mask] = rgb
    overlay = (1 - alpha) * img01 + alpha * color
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def save_viz(fig_path: Path, img01: np.ndarray, gt_type: np.ndarray, pr_type: np.ndarray, class_names: dict):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    gt_overlay = make_type_overlay(img01, gt_type, class_names)
    pr_overlay = make_type_overlay(img01, pr_type, class_names)

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img01); ax0.set_title('Input'); ax0.axis('off')
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(gt_overlay); ax1.set_title('GT Type'); ax1.axis('off')
    ax2 = fig.add_subplot(gs[1, 0]); ax2.imshow(pr_overlay); ax2.set_title('Pred Type'); ax2.axis('off')
    ax3 = fig.add_subplot(gs[1, 1]); ax3.axis('off'); ax3.set_title('Legend')

    patches = []
    for cid in sorted(class_names.keys()):
        if cid == 0:
            continue
        name = f"{cid}: {class_names[cid]}"
        # same palette as overlay
        color_map = {1:(0.21,0.49,0.00), 2:(1.00,0.55,0.00), 3:(0.27,0.56,0.96), 4:(0.91,0.00,0.91)}
        c = color_map.get(cid, (0.2,0.2,0.2))
        patches.append(mpatches.Patch(color=c, label=name))
    leg = ax3.legend(handles=patches, title='Cell types', loc='center')

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _first_present(y, keys):
    for k in keys:
        if k in y and y[k] is not None:
            return y[k]
    return None


def extract_pred_maps(y: dict):
    """Pick model outputs by tolerant key search.
    Returns (bin_like, hv_map) where:
        bin_like: (B,1|2,H,W) or (B,H,W)
        hv_map:   (B,2,H,W) or (B,H,W,2)
    """
    bin_like = _first_present(y, [
        'bin_logits', 'nuclei_binary_map', 'binary_map', 'np_bin', 'bin'
    ])
    hv_map = _first_present(y, ['hv_map', 'hv', 'np_hv'])

    if bin_like is None or hv_map is None:
        raise KeyError(
            f"Could not find required heads in model output. Got keys={list(y.keys())}.\n"
            "Expected one of ['bin_logits','nuclei_binary_map','binary_map','np_bin','bin'] "
            "and one of ['hv_map','hv','np_hv']."
        )
    return bin_like, hv_map


def postproc_to_instances(bin_prob: np.ndarray, hv_pred: np.ndarray,
                          bin_thresh: float = 0.5,
                          min_area: int = 10) -> np.ndarray:
    """Lightweight instance post-processing using watershed.

    Args:
        bin_prob: (H, W) predicted nuclei probability (after sigmoid)
        hv_pred:  (H, W, 2) predicted HV maps
        bin_thresh: threshold for nuclear region
        min_area: remove instances smaller than this

    Returns:
        inst_map: (H, W) labeled instance mask (0=background, 1..N instances)
    """
    H, W = bin_prob.shape

    # Smooth prob for stability
    p = gaussian_filter(bin_prob, sigma=1.0)

    # foreground mask
    fg = p >= bin_thresh
    fg = remove_small_holes(fg, area_threshold=32)
    fg = remove_small_objects(fg, min_size=min_area)

    if not np.any(fg):
        return np.zeros((H, W), np.int32)

    # Elevation for watershed: combine inverse prob + hv magnitude (encourage split)
    hv_mag = np.sqrt(np.sum(hv_pred ** 2, axis=-1))  # (H,W)
    elev = (1.0 - p) + 0.5 * (hv_mag / (hv_mag.max() + 1e-6))

    # Seeds via distance transform peaks
    dist = distance_transform_edt(fg)
    dist_s = gaussian_filter(dist, sigma=1.0)

    # Create markers: threshold local peaks of smoothed distance
    thr = np.quantile(dist_s[fg], 0.75) if np.any(fg) else 0.0
    markers = (dist_s > max(thr, 1.0)).astype(np.int32)
    markers = remove_small_objects(markers.astype(bool), min_size=min_area).astype(np.int32)
    markers = label(markers)

    if markers.max() == 0:
        # Fallback: connected components on fg
        markers = label(fg.astype(np.uint8))

    inst_map = watershed(elev, markers=markers, mask=fg)

    # Remove tiny segments
    for r in regionprops(inst_map):
        if r.area < min_area:
            inst_map[inst_map == r.label] = 0
    inst_map = label(inst_map > 0)
    return inst_map.astype(np.int32)


# -------------------------------
# Metrics (Hover-Net style)
# -------------------------------
# -------------------------------
# Type-wise helpers
# -------------------------------


def compute_instance_types_from_pixel_types(inst_map: np.ndarray, type_map: np.ndarray, num_classes: int, bg_index: int) -> dict:
    """Return {instance_id: class_id} by majority vote over pixel-level types.
    Background class is suppressed from voting.
    """
    out = {}
    ids = [i for i in np.unique(inst_map) if i != 0]
    for i_id in ids:
        mask = (inst_map == i_id)
        vals = type_map[mask]
        if vals.size == 0:
            out[i_id] = bg_index
            continue
        hist = np.bincount(vals, minlength=max(num_classes, vals.max()+1))
        if bg_index < len(hist):
            hist[bg_index] = 0
        out[i_id] = int(np.argmax(hist)) if hist.sum() > 0 else bg_index
    return out


def filter_and_relabel(inst_map: np.ndarray, keep_ids: set) -> np.ndarray:
    """Keep only instances whose id is in keep_ids; relabel to 1..K.
    """
    if len(keep_ids) == 0:
        return np.zeros_like(inst_map)
    out = np.zeros_like(inst_map)
    cur = 0
    for lab in np.unique(inst_map):
        if lab == 0:
            continue
        if lab in keep_ids:
            cur += 1
            out[inst_map == lab] = cur
    return out
def dice_coefficient(gt_bin: np.ndarray, pr_bin: np.ndarray) -> float:
    gt = gt_bin.astype(bool)
    pr = pr_bin.astype(bool)
    inter = (gt & pr).sum()
    denom = gt.sum() + pr.sum()
    return (2.0 * inter / denom) if denom > 0 else 1.0


def pairwise_iou(gt_labs: np.ndarray, pr_labs: np.ndarray) -> np.ndarray:
    gt_ids = [i for i in np.unique(gt_labs) if i != 0]
    pr_ids = [i for i in np.unique(pr_labs) if i != 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    iou = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    for gi, g in enumerate(gt_ids):
        gmask = (gt_labs == g)
        gsum = gmask.sum()
        for pj, p in enumerate(pr_ids):
            pmask = (pr_labs == p)
            inter = np.logical_and(gmask, pmask).sum()
            if inter == 0:
                continue
            uni = gsum + pmask.sum() - inter
            iou[gi, pj] = inter / max(uni, 1)
    return iou


def match_by_iou(iou: np.ndarray, thr: float = 0.5):
    """Greedy matching of instances by IoU threshold.
    Returns: list of (gi, pj, iou), unmatched_gt_ids, unmatched_pr_ids
    """
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    iou_copy = iou.copy()
    matches = []
    used_g = set()
    used_p = set()
    while True:
        gi, pj = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
        best = iou_copy[gi, pj]
        if best < thr or best <= 0:
            break
        matches.append((gi, pj, float(best)))
        used_g.add(gi)
        used_p.add(pj)
        iou_copy[gi, :] = -1
        iou_copy[:, pj] = -1
    all_g = set(range(iou.shape[0]))
    all_p = set(range(iou.shape[1]))
    um_g = sorted(list(all_g - used_g))
    um_p = sorted(list(all_p - used_p))
    return matches, um_g, um_p


def compute_pq(gt_labs: np.ndarray, pr_labs: np.ndarray, iou_thr: float = 0.5):
    iou = pairwise_iou(gt_labs, pr_labs)
    matches, um_g, um_p = match_by_iou(iou, thr=iou_thr)
    tp = len(matches)
    fp = len(um_p)
    fn = len(um_g)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 1.0
    sq = (np.mean([m[2] for m in matches]) if tp > 0 else 0.0)
    pq = dq * sq
    return pq, dq, sq, tp, fp, fn


def compute_aji(gt_labs: np.ndarray, pr_labs: np.ndarray) -> float:
    gt_ids = [i for i in np.unique(gt_labs) if i != 0]
    pr_ids = [i for i in np.unique(pr_labs) if i != 0]
    if len(gt_ids) == 0 and len(pr_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return 0.0

    iou = pairwise_iou(gt_labs, pr_labs)
    # For each GT, take best-matching pred (may be 0 IoU)
    used_pred = set()
    inter_sum = 0
    union_sum = 0
    for gi, g in enumerate(gt_ids):
        # Best pred for this gt
        pj = int(np.argmax(iou[gi])) if iou.shape[1] > 0 else -1
        if pj >= 0:
            chosen_pred = pr_ids[pj]
            used_pred.add(chosen_pred)
            gmask = (gt_labs == g)
            pmask = (pr_labs == chosen_pred)
            inter = np.logical_and(gmask, pmask).sum()
            uni = np.logical_or(gmask, pmask).sum()
            inter_sum += inter
            union_sum += uni
        else:
            # No preds at all
            gmask = (gt_labs == g)
            inter_sum += 0
            union_sum += gmask.sum()

    # Add unmatched preds area to denominator (per AJI def)
    unmatched_preds = [p for p in pr_ids if p not in used_pred]
    for p in unmatched_preds:
        pmask = (pr_labs == p)
        union_sum += pmask.sum()

    return inter_sum / max(union_sum, 1)


# -------------------------------
# Main
# -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate CellViT on CoNSeP (merged) test set")
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--test_root', type=Path, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)

    # Model hyperparams (must match training)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--patch_size', type=int, default=16)
    ap.add_argument('--vit_embed_dim', type=int, default=384)
    ap.add_argument('--vit_depth', type=int, default=12)
    ap.add_argument('--vit_heads', type=int, default=6)
    ap.add_argument('--vit_mlp_ratio', type=float, default=4.0)
    ap.add_argument('--num_classes', type=int, default=5, help='nuclei classes incl. bg (merged=5)')

    # Postproc / matching
    ap.add_argument('--bin_thresh', type=float, default=0.5)
    ap.add_argument('--min_area', type=int, default=10)
    ap.add_argument('--iou_thr', type=float, default=0.5)

    # Binary/type head handling
    ap.add_argument('--fg_index', type=int, default=-1, help='if bin head has 2 channels (bg/fg), which index is foreground (default=-1)')
    ap.add_argument('--bg_index', type=int, default=0, help='background class index in type maps')
    ap.add_argument('--class_names', type=str, default='bg,other,inflammatory,epithelial,spindle', help='comma-separated names for classes incl. background at index 0')
    ap.add_argument('--disable_amp', action='store_true', help='disable autocast mixed precision')

    # Visualization
    ap.add_argument('--save_viz_dir', type=Path, default=None, help='directory to save qualitative overlays')
    ap.add_argument('--viz_max', type=int, default=32, help='maximum number of images to save (per run)')

    # Output
    ap.add_argument('--save_csv', type=Path, default=None, help='optional path to save per-image metrics csv')
    return ap.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset / loader
    ds_te = ConsepPatchDatasetMerged(args.test_root, label_scheme='consep_merged')
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
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
    ckpt = torch.load(str(args.ckpt), map_location='cpu')
    sd = ckpt.get('model', ckpt)
    _ = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    # Check GT instance availability (probe first sample)
    first = ds_te[0]
    gt_inst_key = None
    for k in ['inst_map', 'instance_map', 'inst', 'instances']:
        if k in first:
            gt_inst_key = k
            break
    if gt_inst_key is None:
        print('[ERROR] Ground-truth instance map not found in dataset item.\n'
              'Make sure ConsepPatchDatasetMerged.__getitem__ returns an instance label map under one of the keys\n'
              "['inst_map', 'instance_map']")
        sys.exit(1)

    # Accumulators
    dices, ajis, pqs, dqs, sqs = [], [], [], [], []
    per_image = []  # for optional CSV

    amp_enabled = (torch.cuda.is_available() and (not args.disable_amp))

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        idx_offset = 0
        # Prepare per-class accumulators (exclude background)
        K = args.num_classes
        class_ids = [c for c in range(K) if c != args.bg_index]
        class_names = None
        raw = [s.strip() for s in str(args.class_names).split(',') if s.strip()]
        if raw:
            class_names = {i: (raw[i] if i < len(raw) else f'class{i}') for i in range(K)}
        else:
            class_names = {i: f'class{i}' for i in range(K)}

        perclass = {c: {'dice': [], 'aji': [], 'pq': [], 'dq': [], 'sq': []} for c in class_ids}

        for batch in tqdm(dl_te, ncols=100, desc='Eval'):
            x = batch['image'].to(device, non_blocking=True)
            gt_bin = batch['bin_map']        # (B, H, W)
            gt_inst = batch[gt_inst_key]     # (B, H, W) labeled
            gt_type = batch.get('type_map', None)  # (B, H, W) class ids, optional but recommended

            y = model(x)
            bin_like, hv_map = extract_pred_maps(y)
            type_like = _first_present(y, ['type_logits', 'nuclei_type_map', 'type', 'np_type', 'type_map_pred'])

            # bin_like -> (B,H,W)
            if bin_like.ndim == 4:
                if bin_like.shape[1] == 1:
                    bin_like = bin_like[:, 0]
                elif bin_like.shape[1] == 2:  # bg/fg 2채널 (choose foreground by fg_index)
                    fg_idx = args.fg_index if args.fg_index in (0, 1, -1) else -1
                    bin_like = bin_like[:, fg_idx]
                else:
                    raise ValueError(f"Unexpected bin_like shape: {tuple(bin_like.shape)}")
            elif bin_like.ndim != 3:
                raise ValueError(f"Unexpected bin_like shape: {tuple(bin_like.shape)}")

            # logits인지 prob인지 값 범위로 판별 후 sigmoid
            minv = float(bin_like.min().detach().cpu())
            maxv = float(bin_like.max().detach().cpu())
            bin_prob = bin_like.sigmoid() if (minv < 0.0 or maxv > 1.0) else bin_like  # (B,H,W)

            # hv_map -> (B,H,W,2)
            if hv_map.ndim == 4 and hv_map.shape[1] == 2:
                hv = hv_map.permute(0, 2, 3, 1).contiguous()
            elif hv_map.ndim == 4 and hv_map.shape[-1] == 2:
                hv = hv_map.contiguous()
            else:
                raise ValueError(f"hv_map must be (B,2,H,W) or (B,H,W,2), got {tuple(hv_map.shape)}")

            # type_like -> (B,H,W) of discrete class ids (optional)
            type_argmax = None
            if type_like is not None:
                if type_like.ndim == 4:
                    # assume (B,C,H,W)
                    type_argmax = type_like.argmax(dim=1)
                elif type_like.ndim == 3:
                    type_argmax = type_like  # already indices
                else:
                    raise ValueError(f"Unexpected type_like shape: {tuple(type_like.shape)}")

            B = x.size(0)
            for b in range(B):
                bp = tensor_to_numpy(bin_prob[b]).astype(np.float32, copy=False)
                hvp = tensor_to_numpy(hv[b]).astype(np.float32, copy=False)
                gt_b = gt_bin[b].cpu().numpy().astype(np.uint8)
                gt_i = gt_inst[b].cpu().numpy().astype(np.int32)

                pr_inst = postproc_to_instances(bp, hvp, bin_thresh=args.bin_thresh, min_area=args.min_area)

                # Metrics
                pr_bin = (pr_inst > 0).astype(np.uint8)
                dice = dice_coefficient(gt_b, pr_bin)
                aji = compute_aji(gt_i, pr_inst)
                pq, dq, sq, tp, fp, fn = compute_pq(gt_i, pr_inst, iou_thr=args.iou_thr)

                dices.append(dice); ajis.append(aji); pqs.append(pq); dqs.append(dq); sqs.append(sq)

                # Per-class metrics (need both GT type_map and pred type_like)
                if gt_type is not None and type_argmax is not None:
                    gt_type_map = gt_type[b].cpu().numpy().astype(np.int32)
                    pr_type_map = type_argmax[b].detach().cpu().numpy().astype(np.int32)

                    # Instance -> class by majority vote
                    gt_inst_types = compute_instance_types_from_pixel_types(gt_i, gt_type_map, args.num_classes, args.bg_index)
                    pr_inst_types = compute_instance_types_from_pixel_types(pr_inst, pr_type_map, args.num_classes, args.bg_index)

                    for c in class_ids:
                        gt_keep = {gid for gid, t in gt_inst_types.items() if t == c}
                        pr_keep = {pid for pid, t in pr_inst_types.items() if t == c}
                        gt_c = filter_and_relabel(gt_i, gt_keep)
                        pr_c = filter_and_relabel(pr_inst, pr_keep)

                        dice_c = dice_coefficient((gt_c > 0).astype(np.uint8), (pr_c > 0).astype(np.uint8))
                        aji_c  = compute_aji(gt_c, pr_c)
                        pq_c, dq_c, sq_c, *_ = compute_pq(gt_c, pr_c, iou_thr=args.iou_thr)

                        perclass[c]['dice'].append(dice_c)
                        perclass[c]['aji'].append(aji_c)
                        perclass[c]['pq'].append(pq_c)
                        perclass[c]['dq'].append(dq_c)
                        perclass[c]['sq'].append(sq_c)

                # Visualization (if enabled and within limit)
                if args.save_viz_dir is not None and (idx_offset + b) < args.viz_max:
                    img01 = to_hwc01(batch['image'][b])
                    # Prefer GT type if available for GT pane; else binary
                    gt_type_map_vis = gt_type[b].cpu().numpy().astype(np.int32) if gt_type is not None else (gt_b > 0).astype(np.int32)
                    # Prefer predicted type if available; else binary pred
                    if type_argmax is not None:
                        pr_type_map_vis = type_argmax[b].detach().cpu().numpy().astype(np.int32)
                    else:
                        pr_type_map_vis = (pr_inst > 0).astype(np.int32)
                    save_viz(Path(args.save_viz_dir) / f"{idx_offset + b:05d}.png", img01, gt_type_map_vis, pr_type_map_vis, class_names)

                per_image.append({
                    'index': idx_offset + b,
                    'dice': dice,
                    'aji': aji,
                    'pq': pq,
                    'dq': dq,
                    'sq': sq,
                    'tp': tp, 'fp': fp, 'fn': fn
                })
            idx_offset += B

    # Aggregate
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_aji  = float(np.mean(ajis)) if ajis else 0.0
    mean_pq   = float(np.mean(pqs)) if pqs else 0.0
    mean_dq   = float(np.mean(dqs)) if dqs else 0.0
    mean_sq   = float(np.mean(sqs)) if sqs else 0.0

    print("==== Evaluation (CoNSeP merged) ====")
    print(f"DICE: {mean_dice:.4f}")
    print(f" AJI: {mean_aji:.4f}")
    print(f"  DQ: {mean_dq:.4f}")
    print(f"  SQ: {mean_sq:.4f}")
    print(f"  PQ: {mean_pq:.4f}")

    # Per-class summary (if computed)
    if 'perclass' in locals() and any(len(v['pq']) for v in perclass.values()):
        print("-- Per-class metrics (exclude background) --")
        print(f"{'Class':<16} {'DICE':>8} {'AJI':>8} {'DQ':>8} {'SQ':>8} {'PQ':>8}")
        for c in class_ids:
            d = perclass[c]
            md = np.mean(d['dice']) if d['dice'] else 0.0
            ma = np.mean(d['aji'])  if d['aji']  else 0.0
            mdq= np.mean(d['dq'])   if d['dq']   else 0.0
            msq= np.mean(d['sq'])   if d['sq']   else 0.0
            mpq= np.mean(d['pq'])   if d['pq']   else 0.0
            print(f"{class_names[c]:<16} {md:8.4f} {ma:8.4f} {mdq:8.4f} {msq:8.4f} {mpq:8.4f}")

        macro_pq = np.mean([np.mean(perclass[c]['pq']) for c in class_ids if perclass[c]['pq']]) if class_ids else 0.0
        print(f"Macro PQ (over {len(class_ids)} classes): {macro_pq:.4f}")

    # Optional CSV
    if per_image and args.save_csv is not None:
        import csv
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            for row in per_image:
                w.writerow(row)
        print(f"Saved per-image metrics to: {args.save_csv}")

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_aji  = float(np.mean(ajis)) if ajis else 0.0
    mean_pq   = float(np.mean(pqs)) if pqs else 0.0
    mean_dq   = float(np.mean(dqs)) if dqs else 0.0
    mean_sq   = float(np.mean(sqs)) if sqs else 0.0

    print("\n==== Evaluation (CoNSeP merged) ====")
    print(f"DICE: {mean_dice:.4f}")
    print(f" AJI: {mean_aji:.4f}")
    print(f"  DQ: {mean_dq:.4f}")
    print(f"  SQ: {mean_sq:.4f}")
    print(f"  PQ: {mean_pq:.4f}")

    # Optional CSV
    if per_image and args.save_csv is not None:
        import csv
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            for row in per_image:
                w.writerow(row)
        print(f"Saved per-image metrics to: {args.save_csv}")


if __name__ == '__main__':
    main()


