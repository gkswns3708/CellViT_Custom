# eval_consep_metrics.py
# -*- coding: utf-8 -*-
"""
Evaluate CellViT (ViT-256) on CoNSeP (merged) test set and compute
DICE, AJI, DQ, SQ, PQ (overall + per-class).

Requires:
    pip install scikit-image scipy

Example:
    python eval_consep_metrics.py \
      --ckpt Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth \
      --test_root Dataset/CoNSeP/Preprocessed/Test \
      --batch_size 8 --device cuda \
      --out eval_out_consep_merged \
      --save_csv eval_out_consep_merged/per_image_metrics.csv
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

# repo-local imports (경로는 레포 구조에 맞춰 유지)
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
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _first_present(y, keys):
    for k in keys:
        if k in y and y[k] is not None:
            return y[k]
    return None


def extract_pred_maps(y: dict):
    """Pick model outputs by tolerant key search.
    Returns (bin_logits_or_prob, hv_map, type_logits_or_idx) where:
        bin_like: (B,1|2,H,W) or (B,H,W)
        hv_map:   (B,2,H,W) or (B,H,W,2)
        type_like: (B,C,H,W) logits or (B,H,W) indices or None
    """
    bin_like = _first_present(y, [
        'nuclei_binary_map', 'bin_logits', 'binary_map', 'np_bin', 'bin'
    ])
    hv_map = _first_present(y, ['hv_map', 'hv', 'np_hv'])
    type_like = _first_present(y, ['nuclei_type_map', 'type_logits', 'type', 'np_type', 'type_map_pred'])

    if bin_like is None or hv_map is None:
        raise KeyError(
            f"Could not find required heads in model output. Got keys={list(y.keys())}.\n"
            "Expected one of ['nuclei_binary_map','bin_logits','binary_map','np_bin','bin'] "
            "and one of ['hv_map','hv','np_hv']."
        )
    return bin_like, hv_map, type_like


def postproc_to_instances(bin_prob: np.ndarray, hv_pred: np.ndarray,
                          bin_thresh: float = 0.5,
                          min_area: int = 10) -> np.ndarray:
    """Lightweight instance post-processing using watershed.
    Args:
        bin_prob: (H, W) nuclei probability in [0,1]
        hv_pred:  (H, W, 2) predicted HV maps
    Returns:
        inst_map: (H, W) labeled instance mask (0=background, 1..N)
    """
    H, W = bin_prob.shape
    p = gaussian_filter(bin_prob, sigma=1.0)

    # foreground mask
    fg = p >= bin_thresh
    fg = remove_small_holes(fg, area_threshold=32)
    fg = remove_small_objects(fg, min_size=min_area)
    if not np.any(fg):
        return np.zeros((H, W), np.int32)

    # Elevation: inverse prob + hv magnitude
    hv_mag = np.sqrt(np.sum(hv_pred ** 2, axis=-1))
    elev = (1.0 - p) + 0.5 * (hv_mag / (hv_mag.max() + 1e-6))

    # Seeds via distance peaks
    dist = distance_transform_edt(fg)
    dist_s = gaussian_filter(dist, sigma=1.0)
    thr = np.quantile(dist_s[fg], 0.75) if np.any(fg) else 0.0
    markers = (dist_s > max(thr, 1.0)).astype(np.int32)
    markers = remove_small_objects(markers.astype(bool), min_size=min_area).astype(np.int32)
    markers = label(markers)
    if markers.max() == 0:
        markers = label(fg.astype(np.uint8))

    inst_map = watershed(elev, markers=markers, mask=fg)

    # remove tiny segments
    for r in regionprops(inst_map):
        if r.area < min_area:
            inst_map[inst_map == r.label] = 0
    inst_map = label(inst_map > 0)
    return inst_map.astype(np.int32)


# -------------------------------
# Metrics
# -------------------------------

def dice_coefficient(gt_bin: np.ndarray, pr_bin: np.ndarray) -> float:
    gt = gt_bin.astype(bool); pr = pr_bin.astype(bool)
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
    """Greedy matching by IoU threshold."""
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    iou_copy = iou.copy()
    matches = []
    used_g, used_p = set(), set()
    while True:
        gi, pj = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
        best = iou_copy[gi, pj]
        if best < thr or best <= 0:
            break
        matches.append((gi, pj, float(best)))
        used_g.add(gi); used_p.add(pj)
        iou_copy[gi, :] = -1; iou_copy[:, pj] = -1
    all_g = set(range(iou.shape[0])); all_p = set(range(iou.shape[1]))
    um_g = sorted(list(all_g - used_g)); um_p = sorted(list(all_p - used_p))
    return matches, um_g, um_p


def compute_pq(gt_labs: np.ndarray, pr_labs: np.ndarray, iou_thr: float = 0.5):
    iou = pairwise_iou(gt_labs, pr_labs)
    matches, um_g, um_p = match_by_iou(iou, thr=iou_thr)
    tp = len(matches); fp = len(um_p); fn = len(um_g)
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
    used_pred = set()
    inter_sum = 0
    union_sum = 0
    for gi, g in enumerate(gt_ids):
        pj = int(np.argmax(iou[gi])) if iou.shape[1] > 0 else -1
        if pj >= 0:
            chosen_pred = pr_ids[pj]
            used_pred.add(chosen_pred)
            gmask = (gt_labs == g); pmask = (pr_labs == chosen_pred)
            inter = np.logical_and(gmask, pmask).sum()
            uni = np.logical_or(gmask, pmask).sum()
            inter_sum += inter
            union_sum += uni
        else:
            gmask = (gt_labs == g)
            union_sum += gmask.sum()
    # add unmatched preds area to denominator
    unmatched_preds = [p for p in pr_ids if p not in used_pred]
    for p in unmatched_preds:
        pmask = (pr_labs == p)
        union_sum += pmask.sum()
    return inter_sum / max(union_sum, 1)


# -------------------------------
# Type-wise helpers
# -------------------------------

def compute_instance_types_from_pixel_types(inst_map: np.ndarray, type_map: np.ndarray, num_classes: int, bg_index: int) -> dict:
    """Return {instance_id: class_id} by majority vote over pixel-level types (bg suppressed)."""
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
    """Keep only instances whose id is in keep_ids; relabel to 1..K."""
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
    ap.add_argument('--fg_index', type=int, default=-1, help='if bin head has 2 channels (bg/fg), which index is foreground (default=-1 => last)')
    ap.add_argument('--bg_index', type=int, default=0, help='background class index in type maps')
    ap.add_argument('--class_names', type=str, default='bg,other,inflammatory,epithelial,spindle',
                    help='comma-separated names for classes incl. background at index 0')
    ap.add_argument('--disable_amp', action='store_true', help='disable autocast mixed precision')

    # Visualization (optional)
    ap.add_argument('--save_viz_dir', type=Path, default=None, help='directory to save qualitative overlays')
    ap.add_argument('--viz_max', type=int, default=32, help='maximum number of images to save (per run)')

    # Outputs
    ap.add_argument('--out', type=Path, default=Path('eval_out_consep_merged'),
                    help='directory to save summary CSV (segmentation_metrics.csv)')
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

    # Probe GT instance availability
    first = ds_te[0]
    gt_inst_key = None
    for k in ['inst_map', 'instance_map', 'inst', 'instances']:
        if k in first:
            gt_inst_key = k
            break
    if gt_inst_key is None:
        print('[ERROR] GT instance map not found in dataset item.\n'
              "ConsepPatchDatasetMerged.__getitem__ must return an instance map under one of: "
              "['inst_map','instance_map','inst','instances']")
        sys.exit(1)

    # Accumulators
    dices, ajis, pqs, dqs, sqs = [], [], [], [], []
    per_image = []  # optional CSV

    amp_enabled = (torch.cuda.is_available() and (not args.disable_amp))

    # Per-class accumulators (exclude bg)
    K = args.num_classes
    class_ids = [c for c in range(K) if c != args.bg_index]
    raw = [s.strip() for s in str(args.class_names).split(',') if s.strip()]
    class_names = {i: (raw[i] if i < len(raw) else f'class{i}') for i in range(K)} if raw else {i: f'class{i}' for i in range(K)}
    perclass = {
        c: {'dice': [], 'aji': [], 'pq': [], 'dq': [], 'sq': [], 'tp': 0, 'fp': 0, 'fn': 0, 'support': 0}
        for c in class_ids
    }
    overall_tp = overall_fp = overall_fn = 0

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        idx_offset = 0
        for batch in tqdm(dl_te, ncols=100, desc='Eval'):
            x = batch['image'].to(device, non_blocking=True)
            gt_bin = batch['bin_map']            # (B, H, W)
            gt_inst = batch[gt_inst_key]         # (B, H, W) labeled
            gt_type = batch.get('type_map', None)  # (B, H, W) class ids, optional

            y = model(x)
            bin_like, hv_map, type_like = extract_pred_maps(y)

            # ---- nuclei prob (B,H,W)
            if bin_like.ndim == 4:
                if bin_like.shape[1] == 1:
                    prob = bin_like[:, 0]
                    # logits or prob? heuristics
                    minv = float(prob.min().detach().cpu()); maxv = float(prob.max().detach().cpu())
                    bin_prob = prob.sigmoid() if (minv < 0.0 or maxv > 1.0) else prob
                elif bin_like.shape[1] == 2:
                    # 2ch logits -> softmax -> use foreground (fg_index or last)
                    fg_idx = args.fg_index if args.fg_index in (0, 1, -1) else -1
                    sm = torch.softmax(bin_like, dim=1)
                    bin_prob = sm[:, fg_idx]
                else:
                    raise ValueError(f"Unexpected bin_like shape: {tuple(bin_like.shape)}")
            elif bin_like.ndim == 3:
                minv = float(bin_like.min().detach().cpu()); maxv = float(bin_like.max().detach().cpu())
                bin_prob = bin_like.sigmoid() if (minv < 0.0 or maxv > 1.0) else bin_like
            else:
                raise ValueError(f"Unexpected bin_like shape: {tuple(bin_like.shape)}")

            # ---- hv to (B,H,W,2)
            if hv_map.ndim == 4 and hv_map.shape[1] == 2:
                hv = hv_map.permute(0, 2, 3, 1).contiguous()
            elif hv_map.ndim == 4 and hv_map.shape[-1] == 2:
                hv = hv_map.contiguous()
            else:
                raise ValueError(f"hv_map must be (B,2,H,W) or (B,H,W,2), got {tuple(hv_map.shape)}")

            # ---- type logits -> indices (optional)
            type_argmax = None
            if type_like is not None:
                if type_like.ndim == 4:   # (B,C,H,W)
                    type_argmax = type_like.argmax(dim=1)
                elif type_like.ndim == 3: # already indices
                    type_argmax = type_like
                else:
                    raise ValueError(f"Unexpected type_like shape: {tuple(type_like.shape)}")

            B = x.size(0)
            for b in range(B):
                bp = tensor_to_numpy(bin_prob[b]).astype(np.float32, copy=False)
                hvp = tensor_to_numpy(hv[b]).astype(np.float32, copy=False)
                gt_b = gt_bin[b].cpu().numpy().astype(np.uint8)
                gt_i = gt_inst[b].cpu().numpy().astype(np.int32)

                pr_inst = postproc_to_instances(bp, hvp, bin_thresh=args.bin_thresh, min_area=args.min_area)

                # Overall metrics
                pr_bin = (pr_inst > 0).astype(np.uint8)
                dice = dice_coefficient(gt_b, pr_bin)
                aji = compute_aji(gt_i, pr_inst)
                pq, dq, sq, tp, fp, fn = compute_pq(gt_i, pr_inst, iou_thr=args.iou_thr)
                dices.append(dice); ajis.append(aji); pqs.append(pq); dqs.append(dq); sqs.append(sq)
                overall_tp += tp; overall_fp += fp; overall_fn += fn

                # Per-class metrics (need both GT type_map and predicted types)
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
                        pq_c, dq_c, sq_c, tp_c, fp_c, fn_c = compute_pq(gt_c, pr_c, iou_thr=args.iou_thr)

                        pc = perclass[c]
                        pc['dice'].append(dice_c); pc['aji'].append(aji_c)
                        pc['pq'].append(pq_c);     pc['dq'].append(dq_c); pc['sq'].append(sq_c)
                        pc['tp'] += tp_c; pc['fp'] += fp_c; pc['fn'] += fn_c
                        pc['support'] += len(gt_keep)  # GT 인스턴스 개수

                # Per-image CSV(옵션)
                if args.save_csv is not None:
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

    # Aggregate means
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_aji  = float(np.mean(ajis))  if ajis  else 0.0
    mean_pq   = float(np.mean(pqs))   if pqs   else 0.0
    mean_dq   = float(np.mean(dqs))   if dqs   else 0.0
    mean_sq   = float(np.mean(sqs))   if sqs   else 0.0

    print("==== Evaluation (CoNSeP merged) ====")
    print(f"DICE: {mean_dice:.4f}")
    print(f" AJI: {mean_aji:.4f}")
    print(f"  DQ: {mean_dq:.4f}")
    print(f"  SQ: {mean_sq:.4f}")
    print(f"  PQ: {mean_pq:.4f}")

    # Per-class summary (print)
    if any(len(v['pq']) for v in perclass.values()):
        print("-- Per-class metrics (exclude background) --")
        print(f"{'Class':<16} {'DICE':>8} {'AJI':>8} {'DQ':>8} {'SQ':>8} {'PQ':>8} {'SUP':>6}")
        for c in class_ids:
            pc = perclass[c]
            md = float(np.mean(pc['dice'])) if pc['dice'] else 0.0
            ma = float(np.mean(pc['aji']))  if pc['aji']  else 0.0
            mdq= float(np.mean(pc['dq']))   if pc['dq']   else 0.0
            msq= float(np.mean(pc['sq']))   if pc['sq']   else 0.0
            mpq= float(np.mean(pc['pq']))   if pc['pq']   else 0.0
            print(f"{class_names[c]:<16} {md:8.4f} {ma:8.4f} {mdq:8.4f} {msq:8.4f} {mpq:8.4f} {pc['support']:6d}")

    # ===== Save summary CSV (overall + per-class) =====
    import csv
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_csv = out_dir / "segmentation_metrics.csv"

    # (참고) 카운트 기반 DQ도 계산 가능
    dq_overall_counts = (overall_tp / max(overall_tp + 0.5*overall_fp + 0.5*overall_fn, 1e-6)
                         if (overall_tp+overall_fp+overall_fn)>0 else 1.0)
    pq_overall_counts = dq_overall_counts * mean_sq  # SQ는 매칭 IoU 평균이므로 mean_sq 사용

    with open(seg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope","class_id","class_name","DICE","AJI","DQ","SQ","PQ","support","TP","FP","FN"])
        # overall
        w.writerow([
            "overall","-","-",
            f"{mean_dice:.6f}", f"{mean_aji:.6f}",
            f"{mean_dq:.6f}", f"{mean_sq:.6f}", f"{mean_pq:.6f}",
            sum(perclass[c]['support'] for c in perclass), overall_tp, overall_fp, overall_fn
        ])
        # per-class
        for c in class_ids:
            pc = perclass[c]
            md = float(np.mean(pc['dice'])) if pc['dice'] else 0.0
            ma = float(np.mean(pc['aji']))  if pc['aji']  else 0.0
            mdq= float(np.mean(pc['dq']))   if pc['dq']   else 0.0
            msq= float(np.mean(pc['sq']))   if pc['sq']   else 0.0
            mpq= float(np.mean(pc['pq']))   if pc['pq']   else 0.0
            w.writerow([
                "per-class", c, class_names.get(c, f"class{c}"),
                f"{md:.6f}", f"{ma:.6f}", f"{mdq:.6f}", f"{msq:.6f}", f"{mpq:.6f}",
                pc['support'], pc['tp'], pc['fp'], pc['fn']
            ])
    print(f"[Saved] {seg_csv}")

    # Optional per-image CSV
    if per_image and args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            for row in per_image:
                w.writerow(row)
        print(f"Saved per-image metrics to: {args.save_csv}")


if __name__ == '__main__':
    print("[eval] CoNSeP merged segmentation metrics (DICE/AJI/DQ/SQ/PQ) runner")
    main()
