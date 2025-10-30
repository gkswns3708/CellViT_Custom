# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
from torch.utils.data import DataLoader

# local datasets
from Data.CoNSeP_patch import ConsepPatchDataset
from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged
from Data.PanNuke_hf import PanNukeHFDataset

from Model.CellViT_ViT256_Custom import CellViTCustom


# =============================
# Label spec (id->name, palette, remap)
# =============================
def label_spec(dataset: str, merge_consep: bool):
    ds = dataset.lower()

    if ds == 'consep':
        if not merge_consep:
            id2name = {
                1:"other", 2:"inflammatory", 3:"healthy epithelial",
                4:"dysplastic/malignant epithelial", 5:"fibroblast",
                6:"muscle", 7:"endothelial"
            }
            remap = {k:k for k in id2name.keys()}
            palette = {
                0:(0,0,0), 1:(76,153,0), 2:(255,127,0), 3:(0,176,240),
                4:(0,92,230), 5:(255,0,128), 6:(153,51,255), 7:(255,0,0)
            }
            return id2name, palette, remap
        # consep를 merged 뷰로 보고 싶을 때만 이 remap 사용
        id2name = {1:"other", 2:"inflammatory", 3:"epithelial (3+4)", 4:"spindle (5+6+7)"}
        remap   = {1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:4}
        palette = {0:(0,0,0), 1:(76,153,0), 2:(255,127,0), 3:(0,176,240), 4:(255,0,128)}
        return id2name, palette, remap

    elif ds == 'consep_merged':
        # 이미 [0..4]로 합쳐진 라벨 → 항등 remap!
        id2name = {1:"other", 2:"inflammatory", 3:"epithelial (3+4)", 4:"spindle (5+6+7)"}
        remap   = {1:1, 2:2, 3:3, 4:4}
        palette = {0:(0,0,0), 1:(76,153,0), 2:(255,127,0), 3:(0,176,240), 4:(255,0,128)}
        return id2name, palette, remap

    elif ds == 'pannuke_hf':
        id2name = {1:"Neoplastic", 2:"Inflammatory", 3:"Connective", 4:"Dead", 5:"Epithelial"}
        remap   = {k:k for k in id2name.keys()}
        palette = {0:(0,0,0), 1:(0,176,240), 2:(255,127,0), 3:(76,153,0), 4:(128,128,128), 5:(255,0,128)}
        return id2name, palette, remap

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def colorize_label(label: np.ndarray, palette: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        out[label == k] = rgb
    return out


def blend_overlay(img_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float=0.5) -> np.ndarray:
    img = img_rgb.astype(np.float32) / 255.0
    ov  = overlay_rgb.astype(np.float32) / 255.0
    out = (1 - alpha) * img + alpha * ov
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def build_legend_handles(id2name: Dict[int, str], palette: Dict[int, tuple]) -> List[Patch]:
    handles = []
    for cid in sorted(id2name.keys()):
        rgb = palette[cid]; color = tuple([c/255.0 for c in rgb])
        handles.append(Patch(facecolor=color, edgecolor='black', label=f"{cid}: {id2name[cid]}"))
    return handles


# =============================
# Confusion/metrics (classification)
# =============================
def update_confusion(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, C: int):
    y_true = np.clip(y_true, 0, C)
    y_pred = np.clip(y_pred, 0, C)
    idx = C + 1
    cm = np.bincount((y_true * idx + y_pred).ravel(), minlength=idx*idx).reshape(idx, idx)
    conf += cm


def metrics_from_confusion(conf: np.ndarray, classes_to_eval: List[int]):
    per_class = {}
    total = conf.sum()
    overall_acc = np.trace(conf) / total if total > 0 else 0.0
    col_sum = conf.sum(axis=0)
    row_sum = conf.sum(axis=1)

    for c in classes_to_eval:
        tp = conf[c, c]
        fp = col_sum[c] - tp
        fn = row_sum[c] - tp
        support = row_sum[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc_c = rec
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c, support=int(support))
    return overall_acc, per_class


# =============================
# Instance metrics helpers (AJI / PQ / DQ / SQ) + DICE
# =============================
def binary_dice(mask_pred: np.ndarray, mask_true: np.ndarray, eps: float=1e-6) -> float:
    inter = np.sum(mask_pred & mask_true)
    denom = np.sum(mask_pred) + np.sum(mask_true) + eps
    return (2.0 * inter) / denom


def modal_type_per_instance(inst_map: np.ndarray, type_map: np.ndarray) -> Dict[int, int]:
    """각 인스턴스 id별 modal(최빈) type을 반환 (0은 제외)."""
    ids = np.unique(inst_map)
    ids = ids[ids != 0]
    out = {}
    for i in ids:
        t = type_map[inst_map == i]
        if t.size == 0:
            continue
        # background(0) 제외한 modal, 없으면 0
        vals, cnts = np.unique(t[t > 0], return_counts=True)
        out[i] = int(vals[np.argmax(cnts)]) if len(vals) else 0
    return out


def contingency_table(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT/Pred 라벨 쌍 카운트 테이블 (0은 background로 유지)."""
    gt_ids = np.unique(gt)
    pr_ids = np.unique(pred)
    gt_ids = gt_ids[gt_ids != 0]
    pr_ids = pr_ids[pr_ids != 0]
    if gt_ids.size == 0 or pr_ids.size == 0:
        return np.zeros((gt_ids.size, pr_ids.size), dtype=np.int64)

    gt_index = {k:i for i,k in enumerate(gt_ids)}
    pr_index = {k:i for i,k in enumerate(pr_ids)}

    # (gt_id, pr_id) 쌍 카운트
    pairs = gt.astype(np.int64) * (pred.max() + 1) + pred.astype(np.int64)
    uniq, cnt = np.unique(pairs, return_counts=True)

    inter = np.zeros((gt_ids.size, pr_ids.size), dtype=np.int64)
    for u, c in zip(uniq, cnt):
        g = int(u // (pred.max() + 1))
        p = int(u %  (pred.max() + 1))
        if g == 0 or p == 0:
            continue
        gi = gt_index.get(g, None)
        pi = pr_index.get(p, None)
        if gi is not None and pi is not None:
            inter[gi, pi] = c
    return inter, gt_ids, pr_ids


def match_by_iou(gt: np.ndarray, pred: np.ndarray, iou_thr: float=0.5):
    """IoU 기반 1:1 매칭 (그리디, IoU 내림차순)."""
    inter, gt_ids, pr_ids = contingency_table(gt, pred)
    if inter.size == 0:
        return [], gt_ids, pr_ids, inter

    gt_areas = np.array([(gt == i).sum() for i in gt_ids], dtype=np.int64)
    pr_areas = np.array([(pred == j).sum() for j in pr_ids], dtype=np.int64)
    # IoU = inter / (area_g + area_p - inter)
    iou = inter / (gt_areas[:, None] + pr_areas[None, :] - inter + 1e-6)

    # 그리디 매칭
    pairs = []
    used_g = set()
    used_p = set()
    # 내림차순으로 모든 (gi, pj) 후보 정렬
    gi, pj = np.where(iou > 0.0)
    order = np.argsort(-iou[gi, pj])
    for idx in order:
        g = gi[idx]; p = pj[idx]; v = iou[g, p]
        if v < iou_thr:
            break
        if (g not in used_g) and (p not in used_p):
            pairs.append((int(g), int(p), float(v)))
            used_g.add(g); used_p.add(p)

    return pairs, gt_ids, pr_ids, iou


def compute_pq_aji(
    gt_inst: np.ndarray, pr_inst: np.ndarray,
    gt_types: Optional[Dict[int,int]]=None, pr_types: Optional[Dict[int,int]]=None,
    class_id: Optional[int]=None, iou_thr: float=0.5
):
    """
    class_id가 지정되면 해당 타입 인스턴스만 대상으로 PQ/AJI 계산.
      - PQ = SQ * DQ
      - DQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
      - SQ = sum IoU(TP) / |TP|
      - AJI = sum_{matched} |A∩B| / ( sum_{matched} |A∪B| + sum_area(unmatched GT) + sum_area(unmatched PR) )
    """
    gt = gt_inst.copy()
    pr = pr_inst.copy()

    # 타입 필터링: 해당 타입 아닌 인스턴스는 0으로 날림
    if class_id is not None and gt_types is not None and pr_types is not None:
        # 리라벨링: 해당 타입의 id만 유지
        for gid in np.unique(gt):
            if gid == 0: continue
            if gt_types.get(int(gid), 0) != class_id:
                gt[gt == gid] = 0
        for pid in np.unique(pr):
            if pid == 0: continue
            if pr_types.get(int(pid), 0) != class_id:
                pr[pr == pid] = 0

    pairs, gt_ids, pr_ids, iou = match_by_iou(gt, pr, iou_thr=iou_thr)

    # TP, FP, FN
    tp = len(pairs)
    fp = int(len(pr_ids) - tp)
    fn = int(len(gt_ids) - tp)

    # DQ, SQ, PQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6) if (tp + fp + fn) > 0 else 0.0
    sq = (sum(v for _, _, v in pairs) / (tp + 1e-6)) if tp > 0 else 0.0
    pq = dq * sq

    # AJI 계산
    # 매칭된 쌍의 intersection / union 합
    inter, gt_ids_all, pr_ids_all = contingency_table(gt, pr)[0:3]
    # gt/pr id -> index 맵
    gt_index = {k:i for i,k in enumerate(gt_ids_all)}
    pr_index = {k:i for i,k in enumerate(pr_ids_all)}

    sum_inter = 0.0
    sum_union = 0.0
    matched_gt = set(); matched_pr = set()
    for gi, pj, _ in pairs:
        g_id = gt_ids[gi]; p_id = pr_ids[pj]
        gmask = (gt == g_id); pmask = (pr == p_id)
        inter_ = np.sum(gmask & pmask)
        union_ = np.sum(gmask | pmask)
        sum_inter += inter_; sum_union += union_
        matched_gt.add(int(g_id)); matched_pr.add(int(p_id))

    # unmatched GT/PR 영역은 union에 그대로 더해준다
    for g_id in gt_ids:
        if int(g_id) not in matched_gt:
            sum_union += float((gt == g_id).sum())
    for p_id in pr_ids:
        if int(p_id) not in matched_pr:
            sum_union += float((pr == p_id).sum())

    aji = (sum_inter / (sum_union + 1e-6)) if sum_union > 0 else 0.0
    return dict(PQ=pq, DQ=dq, SQ=sq, AJI=aji, TP=tp, FP=fp, FN=fn)


# =============================
# Eval + Visualization + Metrics Aggregation
# =============================
@torch.no_grad()
def evaluate_and_visualize(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset/Loader
    if args.dataset == 'pannuke_hf':
        ds = PanNukeHFDataset(
            repo_id=args.hf_repo, split=args.hf_split, fold=args.hf_fold, normalize=True
        )
    elif args.dataset == 'consep_merged':
        ds = ConsepPatchDatasetMerged(args.root, label_scheme='consep_merged')
    elif args.dataset == 'consep':
        ds = ConsepPatchDataset(args.root, num_nuclei_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # remap/legend
    merge_flag = (args.dataset.lower() == 'consep') and args.merge_consep
    id2name, palette, remap = label_spec(args.dataset, merge_flag)

    # Model
    model = CellViTCustom(
        num_nuclei_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_ratio=args.vit_mlp_ratio,
    ).to(device).eval()

    # Load weights
    if args.ckpt is not None and Path(args.ckpt).exists():
        ck = torch.load(str(args.ckpt), map_location='cpu')
        sd = ck.get('model', ck)
        model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded model weights: {args.ckpt}")
    else:
        print("[eval] WARNING: no checkpoint provided — random weights!")

    # confusion for type classification
    C = args.num_classes - 1
    conf = np.zeros((C+1, C+1), dtype=np.int64)
    classes_to_eval = list(range(1, C+1)) if args.exclude_bg else list(range(0, C+1))

    # Aggregators for new metrics
    dice_overall_num = 0.0; dice_overall_den = 0.0
    dice_per_class_num = {c:0.0 for c in range(1, C+1)}
    dice_per_class_den = {c:0.0 for c in range(1, C+1)}

    # PQ/AJI aggregators (sum over images, then average)
    pq_sum = dict(overall=dict(PQ=0.0, DQ=0.0, SQ=0.0, AJI=0.0, N=0))
    for c in range(1, C+1):
        pq_sum[c] = dict(PQ=0.0, DQ=0.0, SQ=0.0, AJI=0.0, N=0)

    saved = 0
    for batch in tqdm(dl, ncols=100, desc="Eval"):
        x = batch['image'].to(device, non_blocking=True)
        t_type = batch['type_map'].cpu().numpy()
        paths = batch.get('path_image', [''] * x.size(0))

        # 예측
        with torch.no_grad():
            y = model(x)
        pr_type_logits = y['nuclei_type_map']
        pr_type = torch.argmax(pr_type_logits, dim=1).cpu().numpy()

        # 필요 시 리매핑 (consep + merge_consep)
        if args.dataset.startswith('consep') and merge_flag:
            vt = np.vectorize(lambda k: remap.get(int(k), int(k)))
            t_type = np.stack([vt(t_type[i]) for i in range(t_type.shape[0])], axis=0)
            pr_type = np.stack([vt(pr_type[i]) for i in range(pr_type.shape[0])], axis=0)

        # confusion 업데이트 (타입 분류 관점)
        t_clamped = np.clip(t_type, 0, C)
        p_clamped = np.clip(pr_type, 0, C)
        update_confusion(conf, t_clamped.reshape(-1), p_clamped.reshape(-1), C)

        # DICE (pixel-wise)
        # overall: nucleus vs background
        gt_nuc = (t_clamped > 0)
        pr_nuc = (p_clamped > 0)
        dice_overall_num += 2.0 * np.sum(gt_nuc & pr_nuc)
        dice_overall_den += np.sum(gt_nuc) + np.sum(pr_nuc)

        # per-class
        for c in range(1, C+1):
            gt_c = (t_clamped == c); pr_c = (p_clamped == c)
            dice_per_class_num[c] += 2.0 * np.sum(gt_c & pr_c)
            dice_per_class_den[c] += np.sum(gt_c) + np.sum(pr_c)

        # ===== Instance metrics (AJI/PQ/DQ/SQ) =====
        # GT inst map 확보: 'inst_map' 또는 'instance_map' 키 기대
        gt_inst_map = None
        for k in ['inst_map', 'instance_map']:
            if k in batch:
                gt_inst_map = batch[k].cpu().numpy()
                break

        # Pred inst map: 모델 postproc 사용 (HV+NP+Type)
        # (없으면 스킵)
        pr_inst_map = None
        try:
            preds = {
                "nuclei_type_map": y["nuclei_type_map"],
                "nuclei_binary_map": y["nuclei_binary_map"],
                "hv_map": y["hv_map"],
            }
            inst_maps_t, _ = model.calculate_instance_map(preds, magnification=40)
            pr_inst_map = inst_maps_t.cpu().numpy()
        except Exception as e:
            pr_inst_map = None

        if gt_inst_map is None or pr_inst_map is None:
            # 인스턴스 지표는 스킵
            pass
        else:
            bs = pr_inst_map.shape[0]
            for i in range(bs):
                gt_i = gt_inst_map[i].astype(np.int32)
                pr_i = pr_inst_map[i].astype(np.int32)
                gt_t = t_clamped[i].astype(np.int32)
                pr_t = p_clamped[i].astype(np.int32)

                # 인스턴스별 타입 (modal)
                gt_types = modal_type_per_instance(gt_i, gt_t)
                pr_types = modal_type_per_instance(pr_i, pr_t)

                # overall
                res_all = compute_pq_aji(gt_i, pr_i, gt_types, pr_types, class_id=None, iou_thr=0.5)
                for k in ['PQ','DQ','SQ','AJI']:
                    pq_sum['overall'][k] += res_all[k]
                pq_sum['overall']['N'] += 1

                # per-class
                for c in range(1, C+1):
                    res_c = compute_pq_aji(gt_i, pr_i, gt_types, pr_types, class_id=c, iou_thr=0.5)
                    for k in ['PQ','DQ','SQ','AJI']:
                        pq_sum[c][k] += res_c[k]
                    pq_sum[c]['N'] += 1

        # ===== Visualization (선택) =====
        if args.max_samples > 0:
            for b in range(x.size(0)):
                if saved >= args.max_samples: break
                img = (x[b].cpu().numpy().transpose(1,2,0) * 255.0).round().astype(np.uint8)
                gt_rgb   = colorize_label(t_clamped[b], palette)
                pr_rgb   = colorize_label(p_clamped[b], palette)
                gt_ov    = blend_overlay(img, gt_rgb, alpha=args.type_alpha)
                pr_ov    = blend_overlay(img, pr_rgb, alpha=args.type_alpha)

                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(2,2,1); ax1.imshow(img);   ax1.set_title("Input");       ax1.axis('off')
                ax2 = plt.subplot(2,2,2); ax2.imshow(gt_ov); ax2.set_title("GT Type");     ax2.axis('off')
                ax3 = plt.subplot(2,2,3); ax3.imshow(pr_ov); ax3.set_title("Pred Type");   ax3.axis('off')

                handles = build_legend_handles(id2name, palette)
                ax4 = plt.subplot(2,2,4); ax4.axis('off')
                ax4.legend(handles=handles, title="Cell types", loc='center'); ax4.set_title("Legend")

                stem = Path(paths[b]).stem if isinstance(paths, list) else f"sample_{saved}"
                out_path = out_dir / f"{stem}_typeviz_gt_pred.png"
                plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
                saved += 1

    # ====== Classification metrics (기존) ======
    overall_acc, per_class = metrics_from_confusion(conf, classes_to_eval)

    # ====== New: DICE overall / per-class ======
    dice_overall = (dice_overall_num / (dice_overall_den + 1e-6)) if dice_overall_den > 0 else 0.0
    dice_perclass = {c: (dice_per_class_num[c] / (dice_per_class_den[c] + 1e-6)
                         if dice_per_class_den[c] > 0 else 0.0)
                     for c in range(1, C+1)}

    # ====== New: PQ/AJI/DQ/SQ overall & per-class (평균) ======
    def avg_block(d):
        n = max(1, d['N'])
        return {k: (d[k]/n if k in d else 0.0) for k in ['PQ','DQ','SQ','AJI']}

    pq_overall = avg_block(pq_sum['overall'])
    pq_perclass = {c: avg_block(pq_sum[c]) for c in range(1, C+1)}

    # ====== Print summary ======
    print("\n[Type classification metrics]")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class (exclude_bg={}):".format(args.exclude_bg))
    for c in classes_to_eval:
        name = id2name.get(c, f"class_{c}")
        m = per_class[c]
        print(f"  {c:>2} ({name:>27s}) | F1={m['f1']:.4f}  Acc={m['acc']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (support={m['support']})")

    print("\n[Segmentation metrics]")
    print(f"Overall DICE (nucleus vs bg): {dice_overall:.4f}")
    print(f"Overall PQ/DQ/SQ/AJI: PQ={pq_overall['PQ']:.4f}  DQ={pq_overall['DQ']:.4f}  SQ={pq_overall['SQ']:.4f}  AJI={pq_overall['AJI']:.4f}")
    print("Per-type DICE & PQ/DQ/SQ/AJI:")
    for c in range(1, C+1):
        name = id2name.get(c, f"class_{c}")
        print(f"  {c:>2} ({name:>27s}) | DICE={dice_perclass[c]:.4f}  "
              f"PQ={pq_perclass[c]['PQ']:.4f}  DQ={pq_perclass[c]['DQ']:.4f}  SQ={pq_perclass[c]['SQ']:.4f}  AJI={pq_perclass[c]['AJI']:.4f}")

    # ====== Save CSVs ======
    import csv
    # classification
    csv_path_cls = out_dir / "type_metrics.csv"
    with open(csv_path_cls, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id","class_name","f1","acc(per-class)","precision","recall","support"])
        for c in classes_to_eval:
            name = id2name.get(c, f"class_{c}"); m = per_class[c]
            writer.writerow([c, name, f"{m['f1']:.6f}", f"{m['acc']:.6f}",
                             f"{m['precision']:.6f}", f"{m['recall']:.6f}", m['support']])

    # segmentation (overall + per-class)
    csv_path_seg = out_dir / "segmentation_metrics.csv"
    with open(csv_path_seg, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope","class_id","class_name","DICE","PQ","DQ","SQ","AJI"])
        writer.writerow(["overall","-","-",
                         f"{dice_overall:.6f}", f"{pq_overall['PQ']:.6f}",
                         f"{pq_overall['DQ']:.6f}", f"{pq_overall['SQ']:.6f}",
                         f"{pq_overall['AJI']:.6f}"])
        for c in range(1, C+1):
            name = id2name.get(c, f"class_{c}")
            writer.writerow(["per-class", c, name,
                             f"{dice_perclass[c]:.6f}", f"{pq_perclass[c]['PQ']:.6f}",
                             f"{pq_perclass[c]['DQ']:.6f}", f"{pq_perclass[c]['SQ']:.6f}",
                             f"{pq_perclass[c]['AJI']:.6f}"])

    print(f"\n[Saved] {csv_path_cls}")
    print(f"[Saved] {csv_path_seg}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate metrics and visualize GT/PRED type maps.")
    # dataset selector
    p.add_argument('--dataset', type=str, default='consep',
                   choices=['consep', 'consep_merged', 'pannuke_hf'])

    # CoNSeP: needs root
    p.add_argument('--root', type=Path, default=None,
                   help='Dataset root containing images/ and labels/ (required for consep/consep_merged)')
    p.add_argument('--merge_consep', action='store_true',
                   help='Only for CoNSeP: view merged (3,4)->epithelial; (5,6,7)->spindle')

    # PanNuke(HF) options
    p.add_argument('--hf_repo', type=str, default='tio-ikim/pannuke')
    p.add_argument('--hf_split', type=str, default='validation')  # train/validation/test
    p.add_argument('--hf_fold', type=int, default=None)

    # common
    p.add_argument('--ckpt', type=Path, required=True, help='Trained CellViT checkpoint (.pth with key \"model\")')
    p.add_argument('--out', type=Path, default=Path('eval_out'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max_samples', type=int, default=40)
    p.add_argument('--type_alpha', type=float, default=0.45)
    p.add_argument('--exclude_bg', action='store_true', help='Do not report background in per-class metrics')

    # ViT-256 hyperparams (match your model)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--vit_embed_dim', type=int, default=384)
    p.add_argument('--vit_depth', type=int, default=12)
    p.add_argument('--vit_heads', type=int, default=6)
    p.add_argument('--vit_mlp_ratio', type=float, default=4.0)

    # num_classes (incl. background!)
    # consep(original)=8, consep(merged)=5, pannuke=6
    p.add_argument('--num_classes', type=int, default=6,
                   help='#type classes incl. background. consep=8, consep_merged=5, pannuke=6')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # quick sanity for --root
    if args.dataset in ('consep', 'consep_merged') and args.root is None:
        raise SystemExit("--root is required for CoNSeP datasets.")
    if args.dataset == 'consep' and args.merge_consep:
        print("[warn] --merge_consep has no effect when --dataset=consep_merged is available.")
    evaluate_and_visualize(args)
