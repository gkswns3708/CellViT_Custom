# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from Data.CoNSeP_patch import ConsepPatchDataset
from Model.CellViT_ViT256_Custom import CellViTCustom  # 예측/후처리에 사용

# ------------------------
# 유틸: 클래스 합치기(옵션)
# ------------------------
def build_class_map_consep(merge: bool) -> Tuple[Dict[int,int], Dict[int,str]]:
    """
    CoNSeP 라벨:
      1=other, 2=inflammatory, 3=healthy epithelial, 4=dysplastic/malignant epithelial,
      5=fibroblast, 6=muscle, 7=endothelial (0=bg)
    merge=False: 원클래스(1..7)
    merge=True : epithelial(3+4), spindle(5+6+7)로 합치기
    """
    if not merge:
        id2name = {
            1:"other", 2:"inflammatory", 3:"healthy_epith", 4:"dysplastic_epith",
            5:"fibroblast", 6:"muscle", 7:"endothelial"
        }
        remap = {k:k for k in id2name.keys()}
        return remap, id2name
    # 합친 클래스: 1 other / 2 inflammatory / 3 epithelial / 4 spindle / 5 misc? -> 논문 예시대로 4클래스
    id2name = {1:"other", 2:"inflammatory", 3:"epithelial", 4:"spindle"}
    remap = {1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:4}
    return remap, id2name

# ------------------------
# 인스턴스 뽑기/요약
# ------------------------
def extract_instances(inst_map: np.ndarray, type_map: np.ndarray, class_remap: Dict[int,int]) -> List[dict]:
    """
    inst_map: (H,W) [0=bg, 1..N]
    type_map: (H,W) [0=bg 포함]
    class_remap: {orig_id -> merged_id}
    return: [{'id':int, 'mask':bool(H,W), 'cls':int, 'centroid':(y,x)}]
    """
    assert inst_map.ndim == 2 and type_map.ndim == 2
    ids = np.unique(inst_map)
    out = []
    for i in ids:
        if i == 0:
            continue
        m = (inst_map == i)
        if not np.any(m):
            continue
        # 다수결 타입 (bg는 제외)
        cls_vals = type_map[m]
        cls_vals = cls_vals[cls_vals > 0]
        if cls_vals.size == 0:
            # 타입 정보가 없으면 other로 보정
            maj = 1
        else:
            maj = int(np.bincount(cls_vals).argmax())
        maj = class_remap.get(maj, maj)  # remap
        # centroid
        ys, xs = np.nonzero(m)
        cy = float(ys.mean()) if ys.size else 0.0
        cx = float(xs.mean()) if xs.size else 0.0
        out.append({'id': int(i), 'mask': m, 'cls': maj, 'centroid': (cy, cx)})
    return out

# ------------------------
# IoU 매칭 (PQ용) - greedy (IoU 내림차순)
# ------------------------
def pair_iou_matrix(gts: List[dict], preds: List[dict]) -> np.ndarray:
    M, N = len(gts), len(preds)
    ious = np.zeros((M, N), dtype=np.float32)
    for i, gi in enumerate(gts):
        gm = gi['mask']
        gsum = gm.sum()
        if gsum == 0: 
            continue
        for j, pj in enumerate(preds):
            pm = pj['mask']
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            uni = gsum + pm.sum() - inter
            if uni > 0:
                ious[i, j] = inter / float(uni)
    return ious

def greedy_match_iou(ious: np.ndarray, thr: float = 0.5) -> List[Tuple[int,int,float]]:
    matches = []
    used_g = set()
    used_p = set()
    # 후보를 (IoU, g, p) 내림차순 정렬
    cand = []
    M, N = ious.shape
    for g in range(M):
        for p in range(N):
            if ious[g, p] >= thr:
                cand.append((ious[g, p], g, p))
    cand.sort(reverse=True, key=lambda x: x[0])
    for iou, g, p in cand:
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p)
        matches.append((g, p, float(iou)))
    return matches

# ------------------------
# Detection 매칭 (중심점 반경) - greedy (거리 오름차순)
# ------------------------
def pair_dist_matrix(gts: List[dict], preds: List[dict]) -> np.ndarray:
    M, N = len(gts), len(preds)
    D = np.full((M, N), np.inf, dtype=np.float32)
    for i, gi in enumerate(gts):
        gy, gx = gi['centroid']
        for j, pj in enumerate(preds):
            py, px = pj['centroid']
            D[i, j] = ((gy - py) ** 2 + (gx - px) ** 2) ** 0.5
    return D

def greedy_match_dist(D: np.ndarray, radius: float) -> List[Tuple[int,int,float]]:
    matches = []
    used_g = set()
    used_p = set()
    M, N = D.shape
    # 후보 (dist, g, p) 정렬
    cand = []
    for g in range(M):
        for p in range(N):
            if D[g, p] <= radius:
                cand.append((D[g, p], g, p))
    cand.sort(key=lambda x: x[0])  # 가까운 것부터
    for dist, g, p in cand:
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p)
        matches.append((g, p, float(dist)))
    return matches

# ------------------------
# PQ / bPQ / mPQ
# ------------------------
def compute_pq(gt_all: List[dict], pr_all: List[dict], class_ids: List[int], iou_thr: float=0.5):
    # 클래스별 분할
    pq_per_c = {}
    for c in class_ids:
        gtc = [x for x in gt_all if x['cls'] == c]
        prc = [x for x in pr_all if x['cls'] == c]
        ious = pair_iou_matrix(gtc, prc)
        matches = greedy_match_iou(ious, iou_thr)
        TP = len(matches)
        FP = max(len(prc) - TP, 0)
        FN = max(len(gtc) - TP, 0)
        sum_iou = sum(m[2] for m in matches)
        denom = TP + 0.5*FP + 0.5*FN
        pq = (sum_iou / denom) if denom > 0 else 0.0
        dq = (TP / denom) if denom > 0 else 0.0
        sq = (sum_iou / TP) if TP > 0 else 0.0
        pq_per_c[c] = dict(PQ=pq, DQ=dq, SQ=sq, TP=TP, FP=FP, FN=FN, SUM_IOU=sum_iou)

    # bPQ: 클래스 무시하고 통합
    ious_b = pair_iou_matrix(gt_all, pr_all)
    matches_b = greedy_match_iou(ious_b, iou_thr)
    TPb = len(matches_b)
    FPb = max(len(pr_all) - TPb, 0)
    FNb = max(len(gt_all) - TPb, 0)
    sum_iou_b = sum(m[2] for m in matches_b)
    denom_b = TPb + 0.5*FPb + 0.5*FNb
    bPQ = (sum_iou_b / denom_b) if denom_b > 0 else 0.0

    # mPQ: 클래스 평균
    mPQ = float(np.mean([pq_per_c[c]['PQ'] for c in class_ids])) if class_ids else 0.0
    return pq_per_c, bPQ, mPQ

# ------------------------
# Detection/Classification (PanNuke 공식식)
# ------------------------
def compute_detection_and_classification(gt_all: List[dict], pr_all: List[dict],
                                         class_ids: List[int], radius_px: float):
    # detection 매칭(클래스 무시)
    D = pair_dist_matrix(gt_all, pr_all)
    matches = greedy_match_dist(D, radius_px)
    matched_gt_idx = set(m[0] for m in matches)
    matched_pr_idx = set(m[1] for m in matches)

    FPd = max(len(pr_all) - len(matched_pr_idx), 0)  # unmatched preds
    FNd = max(len(gt_all) - len(matched_gt_idx), 0)  # unmatched gts
    TPd = len(matches)

    Pd = TPd / (TPd + FPd) if (TPd + FPd) > 0 else 0.0
    Rd = TPd / (TPd + FNd) if (TPd + FNd) > 0 else 0.0
    F1d = 2*Pd*Rd / (Pd + Rd) if (Pd + Rd) > 0 else 0.0

    # PanNuke 분해: 클래스별 TPc, FNc, FPc, TNc
    per_c = {}
    for c in class_ids:
        TPc = FNc = FPc = TNc = 0
        for (gi, pj, _) in matches:
            g_cls = gt_all[gi]['cls']
            p_cls = pr_all[pj]['cls']
            if g_cls == c and p_cls == c:
                TPc += 1
            elif g_cls == c and p_cls != c:
                FNc += 1
            elif g_cls != c and p_cls == c:
                FPc += 1
            elif g_cls != c and p_cls != c and g_cls == p_cls:
                # 다른 클래스이지만 맞게 분류된 매칭 → TNc로 취급 (PanNuke 가중식)
                TNc += 1
            # (그 외: 다른클래스-다른클래스-오분류는 TN/TP/FP/FN에 기여하지 않음)

        # PanNuke 공식식 (논문 4.3절)
        # Pc = (TPc+TNc) / (TPc+TNc + 2*FPc + FPd)
        # Rc = (TPc+TNc) / (TPc+TNc + 2*FNc + FNd)
        # F1c = 2*(TPc+TNc) / [ 2*(TPc+TNc) + 2*FPc + 2*FNc + FPd + FNd ]
        num = (TPc + TNc)
        Pc = num / (num + 2*FPc + FPd) if (num + 2*FPc + FPd) > 0 else 0.0
        Rc = num / (num + 2*FNc + FNd) if (num + 2*FNc + FNd) > 0 else 0.0
        F1c = (2*num) / (2*num + 2*FPc + 2*FNc + FPd + FNd) if (2*num + 2*FPc + 2*FNc + FPd + FNd) > 0 else 0.0
        per_c[c] = dict(P=Pc, R=Rc, F1=F1c, TPc=TPc, FNc=FNc, FPc=FPc, TNc=TNc)

    return dict(Pd=Pd, Rd=Rd, F1d=F1d, FPd=FPd, FNd=FNd, TPd=TPd), per_c

# ------------------------
# 메인 평가 파이프라인
# ------------------------
@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 데이터
    ds = ConsepPatchDataset(args.root, num_nuclei_classes=args.num_classes)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 클래스 맵/이름
    if args.dataset.lower() == "consep":
        remap, id2name = build_class_map_consep(args.merge_consep)
        class_ids = sorted(set(remap.values()))
    else:
        # PanNuke (5 classes): 1..5 라고 가정 (실 사용 시 데이터셋 라벨에 맞게 수정)
        class_ids = list(range(1, args.num_classes))
        id2name = {1:"neoplastic", 2:"inflammatory", 3:"epithelial", 4:"dead", 5:"connective"}
        remap = {k:k for k in class_ids}

    # 모델
    model = CellViTCustom(num_nuclei_classes=args.num_classes)
    if args.ckpt is not None and Path(args.ckpt).exists():
        ck = torch.load(str(args.ckpt), map_location='cpu')
        sd = ck.get('model', ck)  # state_dict 혹은 직접 state_dict 저장일 때 대응
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    all_pq_counts = []   # (gt_insts, pr_insts) 를 누적하지 않고 바로 최종에서 합산 계산 -> PQ는 전역 단일 매칭이 아님에 유의
    # detection/classification은 전 이미지 합쳐서 매칭 → 전역 모드로 구현(간단히 concat)
    GT_concat: List[dict] = []
    PR_concat: List[dict] = []

    for batch in tqdm(dl, ncols=100, desc="Eval"):
        x = batch['image'].to(device, non_blocking=True)
        gt_inst = batch['inst_map'].numpy()          # (B,H,W)
        gt_type = batch['type_map'].numpy()          # (B,H,W)

        # 예측
        y = model(x)
        # 인스턴스 후처리 (HV/NP/NT를 결합) – 모델의 helper 사용
        try:
            pred_inst_batch, _ = model.calculate_instance_map(y)  # Tensor(B,H,W), _
            pred_inst_batch = pred_inst_batch.cpu().numpy()
        except Exception as e:
            raise RuntimeError("DetectionCellPostProcessor가 필요합니다. post_utils가 설치/임포트 가능한지 확인하세요.") from e

        # 픽셀 단위 타입 예측 (NT argmax)
        pred_type_batch = torch.argmax(y['nuclei_type_map'], dim=1).cpu().numpy()  # (B,H,W)

        B = x.size(0)
        for b in range(B):
            # remap 타입
            if args.dataset.lower() == "consep":
                vt = np.vectorize(lambda t: remap.get(int(t), int(t)))
                gt_type_b = vt(gt_type[b])
                pr_type_b = vt(pred_type_batch[b])
            else:
                gt_type_b = gt_type[b]
                pr_type_b = pred_type_batch[b]

            g_list = extract_instances(gt_inst[b], gt_type_b, class_remap={i:i for i in range(100)})
            p_list = extract_instances(pred_inst_batch[b], pr_type_b, class_remap={i:i for i in range(100)})

            # 클래스 밖(=0/bad)은 제거
            g_list = [g for g in g_list if g['cls'] in class_ids]
            p_list = [p for p in p_list if p['cls'] in class_ids]

            # PQ는 이미지별로 계산 후(클래스 단위), 나중에 평균이 아니라 “전역 집계”가 더 보수적.
            # 여기서는 PanNuke 관행에 맞춰 전 데이터 합쳐 단일 PQ를 내기 위해 누적 리스트 사용.
            GT_concat.extend(g_list)
            PR_concat.extend(p_list)

    # 최종 계산
    pq_per_c, bPQ, mPQ = compute_pq(GT_concat, PR_concat, class_ids, iou_thr=args.iou_thr)
    det_global, cls_per_c = compute_detection_and_classification(GT_concat, PR_concat, class_ids, radius_px=args.det_radius)

    # 출력
    print("\n=== Panoptic Quality (IoU>{:.2f}) ===".format(args.iou_thr))
    for c in class_ids:
        nm = id2name.get(c, str(c))
        d = pq_per_c[c]
        print(f"[{c}:{nm}] PQ={d['PQ']:.3f}  (DQ={d['DQ']:.3f}, SQ={d['SQ']:.3f})  TP={d['TP']} FP={d['FP']} FN={d['FN']}")
    print(f"bPQ={bPQ:.3f} | mPQ={mPQ:.3f}")

    print("\n=== Detection (centroid match, radius={:.1f}px) ===".format(args.det_radius))
    print("Pd={:.3f}  Rd={:.3f}  F1d={:.3f}  (TPd={} FPd={} FNd={})".format(
        det_global['Pd'], det_global['Rd'], det_global['F1d'],
        det_global['TPd'], det_global['FPd'], det_global['FNd']
    ))

    print("\n=== Classification per class (PanNuke metric) ===")
    for c in class_ids:
        nm = id2name.get(c, str(c))
        d = cls_per_c[c]
        print(f"[{c}:{nm}] P={d['P']:.3f}  R={d['R']:.3f}  F1={d['F1']:.3f}  (TPc={d['TPc']} TNc={d['TNc']} FPc={d['FPc']} FNc={d['FNc']})")


def parse_args():
    p = argparse.ArgumentParser(description="Cell-type metrics (PQ / bPQ / mPQ + detection/classification) as in CellViT paper")
    p.add_argument('--root', type=Path, required=True, help='Dataset root containing images/ and labels/')
    p.add_argument('--ckpt', type=Path, required=True, help='Trained CellViT checkpoint (.pth with state_dict under key "model")')
    p.add_argument('--dataset', type=str, default='consep', choices=['consep', 'pannuke'])
    p.add_argument('--merge_consep', action='store_true', help='Merge CoNSeP classes: (3,4)->epithelial; (5,6,7)->spindle')
    p.add_argument('--num_classes', type=int, default=8, help='include bg (0). For CoNSeP raw 8 (0..7).')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    # Metric params
    p.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold for PQ matching')
    p.add_argument('--det_radius', type=float, default=12.0, help='Centroid match radius in pixels (12 for 0.25µm/px, 6 for 0.50µm/px)')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
