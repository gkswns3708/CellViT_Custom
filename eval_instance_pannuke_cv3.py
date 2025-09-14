# -*- coding: utf-8 -*-
"""
PanNuke 3-fold CV 인스턴스 수준 평가 (CellViTCustom)

- HF Hub의 RationAI/PanNuke 로부터 fold1/2/3를 읽습니다.
- 각 fold에 대해:
    * 해당 fold의 체크포인트로 추론
    * 모델의 post processor(DetectionCellPostProcessor)를 통해 인스턴스 분할
    * 매칭 방법 선택:
        - --match iou     : GT/Pred 인스턴스를 IoU≥thr 로 1:1 그리디 매칭
        - --match paper   : 인스턴스 중심점(centroid) 거리 ≤ 반경(px) 으로 1:1 그리디 매칭
                            (반경 미지정 시 배율(x40/ x20)에 따라 자동: 12px / 6px)
    * Detection Pd/Rd/F1d 집계
    * 분류 지표:
        - --match iou   : "매칭된 인스턴스"만으로 per-class P/R/F1 (tp/pred/gt 기반)
        - --match paper : PanNuke/CellViT 논문식(TPc/TNc/FPc/FNc, FPd/FNd 포함) per-class P/R/F1
- 3개 fold 결과 평균 출력 + CSV 저장
"""
from __future__ import annotations
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from Model.CellViT_ViT256_Custom import CellViTCustom


# ----------------------------
# DataLoader collate (이미지 Tensor, 나머지는 list로 유지)
# ----------------------------
def _collate_eval(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)
    return {
        "image": images,
        "gt_inst_map": [b["gt_inst_map"] for b in batch],  # list[np.ndarray]
        "gt_types":    [b["gt_types"] for b in batch],     # list[dict]
        "name":        [b["name"] for b in batch],         # list[str]
    }


# ----------------------------
# Pred 타입 dict 안전 파서 (postproc 출력 다양성 대응)
# ----------------------------
def _extract_cls_from_value(v):
    """value가 int/float/dict(확률)/list/np.ndarray일 수 있으니 클래스 id(int)를 추출."""
    import numpy as _np
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(round(float(v)))
    if isinstance(v, dict):
        for key in ('type', 'class', 'cls', 'label'):
            if key in v and isinstance(v[key], (int, float, np.integer, np.floating)):
                return int(v[key])
        for key in ('prob', 'probs', 'logits', 'type_prob', 'type_probs'):
            if key in v:
                arr = _np.asarray(v[key])
                if arr.size > 0:
                    return int(arr.argmax())
        return None
    if isinstance(v, (list, tuple, _np.ndarray)):
        arr = _np.asarray(v)
        if arr.size > 0:
            return int(arr.argmax())
        return None
    return None


def _majority_type_from_maps(type_argmax_map: np.ndarray, inst_map: np.ndarray, inst_id: int) -> int:
    """inst_id 영역에서 type_argmax 최빈값(>0)을 사용. 없으면 0."""
    mask = (inst_map == inst_id)
    vals = type_argmax_map[mask]
    if vals.size == 0:
        return 0
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(np.bincount(vals).argmax())


def _normalize_pred_types(pred_type_entry, type_argmax_map: np.ndarray, inst_map: np.ndarray) -> dict[int, int]:
    """
    post-processor가 돌려준 pred_type_entry(보통 dict)를 id->int class로 표준화.
    value가 dict/리스트/확률벡터여도 처리. 실패하면 majority fallback.
    """
    res: dict[int, int] = {}
    if isinstance(pred_type_entry, dict):
        items = pred_type_entry.items()
    elif isinstance(pred_type_entry, list):
        items = []
        for it in pred_type_entry:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                items.append((it[0], it[1]))
    else:
        items = []

    for k, v in items:
        try:
            inst_id = int(k)
        except Exception:
            continue
        cls_id = _extract_cls_from_value(v)
        if cls_id is None:
            cls_id = _majority_type_from_maps(type_argmax_map, inst_map, inst_id)
        res[inst_id] = int(cls_id)
    return res


# ----------------------------
# PanNuke(HF) fold dataset
# ----------------------------
class PanNukeHFFoldDataset(Dataset):
    """
    RationAI/PanNuke의 한 fold(fold1/2/3)를 인스턴스 GT로 변환하는 래퍼.
    각 item:
      {
        'image': Tensor float (3,H,W) [0..1],
        'gt_inst_map': np.ndarray (H,W) int32  # 1..N
        'gt_types': Dict[int,int]  # instance_id -> class_id (1..5)
        'name': str
      }
    """
    def __init__(self, repo_id: str, fold: str, normalize: bool = True) -> None:
        super().__init__()
        assert fold in ("fold1", "fold2", "fold3"), "fold must be one of fold1/fold2/fold3"
        self.ds = load_dataset(repo_id, split=fold)
        self.normalize = normalize

    def __len__(self) -> int:
        return self.ds.num_rows

    @staticmethod
    def _build_gt_from_record(rec) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        rec['instances']: list of PIL 1-bit masks (256x256), each exactly one nucleus.
        rec['categories']: list of class ids in [0..4], mapping to [1..5].
        returns:
          gt_inst_map: (H,W) ints, 0=bg, 1..N instances
          gt_types: {inst_id -> class_id(1..5)}
        """
        H = W = 256
        inst_map = np.zeros((H, W), dtype=np.int32)
        types: Dict[int, int] = {}
        inst_imgs = rec["instances"]
        cats = rec["categories"]
        assert len(inst_imgs) == len(cats), "instances and categories length mismatch"

        next_id = 1
        for pil_bin, c0 in zip(inst_imgs, cats):
            m = np.array(pil_bin.convert("1"), dtype=np.uint8)  # 0/1
            if m.max() == 0:
                continue
            inst_map[m > 0] = next_id
            types[next_id] = int(c0) + 1  # 0..4 -> 1..5
            next_id += 1
        return inst_map, types

    def __getitem__(self, idx: int):
        rec = self.ds[int(idx)]
        img: Image.Image = rec["image"]
        img_np = np.array(img).astype(np.uint8)
        if img_np.ndim == 2:
            img_np = np.repeat(img_np[..., None], 3, axis=2)

        x = torch.from_numpy(img_np).float().permute(2, 0, 1)  # (3,H,W)
        if self.normalize and x.max() > 1.0:
            x = x / 255.0

        gt_inst, gt_types = self._build_gt_from_record(rec)
        name = f"{self.ds.split}_{idx:06d}" if hasattr(self.ds, "split") else f"pannuke_{idx:06d}"
        return {
            "image": x,
            "gt_inst_map": gt_inst,
            "gt_types": gt_types,
            "name": name,
        }


# ----------------------------
# IoU 매칭 유틸
# ----------------------------
def iou_matrix(gt_map: np.ndarray, pr_map: np.ndarray) -> np.ndarray:
    """Compute IoU for all GT x Pred instance pairs."""
    gt_ids = [i for i in np.unique(gt_map) if i != 0]
    pr_ids = [j for j in np.unique(pr_map) if j != 0]
    G, P = len(gt_ids), len(pr_ids)
    if G == 0 or P == 0:
        return np.zeros((G, P), dtype=np.float32)

    iou = np.zeros((G, P), dtype=np.float32)
    pr_masks = {pid: (pr_map == pid) for pid in pr_ids}
    for gi, gid in enumerate(gt_ids):
        gmask = (gt_map == gid)
        g_area = gmask.sum()
        if g_area == 0:
            continue
        for pj, pid in enumerate(pr_ids):
            pmask = pr_masks[pid]
            inter = np.logical_and(gmask, pmask).sum()
            if inter == 0:
                continue
            union = g_area + pmask.sum() - inter
            if union > 0:
                iou[gi, pj] = inter / union
    return iou


def greedy_match(iou: np.ndarray, thr: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy 1:1 matching by descending IoU, keep only IoU >= thr."""
    if iou.size == 0:
        return []
    G, P = iou.shape
    flat = [(float(iou[gi, pj]), gi, pj)
            for gi in range(G) for pj in range(P) if iou[gi, pj] >= thr]
    flat.sort(reverse=True, key=lambda t: t[0])

    used_g = set()
    used_p = set()
    pairs = []
    for v, gi, pj in flat:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        pairs.append((gi, pj, v))
    return pairs


# ----------------------------
# 중심점(centroid) 매칭 유틸
# ----------------------------
def _centroids_from_inst_map(inst_map: np.ndarray) -> dict[int, tuple[float, float]]:
    """각 인스턴스 id -> (y, x) 중심점(centroid)."""
    ids = [i for i in np.unique(inst_map) if i != 0]
    cents = {}
    for i in ids:
        ys, xs = np.nonzero(inst_map == i)
        if ys.size == 0:
            continue
        cents[i] = (float(ys.mean()), float(xs.mean()))
    return cents

def _cdist(Y1: List[tuple[float,float]], Y2: List[tuple[float,float]]) -> np.ndarray:
    """유클리드 거리 행렬 (len(Y1) x len(Y2))."""
    if len(Y1) == 0 or len(Y2) == 0:
        return np.zeros((len(Y1), len(Y2)), dtype=np.float32)
    A = np.asarray(Y1, dtype=np.float32)
    B = np.asarray(Y2, dtype=np.float32)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

def greedy_centroid_match(gt_map: np.ndarray, pr_map: np.ndarray, radius_px: float) -> List[Tuple[int,int,float]]:
    """
    중심점 거리 기반 1:1 그리디 매칭.
    반환: [(gt_idx, pr_idx, dist), ...], dist <= radius_px
    """
    gt_ids = [i for i in np.unique(gt_map) if i != 0]
    pr_ids = [j for j in np.unique(pr_map) if j != 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return []

    gt_c = _centroids_from_inst_map(gt_map)
    pr_c = _centroids_from_inst_map(pr_map)
    G, P = len(gt_ids), len(pr_ids)

    Yg = [gt_c[i] for i in gt_ids]
    Yp = [pr_c[j] for j in pr_ids]
    D = _cdist(Yg, Yp)

    cand = [(float(D[gi, pj]), gi, pj) for gi in range(G) for pj in range(P) if D[gi, pj] <= radius_px]
    cand.sort(key=lambda t: t[0])  # 가까운 것부터

    used_g, used_p, pairs = set(), set(), []
    for dist, gi, pj in cand:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi); used_p.add(pj)
        pairs.append((gi, pj, dist))
    return pairs


# ----------------------------
# Accumulator: IoU 방식 (matched-only per-class)
# ----------------------------
class InstanceMetricsAccumulatorIOU:
    """Detection P/R/F1 + per-class P/R/F1 (matched nuclei only, IoU 매칭)."""
    def __init__(self, num_classes_including_bg: int = 6) -> None:
        self.tp_d = 0; self.fp_d = 0; self.fn_d = 0
        C = num_classes_including_bg - 1
        self.C = C
        self.tp_c = np.zeros(C + 1, dtype=np.int64)
        self.pred_c = np.zeros(C + 1, dtype=np.int64)
        self.gt_c = np.zeros(C + 1, dtype=np.int64)

    def add_image(
        self,
        gt_map: np.ndarray,
        gt_types: Dict[int, int],
        pr_map: np.ndarray,
        pr_types: Dict[int, int],
        iou_thr: float = 0.5
    ):
        gt_ids = [i for i in np.unique(gt_map) if i != 0]
        pr_ids = [j for j in np.unique(pr_map) if j != 0]

        M = iou_matrix(gt_map, pr_map)
        matches = greedy_match(M, thr=iou_thr)

        tp = len(matches)
        fp = max(len(pr_ids) - tp, 0)
        fn = max(len(gt_ids) - tp, 0)
        self.tp_d += tp; self.fp_d += fp; self.fn_d += fn

        # Classification on matched only
        for gi, pj, _ in matches:
            gid = gt_ids[gi]; pid = pr_ids[pj]
            gcls = int(gt_types.get(gid, 0))
            pcls = int(pr_types.get(pid, 0))
            if 1 <= gcls <= self.C:
                self.gt_c[gcls] += 1
            if 1 <= pcls <= self.C:
                self.pred_c[pcls] += 1
            if 1 <= gcls <= self.C and gcls == pcls:
                self.tp_c[gcls] += 1

    def summarize(self) -> Dict[str, float | Dict[int, Dict[str, float]]]:
        eps = 1e-9
        Pd = self.tp_d / (self.tp_d + self.fp_d + eps)
        Rd = self.tp_d / (self.tp_d + self.fn_d + eps)
        F1d = 2 * Pd * Rd / (Pd + Rd + eps)

        per_class = {}
        for k in range(1, self.C + 1):
            tp = self.tp_c[k]; pred = self.pred_c[k]; gt = self.gt_c[k]
            P = tp / (pred + eps)
            R = tp / (gt + eps)
            F1 = 2 * P * R / (P + R + eps) if (P + R) > 0 else 0.0
            per_class[k] = {"P": float(P), "R": float(R), "F1": float(F1), "support": int(gt)}

        return {
            "Pd": float(Pd), "Rd": float(Rd), "F1d": float(F1d),
            "per_class": per_class,
            "det_counts": {"TP": self.tp_d, "FP": self.fp_d, "FN": self.fn_d},
        }


# ----------------------------
# Accumulator: 논문 방식 (centroid + PanNuke/CellViT 식)
# ----------------------------
class InstanceMetricsAccumulatorPaper:
    """
    Detection Pd/Rd/F1d + per-class Pc/Rc/F1c (PanNuke/CellViT 식).
    class ids: 1..C (PanNuke: C=5)
    """
    def __init__(self, num_classes_including_bg: int = 6) -> None:
        C = num_classes_including_bg - 1
        self.C = C
        # detection
        self.tp_d = 0; self.fp_d = 0; self.fn_d = 0
        # class terms
        self.TPc = np.zeros(C + 1, dtype=np.int64)   # index 0 unused
        self.TNc = np.zeros(C + 1, dtype=np.int64)
        self.FPc = np.zeros(C + 1, dtype=np.int64)
        self.FNc = np.zeros(C + 1, dtype=np.int64)
        self.gt_support = np.zeros(C + 1, dtype=np.int64)

    def add_image(
        self,
        gt_map: np.ndarray,
        gt_types: Dict[int, int],
        pr_map: np.ndarray,
        pr_types: Dict[int, int],
        magnification: int = 40,
        radius_override: Optional[float] = None,
    ):
        # 반경(px): override 우선, 없으면 배율 기반 자동
        if radius_override is not None:
            radius_px = float(radius_override)
        else:
            radius_px = 12.0 if magnification >= 30 else 6.0

        gt_ids = [i for i in np.unique(gt_map) if i != 0]
        pr_ids = [j for j in np.unique(pr_map) if j != 0]

        # GT per-class support
        for gid in gt_ids:
            c = int(gt_types.get(gid, 0))
            if 1 <= c <= self.C:
                self.gt_support[c] += 1

        # 중심점 매칭
        matches = greedy_centroid_match(gt_map, pr_map, radius_px)
        gid_list = gt_ids; pid_list = pr_ids
        matched_g = set(gid_list[gi] for gi, _, _ in matches)
        matched_p = set(pid_list[pj] for _, pj, _ in matches)

        # detection TP/FP/FN
        TPd = len(matches)
        FPd = max(len(pr_ids) - TPd, 0)
        FNd = max(len(gt_ids) - TPd, 0)
        self.tp_d += TPd; self.fp_d += FPd; self.fn_d += FNd

        # 클래스 분해
        # 1) 매칭된 쌍
        for gi, pj, _ in matches:
            gid = gid_list[gi]; pid = pid_list[pj]
            gcls = int(gt_types.get(gid, 0))
            pcls = int(pr_types.get(pid, 0))
            if 1 <= gcls <= self.C and 1 <= pcls <= self.C:
                if gcls == pcls:
                    self.TPc[gcls] += 1
                    for c in range(1, self.C + 1):
                        if c != gcls:
                            self.TNc[c] += 1
                else:
                    self.FPc[pcls] += 1
                    self.FNc[gcls] += 1
            elif 1 <= gcls <= self.C and not (1 <= pcls <= self.C):
                self.FNc[gcls] += 1
            elif 1 <= pcls <= self.C and not (1 <= gcls <= self.C):
                self.FPc[pcls] += 1

        # 2) 매칭 실패 항목을 클래스별로 분배
        for pid in pr_ids:
            if pid not in matched_p:
                pcls = int(pr_types.get(pid, 0))
                if 1 <= pcls <= self.C:
                    self.FPc[pcls] += 1
        for gid in gt_ids:
            if gid not in matched_g:
                gcls = int(gt_types.get(gid, 0))
                if 1 <= gcls <= self.C:
                    self.FNc[gcls] += 1

    def summarize(self) -> Dict[str, float | Dict[int, Dict[str, float]]]:
        eps = 1e-9
        Pd = self.tp_d / (self.tp_d + self.fp_d + eps)
        Rd = self.tp_d / (self.tp_d + self.fn_d + eps)
        F1d = 2 * Pd * Rd / (Pd + Rd + eps)

        per_class = {}
        for c in range(1, self.C + 1):
            TPc = self.TPc[c]; TNc = self.TNc[c]
            FPc = self.FPc[c]; FNc = self.FNc[c]
            Pc = (TPc + TNc) / (TPc + TNc + 2*FPc + self.fp_d + eps)
            Rc = (TPc + TNc) / (TPc + TNc + 2*FNc + self.fn_d + eps)
            F1 = 2*(TPc + TNc) / (2*(TPc + TNc) + 2*FPc + 2*FNc + self.fp_d + self.fn_d + eps)
            per_class[c] = {"P": float(Pc), "R": float(Rc), "F1": float(F1), "support": int(self.gt_support[c])}

        return {
            "Pd": float(Pd), "Rd": float(Rd), "F1d": float(F1d),
            "per_class": per_class,
            "det_counts": {"TP": int(self.tp_d), "FP": int(self.fp_d), "FN": int(self.fn_d)},
        }


# ----------------------------
# One fold evaluation
# ----------------------------
@torch.no_grad()
def eval_one_fold(
    repo_id: str,
    fold: str,
    ckpt_path: Path,
    model_cfg: dict,
    device: torch.device,
    batch_size: int,
    workers: int,
    magnification: int = 40,
    iou_thr: float = 0.5,
    use_postproc_types: bool = True,
    match_method: str = "paper",                 # 'iou' or 'paper'
    radius_px: Optional[float] = None,           # used only if match_method='paper'
) -> Dict[str, float | Dict]:
    ds = PanNukeHFFoldDataset(repo_id=repo_id, fold=fold, normalize=True)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate_eval,
    )

    # Model
    model = CellViTCustom(
        num_nuclei_classes=model_cfg["num_classes"],  # 6 (bg+5)
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
    ).to(device).eval()

    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck.get("model", ck)
    model.load_state_dict(sd, strict=False)
    print(f"[fold {fold}] loaded ckpt: {ckpt_path}")

    if match_method == "iou":
        acc = InstanceMetricsAccumulatorIOU(num_classes_including_bg=model_cfg["num_classes"])
    else:
        acc = InstanceMetricsAccumulatorPaper(num_classes_including_bg=model_cfg["num_classes"])

    for batch in tqdm(dl):
        x = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
        gt_inst_maps = batch["gt_inst_map"]               # list[np.ndarray]
        gt_types_list = batch["gt_types"]

        # forward
        y = model(x)

        # instance postproc
        pred_inst_maps_t, pred_type_dicts = model.calculate_instance_map(
            y, magnification=magnification  # PanNuke is x40
        )
        pred_inst_maps = pred_inst_maps_t.cpu().numpy().astype(np.int32)

        # fallback용 pixel-wise type argmax
        type_argmax = torch.argmax(y["nuclei_type_map"], dim=1).cpu().numpy()  # (B,H,W)

        for b in range(x.size(0)):
            gt_map = gt_inst_maps[b]
            pr_map = pred_inst_maps[b]
            ta_map = type_argmax[b]
            gt_types: Dict[int, int] = gt_types_list[b]

            if use_postproc_types:
                pr_types = _normalize_pred_types(pred_type_dicts[b], ta_map, pr_map)
            else:
                pr_types = {}
                ids = [i for i in np.unique(pr_map) if i != 0]
                for pid in ids:
                    pr_types[pid] = _majority_type_from_maps(ta_map, pr_map, pid)

            if isinstance(acc, InstanceMetricsAccumulatorIOU):
                acc.add_image(gt_map, gt_types, pr_map, pr_types, iou_thr=iou_thr)
            else:
                acc.add_image(gt_map, gt_types, pr_map, pr_types,
                              magnification=magnification, radius_override=radius_px)

    return acc.summarize()


# ----------------------------
# 3-fold wrapper
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PanNuke 3-fold instance-level evaluation (CellViTCustom)")
    p.add_argument("--hf_repo", type=str, default="RationAI/PanNuke")

    p.add_argument("--ckpt_root", type=Path,
                   default=Path("/workspace/CellViT_Custom/Checkpoints/CellViT/cv3"),
                   help="폴더 내부에 cv3_fold1/best.pth ... 가 있다고 가정")
    p.add_argument("--ckpt_list", type=str, default="",
                   help="쉼표로 구분된 3개 경로를 직접 지정 (fold1,fold2,fold3 순서)")

    p.add_argument("--out", type=Path, default=Path("eval_instance_cv3"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)

    # Model hyperparams (must match training)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--num_classes", type=int, default=6, help="PanNuke: bg+5 classes = 6")

    # Matching options
    p.add_argument("--match", type=str, default="paper", choices=["iou", "paper"],
                   help="매칭 방법 선택: 'iou' 또는 'paper'(centroid)")
    p.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold (match=iou 일 때 사용)")
    p.add_argument("--magnification", type=int, default=40, help="PanNuke 배율 (match=paper 기본 반경 결정)")
    p.add_argument("--radius_px", type=float, default=None,
                   help="match=paper일 때 중심점 반경(px) 수동 지정 (미지정 시 배율로 자동: x40→12, x20→6)")

    p.add_argument("--no_postproc_types", action="store_true",
                   help="postproc 타입 dict 사용 끄고 majority(type-argmax)로만 예측 타입 결정")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    folds = ["fold1", "fold2", "fold3"]

    # resolve ckpts
    if args.ckpt_list:
        parts = [Path(s.strip()) for s in args.ckpt_list.split(",")]
        assert len(parts) == 3, "--ckpt_list must contain 3 paths for fold1,fold2,fold3"
        ckpts: Dict[str, Path] = {f: parts[i] for i, f in enumerate(folds)}
    else:
        ckpts = {}
        for f in folds:
            n = int(f.replace("fold", ""))
            ckpts[f] = args.ckpt_root / f"cv3_fold{n}" / "best.pth"

    # model cfg
    model_cfg = dict(
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )

    # run folds
    fold_results = {}
    for f in folds:
        ck = ckpts[f]
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint for {f} not found: {ck}")
        res = eval_one_fold(
            repo_id=args.hf_repo,
            fold=f,
            ckpt_path=ck,
            model_cfg=model_cfg,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            magnification=args.magnification,
            iou_thr=args.iou_thr,
            use_postproc_types=not args.no_postproc_types,
            match_method=args.match,
            radius_px=args.radius_px,
        )
        fold_results[f] = res
        print(f"\n[{f}] Detection: Pd={res['Pd']:.3f}  Rd={res['Rd']:.3f}  F1d={res['F1d']:.3f}")
        names = ["Neoplastic","Inflammatory","Connective","Dead","Epithelial"]
        for k, name in enumerate(names, start=1):
            pc = res["per_class"][k]
            print(f"  {name:>12s}: P={pc['P']:.3f}  R={pc['R']:.3f}  F1={pc['F1']:.3f} (n={pc['support']})")

    # average across folds
    def avg(vals: List[float]) -> float:
        return float(np.mean(vals)) if len(vals) else 0.0

    avg_det = {
        "Pd": avg([fold_results[f]["Pd"] for f in folds]),
        "Rd": avg([fold_results[f]["Rd"] for f in folds]),
        "F1d": avg([fold_results[f]["F1d"] for f in folds]),
    }
    avg_cls = {}
    for k in range(1, args.num_classes):  # 1..5
        P = avg([fold_results[f]["per_class"][k]["P"] for f in folds])
        R = avg([fold_results[f]["per_class"][k]["R"] for f in folds])
        F1 = avg([fold_results[f]["per_class"][k]["F1"] for f in folds])
        support = int(np.sum([fold_results[f]["per_class"][k]["support"] for f in folds]))
        avg_cls[k] = {"P": P, "R": R, "F1": F1, "support": support}

    print("\n=== 3-Fold Average (PanNuke, instance-level) ===")
    print(f"Detection: Pd={avg_det['Pd']:.3f}  Rd={avg_det['Rd']:.3f}  F1d={avg_det['F1d']:.3f}")
    names = ["Neoplastic","Inflammatory","Connective","Dead","Epithelial"]
    for k, name in enumerate(names, start=1):
        pc = avg_cls[k]
        print(f"  {name:>12s}: P={pc['P']:.3f}  R={pc['R']:.3f}  F1={pc['F1']:.3f} (n={pc['support']})")

    # save CSVs
    import csv
    out_dir = args.out
    # per-fold detection
    with open(out_dir / "detection_per_fold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "Pd", "Rd", "F1d", "TP", "FP", "FN"])
        for f in folds:
            detc = fold_results[f]["det_counts"]
            w.writerow([f, f"{fold_results[f]['Pd']:.6f}", f"{fold_results[f]['Rd']:.6f}",
                        f"{fold_results[f]['F1d']:.6f}", detc["TP"], detc["FP"], detc["FN"]])
        w.writerow(["avg", f"{avg_det['Pd']:.6f}", f"{avg_det['Rd']:.6f}", f"{avg_det['F1d']:.6f}", "", "", ""])
    # per-fold classification
    with open(out_dir / "classification_per_fold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "class_id", "class_name", "P", "R", "F1", "support"])
        for f in folds:
            for k, name in enumerate(names, start=1):
                pc = fold_results[f]["per_class"][k]
                w.writerow([f, k, name, f"{pc['P']:.6f}", f"{pc['R']:.6f}", f"{pc['F1']:.6f}", pc["support"]])
        # avg
        for k, name in enumerate(names, start=1):
            pc = avg_cls[k]
            w.writerow(["avg", k, name, f"{pc['P']:.6f}", f"{pc['R']:.6f}", f"{pc['F1']:.6f}", pc["support"]])

    print(f"\n[Saved] {out_dir/'detection_per_fold.csv'}")
    print(f"[Saved] {out_dir/'classification_per_fold.csv'}")


if __name__ == "__main__":
    main()
