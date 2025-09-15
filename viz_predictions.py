# -*- coding: utf-8 -*-
"""
CoNSeP / PanNuke 시각화 스크립트
- PanNuke: CV3(3-fold) 각 fold의 테스트 세트 전체를 inference & 시각화 저장
- CoNSeP: 타입 맵 기반 분류 시각화

Usage examples
--------------
# PanNuke, 논문 방식(centroid) 매칭, x40 -> 자동 반경 12px
python vis_eval.py --dataset pannuke_hf \
  --hf_repo RationAI/PanNuke \
  --ckpt_root /workspace/CellViT_Custom/Checkpoints/CellViT/cv3 \
  --out vis_out --match paper --magnification 40

# PanNuke, IoU 매칭
python vis_eval.py --dataset pannuke_hf --match iou --iou_thr 0.5

# CoNSeP (원래 예시와 동일한 타입 분류 시각화)
python vis_eval.py --dataset consep --root /data/CoNSeP --ckpt /path/to/model.pth --out vis_out_consep
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# -----------------------------
# (필요 시) 로컬 데이터셋
# -----------------------------
try:
    from Data.CoNSeP_patch import ConsepPatchDataset
    from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged
except Exception:
    ConsepPatchDataset = None
    ConsepPatchDatasetMerged = None

# -----------------------------
# 모델
# -----------------------------
from Model.CellViT_ViT256_Custom import CellViTCustom


# -----------------------------
# 공통: 라벨 스펙
# -----------------------------
def label_spec(dataset: str, merge_consep: bool) -> Tuple[Dict[int,str], Dict[int,Tuple[int,int,int]], Dict[int,int]]:
    dataset = dataset.lower()
    if dataset == 'consep':
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
        # merged
        id2name = {1:"other", 2:"inflammatory", 3:"epithelial (3+4)", 4:"spindle (5+6+7)"}
        remap = {1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:4}
        palette = {0:(0,0,0), 1:(76,153,0), 2:(255,127,0), 3:(0,176,240), 4:(255,0,128)}
        return id2name, palette, remap

    elif dataset == 'pannuke_hf':
        id2name = {1:"Neoplastic", 2:"Inflammatory", 3:"Connective", 4:"Dead", 5:"Epithelial"}
        remap = {k:k for k in id2name.keys()}
        palette = {
            0:(0,0,0), 1:(0,176,240), 2:(255,127,0), 3:(76,153,0),
            4:(128,128,128), 5:(255,0,128)
        }
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


# -----------------------------
# PanNuke HF fold dataset (자체 래퍼)
# -----------------------------
class PanNukeHFFoldDataset(torch.utils.data.Dataset):
    """
    HF Hub 'RationAI/PanNuke' 의 foldN(split='fold1/2/3')을 로드, 인스턴스 GT 구성
    item:
      {
        'image': Tensor float (3,H,W) [0..1],
        'gt_inst_map': np.ndarray (H,W), 0=bg, 1..N,
        'gt_types': Dict[int,int], 1..N -> class_id(1..5),
        'type_map': np.ndarray (H,W), 픽셀 타입(majority) [선택적, 여기선 보조]
        'name': str
      }
    """
    def __init__(self, repo_id: str, fold: str, normalize: bool=True):
        super().__init__()
        from datasets import load_dataset
        assert fold in ("fold1","fold2","fold3")
        self.ds = load_dataset(repo_id, split=fold)
        self.normalize = normalize

    def __len__(self) -> int:
        return self.ds.num_rows

    @staticmethod
    def _build_gt_from_record(rec) -> Tuple[np.ndarray, Dict[int,int], np.ndarray]:
        H = W = 256
        inst_map = np.zeros((H, W), dtype=np.int32)
        types: Dict[int, int] = {}
        type_map = np.zeros((H, W), dtype=np.int32)

        inst_imgs = rec["instances"]           # list of PIL 1-bit masks
        cats = rec["categories"]               # list of ints (0..4)
        assert len(inst_imgs) == len(cats)
        next_id = 1
        for pil_bin, c0 in zip(inst_imgs, cats):
            m = np.array(pil_bin.convert("1"), dtype=np.uint8)  # 0/1
            if m.max() == 0:
                continue
            inst_map[m > 0] = next_id
            cls_id = int(c0) + 1               # 1..5
            types[next_id] = cls_id
            type_map[m > 0] = cls_id
            next_id += 1
        return inst_map, types, type_map

    def __getitem__(self, idx: int):
        rec = self.ds[int(idx)]
        img: Image.Image = rec["image"]
        img_np = np.array(img).astype(np.uint8)
        if img_np.ndim == 2:
            img_np = np.repeat(img_np[..., None], 3, axis=2)
        x = torch.from_numpy(img_np).float().permute(2,0,1)  # (3,H,W)
        if self.normalize and x.max() > 1.0:
            x = x / 255.0

        gt_inst, gt_types, type_map = self._build_gt_from_record(rec)
        name = f"{self.ds.split}_{idx:06d}" if hasattr(self.ds, "split") else f"pannuke_{idx:06d}"
        return {
            "image": x, "gt_inst_map": gt_inst, "gt_types": gt_types,
            "type_map": type_map, "name": name
        }

# (추가) PanNuke 배치용 collate
def _collate_pannuke(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),   # (B,3,H,W) 텐서
        "gt_inst_map": [b["gt_inst_map"] for b in batch],           # list[np.ndarray]
        "gt_types":    [b["gt_types"]    for b in batch],           # list[dict]  <-- 핵심: dict를 그대로 유지
        "type_map":    [b["type_map"]    for b in batch],           # list[np.ndarray]
        "name":        [b["name"]        for b in batch],           # list[str]
    }


# --- add: instance boundary & type-colored instance overlay helpers ---

def _instance_boundary_mask(inst_map: np.ndarray) -> np.ndarray:
    """인스턴스 경계(bool). 이웃 픽셀과 id가 다르면 경계."""
    h, w = inst_map.shape
    m = inst_map
    # pad로 경계 처리 후 4-연결 기준 경계 검출
    up    = np.pad(m[1:, :],   ((0,1),(0,0)), mode='edge')
    down  = np.pad(m[:-1, :],  ((1,0),(0,0)), mode='edge')
    left  = np.pad(m[:, 1:],   ((0,0),(0,1)), mode='edge')
    right = np.pad(m[:, :-1],  ((0,0),(1,0)), mode='edge')
    b = (m != up) | (m != down) | (m != left) | (m != right)
    # 배경 0은 경계에서 제외 (선택)
    b &= (m != 0)
    return b

def _colorize_instances_by_type(inst_map: np.ndarray,
                                types_dict: Dict[int,int],
                                palette: Dict[int, Tuple[int,int,int]],
                                draw_boundaries: bool = True,
                                boundary_color: Tuple[int,int,int] = (255,255,255)) -> np.ndarray:
    """
    각 인스턴스를 '그 인스턴스의 타입 색'으로 채운 RGB 맵 반환.
    types_dict: {inst_id -> class_id(1..C)}, palette: {class_id -> (R,G,B)}
    """
    h, w = inst_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = [i for i in np.unique(inst_map) if i != 0]
    for iid in ids:
        cls_id = int(types_dict.get(iid, 0))
        rgb = palette.get(cls_id, (0, 0, 0))
        out[inst_map == iid] = rgb

    if draw_boundaries:
        bmask = _instance_boundary_mask(inst_map)
        out[bmask] = np.array(boundary_color, dtype=np.uint8)
    return out


# -----------------------------
# 매칭 유틸 (IoU / Centroid)
# -----------------------------
def iou_matrix(gt_map: np.ndarray, pr_map: np.ndarray) -> np.ndarray:
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


def greedy_match_by_iou(iou: np.ndarray, thr: float) -> List[Tuple[int,int,float]]:
    if iou.size == 0:
        return []
    G, P = iou.shape
    flat = [(float(iou[gi,pj]), gi, pj) for gi in range(G) for pj in range(P) if iou[gi,pj] >= thr]
    flat.sort(reverse=True, key=lambda t: t[0])
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pj in flat:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi); used_p.add(pj)
        pairs.append((gi, pj, v))
    return pairs


def _centroids_from_inst_map(inst_map: np.ndarray) -> Dict[int, Tuple[float,float]]:
    ids = [i for i in np.unique(inst_map) if i != 0]
    out = {}
    for i in ids:
        ys, xs = np.nonzero(inst_map == i)
        if ys.size > 0:
            out[i] = (float(ys.mean()), float(xs.mean()))
    return out


def greedy_match_by_centroid(
    gt_map: np.ndarray, pr_map: np.ndarray, radius_px: float
) -> Tuple[List[Tuple[int,int,float]], Dict[int,Tuple[float,float]], Dict[int,Tuple[float,float]]]:
    gt_ids = [i for i in np.unique(gt_map) if i != 0]
    pr_ids = [j for j in np.unique(pr_map) if j != 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return [], _centroids_from_inst_map(gt_map), _centroids_from_inst_map(pr_map)

    gt_c = _centroids_from_inst_map(gt_map)
    pr_c = _centroids_from_inst_map(pr_map)
    G, P = len(gt_ids), len(pr_ids)
    Yg = [gt_c[i] for i in gt_ids]
    Yp = [pr_c[j] for j in pr_ids]
    A = np.asarray(Yg, dtype=np.float32)
    B = np.asarray(Yp, dtype=np.float32)
    D = np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))

    cand = [(float(D[gi,pj]), gi, pj) for gi in range(G) for pj in range(P) if D[gi,pj] <= radius_px]
    cand.sort(key=lambda t: t[0])  # 가까운 것부터
    used_g, used_p, pairs = set(), set(), []
    for dist, gi, pj in cand:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi); used_p.add(pj)
        pairs.append((gi, pj, dist))
    return pairs, gt_c, pr_c


# -----------------------------
# 시각화 도우미
# -----------------------------
def _overlay_instances(inst_map: np.ndarray, palette: Dict[int,Tuple[int,int,int]]) -> np.ndarray:
    # instance id는 무작위 색 → 여기선 type palette가 없으므로 파스텔 톤 생성
    rng = np.random.default_rng(12345)
    h, w = inst_map.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    ids = [i for i in np.unique(inst_map) if i != 0]
    colors = {}
    for i in ids:
        colors[i] = tuple(int(c) for c in rng.integers(low=64, high=224, size=3))
        out[inst_map == i] = colors[i]
    return out


def _draw_centroids_and_matches(ax, gt_c: Dict[int,Tuple[float,float]], pr_c: Dict[int,Tuple[float,float]],
                                matches: List[Tuple[int,int,float]],
                                gt_ids: List[int], pr_ids: List[int],
                                radius_px: float):
    # 매칭된 쌍 표시(초록 선 + 점), 미매칭 GT(빨강), 미매칭 Pred(주황)
    matched_g = set([gt_ids[gi] for gi,_,_ in matches])
    matched_p = set([pr_ids[pj] for _,pj,_ in matches])

    # 반경 원은 GT 중심에만 예시로 그립니다.
    for gid in gt_ids:
        yx = gt_c.get(gid, None)
        if yx is None: continue
        circ = Circle((yx[1], yx[0]), radius_px, fill=False, edgecolor='blue', linewidth=0.6, alpha=0.5)
        ax.add_patch(circ)

    # 매칭선
    for gi, pj, _ in matches:
        gy, gx = gt_c[gt_ids[gi]]
        py, px = pr_c[pr_ids[pj]]
        ax.plot([gx, px], [gy, py], '-', color='lime', linewidth=1.0, alpha=0.9)

    # 점들
    for gid, (gy,gx) in gt_c.items():
        color = 'lime' if gid in matched_g else 'red'
        ax.plot(gx, gy, 'o', markersize=3, color=color, alpha=0.9)
    for pid, (py,px) in pr_c.items():
        color = 'lime' if pid in matched_p else 'orange'
        ax.plot(px, py, 'x', markersize=3, color=color, alpha=0.9)

    ax.set_title("Centroids & Matches\n(green=TP, red=FN, orange=FP, blue circle=radius)")
    ax.axis('off')


def _visualize_panel(
    img: np.ndarray,
    gt_type_map: Optional[np.ndarray],
    pr_type_map: Optional[np.ndarray],
    gt_inst: Optional[np.ndarray],
    pr_inst: Optional[np.ndarray],
    id2name: Dict[int,str], palette: Dict[int,tuple],
    centroid_panel_fn=None,
    save_path: Path = None,
    alpha: float = 0.45,
    gt_types_dict: Optional[Dict[int,int]] = None,   
    pr_types_dict: Optional[Dict[int,int]] = None,   
):
    fig = plt.figure(figsize=(12, 13))

    # 1) Input
    ax1 = plt.subplot(4,2,1); ax1.imshow(img); ax1.set_title("Input"); ax1.axis('off')

    # 2) GT Type Overlay
    if gt_type_map is not None:
        gt_rgb = colorize_label(gt_type_map, palette)
        gt_ov  = blend_overlay(img, gt_rgb, alpha)
        ax2 = plt.subplot(4,2,2); ax2.imshow(gt_ov); ax2.set_title("GT Type Overlay"); ax2.axis('off')
    else:
        ax2 = plt.subplot(4,2,2); ax2.axis('off')

    # 3) Pred Type Overlay
    if pr_type_map is not None:
        pr_rgb = colorize_label(pr_type_map, palette)
        pr_ov  = blend_overlay(img, pr_rgb, alpha)
        ax3 = plt.subplot(4,2,3); ax3.imshow(pr_ov); ax3.set_title("Pred Type Overlay"); ax3.axis('off')
    else:
        ax3 = plt.subplot(4,2,3); ax3.axis('off')

    # 4) Centroids & Matches
    ax4 = plt.subplot(4,2,4)
    if centroid_panel_fn is not None:
        ax4.imshow(img); centroid_panel_fn(ax4)
    ax4.axis('off')

    # 5) GT Instance Overlay (랜덤색 인스턴스)
    if gt_inst is not None:
        gi_rgb = _overlay_instances(gt_inst, palette)
        gi_ov  = blend_overlay(img, gi_rgb, alpha=0.35)
        ax5 = plt.subplot(4,2,5); ax5.imshow(gi_ov); ax5.set_title("GT Instance Overlay"); ax5.axis('off')
    else:
        ax5 = plt.subplot(4,2,5); ax5.axis('off')

    # 6) Pred Instance Overlay (랜덤색 인스턴스)
    if pr_inst is not None:
        pi_rgb = _overlay_instances(pr_inst, palette)
        pi_ov  = blend_overlay(img, pi_rgb, alpha=0.35)
        ax6 = plt.subplot(4,2,6); ax6.imshow(pi_ov); ax6.set_title("Pred Instance Overlay"); ax6.axis('off')
    else:
        ax6 = plt.subplot(4,2,6); ax6.axis('off')

    # 7) GT Type + Instance Overlay (인스턴스를 '타입 색'으로 채움 + 경계)
    ax7 = plt.subplot(4,2,7)
    if gt_inst is not None and gt_types_dict is not None:
        gt_ti_rgb = _colorize_instances_by_type(gt_inst, gt_types_dict, palette, draw_boundaries=True)
        gt_ti_ov  = blend_overlay(img, gt_ti_rgb, alpha=0.40)
        ax7.imshow(gt_ti_ov); ax7.set_title("GT Type+Instance Overlay"); ax7.axis('off')
    else:
        ax7.axis('off')

    # 8) Pred Type+Instance Overlay
    ax8 = plt.subplot(4,2,8)
    if pr_inst is not None and pr_types_dict is not None:
        pr_ti_rgb = _colorize_instances_by_type(pr_inst, pr_types_dict, palette, draw_boundaries=True)
        pr_ti_ov  = blend_overlay(img, pr_ti_rgb, alpha=0.40)
        ax8.imshow(pr_ti_ov); ax8.set_title("Pred Type+Instance Overlay"); ax8.axis('off')
    else:
        ax8.axis('off')

    # 범례
    handles = build_legend_handles(id2name, palette)
    fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)), frameon=True)

    plt.tight_layout(rect=[0,0.05,1,1])
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)



# -----------------------------
# PanNuke CV3: fold별 전체 테스트 세트 시각화
# -----------------------------
@torch.no_grad()
def run_pannuke_cv3_visualize(
    repo_id: str,
    ckpt_root: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    match: str = "paper",
    iou_thr: float = 0.5,
    magnification: int = 40,
    radius_px: Optional[float] = None,
    max_samples_per_fold: int = -1,
    use_postproc_types: bool = True,
    model_cfg: Dict = None,
):
    id2name, palette, _ = label_spec('pannuke_hf', merge_consep=False)

    folds = ["fold1","fold2","fold3"]
    for f in folds:
        # 모델 로드
        n = int(f.replace("fold",""))
        ckpt_path = ckpt_root / f"cv3_fold{n}" / "best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[{f}] checkpoint not found: {ckpt_path}")

        model = CellViTCustom(
            num_nuclei_classes=model_cfg["num_classes"],
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
        print(f"[{f}] loaded ckpt: {ckpt_path}")

        ds = PanNukeHFFoldDataset(repo_id=repo_id, fold=f, normalize=True)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=_collate_pannuke,   # <-- 요거 추가
        )

        saved = 0
        for batch in tqdm(dl, desc=f"[{f}] infer+viz"):
            x = batch["image"].to(device, non_blocking=True)
            gt_inst_maps = batch["gt_inst_map"]
            gt_types_list = batch["gt_types"]
            gt_type_maps  = batch["type_map"]
            names         = batch["name"] if isinstance(batch["name"], list) else [batch["name"]]

            # forward
            y = model(x)

            # postproc → 인스턴스 맵 & 타입 딕셔너리
            pred_inst_maps_t, pred_type_dicts = model.calculate_instance_map(
                y, magnification=magnification
            )
            pred_inst_maps = pred_inst_maps_t.cpu().numpy().astype(np.int32)

            # type argmax (fallback)
            type_argmax = torch.argmax(y["nuclei_type_map"], dim=1).cpu().numpy()

            for b in range(x.size(0)):
                img = (x[b].cpu().numpy().transpose(1,2,0) * 255.0).round().astype(np.uint8)
                gt_inst = gt_inst_maps[b]
                gt_types: Dict[int,int] = gt_types_list[b]
                gt_type = gt_type_maps[b]

                pr_inst = pred_inst_maps[b]
                # 예측 타입 dict 정규화
                if use_postproc_types:
                    pr_types = {}
                    entry = pred_type_dicts[b]
                    # entry는 {inst_id: {cls/prob/...}} 형태일 수 있음
                    for k, v in (entry.items() if isinstance(entry, dict) else []):
                        try:
                            inst_id = int(k)
                        except Exception:
                            continue
                        cls_id = None
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            cls_id = int(round(float(v)))
                        elif isinstance(v, dict):
                            if "type" in v: cls_id = int(v["type"])
                            elif "class" in v: cls_id = int(v["class"])
                            elif "cls" in v: cls_id = int(v["cls"])
                            elif "label" in v: cls_id = int(v["label"])
                            else:
                                for key in ("prob","probs","logits","type_prob","type_probs"):
                                    if key in v:
                                        arr = np.asarray(v[key])
                                        if arr.size > 0:
                                            cls_id = int(arr.argmax()); break
                        if cls_id is None:
                            # fall back: majority from pixel-wise type
                            mask = (pr_inst == inst_id)
                            vals = type_argmax[b][mask]
                            vals = vals[vals > 0]
                            cls_id = int(np.bincount(vals).argmax()) if vals.size>0 else 0
                        pr_types[inst_id] = int(cls_id)
                else:
                    pr_types = {}
                    ids = [i for i in np.unique(pr_inst) if i != 0]
                    for pid in ids:
                        mask = (pr_inst == pid)
                        vals = type_argmax[b][mask]
                        vals = vals[vals > 0]
                        pr_types[pid] = int(np.bincount(vals).argmax()) if vals.size>0 else 0

                # 매칭 (paper / iou)
                gt_ids = [i for i in np.unique(gt_inst) if i != 0]
                pr_ids = [j for j in np.unique(pr_inst) if j != 0]
                if match == "iou":
                    M = iou_matrix(gt_inst, pr_inst)
                    pairs = greedy_match_by_iou(M, thr=iou_thr)
                    # centroid 좌표는 시각화 편의상 계산
                    gt_c = _centroids_from_inst_map(gt_inst)
                    pr_c = _centroids_from_inst_map(pr_inst)
                    radius_draw = 0.0
                else:
                    r = radius_px if radius_px is not None else (12.0 if magnification >= 30 else 6.0)
                    pairs, gt_c, pr_c = greedy_match_by_centroid(gt_inst, pr_inst, radius_px=r)
                    radius_draw = r

                # 시각화 콜백(centroid + matches)
                def centroid_panel(ax):
                    _draw_centroids_and_matches(ax, gt_c, pr_c, pairs, gt_ids, pr_ids, radius_px=radius_draw)

                # 타입 예측 맵(픽셀) — 인스턴스 타입 dict과 별개로 보조용
                pr_type_pix = type_argmax[b]

                save_dir = out_dir / "pannuke_cv3" / f"{f}"
                save_path = save_dir / f"{names[b]}_vis.png"
                _visualize_panel(
                    img=img,
                    gt_type_map=gt_type,
                    pr_type_map=pr_type_pix,
                    gt_inst=gt_inst,
                    pr_inst=pr_inst,
                    id2name=id2name, palette=palette,
                    centroid_panel_fn=centroid_panel,
                    save_path=save_path,
                    alpha=0.45,
                    gt_types_dict=gt_types,    # <-- 추가
                    pr_types_dict=pr_types,    # <-- 추가
                )
                saved += 1
                if (max_samples_per_fold > 0) and (saved >= max_samples_per_fold):
                    break

        print(f"[{f}] saved visualizations to: {out_dir/'pannuke_cv3'/f}")


# -----------------------------
# CoNSeP: 타입 분류 시각화 (픽셀 단위)
# -----------------------------
@torch.no_grad()
def run_consep_visualize(
    root: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    merge_consep: bool,
    model_cfg: Dict,
    ckpt_path: Optional[Path],
    max_samples: int = 40,
    type_alpha: float = 0.45,
):
    if ConsepPatchDataset is None:
        raise SystemExit("ConsepPatchDataset modules not found in your environment.")
    ds = ConsepPatchDatasetMerged(root, label_scheme='consep_merged') if merge_consep \
         else ConsepPatchDataset(root, num_nuclei_classes=model_cfg["num_classes"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    id2name, palette, remap = label_spec('consep', merge_consep)

    model = CellViTCustom(
        num_nuclei_classes=model_cfg["num_classes"],
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
    ).to(device).eval()

    if ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location='cpu')
        sd = ck.get('model', ck)
        model.load_state_dict(sd, strict=False)
        print(f"[consep] loaded weights: {ckpt_path}")
    else:
        print("[consep] WARNING: no checkpoint provided — random weights")

    saved = 0
    for batch in tqdm(dl, desc="[consep] infer+viz"):
        x = batch['image'].to(device, non_blocking=True)
        t = batch['type_map'].cpu().numpy()
        paths = batch['path_image']

        y = model(x)
        nt_pred = torch.argmax(y['nuclei_type_map'], dim=1).cpu().numpy()

        # (선택) merged remap
        if merge_consep:
            vt = np.vectorize(lambda k: remap.get(int(k), int(k)))
            t = np.stack([vt(t[i]) for i in range(t.shape[0])], axis=0)
            nt_pred = np.stack([vt(nt_pred[i]) for i in range(nt_pred.shape[0])], axis=0)

        for b in range(x.size(0)):
            img = (x[b].cpu().numpy().transpose(1,2,0) * 255.0).round().astype(np.uint8)
            gt_rgb = colorize_label(t[b], palette)
            pr_rgb = colorize_label(nt_pred[b], palette)
            gt_ov  = blend_overlay(img, gt_rgb, alpha=type_alpha)
            pr_ov  = blend_overlay(img, pr_rgb, alpha=type_alpha)

            fig = plt.figure(figsize=(12,8))
            ax1 = plt.subplot(2,2,1); ax1.imshow(img);   ax1.set_title("Input");   ax1.axis('off')
            ax2 = plt.subplot(2,2,2); ax2.imshow(gt_ov); ax2.set_title("GT Type"); ax2.axis('off')
            ax3 = plt.subplot(2,2,3); ax3.imshow(pr_ov); ax3.set_title("Pred Type"); ax3.axis('off')
            handles = build_legend_handles(id2name, palette)
            ax4 = plt.subplot(2,2,4); ax4.axis('off'); ax4.legend(handles=handles, title="Cell types", loc='center')

            stem = Path(paths[b]).stem if isinstance(paths[b], str) else f"consep_{saved:06d}"
            save_path = out_dir / "consep" / f"{stem}_typeviz.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)

            saved += 1
            if saved >= max_samples:
                break

    print(f"[consep] saved visualizations to: {out_dir/'consep'}")


# -----------------------------
# Args & Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CoNSeP / PanNuke visualization (inference + matplotlib saving)")
    p.add_argument('--dataset', type=str, required=True, choices=['consep', 'pannuke_hf'])

    # PanNuke(HF)
    p.add_argument('--hf_repo', type=str, default='RationAI/PanNuke')
    p.add_argument('--ckpt_root', type=Path, default=Path('/workspace/CellViT_Custom/Checkpoints/CellViT/cv3'))
    p.add_argument('--match', type=str, default='paper', choices=['paper','iou'])
    p.add_argument('--iou_thr', type=float, default=0.5)
    p.add_argument('--magnification', type=int, default=40, help='x40: 12px, x20: 6px')
    p.add_argument('--radius_px', type=float, default=None, help='centroid match radius override')
    p.add_argument('--max_samples_per_fold', type=int, default=-1, help='-1이면 전체 저장')

    # CoNSeP
    p.add_argument('--root', type=Path, default=None, help='CoNSeP root (consep only)')
    p.add_argument('--merge_consep', action='store_true')
    p.add_argument('--ckpt', type=Path, default=None, help='CoNSeP/단일 모델 가중치')

    # 공통
    p.add_argument('--out', type=Path, default=Path('vis_out'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)

    # CellViT hyperparams
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--embed_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--num_classes', type=int, default=6, help='incl. bg; PanNuke=6')

    # 기타
    p.add_argument('--no_postproc_types', action='store_true',
                   help="postproc 타입 dict 사용 끄고 pixel argmax majority로만 타입 결정")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )

    if args.dataset == 'pannuke_hf':
        run_pannuke_cv3_visualize(
            repo_id=args.hf_repo,
            ckpt_root=args.ckpt_root,
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            match=args.match,
            iou_thr=args.iou_thr,
            magnification=args.magnification,
            radius_px=args.radius_px,
            max_samples_per_fold=args.max_samples_per_fold,
            use_postproc_types=not args.no_postproc_types,
            model_cfg=model_cfg,
        )
    else:
        if args.root is None:
            raise SystemExit("--root is required for CoNSeP dataset.")
        run_consep_visualize(
            root=args.root,
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            merge_consep=args.merge_consep,
            model_cfg=model_cfg,
            ckpt_path=args.ckpt,
            max_samples=40,
            type_alpha=0.45,
        )


if __name__ == "__main__":
    main()
