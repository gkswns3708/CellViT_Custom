# -*- coding: utf-8 -*-
"""
PanNuke 3-fold CV 인스턴스 수준 평가 (HoverNet/StarDist/CPP-Net 디코더 선택 가능)

- HF Hub의 RationAI/PanNuke 로부터 fold1/2/3를 읽습니다.
- 각 fold에 대해:
    * 해당 fold의 체크포인트로 추론
    * 디코더 유형에 맞는 post processor를 통해 인스턴스 분할
    * GT 인스턴스와 IoU≥0.5로 1:1 매칭 (그리디 매칭)
    * Detection Pd/Rd/F1d와, 5개 클래스(Neoplastic/Inflammatory/Connective/Dead/Epithelial)의 P/R/F1을
      "매칭된 인스턴스" 기준으로 집계
- 3개 fold의 결과를 평균 내어 출력 + CSV로 저장

체크포인트 경로 형태(기본):
  /workspace/CellViT_Custom/Checkpoints/CellViT/cv3/cv3_fold{1,2,3}/best.pth
혹은 --ckpt_list 로 각 fold별 경로를 직접 지정할 수 있습니다.

추가: --decoder 로 디코더 선택 (hovernet, stardist, cpp)
      --nrays   로 StarDist/CPP-Net의 ray 개수 조정 가능
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# --- 모델들 ---
from Model.CellViT_ViT256_Custom import CellViTCustom  # Hover-Net style
from Model.cellvit_stardist import CellViTStarDist     # StarDist decoder
from Model.cellvit_cpp_net import CellViTCPP           # CPP-Net decoder


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

    HF 레코드 구조 (fold1/2/3):
      - image: PIL.Image (256x256 RGB)
      - instances: 길이 N 리스트 (각각이 PIL 1-bit 바이너리 이미지)
      - categories: 길이 N 리스트 (각 인스턴스의 클래스 0..4)
      - tissue: int (unused here)
    """
    def __init__(self, repo_id: str, fold: str, normalize: bool = True) -> None:
        super().__init__()
        assert fold in ("fold1", "fold2", "fold3"), "fold must be one of fold1/fold2/fold3"
        self.ds = load_dataset(repo_id, split=fold)
        self.normalize = normalize

    def __len__(self) -> int:
        return self.ds.num_rows

    @staticmethod
    def _build_gt_from_record(rec) -> Tuple[np.ndarray, Dict[int,int]]:
        """
        rec['instances']: list of PIL 1-bit masks (256x256), each exactly one nucleus.
        rec['categories']: list of class ids in [0..4], mapping to [1..5].
        returns:
          gt_inst_map: (H,W) ints, 0=bg, 1..N instances
          gt_types: {inst_id -> class_id(1..5)}
        """
        H = W = 256  # PanNuke fixed size
        inst_map = np.zeros((H, W), dtype=np.int32)
        types: Dict[int, int] = {}
        inst_imgs = rec["instances"]
        cats = rec["categories"]
        assert len(inst_imgs) == len(cats), "instances and categories length mismatch"

        for i, (pil_bin, c0) in enumerate(zip(inst_imgs, cats), start=1):
            # to binary mask
            m = np.array(pil_bin.convert("1"), dtype=np.uint8)  # 0/1
            if m.max() == 0:
                continue
            # assign unique id i where mask==1 (if overlap occurs, later instance overrides)
            inst_map[m > 0] = i
            # class: 0..4 -> 1..5
            types[i] = int(c0) + 1
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
        name = f"{self.ds.split}_{idx:06d}"
        return {
            "image": x,
            "gt_inst_map": gt_inst,    # numpy
            "gt_types": gt_types,      # dict
            "name": name,
        }


# ----------------------------
# IoU & matching (greedy)
# ----------------------------
def iou_matrix(gt_map: np.ndarray, pr_map: np.ndarray) -> np.ndarray:
    """Compute IoU for all GT x Pred instance pairs."""
    gt_ids = [i for i in np.unique(gt_map) if i != 0]
    pr_ids = [j for j in np.unique(pr_map) if j != 0]
    G, P = len(gt_ids), len(pr_ids)
    if G == 0 or P == 0:
        return np.zeros((G, P), dtype=np.float32)

    iou = np.zeros((G, P), dtype=np.float32)
    for gi, gid in enumerate(gt_ids):
        gmask = (gt_map == gid)
        g_area = gmask.sum()
        if g_area == 0:
            continue
        for pj, pid in enumerate(pr_ids):
            pmask = (pr_map == pid)
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
    pairs = []
    G, P = iou.shape
    flat = []
    for gi in range(G):
        for pj in range(P):
            v = float(iou[gi, pj])
            if v >= thr:
                flat.append((v, gi, pj))
    flat.sort(reverse=True, key=lambda t: t[0])

    used_g = set()
    used_p = set()
    for v, gi, pj in flat:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        pairs.append((gi, pj, v))
    return pairs


# ----------------------------
# Per-image → accumulators
# ----------------------------
class InstanceMetricsAccumulator:
    """Accumulate counts across images to compute Detection P/R/F1 and per-class P/R/F1."""
    def __init__(self, num_classes_including_bg: int = 6) -> None:
        # Detection
        self.tp_d = 0
        self.fp_d = 0
        self.fn_d = 0
        # Classification (classes 1..5 for PanNuke)
        C = num_classes_including_bg - 1  # 5
        self.C = C
        self.tp_c = np.zeros(C + 1, dtype=np.int64)    # 0 unused
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
        self.tp_d += tp
        self.fp_d += fp
        self.fn_d += fn

        # classification: only on matched pairs
        for gi, pj, _ in matches:
            gid = gt_ids[gi]
            pid = pr_ids[pj]
            gcls = int(gt_types.get(gid, 0))  # 1..5
            pcls = int(pr_types.get(pid, 0))
            if gcls < 1 or gcls > self.C:   # ignore background or out-of-range
                continue
            if pcls < 1 or pcls > self.C:
                continue
            # accumulate
            self.gt_c[gcls] += 1
            self.pred_c[pcls] += 1
            if gcls == pcls:
                self.tp_c[gcls] += 1

    def summarize(self) -> Dict[str, float | Dict[int, Dict[str, float]]]:
        # detection
        Pd = self.tp_d / (self.tp_d + self.fp_d + 1e-9)
        Rd = self.tp_d / (self.tp_d + self.fn_d + 1e-9)
        F1d = 2 * Pd * Rd / (Pd + Rd + 1e-9)

        # classification per-class
        per_class = {}
        for k in range(1, self.C + 1):
            tp = self.tp_c[k]
            pred = self.pred_c[k]
            gt = self.gt_c[k]
            P = tp / (pred + 1e-9)
            R = tp / (gt + 1e-9)
            F1 = 2 * P * R / (P + R + 1e-9) if (P + R) > 0 else 0.0
            per_class[k] = {"P": float(P), "R": float(R), "F1": float(F1), "support": int(gt)}

        return {
            "Pd": float(Pd),
            "Rd": float(Rd),
            "F1d": float(F1d),
            "per_class": per_class,
            "det_counts": {"TP": self.tp_d, "FP": self.fp_d, "FN": self.fn_d},
        }


# ----------------------------
# Pred instance types (majority)
# ----------------------------
def types_from_map_majority(type_argmax: np.ndarray, inst_map: np.ndarray) -> Dict[int, int]:
    """
    type_argmax: (H,W)  predicted type id [0..5], where 0=bg, 1..5 classes
    inst_map:    (H,W)  predicted instance ids (0 bg, 1..N)
    return: {inst_id -> class_id(1..5)}
    """
    types: Dict[int, int] = {}
    ids = [i for i in np.unique(inst_map) if i != 0]
    for pid in ids:
        mask = (inst_map == pid)
        if mask.sum() == 0:
            continue
        vals = type_argmax[mask]
        vals = vals[vals > 0]
        cls = int(np.bincount(vals).argmax()) if vals.size > 0 else 0
        types[pid] = cls
    return types


# ----------------------------
# 디코더별 instance 생성 래퍼
# ----------------------------
def build_model(decoder: str, model_cfg: dict) -> torch.nn.Module:
    dec = decoder.lower()
    if dec == "hovernet":
        return CellViTCustom(
            num_nuclei_classes=model_cfg["num_classes"],
            num_tissue_classes=model_cfg.get("num_tissue_classes", 0),
            img_size=model_cfg["img_size"],
            patch_size=model_cfg["patch_size"],
            embed_dim=model_cfg["embed_dim"],
            input_channels=3,
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_ratio=model_cfg["mlp_ratio"],
        )
    elif dec == "stardist":
        return CellViTStarDist(
            num_nuclei_classes=model_cfg["num_classes"],
            num_tissue_classes=model_cfg.get("num_tissue_classes", 0),
            embed_dim=model_cfg["embed_dim"],
            input_channels=3,
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            extract_layers=model_cfg.get("extract_layers", [model_cfg["depth"] // 4, model_cfg["depth"] // 2, 3 * model_cfg["depth"] // 4, model_cfg["depth"]]),
            nrays=model_cfg.get("nrays", 32),
            mlp_ratio=model_cfg["mlp_ratio"],
        )
    elif dec == "cpp":
        return CellViTCPP(
            num_nuclei_classes=model_cfg["num_classes"],
            num_tissue_classes=model_cfg.get("num_tissue_classes", 0),
            embed_dim=model_cfg["embed_dim"],
            input_channels=3,
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            extract_layers=model_cfg.get("extract_layers", [model_cfg["depth"] // 4, model_cfg["depth"] // 2, 3 * model_cfg["depth"] // 4, model_cfg["depth"]]),
            nrays=model_cfg.get("nrays", 32),
            mlp_ratio=model_cfg["mlp_ratio"],
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder}")


def predictions_to_instances(
    decoder: str,
    model: torch.nn.Module,
    y: Dict[str, torch.Tensor],
    magnification: int,
) -> Tuple[np.ndarray, List[Dict[int, int]], np.ndarray]:
    """디코더 유형에 맞게 인스턴스 맵과 타입 딕트를 생성한다.

    Returns:
        pred_inst_maps: (B,H,W) int32
        pred_type_dicts: list of dict(instance_id -> class_id)
        type_argmax: (B,H,W) int32 (for fallback majority)
    """
    dec = decoder.lower()

    if "nuclei_type_map" not in y:
        raise KeyError("Model output must contain 'nuclei_type_map'.")

    # 공통: 픽셀 단위 타입 argmax (bg=0, classes=1..C-1)
    type_argmax = torch.argmax(y["nuclei_type_map"], dim=1).cpu().numpy()

    if dec == "hovernet":
        # HoverNet-style: 모델 내부 postproc 사용 (HV 기반)
        pred_inst_maps_t, pred_type_dicts = model.calculate_instance_map(y, magnification=magnification)
        pred_inst_maps = pred_inst_maps_t.cpu().numpy().astype(np.int32)
        # pred_type_dicts 는 보통 {inst_id: cls} 형태. 안전하게 후처리
        out_dicts: List[Dict[int, int]] = []
        for d in pred_type_dicts:
            if isinstance(d, dict) and all(isinstance(k, (int, np.integer)) for k in d.keys()):
                out_dicts.append({int(k): int(v) for k, v in d.items()})
            else:
                out_dicts.append({})
        return pred_inst_maps, out_dicts, type_argmax

    # StarDist/CPP: 별도 시그니처
    stardist_key = "stardist_map_refined" if "stardist_map_refined" in y else "stardist_map"
    if stardist_key not in y or "dist_map" not in y:
        raise KeyError("StarDist/CPP decoders require 'dist_map' and 'stardist_map' (or 'stardist_map_refined') in outputs.")

    inst_t, type_list, _ = model.calculate_instance_map(
        dist_map=y["dist_map"],
        stardist_map=y[stardist_key],
        nuclei_type_map=y["nuclei_type_map"],
    )
    pred_inst_maps = inst_t.cpu().numpy().astype(np.int32)

    # type_list 가 {inst_id: {"type": cls, ...}} 혹은 {inst_id: cls} 일 수 있음 → 안전 변환
    out_dicts: List[Dict[int, int]] = []
    B = pred_inst_maps.shape[0]
    for b in range(B):
        d = type_list[b] if isinstance(type_list, list) else {}
        mapping: Dict[int, int] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and ("type" in v):
                    mapping[int(k)] = int(v["type"])  # 1..C-1
                elif isinstance(v, (int, np.integer)):
                    mapping[int(k)] = int(v)
        out_dicts.append(mapping)

    return pred_inst_maps, out_dicts, type_argmax


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
    decoder: str,
    magnification: int = 40,
) -> Dict[str, float | Dict]:
    ds = PanNukeHFFoldDataset(repo_id=repo_id, fold=fold, normalize=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)

    # Model
    model = build_model(decoder, model_cfg).to(device).eval()

    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck.get("model", ck)
    missing_unexp = model.load_state_dict(sd, strict=False)
    try:
        mk, uk = missing_unexp.missing_keys, missing_unexp.unexpected_keys
        print(f"[fold {fold}] load ckpt: {ckpt_path}  missing={len(mk)} unexpected={len(uk)}")
    except Exception:
        print(f"[fold {fold}] loaded ckpt: {ckpt_path}")

    acc = InstanceMetricsAccumulator(num_classes_including_bg=model_cfg["num_classes"])  # PanNuke: 6

    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
        gt_inst_maps = batch["gt_inst_map"]               # list/np arrays
        gt_types_list = batch["gt_types"]

        # forward
        y = model(x)
        # 인스턴스 후처리 (디코더별)
        pred_inst_maps, pred_type_dicts, type_argmax = predictions_to_instances(
            decoder=decoder, model=model, y=y, magnification=magnification
        )

        # 배치 단위 평가 누적
        for b in range(x.size(0)):
            gt_map = gt_inst_maps[b] if isinstance(gt_inst_maps, list) else gt_inst_maps[b].numpy()
            pr_map = pred_inst_maps[b]
            # ensure shape alignment
            if gt_map.shape != pr_map.shape:
                H = min(gt_map.shape[0], pr_map.shape[0])
                W = min(gt_map.shape[1], pr_map.shape[1])
                gt_map = gt_map[:H, :W]
                pr_map = pr_map[:H, :W]
                ta = type_argmax[b][:H, :W]
            else:
                ta = type_argmax[b]

            # GT dict
            gt_types: Dict[int, int] = gt_types_list[b]

            # Pred dict: 가능하면 postproc 타입, 아니면 majority
            pr_types = pred_type_dicts[b] if (b < len(pred_type_dicts)) else {}
            if not pr_types:
                pr_types = types_from_map_majority(ta, pr_map)

            acc.add_image(gt_map, gt_types, pr_map, pr_types, iou_thr=0.5)

    return acc.summarize()


# ----------------------------
# 3-fold wrapper
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PanNuke 3-fold instance-level evaluation (CellViT Multi-Decoder)")
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
    p.add_argument("--decoder", type=str, choices=["hovernet", "stardist", "cpp"], default="hovernet",
                   help="디코더 선택: hovernet(=Custom), stardist, cpp")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--num_classes", type=int, default=6, help="PanNuke: bg+5 classes = 6")
    p.add_argument("--num_tissue_classes", type=int, default=0, help="전역 tissue 분류 사용 시 >0")
    p.add_argument("--nrays", type=int, default=32, help="StarDist/CPP-Net에서 사용하는 ray 수")

    p.add_argument("--magnification", type=int, default=40, help="Detection postproc magnification (HoverNet=x40)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    folds = ["fold1", "fold2", "fold3"]

    # resolve ckpts
    ckpts: Dict[str, Path] = {}
    if args.ckpt_list:
        parts = [Path(s.strip()) for s in args.ckpt_list.split(",")]
        assert len(parts) == 3, "--ckpt_list must contain 3 paths for fold1,fold2,fold3"
        ckpts = {f: parts[i] for i, f in enumerate(folds)}
    else:
        # default pattern: {ckpt_root}/cv3_fold{n}/best.pth
        for f in folds:
            n = int(f.replace("fold", ""))
            ckpts[f] = args.ckpt_root / f"cv3_fold{n}" / "best.pth"

    # model cfg
    model_cfg = dict(
        num_classes=args.num_classes,
        num_tissue_classes=args.num_tissue_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        extract_layers=[args.depth // 4, args.depth // 2, 3 * args.depth // 4, args.depth],
        nrays=args.nrays,
    )

    print(f"Decoder: {args.decoder}  |  ModelCfg: {model_cfg}")

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
            decoder=args.decoder,
            magnification=args.magnification,
        )
        fold_results[f] = res
        print(f"\n[{f}] Detection: Pd={res['Pd']:.3f}  Rd={res['Rd']:.3f}  F1d={res['F1d']:.3f}")
        for k, name in enumerate(["Neoplastic","Inflammatory","Connective","Dead","Epithelial"], start=1):
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
