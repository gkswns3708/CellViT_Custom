# -*- coding: utf-8 -*-
"""
PanNuke 3-fold cross-validation driver (HF: RationAI/PanNuke)
- 각 라운드에서 1개 fold를 검증/테스트로 두고(예: fold1),
  나머지 두 fold(예: fold2, fold3)로 학습
- fold1/2/3을 모두 검증으로 번갈아 수행 → 3회 평균 리포트

출력:
  Checkpoints/CellViT/
    ├─ cv3_fold1/
    │   ├─ best.pth
    │   ├─ last.pth
    │   ├─ confusion_fold1.npy
    │   └─ metrics_fold1.csv
    ├─ cv3_fold2/ ...
    ├─ cv3_fold3/ ...
    └─ metrics_cv3_summary.csv   # 3fold 평균 요약
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import csv
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# 모델/로스
from Model.CellViT_ViT256_Custom import CellViTCustom, cellvit_losses
# PanNuke(HF, instances+categories 기반) 데이터셋
from Data.PanNuke_hf_instances import PanNukeHFInstancesDataset


# -----------------------------
# Metrics: confusion → per-class F1/ACC
# -----------------------------
def update_confusion(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, C: int):
    """0..C(포함) 레이블에 대한 빠른 혼동행렬 업데이트"""
    y_true = np.clip(y_true, 0, C)
    y_pred = np.clip(y_pred, 0, C)
    idx = C + 1
    cm = np.bincount((y_true * idx + y_pred).ravel(), minlength=idx * idx).reshape(idx, idx)
    conf += cm


def metrics_from_confusion(conf: np.ndarray, classes_to_eval: List[int]):
    """per-class precision/recall/F1, per-class acc(=recall), overall acc"""
    per_class: Dict[int, Dict[str, float]] = {}
    total = conf.sum()
    overall_acc = float(np.trace(conf) / total) if total > 0 else 0.0

    col_sum = conf.sum(axis=0)  # predicted count per class
    row_sum = conf.sum(axis=1)  # true count per class

    for c in classes_to_eval:
        tp = conf[c, c]
        fp = col_sum[c] - tp
        fn = row_sum[c] - tp
        support = row_sum[c]
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc_c = rec  # per-class accuracy = recall
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c, support=int(support))
    return overall_acc, per_class


# -----------------------------
# Data loaders
# -----------------------------
def build_dl_for_folds(repo: str, train_folds: List[str], val_fold: str, batch_size: int, workers: int):
    ds_tr = PanNukeHFInstancesDataset(repo_id=repo, folds=train_folds, normalize=True)
    ds_va = PanNukeHFInstancesDataset(repo_id=repo, folds=[val_fold],   normalize=True)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True,
                       persistent_workers=(workers > 0))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True,
                       persistent_workers=(workers > 0))
    return dl_tr, dl_va


# -----------------------------
# Train / Validate loop
# -----------------------------
@torch.no_grad()
def validate_loss(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {}
    total = 0
    for batch in dl:
        x = batch['image'].to(device, non_blocking=True)
        type_map = batch['type_map'].to(device, non_blocking=True)
        hv_map   = batch['hv_map'].to(device, non_blocking=True)
        bin_map  = batch['bin_map'].to(device, non_blocking=True)
        y = model(x)
        losses = cellvit_losses(y, type_map=type_map, hv_map=hv_map, bin_map=bin_map)
        bs = x.size(0)
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        total += bs
    return {k: v / max(total, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_types(model: torch.nn.Module, dl: DataLoader, device: torch.device, num_classes: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """검증 fold에서 type 분류 혼동행렬/지표 계산 (배경 제외 1..5)"""
    model.eval()
    C = num_classes - 1  # 0..C
    conf = np.zeros((C + 1, C + 1), dtype=np.int64)
    for batch in dl:
        x = batch['image'].to(device, non_blocking=True)
        t = batch['type_map'].cpu().numpy()     # (B,H,W)
        logits = model(x)['nuclei_type_map']    # (B,C+1,H,W)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        # clamp
        t = np.clip(t, 0, C)
        pred = np.clip(pred, 0, C)
        update_confusion(conf, t.reshape(-1), pred.reshape(-1), C)

    classes_to_eval = list(range(1, C + 1))  # 배경 제외
    overall_acc, per_class = metrics_from_confusion(conf, classes_to_eval)

    # fold 요약(매크로 평균)
    macro_f1  = float(np.mean([per_class[c]['f1']  for c in classes_to_eval])) if classes_to_eval else 0.0
    macro_acc = float(np.mean([per_class[c]['acc'] for c in classes_to_eval])) if classes_to_eval else 0.0
    summary = {"overall_acc": overall_acc, "macro_f1": macro_f1, "macro_acc": macro_acc}
    return conf, summary, per_class


def train_one_fold(
    repo: str, train_folds: List[str], val_fold: str,
    out_dir: Path, device: torch.device,
    epochs: int, freeze_epochs: int,
    batch_size: int, lr: float, workers: int,
    vit_embed_dim: int, vit_depth: int, vit_heads: int, vit_mlp_ratio: float,
    init_full_ckpt: Path | None, vit_ckpt: Path | None,
    num_classes: int = 6
) -> Tuple[Path, Dict[str, float]]:
    """한 fold를 학습 후 best.pth를 저장하고, 최종 val loss 리턴"""

    out_dir.mkdir(parents=True, exist_ok=True)
    dl_tr, dl_va = build_dl_for_folds(repo, train_folds, val_fold, batch_size, workers)

    # model
    model = CellViTCustom(
        num_nuclei_classes=num_classes,  # 0..5 (bg + 5 classes)
        num_tissue_classes=0,
        img_size=256, patch_size=16,
        embed_dim=vit_embed_dim, depth=vit_depth, num_heads=vit_heads, mlp_ratio=vit_mlp_ratio,
    ).to(device)

    # init
    if init_full_ckpt is not None and Path(init_full_ckpt).exists():
        sd = torch.load(str(init_full_ckpt), map_location='cpu')
        sd = sd.get('model', sd)
        msg = model.load_state_dict(sd, strict=False)
        print(f"[{val_fold}] init: full ckpt loaded (strict=False).")
        try:
            print(f"  missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        except Exception:
            pass
    elif vit_ckpt is not None and Path(vit_ckpt).exists():
        model.load_vit_checkpoint(str(vit_ckpt), strict=False)
        print(f"[{val_fold}] init: encoder ckpt loaded.")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float('inf')
    best_path = out_dir / "best.pth"

    # optionally freeze encoder for first N epochs
    if freeze_epochs > 0:
        print(f"[{val_fold}] freeze encoder for first {freeze_epochs} epochs")
        model.freeze_encoder()

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            model.unfreeze_encoder()
            print(f"[{val_fold}] unfreeze encoder")

        model.train()
        sums = {}
        seen = 0
        for batch in tqdm(dl_tr, ncols=100, desc=f"[{val_fold}] Epoch {epoch:03d}"):
            x = batch['image'].to(device, non_blocking=True)
            type_map = batch['type_map'].to(device, non_blocking=True)
            hv_map   = batch['hv_map'].to(device, non_blocking=True)
            bin_map  = batch['bin_map'].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                y = model(x)
                losses = cellvit_losses(y, type_map=type_map, hv_map=hv_map, bin_map=bin_map)
                loss = losses['loss']

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            seen += bs

        # epoch end
        train_m = {k: v / max(seen, 1) for k, v in sums.items()}
        val_m   = validate_loss(model, dl_va, device)
        print(f"[{val_fold}] Epoch {epoch:03d} | train/loss={train_m['loss']:.4f} | val/loss={val_m['loss']:.4f}")

        # save best
        if val_m['loss'] < best_val:
            best_val = val_m['loss']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_m}, best_path)
            print(f"[{val_fold}] [ckpt] saved: {best_path}")

    # save last
    last_path = out_dir / "last.pth"
    torch.save({'model': model.state_dict(), 'epoch': epochs}, last_path)
    print(f"[{val_fold}] [ckpt] saved: {last_path}")

    return best_path, {"val_loss": best_val}


# -----------------------------
# CV3 main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PanNuke 3-fold CV (HF: RationAI/PanNuke)")
    p.add_argument('--hf_repo', type=str, default='RationAI/PanNuke')

    # train
    p.add_argument('--epochs', type=int, default=130)
    p.add_argument('--freeze_epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out', type=Path, default=Path('Checkpoints/CellViT/cv3'))

    # ViT-256
    p.add_argument('--vit_embed_dim', type=int, default=384)
    p.add_argument('--vit_depth', type=int, default=12)
    p.add_argument('--vit_heads', type=int, default=6)
    p.add_argument('--vit_mlp_ratio', type=float, default=4.0)

    # init ckpts
    p.add_argument('--init_full_ckpt', type=Path, default=None,
                   help='CellViT-256-x40.pth 같이 전체 가중치(.pth, key=model 포함/미포함 모두 허용)')
    p.add_argument('--vit_ckpt', type=Path, default=None,
                   help='백본 전용 ckpt (HIPT/DINO 등)')

    # num_classes (PanNuke: 0=bg + 5 classes)
    p.add_argument('--num_classes', type=int, default=6)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True, exist_ok=True)

    folds = ['fold1', 'fold2', 'fold3']
    all_fold_summaries = []

    for val_fold in folds:
        train_folds = [f for f in folds if f != val_fold]
        fold_out = args.out / f"cv3_{val_fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        print(f"\n========== CV round (val={val_fold} | train={train_folds}) ==========")
        best_ckpt, best_info = train_one_fold(
            repo=args.hf_repo, train_folds=train_folds, val_fold=val_fold,
            out_dir=fold_out, device=device,
            epochs=args.epochs, freeze_epochs=args.freeze_epochs,
            batch_size=args.batch_size, lr=args.lr, workers=args.workers,
            vit_embed_dim=args.vit_embed_dim, vit_depth=args.vit_depth,
            vit_heads=args.vit_heads, vit_mlp_ratio=args.vit_mlp_ratio,
            init_full_ckpt=args.init_full_ckpt, vit_ckpt=args.vit_ckpt,
            num_classes=args.num_classes
        )

        # ----- evaluation on val fold (type classification metrics) -----
        # 로더(다시 생성; val fold만)
        dl_tr, dl_va = build_dl_for_folds(args.hf_repo, train_folds, val_fold, args.batch_size, args.workers)

        # best ckpt 로드해서 평가
        model = CellViTCustom(
            num_nuclei_classes=args.num_classes, num_tissue_classes=0,
            img_size=256, patch_size=16,
            embed_dim=args.vit_embed_dim, depth=args.vit_depth,
            num_heads=args.vit_heads, mlp_ratio=args.vit_mlp_ratio,
        ).to(device).eval()
        sd = torch.load(str(best_ckpt), map_location='cpu')
        sd = sd.get('model', sd)
        model.load_state_dict(sd, strict=False)

        conf, summary, per_class = evaluate_types(model, dl_va, device, num_classes=args.num_classes)
        np.save(fold_out / f"confusion_{val_fold}.npy", conf)

        # save per-fold CSV
        csv_path = fold_out / f"metrics_{val_fold}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "f1", "acc(per-class)", "precision", "recall", "support"])
            id2name = {1:"Neoplastic", 2:"Inflammatory", 3:"Connective", 4:"Dead", 5:"Epithelial"}
            for c in range(1, args.num_classes):  # 1..5
                name = id2name.get(c, f"class_{c}")
                m = per_class[c]
                writer.writerow([c, name, f"{m['f1']:.6f}", f"{m['acc']:.6f}", f"{m['precision']:.6f}", f"{m['recall']:.6f}", m['support']])
            writer.writerow([])
            writer.writerow(["overall_acc", f"{summary['overall_acc']:.6f}"])
            writer.writerow(["macro_f1",    f"{summary['macro_f1']:.6f}"])
            writer.writerow(["macro_acc",   f"{summary['macro_acc']:.6f}"])
        print(f"[{val_fold}] saved metrics -> {csv_path}")

        all_fold_summaries.append((val_fold, summary))

    # ----- CV3 summary -----
    macro_f1s  = [s['macro_f1'] for _, s in all_fold_summaries]
    macro_accs = [s['macro_acc'] for _, s in all_fold_summaries]
    overalls   = [s['overall_acc'] for _, s in all_fold_summaries]

    cv_summary = {
        "macro_f1_mean":  float(np.mean(macro_f1s)),
        "macro_f1_std":   float(np.std(macro_f1s)),
        "macro_acc_mean": float(np.mean(macro_accs)),
        "macro_acc_std":  float(np.std(macro_accs)),
        "overall_acc_mean": float(np.mean(overalls)),
        "overall_acc_std":  float(np.std(overalls)),
    }

    sum_csv = args.out / "metrics_cv3_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "overall_acc", "macro_f1", "macro_acc"])
        for fold_name, s in all_fold_summaries:
            writer.writerow([fold_name, f"{s['overall_acc']:.6f}", f"{s['macro_f1']:.6f}", f"{s['macro_acc']:.6f}"])
        writer.writerow([])
        writer.writerow(["overall_acc_mean", f"{cv_summary['overall_acc_mean']:.6f}"])
        writer.writerow(["overall_acc_std",  f"{cv_summary['overall_acc_std']:.6f}"])
        writer.writerow(["macro_f1_mean",    f"{cv_summary['macro_f1_mean']:.6f}"])
        writer.writerow(["macro_f1_std",     f"{cv_summary['macro_f1_std']:.6f}"])
        writer.writerow(["macro_acc_mean",   f"{cv_summary['macro_acc_mean']:.6f}"])
        writer.writerow(["macro_acc_std",    f"{cv_summary['macro_acc_std']:.6f}"])

    print("\n=== CV3 Summary ===")
    for k, v in cv_summary.items():
        print(f"{k}: {v:.6f}")
    print(f"[Saved] {sum_csv}")


if __name__ == "__main__":
    main()
