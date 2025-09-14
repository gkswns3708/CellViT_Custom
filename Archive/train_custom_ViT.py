# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

# ✅ 모델/로스: 우리 custom(ViT-256, 공식 아키텍처) 버전
from Model.CellViT_ViT256_Custom import CellViTCustom, cellvit_losses

# ✅ 데이터셋: 머지 버전 래퍼
from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged


def parse_args():
    p = argparse.ArgumentParser(description='Train CellViT (ViT-256 encoder) on CoNSeP patches with label merge option')
    p.add_argument('--train_root', type=Path, required=True, help='Dataset/transformed/CoNSeP/Train or ...')
    p.add_argument('--val_root',   type=Path, required=True, help='Dataset/transformed/CoNSeP/Test or ...')

    # 라벨 스킴
    p.add_argument('--label_scheme', type=str, default='consep_merged',
                   choices=['original', 'consep_merged'],
                   help="original(0..7 유지) | consep_merged(3&4→epithelial, 5/6/7→spindle)")

    # 모델/학습 하이퍼파라미터
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('Checkpoints/CellViT'))
    p.add_argument('--device', type=str, default='cuda')

    # ViT-256 백본 하이퍼파라미터 (Model/CellViT_ViT256_Custom 시그니처와 동일)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--vit_embed_dim', type=int, default=384)
    p.add_argument('--vit_depth', type=int, default=12)
    p.add_argument('--vit_heads', type=int, default=6)
    p.add_argument('--vit_mlp_ratio', type=float, default=4.0)
    # num_tissue_classes는 기본 0 (CoNSeP는 대개 없음)
    p.add_argument('--num_tissue_classes', type=int, default=0)

    # 백본 ckpt (HIPT/DINO 형식 teacher/state_dict 키 자동 처리)
    p.add_argument('--vit_ckpt', type=Path, default=None)

    # 클래스 수: 기본 0=자동(스킴에 맞춤). 필요시 강제로 지정 가능
    p.add_argument('--num_classes', type=int, default=0,
                   help='0이면 label_scheme에 따라 자동 설정: original=8, consep_merged=5')

    # W&B
    p.add_argument('--wandb_project', type=str, default='',
                   help='예: "CellViT". 빈 문자열이면 W&B 비활성화')
    p.add_argument('--wandb_run_name', type=str, default='')
    p.add_argument('--wandb_offline', action='store_true',
                   help='--wandb_project가 비어있지 않으면, offline 모드로 로깅')
    return p.parse_args()


def build_loaders(args):
    ds_tr = ConsepPatchDatasetMerged(args.train_root, label_scheme=args.label_scheme)
    ds_va = ConsepPatchDatasetMerged(args.val_root,   label_scheme=args.label_scheme)

    # num_classes 자동결정
    out_classes = ds_tr.out_num_nuclei_classes
    if args.num_classes and args.num_classes > 0:
        out_classes = args.num_classes

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    return dl_tr, dl_va, out_classes


@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    sums = {}
    count = 0
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
        count += bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def maybe_init_wandb(args, num_classes):
    if not args.wandb_project:
        return None
    try:
        import wandb
        mode = 'offline' if args.wandb_offline else 'online'
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, mode=mode,
                   config={
                       'label_scheme': args.label_scheme,
                       'num_classes': num_classes,
                       'epochs': args.epochs,
                       'batch_size': args.batch_size,
                       'lr': args.lr,
                       'img_size': args.img_size,
                       'patch_size': args.patch_size,
                       'vit_embed_dim': args.vit_embed_dim,
                       'vit_depth': args.vit_depth,
                       'vit_heads': args.vit_heads,
                       'vit_mlp_ratio': args.vit_mlp_ratio,
                       'vit_ckpt': str(args.vit_ckpt) if args.vit_ckpt else '',
                   })
        return wandb
    except Exception as e:
        print(f"[W&B] init failed -> continue without W&B: {e}")
        return None


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dl_tr, dl_va, out_classes = build_loaders(args)

    # 모델 생성
    model = CellViTCustom(
        num_nuclei_classes=out_classes,            # ✅ 라벨 스킴 반영된 클래스 수
        num_tissue_classes=args.num_tissue_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_ratio=args.vit_mlp_ratio,
        # drop_rate/attn_drop_rate/drop_path_rate는 기본 0
    )

    # 백본 ckpt 로딩
    if args.vit_ckpt is not None and Path(args.vit_ckpt).exists():
        model.load_vit_checkpoint(str(args.vit_ckpt), strict=False)

    model.to(device)
    print(f"[Model] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | classes={out_classes} | scheme={args.label_scheme}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    wandb = maybe_init_wandb(args, out_classes)

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        sums = {}
        seen = 0
        for batch in tqdm(dl_tr, ncols=100, desc=f"Epoch {epoch:03d}"):
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

        train_metrics = {f"train/{k}": v / max(seen, 1) for k, v in sums.items()}
        val_metrics = {f"val/{k}": v for k, v in validate(model, dl_va, device).items()}
        print(f"Epoch {epoch:03d} | "
              + " | ".join([f"{k}:{v:.4f}" for k, v in train_metrics.items() if k.endswith('/loss')])
              + " || "
              + " | ".join([f"{k}:{v:.4f}" for k, v in val_metrics.items() if k.endswith('/loss')]))

        if wandb is not None:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        # best ckpt 기준: val/loss
        val_score = val_metrics.get('val/loss', 1e9)
        if val_score < best_val:
            best_val = val_score
            ckpt_path = args.out / f"cellvit_vit256_{args.label_scheme}_best.pth"
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_metrics}, ckpt_path)
            print(f"[ckpt] saved: {ckpt_path}")

    last_path = args.out / f"cellvit_vit256_{args.label_scheme}_last.pth"
    torch.save({'model': model.state_dict(), 'epoch': args.epochs}, last_path)
    print(f"[ckpt] saved: {last_path} | Done.")


if __name__ == '__main__':
    main()
