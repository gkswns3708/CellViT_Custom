# train_custom_ViT.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

# 모델/로스
from Model.CellViT_ViT256_Custom import CellViTCustom, cellvit_losses

# 데이터셋
from Data.PanNuke_hf_instances import PanNukeHFInstancesDataset
from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged  # (옵션) 여전히 사용 가능


def parse_args():
    p = argparse.ArgumentParser(description='Train CellViT (ViT-256) with HF PanNuke or CoNSeP')

    # ---------------- common ----------------
    p.add_argument('--dataset', type=str, default='pannuke_hf',
                   choices=['pannuke_hf', 'consep_merged'],
                   help='pannuke_hf (RationAI/PanNuke) | consep_merged (local CoNSeP)')
    p.add_argument('--out', type=Path, default=Path('Checkpoints/CellViT'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--freeze_epochs', type=int, default=0,
                   help='>0이면 처음 N epoch 동안 encoder 동결')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)

    # ---------------- PanNuke(HF) ----------------
    p.add_argument('--hf_repo', type=str, default='RationAI/PanNuke')
    p.add_argument('--hf_train_folds', type=str, default='fold1',
                   help='쉼표로 구분: fold1,fold2 ...')
    p.add_argument('--hf_val_folds', type=str, default='fold2',
                   help='쉼표로 구분: fold2 ...')
    # PanNuke 클래스 수: 0=bg + 5(Neopl,Infl,Conn,Dead,Epit) = 6
    p.add_argument('--num_classes', type=int, default=6,
                   help='type head classes incl. background; PanNuke=6')

    # ---------------- CoNSeP(local, merged) ----------------
    p.add_argument('--train_root', type=Path, default=None,
                   help='Dataset/transformed/CoNSeP/Train')
    p.add_argument('--val_root', type=Path, default=None,
                   help='Dataset/transformed/CoNSeP/Test')

    # ---------------- ViT-256 ----------------
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--vit_embed_dim', type=int, default=384)
    p.add_argument('--vit_depth', type=int, default=12)
    p.add_argument('--vit_heads', type=int, default=6)
    p.add_argument('--vit_mlp_ratio', type=float, default=4.0)
    p.add_argument('--num_tissue_classes', type=int, default=0)

    # ---------------- init/ckpt ----------------
    p.add_argument('--init_full_ckpt', type=Path, default=None,
                   help='전체 모델 state_dict를 로드(헤드 포함, strict=False 권장)')
    p.add_argument('--vit_ckpt', type=Path, default=None,
                   help='백본 전용 ckpt (HIPT/DINO 스타일 등); encoder에만 로드')

    # ---------------- W&B ----------------
    p.add_argument('--wandb_project', type=str, default='',
                   help='예: "CellViT"; 빈 문자열이면 사용 안 함')
    p.add_argument('--wandb_run_name', type=str, default='')
    p.add_argument('--wandb_offline', action='store_true')
    return p.parse_args()


def build_loaders(args):
    if args.dataset == 'pannuke_hf':
        train_folds = [s.strip() for s in args.hf_train_folds.split(',') if s.strip()]
        val_folds   = [s.strip() for s in args.hf_val_folds.split(',') if s.strip()]
        ds_tr = PanNukeHFInstancesDataset(repo_id=args.hf_repo, folds=train_folds, normalize=True)
        ds_va = PanNukeHFInstancesDataset(repo_id=args.hf_repo, folds=val_folds,   normalize=True)
        out_classes = args.num_classes  # 6 권장
    else:
        # local CoNSeP merged
        if args.train_root is None or args.val_root is None:
            raise SystemExit("--train_root/--val_root must be provided for consep_merged.")
        ds_tr = ConsepPatchDatasetMerged(args.train_root, label_scheme='consep_merged')
        ds_va = ConsepPatchDatasetMerged(args.val_root,   label_scheme='consep_merged')
        out_classes = 5  # 0..4 (bg+4 merged classes)

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
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            mode=mode,
            config={
                'dataset': args.dataset,
                'hf_repo': args.hf_repo,
                'hf_train_folds': args.hf_train_folds,
                'hf_val_folds': args.hf_val_folds,
                'num_classes': num_classes,
                'epochs': args.epochs,
                'freeze_epochs': args.freeze_epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'img_size': args.img_size,
                'patch_size': args.patch_size,
                'vit_embed_dim': args.vit_embed_dim,
                'vit_depth': args.vit_depth,
                'vit_heads': args.vit_heads,
                'vit_mlp_ratio': args.vit_mlp_ratio,
                'init_full_ckpt': str(args.init_full_ckpt) if args.init_full_ckpt else '',
                'vit_ckpt': str(args.vit_ckpt) if args.vit_ckpt else '',
            }
        )
        return wandb
    except Exception as e:
        print(f"[W&B] init failed -> continue without W&B: {e}")
        return None


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dl_tr, dl_va, out_classes = build_loaders(args)

    # ----- Model -----
    model = CellViTCustom(
        num_nuclei_classes=out_classes,
        num_tissue_classes=args.num_tissue_classes,  # PanNuke tissue label은 미사용(0)
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_ratio=args.vit_mlp_ratio,
    )

    # init: full-model ckpt 우선
    if args.init_full_ckpt is not None and Path(args.init_full_ckpt).exists():
        print(f"[init] loading FULL model ckpt: {args.init_full_ckpt}")
        sd = torch.load(str(args.init_full_ckpt), map_location='cpu')
        sd = sd.get('model', sd)
        msg = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded (strict=False).")
        try:
            print("  missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
        except Exception:
            pass

    # init: encoder-only ckpt (옵션)
    elif args.vit_ckpt is not None and Path(args.vit_ckpt).exists():
        print(f"[init] loading VIT encoder ckpt: {args.vit_ckpt}")
        model.load_vit_checkpoint(str(args.vit_ckpt), strict=False)

    model.to(device)
    print(f"[Model] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | classes={out_classes}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    wandb = maybe_init_wandb(args, out_classes)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        # freeze schedule
        if epoch == 1 and args.freeze_epochs > 0:
            print(f"[freeze] encoder frozen for first {args.freeze_epochs} epochs")
            model.freeze_encoder()
        if epoch == args.freeze_epochs + 1:
            print("[unfreeze] encoder")
            model.unfreeze_encoder()

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
        print(
            f"Epoch {epoch:03d} | "
            + " | ".join([f"{k}:{v:.4f}" for k, v in train_metrics.items() if k.endswith('/loss')])
            + " || "
            + " | ".join([f"{k}:{v:.4f}" for k, v in val_metrics.items() if k.endswith('/loss')])
        )

        if wandb is not None:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        # best ckpt: val/loss
        val_score = val_metrics.get('val/loss', 1e9)
        if val_score < best_val:
            best_val = val_score
            tag = "pannuke_hf" if args.dataset == 'pannuke_hf' else "consep_merged"
            ckpt_path = args.out / f"cellvit_vit256_{tag}_best.pth"
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_metrics}, ckpt_path)
            print(f"[ckpt] saved: {ckpt_path}")

    last_tag = "pannuke_hf" if args.dataset == 'pannuke_hf' else "consep_merged"
    last_path = args.out / f"cellvit_vit256_{last_tag}_last.pth"
    torch.save({'model': model.state_dict(), 'epoch': args.epochs}, last_path)
    print(f"[ckpt] saved: {last_path} | Done.")


if __name__ == '__main__':
    main()
