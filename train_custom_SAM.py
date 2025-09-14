
# -*- coding: utf-8 -*-
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Data.CoNSeP_patch import ConsepPatchDataset
from Model.CellViT_SAM_Custom import CellViTCustom, cellvit_losses


def parse_args():
    p = argparse.ArgumentParser(description='Train CellViTCustom with ViT encoder on patched CoNSeP')
    p.add_argument('--train_root', type=Path, required=True, help='e.g., Dataset/transformed/CoNSeP/Train')
    p.add_argument('--val_root', type=Path, required=True, help='e.g., Dataset/transformed/CoNSeP/Test')
    p.add_argument('--num_classes', type=int, default=6, help='num nuclei classes incl. background (CoNSeP default=6)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('Checkpoints/CellViT'))
    p.add_argument('--vit_ckpt', type=Path, default=None, help='optional path to ViT encoder checkpoint (.pth)')
    p.add_argument('--pos_weight', type=float, default=None, help='binary CE positive class weight, e.g. 2.0')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def build_loaders(train_root, val_root, num_classes, batch_size, workers):
    ds_tr = ConsepPatchDataset(train_root, num_nuclei_classes=num_classes)
    ds_va = ConsepPatchDataset(val_root,   num_nuclei_classes=num_classes)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return dl_tr, dl_va


def validate(model, dl, device):
    model.eval()
    m = {'loss':0.0, 'loss_bin':0.0, 'loss_type':0.0, 'loss_hv':0.0}
    n = 0
    with torch.no_grad():
        for batch in dl:
            x = batch['image'].to(device)
            type_map = batch['type_map'].to(device)
            hv_map   = batch['hv_map'].to(device)
            bin_map  = batch['bin_map'].to(device)

            y = model(x)
            losses = cellvit_losses(y, type_map=type_map, hv_map=hv_map, bin_map=bin_map)
            bs = x.size(0)
            for k in m: m[k] += float(losses.get(k, 0.0)) * bs
            n += bs
    for k in m: m[k] /= max(n,1)
    return m


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data
    dl_tr, dl_va = build_loaders(args.train_root, args.val_root, args.num_classes, args.batch_size, args.workers)

    # Model
    model = CellViTCustom(num_nuclei_classes=args.num_classes)
    if args.vit_ckpt is not None and Path(args.vit_ckpt).exists():
        model.load_vit_checkpoint(str(args.vit_ckpt), strict=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        model.train()
        running = {'loss':0.0, 'loss_bin':0.0, 'loss_type':0.0, 'loss_hv':0.0}
        seen = 0
        for batch in tqdm(dl_tr):
            x = batch['image'].to(device)
            type_map = batch['type_map'].to(device)
            hv_map   = batch['hv_map'].to(device)
            bin_map  = batch['bin_map'].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                y = model(x)
                losses = cellvit_losses(
                    y, type_map=type_map, hv_map=hv_map, bin_map=bin_map,
                    pos_weight=args.pos_weight,
                )
                loss = losses['loss']

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            for k in running: running[k] += float(losses.get(k, 0.0)) * bs
            seen += bs

        for k in running: running[k] /= max(seen,1)

        # Validation
        val_m = validate(model, dl_va, device)
        print(f"Epoch {epoch:03d} | train: {running} | val: {val_m}")

        # checkpoint
        val_score = val_m['loss']
        if val_score < best_val:
            best_val = val_score
            ckpt_path = args.out / f"cellvitcustom_SAM_best.pth"
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_m}, ckpt_path)
            print(f"[ckpt] saved: {ckpt_path}")

    # final save
    last_path = args.out / f"cellvitcustom_last.pth"
    torch.save({'model': model.state_dict(), 'epoch': args.epochs}, last_path)
    print(f"[ckpt] saved: {last_path}Done.")


if __name__ == '__main__':
    main()
