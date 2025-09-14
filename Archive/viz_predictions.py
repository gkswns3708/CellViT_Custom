# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# local datasets
from Data.CoNSeP_patch import ConsepPatchDataset
from Data.CoNSeP_patch_merged import ConsepPatchDatasetMerged
# HF PanNuke loader
from Data.PanNuke_hf import PanNukeHFDataset

from Model.CellViT_ViT256_Custom import CellViTCustom


# -----------------------------
# Label spec (id->name, palette, remap)
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
        # PanNuke: 0=bg, 1..5 = Neoplastic, Inflammatory, Connective, Dead, Epithelial
        id2name = {
            1:"Neoplastic", 2:"Inflammatory", 3:"Connective", 4:"Dead", 5:"Epithelial"
        }
        remap = {k:k for k in id2name.keys()}  # no merge
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
# Confusion/metrics
# -----------------------------
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


# -----------------------------
# Eval + Visualization
# -----------------------------
@torch.no_grad()
def evaluate_and_visualize(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset/Loader
    if args.dataset == 'pannuke_hf':
        ds = PanNukeHFDataset(
            repo_id=args.hf_repo,
            split=args.hf_split,   # 기본 fold1
            fold=args.hf_fold,
            normalize=True
        )
    elif args.dataset == 'consep_merged':
        ds = ConsepPatchDatasetMerged(args.root, label_scheme='consep_merged')
    elif args.dataset == 'consep':
        ds = ConsepPatchDataset(args.root, num_nuclei_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # Label spec
    id2name, palette, remap = label_spec(args.dataset, args.merge_consep)

    # Model
    model = CellViTCustom(
        num_nuclei_classes=args.num_classes,
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim, depth=args.vit_depth,
        num_heads=args.vit_heads, mlp_ratio=args.vit_mlp_ratio,
    ).to(device).eval()

    # Load weights
    if args.ckpt is not None and Path(args.ckpt).exists():
        ck = torch.load(str(args.ckpt), map_location='cpu')
        sd = ck.get('model', ck)
        model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded model weights: {args.ckpt}")
    else:
        print("[eval] WARNING: no checkpoint provided — random weights!")

    # confusion: include bg (0..C)
    C = args.num_classes - 1
    conf = np.zeros((C+1, C+1), dtype=np.int64)
    classes_to_eval = list(range(1, C+1)) if args.exclude_bg else list(range(0, C+1))

    saved = 0
    for batch in dl:
        x = batch['image'].to(device, non_blocking=True)
        t = batch['type_map'].cpu().numpy()
        paths = batch['path_image']

        # forward
        nt_logits = model(x)['nuclei_type_map']
        nt_pred = torch.argmax(nt_logits, dim=1).cpu().numpy()

        # CoNSeP merged remap
        if args.dataset.startswith('consep') and args.merge_consep:
            vt = np.vectorize(lambda k: remap.get(int(k), int(k)))
            t = np.stack([vt(t[i]) for i in range(t.shape[0])], axis=0)
            nt_pred = np.stack([vt(nt_pred[i]) for i in range(nt_pred.shape[0])], axis=0)

        t = np.clip(t, 0, C); nt_pred = np.clip(nt_pred, 0, C)
        update_confusion(conf, t.reshape(-1), nt_pred.reshape(-1), C)

        if args.max_samples > 0:
            for b in range(x.size(0)):
                if saved >= args.max_samples: break
                img = (x[b].cpu().numpy().transpose(1,2,0) * 255.0).round().astype(np.uint8)
                gt_rgb = colorize_label(t[b], palette)
                pr_rgb = colorize_label(nt_pred[b], palette)
                gt_ov  = blend_overlay(img, gt_rgb, alpha=args.type_alpha)
                pr_ov  = blend_overlay(img, pr_rgb, alpha=args.type_alpha)

                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(2,2,1); ax1.imshow(img);   ax1.set_title("Input");       ax1.axis('off')
                ax2 = plt.subplot(2,2,2); ax2.imshow(gt_ov); ax2.set_title("GT Type");     ax2.axis('off')
                ax3 = plt.subplot(2,2,3); ax3.imshow(pr_ov); ax3.set_title("Pred Type");   ax3.axis('off')

                handles = build_legend_handles(id2name, palette)
                ax4 = plt.subplot(2,2,4); ax4.axis('off')
                ax4.legend(handles=handles, title="Cell types", loc='center'); ax4.set_title("Legend")

                stem = Path(paths[b]).stem if isinstance(paths[b], str) else str(paths[b])
                out_path = out_dir / f"{stem}_typeviz_gt_pred.png"
                plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
                saved += 1

    overall_acc, per_class = metrics_from_confusion(conf, classes_to_eval)
    print("\n[Type classification metrics]")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class (exclude_bg={}):".format(args.exclude_bg))
    for c in classes_to_eval:
        name = id2name.get(c, f"class_{c}")
        m = per_class[c]
        print(f"  {c:>2} ({name:>27s}) | F1={m['f1']:.4f}  Acc={m['acc']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (support={m['support']})")

    # save CSV + confusion
    import csv
    csv_path = out_dir / "type_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id","class_name","f1","acc(per-class)","precision","recall","support"])
        for c in classes_to_eval:
            name = id2name.get(c, f"class_{c}"); m = per_class[c]
            writer.writerow([c, name, f"{m['f1']:.6f}", f"{m['acc']:.6f}",
                             f"{m['precision']:.6f}", f"{m['recall']:.6f}", m['support']])
    np.save(out_dir / "confusion.npy", conf)
    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {out_dir/'confusion.npy'}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate per-class F1/ACC and visualize GT/PRED type maps.")
    # dataset selector
    p.add_argument('--dataset', type=str, default='pannuke_hf',
                   choices=['consep', 'consep_merged', 'pannuke_hf'])

    # CoNSeP: needs root
    p.add_argument('--root', type=Path, default=None,
                   help='Dataset root containing images/ and labels/ (for consep/consep_merged)')
    p.add_argument('--merge_consep', action='store_true',
                   help='Only for CoNSeP: merge (3,4)->epithelial; (5,6,7)->spindle')

    # PanNuke(HF)
    p.add_argument('--hf_repo', type=str, default='tio-ikim/pannuke')
    p.add_argument('--hf_split', type=str, default='fold1')  # <-- 기본 fold1 (이전 validation 에러 방지)
    p.add_argument('--hf_fold', type=int, default=None)

    # common
    p.add_argument('--ckpt', type=Path, required=True, help='Trained CellViT checkpoint (.pth with key "model")')
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
    # consep=8, consep_merged=5, pannuke=6
    p.add_argument('--num_classes', type=int, default=6,
                   help='#type classes incl. background. consep=8, consep_merged=5, pannuke=6')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset in ('consep', 'consep_merged') and args.root is None:
        raise SystemExit("--root is required for CoNSeP datasets.")
    evaluate_and_visualize(args)
