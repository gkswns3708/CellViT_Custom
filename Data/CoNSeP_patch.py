# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class ConsepPatchDataset(Dataset):
    """Reads transformed patches produced by preprocess script.

    Expects directory with:
      - images/*.png
      - labels/*.npz  (inst_map, type_map, hv_map[, bin_map])
      - meta.csv      (optional; if present, used for deterministic ordering)
    """
    def __init__(
        self,
        root: str | Path,
        num_nuclei_classes: int,
        use_meta: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.lbl_dir = self.root / 'labels'
        self.num_nuclei_classes = num_nuclei_classes
        self.normalize = normalize

        if use_meta and (self.root / 'meta.csv').exists():
            import csv
            rows = []
            with open(self.root / 'meta.csv', 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            # trust the recorded paths if exist
            items = []
            for r in rows:
                ip = Path(r['image_path']) if 'image_path' in r else (self.img_dir / f"{Path(r['case_id']).stem}_r{r['row']}_c{r['col']}.png")
                lp = Path(r['label_path']) if 'label_path' in r else (self.lbl_dir / f"{Path(r['case_id']).stem}_r{r['row']}_c{r['col']}.npz")
                if ip.exists() and lp.exists():
                    items.append((ip, lp))
            self.items = items
        else:
            # scan directories
            pngs = sorted(self.img_dir.glob('*.png'))
            items = []
            for p in pngs:
                stem = p.stem
                lbl = self.lbl_dir / f"{stem}.npz"
                if lbl.exists():
                    items.append((p, lbl))
            self.items = items
        if not self.items:
            raise FileNotFoundError(f"No patches found under {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _to_tensor_img(arr: np.ndarray, normalize: bool) -> torch.Tensor:
        # arr: HxWx3 uint8
        x = torch.from_numpy(arr).permute(2,0,1).float()  # C,H,W
        if normalize:
            x = x / 255.0
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, lbl_path = self.items[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        lab = np.load(lbl_path)
        inst_map = lab['inst_map'].astype(np.int64)
        type_map = lab['type_map'].astype(np.int64)
        hv_map   = lab['hv_map'].astype(np.float32)  # (2,H,W)
        bin_map  = lab['bin_map'].astype(np.int64) if 'bin_map' in lab else (type_map > 0).astype(np.int64)

        # tensors
        x = self._to_tensor_img(img, self.normalize)
        t_type = torch.from_numpy(type_map)
        t_inst = torch.from_numpy(inst_map)
        t_hv   = torch.from_numpy(hv_map)
        t_bin  = torch.from_numpy(bin_map)

        return {
            'image': x,
            'type_map': t_type,   # (H,W) long
            'inst_map': t_inst,   # (H,W) long
            'hv_map': t_hv,       # (2,H,W) float
            'bin_map': t_bin,     # (H,W) long {0,1}
            'path_image': str(img_path),
            'path_label': str(lbl_path),
        }