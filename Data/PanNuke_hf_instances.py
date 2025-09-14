# Data/PanNuke_hf_instances.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    # optional: SciPy가 있으면 더 빠르고 정확한 center_of_mass 사용
    from scipy.ndimage import center_of_mass
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from datasets import load_dataset


def _center_of_mass(mask: np.ndarray) -> tuple[float, float]:
    if _HAS_SCIPY:
        cy, cx = center_of_mass(mask)
        return float(cy), float(cx)
    # SciPy 없을 때 fallback
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """
    inst_map: (H,W) int, 0=bg, 1..K instance id
    return: (2,H,W) float32 in [-1,1] approximately (center-of-mass normalized)
    """
    h, w = inst_map.shape[:2]
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)

    ids = [i for i in np.unique(inst_map) if i != 0]
    for inst_id in ids:
        m = (inst_map == inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        cy, cx = _center_of_mass(m)
        yy, xx = np.nonzero(m)
        dx = (xx - cx).astype(np.float32)
        dy = (yy - cy).astype(np.float32)
        # -1..1 정규화 (좌/우, 상/하 각각 최대 절대값 기준)
        if np.any(dx < 0):
            dx[dx < 0] /= -np.min(dx[dx < 0])
        if np.any(dx > 0):
            dx[dx > 0] /= np.max(dx[dx > 0])
        if np.any(dy < 0):
            dy[dy < 0] /= -np.min(dy[dy < 0])
        if np.any(dy > 0):
            dy[dy > 0] /= np.max(dy[dy > 0])
        x_map[yy, xx] = dx
        y_map[yy, xx] = dy
    return np.stack([x_map, y_map], axis=0)


def _to_bool_mask(pil_img: Image.Image) -> np.ndarray:
    """
    PIL '1' 모드(흑백 바이너리) 또는 L/RGB라도 True/False로 안전 변환
    """
    arr = np.array(pil_img)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)


class PanNukeHFInstancesDataset(Dataset):
    """
    HF Datasets: RationAI/PanNuke
      folds: 'fold1' | 'fold2' | 'fold3'
      record:
        - image: PIL Image (256x256, RGB)
        - instances: list[PIL binary masks], one per nucleus
        - categories: list[int] (0..4) per nucleus
        - tissue: int (0..18)  -- (옵션) 사용 안 함
    반환 텐서:
      image: FloatTensor (3,H,W) [0..1]
      inst_map: LongTensor (H,W)
      type_map: LongTensor (H,W)   # 0=bg, 1..5(Neopl,Infl,Conn,Dead,Epit)
      bin_map : LongTensor (H,W)   # (inst_map>0)
      hv_map  : FloatTensor (2,H,W)
      path_image: str              # fold_idx 형태의 식별자
    """
    def __init__(
        self,
        repo_id: str = "RationAI/PanNuke",
        folds: Iterable[str] = ("fold1",),
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        folds = list(folds)
        self.normalize = normalize

        # 전체 DatasetDict 로드 후 필요한 폴드만 보관
        ds_dict = load_dataset(repo_id, cache_dir=cache_dir)
        # 유효 폴드만
        self._folds: List[str] = []
        self._ds_by_fold: Dict[str, Any] = {}
        for f in folds:
            if f not in ds_dict:
                raise ValueError(f"Fold '{f}' not in dataset. Available: {list(ds_dict.keys())}")
            self._folds.append(f)
            self._ds_by_fold[f] = ds_dict[f]

        # 인덱스 매핑: [(fold_name, local_idx)]
        self._index: List[tuple[str, int]] = []
        for f in self._folds:
            n = self._ds_by_fold[f].num_rows
            self._index.extend([(f, i) for i in range(n)])

    def __len__(self) -> int:
        return len(self._index)

    def _compose_maps(self, instances: List[Image.Image], categories: List[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        instances: list of PIL bin masks
        categories: list[int] (0..4)
        return: (inst_map[H,W], type_map[H,W])
          - inst_map: 1..K
          - type_map: 0..5 (0=bg, 1..5 as 5 categories)
        """
        h = instances[0].height if instances else 256
        w = instances[0].width  if instances else 256
        inst_map = np.zeros((h, w), dtype=np.int32)
        type_map = np.zeros((h, w), dtype=np.int32)

        # 인스턴스 ID는 1부터, 카테고리는 0..4 -> (배경 포함) 1..5로 +1
        for k, (m_img, c) in enumerate(zip(instances, categories), start=1):
            m = _to_bool_mask(m_img)  # (H,W) {0,1}
            if m.sum() == 0:
                continue
            inst_map[m > 0] = k
            type_map[m > 0] = (int(c) + 1)
        return inst_map, type_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        fold, j = self._index[idx]
        rec = self._ds_by_fold[fold][j]

        # image -> (3,H,W) float
        img = rec["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img_arr = np.array(img).astype(np.uint8)
        if img_arr.ndim == 2:
            img_arr = np.repeat(img_arr[..., None], 3, axis=2)
        img_t = torch.from_numpy(img_arr).float().permute(2, 0, 1)
        if self.normalize and img_t.max() > 1.0:
            img_t = img_t / 255.0

        # instances + categories -> inst_map, type_map
        instances = rec["instances"]
        categories = rec["categories"]
        inst_map, type_map = self._compose_maps(instances, categories)

        # aux maps
        bin_map = (inst_map > 0).astype(np.int64)
        hv_map = gen_instance_hv_map(inst_map).astype(np.float32)

        return {
            "image": img_t,                                  # (3,H,W) float
            "inst_map": torch.from_numpy(inst_map).long(),   # (H,W)
            "type_map": torch.from_numpy(type_map).long(),   # (H,W) 0..5
            "bin_map":  torch.from_numpy(bin_map).long(),    # (H,W)
            "hv_map":   torch.from_numpy(hv_map).float(),    # (2,H,W)
            "path_image": f"{fold}_{j:06d}.png",
        }
