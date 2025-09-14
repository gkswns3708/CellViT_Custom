# Data/PanNuke_hf.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import io

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    # 프로젝트에 PanNukeDataset 유틸이 있으면 재사용
    from Data.PanNukeDataset import PanNukeDataset as _PNK
    _has_pannuke_utils = True
except Exception:
    _has_pannuke_utils = False

from datasets import load_dataset


def _gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """PanNukeDataset.gen_instance_hv_map 대체(없을 때만 사용)."""
    if _has_pannuke_utils:
        return _PNK.gen_instance_hv_map(inst_map)

    # fallback: 간단한 CoM 기반 -1..1 정규화
    from scipy.ndimage import center_of_mass
    h, w = inst_map.shape[:2]
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    ids = [i for i in np.unique(inst_map) if i != 0]
    for inst_id in ids:
        m = (inst_map == inst_id).astype(np.uint8)
        if m.sum() == 0: continue
        (cy, cx) = center_of_mass(m)
        yy, xx = np.nonzero(m)
        dx = (xx - cx).astype(np.float32)
        dy = (yy - cy).astype(np.float32)
        if np.any(dx < 0): dx[dx < 0] /= -np.min(dx[dx < 0])
        if np.any(dx > 0): dx[dx > 0] /=  np.max(dx[dx > 0])
        if np.any(dy < 0): dy[dy < 0] /= -np.min(dy[dy < 0])
        if np.any(dy > 0): dy[dy > 0] /=  np.max(dy[dy > 0])
        x_map[yy, xx] = dx
        y_map[yy, xx] = dy
    return np.stack([x_map, y_map], axis=0)  # (2,H,W)


def _smart_load_split(repo_id: str, split: str, cache_dir: Optional[str]):
    """
    리포지토리마다 split 이름이 다름.
    - 바로 로드 시도
    - 실패하면: 전체를 로드해 키 목록을 보고 mapping 시도
    """
    try:
        return load_dataset(repo_id, split=split, cache_dir=cache_dir), split
    except Exception as e:
        # 전체 로드해서 split 키 노출
        ds_dict = load_dataset(repo_id, cache_dir=cache_dir)
        available = list(ds_dict.keys())
        # alias 매핑 시도
        alias = {'train': 'fold1', 'validation': 'fold2', 'test': 'fold3'}
        if split in alias and alias[split] in available:
            print(f"[PanNukeHFDataset] mapping split '{split}' -> '{alias[split]}' (available: {available})")
            return ds_dict[alias[split]], alias[split]
        if split not in available:
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")
        return ds_dict[split], split


class PanNukeHFDataset(Dataset):
    """
    HuggingFace Datasets로 PanNuke를 읽어오는 래퍼.
    각 item:
      {
        'image': FloatTensor (3,H,W) [0..1],
        'inst_map': LongTensor (H,W),
        'type_map': LongTensor (H,W),      # 0=bg, 1..5 classes
        'bin_map' : LongTensor (H,W),      # (inst_map>0)
        'hv_map'  : FloatTensor (2,H,W),   # [-1..1] approx
        'path_image': str
      }
    """
    def __init__(
        self,
        repo_id: str = "RationAI/PanNuke",
        split: str = "fold1",               # repo에 따라 fold1/2/3 또는 train/validation/test
        fold: Optional[int] = None,         # fold 컬럼 존재 시 필터링 옵션
        cache_dir: Optional[str] = None,
        image_key: str = "image",
        mask_key_candidates: tuple = ("mask", "label", "mask_path", "npy_path"),
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize

        ds, used_split = _smart_load_split(repo_id, split, cache_dir)
        # fold 컬럼이 있으면 필터링
        if fold is not None and "fold" in ds.column_names:
            ds = ds.filter(lambda x: int(x["fold"]) == int(fold))

        # 필수 컬럼 점검
        assert image_key in ds.column_names, f"'{image_key}' column not found in HF dataset."
        self.image_key = image_key
        self.mask_key = None
        for k in mask_key_candidates:
            if k in ds.column_names:
                self.mask_key = k
                break
        if self.mask_key is None:
            raise KeyError(f"Mask column not found. Tried: {mask_key_candidates}")

        self.ds = ds

        # 이미지 이름/경로 유사 컬럼 추정
        self._name_key = None
        for cand in ("img", "image_id", "image_name", "file_name", "filename"):
            if cand in self.ds.column_names:
                self._name_key = cand
                break
        self._name_prefix = f"{Path(repo_id).name}_{used_split}"

    def __len__(self) -> int:
        return self.ds.num_rows

    def _load_image(self, rec: Dict[str, Any]) -> np.ndarray:
        img = rec[self.image_key]
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.uint8)
        elif isinstance(img, (np.ndarray, list)):
            arr = np.array(img, dtype=np.uint8)
        else:
            arr = np.array(Image.open(img)).astype(np.uint8)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.repeat(arr[..., None], 3, axis=2)
        return arr

    def _load_mask(self, rec: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        m = rec[self.mask_key]

        def _from_npy_like(obj):
            if isinstance(obj, (str, Path)):
                return np.load(obj, allow_pickle=True)
            if isinstance(obj, (bytes, bytearray)):
                return np.load(io.BytesIO(obj), allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                return obj.item()
            if isinstance(obj, dict):
                return obj
            raise TypeError(f"Unsupported mask object type: {type(obj)}")

        mm = _from_npy_like(m)
        mm = mm if isinstance(mm, dict) else mm[()]

        # {'inst_map':..., 'type_map':...} 기대
        inst_map = mm["inst_map"].astype(np.int32)
        type_map = mm["type_map"].astype(np.int32)
        return inst_map, type_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rec = self.ds[int(idx)]

        # image
        img = self._load_image(rec)  # (H,W,3) uint8
        img_t = torch.from_numpy(img).float().permute(2, 0, 1)  # (3,H,W)
        if self.normalize and img_t.max() > 1.0:
            img_t = img_t / 255.0

        # mask -> inst/type/bin/hv
        inst_map, type_map = self._load_mask(rec)
        bin_map = (inst_map > 0).astype(np.int64)
        hv_map = _gen_instance_hv_map(inst_map)

        name = rec.get(self._name_key, f"{self._name_prefix}_{idx:06d}")

        sample = {
            "image": img_t,                                              # (3,H,W) float32
            "inst_map": torch.from_numpy(inst_map).long(),               # (H,W) int64
            "type_map": torch.from_numpy(type_map).long(),               # (H,W) int64
            "bin_map":  torch.from_numpy(bin_map).long(),                # (H,W) int64
            "hv_map":   torch.from_numpy(hv_map).float(),                # (2,H,W) float32
            "path_image": name,                                          # string
        }
        return sample
