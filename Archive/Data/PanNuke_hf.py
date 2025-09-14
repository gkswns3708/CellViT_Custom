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
    # 공식 PanNuke util이 이미 프로젝트에 있다면 재사용
    from Data.PanNukeDataset import PanNukeDataset as _PNK
    _has_pannuke_utils = True
except Exception:
    _has_pannuke_utils = False

from datasets import load_dataset


def _gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """PanNukeDataset.gen_instance_hv_map 대체(없을 때만 사용)."""
    if _has_pannuke_utils:
        return _PNK.gen_instance_hv_map(inst_map)

    # --- 간결한 fallback: center-of-mass 기반 -1..1 정규화(H/V) ---
    from scipy.ndimage import center_of_mass
    h, w = inst_map.shape[:2]
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    ids = [i for i in np.unique(inst_map) if i != 0]
    for inst_id in ids:
        m = (inst_map == inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        (cy, cx) = center_of_mass(m)
        yy, xx = np.nonzero(m)
        # 좌표 차이
        dx = (xx - cx).astype(np.float32)
        dy = (yy - cy).astype(np.float32)
        # -1..1 정규화
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
    return np.stack([x_map, y_map], axis=0)  # (2,H,W)


class PanNukeHFDataset(Dataset):
    """
    HuggingFace Datasets로 PanNuke를 읽어오는 래퍼.
    각 item은 다음을 반환:
      {
        'image': FloatTensor (3,H,W) [0..1],
        'inst_map': LongTensor (H,W),
        'type_map': LongTensor (H,W),      # 0=bg, 1..5 classes
        'bin_map' : LongTensor (H,W),      # (inst_map>0)
        'hv_map'  : FloatTensor (2,H,W),   # [-1..1] 대략
        'path_image': str                  # 원본 이미지 이름/경로 유사 표기
      }
    """
    def __init__(
        self,
        repo_id: str = "RationAI/PanNuke",   # 실제 Hub id에 맞게 바꾸세요
        split: str = "validation",           # "train"/"validation"/"test" 등
        fold: Optional[int] = None,          # fold 컬럼이 있을 때 필터링
        cache_dir: Optional[str] = None,
        image_key: str = "image",            # 이미지가 저장된 컬럼명
        mask_key_candidates: tuple = ("mask", "label", "mask_path", "npy_path"),
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize

        ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        # fold 컬럼이 존재하면 필터링
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

        # 편의: 이미지 이름/경로 비슷한 컬럼 추정
        self._name_key = None
        for cand in ("img", "image_id", "image_name", "file_name", "filename"):
            if cand in self.ds.column_names:
                self._name_key = cand
                break

    def __len__(self) -> int:
        return self.ds.num_rows

    def _load_image(self, rec: Dict[str, Any]) -> np.ndarray:
        img = rec[self.image_key]
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.uint8)
        elif isinstance(img, (np.ndarray, list)):
            arr = np.array(img, dtype=np.uint8)
        else:
            # datasets에서 경로 문자열만 줄 수도 있음
            arr = np.array(Image.open(img)).astype(np.uint8)
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

        # mm가 dict 형태이길 기대: {'inst_map':..., 'type_map':...}
        if isinstance(mm, dict) and "inst_map" in mm and "type_map" in mm:
            inst_map = mm["inst_map"].astype(np.int32)
            type_map = mm["type_map"].astype(np.int32)
        else:
            # .npy가 dict를 래핑한 object-array인 일반 케이스 처리
            mm = mm if isinstance(mm, dict) else mm[()]
            inst_map = mm["inst_map"].astype(np.int32)
            type_map = mm["type_map"].astype(np.int32)

        return inst_map, type_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rec = self.ds[int(idx)]

        # image
        img = self._load_image(rec)  # (H,W,3) uint8
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        img_t = torch.from_numpy(img).float().permute(2, 0, 1)  # (3,H,W)
        if self.normalize and img_t.max() > 1.0:
            img_t = img_t / 255.0

        # mask -> inst/type/bin/hv
        inst_map, type_map = self._load_mask(rec)
        bin_map = (inst_map > 0).astype(np.int64)
        hv_map = _gen_instance_hv_map(inst_map)

        sample = {
            "image": img_t,                                              # (3,H,W) float32
            "inst_map": torch.from_numpy(inst_map).long(),               # (H,W) int64
            "type_map": torch.from_numpy(type_map).long(),               # (H,W) int64
            "bin_map":  torch.from_numpy(bin_map).long(),                # (H,W) int64
            "hv_map":   torch.from_numpy(hv_map).float(),                # (2,H,W) float32
            "path_image": rec.get(self._name_key, f"pannuke_{idx:06d}"), # string
        }
        return sample
