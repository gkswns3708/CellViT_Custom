# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

# 기존 데이터셋 그대로 사용
from Data.CoNSeP_patch import ConsepPatchDataset


def build_label_mapping(scheme: str) -> np.ndarray:
    """
    반환: 크기 >= 8 의 룩업 테이블 (0..7 사용)
    scheme='original': identity
    scheme='consep_merged': 3&4->3 (epithelial), 5,6,7->4 (spindle)
    최종 라벨 세트:
      0: background
      1: other
      2: inflammatory
      3: epithelial (3,4)
      4: spindle (5,6,7)
    """
    scheme = scheme.lower()
    lut = np.arange(256, dtype=np.int64)
    if scheme == 'original':
        # 0..7 유지 (num_classes=8)
        return lut
    elif scheme == 'consep_merged':
        lut[0] = 0
        lut[1] = 1
        lut[2] = 2
        lut[3] = 3
        lut[4] = 3
        lut[5] = 4
        lut[6] = 4
        lut[7] = 4
        return lut
    else:
        raise ValueError(f"Unknown label scheme: {scheme}")


class ConsepPatchDatasetMerged(ConsepPatchDataset):
    """
    ConsepPatchDataset에 라벨 머지(post-map)를 얹은 버전.
    - type_map을 scheme에 맞춰 변환
    - bin_map도 변환된 type_map 기준으로 재계산
    """
    def __init__(
        self,
        root: str | Path,
        label_scheme: str = 'original',          # 'original' | 'consep_merged'
        use_meta: bool = True,
        normalize: bool = True,
    ) -> None:
        # num_nuclei_classes는 부모에서 쓰지 않으므로 넣지 않아도 됨
        super().__init__(root=root, num_nuclei_classes=8, use_meta=use_meta, normalize=normalize)
        self.label_scheme = label_scheme
        self._lut = build_label_mapping(label_scheme)

        # 출력 클래스 수 (배경 포함)
        if label_scheme == 'original':
            self.out_num_nuclei_classes = 8  # 0..7
        elif label_scheme == 'consep_merged':
            self.out_num_nuclei_classes = 5  # 0..4
        else:
            raise ValueError(f"Unknown label scheme: {label_scheme}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        # np -> remap -> torch
        t_type_np = sample['type_map'].numpy()
        # 라벨 값이 혹시 0..7 벗어나도 안전하게 클리핑
        t_type_np = np.clip(t_type_np, 0, 255)
        t_type_np = self._lut[t_type_np]  # vectorized

        # bin_map 재계산 (머지된 타입 기준)
        t_bin_np = (t_type_np > 0).astype(np.int64)

        sample['type_map'] = torch.from_numpy(t_type_np)
        sample['bin_map']  = torch.from_numpy(t_bin_np)
        # inst_map, hv_map, image 는 그대로
        return sample
