
#!/usr/bin/env python3
"""
Preprocessing for CellViT/SAM training (CoNSeP-friendly, .mat labels).

- Inputs per case:
  * 1000x1000 RGB PNG image
  * MATLAB .mat label file with keys:
      - 'inst_map': (H, W) integer instance map (0 = background)
      - 'type_map': (H, W) integer class map (0 = background)
    (Both classic MAT and v7.3 HDF5 are supported.)

- Operations:
  1) Pad each 1000x1000 sample to 1024x1024 (constant=0 for labels & image by default)
  2) Split into 4x4 non-overlapping patches of size 256x256
  3) For each patch, reindex instance ids to 1..N (patch-local)
  4) Generate HV maps (2, 256, 256) from the instance map (centroid-offset style)
  5) Optionally generate binary nuclei map from inst_map
  6) Save each patch:
     - Image: PNG
     - Labels: NPZ with keys: 'inst_map', 'type_map', 'hv_map' and optional 'bin_map'

Usage example:

python preprocess_cellvit_patches.py \
  --images_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Train/Images \
  --labels_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Original/Train/Labels \
  --out_dir /workspace/CellViT_Custom/Dataset/CoNSeP/Preprocessed/Train \
  --image_suffix .png \
  --label_suffix .mat \
  --save_binary



"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional deps for .mat support
try:
    import scipy.io as sio
except Exception:  # pragma: no cover
    sio = None
try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

# ------------------------------
# Core helpers
# ------------------------------

def pad_to_size(img: np.ndarray, target_hw: Tuple[int, int], mode: str = "constant", constant_values: int | Tuple[int, int] = 0) -> np.ndarray:
    """Pad HxWx[C] array to target size (H, W) by padding bottom/right sides."""
    th, tw = target_hw
    assert img.ndim in (2, 3)
    h, w = img.shape[:2]
    if h == th and w == tw:
        return img
    assert h <= th and w <= tw, f"Input larger than target: {(h,w)} > {(th,tw)}"
    pad_h, pad_w = th - h, tw - w
    # Cast mode to Any to appease numpy stubs that expect a callable protocol for mode
    if mode == "constant":
        if img.ndim == 2:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode=cast(Any, mode), constant_values=constant_values)
        else:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=cast(Any, mode), constant_values=constant_values)
    else:
        if img.ndim == 2:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode=cast(Any, mode))
        else:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=cast(Any, mode))


def split_grid(arr: np.ndarray, patch: int = 256) -> List[np.ndarray]:
    """Split 2D/3D array (1024x1024[,C]) into 16 tiles of 256x256 in row-major order."""
    H, W = arr.shape[:2]
    assert H % patch == 0 and W % patch == 0, f"Array size must be multiple of patch: {(H,W)}"
    tiles: List[np.ndarray] = []
    for r in range(H // patch):
        for c in range(W // patch):
            y0, x0 = r * patch, c * patch
            tiles.append(arr[y0 : y0 + patch, x0 : x0 + patch, ...])
    return tiles


def reindex_instances(inst_map: np.ndarray) -> np.ndarray:
    """Reindex instance ids in a (H, W) map to 1..N while keeping background at 0."""
    inst = inst_map.astype(np.int64, copy=False)
    uniq = np.unique(inst)
    uniq = uniq[uniq != 0]
    if uniq.size == 0:
        return inst.copy()
    mapping = {int(k): i + 1 for i, k in enumerate(sorted(uniq))}
    out = np.zeros_like(inst)
    for k, v in mapping.items():
        out[inst == k] = v
    return out


def hv_from_instance_map(inst_map: np.ndarray, target_range: str = "neg1to1") -> np.ndarray:
    """Compute HV map (2, H, W) from an instance map (H, W)."""
    H, W = inst_map.shape
    hv = np.zeros((2, H, W), dtype=np.float32)
    ids = np.unique(inst_map)
    ids = ids[ids != 0]
    if ids.size == 0:
        return hv

    yy, xx = np.mgrid[0:H, 0:W]

    for k in ids:
        mask = inst_map == k
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        cy = ys.mean()
        cx = xs.mean()
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        hx = max(1.0, (x_max - x_min + 1) / 2.0)
        hy = max(1.0, (y_max - y_min + 1) / 2.0)
        hv_x = (xx[mask] - cx) / hx
        hv_y = (yy[mask] - cy) / hy
        hv_x = np.clip(hv_x, -1.0, 1.0)
        hv_y = np.clip(hv_y, -1.0, 1.0)
        hv[0][mask] = hv_x
        hv[1][mask] = hv_y

    if target_range == "0to1":
        hv = (hv + 1.0) * 0.5
        hv = np.clip(hv, 0.0, 1.0)
    return hv


# ------------------------------
# .mat label loading (classic + v7.3)
# ------------------------------

def _maybe_squeeze(arr: np.ndarray) -> np.ndarray:
    a = np.array(arr)
    while a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a)
    return a


def load_labels_mat(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (inst_map, type_map) from a .mat file. Supports classic MAT via scipy and v7.3 via h5py.
    Returns int arrays shaped (H, W).
    """
    # Try scipy first (classic MAT)
    if sio is not None:
        try:
            mdict = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            # direct keys
            inst_map = mdict.get("inst_map", None)
            type_map = mdict.get("type_map", None)
            # Sometimes nested in a struct like mdict['label']
            if inst_map is None or type_map is None:
                for k, v in mdict.items():
                    if k.startswith("__"):
                        continue
                    try:
                        if hasattr(v, "__dict__"):
                            inst_map = getattr(v, "inst_map", inst_map)
                            type_map = getattr(v, "type_map", type_map)
                    except Exception:
                        pass
            if inst_map is not None and type_map is not None:
                inst = _maybe_squeeze(inst_map).astype(np.int32)
                typ = _maybe_squeeze(type_map).astype(np.int16)
                return inst, typ
        except NotImplementedError:
            # v7.3 goes here
            pass
        except Exception:
            # fallthrough to h5py
            pass

    # v7.3 HDF5 via h5py
    if h5py is None:
        raise RuntimeError("h5py is required to read v7.3 .mat files. Install h5py.")
    with h5py.File(mat_path, 'r') as f:
        # datasets may be stored in column-major; transpose to (H, W)
        def read_key(key):
            if key in f:
                arr = np.array(f[key])
                if arr.ndim >= 2:
                    arr = arr.T  # MATLAB col-major
                return np.squeeze(arr)
            return None
        inst = read_key('inst_map')
        typ  = read_key('type_map')
        if inst is None or typ is None:
            # try nested group like '/label/inst_map'
            for g in f.keys():
                try:
                    inst = inst or read_key(f"{g}/inst_map")
                    typ  = typ  or read_key(f"{g}/type_map")
                except Exception:
                    continue
        if inst is None or typ is None:
            raise KeyError(f"Could not find 'inst_map' and 'type_map' in {mat_path}")
        return inst.astype(np.int32), typ.astype(np.int16)


# ------------------------------
# Main IO helpers
# ------------------------------

@dataclass
class Paths:
    images_dir: Path
    labels_dir: Path
    out_dir: Path


def save_patch(
    case_id: str,
    r: int,
    c: int,
    img_patch: np.ndarray,
    inst_patch: np.ndarray,
    type_patch: np.ndarray,
    hv_patch: np.ndarray,
    out_images: Path,
    out_labels: Path,
    save_binary: bool,
):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stem = f"{case_id}_r{r}_c{c}"
    Image.fromarray(img_patch).save(out_images / f"{stem}.png")
    save_dict = {
        "inst_map": inst_patch.astype(np.int32),
        "type_map": type_patch.astype(np.int16),
        "hv_map": hv_patch.astype(np.float32),  # (2, 256, 256)
    }
    if save_binary:
        save_dict["bin_map"] = (inst_patch > 0).astype(np.uint8)
    np.savez_compressed(out_labels / f"{stem}.npz", **save_dict)


def load_labels_any(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ext = label_path.suffix.lower()
    if ext == '.mat':
        return load_labels_mat(label_path)
    elif ext == '.npz':
        nz = np.load(label_path, allow_pickle=True)
        if 'inst_map' in nz and 'type_map' in nz:
            return nz['inst_map'], nz['type_map']
        # fallback: single object array
        label_dict = nz[list(nz.keys())[0]].item()
        return label_dict['inst_map'], label_dict['type_map']
    elif ext == '.npy':
        obj = np.load(label_path, allow_pickle=True)
        label_dict = obj.item()
        return label_dict['inst_map'], label_dict['type_map']
    else:
        raise ValueError(f"Unsupported label extension: {ext}")


def process_case(
    img_path: Path,
    label_path: Path,
    paths: Paths,
    pad_mode_img: str = "constant",
    pad_mode_lbl: str = "constant",
    hv_target_range: str = "neg1to1",
    save_binary: bool = False,
) -> List[Dict[str, str]]:
    """Process a single case and return rows for the meta table."""
    # load image
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # load labels (.mat/.npz/.npy)
    inst_map, type_map = load_labels_any(label_path)

    assert img_np.shape[:2] == inst_map.shape == type_map.shape, (
        f"Shape mismatch: img {img_np.shape[:2]} vs inst {inst_map.shape} vs type {type_map.shape}"
    )

    # Pad to 1024x1024
    target_hw = (1024, 1024)
    img_np = pad_to_size(img_np, target_hw, mode=pad_mode_img, constant_values=0)
    inst_map = pad_to_size(inst_map, target_hw, mode=pad_mode_lbl, constant_values=0)
    type_map = pad_to_size(type_map, target_hw, mode=pad_mode_lbl, constant_values=0)

    # Split into 16 patches
    img_tiles = split_grid(img_np, 256)
    inst_tiles = split_grid(inst_map, 256)
    type_tiles = split_grid(type_map, 256)

    rows: List[Dict[str, str]] = []
    case_id = img_path.stem

    out_images = paths.out_dir / "images"
    out_labels = paths.out_dir / "labels"

    for r in range(4):
        for c in range(4):
            tile_i = r * 4 + c
            img_patch = img_tiles[tile_i]
            inst_patch = reindex_instances(inst_tiles[tile_i])
            type_patch = type_tiles[tile_i]
            hv_patch = hv_from_instance_map(inst_patch, target_range=hv_target_range)

            save_patch(
                case_id, r, c, img_patch, inst_patch, type_patch, hv_patch,
                out_images, out_labels, save_binary
            )

            rows.append({
                "case_id": case_id,
                "row": str(r),
                "col": str(c),
                "image_path": str((out_images / f"{case_id}_r{r}_c{c}.png").resolve()),
                "label_path": str((out_labels / f"{case_id}_r{r}_c{c}.npz").resolve()),
                "orig_image": str(img_path.resolve()),
                "orig_label": str(label_path.resolve()),
            })

    return rows


def find_pairs(images_dir: Path, labels_dir: Path, image_suffix: str, label_suffix: str) -> List[Tuple[Path, Path]]:
    images = sorted(images_dir.glob(f"*{image_suffix}"))
    pairs: List[Tuple[Path, Path]] = []
    for img_path in images:
        base = img_path.stem
        lbl = labels_dir / f"{base}{label_suffix}"
        if not lbl.exists():
            # try alternate suffixes
            for alt in ('.mat', '.npz', '.npy'):
                cand = labels_dir / f"{base}{alt}"
                if cand.exists():
                    lbl = cand
                    break
            else:
                print(f"[WARN] Label not found for {img_path.name}")
                continue
        pairs.append((img_path, lbl))
    return pairs


# ------------------------------
# CLI
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="Pad 1000x1000 → 1024x1024 and split into 16×256 patches with HV maps (.mat labels).")
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--labels_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--image_suffix", type=str, default=".png")
    p.add_argument("--label_suffix", type=str, default=".mat", help=".mat/.npz/.npy; if not found, tries alternates automatically")
    p.add_argument("--pad_mode_img", type=str, default="constant", choices=["constant", "edge", "reflect"])
    p.add_argument("--pad_mode_lbl", type=str, default="constant", choices=["constant", "edge", "reflect"], help="Usually 'constant' is safest for labels")
    p.add_argument("--hv_target_range", type=str, default="neg1to1", choices=["neg1to1", "0to1"], help="Output range for HV maps")
    p.add_argument("--save_binary", action="store_true", help="Also save a binary nuclei map (inst>0)")
    p.add_argument("--meta_name", type=str, default="meta.csv")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = Paths(images_dir=args.images_dir, labels_dir=args.labels_dir, out_dir=args.out_dir)
    pairs = find_pairs(args.images_dir, args.labels_dir, args.image_suffix, args.label_suffix)
    if not pairs:
        print("No image/label pairs found. Check your directories and suffixes.")
        return

    meta_rows: List[Dict[str, str]] = []
    for img_path, lbl_path in tqdm(pairs):
        rows = process_case(
            img_path,
            lbl_path,
            paths,
            pad_mode_img=args.pad_mode_img,
            pad_mode_lbl=args.pad_mode_lbl,
            hv_target_range=args.hv_target_range,
            save_binary=args.save_binary,
        )
        meta_rows.extend(rows)

    meta_path = args.out_dir / args.meta_name
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        for r in meta_rows:
            writer.writerow(r)
    print(f"Saved {len(meta_rows)} patches to {args.out_dir}")


if __name__ == "__main__":
    main()
