# -*- coding: utf-8 -*-
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 네 레포 구조에 맞춰 import (official utils를 그대로 사용)
from Model.utils import Conv2DBlock, Deconv2DBlock, ViTCellViT
from Model.post_utils.post_proc_cellvit import DetectionCellPostProcessor


# =========================
# 공식 구조: plain ViT + 멀티스케일 토큰 skip + 분기별 업샘플러
# =========================
class CellViTCustom(nn.Module):
    """
    Official-like CellViT:
      - Encoder: ViTCellViT (plain ViT, cls token + 1D abs pos), extract_layers=[L/4, L/2, 3L/4, L]
      - Shared skip transforms: decoder0..decoder3
      - Branch-specific upsampling heads: NP (binary), HV, NT (types)
      - Optional TC head is given by encoder's classifier head (if num_tissue_classes>0)

    Outputs (dict):
      - tissue_types: (B, num_tissue_classes) logits  [num_tissue_classes>0일 때]
      - nuclei_binary_map: (B, 2, H, W) logits
      - hv_map:            (B, 2, H, W) regression
      - nuclei_type_map:   (B, C_types, H, W) logits
    """
    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int = 0,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 384,    # ViT-256
        input_channels: int = 3,
        depth: int = 12,
        num_heads: int = 6,
        extract_layers: Optional[List[int]] = None,  # e.g. [3,6,9,12]
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        regression_loss: bool = False,  # (옵션) binary branch에 회귀 채널 2개 추가
    ):
        super().__init__()
        if extract_layers is None:
            extract_layers = [depth // 4, depth // 2, 3 * depth // 4, depth]  # [3,6,9,12] for depth=12
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.regression_loss = regression_loss

        # --- Encoder (official ViTCellViT) ---
        self.encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=self.num_tissue_classes,            # TC 분기 로짓
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
        )

        # --- skip dims / bottleneck dims (official heuristic) ---
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # --- Shared skip path transforms (decoder0..decoder3) ---
        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.input_channels, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # z0 (RGB) -> 64ch

        self.decoder1 = nn.Sequential(  # z1 (token fmap) -> 128ch
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(  # z2 -> 256ch
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(  # z3 -> bottleneck_dim
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # --- Branch heads' upsampling paths (decoder4 + headers) ---
        offset_branches = 2 if self.regression_loss else 0
        self.branches_out = {
            "nuclei_binary_map": 2 + offset_branches,      # (2 [+ 2 regression])
            "hv_map": 2,                                   # (u,v)
            "nuclei_type_map": self.num_nuclei_classes,    # (C_types)
        }

        self.nuclei_binary_map_decoder = self._create_branch(self.branches_out["nuclei_binary_map"])
        self.hv_map_decoder            = self._create_branch(self.branches_out["hv_map"])
        self.nuclei_type_maps_decoder  = self._create_branch(self.branches_out["nuclei_type_map"])  # 'maps' 복수 주의

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """
        x: (B,3,H,W), H,W divisible by patch_size
        returns dict with 3 maps (+ tissue_types if available)
        """
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0, \
            "Input size must be divisible by patch_size."

        out = {}

        # Encoder: logits (TC), cls_tokens, token_list
        cls_logits, _, tokens = self.encoder(x)   # tokens list @ extract layers
        if self.num_tissue_classes > 0:
            out["tissue_types"] = cls_logits

        # z0 = input image; z1..z4 = token tensors at layers
        z0 = x
        z1, z2, z3, z4 = tokens  # each: (B, N+1, D) where token 0 is [CLS]

        # reshape tokens -> (B, D, H/ps, W/ps)
        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size

        def tok2fmap(z: torch.Tensor) -> torch.Tensor:
            z = z[:, 1:, :].transpose(-1, -2)  # drop CLS -> (B, D, N)
            return z.view(-1, self.embed_dim, patch_h, patch_w)

        z1 = tok2fmap(z1)
        z2 = tok2fmap(z2)
        z3 = tok2fmap(z3)
        z4 = tok2fmap(z4)

        # per-branch upsampling
        if self.regression_loss:
            nb_full = self._forward_branch(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder) # type: ignore
            out["nuclei_binary_map"] = nb_full[:, :2, :, :]
            out["regression_map"]    = nb_full[:, 2:, :, :]
        else:
            out["nuclei_binary_map"] = self._forward_branch(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)

        out["hv_map"]          = self._forward_branch(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out["nuclei_type_map"] = self._forward_branch(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)

        if retrieve_tokens:
            out["tokens"] = z4
        return out

    # ---------- building blocks ----------
    def _forward_branch(
        self,
        z0: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4: torch.Tensor,
        branch: nn.Sequential,
    ) -> torch.Tensor:
        # bottleneck up from z4
        b4 = branch.bottleneck_upsampler(z4)       # -> bottleneck_dim, x2
        # skip z3
        b3 = self.decoder3(z3)
        b3 = branch.decoder3_upsampler(torch.cat([b3, b4], dim=1))  # -> 256, x2
        # skip z2
        b2 = self.decoder2(z2)
        b2 = branch.decoder2_upsampler(torch.cat([b2, b3], dim=1))  # -> 128, x2
        # skip z1
        b1 = self.decoder1(z1)
        b1 = branch.decoder1_upsampler(torch.cat([b1, b2], dim=1))  # -> 64, x2
        # skip z0 (RGB conv)
        b0 = self.decoder0(z0)                                      # 64
        y  = branch.decoder0_header(torch.cat([b0, b1], dim=1))     # head
        return y

    def _create_branch(self, num_out: int) -> nn.Module:
        # branch-specific upsampling path (공식 구조를 그대로)
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim, out_channels=self.bottleneck_dim,
            kernel_size=2, stride=2, padding=0, output_padding=0
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(64, num_out, kernel_size=1, stride=1, padding=0),
        )

        return nn.Sequential(OrderedDict([
            ("bottleneck_upsampler", bottleneck_upsampler),
            ("decoder3_upsampler", decoder3_upsampler),
            ("decoder2_upsampler", decoder2_upsampler),
            ("decoder1_upsampler", decoder1_upsampler),
            ("decoder0_header", decoder0_header),
        ]))

    # ---------- encoder weight load ----------
    def load_vit_checkpoint(self, ckpt_path: Union[str, Path], strict: bool = False):
        """
        다양한 포맷(전체 모델 ckpt, Lightning, DINO/HIPT 등)을 받아
        encoder(ViT) 파트만 추출·정규화해서 self.encoder에 로드.
        """
        blob = torch.load(str(ckpt_path), map_location="cpu")

        # 1) 루트에서 state_dict 고르기 (우선순위)
        sd = blob
        if isinstance(blob, dict):
            for k in ["model_state_dict", "state_dict", "teacher", "student", "model"]:
                if k in blob and isinstance(blob[k], dict):
                    sd = blob[k]
                    print(f"[CellViTCustom] load_vit_checkpoint: picked '{k}'")
                    break

        # 2) 만약 전체 모델의 state_dict라면 encoder.* 서브트리만 추출
        if isinstance(sd, dict) and any(k.startswith("encoder.") for k in sd.keys()):
            sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            print(f"[CellViTCustom] load_vit_checkpoint: kept encoder.* subset ({len(sd)} keys)")

        # 3) 흔한 prefix 제거 (DP/백본/라이트닝)
        sd = {k.replace("module.", "", 1).replace("backbone.", "", 1).replace("model.", "", 1): v
              for k, v in sd.items()}

        # 4) tissue head는 쓰지 않으면 제외
        if self.num_tissue_classes == 0:
            drop = [k for k in sd.keys() if k.startswith("head.")]
            for k in drop: sd.pop(k)
            if drop:
                print(f"[CellViTCustom] load_vit_checkpoint: dropped head.* ({len(drop)} keys, num_tissue_classes=0)")

        # (옵션) pos_embed shape 불일치 시 건너뛰고 로드
        # 필요할 때 주석 해제
        # if "pos_embed" in sd and sd["pos_embed"].shape != self.encoder.pos_embed.shape:
        #     print("[CellViTCustom] load_vit_checkpoint: skipping pos_embed due to shape mismatch")
        #     sd.pop("pos_embed")

        msg = self.encoder.load_state_dict(sd, strict=strict)
        try:
            mk, uk = msg.missing_keys, msg.unexpected_keys
            print(f"[CellViTCustom] load_vit_checkpoint: missing={len(mk)} unexpected={len(uk)}")
        except Exception:
            print(f"[CellViTCustom] load_vit_checkpoint: loaded (strict={strict})")
        return msg

    # ---------- optional: instance postproc ----------
    @torch.no_grad()
    def calculate_instance_map(self, predictions: Dict[str, torch.Tensor], magnification: Literal[20, 40] = 40
                               ) -> Tuple[torch.Tensor, List[dict]]:
        preds = {
            "nuclei_type_map": predictions["nuclei_type_map"].permute(0, 2, 3, 1),
            "nuclei_binary_map": predictions["nuclei_binary_map"].permute(0, 2, 3, 1),
            "hv_map": predictions["hv_map"].permute(0, 2, 3, 1),
        }
        cell_post = DetectionCellPostProcessor(nr_types=self.num_nuclei_classes, magnification=magnification, gt=False)
        inst_preds, type_preds = [], []
        for i in range(preds["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate([
                torch.argmax(preds["nuclei_type_map"], dim=-1)[i].detach().cpu()[..., None],
                torch.argmax(preds["nuclei_binary_map"], dim=-1)[i].detach().cpu()[..., None],
                preds["hv_map"][i].detach().cpu(),
            ], axis=-1)
            inst_pred = cell_post.post_process_cell_segmentation(pred_map)
            inst_preds.append(inst_pred[0]); type_preds.append(inst_pred[1])
        return torch.Tensor(np.stack(inst_preds)), type_preds

    def freeze_encoder(self):
        for name, p in self.encoder.named_parameters():
            if name.split(".")[0] != "head":  # head는 학습
                p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# =========================
# Loss functions (논문식 구성)
#   L = L_NP + L_HV + L_NT (+ L_TC if 제공)
#   - NP: Focal-Tversky + Dice (binary)
#   - HV: MSE + gradient-MSE
#   - NT: Focal-Tversky + Dice + CE (multi-class)
# =========================
def _one_hot_ignore(labels: torch.Tensor, num_classes: int, ignore_index: int = 255) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    labels: (B,H,W) long. ignore_index=255는 무시
    return: (onehot[B,C,H,W], valid_mask[B,1,H,W])
    """
    device = labels.device
    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    onehot = torch.zeros((labels.size(0), num_classes, labels.size(1), labels.size(2)), device=device, dtype=torch.float)
    onehot.scatter_(1, labels.clamp(0, num_classes - 1).unsqueeze(1), 1.0)
    onehot = onehot * valid.unsqueeze(1).float()
    return onehot, valid.unsqueeze(1).float()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    """
    Multiclass Dice loss (mean over classes), ignore_index 지원
    logits: (B,C,H,W), target: (B,H,W) long
    """
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    tgt_oh, valid = _one_hot_ignore(target, C, ignore_index)
    probs = probs * valid
    tgt_oh = tgt_oh * valid

    dims = (0, 2, 3)
    num = 2 * torch.sum(probs * tgt_oh, dim=dims)
    den = torch.sum(probs + tgt_oh, dim=dims) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()


def focal_tversky_from_logits(
    logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.7, beta: float = 0.3, gamma: float = 4/3, ignore_index: int = 255, eps: float = 1e-6
) -> torch.Tensor:
    """
    Multiclass Focal-Tversky loss(평균) - Salehi17
    """
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    tgt_oh, valid = _one_hot_ignore(target, C, ignore_index)
    probs = probs * valid
    tgt_oh = tgt_oh * valid

    dims = (0, 2, 3)
    TP = torch.sum(probs * tgt_oh, dim=dims)
    FP = torch.sum(probs * (1 - tgt_oh), dim=dims)
    FN = torch.sum((1 - probs) * tgt_oh, dim=dims)
    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    ft = torch.pow(1 - tversky, gamma)
    return ft.mean()


def binary_dice_from_logits(logits_2ch: torch.Tensor, target01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Binary Dice (positive class만) - logits_2ch: (B,2,H,W), target01: (B,H,W) {0,1}
    """
    probs = F.softmax(logits_2ch, dim=1)[:, 1]  # nucleus prob
    target = target01.float()
    num = 2 * torch.sum(probs * target)
    den = torch.sum(probs + target) + eps
    return 1 - (num + eps) / den


def binary_focal_tversky_from_logits(
    logits_2ch: torch.Tensor, target01: torch.Tensor,
    alpha: float = 0.7, beta: float = 0.3, gamma: float = 4/3, eps: float = 1e-6
) -> torch.Tensor:
    p = F.softmax(logits_2ch, dim=1)[:, 1]  # nucleus prob
    t = target01.float()
    TP = torch.sum(p * t)
    FP = torch.sum(p * (1 - t))
    FN = torch.sum((1 - p) * t)
    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return torch.pow(1 - tversky, gamma)


def hv_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pred/target: (B,2,H,W), valid: (B,1,H,W) mask(optional)
    """
    if valid is None:
        return F.mse_loss(pred, target)
    return F.mse_loss(pred * valid, target * valid, reduction="sum") / (valid.sum() * 2 + 1e-6)


def hv_gradient_mse(pred: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sobel gradient MSE on HV channels
    pred/target: (B,2,H,W), valid: (B,1,H,W)
    """
    device = pred.device
    kx = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=torch.float, device=device).view(1,1,3,3) / 8.0
    ky = kx.transpose(2,3)
    # apply per-channel (groups=2)
    wx = torch.cat([kx, kx], dim=0)  # (2,1,3,3)
    wy = torch.cat([ky, ky], dim=0)

    def grad2(x):
        gx = F.conv2d(x, wx, padding=1, groups=2)
        gy = F.conv2d(x, wy, padding=1, groups=2)
        return gx, gy

    pgx, pgy = grad2(pred)
    tgx, tgy = grad2(target)
    if valid is None:
        return F.mse_loss(pgx, tgx) + F.mse_loss(pgy, tgy)
    return (
        F.mse_loss(pgx * valid, tgx * valid, reduction="sum")
        + F.mse_loss(pgy * valid, tgy * valid, reduction="sum")
    ) / (valid.sum() * 2 + 1e-6)


def cellvit_losses(
    outputs: Dict[str, torch.Tensor],
    type_map: torch.Tensor,                  # (B,H,W) long, 0=bg, 1..C-1 nuclei
    hv_map: Optional[torch.Tensor] = None,   # (B,2,H,W) float
    bin_map: Optional[torch.Tensor] = None,  # (B,H,W) long {0,1}; 없으면 type_map으로부터 생성
    tissue_label: Optional[torch.Tensor] = None,  # (B,) long (옵션)
    # --- 가중치(논문 기본 구성; 값은 보수적으로 1.0으로 두고 필요 시 조정) ---
    w_np_ft: float = 1.0, w_np_dice: float = 1.0,
    w_nt_ft: float = 1.0, w_nt_dice: float = 1.0, w_nt_ce: float = 1.0,
    w_hv_mse: float = 1.0, w_hv_msge: float = 1.0,
    w_tc_ce: float = 0.0,                   # 기본 0.0 (데이터에 TC 라벨 없으니 꺼둠)
) -> Dict[str, torch.Tensor]:
    """
    논문식 총손실:
      L = (w_np_ft * FT_bin + w_np_dice * Dice_bin)
        + (w_hv_mse * MSE_hv + w_hv_msge * MSE_grad_hv)
        + (w_nt_ft * FT_multi + w_nt_dice * Dice_multi + w_nt_ce * CE_multi)
        + (w_tc_ce * CE_tissue) [옵션]
    """
    device = outputs["nuclei_type_map"].device
    # --- derive binary map if not provided
    if bin_map is None:
        bin_map = (type_map > 0).long()

    # ----- NP (binary) -----
    np_logits = outputs["nuclei_binary_map"]  # (B,2,H,W)
    loss_np_ft   = binary_focal_tversky_from_logits(np_logits, bin_map)
    loss_np_dice = binary_dice_from_logits(np_logits, bin_map)

    # ----- HV -----
    loss_hv_mse  = torch.tensor(0.0, device=device)
    loss_hv_msge = torch.tensor(0.0, device=device)
    if hv_map is not None and "hv_map" in outputs:
        valid = (bin_map > 0).unsqueeze(1).float()  # nucleus pixels만
        loss_hv_mse  = hv_mse_loss(outputs["hv_map"], hv_map, valid=valid)
        loss_hv_msge = hv_gradient_mse(outputs["hv_map"], hv_map, valid=valid)

    # ----- NT (types, multi-class) -----
    nt_logits = outputs["nuclei_type_map"]  # (B,C,H,W)
    C = nt_logits.shape[1]
    tt = type_map.clone()
    tt[(tt < 0) | (tt >= C)] = 255  # ignore invalid; bg(0)은 포함
    # Focal-Tversky & Dice는 멀티클래스 버전
    loss_nt_ft   = focal_tversky_from_logits(nt_logits, tt, ignore_index=255)
    loss_nt_dice = dice_loss_from_logits(nt_logits, tt, ignore_index=255)
    # CE (멀티클래스)
    loss_nt_ce   = F.cross_entropy(nt_logits, tt.long(), ignore_index=255)

    # ----- TC (optional, if provided) -----
    loss_tc_ce = torch.tensor(0.0, device=device)
    if w_tc_ce > 0 and "tissue_types" in outputs and tissue_label is not None:
        loss_tc_ce = F.cross_entropy(outputs["tissue_types"], tissue_label.long())

    # total
    loss = (
        w_np_ft * loss_np_ft + w_np_dice * loss_np_dice
        + w_hv_mse * loss_hv_mse + w_hv_msge * loss_hv_msge
        + w_nt_ft * loss_nt_ft + w_nt_dice * loss_nt_dice + w_nt_ce * loss_nt_ce
        + w_tc_ce * loss_tc_ce
    )

    return {
        "loss": loss,
        "loss_np_ft": loss_np_ft.detach(),
        "loss_np_dice": loss_np_dice.detach(),
        "loss_hv_mse": loss_hv_mse.detach(),
        "loss_hv_msge": loss_hv_msge.detach(),
        "loss_nt_ft": loss_nt_ft.detach(),
        "loss_nt_dice": loss_nt_dice.detach(),
        "loss_nt_ce": loss_nt_ce.detach(),
        "loss_tc_ce": loss_tc_ce.detach(),
    }
