# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the official ViT encoder you provided (lives in Model/VIT)
from Model.VIT.SAM.image_encoder import ImageEncoderViT


# ----------------------------
# Small conv/upsample building blocks
# ----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBNAct(out_ch, out_ch)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# ----------------------------
# CellViTCustom: ViT encoder → patch-grid decoding → multi-branch heads
# ----------------------------
class CellViTCustom(nn.Module):
    """Cell segmentation with ViT encoder and UNETR-like upsampling (no skip taps).

    Outputs:
      - nuclei_binary_map: (B, 2, H, W) logits
      - nuclei_type_map:   (B, C_types, H, W) logits
      - hv_map:            (B, 2, H, W) regression (offsets)
    """
    def __init__(
        self,
        num_nuclei_classes: int,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_out_chans: int = 256,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: tuple = (),
    ):
        super().__init__()
        self.num_nuclei_classes = num_nuclei_classes
        self.img_size = img_size
        self.patch = patch_size

        # ViT encoder from official code (returns feature map after neck): (B, C=out_chans, H/16, W/16)
        self.encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            out_chans=vit_out_chans,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=True,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

        C = vit_out_chans  # encoder output channels (default 256)
        # Decoder: 16→32→64→128→256 with progressive channel reduction
        self.dec3 = UpBlock(C, 256)   # 16→32
        self.dec2 = UpBlock(256, 128) # 32→64
        self.dec1 = UpBlock(128, 64)  # 64→128
        self.dec0 = UpBlock(64, 64)   # 128→256

        self.dec_final = ConvBNAct(64, 64)

        # Heads
        self.head_binary = nn.Conv2d(64, 2, kernel_size=1)
        self.head_types  = nn.Conv2d(64, num_nuclei_classes, kernel_size=1)
        self.head_hv     = nn.Conv2d(64, 2, kernel_size=1)

    @torch.no_grad()
    def _resize_abs_pos_(self, state_dict: Dict[str, torch.Tensor]):
        """If loading a checkpoint trained at a different img_size, interpolate pos_embed."""
        if 'pos_embed' not in state_dict or getattr(self.encoder, 'pos_embed', None) is None:
            return
        pe_ckpt = state_dict['pos_embed']  # (1, Hc, Wc, C)
        pe_curr = self.encoder.pos_embed
        if pe_ckpt.shape == pe_curr.shape:
            return  # no change
        # interpolate spatially (H, W)
        Bc, Hc, Wc, C = pe_ckpt.shape
        Bn, Hn, Wn, Cn = pe_curr.shape
        pe_ckpt_nchw = pe_ckpt.permute(0, 3, 1, 2)  # (1, C, H, W)
        pe_resized = F.interpolate(pe_ckpt_nchw, size=(Hn, Wn), mode='bicubic', align_corners=False)
        state_dict['pos_embed'] = pe_resized.permute(0, 2, 3, 1)

    def load_vit_checkpoint(self, ckpt_path: str, strict: bool = False) -> None:
        sd = torch.load(ckpt_path, map_location='cpu')
        # support either pure sd or wrapped
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        self._resize_abs_pos_(sd)
        missing, unexpected = self.encoder.load_state_dict(sd, strict=strict)
        print(f"[CellViTCustom] load_vit_checkpoint: missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, 3, 256, 256)
        f = self.encoder(x)    # (B, C=256, 16, 16)
        y = self.dec3(f)       # (B, 256, 32, 32)
        y = self.dec2(y)       # (B, 128, 64, 64)
        y = self.dec1(y)       # (B, 64, 128, 128)
        y = self.dec0(y)       # (B, 64, 256, 256)
        y = self.dec_final(y)  # (B, 64, 256, 256)

        out = {
            'nuclei_binary_map': self.head_binary(y),   # logits
            'nuclei_type_map':   self.head_types(y),    # logits
            'hv_map':            self.head_hv(y),       # regression
        }
        return out


# ----------------------------
# Loss wrapper (type-map–aware)
# ----------------------------
def cellvit_losses(
    outputs: Dict[str, torch.Tensor],
    type_map: torch.Tensor,  # (B, H, W) long (0=bg, 1..C-1 types)
    hv_map: Optional[torch.Tensor] = None,  # (B, 2, H, W) float
    bin_map: Optional[torch.Tensor] = None, # (B, H, W) long {0,1}
    w_bin: float = 1.0,
    w_type: float = 1.0,
    w_hv: float = 1.0,
    pos_weight: Optional[float] = None,  # for binary CE (class imbalance)
):
    device = outputs['nuclei_binary_map'].device
    # derive binary from type_map if not provided
    if bin_map is None:
        bin_map = (type_map > 0).long()

    # Binary CE on all pixels (optionally class-weighted)
    bin_logits = outputs['nuclei_binary_map']  # (B,2,H,W)
    if pos_weight is not None:
        # convert to per-class weights for CE: weight[c]
        # weight for background=1.0, for nucleus=pos_weight
        weight = torch.tensor([1.0, float(pos_weight)], device=device)
        loss_bin = F.cross_entropy(bin_logits, bin_map, weight=weight)
    else:
        loss_bin = F.cross_entropy(bin_logits, bin_map)

    # Type CE but only on nucleus pixels (mask type>0)
    type_logits = outputs['nuclei_type_map']  # (B,C,H,W)
    mask = (type_map > 0)
    if mask.any():
        # gather masked positions
        tl = type_logits.permute(0,2,3,1)[mask]  # (Nmask, C)
        tt = type_map[mask]                       # (Nmask,)
        loss_type = F.cross_entropy(tl, tt)
    else:
        loss_type = torch.tensor(0.0, device=device)

    # HV SmoothL1 (L1Huber) only on nucleus pixels if hv provided
    loss_hv = torch.tensor(0.0, device=device)
    if hv_map is not None and 'hv_map' in outputs:
        hv_pred = outputs['hv_map']  # (B,2,H,W)
        if mask.any():
            # mask per-channel
            mask_f = mask.unsqueeze(1).float()
            l1 = F.smooth_l1_loss(hv_pred*mask_f, hv_map*mask_f, reduction='sum')
            loss_hv = l1 / (mask_f.sum() + 1e-6)
        else:
            loss_hv = F.smooth_l1_loss(hv_pred, hv_map)

    loss = w_bin*loss_bin + w_type*loss_type + w_hv*loss_hv
    return {
        'loss': loss,
        'loss_bin': loss_bin.detach(),
        'loss_type': loss_type.detach(),
        'loss_hv': loss_hv.detach(),
    }

