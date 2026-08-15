"""Vision encoder (see) and image decoder (create). Tiny conv + transformer."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import SARAConfig
from .modules import RMSNorm, TransformerBlock


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, dim: int, in_ch: int = 3):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [-1, 1]
        x = self.proj(x)  # (B, dim, gh, gw)
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)  # (B, N, dim)


class VisionEncoder(nn.Module):
    """Image → visual tokens in transformer width."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.config = config
        self.embed = PatchEmbed(config.img_size, config.patch_size, config.dim)
        n = config.n_patches
        self.pos = nn.Parameter(torch.randn(1, n, config.dim) * 0.02)
        # reuse TransformerBlock but disable cross-attn; causal would be wrong for patches
        self.blocks = nn.ModuleList(
            [VisionBlock(config) for _ in range(config.vision_layers)]
        )
        self.norm = RMSNorm(config.dim, config.rms_eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x, _ = self.embed(images)
        n = x.shape[1]
        x = x + self.pos[:, :n]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class VisionBlock(nn.Module):
    """Non-causal transformer block for patches (bidirectional)."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        from .modules import GQAAttention, SwiGLU

        self.n1 = RMSNorm(config.dim, config.rms_eps)
        self.attn = GQAAttention(config)
        self.n2 = RMSNorm(config.dim, config.rms_eps)
        self.ffn = SwiGLU(config.dim, config.mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force non-causal: pass a dummy all-ones mask so GQAAttention won't set is_causal
        b, t, _ = x.shape
        mask = torch.zeros(b, 1, t, t, device=x.device, dtype=x.dtype)
        h, _ = self.attn(self.n1(x), mask=mask)
        x = x + h
        x = x + self.ffn(self.n2(x))
        return x


class ImageDecoder(nn.Module):
    """Conditioned conv-transpose decoder: pooled hidden → RGB image."""

    def __init__(self, config: SARAConfig, out_size: Optional[int] = None):
        super().__init__()
        self.size = out_size or config.img_size
        dim = config.dim
        self.fc = nn.Linear(dim, 128 * 4 * 4)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32
            nn.GELU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 64
            nn.Tanh(),
        )
        self.resize = self.size != 64

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, dim)
        x = self.fc(cond).view(cond.shape[0], 128, 4, 4)
        x = self.up(x)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return x


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().clamp(-1, 1)
    arr = ((t.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_shape_image(kind: str, color: str, size: int = 64) -> Image.Image:
    """Synthetic labeled shapes for SEE / CREATE training."""
    from PIL import ImageDraw

    palette = {
        "red": (220, 40, 40),
        "green": (40, 180, 70),
        "blue": (40, 90, 220),
        "yellow": (240, 200, 40),
        "white": (240, 240, 240),
        "purple": (160, 60, 200),
    }
    bg = (18, 18, 24)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    c = palette.get(color, (200, 200, 200))
    m = size // 8
    if kind == "circle":
        d.ellipse([m, m, size - m, size - m], fill=c)
    elif kind == "square":
        d.rectangle([m, m, size - m, size - m], fill=c)
    elif kind == "triangle":
        d.polygon([(size // 2, m), (size - m, size - m), (m, size - m)], fill=c)
    else:  # stripe
        d.rectangle([0, size // 3, size, 2 * size // 3], fill=c)
    return img
