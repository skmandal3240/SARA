"""Short video / GIF generation: time-conditioned frame decoder."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import SARAConfig
from .vision import ImageDecoder, tensor_to_pil


class VideoDecoder(nn.Module):
    """cond + frame index → RGB frames. Shares the same upsampling recipe as images."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.n_frames = config.n_video_frames
        self.size = config.video_size
        self.frame_emb = nn.Embedding(config.n_video_frames, config.dim)
        self.mix = nn.Linear(config.dim * 2, config.dim)
        self.dec = ImageDecoder(config, out_size=config.video_size)

    def forward(self, cond: torch.Tensor, n_frames: int | None = None) -> torch.Tensor:
        n_frames = n_frames or self.n_frames
        b = cond.shape[0]
        frames = []
        for t in range(n_frames):
            idx = torch.full((b,), t, device=cond.device, dtype=torch.long)
            h = torch.cat([cond, self.frame_emb(idx)], dim=-1)
            h = self.mix(h)
            frames.append(self.dec(h))
        return torch.stack(frames, dim=1)  # (B, T, 3, H, W)


def frames_to_gif(frames: torch.Tensor, path: str | Path, duration_ms: int = 120) -> None:
    """frames: (T, 3, H, W) or (B, T, 3, H, W)."""
    if frames.dim() == 5:
        frames = frames[0]
    imgs = [tensor_to_pil(frames[t]).convert("P", palette=Image.Palette.ADAPTIVE) for t in range(frames.shape[0])]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
