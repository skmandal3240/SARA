"""From-scratch transformer primitives: RMSNorm, RoPE, GQA, SwiGLU, blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SARAConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device=None, dtype=None):
    """Precompute cos/sin with paired frequencies (first half / second half)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, half)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (max_T, D) or (T, D)."""
    t = x.shape[2]
    cos = cos[start_pos : start_pos + t].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin[start_pos : start_pos + t].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GQAAttention(nn.Module):
    """Grouped-query self-attention with RoPE and optional QK-norm."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = config.qk_norm

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        cos, sin = build_rope_cache(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, self.rope_cos, self.rope_sin, start_pos)
        k = apply_rope(k, self.rope_cos, self.rope_sin, start_pos)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # causal unless an explicit mask is provided (then SDPA uses that mask)
        is_causal = mask is None and seqlen > 1 and kv_cache is None
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=self.scale
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(attn), new_cache


class CrossAttention(nn.Module):
    """Text queries attend to a modality memory (vision / audio tokens)."""

    def __init__(self, dim: int, context_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        ctx_len = context.shape[1]
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, config: SARAConfig, use_cross: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.rms_eps)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_eps)
        self.ffn = SwiGLU(config.dim, config.mlp_hidden)
        self.use_cross = use_cross
        if use_cross:
            self.cross_norm = RMSNorm(config.dim, config.rms_eps)
            self.cross = CrossAttention(config.dim, config.dim, max(1, config.n_heads // 2))
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache=None,
    ):
        h, kv_cache = self.attn(self.attn_norm(x), mask=mask, start_pos=start_pos, kv_cache=kv_cache)
        x = x + self.drop(h)
        if self.use_cross and context is not None:
            x = x + self.drop(self.cross(self.cross_norm(x), context))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x, kv_cache
