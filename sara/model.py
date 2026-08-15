"""SARA unified multimodal transformer + generation heads + tool head."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SARAConfig
from .modules import RMSNorm, TransformerBlock
from .vision import VisionEncoder, ImageDecoder
from .audio import AudioEncoder, MelDecoder
from .video import VideoDecoder
from .music import SongHead


class ToolHead(nn.Module):
    """Classify which registered tool to call from a hidden state (constrained channel)."""

    def __init__(self, dim: int, max_tools: int = 32):
        super().__init__()
        self.max_tools = max_tools
        self.cls = nn.Linear(dim, max_tools)
        self.stop = nn.Linear(dim, 2)  # continue vs final-answer

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"tool": self.cls(h), "stop": self.stop(h)}


class NumberHead(nn.Module):
    """xVal-style scalar: pooled hidden → is-number gate and a value.

    Decode as the [NUM] token times `value`. Tiny Linear stub — not a numeric LM.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.value = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        g = self.gate(h).squeeze(-1)
        v = self.value(h).squeeze(-1)
        return {"is_num": g, "value": v, "num": torch.sigmoid(g) * v}


class SARA(nn.Module):
    """See / Articulate / Reason / Author — one transformer, many heads."""

    def __init__(self, config: SARAConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        # modality type: text, vision, audio, code, video, song, tool
        self.type_emb = nn.Embedding(8, config.dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, use_cross=(i % 2 == 1))
                for i in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, config.rms_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.vision = VisionEncoder(config)
        self.image_dec = ImageDecoder(config)
        self.audio_enc = AudioEncoder(config)
        self.mel_dec = MelDecoder(config)
        self.video_dec = VideoDecoder(config)
        self.song_head = SongHead(config)
        self.tool_head = ToolHead(config.dim, config.max_tools)
        self.number_head = NumberHead(config.dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def pool(self, h: torch.Tensor, pad_id: Optional[int] = None, tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-pool hidden states, optionally ignoring pads."""
        if tokens is None or pad_id is None:
            return h.mean(dim=1)
        mask = (tokens != pad_id).float().unsqueeze(-1)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches=None,
    ) -> dict:
        bsz, seqlen = tokens.shape
        if type_ids is None:
            type_ids = torch.zeros_like(tokens)
        h = self.tok_emb(tokens) + self.type_emb(type_ids)

        # Concatenate encoded modalities as extra memory for cross-attn
        memories = []
        if images is not None:
            memories.append(self.vision(images))
        if mel is not None:
            memories.append(self.audio_enc(mel))
        if memories:
            context = torch.cat(memories, dim=1) if context is None else torch.cat([context, *memories], dim=1)

        new_caches = []
        for i, blk in enumerate(self.blocks):
            cache_i = None if kv_caches is None else kv_caches[i]
            h, cache_i = blk(h, context=context, start_pos=start_pos, kv_cache=cache_i)
            new_caches.append(cache_i)
        h = self.norm(h)
        logits = self.lm_head(h)

        out = {"logits": logits, "hidden": h, "kv_caches": new_caches, "context": context}
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=self.config.pad_id,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = None,
        images=None,
        mel=None,
        type_ids: Optional[torch.Tensor] = None,
        stop_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        self.eval()
        eos_id = self.config.eos_id if eos_id is None else eos_id
        stop = set(stop_ids or [])
        stop.add(eos_id)
        max_len = int(self.config.max_seq_len)
        if tokens.numel() == 0 or tokens.shape[1] == 0:
            tokens = torch.full((1, 1), self.config.bos_id, dtype=torch.long, device=tokens.device)
            type_ids = None
        if tokens.shape[1] >= max_len:
            tokens = tokens[:, -(max_len - 1) :]
            if type_ids is not None:
                type_ids = type_ids[:, -tokens.shape[1] :]
        room = max(1, max_len - tokens.shape[1])
        max_new = min(max_new, room)
        kv = None
        # prefills
        out = self.forward(tokens, type_ids=type_ids, images=images, mel=mel, kv_caches=None, start_pos=0)
        kv = out["kv_caches"]
        context = out["context"]
        generated = tokens
        cur_type = type_ids
        pos = tokens.shape[1]
        for _ in range(max_new):
            last_logits = out["logits"][:, -1, :]
            if temperature <= 0:
                nxt = last_logits.argmax(dim=-1, keepdim=True)
            else:
                logits = last_logits / max(temperature, 1e-5)
                if top_k and top_k < logits.size(-1):
                    v, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, nxt], dim=1)
            if cur_type is not None:
                z = torch.zeros_like(nxt)
                cur_type = torch.cat([cur_type, z], dim=1)
            if int(nxt[0, 0]) in stop:
                break
            if pos >= max_len - 1:
                break
            out = self.forward(
                nxt,
                type_ids=None if cur_type is None else cur_type[:, -1:],
                context=context,
                start_pos=pos,
                kv_caches=kv,
            )
            kv = out["kv_caches"]
            pos += 1
        return generated

    def cond_from_tokens(self, tokens: torch.Tensor, images=None, mel=None) -> torch.Tensor:
        out = self.forward(tokens, images=images, mel=mel)
        return self.pool(out["hidden"], pad_id=self.config.pad_id, tokens=tokens)

    def generate_image(self, tokens: torch.Tensor) -> torch.Tensor:
        cond = self.cond_from_tokens(tokens)
        return self.image_dec(cond)

    def generate_video(self, tokens: torch.Tensor) -> torch.Tensor:
        cond = self.cond_from_tokens(tokens)
        return self.video_dec(cond)

    def generate_mel(self, tokens: torch.Tensor) -> torch.Tensor:
        cond = self.cond_from_tokens(tokens)
        return self.mel_dec(cond)

    def generate_song(self, tokens: torch.Tensor) -> dict:
        cond = self.cond_from_tokens(tokens)
        return self.song_head(cond)

    def tool_decision(self, tokens: torch.Tensor) -> dict:
        cond = self.cond_from_tokens(tokens)
        return self.tool_head(cond)

    def read_number(self, tokens: torch.Tensor) -> dict:
        cond = self.cond_from_tokens(tokens)
        return self.number_head(cond)


def causal_mask(t: int, device, dtype) -> torch.Tensor:
    m = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1).unsqueeze(0).unsqueeze(0)
