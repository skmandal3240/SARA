"""SARA model and training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class SARAConfig:
    """Nano-first config. Larger presets exist as factories, not trained here."""

    # Transformer
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA
    mlp_mult: float = 2.5
    max_seq_len: int = 384
    vocab_size: int = 4096
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = True  # RMSNorm on Q/K before attention

    # Special ids (filled from tokenizer after load)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    # Vision
    img_size: int = 64
    patch_size: int = 8
    vision_layers: int = 2
    vision_heads: int = 4
    n_img_tokens: int = 16  # 4x4 after pool, or 8x8 raw

    # Audio / speech / song
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    audio_frames: int = 80
    audio_layers: int = 2

    # Video
    n_video_frames: int = 8
    video_size: int = 48

    # Song score
    n_pitches: int = 25  # rest + 2 octaves of 12
    n_note_steps: int = 32
    n_keys: int = 12

    # Tools / agents
    max_tools: int = 32
    max_agent_steps: int = 8
    max_tool_calls: int = 12

    # Train
    dtype: str = "fp32"

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0
        return self.dim // self.n_heads

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def mlp_hidden(self) -> int:
        h = int(self.dim * self.mlp_mult)
        return max(64, (h + 63) // 64 * 64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SARAConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def nano(cls) -> "SARAConfig":
        return cls()

    @classmethod
    def small(cls) -> "SARAConfig":
        """~45M params — demo/showcase scale. Still CPU-trainable for a few hundred steps;
        real training needs a GPU (Colab T4 / IndiaAI)."""
        return cls(
            dim=384,
            n_layers=16,
            n_heads=8,
            n_kv_heads=4,
            mlp_mult=2.5,
            max_seq_len=512,
            vocab_size=4096,
            vision_layers=4,
            vision_heads=8,
            audio_layers=3,
            img_size=96,
            patch_size=8,  # 12x12 = 144 patches
            video_size=64,
            n_video_frames=8,
            audio_frames=80,
            n_note_steps=32,
        )

    @classmethod
    def large(cls) -> "SARAConfig":
        """~503M params — Kaggle-scale training on public datasets. Does NOT fit
        T4 x2 for full training; needs P100 (32GB) with bf16 + small batch, or
        multi-GPU / IndiaAI GPUs. Inference fits a single 16GB GPU in int8."""
        return cls(
            dim=1024,
            n_layers=28,
            n_heads=16,
            n_kv_heads=4,
            mlp_mult=3.0,
            max_seq_len=2048,
            vocab_size=32768,
            vision_layers=8,
            vision_heads=16,
            audio_layers=6,
            img_size=128,
            patch_size=8,
            video_size=96,
            n_video_frames=8,
            audio_frames=80,
            n_note_steps=32,
        )

    @classmethod
    def medium(cls) -> "SARAConfig":
        """~128M params — for ~4 GB of text (~1B tokens). Needs Kaggle T4 x2 / P100
        (bf16, grad checkpointing). Trains ~12-18h for 30-50k steps. This is the
        biggest preset that still fits free GPUs."""
        return cls(
            dim=576,
            n_layers=22,
            n_heads=12,
            n_kv_heads=4,
            mlp_mult=2.5,
            max_seq_len=1024,
            vocab_size=16384,
            vision_layers=6,
            vision_heads=12,
            audio_layers=4,
            img_size=96,
            patch_size=8,
            video_size=64,
            n_video_frames=8,
            audio_frames=80,
            n_note_steps=32,
        )

    @classmethod
    def tiny(cls) -> "SARAConfig":
        """Even smaller CPU smoke config."""
        return cls(
            dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
            vocab_size=2048,
            vision_layers=1,
            audio_layers=1,
            img_size=32,
            patch_size=8,
            video_size=32,
            n_video_frames=6,
            audio_frames=48,
            n_note_steps=16,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SARAConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        preset = data.pop("preset", None)
        if preset == "tiny":
            cfg = cls.tiny()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        if preset == "small":
            cfg = cls.small()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        if preset == "medium":
            cfg = cls.medium()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        if preset == "large":
            cfg = cls.large()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        return cls.from_dict(data)


def load_config(path: Optional[str | Path] = None) -> SARAConfig:
    if path is None:
        default = Path(__file__).resolve().parents[1] / "configs" / "sara_nano.yaml"
        if default.exists():
            path = default
    if path is None:
        return SARAConfig.nano()
    return SARAConfig.from_yaml(path)
