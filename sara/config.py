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
        return cls.from_dict(data)


def load_config(path: Optional[str | Path] = None) -> SARAConfig:
    if path is None:
        default = Path(__file__).resolve().parents[1] / "configs" / "sara_nano.yaml"
        if default.exists():
            path = default
    if path is None:
        return SARAConfig.nano()
    return SARAConfig.from_yaml(path)
