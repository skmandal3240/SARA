"""Unified BPE tokenizer with multimodal and agent special tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    # vision
    "<|img_start|>",
    "<|img_end|>",
    "<|img_gen|>",
    "<|img_patch|>",
    # audio / speech
    "<|aud_start|>",
    "<|aud_end|>",
    "<|speech_in|>",
    "<|speech_out|>",
    "<|aud_frame|>",
    # code
    "<|code_start|>",
    "<|code_end|>",
    "<|exec|>",
    "<|result|>",
    # video
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_gen|>",
    # song
    "<|song_start|>",
    "<|song_end|>",
    "<|song_gen|>",
    # tools / agents
    "<|thought|>",
    "<|plan|>",
    "<|tool_call|>",
    "<|tool_end|>",
    "<|observation|>",
    "<|final|>",
    "<|agent|>",
    "<|delegate|>",
    "<|scratch|>",
]


class SARATokenizer:
    def __init__(self, tokenizer, special_map: Optional[dict[str, int]] = None):
        self._tok = tokenizer
        self.special = special_map or {t: i for i, t in enumerate(SPECIAL_TOKENS)}

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.special["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special["<|eos|>"]

    def id(self, token: str) -> int:
        if token in self.special:
            return self.special[token]
        vid = self._tok.token_to_id(token)
        if vid is None:
            return self.special["<|unk|>"]
        return vid

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        ids = list(ids)
        if skip_special:
            special_ids = set(self.special.values())
            ids = [i for i in ids if i not in special_ids]
        return self._tok.decode(ids)

    def wrap_user(self, text: str) -> str:
        return f"<|bos|><|user|>{text}<|assistant|>"

    def wrap_code(self, code: str) -> str:
        return f"<|code_start|>{code}<|code_end|>"

    def wrap_tool(self, payload: str) -> str:
        return f"<|tool_call|>{payload}<|tool_end|>"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))
        meta = path.with_suffix(".special.json")
        meta.write_text(json.dumps(self.special, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "SARATokenizer":
        from tokenizers import Tokenizer

        path = Path(path)
        tok = Tokenizer.from_file(str(path))
        meta = path.with_suffix(".special.json")
        special = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
        return cls(tok, special)

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 4096,
        save_path: Optional[str | Path] = None,
    ) -> "SARATokenizer":
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders

        vocab_size = max(vocab_size, len(SPECIAL_TOKENS) + 256)
        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=1,
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        tok.post_processor = processors.ByteLevel(trim_offsets=True)

        special = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
        inst = cls(tok, special)
        if save_path is not None:
            inst.save(save_path)
        return inst


def default_tokenizer_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tokenizer" / "sara.json"


def load_or_none(path: Optional[str | Path] = None) -> Optional[SARATokenizer]:
    path = Path(path) if path else default_tokenizer_path()
    if path.exists():
        return SARATokenizer.from_file(path)
    return None
