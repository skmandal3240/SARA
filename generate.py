#!/usr/bin/env python3
"""Text / code generation CLI for a trained (or random-init) SARA checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sara.config import SARAConfig
from sara.model import SARA
from sara.tokenizer import SARATokenizer, default_tokenizer_path

ROOT = Path(__file__).resolve().parent


def load_sara(ckpt: Path | None, device: str = "cpu"):
    tok = SARATokenizer.from_file(default_tokenizer_path())
    if ckpt and ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = SARAConfig.from_dict(blob["config"])
        cfg.vocab_size = tok.vocab_size
        cfg.pad_id, cfg.bos_id, cfg.eos_id = tok.pad_id, tok.bos_id, tok.eos_id
        model = SARA(cfg)
        model.load_state_dict(blob["model"], strict=False)
    else:
        cfg = SARAConfig.nano()
        cfg.vocab_size = tok.vocab_size
        cfg.pad_id, cfg.bos_id, cfg.eos_id = tok.pad_id, tok.bos_id, tok.eos_id
        model = SARA(cfg)
    model.to(device).eval()
    return model, tok, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The sun is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sara_nano" / "sara.pt"))
    ap.add_argument("--code", action="store_true")
    args = ap.parse_args()
    model, tok, cfg = load_sara(Path(args.ckpt))
    if args.code:
        text = f"<|bos|><|user|>{args.prompt}<|code_start|>"
    else:
        text = tok.wrap_user(args.prompt)
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    out = model.generate(ids, max_new=args.max_new, temperature=args.temperature, eos_id=tok.eos_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
