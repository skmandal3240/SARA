#!/usr/bin/env python3
"""Train the SARA nano model on CPU (fp32). Language modeling + light aux heads."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from sara.config import SARAConfig, load_config
from sara.model import SARA
from sara.tokenizer import SARATokenizer, default_tokenizer_path
from sara.vision import make_shape_image, pil_to_tensor

ROOT = Path(__file__).resolve().parent
COLORS = ["red", "green", "blue", "yellow"]
KINDS = ["circle", "square", "triangle"]


def cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * t))


def load_tokens(path: Path, seq_len: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    n = (len(arr) // seq_len) * seq_len
    return arr[:n].reshape(-1, seq_len)


def random_shape(cfg: SARAConfig, tok: SARATokenizer, device):
    color = COLORS[np.random.randint(0, len(COLORS))]
    kind = KINDS[np.random.randint(0, len(KINDS))]
    img = make_shape_image(kind, color, size=cfg.img_size)
    ten = pil_to_tensor(img, cfg.img_size).unsqueeze(0).to(device)
    cap = f"<|bos|><|user|>a {color} {kind}<|img_gen|>"
    ids = tok.encode(cap)
    ids = ids[: cfg.max_seq_len]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    return ten, tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "sara_nano.yaml"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sara_nano" / "sara.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "train.bin"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    tok_path = default_tokenizer_path()
    if not tok_path.exists():
        raise SystemExit("tokenizer missing — run prepare_data.py first")
    tok = SARATokenizer.from_file(tok_path)
    cfg.vocab_size = tok.vocab_size
    cfg.pad_id = tok.pad_id
    cfg.bos_id = tok.bos_id
    cfg.eos_id = tok.eos_id
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    device = torch.device(args.device)
    model = SARA(cfg).to(device)
    print(f"SARA params={model.n_params()/1e6:.2f}M dim={cfg.dim} layers={cfg.n_layers} vocab={cfg.vocab_size}")

    data = load_tokens(Path(args.data), cfg.max_seq_len)
    print(f"train sequences={len(data)}")

    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.04)
    model.train()
    t0 = time.time()
    running = 0.0
    last_loss = None

    nseq = len(data)
    step = 0
    accum = args.accum
    opt.zero_grad(set_to_none=True)

    while step < args.steps:
        idx = np.random.randint(0, nseq, size=args.batch)
        batch = torch.tensor(data[idx].astype(np.int64), device=device)
        out = model(batch, targets=batch)
        loss = out["loss"]

        # auxiliary image reconstruction/generation every other step
        if step % 2 == 0:
            img, cap = random_shape(cfg, tok, device)
            vis = model.vision(img)
            # see: visual tokens should be a valid context (no crash + mild recon)
            recon = model.image_dec(vis.mean(dim=1))
            loss = loss + 0.4 * F.mse_loss(recon, img)
            # create: caption hidden → same image
            gen = model.generate_image(cap)
            loss = loss + 0.4 * F.mse_loss(gen, img)

        if step % 5 == 0:
            # song / mel heads: just keep them in the graph with a small prior
            cond = model.cond_from_tokens(batch[:1])
            song = model.song_head(cond)
            # prefer non-rest notes slightly
            note_prior = F.softmax(song["notes"], dim=-1)[:, :, 0].mean()
            loss = loss + 0.05 * note_prior
            mel = model.mel_dec(cond)
            loss = loss + 0.02 * (mel ** 2).mean()

        (loss / accum).backward()
        running += float(loss.item())
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
        last_loss = float(loss.item())
        if step % 25 == 0 or step == args.steps - 1:
            avg = running / 25 if step else last_loss
            running = 0.0
            dt = time.time() - t0
            print(f"step {step:4d}/{args.steps} loss={last_loss:.4f} lr={opt.param_groups[0]['lr']:.2e} t={dt:.1f}s")
        step += 1

    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "step": args.steps,
            "loss": last_loss,
            "vocab_size": cfg.vocab_size,
        },
        ckpt,
    )
    print(f"saved {ckpt} final_loss={last_loss:.4f} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
