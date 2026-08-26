#!/usr/bin/env python3
"""Train SARA medium (~132M params) on a big corpus — Kaggle T4 x2 / P100.

Local CPU smoke:   python train_medium.py --steps 2 --device cpu
Kaggle real run:   python train_medium.py --steps 40000

Differences from train_small.py:
- bf16 autocast (T4 supports fp16; bf16 on Ampere+; script picks automatically)
- gradient checkpointing to fit 128M in <14 GB
- cosine LR schedule with warmup
- periodic checkpoint saves so a crashed session isn't lost work
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from sara.config import load_config
from sara.model import SARA


def get_batch(data: torch.Tensor, batch: int, seq: int, device: str):
    ix = torch.randint(0, len(data) - seq - 1, (batch,))
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sara_medium.yaml")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/sara_medium")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SARA(cfg).to(args.device)
    print(f"SARA medium: {model.n_params()/1e6:.1f}M params on {args.device}")

    p = Path("data/train.bin")
    if not p.exists():
        raise SystemExit("run prepare_data.py first")
    data = torch.from_numpy(np.fromfile(p, dtype=np.uint16).astype(np.int64))
    print(f"train tokens: {len(data)/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup,
                           0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))))

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    seq = min(args.seq, cfg.max_seq_len - 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch, seq, args.device)
        with torch.autocast(device_type=args.device, dtype=amp_dtype,
                            enabled=args.device.startswith("cuda")):
            logits = model(tokens=x)["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            tok_s = step * args.batch * seq / max(time.time() - t0, 1)
            print(f"step {step:6d} | loss {loss.item():.3f} | lr {sched.get_last_lr()[0]:.2e} | {tok_s:.0f} tok/s | {time.time()-t0:.0f}s")
        if step % 2000 == 0:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                       out_dir / "sara.pt")
            print(f"  [checkpoint saved @ step {step}]")

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": args.steps},
               out_dir / "sara.pt")
    print(f"saved {out_dir/'sara.pt'}")


if __name__ == "__main__":
    main()
