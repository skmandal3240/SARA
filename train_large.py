#!/usr/bin/env python3
"""Train SARA large (~550M params) — Kaggle P100 / multi-GPU / IndiaAI GPUs.

CPU smoke:   python train_large.py --steps 1 --batch 1 --device cpu
P100 real:   python train_large.py --steps 60000 --batch 4 --seq 1024

Adds over train_medium.py:
- gradient accumulation (--accum) to reach effective batch on one GPU
- bf16 autocast always-on for CUDA
- checkpoint resume via --init-from
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
    ap.add_argument("--config", default="configs/sara_large.yaml")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation steps")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/sara_large")
    ap.add_argument("--init-from", default=None, help="resume from a sara.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SARA(cfg).to(args.device)
    start_step = 0
    if args.init_from and Path(args.init_from).exists():
        # our own checkpoint format: {"model": state_dict, "config": dict, "step": int}
        blob = torch.load(args.init_from, map_location=args.device, weights_only=False)
        model.load_state_dict(blob["model"])
        start_step = blob.get("step", 0)
        print(f"resumed from step {start_step}")
    print(f"SARA large: {model.n_params()/1e6:.1f}M params on {args.device}")

    p = Path("data/train.bin")
    if not p.exists():
        raise SystemExit("run prepare_data.py first")
    data = torch.from_numpy(np.fromfile(p, dtype=np.uint16).astype(np.int64))
    print(f"train tokens: {len(data)/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            fused=args.device.startswith("cuda"))
    total_steps = args.steps - start_step
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup,
                           0.5 * (1 + math.cos(math.pi * min((start_step + s) / args.steps, 1.0)))))

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    seq = min(args.seq, cfg.max_seq_len - 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(step):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                   out_dir / "sara.pt")

    t0 = time.time()
    opt.zero_grad()
    for step in range(start_step + 1, args.steps + 1):
        rel = step - start_step
        loss_acc = 0.0
        for _ in range(args.accum):
            x, y = get_batch(data, args.batch, seq, args.device)
            with torch.autocast(device_type=args.device, dtype=amp_dtype,
                                enabled=args.device.startswith("cuda")):
                logits = model(tokens=x)["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1)) / args.accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if rel % 50 == 0 or rel == 1:
            tok_s = rel * args.batch * args.accum * seq / max(time.time() - t0, 1)
            print(f"step {step:6d} | loss {loss_acc:.3f} | lr {sched.get_last_lr()[0]:.2e} | {tok_s:.0f} tok/s | {time.time()-t0:.0f}s")
        if rel % 2000 == 0:
            save(step)
            print(f"  [checkpoint @ step {step}]")

    save(args.steps)
    print(f"saved {out_dir/'sara.pt'}")


if __name__ == "__main__":
    main()
