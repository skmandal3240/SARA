#!/usr/bin/env python3
"""Train SARA small (~45M params) — designed for a free Colab T4 GPU.

Local CPU smoke run:   python train_small.py --steps 5 --device cpu
Colab T4 real run:     python train_small.py --steps 5000

Data: same synthetic corpus as nano (prepare_data.py output). This proves the
architecture trains at 45M scale; production data comes later via sara/data adapters.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from sara.config import SARAConfig, load_config
from sara.model import SARA


def get_batch(data: torch.Tensor, batch: int, seq: int, device: str):
    ix = torch.randint(0, len(data) - seq - 1, (batch,))
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sara_small.yaml")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/sara_small")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SARA(cfg).to(args.device)
    print(f"SARA small: {model.n_params()/1e6:.1f}M params on {args.device}")

    tok_path = Path("data/train.bin")
    if not tok_path.exists():
        raise SystemExit("run prepare_data.py first")
    # pack_bin writes (N, seq_len) uint16 rows
    flat = np.fromfile(tok_path, dtype=np.uint16).astype(np.int64)
    data = torch.from_numpy(flat)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    seq = min(256, cfg.max_seq_len - 1)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch, seq, args.device)
        logits = model(tokens=x)["logits"]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.3f} | {time.time()-t0:.0f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out / "sara.pt")
    print(f"saved {out/'sara.pt'}")


if __name__ == "__main__":
    main()
