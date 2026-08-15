"""Local audit journal + signed inference log (model hash + quant + adapter id)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def model_hash(model: Any) -> str:
    if model is None:
        return "none"
    h = hashlib.sha256()
    try:
        n = int(sum(p.numel() for p in model.parameters()))
        h.update(str(n).encode())
        # a few tensors — enough to change when weights change, cheap on nano
        for i, p in enumerate(model.parameters()):
            if i >= 4:
                break
            h.update(p.detach().cpu().flatten()[:32].numpy().tobytes())
    except Exception:
        h.update(type(model).__name__.encode())
    cfg = getattr(model, "config", None)
    if cfg is not None:
        h.update(str(getattr(cfg, "dim", "")).encode())
        h.update(str(getattr(cfg, "n_layers", "")).encode())
    return h.hexdigest()[:16]


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **{k: _plain(v) for k, v in fields.items()},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def inference(
        self,
        event: str = "infer",
        model: Any = None,
        quant: str = "fp32",
        adapter_id: str = "none",
        placement: str = "local",
        **extra: Any,
    ) -> dict[str, Any]:
        return self.record(
            "inference",
            kind=event,
            model_hash=model_hash(model),
            quant=quant,
            adapter_id=adapter_id,
            placement=placement,
            **extra,
        )

    def read(self) -> list[dict[str, Any]]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out = []
        for ln in lines:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
        return out


def _plain(v: Any) -> Any:
    if hasattr(v, "where") and hasattr(v, "reason"):
        return {"where": v.where, "reason": v.reason}
    return v
