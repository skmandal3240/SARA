"""CPU INT8 path (SIA demo 9 restored) + kernel bench + ONNX stub.

Dynamic quantize on `nn.Linear` only. No NVIDIA ifdef. ONNX export is
hardware-agnostic later — Phase A returns a stub rather than a fake file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def quantize_linear_int8(model: nn.Module) -> nn.Module:
    """Torch dynamic INT8 on Linear. Untie lm_head if it shares Embedding storage."""
    if hasattr(model, "lm_head") and hasattr(model, "tok_emb"):
        w = getattr(model.lm_head, "weight", None)
        e = getattr(model.tok_emb, "weight", None)
        if w is not None and e is not None and w.data_ptr() == e.data_ptr():
            model.lm_head.weight = nn.Parameter(w.detach().clone())
    qmod = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return qmod


def _is_quantized_linear(mod: nn.Module) -> bool:
    name = type(mod).__name__.lower()
    modu = (type(mod).__module__ or "").lower()
    blob = name + " " + modu + " " + type(mod).__qualname__.lower()
    if "linear" not in blob:
        return False
    return any(s in blob for s in ("quant", "dynamic", "qint", "torch.ao.nn"))


def count_quantized_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if _is_quantized_linear(m))


def export_onnx_stub(model: nn.Module, path: str | Path | None = None) -> dict[str, Any]:
    """Phase B. Do not pretend we shipped a portable NPU graph today."""
    return {
        "ok": False,
        "reason": "ONNX / delegate export is Phase B; Phase A is CPU INT8 + SARA-ISA doc",
        "path": None if path is None else str(path),
    }


def bench_kernels(dim: int = 256, seq: int = 64, steps: int = 40) -> dict[str, Any]:
    """Tiny CPU timings for the ISA ops. Not a TOPS datasheet."""
    from sara.modules import RMSNorm, SwiGLU

    device = torch.device("cpu")
    x = torch.randn(2, seq, dim, device=device)
    rms = RMSNorm(dim)
    swiglu = SwiGLU(dim, hidden=max(64, dim * 2))
    wq = nn.Linear(dim, dim, bias=False)

    def _ms(fn) -> float:
        fn()
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / steps

    out = {
        "rmsnorm_ms": _ms(lambda: rms(x)),
        "swiglu_ms": _ms(lambda: swiglu(x)),
        "linear_fp32_ms": _ms(lambda: wq(x)),
        "device": "cpu",
        "dim": dim,
        "seq": seq,
    }
    try:
        q = quantize_linear_int8(nn.Linear(dim, dim, bias=False))
        out["linear_int8_ms"] = _ms(lambda: q(x))
        out["int8"] = True
    except Exception as e:
        out["int8"] = False
        out["int8_error"] = str(e)
    return out
