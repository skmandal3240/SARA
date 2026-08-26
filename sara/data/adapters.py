"""Loader stubs: stream a tiny HF sample with a max_tokens cap.

Never download TB. Cache dir is gitignored. Manifests are committed separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import catalog

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "hf_cache"


def _text_of(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    for k in ("text", "content", "src", "tgt", "sentence", "transcript", "code"):
        if k in row and isinstance(row[k], str):
            return row[k]
    # concatenate a couple of string fields
    parts = [str(v) for v in row.values() if isinstance(v, str)]
    return " ".join(parts)[:2000]


def stream_hf(
    hf_id: str,
    max_tokens: int = 2048,
    split: str = "train",
    cache_dir: str | Path | None = None,
    license_filter: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Fetch at most max_tokens of text. Skip if `datasets` is missing."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return {
            "ok": False,
            "reason": "datasets library not installed; skip (no TB download)",
            "hf": hf_id,
            "texts": [],
        }
    cache = Path(cache_dir or CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    n_chars = 0
    # rough: 1 token ~ 4 chars for the cap
    char_budget = max(32, int(max_tokens) * 4)
    try:
        ds = load_dataset(
            hf_id,
            split=split,
            streaming=True,
            cache_dir=str(cache),
            trust_remote_code=False,
        )
        for row in ds:
            if license_filter:
                lic = str((row.get("license") if isinstance(row, dict) else "") or "")
                if lic and lic not in license_filter and lic.upper() not in {
                    x.upper() for x in license_filter
                }:
                    continue
            t = _text_of(row).strip()
            if not t:
                continue
            texts.append(t)
            n_chars += len(t)
            if n_chars >= char_budget or len(texts) >= 32:
                break
    except Exception as e:
        return {
            "ok": False,
            "reason": f"hf stream failed: {type(e).__name__}: {e}",
            "hf": hf_id,
            "texts": [],
        }
    return {
        "ok": True,
        "hf": hf_id,
        "texts": texts,
        "n_chars": n_chars,
        "max_tokens": max_tokens,
        "cache_dir": str(cache),
    }


def load_sample(dataset_id: str, max_tokens: int = 512) -> dict[str, Any]:
    meta = catalog.get(dataset_id)
    hf = meta.get("hf")
    if not hf:
        return {
            "ok": True,
            "id": dataset_id,
            "reason": "local / no hf id",
            "texts": [],
            "meta": meta,
        }
    rec = stream_hf(
        hf,
        max_tokens=max_tokens,
        license_filter=meta.get("license_filter"),
    )
    rec["id"] = dataset_id
    rec["meta"] = meta
    return rec


def ingest_source(spec: str, max_tokens: int = 2048) -> list[str]:
    """prepare_data.py --source hf:org/name or a catalog id."""
    spec = spec.strip()
    if spec.startswith("hf:"):
        rec = stream_hf(spec[3:], max_tokens=max_tokens)
        return list(rec.get("texts") or [])
    if spec.startswith("file:"):
        # local text corpus (one doc per <|doc|> separator)
        from pathlib import Path as _P

        p = _P(spec[5:])
        raw = p.read_text(encoding="utf-8")
        docs = [d.strip() for d in raw.split("<|doc|>") if len(d.strip()) >= 200]
        return docs
    if spec in catalog.DATASETS:
        rec = load_sample(spec, max_tokens=max_tokens)
        return list(rec.get("texts") or [])
    raise ValueError(f"unknown source {spec!r}; use hf:org/name or a catalog id")
