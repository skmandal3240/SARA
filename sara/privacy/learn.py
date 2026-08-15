"""On-device learning hook: store correction pairs. Do not upload raw data.

Federated path (later) averages **LoRA-class deltas / adapters only**,
never chat logs or camera frames.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class LocalLearner:
    adapter_id = "local-v0"

    def __init__(self, path: str | Path, audit: Any = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def correct(self, prompt: str, expected: str, actual: Optional[str] = None) -> dict[str, Any]:
        items = self._load()
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "expected": expected,
            "actual": actual,
        }
        items.append(row)
        self._save(items)
        if self.audit is not None:
            self.audit.record("learn_correct", n=len(items), uploaded=False)
        return row

    def pairs(self) -> list[dict[str, Any]]:
        return self._load()

    def export_adapter(self) -> dict[str, Any]:
        """Adapter-only payload. Raw video/chat is not included."""
        pairs = self._load()
        return {
            "adapter_id": self.adapter_id,
            "kind": "correction-pairs",
            "n": len(pairs),
            # hashes, not raw prompts — fleet average must not see user text
            "delta": [{"i": i, "n_chars": len(p.get("expected") or "")} for i, p in enumerate(pairs)],
            "uploaded": False,
        }

    def import_federated_average(self, deltas: list[dict[str, Any]]) -> dict[str, Any]:
        """Accept adapter deltas from *our* fleet. Reject payloads that look like raw media."""
        for d in deltas:
            if any(k in d for k in ("frames", "wav", "chat", "raw")):
                raise ValueError("federated import refuses raw data; adapters only")
        if self.audit is not None:
            self.audit.record("learn_federated_in", n=len(deltas), uploaded=False)
        return {"ok": True, "n": len(deltas), "adapter_id": self.adapter_id}
