"""Short-term scratchpad + optional file-backed long-term notes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Event:
    kind: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


class Scratchpad:
    def __init__(self, max_events: int = 64):
        self.max_events = max_events
        self.events: list[Event] = []

    def add(self, kind: str, content: str, **meta: Any) -> None:
        self.events.append(Event(kind, content, meta))
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def render(self, last_n: int = 24) -> str:
        chunks = []
        for e in self.events[-last_n:]:
            chunks.append(f"<|{e.kind}|>{e.content}")
        return "".join(chunks)

    def last_observation(self) -> Optional[str]:
        for e in reversed(self.events):
            if e.kind == "observation":
                return e.content
        return None


class LongTermMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def note(self, text: str, tag: str = "note") -> None:
        items = self._load()
        items.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": tag,
                "text": text,
            }
        )
        self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def recall(self, query: str, k: int = 5) -> list[dict]:
        q = query.lower().split()
        scored = []
        for item in self._load():
            text = item.get("text", "").lower()
            score = sum(1 for w in q if w in text)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for s, it in scored[:k] if s > 0] or [s[1] for s in scored[:k]]
