"""Capability ledger. Off until preview + approve. Default deny."""

from __future__ import annotations

from typing import Any, Optional

CAPABILITIES = ("camera", "mic", "files", "mesh", "cloud", "learn", "share_raw")

# Tool name -> capability. Attached only when a GrantLedger is bound.
HIGH_RISK_TOOLS = {
    "shell": "files",
    "file_write": "files",
    "file_read": "files",
    "web_search": "cloud",
    "browser": "cloud",
}


class GrantError(PermissionError):
    pass


class GrantLedger:
    def __init__(self, previews: Optional[dict[str, str]] = None):
        self._on = {c: False for c in CAPABILITIES}
        self._previewed: dict[str, str] = dict(previews or {})
        self.history: list[dict[str, Any]] = []

    def allowed(self, cap: str) -> bool:
        return bool(self._on.get(cap, False))

    def preview(self, cap: str, reason: str) -> dict[str, Any]:
        if cap not in CAPABILITIES:
            raise GrantError(f"unknown capability {cap!r}")
        self._previewed[cap] = reason
        rec = {"op": "preview", "cap": cap, "reason": reason, "on": False}
        self.history.append(rec)
        return {"cap": cap, "reason": reason, "currently": self._on[cap]}

    def approve(self, cap: str) -> None:
        if cap not in CAPABILITIES:
            raise GrantError(f"unknown capability {cap!r}")
        if cap not in self._previewed:
            raise GrantError(f"{cap} must be previewed before approve")
        self._on[cap] = True
        self.history.append({"op": "approve", "cap": cap, "on": True})

    def revoke(self, cap: str) -> None:
        if cap not in self._on:
            raise GrantError(f"unknown capability {cap!r}")
        self._on[cap] = False
        self.history.append({"op": "revoke", "cap": cap, "on": False})

    def require(self, cap: str) -> None:
        if not self.allowed(cap):
            raise GrantError(f"capability {cap!r} denied (default-deny; preview+approve first)")

    def snapshot(self) -> dict[str, bool]:
        return dict(self._on)

    def check_tool(self, name: str) -> Optional[str]:
        """Return a denial string if the tool is high-risk and not granted."""
        cap = HIGH_RISK_TOOLS.get(name)
        if cap is None:
            return None
        if self.allowed(cap):
            return None
        return f"grant denied: {name} needs {cap} (preview+approve)"
