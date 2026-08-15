"""SHADE-shaped browser kernel interface. Not a browser.

SARA does not vendor Shade, Firecrawl, or a headless Chrome. Agents may
attach a *separate* Shade MCP process. SSRF / LAN policy live on the
device profile and grant ledger, not here.
"""

from __future__ import annotations

from typing import Any, Optional


class BrowserMCP:
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint

    def fetch(self, url: str) -> dict[str, Any]:
        return {
            "ok": False,
            "reason": "no browser in this repo; attach Shade via MCP",
            "url": url,
            "endpoint": self.endpoint,
        }
