"""Per-user vault: JSONL encrypted at rest. Fernet if present, else xor+key file."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


def _xor(data: bytes, key: bytes) -> bytes:
    if not key:
        key = b"\x00"
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


class Vault:
    def __init__(
        self,
        path: str | Path,
        key_path: str | Path | None = None,
        audit: Optional[Callable[..., None]] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = Path(key_path) if key_path else self.path.with_suffix(".key")
        self.audit = audit
        self._fernet = None
        self._key = self._load_or_create_key()
        self._init_cipher()
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes()
        key = os.urandom(32)
        self.key_path.write_bytes(key)
        try:
            os.chmod(self.key_path, 0o600)
        except OSError:
            pass
        return key

    def _init_cipher(self) -> None:
        try:
            import hashlib
            from cryptography.fernet import Fernet

            fkey = base64.urlsafe_b64encode(hashlib.sha256(self._key).digest())
            self._fernet = Fernet(fkey)
        except Exception:
            self._fernet = None

    def _encrypt(self, blob: bytes) -> dict[str, str]:
        if self._fernet is not None:
            ct = self._fernet.encrypt(blob)
            return {"alg": "fernet", "ct": base64.b64encode(ct).decode("ascii")}
        return {"alg": "xor", "ct": base64.b64encode(_xor(blob, self._key)).decode("ascii")}

    def _decrypt(self, rec: dict[str, str]) -> bytes:
        raw = base64.b64decode(rec["ct"].encode("ascii"))
        if rec.get("alg") == "fernet" and self._fernet is not None:
            return self._fernet.decrypt(raw)
        return _xor(raw, self._key)

    def put(self, record: dict[str, Any], key: str | None = None) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "key": key or record.get("key") or record.get("id"),
            "record": record,
        }
        blob = json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")
        line = json.dumps(self._encrypt(blob), separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.audit:
            fn = self.audit.record if hasattr(self.audit, "record") else self.audit
            fn("vault_write", key=row["key"], n=1)

    def items(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            payload = json.loads(self._decrypt(rec).decode("utf-8"))
            out.append(payload)
        return out

    def get(self, key: str) -> Optional[dict[str, Any]]:
        found = None
        for item in self.items():
            if item.get("key") == key:
                found = item
        return found
