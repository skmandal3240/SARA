"""Typed tool registry: name, schema, description, implementation."""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .protocol import ToolCall


class ToolError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, str]
    fn: Callable[..., Any]
    required: list[str] = field(default_factory=list)

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        missing = [k for k in self.required if k not in args]
        if missing:
            raise ToolError(f"{self.name}: missing args {missing}")
        # drop unknown keys to keep impls strict-ish
        known = set(self.args_schema) | set(self.required)
        return {k: v for k, v in args.items() if (not known or k in known or k.startswith("_"))}


class ToolRegistry:
    def __init__(self, grants=None, audit=None):
        self._tools: dict[str, ToolSpec] = {}
        self.grants = grants
        self.audit = audit

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def add(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        args_schema: dict[str, str],
        required: Optional[list[str]] = None,
    ) -> None:
        self.register(ToolSpec(name, description, args_schema, fn, required or list(args_schema)))

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"unknown tool {name!r}. known: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def catalog(self) -> str:
        lines = []
        for spec in self._tools.values():
            args = ", ".join(f"{k}: {v}" for k, v in spec.args_schema.items())
            lines.append(f"- {spec.name}({args}) — {spec.description}")
        return "\n".join(lines)

    def id_map(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self._tools)}

    def name_for_id(self, i: int) -> Optional[str]:
        names = self.names()
        if 0 <= i < len(names):
            return names[i]
        return None

    def dispatch(self, call: ToolCall) -> dict[str, Any]:
        if self.grants is not None:
            deny = self.grants.check_tool(call.name)
            if deny:
                if self.audit is not None:
                    self.audit.record("tool_denied", tool=call.name, error=deny)
                return {"ok": False, "tool": call.name, "error": deny, "grant_denied": True}
        try:
            spec = self.get(call.name)
            args = spec.validate(call.args)
            sig = inspect.signature(spec.fn)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                result = spec.fn(**args)
            else:
                allowed = {
                    k: args[k]
                    for k in sig.parameters
                    if k in args and k != "self"
                }
                result = spec.fn(**allowed)
            if self.audit is not None:
                self.audit.record("tool_call", tool=call.name, ok=True)
            return {"ok": True, "tool": call.name, "result": result}
        except Exception as e:
            return {
                "ok": False,
                "tool": call.name,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4),
            }
