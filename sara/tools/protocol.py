"""Structured tool-call protocol.

The model emits:

    <|thought|>...
    <|tool_call|>{"name": "calc", "args": {"expr": "2+2"}}<|tool_end|>
    <|observation|>...
    <|final|>...

A dedicated ToolHead can also pick a tool id; args still travel as text/JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

TOOL_JSON_RE = re.compile(
    r"<\|tool_call\|>(?P<body>.*?)<\|tool_end\|>",
    re.DOTALL,
)
# fallback XML-ish / bracket form
ALT_RE = re.compile(
    r"\[\[tool:(?P<name>[A-Za-z0-9_]+)\((?P<args>.*?)\)\]\]",
    re.DOTALL,
)
THOUGHT_RE = re.compile(r"<\|thought\|>(.*?)(?=<\|(?:tool_call|final|plan|observation|assistant)\|>|$)", re.DOTALL)
FINAL_RE = re.compile(r"<\|final\|>(.*?)(?=<\||$)", re.DOTALL)
PLAN_RE = re.compile(r"<\|plan\|>(.*?)(?=<\|(?:thought|tool_call|final|assistant)\|>|$)", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "args": self.args}, ensure_ascii=False)


@dataclass
class ParsedTurn:
    thought: Optional[str] = None
    plan: Optional[str] = None
    calls: list[ToolCall] = field(default_factory=list)
    final: Optional[str] = None
    text: str = ""

    @property
    def is_final(self) -> bool:
        return self.final is not None and not self.calls

    @property
    def has_call(self) -> bool:
        return bool(self.calls)


def _loads_maybe(s: str) -> Any:
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # tolerate trailing commas / single quotes lightly
        try:
            return json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            return {"_raw": s}


def parse_tool_body(body: str) -> ToolCall:
    data = _loads_maybe(body)
    if isinstance(data, dict) and "name" in data:
        args = data.get("args") or data.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        return ToolCall(name=str(data["name"]), args=args, raw=body)
    if isinstance(data, dict) and len(data) == 1:
        name, args = next(iter(data.items()))
        if not isinstance(args, dict):
            args = {"value": args}
        return ToolCall(name=str(name), args=args, raw=body)
    return ToolCall(name="unknown", args={"_raw": body}, raw=body)


def parse_turn(text: str) -> ParsedTurn:
    text = text or ""
    calls: list[ToolCall] = []
    for m in TOOL_JSON_RE.finditer(text):
        calls.append(parse_tool_body(m.group("body")))
    if not calls:
        for m in ALT_RE.finditer(text):
            args = _loads_maybe(m.group("args"))
            if not isinstance(args, dict):
                # positional fallback: expr=... or a single string
                raw_args = m.group("args").strip()
                if "=" in raw_args and "{" not in raw_args:
                    args = {}
                    for part in raw_args.split(","):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            args[k.strip()] = v.strip().strip("'\"")
                else:
                    args = {"value": raw_args}
            calls.append(ToolCall(name=m.group("name"), args=args, raw=m.group(0)))
    thought = None
    m = THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()
    final = None
    m = FINAL_RE.search(text)
    if m:
        final = m.group(1).strip()
    plan = None
    m = PLAN_RE.search(text)
    if m:
        plan = m.group(1).strip()
    return ParsedTurn(thought=thought, plan=plan, calls=calls, final=final, text=text)


def emit_tool_call(name: str, args: dict[str, Any]) -> str:
    return f"<|tool_call|>{json.dumps({'name': name, 'args': args}, ensure_ascii=False)}<|tool_end|>"


def emit_observation(obs: str) -> str:
    return f"<|observation|>{obs}"


def emit_final(text: str) -> str:
    return f"<|final|>{text}"


def emit_thought(text: str) -> str:
    return f"<|thought|>{text}"
