#!/usr/bin/env python3
"""CLI for SARA tools — list, call, or drop into the agent loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sara.agent.loop import AgentRuntime
from sara.tools.protocol import ToolCall

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="SARA tool / agent CLI")
    ap.add_argument("action", choices=["list", "call", "agent", "catalog"])
    ap.add_argument("--name", help="tool name for call")
    ap.add_argument("--args", default="{}", help="JSON args")
    ap.add_argument("--goal", default="What is 21*2?")
    args = ap.parse_args()

    rt = AgentRuntime(ROOT)
    if args.action in {"list", "catalog"}:
        print(rt.registry.catalog())
        return
    if args.action == "call":
        if not args.name:
            raise SystemExit("--name required")
        payload = json.loads(args.args)
        result = rt.registry.dispatch(ToolCall(args.name, payload))
        print(json.dumps(result, indent=2, default=str))
        return
    if args.action == "agent":
        # optional model
        ckpt = ROOT / "checkpoints" / "sara_nano" / "sara.pt"
        tok = model = cfg = None
        try:
            from generate import load_sara

            model, tok, cfg = load_sara(ckpt if ckpt.exists() else None)
            rt = AgentRuntime(ROOT, model=model, tokenizer=tok, cfg=cfg)
        except Exception as e:
            print("running without weights:", e)
        res = rt.run(args.goal)
        print("FINAL:", res.final)
        print("steps", res.steps, "tools", res.tool_calls)


if __name__ == "__main__":
    main()
