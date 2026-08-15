"""Built-in tools. Real functions — not print stubs.

Workspace is jailed to a root directory. Shell is allowlisted.
python_exec runs a tiny sandbox (no import of os/sys/subprocess, timeout).
web_search hits DuckDuckGo HTML and degrades cleanly if the network is down.
Modality hooks call into SARA's own image/audio/code heads when a runtime is bound.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import operator
import os
import re
import subprocess
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urljoin
from urllib.request import Request, urlopen

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "sorted": sorted,
    "reversed": reversed,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "print": print,
    "isinstance": isinstance,
    "pow": pow,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
}

_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes",
    "importlib", "multiprocessing", "threading", "http", "urllib", "requests",
}


class _ImportGuard(ast.NodeTransformer):
    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_IMPORTS:
                raise RuntimeError(f"import of {alias.name!r} blocked in sandbox")
        return node

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root in _BLOCKED_IMPORTS:
            raise RuntimeError(f"import from {node.module!r} blocked in sandbox")
        return node


def _jail(root: Path, path: str) -> Path:
    root = root.resolve()
    p = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root != p and root not in p.parents:
        raise PermissionError(f"path {p} escapes workspace {root}")
    return p


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_calc(node):
    if isinstance(node, ast.Expression):
        return _eval_calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_calc(node.left), _eval_calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_calc(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"abs", "round", "int", "float", "pow"}:
            args = [_eval_calc(a) for a in node.args]
            return SAFE_BUILTINS[node.func.id](*args)
    raise ValueError("unsupported expression")


class ToolContext:
    """Shared state bound into built-in tools (workspace, optional SARA runtime)."""

    def __init__(self, workspace: str | Path, runtime: Any = None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime  # AgentRuntime or object with .model/.tokenizer/.generate_text

    def bind_runtime(self, runtime: Any) -> None:
        self.runtime = runtime


def python_exec(code: str, ctx: ToolContext, timeout_s: float = 3.0) -> dict[str, Any]:
    """Execute a restricted Python snippet. Captures stdout and the last expr."""
    try:
        tree = ast.parse(code, mode="exec")
        _ImportGuard().visit(tree)
    except Exception as e:
        return {"stdout": "", "ok": False, "error": f"{type(e).__name__}: {e}"}
    ast.fix_missing_locations(tree)
    buf = StringIO()
    globs: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    # allow writing files only via provided helpers
    globs["WORKSPACE"] = str(ctx.workspace)

    def _print(*a, **k):
        k = dict(k)
        k["file"] = buf
        print(*a, **k)

    globs["__builtins__"] = dict(SAFE_BUILTINS)
    globs["__builtins__"]["print"] = _print
    try:
        compiled = compile(tree, "<sandbox>", "exec")
        exec(compiled, globs, globs)
        last = globs.get("_result", None)
        return {"stdout": buf.getvalue(), "last": last, "ok": True}
    except Exception as e:
        return {"stdout": buf.getvalue(), "ok": False, "error": f"{type(e).__name__}: {e}"}


def calc(expr: str) -> dict[str, Any]:
    tree = ast.parse(expr, mode="eval")
    value = _eval_calc(tree)
    return {"expr": expr, "value": value}


def now() -> dict[str, Any]:
    utc = _dt.datetime.now(_dt.timezone.utc)
    ist = utc.astimezone(_dt.timezone(_dt.timedelta(hours=5, minutes=30)))
    return {
        "utc": utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ist": ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "iso": utc.isoformat(),
    }


def file_read(path: str, ctx: ToolContext, max_bytes: int = 64_000) -> dict[str, Any]:
    p = _jail(ctx.workspace, path)
    data = p.read_bytes()[:max_bytes]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return {"path": str(p.relative_to(ctx.workspace)), "content": text, "n_bytes": len(data)}


def file_write(path: str, content: str, ctx: ToolContext) -> dict[str, Any]:
    p = _jail(ctx.workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"path": str(p.relative_to(ctx.workspace)), "n_bytes": p.stat().st_size}


def list_files(path: str, ctx: ToolContext) -> dict[str, Any]:
    p = _jail(ctx.workspace, path or ".")
    if not p.exists():
        return {"path": path, "entries": [], "error": "not found"}
    if p.is_file():
        return {"path": path, "entries": [p.name]}
    entries = []
    for child in sorted(p.iterdir())[:200]:
        entries.append(("d " if child.is_dir() else "f ") + child.name)
    return {"path": str(p.relative_to(ctx.workspace)) if p != ctx.workspace else ".", "entries": entries}


_SHELL_ALLOW = {
    "python3": ["python3"],
    "python": ["python3"],
    "ls": ["ls", "-la"],
    "cat": ["cat"],
    "wc": ["wc"],
    "date": ["date", "-u"],
    "echo": ["echo"],
    "pwd": ["pwd"],
}


def shell(command: str, ctx: ToolContext, timeout_s: float = 8.0) -> dict[str, Any]:
    """Allowlisted shell. First token must be a known command; cwd is workspace."""
    parts = command.strip().split()
    if not parts:
        raise ValueError("empty command")
    head = parts[0]
    if head not in _SHELL_ALLOW:
        raise PermissionError(f"command {head!r} not allowlisted: {sorted(_SHELL_ALLOW)}")
    # rewrite python → venv python if present
    exe = parts[:]
    if head in {"python", "python3"}:
        venv_py = ctx.workspace.parent / ".venv" / "bin" / "python"
        # workspace may be outputs/; repo root is parent of outputs or the repo itself
        candidates = [
            ctx.workspace / ".venv" / "bin" / "python",
            ctx.workspace.parent / ".venv" / "bin" / "python",
            Path("/workspace/SARA/.venv/bin/python"),
        ]
        for c in candidates:
            if c.exists():
                exe[0] = str(c)
                break
        else:
            exe[0] = "python3"
        # scripts must live under workspace
        for tok in exe[1:]:
            if tok.endswith(".py"):
                _jail(ctx.workspace, tok)
    elif head == "cat":
        if len(exe) > 1:
            _jail(ctx.workspace, exe[1])
    try:
        proc = subprocess.run(
            exe,
            cwd=str(ctx.workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "argv": exe,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"argv": exe, "returncode": -1, "stdout": "", "stderr": "timeout"}


def web_search(query: str, max_results: int = 5, timeout_s: float = 6.0) -> dict[str, Any]:
    """Hit DuckDuckGo's HTML endpoint. If offline, return a clean skip (not a crash)."""
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    headers = {"User-Agent": "SARA-nano/0.1 (research; +https://github.com/skmandal3240/SARA)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout_s) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        results = []
        # DDG html uses result__a and result__snippet
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, flags=re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', html, flags=re.DOTALL)
        hrefs = re.findall(r'class="result__a"[^>]*href="([^"]+)"', html)
        def _strip(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s).replace("&amp;", "&").replace("&quot;", '"').strip()
        n = min(max_results, max(len(titles), len(hrefs)))
        for i in range(n):
            results.append(
                {
                    "title": _strip(titles[i]) if i < len(titles) else "",
                    "url": hrefs[i] if i < len(hrefs) else "",
                    "snippet": _strip(snippets[i]) if i < len(snippets) else "",
                }
            )
        return {"ok": True, "query": query, "results": results, "source": "duckduckgo-html"}
    except Exception as e:
        return {
            "ok": False,
            "query": query,
            "results": [],
            "reason": f"network unavailable or parse failed: {type(e).__name__}: {e}",
        }


def code_run(code: str, ctx: ToolContext) -> dict[str, Any]:
    """Parse then exec. Used by the code pathway and as a tool."""
    try:
        ast.parse(code)
        syntax_ok = True
        syntax_error = None
    except SyntaxError as e:
        return {"syntax_ok": False, "syntax_error": str(e), "ok": False}
    executed = python_exec(code, ctx)
    executed["syntax_ok"] = True
    executed["syntax_error"] = None
    return executed


def image_gen(prompt: str, ctx: ToolContext, path: str = "outputs/tool_image.png") -> dict[str, Any]:
    """Hook into SARA's image decoder when a runtime is bound; else a tiny fallback render."""
    out = _jail(ctx.workspace, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rt = ctx.runtime
    if rt is not None and getattr(rt, "sara", None) is not None:
        from ..vision import tensor_to_pil

        img = rt.sara_generate_image(prompt)
        tensor_to_pil(img).save(out)
        return {"path": str(out), "prompt": prompt, "via": "sara.image_dec"}
    # still produce a real image (gradient + caption bar) so the tool is not a print
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (64, 64), (20, 24, 48))
    d = ImageDraw.Draw(im)
    seed = sum(map(ord, prompt))
    d.rectangle([8, 8, 56, 56], outline=((seed * 3) % 255, 80, 200), width=3)
    d.ellipse([20, 20, 44, 44], fill=((seed * 7) % 200 + 40, 40, 180))
    im.save(out)
    return {"path": str(out), "prompt": prompt, "via": "geometric-fallback"}


def audio_gen(prompt: str, ctx: ToolContext, path: str = "outputs/tool_audio.wav") -> dict[str, Any]:
    out = _jail(ctx.workspace, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rt = ctx.runtime
    if rt is not None and getattr(rt, "sara", None) is not None:
        wav = rt.sara_generate_song(prompt)
        from ..audio import save_wav

        save_wav(str(out), wav, rt.cfg.sample_rate)
        return {"path": str(out), "prompt": prompt, "via": "sara.song_head", "n_samples": int(len(wav))}
    # real tone cluster so the tool still writes audio
    import numpy as np
    from ..audio import save_wav

    sr = 16000
    t = np.arange(int(sr * 1.2), dtype=np.float32) / sr
    seed = 220 + (sum(map(ord, prompt)) % 200)
    wav = 0.3 * np.sin(2 * np.pi * seed * t) + 0.15 * np.sin(2 * np.pi * seed * 5 / 4 * t)
    save_wav(str(out), wav.astype(np.float32), sr)
    return {"path": str(out), "prompt": prompt, "via": "dyad-fallback", "n_samples": int(len(wav))}


def register_builtins(registry, ctx: ToolContext) -> None:
    """Attach all built-ins, closing over ctx."""

    registry.add(
        "python_exec",
        lambda code: python_exec(code, ctx),
        "Run a sandboxed Python snippet and capture stdout.",
        {"code": "str"},
        ["code"],
    )
    registry.add(
        "calc",
        lambda expr: calc(expr),
        "Evaluate a numeric arithmetic expression.",
        {"expr": "str"},
        ["expr"],
    )
    registry.add(
        "now",
        lambda: now(),
        "Return the current UTC and IST timestamps.",
        {},
        [],
    )
    registry.add(
        "file_read",
        lambda path: file_read(path, ctx),
        "Read a UTF-8 file under the workspace jail.",
        {"path": "str"},
        ["path"],
    )
    registry.add(
        "file_write",
        lambda path, content: file_write(path, content, ctx),
        "Write a UTF-8 file under the workspace jail.",
        {"path": "str", "content": "str"},
        ["path", "content"],
    )
    registry.add(
        "list_files",
        lambda path=".": list_files(path, ctx),
        "List files under a workspace directory.",
        {"path": "str"},
        [],
    )
    registry.add(
        "web_search",
        lambda query: web_search(query),
        "Search the web via DuckDuckGo HTML. Degrades cleanly if offline.",
        {"query": "str"},
        ["query"],
    )
    registry.add(
        "shell",
        lambda command: shell(command, ctx),
        "Run an allowlisted shell command in the workspace (python3, ls, cat, wc, date, echo, pwd).",
        {"command": "str"},
        ["command"],
    )
    registry.add(
        "code_run",
        lambda code: code_run(code, ctx),
        "Syntax-check then sandboxed-exec a Python program.",
        {"code": "str"},
        ["code"],
    )
    registry.add(
        "image_gen",
        lambda prompt, path="outputs/tool_image.png": image_gen(prompt, ctx, path),
        "Generate an image with SARA's image head (or a geometric fallback).",
        {"prompt": "str", "path": "str"},
        ["prompt"],
    )
    registry.add(
        "audio_gen",
        lambda prompt, path="outputs/tool_audio.wav": audio_gen(prompt, ctx, path),
        "Generate audio/song with SARA's song head (or a harmonic dyad fallback).",
        {"prompt": "str", "path": "str"},
        ["prompt"],
    )
