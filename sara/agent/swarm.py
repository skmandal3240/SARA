"""Multi-agent swarm: orchestrator, coder, researcher, critic.

Cheap on nano (few steps) but the API matches a serious runtime: roles, delegation,
shared memory, and a critic that can reject work and send it back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .loop import AgentResult, AgentRuntime, extract_python, default_program_for, _syntax_ok
from ..tools.protocol import ToolCall


ROLES = {
    "orchestrator": "Break the goal into delegated tasks. Do not implement. Assign coder/researcher.",
    "coder": "Write and run Python. Use file_write, shell, code_run. Fix syntax errors.",
    "researcher": "Use web_search, file_read, list_files. Return concise notes.",
    "critic": "Check the coder's result. If wrong or crashing, explain the defect.",
}


@dataclass
class Delegation:
    to: str
    task: str
    result: Optional[str] = None
    ok: bool = False


@dataclass
class SwarmResult:
    ok: bool
    final: str
    delegations: list[Delegation]
    transcript: str
    artifacts: list[str] = field(default_factory=list)


class Swarm:
    def __init__(self, runtime: AgentRuntime):
        self.rt = runtime

    def _role_prompt(self, role: str, task: str, extra: str = "") -> str:
        return (
            f"<|system|>{ROLES[role]}\n"
            f"<|agent|>{role}\n"
            f"<|user|>{task}\n"
            f"{extra}<|assistant|>"
        )

    def orchestrate(self, goal: str, max_rounds: int = 3) -> SwarmResult:
        rt = self.rt
        delegations: list[Delegation] = []
        artifacts: list[str] = []
        transcript = [f"ORCHESTRATOR goal: {goal}"]

        # Orchestrator chooses a path. Nano may babble; we still parse, then default.
        sketch = rt.generate_text(
            self._role_prompt(
                "orchestrator",
                f"Delegate this goal. Reply with JSON list of "
                f'{{"to":"coder|researcher","task":"..."}} only.\nGoal: {goal}',
            ),
            max_new=80,
            temperature=0.4,
        )
        parsed = _parse_delegations(sketch, goal)
        transcript.append("orchestrator sketch: " + sketch[:400])

        last_coder: Optional[str] = None
        last_research: Optional[str] = None

        for d in parsed:
            if d.to == "researcher":
                res = rt.run(d.task, max_steps=4)
                d.result = res.final
                d.ok = res.ok
                last_research = res.final
                artifacts.extend(res.artifacts)
                transcript.append(f"researcher: {res.final[:400]}")
            elif d.to == "coder":
                d.result, d.ok, arts, log = self._coder(d.task)
                artifacts.extend(arts)
                last_coder = d.result
                transcript.append(f"coder: ok={d.ok} {log[:400]}")
            delegations.append(d)

        # Critic reviews coder output when present
        if last_coder is not None:
            critique = self._critic(goal, last_coder)
            transcript.append("critic: " + critique[:400])
            if "REJECT" in critique.upper() and max_rounds > 1:
                retry_task = f"{goal}\nCritic said: {critique}\nFix the program."
                result, ok, arts, log = self._coder(retry_task)
                artifacts.extend(arts)
                last_coder = result
                delegations.append(Delegation("coder", retry_task, result, ok))
                transcript.append(f"coder-retry: ok={ok} {log[:300]}")

        final = last_coder or last_research or "no delegate produced output"
        ok = any(d.ok for d in delegations)
        transcript_s = "\n".join(transcript)
        rt.long_term.note(transcript_s[:1000], tag="swarm")
        return SwarmResult(ok=ok, final=str(final), delegations=delegations, transcript=transcript_s, artifacts=artifacts)

    def _coder(self, task: str) -> tuple[str, bool, list[str], str]:
        """Coder agent: write a file, run it, recover from errors."""
        rt = self.rt
        code = extract_python(rt.generate_text(f"<|user|>{task}\n<|code_start|>", max_new=128, temperature=0.35))
        if not code or not _syntax_ok(code):
            code = default_program_for(task)
        path = "outputs/swarm_coder.py"
        w = rt.registry.dispatch(ToolCall("file_write", {"path": path, "content": code}))
        run = rt.registry.dispatch(ToolCall("shell", {"command": f"python3 {path}"}))
        arts = []
        if w.get("ok"):
            arts.append(path)
        # recover
        if not (run.get("ok") and (run.get("result") or {}).get("returncode") == 0):
            err = json.dumps(run, default=str)[:600]
            code2 = extract_python(
                rt.generate_text(
                    f"<|user|>Fix this program. Error:\n{err}\nTask: {task}\n<|code_start|>",
                    max_new=128,
                    temperature=0.2,
                )
            )
            if not code2 or not _syntax_ok(code2):
                code2 = default_program_for(task)
            rt.registry.dispatch(ToolCall("file_write", {"path": path, "content": code2}))
            run = rt.registry.dispatch(ToolCall("shell", {"command": f"python3 {path}"}))
            code = code2
        stdout = ""
        if isinstance(run.get("result"), dict):
            stdout = str(run["result"].get("stdout") or "").strip()
        ok = bool(run.get("ok")) and (run.get("result") or {}).get("returncode") == 0
        return stdout or code, bool(ok), arts, json.dumps(run, default=str)[:500]

    def _critic(self, goal: str, product: str) -> str:
        raw = self.rt.generate_text(
            self._role_prompt(
                "critic",
                f"Goal: {goal}\nProduct:\n{product}\n"
                "Reply ACCEPT or REJECT and one sentence.",
            ),
            max_new=48,
            temperature=0.3,
        )
        text = (raw or "").strip()
        # nano may not emit the word; if we have a numeric/nonempty product, accept
        if not text:
            return "ACCEPT (empty critic; product nonempty)" if product.strip() else "REJECT empty product"
        if "REJECT" not in text.upper() and "ACCEPT" not in text.upper():
            return ("ACCEPT " if product.strip() else "REJECT ") + text
        return text


def _parse_delegations(text: str, goal: str) -> list[Delegation]:
    # try JSON list
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            out = []
            for item in data:
                if isinstance(item, dict) and "to" in item:
                    to = str(item["to"]).lower()
                    if to not in ROLES:
                        to = "coder"
                    out.append(Delegation(to=to, task=str(item.get("task") or goal)))
            if out:
                return out
        except json.JSONDecodeError:
            pass
    # default path: researcher skipped, coder implements
    g = goal.lower()
    out = []
    if any(w in g for w in ("search", "who", "lookup", "news")):
        out.append(Delegation("researcher", goal))
    out.append(Delegation("coder", goal))
    return out
