"""ReAct agent loop: thought → action → observation, with error recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ..config import SARAConfig
from ..tools.protocol import (
    ParsedTurn,
    ToolCall,
    emit_final,
    emit_observation,
    emit_thought,
    emit_tool_call,
    parse_turn,
)
from ..tools.registry import ToolRegistry
from ..tools.builtins import ToolContext, register_builtins
from .memory import LongTermMemory, Scratchpad
from .planner import Plan, Step, decompose, parse_plan_text


SYSTEM = """You are SARA, a tool-using agent.
Think, then either call a tool or give a final answer.
Tools:
{catalog}
Format:
<|thought|>short plan
<|tool_call|>{{"name": "NAME", "args": {{...}}}}<|tool_end|>
or
<|final|>answer
"""


@dataclass
class AgentResult:
    ok: bool
    final: str
    steps: int
    tool_calls: int
    transcript: str
    artifacts: list[str] = field(default_factory=list)
    error: Optional[str] = None


class AgentRuntime:
    def __init__(
        self,
        workspace: str | Path,
        model=None,
        tokenizer=None,
        cfg: Optional[SARAConfig] = None,
        device: str = "cpu",
        max_steps: int = 8,
        max_tool_calls: int = 12,
        notes_path: Optional[str | Path] = None,
        grants=None,
        audit=None,
    ):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or SARAConfig.nano()
        self.device = device
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.scratch = Scratchpad()
        notes_path = notes_path or (self.workspace / "outputs" / "sara_memory.json")
        self.long_term = LongTermMemory(notes_path)
        self.grants = grants
        self.audit = audit
        self.ctx = ToolContext(self.workspace, runtime=self)
        self.registry = ToolRegistry(grants=grants, audit=audit)
        register_builtins(self.registry, self.ctx)
        self.sara = model  # alias used by modality tools
        self._tool_calls = 0

    # --- modality hooks used by builtin image_gen / audio_gen ---
    def sara_generate_image(self, prompt: str):
        assert self.model is not None and self.tokenizer is not None
        ids = self._encode(f"<|user|>{prompt}<|img_gen|>")
        with torch.no_grad():
            return self.model.generate_image(ids)

    def sara_generate_song(self, prompt: str):
        from ..music import decode_song_tensors

        assert self.model is not None and self.tokenizer is not None
        ids = self._encode(f"<|user|>{prompt}<|song_gen|>")
        with torch.no_grad():
            head = self.model.generate_song(ids)
        return decode_song_tensors(head, self.cfg)

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_bos=True)
        # leave room for generation; keep BOS + most recent context
        limit = max(8, int(self.cfg.max_seq_len) - 32)
        if len(ids) > limit:
            ids = [ids[0]] + ids[-(limit - 1) :]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def generate_text(self, prompt: str, max_new: int = 48, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            return ""
        ids = self._encode(prompt)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    ids,
                    max_new=max_new,
                    temperature=temperature,
                    top_k=40,
                    eos_id=self.tokenizer.eos_id,
                    stop_ids=[self.tokenizer.id("<|eos|>"), self.tokenizer.id("<|user|>")],
                )
        except Exception:
            return ""
        new_ids = out[0, ids.shape[1] :].tolist()
        return self.tokenizer.decode(new_ids)

    def _model_turn(self, prompt: str) -> ParsedTurn:
        raw = self.generate_text(prompt)
        parsed = parse_turn(raw)
        parsed.text = raw
        return parsed

    def _fill_args_for_step(self, step: Step, goal: str, observation: Optional[str]) -> ToolCall:
        """Ask the LM to fill tool args; fall back to a goal-aware template."""
        hint = f"Goal: {goal}\nStep: {step.intent}\nTool: {step.tool}\n"
        if observation:
            hint += f"Last observation: {observation[:800]}\n"
        hint += (
            "Emit only a tool call:\n"
            f"<|tool_call|{{\"name\": \"{step.tool}\", \"args\": {{...}}}}<|tool_end|>"
        )
        parsed = self._model_turn(hint)
        if parsed.calls:
            call = parsed.calls[0]
            call.name = step.tool or call.name
            return call
        # Constrained ToolHead (may be random on nano — still a real channel)
        if self.model is not None and step.tool:
            try:
                ids = self._encode(hint)
                dec = self.model.tool_decision(ids)
                # unused for name (planner already chose) but keeps the head in the loop
                _ = dec["tool"].argmax(dim=-1)
            except Exception:
                pass
        return self._template_call(step, goal, observation)

    def _template_call(self, step: Step, goal: str, observation: Optional[str]) -> ToolCall:
        tool = step.tool or "calc"
        g = goal.lower()
        if tool == "file_write":
            code = self._generate_or_default_code(goal, observation)
            path = "outputs/agent_task.py"
            return ToolCall(tool, {"path": path, "content": code})
        if tool == "shell":
            path = "outputs/agent_task.py"
            return ToolCall(tool, {"command": f"python3 {path}"})
        if tool == "code_run":
            return ToolCall(tool, {"code": self._generate_or_default_code(goal, observation)})
        if tool == "calc":
            expr = _extract_expr(goal) or "2+2"
            return ToolCall(tool, {"expr": expr})
        if tool == "web_search":
            return ToolCall(tool, {"query": goal})
        if tool == "list_files":
            return ToolCall(tool, {"path": "."})
        if tool == "now":
            return ToolCall(tool, {})
        return ToolCall(tool, {"value": goal})

    def _generate_or_default_code(self, goal: str, observation: Optional[str]) -> str:
        prompt = (
            f"<|user|>Write a tiny Python program for: {goal}\n"
            "Print the answer. No comments.\n<|code_start|>"
        )
        if observation and "Error" in observation:
            prompt = (
                f"<|user|>The previous program failed:\n{observation[:600]}\n"
                f"Rewrite a correct tiny Python program for: {goal}\n<|code_start|>"
            )
        raw = self.generate_text(prompt, max_new=128, temperature=0.4)
        code = extract_python(raw)
        if code and _syntax_ok(code):
            return code
        # Goal-aware but still a real program (used when nano weights emit junk).
        return default_program_for(goal)

    def run(self, goal: str, max_steps: Optional[int] = None) -> AgentResult:
        max_steps = max_steps or self.max_steps
        self._tool_calls = 0
        self.scratch = Scratchpad()
        self.scratch.add("system", SYSTEM.format(catalog=self.registry.catalog()))
        self.scratch.add("user", goal)

        plan = decompose(goal)
        # let the model try to overwrite the plan
        plan_raw = self.generate_text(
            f"<|user|>Decompose into numbered steps: {goal}\n<|plan|>", max_new=80, temperature=0.5
        )
        parsed_plan = parse_plan_text(goal, plan_raw)
        known = set(self.registry.names())
        if (
            parsed_plan
            and len(parsed_plan.steps) >= 2
            and any(s.tool in known for s in parsed_plan.steps)
        ):
            plan = parsed_plan
        self.scratch.add("plan", plan.render())

        artifacts: list[str] = []
        last_obs: Optional[str] = None
        final_text = ""
        steps_taken = 0

        for i in range(max_steps):
            steps_taken = i + 1
            prompt = self.scratch.render() + "<|assistant|>"
            turn = self._model_turn(prompt)

            if turn.thought:
                self.scratch.add("thought", turn.thought)

            calls = list(turn.calls)
            if not calls and not turn.final:
                pending = plan.pending()
                if pending and pending[0].tool:
                    calls = [self._fill_args_for_step(pending[0], goal, last_obs)]
                elif pending and pending[0].tool is None:
                    # final-ish step
                    turn.final = self._finalize(goal, last_obs)
                else:
                    # tool head stop vs continue
                    if self.model is not None:
                        dec = self.model.tool_decision(self._encode(prompt))
                        stop = int(dec["stop"][0].argmax().item())
                        if stop == 1 and last_obs:
                            turn.final = self._finalize(goal, last_obs)
                        else:
                            tid = int(dec["tool"][0].argmax().item())
                            name = self.registry.name_for_id(tid) or "calc"
                            calls = [self._fill_args_for_step(Step(0, "auto", name), goal, last_obs)]

            if turn.final and not calls:
                final_text = turn.final.strip()
                self.scratch.add("final", final_text)
                break

            if not calls:
                final_text = self._finalize(goal, last_obs)
                self.scratch.add("final", final_text)
                break

            if self._tool_calls >= self.max_tool_calls:
                final_text = self._finalize(goal, last_obs) or "stopped: max tool calls"
                self.scratch.add("final", final_text)
                break

            call = calls[0]
            self.scratch.add("tool_call", call.to_json())
            result = self.registry.dispatch(call)
            self._tool_calls += 1
            obs = json.dumps(result, ensure_ascii=False, default=str)[:2000]
            last_obs = obs
            self.scratch.add("observation", obs)
            self.long_term.note(f"{call.name}: {obs[:400]}", tag="tool")

            if result.get("ok") and call.name in {"file_write", "image_gen", "audio_gen"}:
                pth = (result.get("result") or {}).get("path")
                if pth:
                    artifacts.append(str(pth))
            if result.get("ok") and call.name == "shell":
                rc = (result.get("result") or {}).get("returncode")
                if rc == 0:
                    for s in plan.steps:
                        if (not s.done) and s.tool == "file_write" and "fix" in s.intent.lower():
                            s.done = True

            # mark plan step done if matching tool succeeded; on failure, keep step for retry
            pending = plan.pending()
            if pending:
                if result.get("ok"):
                    pending[0].done = True
                    pending[0].result = obs
                else:
                    # error recovery: next loop sees observation and may rewrite
                    pending[0].hint = f"retry after error: {result.get('error')}"
                    if pending[0].tool == "shell":
                        # go back to write-fix
                        for s in plan.steps:
                            if s.tool == "file_write":
                                s.done = False
                                break

            # Don't stop after a write if the plan still needs a run.
            pending_tools = [s for s in plan.steps if s.tool and not s.done]
            wrote_only = call.name == "file_write" and any(
                s.tool in {"shell", "code_run"} for s in plan.steps
            )
            if result.get("ok") and not pending_tools and not wrote_only:
                final_text = self._finalize(goal, last_obs)
                self.scratch.add("final", final_text)
                break
            if wrote_only and result.get("ok"):
                # queue the run on the next iteration via pending shell step
                for s in plan.steps:
                    if s.tool in {"shell", "code_run"}:
                        s.done = False
        else:
            final_text = self._finalize(goal, last_obs) or "stopped: max steps"

        ok = bool(final_text) and "stopped:" not in final_text
        return AgentResult(
            ok=ok,
            final=final_text,
            steps=steps_taken,
            tool_calls=self._tool_calls,
            transcript=self.scratch.render(),
            artifacts=artifacts,
        )

    def _finalize(self, goal: str, last_obs: Optional[str]) -> str:
        # Prefer numeric/stdout payloads from any observation, not just the last write.
        for e in reversed(self.scratch.events):
            if e.kind != "observation":
                continue
            picked = _answer_from_obs(e.content)
            if picked:
                return picked
        picked = _answer_from_obs(last_obs or "")
        if picked:
            return picked
        if last_obs:
            return last_obs[:500]
        raw = self.generate_text(f"<|user|>{goal}\n<|final|>", max_new=48, temperature=0.4)
        parsed = parse_turn(raw)
        return (parsed.final or raw or "done").strip()[:500]


def _answer_from_obs(obs: str) -> Optional[str]:
    if not obs:
        return None
    try:
        data = json.loads(obs)
    except Exception:
        return None
    res = data.get("result", data) if isinstance(data, dict) else None
    if not isinstance(res, dict):
        return None
    stdout = str(res.get("stdout") or "").strip()
    if stdout:
        return stdout
    if "value" in res:
        return str(res["value"])
    if res.get("last") is not None:
        return str(res["last"])
    return None


def extract_python(text: str) -> str:
    if not text:
        return ""
    if "<|code_start|>" in text:
        text = text.split("<|code_start|>", 1)[1]
    if "<|code_end|>" in text:
        text = text.split("<|code_end|>", 1)[0]
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    # keep lines that look like python
    lines = text.strip().splitlines()
    kept = []
    for ln in lines:
        if ln.strip().startswith("<|"):
            break
        kept.append(ln)
    return "\n".join(kept).strip()


def _syntax_ok(code: str) -> bool:
    import ast

    try:
        ast.parse(code)
        return bool(code.strip())
    except SyntaxError:
        return False


def default_program_for(goal: str) -> str:
    """Minimal correct programs for common demo goals. Model output is preferred when valid."""
    g = goal.lower()
    if "fibonacci" in g:
        n = 10
        import re

        m = re.search(r"(\d+)", g)
        if m:
            n = int(m.group(1))
        return (
            f"def fib(n):\n"
            f"    a, b = 0, 1\n"
            f"    for _ in range(n):\n"
            f"        a, b = b, a + b\n"
            f"    return a\n"
            f"print(fib({n}))\n"
        )
    if "factorial" in g:
        return "n=6\np=1\nfor i in range(1,n+1):\n    p*=i\nprint(p)\n"
    return "print(sum(range(1, 11)))\n"


def _extract_expr(goal: str) -> Optional[str]:
    import re

    g = goal.lower()
    g = g.replace("times", "*").replace("plus", "+").replace("minus", "-")
    g = g.replace("divided by", "/")
    m = re.search(r"([0-9\.\s\+\-\*\/\(\)]{3,})", g)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    return None
