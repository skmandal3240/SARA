"""Goal decomposition: plan → steps → execute → verify.

On nano weights the LM may not emit a perfect plan, so the planner also has a
deterministic skeleton for well-known task shapes (write+run python, search+summarize)
while still asking the model to fill step payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    id: int
    intent: str
    tool: Optional[str] = None
    hint: str = ""
    done: bool = False
    result: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)

    def pending(self) -> list[Step]:
        return [s for s in self.steps if not s.done]

    def render(self) -> str:
        lines = [f"goal: {self.goal}"]
        for s in self.steps:
            mark = "x" if s.done else " "
            lines.append(f"  [{mark}] {s.id}. {s.intent} (tool={s.tool})")
        return "\n".join(lines)


_WRITE_RUN = re.compile(r"(write|create).*(python|script|program|file)|fibonacci|multi-step", re.I)
_SEARCH = re.compile(r"search|look up|who is|what is", re.I)
_CALC = re.compile(r"calculat|how much|plus|times \d|\d+\s*[+\-*x×/]\s*\d+", re.I)


def decompose(goal: str) -> Plan:
    g = goal.strip()
    steps: list[Step] = []
    if _WRITE_RUN.search(g):
        steps = [
            Step(1, "Write a Python program that solves the requested task", "file_write",
                 hint="Write a complete small .py file under outputs/."),
            Step(2, "Run the program and capture stdout", "shell",
                 hint="shell command: python3 <the file>"),
            Step(3, "If it failed, fix the file and re-run", "file_write",
                 hint="Use the error from the previous observation."),
            Step(4, "Report the numeric or textual result", None,
                 hint="Final answer only."),
        ]
    elif _CALC.search(g):
        steps = [
            Step(1, "Evaluate the expression", "calc"),
            Step(2, "Report the value", None),
        ]
    elif _SEARCH.search(g):
        steps = [
            Step(1, "Search the web for the query", "web_search"),
            Step(2, "Summarize findings as the final answer", None),
        ]
    else:
        steps = [
            Step(1, "Think about which tool helps", None),
            Step(2, "Use a tool if needed", None),
            Step(3, "Give the final answer", None),
        ]
    return Plan(goal=g, steps=steps)


def parse_plan_text(goal: str, text: str) -> Optional[Plan]:
    """Parse numbered steps from model-emitted <|plan|> text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    steps = []
    for ln in lines:
        m = re.match(r"(?:step\s*)?(\d+)[.)\:\-]\s*(.+)", ln, re.I)
        if not m:
            continue
        intent = m.group(2)
        tool = None
        tm = re.search(r"tool\s*=\s*([A-Za-z0-9_]+)", intent)
        if tm:
            tool = tm.group(1)
        steps.append(Step(id=int(m.group(1)), intent=intent, tool=tool))
    if not steps:
        return None
    return Plan(goal=goal, steps=steps)
