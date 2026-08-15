"""Markdown skill contracts. YAGNI: parse a skill before adding a new tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillContract:
    name: str
    when: str = ""
    need_grant: str = ""
    do_not: str = ""
    steps: list[str] = field(default_factory=list)
    path: str = ""


def load_skill(path: str | Path) -> SkillContract:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    name = p.stem
    when = need = do_not = ""
    steps: list[str] = []
    in_steps = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# "):
            name = line[2:].strip()
            in_steps = False
            continue
        low = line.lower()
        if low.startswith("when:"):
            when = line.split(":", 1)[1].strip()
        elif low.startswith("need_grant:"):
            need = line.split(":", 1)[1].strip()
        elif low.startswith("do_not:"):
            do_not = line.split(":", 1)[1].strip()
        elif line.lower() in {"## steps", "# steps"}:
            in_steps = True
        elif in_steps and line[:1].isdigit() and "." in line[:4]:
            steps.append(line.split(".", 1)[1].strip())
        elif in_steps and line.startswith("- "):
            steps.append(line[2:].strip())
    return SkillContract(name=name, when=when, need_grant=need, do_not=do_not, steps=steps, path=str(p))
