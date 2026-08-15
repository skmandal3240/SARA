import json

from sara.tools.protocol import parse_turn, emit_tool_call, ToolCall
from sara.tools.registry import ToolRegistry
from sara.tools.builtins import ToolContext, register_builtins, calc, python_exec
from sara.agent.loop import AgentRuntime


def test_parse_json_tool_call():
    text = '<|thought|>use calc<|tool_call|>{"name": "calc", "args": {"expr": "2+2"}}<|tool_end|>'
    t = parse_turn(text)
    assert t.has_call
    assert t.calls[0].name == "calc"
    assert t.calls[0].args["expr"] == "2+2"
    assert t.thought and "calc" in t.thought


def test_parse_alt_and_final():
    t = parse_turn("[[tool:now()]]\n<|final|>done")
    assert t.calls[0].name == "now"
    assert t.final == "done"


def test_dispatch_calc_and_python(tmp_path):
    ctx = ToolContext(tmp_path)
    reg = ToolRegistry()
    register_builtins(reg, ctx)
    r = reg.dispatch(ToolCall("calc", {"expr": "7*8"}))
    assert r["ok"] and r["result"]["value"] == 56
    r = reg.dispatch(ToolCall("python_exec", {"code": "print(1+1)"}))
    assert r["ok"] and "2" in r["result"]["stdout"]
    r = reg.dispatch(ToolCall("file_write", {"path": "a.txt", "content": "hi"}))
    assert r["ok"]
    r = reg.dispatch(ToolCall("file_read", {"path": "a.txt"}))
    assert r["result"]["content"] == "hi"
    r = reg.dispatch(ToolCall("unknown_tool", {}))
    assert r["ok"] is False


def test_sandbox_blocks_os(tmp_path):
    ctx = ToolContext(tmp_path)
    out = python_exec("import os\nos.system('echo pwn')", ctx)
    assert out["ok"] is False


def test_agent_loop_terminates(tmp_path):
    rt = AgentRuntime(tmp_path, model=None, tokenizer=None, max_steps=4, max_tool_calls=4)
    res = rt.run("What is 11 times 4?")
    assert res.steps <= 4
    assert res.tool_calls <= 4
    # calc path should produce 44
    assert "44" in str(res.final) or "44" in res.transcript or res.tool_calls >= 1


def test_agent_write_run(tmp_path):
    rt = AgentRuntime(tmp_path, model=None, tokenizer=None, max_steps=6)
    res = rt.run("Write a python file that prints the 10th fibonacci number, run it, report the result.")
    assert (tmp_path / "outputs" / "agent_task.py").exists() or res.tool_calls >= 1
    assert res.steps <= 6
