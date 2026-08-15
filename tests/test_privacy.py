"""Grants (default-deny), vault, audit, local learn, typed memory."""

from pathlib import Path

from sara.privacy.grants import GrantLedger, GrantError, CAPABILITIES
from sara.privacy.vault import Vault
from sara.privacy.audit import AuditLog, model_hash
from sara.privacy.learn import LocalLearner
from sara.agent.memory import LongTermMemory
from sara.agent.loop import AgentRuntime
from sara.tools.protocol import ToolCall
from sara.agent.skills import load_skill


def test_default_deny_preview_approve(tmp_path):
    g = GrantLedger()
    for cap in CAPABILITIES:
        assert g.allowed(cap) is False
    try:
        g.approve("camera")
        raise AssertionError("approve without preview")
    except GrantError:
        pass
    g.preview("camera", "owner CCTV")
    g.approve("camera")
    assert g.allowed("camera") is True
    assert g.allowed("cloud") is False
    g.require("camera")
    try:
        g.require("cloud")
        raise AssertionError("cloud should deny")
    except GrantError:
        pass
    g.revoke("camera")
    assert g.allowed("camera") is False


def test_high_risk_tools_need_grant(tmp_path):
    g = GrantLedger()
    rt = AgentRuntime(tmp_path, model=None, tokenizer=None, grants=g, max_steps=2)
    denied = rt.registry.dispatch(ToolCall("file_write", {"path": "x.py", "content": "print(1)"}))
    assert denied["ok"] is False and denied.get("grant_denied")
    denied_web = rt.registry.dispatch(ToolCall("web_search", {"query": "x"}))
    assert denied_web["ok"] is False
    # calc is not high-risk
    ok = rt.registry.dispatch(ToolCall("calc", {"expr": "2+2"}))
    assert ok["ok"] is True
    g.preview("files", "write a demo file")
    g.approve("files")
    wrote = rt.registry.dispatch(ToolCall("file_write", {"path": "x.py", "content": "print(1)"}))
    assert wrote["ok"] is True


def test_vault_roundtrip_and_audit(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    v = Vault(tmp_path / "vault.jsonl", key_path=tmp_path / "vault.key", audit=log)
    v.put({"secret": "lan-only"}, key="k1")
    raw = (tmp_path / "vault.jsonl").read_text(encoding="utf-8")
    assert "lan-only" not in raw
    got = v.get("k1")
    assert got is not None
    assert got["record"]["secret"] == "lan-only"
    events = log.read()
    assert any(e["event"] == "vault_write" for e in events)


def test_signed_inference_log(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    rec = log.inference(event="see", model=None, quant="int8", adapter_id="local-v0", placement="local")
    assert rec["model_hash"]
    assert rec["quant"] == "int8"
    assert rec["adapter_id"] == "local-v0"
    assert rec["placement"] == "local"


def test_local_learn_does_not_upload(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    lr = LocalLearner(tmp_path / "pairs.json", audit=log)
    lr.correct("caption this", "a red circle", actual="a blob")
    assert len(lr.pairs()) == 1
    ad = lr.export_adapter()
    assert ad["uploaded"] is False
    assert "a red circle" not in str(ad["delta"])
    try:
        lr.import_federated_average([{"frames": [0]}])
        raise AssertionError("raw frames must be refused")
    except ValueError:
        pass
    ok = lr.import_federated_average([{"delta": [0.1]}])
    assert ok["ok"]


def test_typed_facts_forget_profile(tmp_path):
    mem = LongTermMemory(tmp_path / "mem.json")
    mem.remember("city", "Pune", kind="profile")
    mem.remember("last_alert", "gate open", kind="fact")
    inj = mem.profile_inject()
    assert "Pune" in inj
    assert mem.forget("last_alert") == 1
    assert all(f["key"] != "last_alert" for f in mem.facts())
    # forgotten facts stay out of recall
    rec = mem.recall("gate")
    assert all(r.get("key") != "last_alert" for r in rec)


def test_skill_contract_see():
    root = Path(__file__).resolve().parents[1]
    sk = load_skill(root / "docs" / "skills" / "see.md")
    assert sk.name == "see"
    assert sk.need_grant == "camera"
    assert sk.steps
