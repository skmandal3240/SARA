#!/usr/bin/env python3
"""Phase A edge demos. CPU only. Must exit 0.

1. cctv profile — see-path with cloud denied
2. phone profile — agent with cloud denied
3. mesh — 2-node DAG across two in-process peers
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _tiny():
    from sara.config import SARAConfig
    from sara.model import SARA

    cfg = SARAConfig.tiny()
    cfg.vocab_size = 128
    cfg.max_seq_len = 48
    model = SARA(cfg)
    model.eval()
    return model, cfg


def demo_cctv() -> None:
    print("=== cctv see-path (cloud denied) ===")
    from sara.edge.profile import DeviceProfile
    from sara.edge.runtime import SARARuntime
    from sara.privacy.grants import GrantLedger
    from sara.privacy.audit import AuditLog
    from sara.vision import make_shape_image, pil_to_tensor

    model, cfg = _tiny()
    profile = DeviceProfile.from_yaml(ROOT / "configs" / "edge_cctv.yaml")
    grants = GrantLedger()
    grants.preview("camera", "owner-operated SARA Cam on the LAN")
    grants.approve("camera")
    # cloud stays default-deny
    assert grants.allowed("cloud") is False
    audit = AuditLog(OUT / "edge_audit.jsonl")
    rt = SARARuntime(
        profile, grants=grants, model=model, cfg=cfg, audit=audit, workspace=ROOT
    )
    pl = rt.place("see", ram_need=32, tops_need=0.1, require_modalities=["vision"])
    assert pl.where == "local", pl
    img = pil_to_tensor(make_shape_image("circle", "red", size=cfg.img_size), cfg.img_size).unsqueeze(0)
    rec = rt.see(img, prompt="describe this image")
    assert rec["placement"].where != "cloud"
    assert rec["cloud"] is False
    # overflowing without mesh/cloud must deny
    profile.used_ram_mb = profile.ram_mb
    profile.mesh_allowed = False
    deny = rt.place("see", ram_need=64, tops_need=1.0)
    assert deny.where == "deny", deny
    (OUT / "edge_cctv_caption.txt").write_text(str(rec["caption"])[:500], encoding="utf-8")
    print("placement", rec["placement"].where, "deny-when-full", deny.where)


def demo_phone() -> None:
    print("=== phone agent (cloud denied) ===")
    from sara.edge.profile import DeviceProfile
    from sara.edge.runtime import SARARuntime
    from sara.privacy.grants import GrantLedger
    from sara.privacy.audit import AuditLog

    profile = DeviceProfile.from_yaml(ROOT / "configs" / "edge_phone.yaml")
    grants = GrantLedger()
    grants.preview("files", "write and run a local calc")
    grants.approve("files")
    assert grants.allowed("cloud") is False
    audit = AuditLog(OUT / "edge_audit.jsonl")
    rt = SARARuntime(profile, grants=grants, audit=audit, workspace=ROOT)
    res, pl = rt.run_agent("What is 17 times 3?", max_steps=4)
    assert pl.where != "cloud"
    print("placement", pl.where, "final", res.final, "tools", res.tool_calls)
    (OUT / "edge_phone_final.txt").write_text(str(res.final), encoding="utf-8")
    # high-risk web still denied without cloud grant
    from sara.tools.protocol import ToolCall
    from sara.agent.loop import AgentRuntime

    agent = AgentRuntime(ROOT, grants=grants, audit=audit, max_steps=1)
    web = agent.registry.dispatch(ToolCall("web_search", {"query": "should not leave device"}))
    assert web.get("ok") is False
    print("web denied without cloud grant")


def demo_mesh() -> None:
    print("=== mesh 2-node DAG (in-process peers) ===")
    from sara.edge.profile import DeviceProfile
    from sara.edge.mesh import Mesh, MeshPeer, TaskDAG, TaskNode

    cam = DeviceProfile.named("cctv")
    hub = DeviceProfile.named("laptop")
    mesh = Mesh()
    mesh.join(MeshPeer("cam", cam))
    mesh.join(MeshPeer("hub", hub))
    dag = TaskDAG(
        nodes=[
            TaskNode(
                "see",
                "see_frame",
                payload={"jpeg": b"not-sent", "embedding": [0.2, 0.1]},
                ram_mb=48,
                tops=0.2,
                modalities=["vision"],
                requires_raw=True,
                tags=["camera"],
            ),
            TaskNode(
                "caption",
                "caption",
                payload={"embedding": [0.2, 0.1]},
                ram_mb=48,
                tops=0.2,
                modalities=["text"],
                tags=["can_exec"],
            ),
        ],
        edges=[("see", "caption")],
    )
    mapping = mesh.place(dag, share_raw=False)
    assert mapping["see"] == "cam"
    assert mapping["caption"] == "hub"
    stripped = mesh.sanitize_payload(dag.nodes[0], share_raw=False)
    assert "jpeg" not in stripped
    print("placed", mapping, "raw_stripped", stripped)
    (OUT / "edge_mesh.json").write_text(str(mapping), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    demo_cctv()
    print("PASS demo_cctv")
    demo_phone()
    print("PASS demo_phone")
    demo_mesh()
    print("PASS demo_mesh")
    print("EDGE DEMOS ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
