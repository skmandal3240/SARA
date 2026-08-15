"""Device profiles, runtime scheduler, INT8, paging, in-process mesh."""

from pathlib import Path

import torch

from sara.config import SARAConfig
from sara.model import SARA
from sara.edge.profile import DeviceProfile, PROFILES, lan_host_allowed
from sara.edge.runtime import SARARuntime
from sara.edge.quantize import quantize_linear_int8, count_quantized_linears, export_onnx_stub, bench_kernels
from sara.edge.paging import LayerPager
from sara.edge.mesh import Mesh, MeshPeer, TaskDAG, TaskNode, PlacementError
from sara.privacy.grants import GrantLedger


def _tiny():
    cfg = SARAConfig.tiny()
    cfg.vocab_size = 128
    cfg.max_seq_len = 48
    return SARA(cfg), cfg


def test_all_sku_profiles():
    for name in PROFILES:
        p = DeviceProfile.named(name)
        assert p.name == name
        assert p.ram_mb > 0 and p.tops > 0
    assert DeviceProfile.named("cctv").cloud_allowed is False
    assert DeviceProfile.named("phone").cloud_allowed is False
    assert DeviceProfile.named("server").cloud_allowed is True


def test_yaml_profiles():
    root = Path(__file__).resolve().parents[1] / "configs"
    c = DeviceProfile.from_yaml(root / "edge_cctv.yaml")
    assert c.name == "cctv" and c.cloud_allowed is False and c.owner_operated
    ph = DeviceProfile.from_yaml(root / "edge_phone.yaml")
    assert ph.name == "phone" and "code" in ph.modalities
    d = DeviceProfile.from_yaml(root / "edge_drone.yaml")
    assert d.name == "drone" and d.battery


def test_place_local_when_headroom():
    p = DeviceProfile.named("cctv")
    rt = SARARuntime(p, grants=GrantLedger())
    pl = rt.place("see", ram_need=32, tops_need=0.1)
    assert pl.where == "local"


def test_place_deny_without_cloud_grant():
    p = DeviceProfile.named("cctv")
    p.used_ram_mb = p.ram_mb
    p.mesh_allowed = False
    g = GrantLedger()
    rt = SARARuntime(p, grants=g)
    pl = rt.place("see", ram_need=64, tops_need=1.0)
    assert pl.where == "deny"
    assert "cloud" in pl.reason.lower() or "not granted" in pl.reason.lower()


def test_place_cloud_when_granted():
    p = DeviceProfile.named("server")
    p.used_ram_mb = p.ram_mb
    p.mesh_allowed = False
    g = GrantLedger()
    g.preview("cloud", "reviewer overflow test")
    g.approve("cloud")
    rt = SARARuntime(p, grants=g)
    pl = rt.place("train", ram_need=100, tops_need=1.0)
    assert pl.where == "cloud"


def test_place_mesh_overflow():
    cam = DeviceProfile.named("cctv")
    cam.used_ram_mb = cam.ram_mb  # local full
    hub = DeviceProfile.named("laptop")
    g = GrantLedger()
    g.preview("mesh", "overflow")
    g.approve("mesh")
    mesh = Mesh()
    mesh.join(MeshPeer("hub", hub, g))
    rt = SARARuntime(cam, grants=g, mesh=mesh)
    pl = rt.place("caption", ram_need=64, tops_need=0.2)
    assert pl.where == "mesh"
    assert pl.peer == "hub"


def test_ssrf_lan_allowlist():
    p = DeviceProfile.named("cctv")
    assert lan_host_allowed("127.0.0.1", p.lan_allowlist)
    assert lan_host_allowed("192.168.1.20", p.lan_allowlist)
    assert not lan_host_allowed("8.8.8.8", p.lan_allowlist)
    assert not lan_host_allowed("evil.example", p.lan_allowlist)
    rt = SARARuntime(p)
    try:
        rt.assert_lan("10.0.0.5")
    except PermissionError:
        raise AssertionError("LAN should pass")
    try:
        rt.assert_lan("1.1.1.1")
        raise AssertionError("WAN should fail")
    except PermissionError:
        pass


def test_int8_dynamic_linear():
    model, cfg = _tiny()
    q = quantize_linear_int8(model)
    assert count_quantized_linears(q) >= 1
    x = torch.randint(1, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = q(x)
    assert out["logits"].shape[-1] == cfg.vocab_size
    stub = export_onnx_stub(q)
    assert stub["ok"] is False


def test_paged_forward_one_block_at_a_time():
    model, cfg = _tiny()
    model.eval()
    x = torch.randint(1, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = model(x)
    pager = LayerPager(model, device="cpu")
    paged = pager.forward(x)
    assert paged["logits"].shape == full["logits"].shape
    assert pager.max_resident == 1
    assert pager.loads == cfg.n_layers
    assert torch.isfinite(paged["logits"]).all()


def test_mesh_two_node_dag_split():
    cam = DeviceProfile.named("cctv")
    hub = DeviceProfile.named("laptop")
    mesh = Mesh()
    mesh.join(MeshPeer("cam", cam))
    mesh.join(MeshPeer("hub", hub))
    dag = TaskDAG(
        nodes=[
            TaskNode("n1", "see_frame", payload={"jpeg": b"raw"}, ram_mb=64, tops=0.2,
                     modalities=["vision"], requires_raw=True, tags=["camera"]),
            TaskNode("n2", "caption", payload={"embedding": [0.1]}, ram_mb=64, tops=0.2,
                     modalities=["text"], tags=["can_exec"]),
        ],
        edges=[("n1", "n2")],
    )
    mapping = mesh.place(dag, share_raw=False)
    assert mapping["n1"] == "cam"
    assert mapping["n2"] == "hub"
    stripped = mesh.sanitize_payload(dag.nodes[0], share_raw=False)
    assert "jpeg" not in stripped


def test_bench_kernels_cpu():
    b = bench_kernels(dim=64, seq=16, steps=5)
    assert "rmsnorm_ms" in b and b["rmsnorm_ms"] >= 0
