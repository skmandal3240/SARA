"""Device SKU profiles — one runtime, YAML/JSON caps per product."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PROFILES = ("cctv", "drone", "phone", "laptop", "tv", "robot", "ev", "server", "db")

# Owner-operated cameras talk on the LAN only. SSRF is deny-by-default.
_LAN = ("127.0.0.1", "::1", "10.0.0.0/8", "192.168.0.0/16", "172.16.0.0/12")


@dataclass
class DeviceProfile:
    name: str
    ram_mb: int
    tops: float
    cameras: int = 0
    radios: list[str] = field(default_factory=list)
    mesh_allowed: bool = True
    cloud_allowed: bool = False
    modalities: list[str] = field(default_factory=list)
    policy_tags: list[str] = field(default_factory=list)
    npu: bool = False
    battery: bool = False
    quantize: str = "int8"
    paging: bool = False
    used_ram_mb: int = 128
    used_tops: float = 0.0
    owner_operated: bool = False
    lan_allowlist: list[str] = field(default_factory=lambda: list(_LAN))
    ssrf_deny: bool = True
    role: str = "node"  # sensor | hub | node

    def has_headroom(self, ram_need: int = 0, tops_need: float = 0.0, frac: float = 0.85) -> bool:
        return (self.used_ram_mb + ram_need) <= self.ram_mb * frac and (
            self.used_tops + tops_need
        ) <= max(self.tops, 1e-6) * frac

    def reserve(self, ram_need: int = 0, tops_need: float = 0.0) -> None:
        self.used_ram_mb += ram_need
        self.used_tops += tops_need

    def release(self, ram_need: int = 0, tops_need: float = 0.0) -> None:
        self.used_ram_mb = max(0, self.used_ram_mb - ram_need)
        self.used_tops = max(0.0, self.used_tops - tops_need)

    def advertise(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "free_ram_mb": max(0, self.ram_mb - self.used_ram_mb),
            "free_tops": max(0.0, self.tops - self.used_tops),
            "modalities": list(self.modalities),
            "policy_tags": list(self.policy_tags),
            "mesh": self.mesh_allowed,
            "cloud": self.cloud_allowed,
            "role": self.role,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceProfile":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeviceProfile":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name = data.pop("profile", None) or data.get("name")
        if name and name in _PRESETS:
            base = _PRESETS[name]()
            for k, v in data.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            return base
        if name:
            data = {**data, "name": name}
        return cls.from_dict(data)

    @classmethod
    def named(cls, name: str) -> "DeviceProfile":
        if name not in _PRESETS:
            raise KeyError(f"unknown profile {name!r}; known {list(_PRESETS)}")
        return _PRESETS[name]()


def lan_host_allowed(host: str, allowlist: list[str]) -> bool:
    """SSRF deny-by-default: only loopback / RFC1918 (or explicit allowlist)."""
    import ipaddress

    host = (host or "").split("%")[0].strip().lower()
    if not host:
        return False
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    for cidr in allowlist:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
            if ip in net:
                return True
        except ValueError:
            if host == cidr.lower():
                return True
    return False


def _cctv() -> DeviceProfile:
    return DeviceProfile(
        name="cctv",
        ram_mb=2048,
        tops=4.0,
        cameras=4,
        radios=["ethernet"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["vision", "text"],
        policy_tags=["camera"],
        npu=True,
        quantize="int8",
        paging=True,
        owner_operated=True,
        role="sensor",
        used_ram_mb=256,
    )


def _drone() -> DeviceProfile:
    return DeviceProfile(
        name="drone",
        ram_mb=2048,
        tops=3.0,
        cameras=1,
        radios=["radio", "wifi"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["vision", "text", "audio"],
        policy_tags=["camera"],
        npu=True,
        battery=True,
        quantize="int8",
        paging=True,
        owner_operated=True,
        role="sensor",
        used_ram_mb=256,
    )


def _phone() -> DeviceProfile:
    return DeviceProfile(
        name="phone",
        ram_mb=6144,
        tops=8.0,
        cameras=2,
        radios=["wifi", "lte"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["text", "vision", "audio", "code"],
        policy_tags=["camera", "vault", "can_exec"],
        npu=True,
        battery=True,
        quantize="int8",
        paging=True,
        role="sensor",
        used_ram_mb=512,
    )


def _laptop() -> DeviceProfile:
    return DeviceProfile(
        name="laptop",
        ram_mb=16384,
        tops=16.0,
        cameras=1,
        radios=["wifi"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["text", "vision", "audio", "code"],
        policy_tags=["vault", "can_exec"],
        npu=False,
        quantize="int8",
        role="hub",
        used_ram_mb=1024,
    )


def _tv() -> DeviceProfile:
    return DeviceProfile(
        name="tv",
        ram_mb=3072,
        tops=2.0,
        cameras=0,
        radios=["wifi"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["text", "audio", "vision"],
        policy_tags=[],
        npu=True,
        quantize="int8",
        paging=True,
        role="node",
        used_ram_mb=384,
    )


def _robot() -> DeviceProfile:
    return DeviceProfile(
        name="robot",
        ram_mb=4096,
        tops=6.0,
        cameras=2,
        radios=["wifi"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["vision", "text", "audio"],
        policy_tags=["camera", "can_exec"],
        npu=True,
        battery=True,
        quantize="int8",
        paging=True,
        role="node",
        used_ram_mb=512,
    )


def _ev() -> DeviceProfile:
    return DeviceProfile(
        name="ev",
        ram_mb=4096,
        tops=8.0,
        cameras=4,
        radios=["can", "lte"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["vision", "text", "audio"],
        policy_tags=["camera", "vault"],
        npu=True,
        quantize="int8",
        paging=True,
        owner_operated=True,
        role="node",
        used_ram_mb=768,
    )


def _server() -> DeviceProfile:
    return DeviceProfile(
        name="server",
        ram_mb=65536,
        tops=64.0,
        cameras=0,
        radios=["ethernet"],
        mesh_allowed=True,
        cloud_allowed=True,
        modalities=["text", "vision", "audio", "code"],
        policy_tags=["can_exec", "vault"],
        npu=False,
        quantize="fp32",
        paging=False,
        role="hub",
        used_ram_mb=2048,
    )


def _db() -> DeviceProfile:
    return DeviceProfile(
        name="db",
        ram_mb=4096,
        tops=2.0,
        cameras=0,
        radios=["ethernet"],
        mesh_allowed=True,
        cloud_allowed=False,
        modalities=["text"],
        policy_tags=["vault"],
        npu=False,
        quantize="int8",
        role="node",
        used_ram_mb=256,
    )


_PRESETS = {
    "cctv": _cctv,
    "drone": _drone,
    "phone": _phone,
    "laptop": _laptop,
    "tv": _tv,
    "robot": _robot,
    "ev": _ev,
    "server": _server,
    "db": _db,
}
