"""SARA Mesh v0 — in-process peer mesh + task DAG placement.

Not a fleet gossip network. Two Python objects in one process is a valid
Phase A proof. Payloads between peers are embeddings / tokens / adapters,
not raw camera/mic/vault files, unless the user granted `share_raw`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .profile import DeviceProfile

RAW_KINDS = frozenset({"see_frame", "raw_audio", "vault_file", "raw_image"})


@dataclass
class TaskNode:
    id: str
    kind: str
    payload: Any = None
    ram_mb: int = 32
    tops: float = 0.1
    modalities: list[str] = field(default_factory=list)
    requires_raw: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskDAG:
    nodes: list[TaskNode]
    edges: list[tuple[str, str]] = field(default_factory=list)

    def node_map(self) -> dict[str, TaskNode]:
        return {n.id: n for n in self.nodes}


class MeshPeer:
    def __init__(self, peer_id: str, profile: DeviceProfile, grants: Any = None):
        self.id = peer_id
        self.profile = profile
        self.grants = grants

    def capable(self, node: TaskNode, share_raw: bool) -> bool:
        if not self.profile.has_headroom(node.ram_mb, node.tops):
            return False
        tags = set(node.tags)
        have = set(self.profile.policy_tags)
        if "camera" in tags and "camera" not in have:
            return False
        if "can_exec" in tags and "can_exec" not in have:
            return False
        if "vault" in tags and "vault" not in have:
            return False
        if node.requires_raw and not share_raw and "camera" in tags and "camera" not in have:
            return False
        need = set(node.modalities) - {"text", "embed", "tokens"}
        have_mod = set(self.profile.modalities) | {"text", "embed"}
        if need and not need.issubset(have_mod):
            return False
        return True

    def cost(self) -> float:
        # cheapest capable = smallest TOPS SKU that still fits (don't burn the hub)
        return float(self.profile.tops)


class Mesh:
    def __init__(self) -> None:
        self.peers: dict[str, MeshPeer] = {}

    def join(self, peer: MeshPeer) -> None:
        self.peers[peer.id] = peer

    def advertise(self) -> list[dict[str, Any]]:
        return [{"id": p.id, **p.profile.advertise()} for p in self.peers.values()]

    def sanitize_payload(self, node: TaskNode, share_raw: bool) -> Any:
        if share_raw or not (node.requires_raw or node.kind in RAW_KINDS):
            return node.payload
        payload = node.payload
        if isinstance(payload, dict):
            keep = ("embedding", "tokens", "caption", "id", "adapter")
            return {k: payload[k] for k in keep if k in payload} or {
                "kind": "embedding",
                "note": "raw stripped; grant share_raw to send frames",
            }
        return {"kind": "embedding", "note": "raw stripped"}

    def place(self, dag: TaskDAG, share_raw: bool = False) -> dict[str, str]:
        """Place each node on the cheapest capable peer the policy allows.

        Returns node_id -> peer_id. Raises PlacementError if the graph cannot
        be placed (caller may then offer cloud — default deny).
        """
        mapping: dict[str, str] = {}
        for node in dag.nodes:
            _ = self.sanitize_payload(node, share_raw)
            cands = [p for p in self.peers.values() if p.capable(node, share_raw)]
            if not cands:
                raise PlacementError(f"cannot place node {node.id!r} kind={node.kind}")
            pick = min(cands, key=lambda p: (p.cost(), p.id))
            mapping[node.id] = pick.id
            pick.profile.reserve(node.ram_mb, node.tops)
        return mapping


class PlacementError(RuntimeError):
    pass
