"""SARARuntime — local if headroom else mesh else cloud-if-granted else deny."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from .mesh import Mesh, PlacementError, TaskDAG
from .profile import DeviceProfile, lan_host_allowed
from .quantize import quantize_linear_int8


@dataclass
class Placement:
    where: str  # local | mesh | cloud | deny
    reason: str
    peer: Optional[str] = None
    node_map: dict[str, str] = field(default_factory=dict)


class SARARuntime:
    """Same binary on every SKU. Profile + grants decide where work runs."""

    def __init__(
        self,
        profile: DeviceProfile,
        grants: Any = None,
        mesh: Optional[Mesh] = None,
        model=None,
        tokenizer=None,
        cfg=None,
        audit: Any = None,
        learner: Any = None,
        vault: Any = None,
        workspace: str | Path = ".",
    ):
        self.profile = profile
        self.grants = grants
        self.mesh = mesh
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.audit = audit
        self.learner = learner
        self.vault = vault
        self.workspace = Path(workspace)
        self.quantized = False

    def maybe_quantize(self) -> None:
        if self.model is None or self.quantized:
            return
        if (self.profile.quantize or "").lower() in {"int8", "dynamic-int8", "qint8"}:
            self.model = quantize_linear_int8(self.model)
            self.quantized = True

    def _cloud_ok(self) -> bool:
        if not self.profile.cloud_allowed:
            return False
        if self.grants is None:
            return False
        return bool(self.grants.allowed("cloud"))

    def _mesh_ok(self) -> bool:
        if not self.profile.mesh_allowed or self.mesh is None:
            return False
        if self.grants is None:
            return True  # mesh radio on; share_raw still gated at payload
        return bool(self.grants.allowed("mesh"))

    def _share_raw(self) -> bool:
        return bool(self.grants and self.grants.allowed("share_raw"))

    def place(
        self,
        task: Any = "infer",
        ram_need: int = 64,
        tops_need: float = 0.5,
        require_modalities: Optional[list[str]] = None,
    ) -> Placement:
        """Rule: if the chip still has headroom, do not leave the device."""
        dag = task if isinstance(task, TaskDAG) else None
        if dag is not None:
            ram_need = sum(n.ram_mb for n in dag.nodes) or ram_need
            tops_need = sum(n.tops for n in dag.nodes) or tops_need

        local_ok = self.profile.has_headroom(ram_need, tops_need)
        if require_modalities:
            have = set(self.profile.modalities) | {"text"}
            if not set(require_modalities).issubset(have):
                local_ok = False

        if local_ok:
            pl = Placement("local", "chip has headroom")
            self._audit("place", where="local", task=str(getattr(task, "id", task))[:80])
            return pl

        if self._mesh_ok():
            if dag is not None:
                try:
                    mapping = self.mesh.place(dag, share_raw=self._share_raw())
                    peer = next(iter(set(mapping.values()))) if mapping else None
                    pl = Placement("mesh", "overflow to peer mesh", peer=peer, node_map=mapping)
                    self._audit("place", where="mesh", node_map=mapping)
                    return pl
                except PlacementError as e:
                    mesh_reason = str(e)
            else:
                # single blob: any peer with headroom
                for pid, peer in self.mesh.peers.items():
                    if peer.profile.has_headroom(ram_need, tops_need):
                        pl = Placement("mesh", "overflow to peer", peer=pid)
                        self._audit("place", where="mesh", peer=pid)
                        return pl
                mesh_reason = "no peer with headroom"
        else:
            mesh_reason = "mesh unavailable or grant denied"

        if self._cloud_ok():
            pl = Placement("cloud", "local+mesh saturated; cloud grant present")
            self._audit("place", where="cloud")
            return pl

        pl = Placement(
            "deny",
            f"no headroom; {mesh_reason}; cloud not granted",
        )
        self._audit("place", where="deny", reason=pl.reason)
        return pl

    def assert_lan(self, host: str) -> None:
        if not self.profile.ssrf_deny:
            return
        if not lan_host_allowed(host, self.profile.lan_allowlist):
            raise PermissionError(f"SSRF deny-by-default: host {host!r} not on LAN allowlist")

    def see(self, image: torch.Tensor, prompt: str = "describe this image") -> dict[str, Any]:
        if self.grants is not None:
            self.grants.require("camera")
        pl = self.place("see", ram_need=48, tops_need=0.2, require_modalities=["vision"])
        if pl.where == "deny":
            raise PermissionError(pl.reason)
        if pl.where == "cloud" and (self.grants is None or not self.grants.allowed("cloud")):
            raise PermissionError("cloud see-path denied")
        if self.model is None:
            raise RuntimeError("no model bound")
        self.model.eval()
        with torch.no_grad():
            vis = self.model.vision(image)
            tokens = None
            caption = ""
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(
                    f"<|bos|><|user|>{prompt}<|assistant|>", add_bos=False
                )
                cap = 32
                if self.cfg is not None:
                    cap = max(4, int(self.cfg.max_seq_len) - 16)
                tokens = torch.tensor([ids[:cap] or [1]], dtype=torch.long)
            else:
                vocab = int(getattr(self.model.config, "vocab_size", 128))
                tokens = torch.randint(1, max(2, vocab), (1, 8))
            out = self.model.generate(
                tokens, max_new=16, temperature=0.0, images=image,
                eos_id=getattr(self.model.config, "eos_id", 2),
            )
            if self.tokenizer is not None:
                caption = self.tokenizer.decode(out[0].tolist())
            else:
                caption = f"tokens:{tuple(out.shape)}"
        rec = {
            "caption": caption,
            "embedding": vis.mean(dim=1).detach().cpu(),
            "placement": pl,
            "cloud": pl.where == "cloud",
        }
        if self.audit is not None:
            self.audit.inference(
                event="see",
                model=self.model,
                quant=self.profile.quantize,
                adapter_id=self._adapter_id(),
                placement=pl.where,
            )
        return rec

    def run_agent(self, goal: str, max_steps: int = 6):
        from sara.agent.loop import AgentRuntime

        if self.grants is not None and self.grants.allowed("cloud"):
            # still fine — but phone demo denies cloud
            pass
        pl = self.place("agent", ram_need=64, tops_need=0.3)
        if pl.where == "deny":
            raise PermissionError(pl.reason)
        if pl.where == "cloud" and not self._cloud_ok():
            raise PermissionError("cloud agent denied")
        rt = AgentRuntime(
            self.workspace,
            model=self.model,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            max_steps=max_steps,
            grants=self.grants,
            audit=self.audit,
        )
        res = rt.run(goal)
        if self.audit is not None:
            self.audit.inference(
                event="agent",
                model=self.model,
                quant=self.profile.quantize,
                adapter_id=self._adapter_id(),
                placement=pl.where,
            )
        return res, pl

    def _adapter_id(self) -> str:
        if self.learner is None:
            return "none"
        return getattr(self.learner, "adapter_id", "local")

    def _audit(self, event: str, **kw: Any) -> None:
        if self.audit is not None:
            self.audit.record(event, **kw)
