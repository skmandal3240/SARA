"""AirLLM-style block streaming: load one TransformerBlock, run, unload.

This is a real loop over `model.blocks`. Inactive layers sit as CPU
state-dicts; at most one block is 'resident' in the working set besides
embeddings / current activations. Do not fake this with a full forward.
"""

from __future__ import annotations

from typing import Optional

import torch

class LayerPager:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self._cpu: list[dict[str, torch.Tensor]] = []
        self.max_resident = 0
        self._resident = 0
        self.loads = 0
        for blk in model.blocks:
            self._cpu.append(
                {k: v.detach().cpu().contiguous().clone() for k, v in blk.state_dict().items()}
            )
            blk.to("cpu")

    def _load(self, i: int) -> None:
        blk = self.model.blocks[i]
        blk.load_state_dict(self._cpu[i], strict=True)
        blk.to(self.device)
        blk.eval()
        self._resident += 1
        self.loads += 1
        self.max_resident = max(self.max_resident, self._resident)

    def _unload(self, i: int) -> None:
        self.model.blocks[i].to("cpu")
        self._resident = max(0, self._resident - 1)

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> dict:
        """Same tensor contract as `SARA.forward`, one block resident at a time."""
        model = self.model
        model.eval()
        if type_ids is None:
            type_ids = torch.zeros_like(tokens)
        h = model.tok_emb(tokens) + model.type_emb(type_ids)
        memories = []
        if images is not None:
            memories.append(model.vision(images))
        if mel is not None:
            memories.append(model.audio_enc(mel))
        if memories:
            context = torch.cat(memories, dim=1) if context is None else torch.cat(
                [context, *memories], dim=1
            )
        new_caches = []
        for i in range(len(model.blocks)):
            self._load(i)
            h, cache_i = model.blocks[i](h, context=context, start_pos=start_pos, kv_cache=None)
            new_caches.append(cache_i)
            self._unload(i)
        h = model.norm(h)
        logits = model.lm_head(h)
        return {"logits": logits, "hidden": h, "kv_caches": new_caches, "context": context}
