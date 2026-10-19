"""Kimi-K3 attention stage for AFD (attention workers).

Everything except the routed experts runs here: KDA/MLA attention, attention
residuals, dense layer 0, the router (hidden space), the latent down projection
(the tensor dispatched to the expert workers), and after combine the latent
RMSNorm + up projection plus the 2 shared experts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import BaseOP, ParallelLMHead
from minisgl.models.afd import ModelStageState

from .kimi_k3 import K3Residual, K3TopK, KimiK3Model

if TYPE_CHECKING:
    from .config import ModelConfig


class KimiK3AfdAttention(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.model = KimiK3Model(config, afd_attention_side=True)
        self.lm_head = ParallelLMHead(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self._config = config
        self.num_routed_experts = self.model._args.n_routed

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("AFD stage model is driven by the AFD worker")

    def post_load(self, **_) -> None:
        for layer in self.model.layers.op_list:
            if not layer.is_kda:
                layer.self_attn._absorb()

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(hidden_states=self.model.embed_tokens.forward(input_ids), residual=None)

    def _resume(self, hidden: torch.Tensor, residual: K3Residual | None) -> K3Residual:
        if residual is None:
            return self.model.new_residual(hidden)
        residual.prefix = residual.prefix + hidden
        return residual

    def forward_attention(self, layer_id: int, hidden: torch.Tensor, residual=None) -> ModelStageState:
        res = self._resume(hidden, residual)
        h, res = self.model.layers.op_list[layer_id].attention_part(res)
        return ModelStageState(hidden_states=h, residual=res)

    def route(self, layer_id: int, hidden: torch.Tensor) -> K3TopK:
        layer = self.model.layers.op_list[layer_id]
        if layer.block_sparse_moe is None:
            return K3TopK(topk_ids=None, topk_weights=None, skip_moe=True, local_output=layer.mlp.forward(hidden))
        moe = layer.block_sparse_moe
        ids, w = moe.gate.forward(hidden)
        latent = moe.routed_expert_down_proj.forward(hidden)
        shared = moe.shared_experts.forward(hidden) if moe.shared_experts is not None else None
        return K3TopK(
            topk_ids=ids,
            topk_weights=w,
            shared_output=shared,
            dispatch_input=latent,
            post_fn=lambda y, _m=moe, _s=shared: _m.post_combine(y, _s),
        )

    def finalize_hidden(self, hidden: torch.Tensor, residual=None) -> torch.Tensor:
        res = self._resume(hidden, residual)
        return self.model.final_hidden(res)

    def forward_lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward(hidden)


__all__ = ["KimiK3AfdAttention"]
