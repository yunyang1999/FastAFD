"""DeepSeek-V4 attention stage for AFD (attention workers).

Runs embed + mHC + attention/compressor/indexer + router per layer and the final
``hc_head``/norm/lm_head. The MegaMoE expert workers compute routed experts plus
the shared expert (protocol slot weight 1/rsf), so after combine the attention
side only applies ``routed_scaling_factor`` (``afd_moe_output``) and the mHC
post-mix of the next layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import BaseOP, ParallelLMHead
from minisgl.models.afd import ModelStageState

from .deepseek_v4 import DSV4Model, DSV4Residual, DSV4TopK

if TYPE_CHECKING:
    from .config import ModelConfig


class DeepseekV4AfdAttention(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.model = DSV4Model(config, afd_attention_side=True)
        self.lm_head = ParallelLMHead(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self._rsf = float(self.model._args.rsf)
        self._config = config

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("AFD stage model is driven by the AFD worker")

    def post_load(self, **_) -> None:
        for layer in self.model.layers.op_list:
            layer.prepare()

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(hidden_states=self.model.embed_hc(input_ids), residual=None)

    def _resume(self, hidden: torch.Tensor, residual: DSV4Residual | None) -> torch.Tensor:
        if residual is None:
            return hidden
        from minisgl.kernel.dsv4_triton import hc_post_fused

        return hc_post_fused(hidden, residual.x, residual.post, residual.comb)

    def forward_attention(self, layer_id: int, hidden: torch.Tensor, residual=None) -> ModelStageState:
        x = self._resume(hidden, residual)
        h, res = self.model.layers.op_list[layer_id].attention_part(x)
        return ModelStageState(hidden_states=h, residual=res)

    def route(self, layer_id: int, hidden: torch.Tensor) -> DSV4TopK:
        ids, w = self.model.layers.op_list[layer_id].ffn.gate.forward(hidden)
        return DSV4TopK(topk_ids=ids, topk_weights=w, routed_scaling_factor=self._rsf)

    def finalize_hidden(self, hidden: torch.Tensor, residual=None) -> torch.Tensor:
        return self.model.head_hidden(self._resume(hidden, residual))

    def forward_lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward(hidden)


__all__ = ["DeepseekV4AfdAttention"]
