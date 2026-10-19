"""Cache / state pool for Kimi-K3 (KDA linear attention + MLA).

* MLA layers: latent cache ``mla_kv[i]`` of shape ``[N, 576]`` bf16 (ckv 512 +
  k_rot 64, token-slot indexed, ``page_size = 1``).
* KDA layers: recurrent state ``ssm[i]`` ``[R, H_local, 128(V), 128(K)]`` (fp32 by
  default, ``MINISGL_KDA_STATE_DTYPE=bf16`` halves it) and short-conv state
  ``conv[i]`` ``[R, 3*H_local*128, 3]`` bf16, indexed by request table slot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .base import BaseKVCachePool


def kda_state_dtype() -> torch.dtype:
    return torch.bfloat16 if os.environ.get("MINISGL_KDA_STATE_DTYPE", "fp32").lower() in ("bf16", "bfloat16") else torch.float32


@dataclass(frozen=True)
class KimiCacheSpec:
    num_layers: int
    kda_layers: tuple[int, ...]
    mla_layers: tuple[int, ...]
    kda_heads_local: int
    kda_head_dim: int
    conv_kernel: int
    kv_lora_rank: int
    rope_dim: int

    @classmethod
    def from_model_config(cls, mc) -> "KimiCacheSpec":
        from minisgl.distributed import get_tp_info

        d = mc.hf_text_config
        la = d["linear_attn_config"]
        L = int(mc.num_layers)
        kda1 = {int(x) for x in la["kda_layers"]}  # 1-indexed in the HF config
        kda = tuple(i for i in range(L) if (i + 1) in kda1)
        mla = tuple(i for i in range(L) if (i + 1) not in kda1)
        tp = get_tp_info().size
        heads = int(la["num_heads"])
        assert heads % tp == 0
        return cls(
            num_layers=L,
            kda_layers=kda,
            mla_layers=mla,
            kda_heads_local=heads // tp,
            kda_head_dim=int(la["head_dim"]),
            conv_kernel=int(la["short_conv_kernel_size"]),
            kv_lora_rank=int(d["kv_lora_rank"]),
            rope_dim=int(d["qk_rope_head_dim"]),
        )

    @property
    def conv_dim(self) -> int:
        return 3 * self.kda_heads_local * self.kda_head_dim

    def bytes_per_token(self) -> float:
        return len(self.mla_layers) * (self.kv_lora_rank + self.rope_dim) * 2

    def bytes_per_request(self) -> int:
        d = self.kda_head_dim
        ssm = self.kda_heads_local * d * d * kda_state_dtype().itemsize
        conv = self.conv_dim * (self.conv_kernel - 1) * 2
        return len(self.kda_layers) * (ssm + conv)


class KimiKVPool(BaseKVCachePool):
    def __init__(self, model_config, *, num_pages: int, page_size: int, num_req_slots: int, device: torch.device):
        spec = KimiCacheSpec.from_model_config(model_config)
        self.spec = spec
        self.num_tokens = int(num_pages) * int(page_size)
        self.num_req_slots = int(num_req_slots)
        self._device = device
        self.layer_kind: dict[int, tuple[str, int]] = {}
        for i, l in enumerate(spec.mla_layers):
            self.layer_kind[l] = ("mla", i)
        for i, l in enumerate(spec.kda_layers):
            self.layer_kind[l] = ("kda", i)
        row = spec.kv_lora_rank + spec.rope_dim
        self.mla_kv = torch.zeros(len(spec.mla_layers), self.num_tokens, row, dtype=torch.bfloat16, device=device)
        d = spec.kda_head_dim
        R = self.num_req_slots
        self.ssm = torch.zeros(len(spec.kda_layers), R, spec.kda_heads_local, d, d, dtype=kda_state_dtype(), device=device)
        self.conv = torch.zeros(
            len(spec.kda_layers), R, spec.conv_dim, spec.conv_kernel - 1, dtype=torch.bfloat16, device=device
        )

    def mla_cache(self, layer_id: int) -> torch.Tensor:
        kind, i = self.layer_kind[layer_id]
        assert kind == "mla"
        return self.mla_kv[i]

    def kda_state(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        kind, i = self.layer_kind[layer_id]
        assert kind == "kda"
        return self.ssm[i], self.conv[i]

    def k_cache(self, index: int) -> torch.Tensor:
        return self.mla_cache(index)

    def v_cache(self, index: int) -> torch.Tensor:
        return self.mla_cache(index)

    def store_kv(self, k, v, out_loc, layer_id) -> None:  # pragma: no cover
        raise RuntimeError("KimiKVPool is written by the Kimi attention modules")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def memory_summary(self) -> dict:
        return {
            "mla_kv_bytes": self.mla_kv.numel() * 2,
            "kda_ssm_bytes": self.ssm.numel() * self.ssm.element_size(),
            "kda_conv_bytes": self.conv.numel() * 2,
        }


__all__ = ["KimiKVPool", "KimiCacheSpec", "kda_state_dtype"]
