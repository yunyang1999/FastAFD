"""KV cache / state pool for DeepSeek-V4 (SWA + compressed sparse attention).

Per layer ``l`` one KV buffer ``kv[l]`` of rows of 576 bf16 values (the 512-d
MQA latent stored as MLA ``ckv`` plus a zero 64-d ``kpe``) laid out as

    [ SWA ring:  R x window rows      row = table_idx * window + pos % window ]
    [ scratch:   S rows               this forward's new tokens (row = S0 + i) ]
    [ compressed: N / ratio rows      row = C0 + token_slot // ratio           ]

with ``R = max_running_req + 1`` (last = graph dummy), ``S`` the max tokens per
forward and ``N = num_pages * page_size`` token slots. ``page_size`` must be a
multiple of the largest compression ratio so a compressed group never spans
pages. C4 layers additionally own an indexer-key buffer ``[N/4, 128]`` and all
compressed layers own fp32 compressor states (kv, score) per request. Indexer
keys are FP8 with a per-entry FP32 scale in pages of 64 c4 entries (page_size
must be 256 so a c4 page never spans two token pages).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import BaseKVCachePool

ROW_DIM = 576  # 512 latent + 64 (zero) rope slot of the MLA kernel layout
ROW_ALIGN = 64  # trtllm-gen MLA kernel views the cache as [-1, 1, 64, 576]


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass(frozen=True)
class DSV4CacheSpec:
    num_layers: int
    ratios: tuple[int, ...]
    window: int
    head_dim: int
    index_head_dim: int

    @classmethod
    def from_model_config(cls, mc) -> "DSV4CacheSpec":
        d = mc.hf_text_config
        L = int(mc.num_layers)
        ratios = tuple(int(r) for r in d["compress_ratios"][:L])
        return cls(
            num_layers=L,
            ratios=ratios,
            window=int(d.get("sliding_window", 128)),
            head_dim=int(d["head_dim"]),
            index_head_dim=int(d["index_head_dim"]),
        )

    def coff(self, ratio: int) -> int:
        return 2 if ratio == 4 else 1

    def bytes_per_token(self) -> float:
        b = 0.0
        for r in self.ratios:
            if r:
                b += ROW_DIM * 2 / r
            if r == 4:
                b += (self.index_head_dim + 4) / 4  # FP8 key + FP32 scale per c4 entry
        return b

    def bytes_per_request(self) -> int:
        b = 0
        for r in self.ratios:
            b += self.window * ROW_DIM * 2
            if r:
                c = self.coff(r)
                b += 2 * (c * r) * (c * self.head_dim) * 4
            if r == 4:
                b += 2 * (2 * 4) * (2 * self.index_head_dim) * 4
        return b

    def fixed_bytes(self, scratch_rows: int) -> int:
        return self.num_layers * scratch_rows * ROW_DIM * 2


class DSV4KVPool(BaseKVCachePool):
    def __init__(
        self,
        model_config,
        *,
        num_pages: int,
        page_size: int,
        num_req_slots: int,
        scratch_rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        spec = DSV4CacheSpec.from_model_config(model_config)
        max_ratio = max([r for r in spec.ratios if r] or [1])
        if page_size % max_ratio != 0:
            raise ValueError(f"DSV4 page_size={page_size} must be a multiple of {max_ratio}")
        self.spec = spec
        self.page_size = int(page_size)
        self.num_tokens = int(num_pages) * int(page_size)
        self.num_req_slots = int(num_req_slots)
        self.window = spec.window
        self.swa_rows = self.num_req_slots * spec.window
        self.scratch_rows = _align(int(scratch_rows), ROW_ALIGN)
        self.scratch_base = self.swa_rows
        self.comp_base = self.swa_rows + self.scratch_rows
        self._device = device
        self._dtype = torch.bfloat16
        self.kv: list[torch.Tensor] = []
        self.idx_keys: list[torch.Tensor | None] = []
        self.state_kv: list[torch.Tensor | None] = []
        self.state_score: list[torch.Tensor | None] = []
        self.idx_state_kv: list[torch.Tensor | None] = []
        self.idx_state_score: list[torch.Tensor | None] = []
        R = self.num_req_slots
        d = spec.head_dim
        for r in spec.ratios:
            comp_rows = self.num_tokens // r if r else 0
            rows = _align(self.comp_base + comp_rows, ROW_ALIGN)
            self.kv.append(torch.zeros(rows, ROW_DIM, dtype=torch.bfloat16, device=device))
            if r:
                c = spec.coff(r)
                self.state_kv.append(torch.zeros(R, c * r, c * d, dtype=torch.float32, device=device))
                self.state_score.append(
                    torch.full((R, c * r, c * d), float("-inf"), dtype=torch.float32, device=device)
                )
            else:
                self.state_kv.append(None)
                self.state_score.append(None)
            if r == 4:
                di = spec.index_head_dim
                # paged FP8 indexer keys (DeepGEMM fused layout): per page of 64
                # c4 entries, 64 x di FP8 values then 64 FP32 scales
                n_idx_pages = (self.num_tokens // 4 + 63) // 64
                self.idx_keys.append(torch.zeros(n_idx_pages, 64 * (di + 4), dtype=torch.uint8, device=device))
                self.idx_state_kv.append(torch.zeros(R, 8, 2 * di, dtype=torch.float32, device=device))
                self.idx_state_score.append(
                    torch.full((R, 8, 2 * di), float("-inf"), dtype=torch.float32, device=device)
                )
            else:
                self.idx_keys.append(None)
                self.idx_state_kv.append(None)
                self.idx_state_score.append(None)

    # ---- BaseKVCachePool interface (MHA-style accessors are not meaningful) ----
    def k_cache(self, index: int) -> torch.Tensor:
        return self.kv[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.kv[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:  # pragma: no cover
        raise RuntimeError("DSV4KVPool is written by the DSV4 attention modules")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def memory_summary(self) -> dict:
        tot = sum(t.numel() * t.element_size() for t in self.kv)
        tot_idx = sum(t.numel() * t.element_size() for t in self.idx_keys if t is not None)
        tot_state = sum(
            t.numel() * t.element_size()
            for lst in (self.state_kv, self.state_score, self.idx_state_kv, self.idx_state_score)
            for t in lst
            if t is not None
        )
        return {"kv_bytes": tot, "indexer_key_bytes": tot_idx, "compressor_state_bytes": tot_state}


__all__ = ["DSV4KVPool", "DSV4CacheSpec", "ROW_DIM"]
