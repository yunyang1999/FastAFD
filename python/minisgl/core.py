from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        # Amortised append into one preallocated host buffer per request. The previous
        # torch.cat re-copied the whole sequence per request and decode step; beyond
        # 32K elements ATen parallelises that copy over the OpenMP pool, which made the
        # colocated scheduler host-bound at 32K context (~0.4 ms per request and step).
        n = len(self.input_ids)
        buf = getattr(self, "_ids_buf", None)
        if buf is None or buf.numel() < n + 1:
            buf = torch.empty(max(self.max_device_len, n + 1), dtype=self.input_ids.dtype)
            buf[:n].copy_(self.input_ids)
            self._ids_buf = buf
        buf[n : n + 1].copy_(next_token)
        self.input_ids = buf[: n + 1]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    positions_host: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # AFD microbatch ping-pong: req-order and token-order split points
    # (both len = m+1). `afd_microbatch_offsets[-1] == padded_size`, slices
    # `padded_reqs`.
    # `afd_microbatch_token_offsets[-1] == total_tokens`, slices flat token
    # tensors (`out_loc`, `positions`, `input_ids`, packed comm buffers).
    # For decode they coincide (1 token per req); for prefill they differ.
    # AFD attention scheduler also populates `attn_metadata_mbs` with one
    # metadata object per MB; for m=1 it is a one-element list.
    afd_microbatch_offsets: tuple[int, ...] | None = field(default=None, init=False)
    afd_microbatch_token_offsets: tuple[int, ...] | None = field(default=None, init=False)
    attn_metadata_mbs: list[BaseAttnMetadata] | None = field(default=None, init=False)
    afd_mb_subbatches: list[Batch] | None = field(default=None, init=False)
    afd_sampling_plan: object | None = field(default=None, init=False)
    afd_last_indices: torch.Tensor | None = field(default=None, init=False)
    afd_num_sampling: int = field(default=0, init=False)
    afd_writeback: object | None = field(default=None, init=False)
    afd_req_table_indices_gpu: torch.Tensor | None = field(default=None, init=False)
    afd_kv_store_merged: bool = field(default=False, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    mlp_deepep_buffer: object | None = None
    moe_num_token_non_padded: torch.Tensor | int | None = None
    moe_deepep_dispatch_max_tokens_per_rank: int | None = None
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
