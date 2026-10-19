"""Attention backend for Kimi-K3: FA2 MLA (absorbed, NoPE) + KDA metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.kvcache.kimi_pool import KimiKVPool
    from minisgl.models import ModelConfig


@dataclass
class KimiMetadata(BaseAttnMetadata):
    is_decode: bool
    num_tokens: int
    cu_seqlens_q: torch.Tensor  # gpu int32 [B+1]
    state_idx: torch.Tensor  # gpu int32 [B]
    has_init: torch.Tensor  # gpu int32 [B]
    max_seqlen_q: int
    # mla plan inputs (cpu) + device indices
    qo_indptr_cpu: torch.Tensor
    kv_indptr_cpu: torch.Tensor
    kv_len_cpu: torch.Tensor
    kv_indices: torch.Tensor
    wrapper: object | None = None
    planned: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class KimiAttnBackend(BaseAttnBackend):
    def __init__(self, config: "ModelConfig") -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        ctx = get_global_ctx()
        self.config = config
        self.pool: KimiKVPool = ctx.kv_cache
        self.device = self.pool.device
        d = config.hf_text_config
        from minisgl.distributed import get_tp_info

        self.num_heads = int(d["num_attention_heads"]) // get_tp_info().size
        self.kv_lora = int(d["kv_lora_rank"])
        self.rope = int(d["qk_rope_head_dim"])
        self.scale = (int(d["qk_nope_head_dim"]) + int(d["qk_rope_head_dim"])) ** -0.5
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.wrapper = BatchMLAPagedAttentionWrapper(self.workspace, backend="fa2")
        self.graph_wrappers: Dict[int, object] = {}
        self._capture_meta: Dict[int, KimiMetadata] = {}
        self._capture_bufs: Dict[str, torch.Tensor] = {}
        self.capture_bs: List[int] = []
        self._last_event = torch.cuda.Event()
        self._last_event.record()

    # ------------------------------------------------------------------
    def _build(self, batch: Batch) -> KimiMetadata:
        reqs = batch.padded_reqs
        dev = self.device
        page_table = get_global_ctx().page_table
        CPU = {"dtype": torch.int32, "pin_memory": True}
        q_lens = [int(r.device_len) - int(r.cached_len) for r in reqs]
        kv_lens = [int(r.device_len) for r in reqs]
        cu_q = [0]
        for n in q_lens:
            cu_q.append(cu_q[-1] + n)
        cu_kv = [0]
        for n in kv_lens:
            cu_kv.append(cu_kv[-1] + n)
        qo_cpu = torch.tensor(cu_q, **CPU)
        kv_cpu = torch.tensor(cu_kv, **CPU)
        kvlen_cpu = torch.tensor(kv_lens, **CPU)
        table = [int(r.table_idx) for r in reqs]
        tbl = torch.tensor(table, **CPU).to(dev, non_blocking=True)
        init = torch.tensor([1 if int(r.cached_len) > 0 else 0 for r in reqs], **CPU).to(dev, non_blocking=True)
        kv_indices = (
            torch.cat([page_table[t, :n] for t, n in zip(table, kv_lens)]) if reqs else page_table.new_empty(0)
        )
        return KimiMetadata(
            is_decode=bool(batch.is_decode),
            num_tokens=cu_q[-1],
            cu_seqlens_q=qo_cpu.to(dev, non_blocking=True),
            state_idx=tbl,
            has_init=init,
            max_seqlen_q=max(q_lens) if q_lens else 0,
            qo_indptr_cpu=qo_cpu,
            kv_indptr_cpu=kv_cpu,
            kv_len_cpu=kvlen_cpu,
            kv_indices=kv_indices,
            wrapper=self.wrapper,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        batch.attn_metadata = self._build(batch)

    def _plan(self, md: KimiMetadata) -> None:
        if md.planned:
            return
        md.planned = True
        self._last_event.synchronize()
        md.wrapper.plan(
            md.qo_indptr_cpu,
            md.kv_indptr_cpu,
            md.kv_indices,
            md.kv_len_cpu,
            self.num_heads,
            self.kv_lora,
            self.rope,
            1,
            not md.is_decode,
            self.scale,
            torch.bfloat16,
            torch.bfloat16,
        )
        self._last_event.record()

    def mla_attn(self, layer_id: int, q_abs: torch.Tensor, q_rope: torch.Tensor) -> torch.Tensor:
        md = get_global_ctx().batch.attn_metadata
        self._plan(md)
        cache = self.pool.mla_cache(layer_id)
        ckv = cache[:, : self.kv_lora].unsqueeze(1)
        kpe = cache[:, self.kv_lora :].unsqueeze(1)
        return md.wrapper.run(q_abs, q_rope, ckv, kpe)

    # ------------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        self._capture_bufs = {
            "kv_indices": torch.zeros(max_bs * max_seq_len, dtype=torch.int32, device=self.device),
        }

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        bs = batch.size
        dev = self.device
        w = BatchMLAPagedAttentionWrapper(
            self.workspace,
            use_cuda_graph=True,
            qo_indptr=torch.arange(bs + 1, dtype=torch.int32, device=dev),
            kv_indptr=torch.zeros(bs + 1, dtype=torch.int32, device=dev),
            kv_indices=self._capture_bufs["kv_indices"],
            kv_len_arr=torch.zeros(bs, dtype=torch.int32, device=dev),
            backend="fa2",
        )
        self.graph_wrappers[bs] = w
        md = self._build(batch)
        md.wrapper = w
        md.cu_seqlens_q = md.cu_seqlens_q.clone()
        md.state_idx = md.state_idx.clone()
        md.has_init = md.has_init.clone()
        self._capture_meta[bs] = md
        batch.attn_metadata = md
        self._plan(md)

    def prepare_for_replay(self, batch: Batch) -> None:
        bs = batch.padded_size
        cap = self._capture_meta[bs]
        real = batch.attn_metadata
        assert isinstance(real, KimiMetadata)
        cap.state_idx.copy_(real.state_idx)
        cap.has_init.copy_(real.has_init)
        cap.cu_seqlens_q.copy_(real.cu_seqlens_q)
        real.wrapper = self.graph_wrappers[bs]
        real.planned = False
        self._plan(real)

    def forward(self, q, k, v, layer_id, batch):  # pragma: no cover
        raise RuntimeError("KimiAttnBackend is driven by the Kimi attention modules")


__all__ = ["KimiAttnBackend", "KimiMetadata"]
