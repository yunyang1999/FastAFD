"""Attention backend for Kimi-K3: MLA (absorbed, NoPE) + KDA metadata.

Decode MLA runs flashinfer's trtllm-gen MLA kernel on a paged BF16 latent cache
(page size 64), the kernel sglang 0.5.18 selects on SM100 (``trtllm_mla``).
Head counts that are not a power of two (TP1/TP2: 96 / 48) are zero-padded to
128 / 64 so one call reads the KV once. Prefill (and MINISGL_K3_MLA_DECODE=fa2) uses the FA2 MLA wrapper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch
import triton
import triton.language as tl
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
    block_tables: torch.Tensor | None = None  # trtllm decode: [B, NP] int32 page ids
    seq_lens: torch.Tensor | None = None  # trtllm decode: [B] int32 kv lengths

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
        self.page_size = int(ctx.page_size)
        self.max_seq_len = int(ctx.page_table.shape[1])
        self.use_trtllm = (
            os.environ.get("MINISGL_K3_MLA_DECODE", "sglang").lower() == "sglang" and self.page_size in (32, 64)
        )
        if self.use_trtllm:
            # trtllm-gen requires a multiple of 128/page_size block-table columns
            mult = 128 // self.page_size
            npg = (self.max_seq_len + self.page_size - 1) // self.page_size
            self.max_pages = (npg + mult - 1) // mult * mult
            self.trtllm_ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.head_chunks = trtllm_mla_head_chunks(self.num_heads)
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
        is_decode = bool(batch.is_decode)
        if is_decode and self.use_trtllm:
            # paged decode: page ids from the token-slot page table (slot // P)
            kv_indices = page_table.new_empty(0)
            P = self.page_size
            bt = torch.div(page_table[:, : self.max_pages * P : P][tbl.long()], P, rounding_mode="floor")
            if bt.shape[1] < self.max_pages:
                bt = torch.nn.functional.pad(bt, (0, self.max_pages - bt.shape[1]))
            block_tables = bt.to(torch.int32)
            seq_lens = kvlen_cpu.to(dev, non_blocking=True)
        else:
            kv_indices = (
                torch.cat([page_table[t, :n] for t, n in zip(table, kv_lens)]) if reqs else page_table.new_empty(0)
            )
            block_tables = seq_lens = None
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
            block_tables=block_tables,
            seq_lens=seq_lens,
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
        if md.block_tables is not None:
            return self._trtllm_decode(layer_id, q_abs, q_rope, md)
        self._plan(md)
        cache = self.pool.mla_cache(layer_id)
        ckv = cache[:, : self.kv_lora].unsqueeze(1)
        kpe = cache[:, self.kv_lora :].unsqueeze(1)
        return md.wrapper.run(q_abs.contiguous(), q_rope.contiguous(), ckv, kpe)

    def _trtllm_decode(self, layer_id: int, q_abs: torch.Tensor, q_rope: torch.Tensor, md: KimiMetadata) -> torch.Tensor:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        B, H, _ = q_abs.shape
        D = self.kv_lora + self.rope
        # the query kernel takes any row strides but needs unit stride inside a head
        q_abs = q_abs if q_abs.stride(-1) == 1 else q_abs.contiguous()
        q_rope = q_rope if q_rope.stride(-1) == 1 else q_rope.contiguous()
        cache = self.pool.mla_cache(layer_id).view(-1, 1, self.page_size, D)
        outs = []
        for h0, hn, hp in self.head_chunks:
            # query [B, 1, hp, 576] = [q_abs | q_rope] with zero-padded heads up to a
            # count every trtllm-gen variant takes; one Triton pass
            q = torch.empty((B, 1, hp, D), dtype=torch.bfloat16, device=q_abs.device)
            if B:
                _mla_query_kernel[(B * hp,)](
                    q_abs, q_rope, q, h0, hn, hp, q_abs.stride(0), q_abs.stride(1), q_rope.stride(0), q_rope.stride(1),
                    LORA=self.kv_lora, ROPE=self.rope, BLOCK=1024, num_warps=4,
                )
            o = trtllm_batch_decode_with_kv_cache_mla(
                query=q, kv_cache=cache, workspace_buffer=self.trtllm_ws,
                qk_nope_head_dim=128, kv_lora_rank=self.kv_lora, qk_rope_head_dim=self.rope,
                block_tables=md.block_tables, seq_lens=md.seq_lens, max_seq_len=self.max_seq_len,
                bmm1_scale=self.scale, bmm2_scale=1.0,
            )
            outs.append(o.view(B, hp, self.kv_lora)[:, :hn])
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

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
        if md.block_tables is not None:
            md.block_tables = md.block_tables.clone()
            md.seq_lens = md.seq_lens.clone()
        self._capture_meta[bs] = md
        batch.attn_metadata = md
        if md.block_tables is None:
            self._plan(md)

    def prepare_for_replay(self, batch: Batch) -> None:
        bs = batch.padded_size
        cap = self._capture_meta[bs]
        real = batch.attn_metadata
        assert isinstance(real, KimiMetadata)
        cap.state_idx.copy_(real.state_idx)
        cap.has_init.copy_(real.has_init)
        cap.cu_seqlens_q.copy_(real.cu_seqlens_q)
        if cap.block_tables is not None:
            cap.block_tables.copy_(real.block_tables)
            cap.seq_lens.copy_(real.seq_lens)
            return
        real.wrapper = self.graph_wrappers[bs]
        real.planned = False
        self._plan(real)

    def forward(self, q, k, v, layer_id, batch):  # pragma: no cover
        raise RuntimeError("KimiAttnBackend is driven by the Kimi attention modules")


@triton.jit
def _mla_query_kernel(qa_ptr, qr_ptr, out_ptr, h0, hn, hp, sa_b, sa_h, sr_b, sr_h,
                      LORA: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr):
    """out [B, hp, LORA+ROPE] = [q_abs | q_rope] of heads h0..h0+hn, zero rows above hn."""
    r = tl.program_id(0)
    b = r // hp
    h = r - b * hp
    j = tl.arange(0, BLOCK)
    valid = h < hn
    a = tl.load(qa_ptr + b * sa_b + (h0 + h) * sa_h + j, mask=valid & (j < LORA), other=0.0)
    rr = tl.load(qr_ptr + b * sr_b + (h0 + h) * sr_h + (j - LORA), mask=valid & (j >= LORA) & (j < LORA + ROPE), other=0.0)
    v = tl.where(j < LORA, a, rr)
    tl.store(out_ptr + r * (LORA + ROPE) + j, v, mask=j < LORA + ROPE)


def trtllm_mla_head_chunks(num_heads: int) -> list[tuple[int, int, int]]:
    """(first head, heads, padded heads) per trtllm-gen MLA call. Default: one call
    with the heads zero-padded to the next power of two (96 -> 128, 48 -> 64; the
    flashinfer 0.6.6 kernels reject non power-of-two head counts for some
    variants); MINISGL_K3_MLA_HEAD_CHUNKS=64,32,... splits into exact chunks
    instead (one KV pass per chunk)."""
    env = os.environ.get("MINISGL_K3_MLA_HEAD_CHUNKS", "")
    if not env:
        chunks, h0, rem = [], 0, num_heads
        while rem:
            hn = min(rem, 128)
            chunks.append((h0, hn, max(16, 1 << (hn - 1).bit_length())))
            h0 += hn
            rem -= hn
        return chunks
    sup = [int(x) for x in env.split(",")]
    chunks, h0, rem = [], 0, num_heads
    for c in sorted(sup, reverse=True):
        while rem >= c:
            chunks.append((h0, c, c))
            h0 += c
            rem -= c
    if rem:
        raise ValueError(f"trtllm-gen MLA: cannot split {num_heads} heads into {sup}")
    return chunks


__all__ = ["KimiAttnBackend", "KimiMetadata", "trtllm_mla_head_chunks"]
