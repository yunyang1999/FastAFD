"""Attention backend for DeepSeek-V4 (sliding window + compressed sparse MQA).

The backend owns per-batch index metadata; the layers (models/deepseek_v4.py)
compute projections, write caches and call :meth:`DSV4AttnBackend.sparse_attn`,
which runs flashinfer's trtllm-gen sparse MLA decode kernel over explicit per-token
row lists (window rows + compressed rows) with per-head attention sinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch
import triton
import triton.language as tl
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.kvcache.dsv4_pool import DSV4KVPool
    from minisgl.models import ModelConfig


def _pad_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@triton.jit
def _pad_q_kernel(q_ptr, out_ptr, H, Hp, stride_t, stride_h, D: tl.constexpr, DP: tl.constexpr, BLOCK: tl.constexpr):
    """out [T, Hp, DP] = q [T, H, D] zero-padded in heads and features (one pass)."""
    r = tl.program_id(0)
    t = r // Hp
    h = r - t * Hp
    j = tl.arange(0, BLOCK)
    v = tl.load(q_ptr + t * stride_t + h * stride_h + j, mask=(j < D) & (h < H), other=0.0)
    tl.store(out_ptr + r * DP + j, v, mask=j < DP)


@triton.jit
def _assemble_index_kernel(
    win_ptr,  # [T, W] int32 window rows (valid prefix of n_window)
    nwin_ptr,  # [T] int32
    comp_ptr,  # [T, Kc] compressed rows (valid prefix)
    ccnt_ptr,  # [T] valid compressed count (clamped to Kc)
    out_ptr,  # [T, Kp] int32
    sl_ptr,  # [T] int32 total valid
    W,
    Kc,
    Kp,
    stride_w,
    stride_c,
    comp_off,
    HAS_COMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    nw = tl.load(nwin_ptr + t)
    nc = 0
    if HAS_COMP:
        nc = tl.minimum(tl.load(ccnt_ptr + t).to(tl.int32), Kc)
    for j0 in range(0, Kp, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        wv = tl.load(win_ptr + t * stride_w + j, mask=(j < nw) & (j < W), other=-1)
        v = tl.where(j < nw, wv, -1)
        if HAS_COMP:
            cj = j - nw
            inc = (cj >= 0) & (cj < nc)
            cv = tl.load(comp_ptr + t * stride_c + tl.where(inc, cj, 0), mask=inc, other=-1).to(tl.int32)
            v = tl.where(inc, cv + comp_off, v)
        tl.store(out_ptr + t * Kp + j, v, mask=j < Kp)
    tl.store(sl_ptr + t, nw + nc)


@dataclass
class DSV4PrefillSeq:
    tok_start: int
    tok_end: int
    table_idx: int
    cached_len: int
    device_len: int


@dataclass
class DSV4Metadata(BaseAttnMetadata):
    is_decode: bool
    num_tokens: int
    tok_table: torch.Tensor  # [T] int32
    tok_cached: torch.Tensor  # [T] int32 cached_len of the token's request
    scratch_rows: torch.Tensor  # [T] int64 absolute rows
    window_idx: torch.Tensor  # [T, W] int32 absolute rows (-1 invalid)
    ring_src: torch.Tensor  # [M] int64
    ring_dst: torch.Tensor  # [M] int64
    c128_idx: torch.Tensor | None  # [T, n128] int32 absolute rows (-1 invalid)
    c4_rows: torch.Tensor | None  # [T, n4] int64 relative compressed rows (decode only)
    n4_valid: torch.Tensor  # [T] int32
    n128_valid: torch.Tensor  # [T] int32
    n_window: torch.Tensor  # [T] int32 valid window entries (a prefix of window_idx)
    comp_write: Dict[int, torch.Tensor]  # ratio -> [T] int64 relative row (-1 none)
    group_start: Dict[int, torch.Tensor]  # ratio -> [T] int32 rope position of completing group
    seq_lens: torch.Tensor  # [T] int32 kernel seq_lens
    cu_seqlens_q: torch.Tensor  # [B+1] int32 (gpu)
    prefill_seqs: List[DSV4PrefillSeq] = field(default_factory=list)
    # sglang indexer path (decode): c4 block table / DeepGEMM schedule / top-k plan
    c4_bt: torch.Tensor | None = None  # [T, NP] int32 page ids (64 c4 entries per page)
    dg_sched: torch.Tensor | None = None  # [num_sms + 1, 2] int32
    topk_plan: torch.Tensor | None = None  # [T + 1, 2] int32

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class DSV4AttnBackend(BaseAttnBackend):
    def __init__(self, config: "ModelConfig") -> None:
        ctx = get_global_ctx()
        self.config = config
        self.pool: DSV4KVPool = ctx.kv_cache
        self.device = self.pool.device
        self.window = self.pool.window
        d = config.hf_text_config
        self.index_topk = int(d["index_topk"])
        self.ratios = self.pool.spec.ratios
        self.max_seq_len = int(ctx.page_table.shape[1])
        self.n4 = (self.max_seq_len + 3) // 4
        self.n128 = (self.max_seq_len + 127) // 128
        self.workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._j = torch.arange(self.window, device=self.device, dtype=torch.int32)
        self._c128 = torch.arange(self.n128, device=self.device, dtype=torch.int32)
        self._c4 = torch.arange(self.n4, device=self.device, dtype=torch.int32)
        self.capture: Dict[str, torch.Tensor] | None = None
        self.capture_bs: List[int] = []
        self._capture_meta: Dict[int, DSV4Metadata] = {}
        self._sinks_pad: Dict[tuple, torch.Tensor] = {}
        import os

        self.indexer_impl = os.environ.get("MINISGL_DSV4_INDEXER", "sglang").lower()
        self.page_size = int(ctx.page_size)
        if self.indexer_impl == "sglang" and self.page_size != 256 and any(r == 4 for r in self.ratios):
            raise ValueError("the sglang DSV4 indexer path needs page_size=256 (64 c4 entries per page)")
        self.n_c4_pages = (self.max_seq_len + self.page_size - 1) // self.page_size
        self.max_c4_len = self.n_c4_pages * (self.page_size // 4)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def _token_layout(self, reqs) -> tuple[list[int], list[int], list[int], list[int]]:
        tok_table, tok_cached, seq_rows = [], [], []
        cu = [0]
        for req in reqs:
            n = int(req.device_len) - int(req.cached_len)
            tok_table.extend([int(req.table_idx)] * n)
            tok_cached.extend([int(req.cached_len)] * n)
            cu.append(cu[-1] + n)
        return tok_table, tok_cached, cu, seq_rows

    def _build(self, batch: Batch, out: Dict[str, torch.Tensor] | None = None) -> DSV4Metadata:
        reqs = batch.padded_reqs
        dev = self.device
        pool = self.pool
        W = self.window
        page_table = get_global_ctx().page_table
        is_decode = bool(batch.is_decode)
        tok_table_l, tok_cached_l, cu_l, _ = self._token_layout(reqs)
        T = len(tok_table_l)
        CPU = {"dtype": torch.int32, "pin_memory": True}
        tok_table = torch.tensor(tok_table_l, **CPU).to(dev, non_blocking=True)
        tok_cached = torch.tensor(tok_cached_l, **CPU).to(dev, non_blocking=True)
        cu = torch.tensor(cu_l, **CPU).to(dev, non_blocking=True)
        # positions from the requests themselves: AFD builds one metadata object
        # per microbatch from a request subset while batch.positions spans all
        pos_l: list[int] = []
        for req in reqs:
            pos_l.extend(range(int(req.cached_len), int(req.device_len)))
        pos = torch.tensor(pos_l, **CPU).to(dev, non_blocking=True)
        t_idx = torch.arange(T, device=dev, dtype=torch.int64)
        scratch_rows = pool.scratch_base + t_idx
        # window rows: positions p - j, j in [0, W)
        p = pos[:, None] - self._j[None, :]
        valid = p >= 0
        in_new = p >= tok_cached[:, None]
        scratch_w = (pool.scratch_base + t_idx[:, None] - self._j[None, :].long()).to(torch.int32)
        ring_w = (tok_table[:, None] * W + torch.remainder(p, W)).to(torch.int32)
        window_idx = torch.where(valid, torch.where(in_new, scratch_w, ring_w), -1).to(torch.int32)
        # scratch -> ring copy for the last W tokens of each request
        dev_len_tok = []
        for req in reqs:
            n = int(req.device_len) - int(req.cached_len)
            dev_len_tok.extend([int(req.device_len)] * n)
        if is_decode:
            ring_src = scratch_rows
            ring_dst = (tok_table.long() * W + torch.remainder(pos, W).long())
        else:
            dlen = torch.tensor(dev_len_tok, **CPU).to(dev, non_blocking=True)
            keep = pos >= (dlen - W)
            ring_src = scratch_rows[keep]
            ring_dst = (tok_table.long() * W + torch.remainder(pos, W).long())[keep]
        tt = tok_table.long()
        # C128: all completed groups
        c128_idx = None
        if any(r == 128 for r in self.ratios):
            cols = (self._c128.long() * 128).clamp(max=page_table.shape[1] - 1)
            slots = page_table[tt[:, None], cols[None, :]]
            v128 = self._c128[None, :] < ((pos + 1) // 128)[:, None]
            c128_idx = torch.where(v128, pool.comp_base + slots // 128, -1).to(torch.int32)
        n4_valid = ((pos + 1) // 4).to(torch.int32)
        n128_valid = ((pos + 1) // 128).to(torch.int32)
        n_window = (pos + 1).clamp(max=W).to(torch.int32)
        c4_rows = None
        c4_bt = dg_sched = topk_plan = None
        if is_decode and any(r == 4 for r in self.ratios):
            if self.indexer_impl == "sglang":
                from minisgl.kernel.dsv4_sgl import paged_mqa_schedule, plan_topk_v2

                P = self.page_size
                c4_bt = torch.div(page_table[:, : self.n_c4_pages * P : P][tt], P, rounding_mode="floor").to(torch.int32)
                if c4_bt.shape[1] < self.n_c4_pages:
                    c4_bt = torch.nn.functional.pad(c4_bt, (0, self.n_c4_pages - c4_bt.shape[1]))
                c4_lens = ((pos + 1) // 4).to(torch.int32)
                dg_sched = paged_mqa_schedule(c4_lens)
                topk_plan = plan_topk_v2(c4_lens)
            else:
                cols4 = (self._c4.long() * 4).clamp(max=page_table.shape[1] - 1)
                c4_rows = (page_table[tt[:, None], cols4[None, :]] // 4).long()
        comp_write: Dict[int, torch.Tensor] = {}
        group_start: Dict[int, torch.Tensor] = {}
        for r in sorted({x for x in self.ratios if x}):
            complete = ((pos + 1) % r) == 0
            c = torch.div(pos, r, rounding_mode="floor").long()
            slot = page_table[tt, (c * r).clamp(max=page_table.shape[1] - 1)]
            comp_write[r] = torch.where(complete, (slot // r).long(), -1)
            group_start[r] = (pos + 1 - r).clamp(min=0).to(torch.int32)
        seq_lens = (pos + 1).to(torch.int32)
        prefill_seqs: List[DSV4PrefillSeq] = []
        if not is_decode:
            for i, req in enumerate(reqs):
                prefill_seqs.append(
                    DSV4PrefillSeq(cu_l[i], cu_l[i + 1], int(req.table_idx), int(req.cached_len), int(req.device_len))
                )
        md = DSV4Metadata(
            is_decode=is_decode,
            num_tokens=T,
            tok_table=tok_table,
            tok_cached=tok_cached,
            scratch_rows=scratch_rows,
            window_idx=window_idx,
            ring_src=ring_src,
            ring_dst=ring_dst,
            c128_idx=c128_idx,
            c4_rows=c4_rows,
            n4_valid=n4_valid,
            n128_valid=n128_valid,
            n_window=n_window,
            comp_write=comp_write,
            group_start=group_start,
            seq_lens=seq_lens,
            cu_seqlens_q=cu,
            prefill_seqs=prefill_seqs,
            c4_bt=c4_bt,
            dg_sched=dg_sched,
            topk_plan=topk_plan,
        )
        return md

    def prepare_metadata(self, batch: Batch) -> None:
        batch.attn_metadata = self._build(batch)

    # ------------------------------------------------------------------
    # CUDA graph
    # ------------------------------------------------------------------
    _GRAPH_FIELDS = (
        "tok_table", "tok_cached", "scratch_rows", "window_idx", "ring_src", "ring_dst",
        "c128_idx", "c4_rows", "n4_valid", "n128_valid", "n_window", "seq_lens", "cu_seqlens_q",
        "c4_bt", "dg_sched", "topk_plan",
    )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        self._capture_meta = {}

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        md = self._build(batch)
        # private persistent copies (graph reads these addresses)
        for name in self._GRAPH_FIELDS:
            t = getattr(md, name)
            if t is not None:
                setattr(md, name, t.clone())
        md.comp_write = {k: v.clone() for k, v in md.comp_write.items()}
        md.group_start = {k: v.clone() for k, v in md.group_start.items()}
        self._capture_meta[bs] = md
        batch.attn_metadata = md

    def prepare_for_replay(self, batch: Batch) -> None:
        bs = batch.padded_size
        cap = self._capture_meta[bs]
        real = batch.attn_metadata
        assert isinstance(real, DSV4Metadata)
        for name in self._GRAPH_FIELDS:
            dst = getattr(cap, name)
            if dst is not None:
                dst.copy_(getattr(real, name))
        for k in cap.comp_write:
            cap.comp_write[k].copy_(real.comp_write[k])
            cap.group_start[k].copy_(real.group_start[k])

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    def forward(self, q, k, v, layer_id, batch):  # pragma: no cover - interface only
        raise RuntimeError("DSV4AttnBackend is driven by the DSV4 attention modules")

    @staticmethod
    def attn_index(
        md: DSV4Metadata, comp: torch.Tensor | None, comp_count: torch.Tensor | None, comp_off: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[window | compressed] row lists compacted so the valid entries form a
        prefix (the sparse kernel does not skip -1 holes), padded with -1 to a
        multiple of 64 columns, plus valid counts. ``comp`` holds a valid prefix
        of ``comp_count`` (clamped to its width) entries; ``comp_off`` is added
        to valid compressed rows. One Triton kernel."""
        win = md.window_idx
        T, W = win.shape
        Kc = 0 if comp is None else comp.shape[1]
        Kp = _pad_to(W + Kc, 64)
        out = torch.empty((T, Kp), dtype=torch.int32, device=win.device)
        seq_lens = torch.empty((T,), dtype=torch.int32, device=win.device)
        if T:
            _assemble_index_kernel[(T,)](
                win, md.n_window, comp if comp is not None else win, comp_count if comp is not None else md.n_window,
                out, seq_lens, W, Kc, Kp, win.stride(0), comp.stride(0) if comp is not None else 0, int(comp_off),
                HAS_COMP=comp is not None, BLOCK=min(1024, triton.next_power_of_2(Kp)), num_warps=4,
            )
        return out, seq_lens

    def sparse_attn(
        self,
        q: torch.Tensor,
        layer_kv: torch.Tensor,
        idx: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """q [T, H, 512] bf16; idx [T, K] int32 rows of ``layer_kv`` (-1 invalid)."""
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        T, H, D = q.shape
        K = idx.shape[1]
        Kp = _pad_to(K, 64)
        if Kp != K:
            idx = torch.cat([idx, idx.new_full((T, Kp - K), -1)], dim=1)
        idx = idx.to(torch.int32)
        # flashinfer picks the low-latency (16-head tile) sparse kernel below 128
        # heads; zero-padding the query heads to 128 selects the 2-CTA
        # high-throughput kernel, which wins once the batch x top-k is large
        Hp = 128 if (H < 128 and T * Kp >= (16384 if H >= 64 else 65536)) else H
        q576 = torch.empty((T, 1, Hp, 576), dtype=q.dtype, device=q.device)
        if T:
            _pad_q_kernel[(T * Hp,)](q, q576, H, Hp, q.stride(0), q.stride(1), D=D, DP=576, BLOCK=1024, num_warps=4)
        if Hp != H:
            key = (sinks.data_ptr(), Hp)
            sp = self._sinks_pad.get(key)
            if sp is None:
                sp = torch.zeros(Hp, dtype=sinks.dtype, device=sinks.device)
                sp[:H] = sinks
                self._sinks_pad[key] = sp
            sinks = sp
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q576,
            kv_cache=layer_kv.view(-1, 1, 64, 576),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=idx.view(T, 1, Kp),
            seq_lens=seq_lens,
            max_seq_len=int(layer_kv.shape[0]),
            sparse_mla_top_k=Kp,
            bmm1_scale=float(scale),
            bmm2_scale=1.0,
            sinks=sinks,
            backend="trtllm-gen",
        )
        return out.view(T, Hp, 512)[:, :H]


__all__ = ["DSV4AttnBackend", "DSV4Metadata", "DSV4PrefillSeq"]
