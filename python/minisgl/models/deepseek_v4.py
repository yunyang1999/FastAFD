"""DeepSeek-V4 (Flash / Pro) for minisgl.

Architecture (official ``inference/model.py``): manifold-constrained
hyper-connections (``hc_mult`` residual copies mixed by Sinkhorn-normalised
weights), MQA attention with a 512-d latent (k = v, RoPE on the last 64 dims),
sliding window (128) plus per-layer KV compression (ratio 0 / 4 / 128) where
ratio-4 layers select compressed entries with a lightning indexer, grouped
low-rank output projection, and a MoE with sqrt-softplus routing (first
``num_hash_layers`` layers hash-routed by token id), 1 shared expert and a
clamped SwiGLU.

The same module tree serves the colocated engine (``forward``) and the AFD
attention stage (``forward_attention`` / ``route`` / ``finalize_hidden``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.layers.linear import _LinearTPImpl
from minisgl.utils import debug_taps, div_even, nvtx_annotate

from .base import BaseLLMModel
from .moe_common import (
    ActivationSpec,
    ColocatedMegaMoE,
    FP4RoutedExperts,
    moe_exec_backend,
    quant_mxfp4,
    torch_ref_moe,
)

if TYPE_CHECKING:
    from .config import ModelConfig


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSV4Args:
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    ratios: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_iters: int
    hc_eps: float
    eps: float
    vocab: int
    n_routed: int
    n_shared: int
    topk: int
    moe_inter: int
    rsf: float
    swiglu_limit: float
    n_hash_layers: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_seq_len: int
    beta_fast: int
    beta_slow: int

    @classmethod
    def from_config(cls, mc: "ModelConfig") -> "DSV4Args":
        d = mc.hf_text_config
        rs = d.get("rope_scaling") or {}
        return cls(
            dim=int(d["hidden_size"]),
            n_layers=int(mc.num_layers),
            n_heads=int(d["num_attention_heads"]),
            head_dim=int(d["head_dim"]),
            rope_dim=int(d["qk_rope_head_dim"]),
            q_lora_rank=int(d["q_lora_rank"]),
            o_groups=int(d["o_groups"]),
            o_lora_rank=int(d["o_lora_rank"]),
            window=int(d.get("sliding_window", 128)),
            ratios=tuple(int(r) for r in d["compress_ratios"][: int(mc.num_layers)]),
            index_n_heads=int(d["index_n_heads"]),
            index_head_dim=int(d["index_head_dim"]),
            index_topk=int(d["index_topk"]),
            hc_mult=int(d["hc_mult"]),
            hc_iters=int(d["hc_sinkhorn_iters"]),
            hc_eps=float(d["hc_eps"]),
            eps=float(d["rms_norm_eps"]),
            vocab=int(d["vocab_size"]),
            n_routed=int(d["n_routed_experts"]),
            n_shared=int(d.get("n_shared_experts", 1)),
            topk=int(d["num_experts_per_tok"]),
            moe_inter=int(d["moe_intermediate_size"]),
            rsf=float(d.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(d.get("swiglu_limit", 0.0) or 0.0),
            n_hash_layers=int(d.get("num_hash_layers", 0) or 0),
            rope_theta=float(d.get("rope_theta", 10000.0)),
            compress_rope_theta=float(d.get("compress_rope_theta", 160000.0)),
            rope_factor=float(rs.get("factor", 1.0)),
            original_seq_len=int(rs.get("original_max_position_embeddings", 0)),
            beta_fast=int(rs.get("beta_fast", 32)),
            beta_slow=int(rs.get("beta_slow", 1)),
        )


def rope_table_len() -> int:
    return int(os.environ.get("MINISGL_ROPE_TABLE_LEN", str(1 << 17)))


@lru_cache(maxsize=8)
def _freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device_str):
    """precompute_freqs_cis of the official reference, returned as (cos, sin) fp32."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn), 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    f = torch.outer(t, freqs)
    return f.cos().to(device_str), f.sin().to(device_str)


def layer_rope(args: DSV4Args, ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    if ratio:
        orig, base = args.original_seq_len, args.compress_rope_theta
    else:
        orig, base = 0, args.rope_theta
    return _freqs(args.rope_dim, rope_table_len(), orig, base, args.rope_factor, args.beta_fast, args.beta_slow, str(device))


@lru_cache(maxsize=4)
def hadamard_matrix(n: int, device_str: str) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * n**-0.5).to(device_str).contiguous()


# ---------------------------------------------------------------------------
# small layers
# ---------------------------------------------------------------------------


class LinearReplicatedQ(_LinearTPImpl):
    """Replicated linear that may hold FP8 block-quantised weights."""

    def __init__(self, in_features: int, out_features: int, quant=None, dtype: torch.dtype | None = None):
        super().__init__(in_features, out_features, in_features, out_features, False, quant=quant)
        if dtype is not None and quant is None:
            self.weight = torch.empty(out_features, in_features, dtype=dtype)


class Fp32Linear(BaseOP):
    """y = x.float() @ W^T with an fp32 weight (reference: gate, compressor)."""

    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.empty(out_features, in_features, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


# ---------------------------------------------------------------------------
# compressor / indexer
# ---------------------------------------------------------------------------


class DSV4Compressor(BaseOP):
    def __init__(self, args: DSV4Args, ratio: int, head_dim: int, rotate: bool):
        self.ratio = ratio
        self.overlap = ratio == 4
        self.head_dim = head_dim
        self.rotate = rotate
        coff = 2 if self.overlap else 1
        self.coff = coff
        self.wkv = Fp32Linear(args.dim, coff * head_dim)
        self.wgate = Fp32Linear(args.dim, coff * head_dim)
        self.ape = torch.empty(ratio, coff * head_dim, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, eps=args.eps)
        self._eps = args.eps
        self._rope_dim = args.rope_dim

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("use run()")

    def run(
        self,
        x: torch.Tensor,
        md,
        cos: torch.Tensor,
        sin: torch.Tensor,
        state_kv: torch.Tensor,
        state_sc: torch.Tensor,
        out_buf: torch.Tensor,
        row_base: int,
    ) -> None:
        """Update compressor state with this forward's tokens and write every
        completed compressed entry into ``out_buf[row_base + row]``."""
        from minisgl.kernel.dsv4_triton import compressor_decode_step, kv_norm_rope_store

        ratio, d = self.ratio, self.head_dim
        quant = 2 if self.rotate else 1
        had = hadamard_matrix(d, str(x.device)) if self.rotate else None
        kv = self.wkv.forward(x)
        sc = self.wgate.forward(x)
        pos = get_global_ctx().batch.positions
        if md.is_decode:
            pooled = compressor_decode_step(
                kv, sc, self.ape, pos, md.tok_table, state_kv, state_sc,
                ratio=ratio, overlap=self.overlap, head_dim=d,
            )
            rows = md.comp_write[ratio]
            rows = torch.where(rows >= 0, rows + row_base, rows)
            kv_norm_rope_store(
                pooled, self.norm.weight, md.group_start[ratio], cos, sin, out_buf, rows,
                eps=self._eps, rope_dim=self._rope_dim, quant=quant, hadamard=had,
            )
            return
        pooled_all, rows_all, gpos_all = [], [], []
        page_table = get_global_ctx().page_table
        for seq in md.prefill_seqs:
            res = self._prefill_seq(kv[seq.tok_start : seq.tok_end], sc[seq.tok_start : seq.tok_end], seq, state_kv, state_sc)
            if res is None:
                continue
            pooled, gpos = res
            rows = page_table[seq.table_idx, gpos.long()].long() // ratio + row_base
            pooled_all.append(pooled)
            rows_all.append(rows)
            gpos_all.append(gpos)
        if pooled_all:
            kv_norm_rope_store(
                torch.cat(pooled_all), self.norm.weight, torch.cat(gpos_all).to(torch.int32), cos, sin,
                out_buf, torch.cat(rows_all), eps=self._eps, rope_dim=self._rope_dim, quant=quant, hadamard=had,
            )

    def _prefill_seq(self, kv, sc, seq, state_kv, state_sc):
        ratio, d, ov = self.ratio, self.head_dim, self.overlap
        L0, L1, r = seq.cached_len, seq.device_len, seq.table_idx
        n = L1 - L0
        if n <= 0:
            return None
        dev = kv.device
        positions = torch.arange(L0, L1, device=dev)
        sc = sc + self.ape[positions % ratio]
        base = ratio if ov else 0
        g0 = L0 // ratio
        first = g0 * ratio
        pre_n = L0 - first
        if pre_n > 0:
            kv = torch.cat([state_kv[r, base : base + pre_n], kv], 0)
            sc = torch.cat([state_sc[r, base : base + pre_n], sc], 0)
        n_all = L1 - first
        n_groups = n_all // ratio
        pooled = gpos = None
        kv_g = sc_g = None
        if n_groups > 0:
            kv_g = kv[: n_groups * ratio].view(n_groups, ratio, -1)
            sc_g = sc[: n_groups * ratio].view(n_groups, ratio, -1)
            if ov:
                if g0 == 0:
                    prev_kv = torch.zeros_like(kv_g[:1])
                    prev_sc = torch.full_like(sc_g[:1], float("-inf"))
                else:
                    prev_kv = state_kv[r, :ratio].unsqueeze(0)
                    prev_sc = state_sc[r, :ratio].unsqueeze(0)
                prev_kv = torch.cat([prev_kv, kv_g[:-1]], 0)
                prev_sc = torch.cat([prev_sc, sc_g[:-1]], 0)
                kvs = torch.cat([prev_kv[..., :d], kv_g[..., d:]], 1)
                scs = torch.cat([prev_sc[..., :d], sc_g[..., d:]], 1)
            else:
                kvs, scs = kv_g, sc_g
            pooled = (kvs * scs.softmax(dim=1)).sum(dim=1)
            gpos = (g0 + torch.arange(n_groups, device=dev)) * ratio
        rem = n_all - n_groups * ratio
        if rem > 0:
            state_kv[r, base : base + rem] = kv[n_groups * ratio :]
            state_sc[r, base : base + rem] = sc[n_groups * ratio :]
        if ov:
            if n_groups > 0:
                state_kv[r, :ratio] = kv_g[-1]
                state_sc[r, :ratio] = sc_g[-1]
            elif g0 == 0:
                state_kv[r, :ratio] = 0
                state_sc[r, :ratio] = float("-inf")
        if pooled is None:
            return None
        return pooled, gpos


class DSV4Indexer(BaseOP):
    def __init__(self, args: DSV4Args, quant):
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.topk = args.index_topk
        self.wq_b = LinearReplicatedQ(args.q_lora_rank, self.n_heads * self.head_dim, quant=quant)
        self.weights_proj = LinearReplicatedQ(args.dim, self.n_heads, dtype=torch.bfloat16)
        self.compressor = DSV4Compressor(args, 4, self.head_dim, rotate=True)
        self._scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self._rope_dim = args.rope_dim

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("use run()")

    def run(self, x, qr, md, cos, sin, pool, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (absolute kv rows of the selected compressed entries [T, k], counts [T])."""
        from minisgl.kernel.dsv4_triton import q_norm_rope_

        dev = x.device
        T = x.shape[0]
        had = hadamard_matrix(self.head_dim, str(dev))
        q = self.wq_b.forward(qr).view(T, self.n_heads, self.head_dim)
        pos = get_global_ctx().batch.positions
        q_norm_rope_(q, pos, cos, sin, eps=0.0, rope_dim=self._rope_dim, do_norm=False, hadamard=had)
        # reference precision: bf16 scores / weights (einsum of bf16 tensors)
        w = self.weights_proj.forward(x) * self._scale  # [T, H] bf16
        keys = pool.idx_keys[layer_id]
        self.compressor.run(
            x, md, cos, sin, pool.idx_state_kv[layer_id], pool.idx_state_score[layer_id], keys, 0
        )
        page_table = get_global_ctx().page_table
        if md.is_decode:
            k_all = keys[md.c4_rows]  # [T, n4, D]
            s = torch.bmm(q, k_all.transpose(1, 2))  # [T, H, n4] bf16
            s = (s.relu_() * w.unsqueeze(-1)).sum(1).float()  # [T, n4]
            n4 = s.shape[1]
            valid = torch.arange(n4, device=dev)[None, :] < md.n4_valid[:, None]
            s = s.masked_fill(~valid, float("-inf"))
            k = min(self.topk, n4)
            vals, idx = s.topk(k, dim=-1)
            rows = md.c4_rows.gather(1, idx)
            rows = torch.where(torch.isfinite(vals), rows + pool.comp_base, -1).to(torch.int32)
            cnt = torch.clamp(md.n4_valid, max=k)
            return rows, cnt
        # prefill: per request, chunked over tokens
        k_cap = self.topk
        out_rows = torch.full((T, k_cap), -1, dtype=torch.int32, device=dev)
        for seq in md.prefill_seqs:
            n_c = seq.device_len // 4
            if n_c == 0:
                continue
            crow = page_table[seq.table_idx, torch.arange(n_c, device=dev) * 4].long() // 4
            ks = keys[crow]  # [n_c, D] bf16
            for c0 in range(seq.tok_start, seq.tok_end, 256):
                c1 = min(c0 + 256, seq.tok_end)
                s = torch.einsum("thd,cd->thc", q[c0:c1], ks)
                s = (s.relu_() * w[c0:c1].unsqueeze(-1)).sum(1).float()  # [t, n_c]
                nv = (pos[c0:c1] + 1) // 4
                valid = torch.arange(n_c, device=dev)[None, :] < nv[:, None]
                s = s.masked_fill(~valid, float("-inf"))
                k = min(k_cap, n_c)
                vals, idx = s.topk(k, dim=-1)
                rows = torch.where(torch.isfinite(vals), crow[idx] + pool.comp_base, -1).to(torch.int32)
                out_rows[c0:c1, :k] = rows
        cnt = torch.clamp(md.n4_valid, max=k_cap)
        return out_rows, cnt


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


class DSV4Attention(BaseOP):
    def __init__(self, args: DSV4Args, layer_id: int, quant):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.ratio = args.ratios[layer_id]
        self.n_heads = args.n_heads
        self.n_local = div_even(args.n_heads, tp.size)
        self.n_groups_local = div_even(args.o_groups, tp.size)
        self.head_dim = args.head_dim
        self.o_lora = args.o_lora_rank
        self._eps = args.eps
        self._rope_dim = args.rope_dim
        self._args = args
        self._tp_rank = tp.rank
        self.wq_a = LinearReplicatedQ(args.dim, args.q_lora_rank, quant=quant)
        self.q_norm = RMSNorm(args.q_lora_rank, eps=args.eps)
        self.wq_b = LinearColParallelMerged(args.q_lora_rank, [args.n_heads * args.head_dim], has_bias=False, quant=quant)
        self.wkv = LinearReplicatedQ(args.dim, args.head_dim, quant=quant)
        self.kv_norm = RMSNorm(args.head_dim, eps=args.eps)
        in_g = args.n_heads * args.head_dim // args.o_groups
        self.wo_a = torch.empty(self.n_groups_local * args.o_lora_rank, in_g, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(args.o_groups * args.o_lora_rank, args.dim, has_bias=False, quant=quant)
        self.attn_sink = torch.empty(self.n_local, dtype=torch.float32)
        self.compressor = DSV4Compressor(args, self.ratio, args.head_dim, rotate=False) if self.ratio else None
        self.indexer = DSV4Indexer(args, quant) if self.ratio == 4 else None

    @nvtx_annotate("DSV4Attn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel.dsv4_triton import inverse_rope_, kv_norm_rope_store, q_norm_rope_

        ctx = get_global_ctx()
        backend = ctx.attn_backend
        pool = backend.pool
        md = ctx.batch.attn_metadata
        pos = ctx.batch.positions
        T = x.shape[0]
        l = self.layer_id
        cos, sin = layer_rope(self._args, self.ratio, x.device)
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).view(T, self.n_local, self.head_dim)
        q_norm_rope_(q, pos, cos, sin, eps=self._eps, rope_dim=self._rope_dim)
        kv = self.wkv.forward(x)
        layer_kv = pool.kv[l]
        kv_norm_rope_store(
            kv, self.kv_norm.weight, pos, cos, sin, layer_kv, md.scratch_rows,
            eps=self._eps, rope_dim=self._rope_dim, quant=1,
        )
        comp = cnt = None
        if self.ratio:
            self.compressor.run(x, md, cos, sin, pool.state_kv[l], pool.state_score[l], layer_kv, pool.comp_base)
            if self.ratio == 4:
                comp, cnt = self.indexer.run(x, qr, md, cos, sin, pool, l)
            else:
                comp, cnt = md.c128_idx, md.n128_valid
        idx, seq_lens = backend.attn_index(md, comp, cnt)
        o = backend.sparse_attn(q, layer_kv, idx, self.attn_sink, self.head_dim**-0.5, seq_lens)
        # persist this forward's window kv into the per-request ring
        if md.ring_src.numel():
            layer_kv.index_copy_(0, md.ring_dst, layer_kv.index_select(0, md.ring_src))
        inverse_rope_(o, pos, cos, sin, self._rope_dim)
        o = o.reshape(T, self.n_groups_local, -1)
        wo_a = self.wo_a.view(self.n_groups_local, self.o_lora, -1)
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        return self.wo_b.forward(o.flatten(1))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class DSV4TopK(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    routed_scaling_factor: float = 1.0
    skip_moe: bool = False
    local_output: torch.Tensor | None = None
    shared_output: torch.Tensor | None = None
    dispatch_fp8: bool = True
    external_quant: bool = False


class DSV4Gate(BaseOP):
    def __init__(self, args: DSV4Args, layer_id: int):
        self.weight = torch.empty(args.n_routed, args.dim, dtype=torch.float32)
        self.is_hash = layer_id < args.n_hash_layers
        if self.is_hash:
            self.tid2eid = torch.empty(args.vocab, args.topk, dtype=torch.int32)
        else:
            self.bias = torch.empty(args.n_routed, dtype=torch.float32)
        self._topk = args.topk

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (ids int32 [T,k], normalised weights fp32 [T,k]) without rsf."""
        scores = F.softplus(F.linear(x.float(), self.weight)).sqrt()
        if self.is_hash:
            ids = self.tid2eid[get_global_ctx().batch.input_ids.long()].long()
        else:
            ids = (scores + self.bias).topk(self._topk, dim=-1).indices
        w = scores.gather(1, ids)
        w = w / w.sum(dim=-1, keepdim=True)
        return ids.to(torch.int32), w


class DSV4SharedExpert(BaseOP):
    def __init__(self, args: DSV4Args, quant):
        self.gate_up = LinearReplicatedQ(args.dim, 2 * args.moe_inter, quant=quant)
        self.down = LinearReplicatedQ(args.moe_inter, args.dim, quant=quant)
        self._inter = args.moe_inter
        self._limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gate_up.forward(x)
        a = ActivationSpec("swiglu", clamp=self._limit)(h[:, : self._inter], h[:, self._inter :])
        return self.down.forward(a.to(x.dtype))

    def as_mxfp4(self) -> tuple[torch.Tensor, ...]:
        """Shared expert as one MXFP4 expert (for the MegaMoE stage)."""
        w13 = _dense_weight_bf16(self.gate_up)
        w2 = _dense_weight_bf16(self.down)
        p13, s13 = quant_mxfp4(w13)
        p2, s2 = quant_mxfp4(w2)
        return p13.unsqueeze(0), s13.unsqueeze(0), p2.unsqueeze(0), s2.unsqueeze(0)


def _dense_weight_bf16(lin: _LinearTPImpl) -> torch.Tensor:
    w = lin.weight
    if w.dtype == torch.float8_e4m3fn:
        return lin._dequant_weight_to_bf16() if lin._bf16_weight is None else lin._bf16_weight
    return w.to(torch.bfloat16)


class DSV4MoE(BaseOP):
    def __init__(self, args: DSV4Args, layer_id: int, quant, *, with_experts: bool = True, with_shared: bool = True):
        tp = get_tp_info()
        self.gate = DSV4Gate(args, layer_id)
        self.shared_experts = DSV4SharedExpert(args, quant) if with_shared else None
        self.n_local = args.n_routed // tp.size
        self.experts = FP4RoutedExperts(self.n_local, args.dim, args.moe_inter) if with_experts else None
        self._args = args
        self._layer_id = layer_id
        self._expert_offset = tp.rank * self.n_local
        self._tp_size = tp.size
        self._act = ActivationSpec("swiglu", clamp=args.swiglu_limit)

    def forward(self, x: torch.Tensor, executor: ColocatedMegaMoE | None) -> torch.Tensor:
        ids, w = self.gate.forward(x)
        debug_taps.tap(f"L{self._layer_id}.topk_ids", ids)
        if executor is not None:
            # colocated MegaMoE runs the routed experts; the shared expert runs
            # locally (the synthetic shared slot of the colocated kernel faults
            # at EP=1, see docs/dsv4_k3_design.md)
            y = executor(x, ids, w, self.experts.megamoe())
            if self.shared_experts is not None:
                y = (y.float() + self.shared_experts.forward(x).float()).to(x.dtype)
            return y
        w13, w2 = self.experts.bf16()
        y = torch_ref_moe(x, ids, w, w13, w2, self._act, expert_offset=self._expert_offset)
        y = y.float() * self._args.rsf
        if self._tp_size > 1:
            y = DistributedCommunicator().all_reduce(y)
        if self.shared_experts is not None:
            y = y + self.shared_experts.forward(x).float()
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# block / model
# ---------------------------------------------------------------------------


@dataclass
class DSV4Residual:
    x: torch.Tensor  # [T, hc, D]
    post: torch.Tensor  # [T, hc]
    comb: torch.Tensor  # [T, hc, hc]


class DSV4Block(BaseOP):
    def __init__(self, args: DSV4Args, layer_id: int, quant, *, afd_attention_side: bool = False):
        mix = (2 + args.hc_mult) * args.hc_mult
        hd = args.hc_mult * args.dim
        self.hc_attn_fn = torch.empty(mix, hd, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix, hd, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)
        self.attn_norm = RMSNorm(args.dim, eps=args.eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.eps)
        self.attn = DSV4Attention(args, layer_id, quant)
        self.ffn = DSV4MoE(
            args, layer_id, quant,
            with_experts=not afd_attention_side,
            with_shared=not afd_attention_side,
        )
        self._args = args
        self._layer_id = layer_id

    def hc_pre(self, x, fn, scale, base):
        from minisgl.kernel.dsv4_triton import hc_pre_fused, hc_prenorm_mixes

        mixes = hc_prenorm_mixes(x, fn, self._args.eps)
        return hc_pre_fused(x, mixes, scale, base, sinkhorn_iters=self._args.hc_iters, eps=self._args.hc_eps)

    def attention_part(self, x: torch.Tensor) -> tuple[torch.Tensor, DSV4Residual]:
        from minisgl.kernel.dsv4_triton import hc_post_fused

        h, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn_norm.forward(h)
        debug_taps.tap(f"L{self._layer_id}.attn_in", h)
        a = self.attn.forward(h)
        debug_taps.tap(f"L{self._layer_id}.attn_out", a)
        x = hc_post_fused(a, x, post, comb)
        h, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn_norm.forward(h)
        return h, DSV4Residual(x, post, comb)

    @nvtx_annotate("DSV4Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, executor) -> torch.Tensor:
        from minisgl.kernel.dsv4_triton import hc_post_fused

        h, res = self.attention_part(x)
        debug_taps.tap(f"L{self._layer_id}.moe_in", h)
        m = self.ffn.forward(h, executor)
        debug_taps.tap(f"L{self._layer_id}.moe_out", m)
        return hc_post_fused(m, res.x, res.post, res.comb)


class DSV4Model(BaseOP):
    def __init__(self, config: "ModelConfig", *, afd_attention_side: bool = False):
        args = DSV4Args.from_config(config)
        quant = config.quant
        self.embed = VocabParallelEmbedding(args.vocab, args.dim)
        self.layers = OPList(
            [DSV4Block(args, i, quant, afd_attention_side=afd_attention_side) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.hc_head_fn = torch.empty(args.hc_mult, args.hc_mult * args.dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(args.hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)
        self._args = args

    def embed_hc(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed.forward(input_ids)
        return h.unsqueeze(1).repeat(1, self._args.hc_mult, 1).contiguous()

    def head_hidden(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel.dsv4_triton import hc_head_fused, hc_prenorm_mixes

        mixes = hc_prenorm_mixes(x, self.hc_head_fn, self._args.eps)
        y = hc_head_fused(x, mixes, self.hc_head_scale, self.hc_head_base, self._args.hc_eps)
        return self.norm.forward(y)

    def forward(self, input_ids: torch.Tensor, executor) -> torch.Tensor:
        x = self.embed_hc(input_ids)
        for layer in self.layers.op_list:
            x = layer.forward(x, executor)
        return self.head_hidden(x)


class DeepseekV4ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = DSV4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
        )
        self._config = config
        self._executor: ColocatedMegaMoE | None = None
        super().__init__()

    def post_load(self, *, max_tokens_per_rank: int) -> None:
        """Build the colocated MoE executor (MegaMoE) after weights are loaded."""
        args = self.model._args
        if os.environ.get("MINISGL_DSV4_SHARED_MXFP4", "0") == "1":
            # Emulate the AFD placement, where the shared expert runs inside the
            # MegaMoE stage with MXFP4 weights (re-quantised from FP8).
            from .moe_common import dequant_mxfp4

            for layer in self.model.layers.op_list:
                sh = layer.ffn.shared_experts
                if sh is None:
                    continue
                p13, s13, p2, s2 = sh.as_mxfp4()
                for lin, (p, sf) in ((sh.gate_up, (p13[0], s13[0])), (sh.down, (p2[0], s2[0]))):
                    lin._bf16_weight = dequant_mxfp4(p, sf)
                    lin._use_bf16_cublas_for_fp8 = True
        if moe_exec_backend() == "megamoe":
            self._executor = ColocatedMegaMoE(
                num_routed=args.n_routed,
                hidden=args.dim,
                intermediate=args.moe_inter,
                top_k=args.topk,
                num_shared=0,
                routed_scaling_factor=args.rsf,
                activation_clamp=args.swiglu_limit or None,
                max_tokens_per_rank=max_tokens_per_rank,
            )
            for layer in self.model.layers.op_list:
                layer.ffn.experts.megamoe()
                layer.ffn.experts.drop_packed()
            torch.cuda.empty_cache()
        else:
            for layer in self.model.layers.op_list:
                layer.ffn.experts.bf16()
                layer.ffn.experts.drop_packed()
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        h = self.model.forward(ctx.batch.input_ids, self._executor)
        return self.lm_head.forward(h)


__all__ = ["DeepseekV4ForCausalLM", "DSV4Model", "DSV4Args", "DSV4TopK", "DSV4Residual"]
