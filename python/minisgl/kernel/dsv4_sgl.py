"""DeepSeek-V4 decode kernels as run by sglang 0.5.18 on SM100.

* JIT CUDA kernels vendored from sglang (``sgl_jit/csrc/deepseek_v4``):
  main-q RMS-norm + RoPE, indexer-q RoPE + Hadamard + FP8 quant (head weights
  folded with the q scale), in-place (inverse) RoPE, and the top-k v2 fused
  top-512 selection + page-table transform.
* DeepGEMM (minisgl's vendored build): ``fp8_paged_mqa_logits`` and its schedule
  metadata over the paged FP8 indexer-key cache (64 entries x (128 FP8 + FP32
  scale) per page, values first then scales -- DeepGEMM's fused layout).
* A Triton store for indexer keys into that cache (norm + RoPE + Hadamard +
  per-token FP8 quant, matching the q-side quantisation).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from minisgl.kernel.sgl_jit import cpp_args, load_jit, supports_pdl

INDEX_DIM = 128
INDEX_BLOCK = 64  # c4 entries per indexer-cache page (page_size 256 tokens / 4)
INDEX_ROW_BYTES = INDEX_DIM + 4


# ---------------------------------------------------------------------------
# rope table in sglang's (re, im)-interleaved fp32 layout
# ---------------------------------------------------------------------------


_FREQS: dict[tuple, torch.Tensor] = {}


def freqs_real(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """[P, R/2] cos / sin -> [P, R] fp32 with (cos, sin) pairs interleaved."""
    key = (cos.data_ptr(), sin.data_ptr(), tuple(cos.shape))
    t = _FREQS.get(key)
    if t is None:
        t = torch.stack([cos.float(), sin.float()], dim=-1).reshape(cos.shape[0], -1).contiguous()
        _FREQS[key] = t
    return t


# ---------------------------------------------------------------------------
# JIT modules
# ---------------------------------------------------------------------------


@functools.cache
def _q_norm_rope_module(head_dim: int, rope_dim: int):
    args = cpp_args(torch.bfloat16, head_dim, rope_dim, supports_pdl())
    return load_jit(
        "dsv4_main_q_norm_rope", args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQNormRopeKernel<{args}>::forward")],
    )


@functools.cache
def _q_indexer_module():
    args = cpp_args(torch.bfloat16, supports_pdl())
    return load_jit(
        "dsv4_main_q_indexer_rope_hadamard_quant", args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward")],
    )


@functools.cache
def _rope_module():
    args = cpp_args(supports_pdl())
    return load_jit(
        "dsv4_fused_rope", args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@functools.cache
def _topk_v2_module():
    return load_jit(
        "dsv4_topk_v2",
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")],
    )


def fused_q_norm_rope(q_in: torch.Tensor, q_out: torch.Tensor, eps: float, freqs: torch.Tensor, positions: torch.Tensor) -> None:
    """Per-head RMS norm (no weight) + interleaved RoPE on the last rope dims."""
    _q_norm_rope_module(q_in.shape[-1], freqs.shape[-1]).forward(q_in, q_out, freqs, positions, float(eps))


def fused_q_indexer_quant(
    q: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer q [T, H, 128] -> (FP8 q, head weights * weight_scale * q_scale [T, H, 1])."""
    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    w_out = torch.empty((*q.shape[:-1], 1), dtype=torch.float32, device=q.device)
    _q_indexer_module().forward(q, q_fp8, weight, w_out, float(weight_scale), freqs, positions)
    return q_fp8, w_out


def fused_rope_inplace(q: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor, inverse: bool = False) -> None:
    """RoPE on q [T, H, rope_dim] (a view of the last rope dims is fine) in place."""
    _rope_module().forward(q, None, freqs, positions, bool(inverse))


def plan_topk_v2(seq_lens: torch.Tensor) -> torch.Tensor:
    meta = seq_lens.new_empty(seq_lens.shape[0] + 1, 2)
    _topk_v2_module().topk_plan(seq_lens, meta, 0)
    return meta


def topk_transform_v2(
    scores: torch.Tensor, seq_lens: torch.Tensor, page_table: torch.Tensor, out: torch.Tensor, page_size: int,
    meta: torch.Tensor,
) -> None:
    _topk_v2_module().topk_transform(scores, seq_lens, page_table, out, int(page_size), meta, None)


@functools.cache
def _wo_a_quant_module():
    args = cpp_args(torch.bfloat16, supports_pdl())
    return load_jit(
        "dsv4_fp8_wo_a_group_major_quant_ue8m0", args,
        cuda_files=["deepseek_v4/fp8_wo_a_group_major_quant.cuh"],
        cuda_wrappers=[("run", f"FP8WoAGroupMajorQuantUE8M0Kernel<{args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def wo_a_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, G, D] bf16 -> (FP8 codes [T, G, D], packed UE8M0 scales in DeepGEMM's
    TMA-aligned layout) for ``fp8_einsum`` (sglang_per_token_group_quant_fp8_dsv4_wo_a)."""
    T, G, D = x.shape
    hg = D // 128
    phg = (hg + 3) // 4
    at = (T + 3) // 4 * 4
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((G, phg, at), device=x.device, dtype=torch.int32)
    if x.numel():
        _wo_a_quant_module().run(x, q, s)
    return q, s.transpose(-1, -2)[:, :T, :].transpose(0, 1)


def wo_a_einsum(o: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    """sglang FP8 wo_a: o [T, G, D] bf16, w [G, R, D] FP8, w_sf transformed -> [T, G, R] bf16."""
    from minisgl.kernel import deepgemm as dg

    T, G, D = o.shape
    R = w.shape[1]
    q, qs = wo_a_quant(o)
    out = torch.empty((T, G, R), device=o.device, dtype=torch.bfloat16)
    dg.fp8_einsum("bhr,hdr->bhd", (q, qs), (w, w_sf), out, None, (1, 1, 128))
    return out


def wo_a_weight_sf(scale: torch.Tensor, G: int, R: int, D: int) -> torch.Tensor:
    """Block scales [G*R/128, D/128] -> DeepGEMM layout for the grouped einsum."""
    from minisgl.kernel import deepgemm as dg

    raw = scale.float().view(G, R // 128, D // 128).contiguous()
    return dg.transform_sf_into_required_layout(raw, R, D, (1, 128, 128), G, False, False)


# ---------------------------------------------------------------------------
# DeepGEMM paged MQA logits
# ---------------------------------------------------------------------------


@functools.cache
def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def paged_mqa_schedule(c4_lens: torch.Tensor) -> torch.Tensor:
    from minisgl.kernel import deepgemm as dg

    return dg.get_paged_mqa_logits_metadata(c4_lens.view(-1, 1), INDEX_BLOCK, num_sms())


def paged_mqa_logits(
    q_fp8: torch.Tensor, cache: torch.Tensor, weights: torch.Tensor, c4_lens: torch.Tensor,
    block_table: torch.Tensor, schedule: torch.Tensor, max_len: int,
) -> torch.Tensor:
    """q_fp8 [T, H, 128], cache [pages, 64*132] uint8, weights [T, H] fp32 -> logits [T, max_len] fp32."""
    from minisgl.kernel import deepgemm as dg

    T, H, D = q_fp8.shape
    kv = cache.view(cache.shape[0], INDEX_BLOCK, 1, INDEX_ROW_BYTES)
    return dg.fp8_paged_mqa_logits(
        q_fp8.view(T, 1, H, D), kv, weights.view(T, H), c4_lens.view(T, 1), block_table, schedule, int(max_len), False
    )


# ---------------------------------------------------------------------------
# indexer-key store: norm + RoPE + Hadamard + per-token FP8 into the paged cache
# ---------------------------------------------------------------------------


@triton.jit
def _index_key_store_kernel(
    kv_ptr,  # [T, D] bf16 / fp32 pooled keys
    w_ptr,  # [D] norm weight
    pos_ptr,  # [T] int32 rope position
    cos_ptr,
    sin_ptr,
    had_ptr,  # [D, D] fp32 Hadamard (normalised)
    val_ptr,  # fp8 view of the cache bytes
    sc_ptr,  # fp32 view of the cache bytes
    row_ptr,  # [T] int64 destination c4 row (negative -> skip)
    stride_kv,
    eps,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    BLK: tl.constexpr,
):
    t = tl.program_id(0)
    row = tl.load(row_ptr + t).to(tl.int64)
    if row < 0:
        return
    NOPE: tl.constexpr = D - ROPE
    HALF: tl.constexpr = ROPE // 2
    offs = tl.arange(0, D)
    x = tl.load(kv_ptr + t * stride_kv + offs).to(tl.bfloat16).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    w = tl.load(w_ptr + offs).to(tl.float32)
    x = (w * (x * tl.rsqrt(ms + eps))).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(pos_ptr + t)
    x2 = tl.reshape(x, (D // 2, 2))
    x0, x1 = tl.split(x2)
    pair = tl.arange(0, D // 2)
    is_rope = pair >= (NOPE // 2)
    ridx = tl.where(is_rope, pair - NOPE // 2, 0)
    cc = tl.load(cos_ptr + pos * HALF + ridx, mask=is_rope, other=1.0)
    ss = tl.load(sin_ptr + pos * HALF + ridx, mask=is_rope, other=0.0)
    y0 = x0 * cc - x1 * ss
    y1 = x0 * ss + x1 * cc
    y = tl.reshape(tl.join(y0, y1), (D,))
    y = y.to(tl.bfloat16).to(tl.float32)
    hm = tl.load(had_ptr + offs[:, None] * D + offs[None, :])
    y = tl.sum(y[:, None] * hm, axis=0)
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-4)
    scale = amax / 448.0
    q = (y / scale).to(tl.float8e4nv)
    page = row // BLK
    off = row - page * BLK
    base = page * (BLK * (D + 4))
    tl.store(val_ptr + base + off * D + offs, q)
    tl.store(sc_ptr + (base + BLK * D) // 4 + off, scale)


def store_index_keys(
    kv: torch.Tensor, norm_weight: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    hadamard: torch.Tensor, cache: torch.Tensor, rows: torch.Tensor, *, eps: float, rope_dim: int,
) -> None:
    """kv [T, 128] pooled indexer keys -> FP8 + scale rows of the paged cache (rows < 0 skipped)."""
    T, D = kv.shape
    if T == 0:
        return
    flat = cache.view(-1)
    _index_key_store_kernel[(T,)](
        kv, norm_weight, positions, cos, sin, hadamard, flat.view(torch.float8_e4m3fn), flat.view(torch.float32),
        rows.to(torch.int64), kv.stride(0), float(eps), D=D, ROPE=rope_dim, BLK=INDEX_BLOCK, num_warps=4,
    )


def gather_index_keys(cache: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Dequantised keys [n, 128] bf16 of the given c4 rows (prefill path)."""
    pages = rows // INDEX_BLOCK
    off = rows - pages * INDEX_BLOCK
    c = cache.view(cache.shape[0], -1)
    vals = c[:, : INDEX_BLOCK * INDEX_DIM].view(cache.shape[0], INDEX_BLOCK, INDEX_DIM)
    scales = c[:, INDEX_BLOCK * INDEX_DIM :].contiguous().view(torch.float32).view(cache.shape[0], INDEX_BLOCK)
    v = vals[pages, off].view(torch.float8_e4m3fn).float()
    return (v * scales[pages, off].unsqueeze(-1)).to(torch.bfloat16)


__all__ = [
    "freqs_real", "fused_q_norm_rope", "fused_q_indexer_quant", "fused_rope_inplace", "plan_topk_v2",
    "topk_transform_v2", "paged_mqa_schedule", "paged_mqa_logits", "store_index_keys", "gather_index_keys",
    "INDEX_BLOCK", "INDEX_ROW_BYTES",
]
