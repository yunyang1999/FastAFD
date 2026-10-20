"""Triton kernels for DeepSeek-V4 attention-side ops.

Semantics follow the official DeepSeek-V4 reference (`inference/model.py` and
`inference/kernel.py`):

* mHC: ``hc_split_sinkhorn`` (pre = sigmoid + eps, post = 2*sigmoid, comb =
  row-softmax + eps then alternating column/row normalisation), the pre-weighted
  reduction of the ``hc_mult`` residual copies, ``hc_post`` and ``hc_head``.
* RoPE on the last ``rope_dim`` features with interleaved (complex-pair) layout.
* FP8 (e4m3, power-of-two block scale) and FP4 (e2m1, power-of-two block scale)
  quantize-dequantize simulation used by the reference for cached KV / indexer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------


@triton.jit
def _hc_pre_kernel(
    mixes_ptr,  # [T, MIX] fp32 (already rsqrt-normalised)
    x_ptr,  # [T, HC, D] bf16
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX] fp32
    y_ptr,  # [T, D] bf16
    post_ptr,  # [T, HC] fp32
    comb_ptr,  # [T, HC, HC] fp32
    D,
    eps,
    ITERS: tl.constexpr,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    pid_d = tl.program_id(1)
    hc = tl.arange(0, HC)
    s0 = tl.load(scale_ptr + 0)
    m_pre = tl.load(mixes_ptr + t * MIX + hc)
    b_pre = tl.load(base_ptr + hc)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + eps  # [HC]

    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    xp = x_ptr + t * HC * D + hc[:, None] * D + offs[None, :]
    xv = tl.load(xp, mask=mask[None, :], other=0.0).to(tl.float32)  # [HC, BLOCK_D]
    y = tl.sum(pre[:, None] * xv, axis=0)
    tl.store(y_ptr + t * D + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    if pid_d == 0:
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        m_post = tl.load(mixes_ptr + t * MIX + HC + hc)
        b_post = tl.load(base_ptr + HC + hc)
        post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
        tl.store(post_ptr + t * HC + hc, post)
        jj = tl.arange(0, HC)[:, None]
        kk = tl.arange(0, HC)[None, :]
        cidx = 2 * HC + jj * HC + kk
        c = tl.load(mixes_ptr + t * MIX + cidx) * s2 + tl.load(base_ptr + cidx)
        # row softmax + eps
        rmax = tl.max(c, axis=1)
        c = tl.exp(c - rmax[:, None])
        rsum = tl.sum(c, axis=1)
        c = c / rsum[:, None] + eps
        # column normalisation
        csum = tl.sum(c, axis=0)
        c = c / (csum[None, :] + eps)
        for _ in range(ITERS - 1):
            rsum = tl.sum(c, axis=1)
            c = c / (rsum[:, None] + eps)
            csum = tl.sum(c, axis=0)
            c = c / (csum[None, :] + eps)
        tl.store(comb_ptr + t * HC * HC + jj * HC + kk, c)


def hc_pre_fused(
    x: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: [T, HC, D] bf16, mixes: [T, (2+HC)*HC] fp32 -> (y [T,D], post [T,HC], comb [T,HC,HC])."""
    T, HC, D = x.shape
    MIX = (2 + HC) * HC
    assert mixes.shape == (T, MIX) and mixes.dtype == torch.float32
    x = x.contiguous()
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    post = torch.empty((T, HC), dtype=torch.float32, device=x.device)
    comb = torch.empty((T, HC, HC), dtype=torch.float32, device=x.device)
    if T == 0:
        return y, post, comb
    BLOCK_D = 1024
    grid = (T, triton.cdiv(D, BLOCK_D))
    _hc_pre_kernel[grid](
        mixes.contiguous(), x, hc_scale.contiguous(), hc_base.contiguous(), y, post, comb,
        D, float(eps), ITERS=int(sinkhorn_iters), HC=HC, MIX=MIX, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return y, post, comb


@triton.jit
def _hc_post_kernel(
    o_ptr,  # [T, D]
    res_ptr,  # [T, HC, D]
    post_ptr,  # [T, HC] fp32
    comb_ptr,  # [T, HC, HC] fp32
    out_ptr,  # [T, HC, D]
    D,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    hc = tl.arange(0, HC)
    o = tl.load(o_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)  # [BD]
    res = tl.load(
        res_ptr + t * HC * D + hc[:, None] * D + offs[None, :], mask=mask[None, :], other=0.0
    ).to(tl.float32)  # [HC(i), BD]
    post = tl.load(post_ptr + t * HC + hc)  # [HC(j)]
    for j in tl.static_range(HC):
        cj = tl.load(comb_ptr + t * HC * HC + hc * HC + j)  # comb[i, j] for all i
        pj = tl.sum(tl.where(hc == j, post, 0.0))
        acc = pj * o + tl.sum(cj[:, None] * res, axis=0)
        tl.store(out_ptr + t * HC * D + j * D + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def hc_post_fused(
    o: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """new[j] = post[j] * o + sum_i comb[i, j] * residual[i]  (reference hc_post)."""
    T, HC, D = residual.shape
    out = torch.empty_like(residual, dtype=o.dtype)
    if T == 0:
        return out
    BLOCK_D = 1024
    grid = (T, triton.cdiv(D, BLOCK_D))
    _hc_post_kernel[grid](
        o.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous(), out,
        D, HC=HC, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


@triton.jit
def _hc_head_kernel(
    mixes_ptr,  # [T, HC] fp32 (rsqrt-normalised)
    x_ptr,  # [T, HC, D]
    scale_ptr,  # [1]
    base_ptr,  # [HC]
    y_ptr,  # [T, D]
    D,
    eps,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    pid_d = tl.program_id(1)
    hc = tl.arange(0, HC)
    s = tl.load(scale_ptr)
    pre = tl.sigmoid(tl.load(mixes_ptr + t * HC + hc) * s + tl.load(base_ptr + hc)) + eps
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    xv = tl.load(
        x_ptr + t * HC * D + hc[:, None] * D + offs[None, :], mask=mask[None, :], other=0.0
    ).to(tl.float32)
    y = tl.sum(pre[:, None] * xv, axis=0)
    tl.store(y_ptr + t * D + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def hc_head_fused(
    x: torch.Tensor, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, eps: float
) -> torch.Tensor:
    T, HC, D = x.shape
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    if T == 0:
        return y
    BLOCK_D = 1024
    grid = (T, triton.cdiv(D, BLOCK_D))
    _hc_head_kernel[grid](
        mixes.contiguous(), x.contiguous(), hc_scale.contiguous(), hc_base.contiguous(), y,
        D, float(eps), HC=HC, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return y


def hc_prenorm_mixes_ref(x: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """mixes = (x_flat @ hc_fn^T) * rsqrt(mean(x_flat^2) + eps), all fp32 (reference)."""
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    return torch.mm(x_flat, hc_fn.t()) * rsqrt


@triton.jit
def _hc_mix_reduce_kernel(
    d_ptr,  # [S, T, N] fp32 split-K partial products
    sq_ptr,  # [S, T] fp32 split-K partial sums of squares
    out_ptr,  # [T, N] fp32
    T,
    S,
    inv_k,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    t = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    mn = n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    sqv = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        sv = s0 + tl.arange(0, BLOCK_S)
        ms = sv < S
        tile = tl.load(d_ptr + (sv[:, None] * T + t) * N + n[None, :], mask=ms[:, None] & mn[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
        sqv += tl.load(sq_ptr + sv * T + t, mask=ms, other=0.0)
    r = 1.0 / tl.sqrt(tl.sum(sqv, axis=0) * inv_k + eps)
    tl.store(out_ptr + t * N + n, acc * r, mask=mn)


def _hc_prenorm_splits(num_tokens: int, k: int) -> int:
    """Split-K factor of the DeepGEMM prenorm GEMM (sglang mhc._compute_num_split_for_mhc_pre)."""
    block_m, block_k = 64, 64
    grid = (num_tokens + block_m - 1) // block_m
    n_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(1, min(n_sms // max(grid, 1), ((k + block_k - 1) // block_k) // 4))


def hc_prenorm_mixes(x: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """mixes = (x_flat @ hc_fn^T) * rsqrt(mean(x_flat^2) + eps) in fp32.

    Production path (as sglang on SM100): DeepGEMM ``tf32_hc_prenorm_gemm`` (TF32
    GEMM + row sum of squares, split-K) and a split reduction; falls back to the
    fp32 reference when DeepGEMM is unavailable or MINISGL_DSV4_HC_GEMM=torch.
    """
    T = x.shape[0]
    if T == 0 or _HC_GEMM_BACKEND == "torch" or x.dtype != torch.bfloat16 or hc_fn.shape[0] < 16:
        return hc_prenorm_mixes_ref(x, hc_fn, eps)
    from minisgl.kernel import deepgemm as dg

    x_flat = x.reshape(T, -1)
    K = x_flat.shape[1]
    N = hc_fn.shape[0]
    S = _hc_prenorm_splits(T, K)
    d = torch.empty((S, T, N), dtype=torch.float32, device=x.device)
    sq = torch.empty((S, T), dtype=torch.float32, device=x.device)
    dg.tf32_hc_prenorm_gemm(x_flat, hc_fn, d, sq, S)
    out = torch.empty((T, N), dtype=torch.float32, device=x.device)
    _hc_mix_reduce_kernel[(T,)](
        d, sq, out, T, S, 1.0 / K, float(eps), N=N, BLOCK_N=triton.next_power_of_2(N),
        BLOCK_S=min(64, triton.next_power_of_2(S)), num_warps=2,
    )
    return out


import os as _os

_HC_GEMM_BACKEND = _os.environ.get("MINISGL_DSV4_HC_GEMM", "deepgemm").lower()


@triton.jit
def _hc_pre_split_kernel(
    d_ptr,  # [S, T, MIX] fp32 split-K partial products (x_flat @ hc_fn^T)
    sq_ptr,  # [S, T] fp32 split-K partial sums of squares of x_flat
    x_ptr,  # [T, HC, D] bf16
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX] fp32
    nw_ptr,  # [D] norm weight (HAS_NORM)
    y_ptr,  # [T, D] bf16
    post_ptr,  # [T, HC] fp32
    comb_ptr,  # [T, HC, HC] fp32
    T,
    S,
    D,
    inv_k,
    rms_eps,
    eps,
    norm_eps,
    ITERS: tl.constexpr,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    """mHC pre (one program per token): split-K reduction of the prenorm GEMM,
    rsqrt normalisation, Sinkhorn mixing weights, pre-weighted sum of the HC
    residual copies and (optionally) the following RMSNorm -- the fused shape of
    sglang's mhc_pre_big_fuse_with_norm."""
    t = tl.program_id(0)
    hc = tl.arange(0, HC)
    cc = tl.arange(0, HC * HC)
    acc_pre = tl.zeros([HC], dtype=tl.float32)
    acc_post = tl.zeros([HC], dtype=tl.float32)
    acc_comb = tl.zeros([HC * HC], dtype=tl.float32)
    sqv = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        sv = s0 + tl.arange(0, BLOCK_S)
        ms = sv < S
        base = d_ptr + (sv[:, None] * T + t) * MIX
        acc_pre += tl.sum(tl.load(base + hc[None, :], mask=ms[:, None], other=0.0), axis=0)
        acc_post += tl.sum(tl.load(base + HC + hc[None, :], mask=ms[:, None], other=0.0), axis=0)
        acc_comb += tl.sum(tl.load(base + 2 * HC + cc[None, :], mask=ms[:, None], other=0.0), axis=0)
        sqv += tl.load(sq_ptr + sv * T + t, mask=ms, other=0.0)
    r = 1.0 / tl.sqrt(tl.sum(sqv, axis=0) * inv_k + rms_eps)
    s0_ = tl.load(scale_ptr + 0)
    s1_ = tl.load(scale_ptr + 1)
    s2_ = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(acc_pre * r * s0_ + tl.load(base_ptr + hc)) + eps
    post = 2.0 * tl.sigmoid(acc_post * r * s1_ + tl.load(base_ptr + HC + hc))
    tl.store(post_ptr + t * HC + hc, post)
    c = tl.reshape(acc_comb * r * s2_ + tl.load(base_ptr + 2 * HC + cc), (HC, HC))
    rmax = tl.max(c, axis=1)
    c = tl.exp(c - rmax[:, None])
    rsum = tl.sum(c, axis=1)
    c = c / rsum[:, None] + eps
    csum = tl.sum(c, axis=0)
    c = c / (csum[None, :] + eps)
    for _ in range(ITERS - 1):
        rsum = tl.sum(c, axis=1)
        c = c / (rsum[:, None] + eps)
        csum = tl.sum(c, axis=0)
        c = c / (csum[None, :] + eps)
    jj = tl.arange(0, HC)[:, None]
    kk = tl.arange(0, HC)[None, :]
    tl.store(comb_ptr + t * HC * HC + jj * HC + kk, c)
    # pre-weighted sum of the residual copies (+ RMSNorm over the full row)
    ss = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        xv = tl.load(x_ptr + t * HC * D + hc[:, None] * D + offs[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
        y = tl.sum(pre[:, None] * xv, axis=0)
        yb = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + t * D + offs, yb, mask=mask)
        yf = yb.to(tl.float32)
        ss += tl.sum(tl.where(mask, yf * yf, 0.0), axis=0)
    if HAS_NORM:
        rn = 1.0 / tl.sqrt(ss / D + norm_eps)
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            yv = tl.load(y_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(nw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_ptr + t * D + offs, (yv * rn * w).to(y_ptr.dtype.element_ty), mask=mask)


def hc_pre_split(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    sinkhorn_iters: int,
    eps: float,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre on DeepGEMM's TF32 split-K prenorm GEMM + one fused Triton kernel
    (split reduction, Sinkhorn, weighted sum, optional RMSNorm).
    x [T, HC, D] bf16 -> (y [T, D], post [T, HC], comb [T, HC, HC])."""
    from minisgl.kernel import deepgemm as dg

    T, HC, D = x.shape
    MIX = (2 + HC) * HC
    K = HC * D
    S = _hc_prenorm_splits(T, K)
    x = x.contiguous()
    d = torch.empty((S, T, MIX), dtype=torch.float32, device=x.device)
    sq = torch.empty((S, T), dtype=torch.float32, device=x.device)
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    post = torch.empty((T, HC), dtype=torch.float32, device=x.device)
    comb = torch.empty((T, HC, HC), dtype=torch.float32, device=x.device)
    if T == 0:
        return y, post, comb
    dg.tf32_hc_prenorm_gemm(x.view(T, K), hc_fn, d, sq, S)
    _hc_pre_split_kernel[(T,)](
        d, sq, x, hc_scale.contiguous(), hc_base.contiguous(),
        norm_weight if norm_weight is not None else hc_scale, y, post, comb,
        T, S, D, 1.0 / K, float(rms_eps), float(eps), float(norm_eps),
        ITERS=int(sinkhorn_iters), HC=HC, MIX=MIX, BLOCK_S=min(64, triton.next_power_of_2(S)), BLOCK_D=1024,
        HAS_NORM=norm_weight is not None, num_warps=4,
    )
    return y, post, comb


def hc_pre_split_supported(x: torch.Tensor, hc_fn: torch.Tensor) -> bool:
    return _HC_GEMM_BACKEND != "torch" and x.dtype == torch.bfloat16 and hc_fn.shape[0] >= 16 and x.shape[0] > 0


# ---------------------------------------------------------------------------
# quant-dequant simulation helpers (device functions)
# ---------------------------------------------------------------------------


@triton.jit
def _pow2_ceil_scale(amax, inv_max):
    # 2 ** ceil(log2(amax * inv_max)) via exponent bit manipulation.
    v = amax * inv_max
    bits = v.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32)
    man = bits & 0x7FFFFF
    e = e + tl.where(man != 0, 1, 0)
    e = tl.minimum(tl.maximum(e, 1), 254)
    return (e.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fp8_e4m3_roundtrip(x):
    return x.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _fp4_e2m1_round(x):
    # round-to-nearest-even on {0, .5, 1, 1.5, 2, 3, 4, 6}, |x| <= 6
    a = tl.abs(x)
    r = tl.where(
        a <= 0.25, 0.0,
        tl.where(a < 0.75, 0.5,
        tl.where(a <= 1.25, 1.0,
        tl.where(a < 1.75, 1.5,
        tl.where(a <= 2.5, 2.0,
        tl.where(a < 3.5, 3.0,
        tl.where(a <= 5.0, 4.0, 6.0)))))))
    return tl.where(x < 0, -r, r)


# ---------------------------------------------------------------------------
# q: per-head RMS (no weight) + interleaved RoPE on the last ROPE dims, in place
# ---------------------------------------------------------------------------


@triton.jit
def _q_norm_rope_kernel(
    q_ptr,  # [T, H, D] bf16 (in/out)
    pos_ptr,  # [T] int32
    cos_ptr,  # [P, ROPE/2] fp32
    sin_ptr,
    stride_t,
    stride_h,
    eps,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    DO_NORM: tl.constexpr,
    DO_HADAMARD_FP4: tl.constexpr,
    had_ptr,  # [D, D] fp32 hadamard (only if DO_HADAMARD_FP4)
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = q_ptr + t * stride_t + h * stride_h
    NOPE: tl.constexpr = D - ROPE
    HALF: tl.constexpr = ROPE // 2
    pos = tl.load(pos_ptr + t)
    offs = tl.arange(0, D)
    x = tl.load(base + offs).to(tl.float32)
    if DO_NORM:
        # reference: q is bf16; rms computed on it, applied, then rounded to bf16
        ms = tl.sum(x * x, axis=0) / D
        x = (x * tl.rsqrt(ms + eps)).to(tl.bfloat16).to(tl.float32)
    # interleaved rope on [NOPE, D): pairs (NOPE+2i, NOPE+2i+1)
    x2 = tl.reshape(x, (D // 2, 2))
    x0, x1 = tl.split(x2)  # [D/2] even / odd
    pair = tl.arange(0, D // 2)
    is_rope = pair >= (NOPE // 2)
    ridx = tl.where(is_rope, pair - NOPE // 2, 0)
    cc = tl.load(cos_ptr + pos * HALF + ridx, mask=is_rope, other=1.0)
    ss = tl.load(sin_ptr + pos * HALF + ridx, mask=is_rope, other=0.0)
    y0 = x0 * cc - x1 * ss
    y1 = x0 * ss + x1 * cc
    y = tl.reshape(tl.join(y0, y1), (D,))
    if DO_HADAMARD_FP4:
        y = y.to(tl.bfloat16).to(tl.float32)
        hm = tl.load(had_ptr + offs[:, None] * D + offs[None, :])  # [D, D]
        y = tl.sum(y[:, None] * hm, axis=0)
        y = y.to(tl.bfloat16).to(tl.float32)
        yb = tl.reshape(y, (D // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(yb), axis=1), 6.0 * 1.1754943508222875e-38)
        sc = _pow2_ceil_scale(amax, 1.0 / 6.0)
        qv = _fp4_e2m1_round(tl.minimum(tl.maximum(yb / sc[:, None], -6.0), 6.0))
        y = tl.reshape(qv * sc[:, None], (D,))
    tl.store(base + offs, y.to(q_ptr.dtype.element_ty))


def q_norm_rope_(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float,
    rope_dim: int,
    do_norm: bool = True,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place on q [T, H, D] (D = 512 attention or 128 indexer)."""
    T, H, D = q.shape
    if T == 0:
        return q
    assert q.stride(-1) == 1
    _q_norm_rope_kernel[(T, H)](
        q, positions, cos, sin, q.stride(0), q.stride(1), float(eps),
        D=D, ROPE=rope_dim, DO_NORM=bool(do_norm),
        DO_HADAMARD_FP4=hadamard is not None,
        had_ptr=hadamard if hadamard is not None else cos,
        num_warps=4 if D <= 128 else 8,
    )
    return q


# ---------------------------------------------------------------------------
# kv: weighted RMSNorm + RoPE (+ fp8 sim of the NoPE part) -> rows of a cache
# ---------------------------------------------------------------------------


@triton.jit
def _kv_norm_rope_store_kernel(
    kv_ptr,  # [T, D] (bf16 or fp32)
    w_ptr,  # [D] fp32 norm weight
    pos_ptr,  # [T] int32
    cos_ptr,
    sin_ptr,
    out_ptr,  # [N, ROW] bf16 cache
    slot_ptr,  # [T] int64 destination row (negative -> skip), offset by row_base
    stride_kv,
    ROW,
    row_base,
    eps,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    QUANT: tl.constexpr,  # 0 none, 1 fp8 sim on NoPE (block 64), 2 hadamard + fp4 sim (block 32)
    had_ptr,
):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    slot = slot + row_base
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
    if QUANT == 1:
        yb = tl.reshape(y, (D // 64, 64))
        amax = tl.maximum(tl.max(tl.abs(yb), axis=1), 1e-4)
        sc = _pow2_ceil_scale(amax, 1.0 / 448.0)
        qv = _fp8_e4m3_roundtrip(tl.minimum(tl.maximum(yb / sc[:, None], -448.0), 448.0))
        yq = tl.reshape(qv * sc[:, None], (D,))
        y = tl.where(offs < NOPE, yq, y)
    elif QUANT == 2:
        hm = tl.load(had_ptr + offs[:, None] * D + offs[None, :])
        y = tl.sum(y[:, None] * hm, axis=0)
        y = y.to(tl.bfloat16).to(tl.float32)
        yb = tl.reshape(y, (D // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(yb), axis=1), 6.0 * 1.1754943508222875e-38)
        sc = _pow2_ceil_scale(amax, 1.0 / 6.0)
        qv = _fp4_e2m1_round(tl.minimum(tl.maximum(yb / sc[:, None], -6.0), 6.0))
        y = tl.reshape(qv * sc[:, None], (D,))
    tl.store(out_ptr + slot * ROW + offs, y.to(out_ptr.dtype.element_ty))


def kv_norm_rope_store(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    slots: torch.Tensor,
    *,
    eps: float,
    rope_dim: int,
    quant: int,
    hadamard: torch.Tensor | None = None,
    row_base: int = 0,
) -> None:
    """Normalise+RoPE(+quant-sim) rows of ``kv`` [T, D] into ``out[row_base + slots, :D]``
    (negative slots skipped)."""
    T, D = kv.shape
    if T == 0:
        return
    assert kv.stride(-1) == 1 and out.stride(-1) == 1
    if slots.dtype != torch.int64:
        slots = slots.to(torch.int64)
    _kv_norm_rope_store_kernel[(T,)](
        kv, norm_weight, positions, cos, sin, out, slots, kv.stride(0),
        out.stride(0), int(row_base), float(eps), D=D, ROPE=rope_dim, QUANT=int(quant),
        had_ptr=hadamard if hadamard is not None else cos,
        num_warps=4 if D <= 128 else 8,
    )


# ---------------------------------------------------------------------------
# inverse RoPE on the last ROPE dims of the attention output, in place
# ---------------------------------------------------------------------------


@triton.jit
def _inv_rope_kernel(
    o_ptr,  # [T, H, D]
    pos_ptr,
    cos_ptr,
    sin_ptr,
    stride_t,
    stride_h,
    D: tl.constexpr,
    ROPE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = ROPE // 2
    base = o_ptr + t * stride_t + h * stride_h + (D - ROPE)
    pos = tl.load(pos_ptr + t)
    i = tl.arange(0, HALF)
    x0 = tl.load(base + 2 * i).to(tl.float32)
    x1 = tl.load(base + 2 * i + 1).to(tl.float32)
    c = tl.load(cos_ptr + pos * HALF + i)
    s = tl.load(sin_ptr + pos * HALF + i)
    tl.store(base + 2 * i, (x0 * c + x1 * s).to(o_ptr.dtype.element_ty))
    tl.store(base + 2 * i + 1, (x1 * c - x0 * s).to(o_ptr.dtype.element_ty))


def inverse_rope_(o: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    T, H, D = o.shape
    if T == 0:
        return o
    _inv_rope_kernel[(T, H)](o, positions, cos, sin, o.stride(0), o.stride(1), D=D, ROPE=rope_dim, num_warps=1)
    return o


# ---------------------------------------------------------------------------
# compressor decode step: state update + (on group completion) softmax pooling
# ---------------------------------------------------------------------------


@triton.jit
def _compress_decode_kernel(
    kv_ptr,  # [B, CD] fp32 projections (CD = coff * d)
    score_ptr,  # [B, CD] fp32 (without ape)
    ape_ptr,  # [RATIO, CD] fp32
    pos_ptr,  # [B] int32
    tidx_ptr,  # [B] int32 state row
    skv_ptr,  # [R, ROWS, CD] fp32 state
    ssc_ptr,  # [R, ROWS, CD] fp32 state
    pooled_ptr,  # [B, d] fp32 output (valid where complete)
    stride_in,  # row stride of kv / score (may be a slice of a wider GEMM output)
    CD,
    d,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROWS: tl.constexpr,  # coff * ratio
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pid = tl.program_id(1)
    pos = tl.load(pos_ptr + b)
    r = tl.load(tidx_ptr + b).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < d
    rsel = pos % RATIO
    cur_row = rsel + (RATIO if OVERLAP else 0)
    st_kv = skv_ptr + r * ROWS * CD
    st_sc = ssc_ptr + r * ROWS * CD
    # write current token's projections (both halves when overlapping)
    for half in tl.static_range(2 if OVERLAP else 1):
        col = offs + half * d
        kvv = tl.load(kv_ptr + b * stride_in + col, mask=mask, other=0.0)
        scv = tl.load(score_ptr + b * stride_in + col, mask=mask, other=0.0) + tl.load(
            ape_ptr + rsel * CD + col, mask=mask, other=0.0
        )
        tl.store(st_kv + cur_row * CD + col, kvv, mask=mask)
        tl.store(st_sc + cur_row * CD + col, scv, mask=mask)
    complete = ((pos + 1) % RATIO) == 0
    if complete:
        tl.debug_barrier()
        rows = tl.arange(0, ROWS)
        if OVERLAP:
            # rows [0, ratio) use the first half of the previous group's rows,
            # rows [ratio, 2*ratio) the second half of the current group's rows
            colsel = tl.where(rows[:, None] < RATIO, offs[None, :], offs[None, :] + d)
        else:
            colsel = offs[None, :] + rows[:, None] * 0
        m2 = mask[None, :] & (rows[:, None] >= 0)
        sc = tl.load(st_sc + rows[:, None] * CD + colsel, mask=m2, other=float("-inf"))
        kv = tl.load(st_kv + rows[:, None] * CD + colsel, mask=m2, other=0.0)
        mx = tl.max(sc, axis=0)
        p = tl.exp(sc - mx[None, :])
        den = tl.sum(p, axis=0)
        pooled = tl.sum(p * kv, axis=0) / den
        tl.store(pooled_ptr + b * d + offs, pooled, mask=mask)
        tl.debug_barrier()
        if OVERLAP:
            # shift: previous-group rows <- current-group rows (both halves)
            for half in tl.static_range(2):
                col = offs + half * d
                for rr in tl.static_range(RATIO):
                    v1 = tl.load(st_kv + (RATIO + rr) * CD + col, mask=mask, other=0.0)
                    v2 = tl.load(st_sc + (RATIO + rr) * CD + col, mask=mask, other=0.0)
                    tl.store(st_kv + rr * CD + col, v1, mask=mask)
                    tl.store(st_sc + rr * CD + col, v2, mask=mask)


def compressor_decode_step(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    table_idx: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    *,
    ratio: int,
    overlap: bool,
    head_dim: int,
) -> torch.Tensor:
    """Returns pooled (pre-norm) compressed kv [B, d]; rows are meaningful only
    for requests whose group completes at this position."""
    B, CD = kv.shape
    pooled = torch.empty((B, head_dim), dtype=torch.float32, device=kv.device)
    if B == 0:
        return pooled
    rows = (2 if overlap else 1) * ratio
    BLOCK = 128 if rows <= 16 else 32
    grid = (B, triton.cdiv(head_dim, BLOCK))
    if kv.stride(-1) != 1 or score.stride(-1) != 1 or kv.stride(0) != score.stride(0):
        kv, score = kv.contiguous(), score.contiguous()
    _compress_decode_kernel[grid](
        kv, score, ape.contiguous(), positions, table_idx,
        state_kv, state_score, pooled, kv.stride(0), CD, head_dim,
        RATIO=int(ratio), OVERLAP=bool(overlap), ROWS=rows, BLOCK=BLOCK, num_warps=4,
    )
    return pooled


__all__ = [
    "hc_pre_fused",
    "hc_post_fused",
    "hc_head_fused",
    "hc_prenorm_mixes",
    "q_norm_rope_",
    "kv_norm_rope_store",
    "inverse_rope_",
    "compressor_decode_step",
]
