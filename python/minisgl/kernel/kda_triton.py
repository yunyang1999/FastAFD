"""Triton kernels for Kimi Delta Attention (KDA).

The recurrence follows the gated delta rule used by Kimi-Linear / Kimi-K3
(`fla.ops.kda.fused_recurrent_kda` with ``use_qk_l2norm_in_kernel``,
``use_gate_in_kernel``, ``use_beta_sigmoid_in_kernel`` and the safe gate
``lower_bound``); the kernel structure is adapted from the MIT-licensed
flash-linear-attention recurrent kernels (Songlin Yang, Yu Zhang) as vendored by
vLLM/SGLang (Apache-2.0). State layout is ``[slots, H, V, K]`` (transposed, as in
the reference ``transpose_state_layout=True``):

    g     = lower_bound * sigmoid(exp(A_log[h]) * (g_raw + dt_bias))   (per k)
    beta  = sigmoid(b_raw)
    q, k  = l2norm(q), l2norm(k);  q *= scale
    S     = S * exp(g)[None, :]
    v'    = beta * (v - S @ k)
    S     = S + v'[:, None] * k[None, :]
    o     = S @ q
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# causal depthwise short convolution (+SiLU) with per-request state
# ---------------------------------------------------------------------------


@triton.jit
def _short_conv_varlen_kernel(
    x_ptr,  # [T, D]
    w_ptr,  # [D, KW]
    state_ptr,  # [R, D, KW-1]
    y_ptr,  # [T, D]
    cu_ptr,  # [N+1] int32
    sidx_ptr,  # [N] int32 state rows
    init_ptr,  # [N] int32 has_initial_state
    D,
    stride_x,
    stride_y,
    KW: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)
    bos = tl.load(cu_ptr + n)
    eos = tl.load(cu_ptr + n + 1)
    L = eos - bos
    t0 = pid_t * BLOCK_T
    if t0 >= L:
        return
    srow = tl.load(sidx_ptr + n).to(tl.int64)
    has_init = tl.load(init_ptr + n)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    md = offs_d < D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    mt = offs_t < L
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(KW):
        src_t = offs_t - (KW - 1) + k  # position within the sequence
        in_seq = src_t >= 0
        xv = tl.load(
            x_ptr + (bos + src_t)[:, None].to(tl.int64) * stride_x + offs_d[None, :],
            mask=(mt & in_seq)[:, None] & md[None, :],
            other=0.0,
        ).to(tl.float32)
        # positions before the sequence start come from the conv state
        sidx = (KW - 1) + src_t  # index into state for src_t in [-(KW-1), 0)
        use_state = (src_t < 0) & mt & (has_init != 0)
        sv = tl.load(
            state_ptr + srow * D * (KW - 1) + offs_d[None, :] * (KW - 1) + tl.where(use_state, sidx, 0)[:, None],
            mask=use_state[:, None] & md[None, :],
            other=0.0,
        ).to(tl.float32)
        wk = tl.load(w_ptr + offs_d * KW + k, mask=md, other=0.0).to(tl.float32)
        acc += (xv + sv) * wk[None, :]
    out = acc * tl.sigmoid(acc)
    tl.store(
        y_ptr + (bos + offs_t)[:, None].to(tl.int64) * stride_y + offs_d[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=mt[:, None] & md[None, :],
    )


@triton.jit
def _short_conv_state_update_kernel(
    x_ptr,
    state_ptr,
    cu_ptr,
    sidx_ptr,
    init_ptr,
    D,
    stride_x,
    KW: tl.constexpr,
    JP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    pid_d = tl.program_id(1)
    bos = tl.load(cu_ptr + n)
    eos = tl.load(cu_ptr + n + 1)
    L = eos - bos
    srow = tl.load(sidx_ptr + n).to(tl.int64)
    has_init = tl.load(init_ptr + n)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    md = offs_d < D
    j = tl.arange(0, JP)
    mj = j < (KW - 1)
    # new_state[j] = input at seq position L - (KW-1) + j
    src = L - (KW - 1) + j
    from_x = src >= 0
    from_x = from_x & mj
    xv = tl.load(
        x_ptr + (bos + tl.where(from_x, src, 0))[None, :].to(tl.int64) * stride_x + offs_d[:, None],
        mask=md[:, None] & from_x[None, :],
        other=0.0,
    )
    old_idx = (KW - 1) + src  # old state index for src < 0
    use_old = (~from_x) & (old_idx >= 0) & (has_init != 0) & mj
    sp = state_ptr + srow * D * (KW - 1) + offs_d[:, None] * (KW - 1)
    sv = tl.load(sp + tl.where(use_old, old_idx, 0)[None, :], mask=md[:, None] & use_old[None, :], other=0.0)
    newv = tl.where(from_x[None, :], xv.to(tl.float32), sv.to(tl.float32))
    tl.debug_barrier()
    tl.store(sp + j[None, :], newv.to(state_ptr.dtype.element_ty), mask=md[:, None] & mj[None, :])


def short_conv_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_idx: torch.Tensor,
    has_initial_state: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """x [T, D] -> silu(causal_conv(x)) [T, D]; updates conv_state[state_idx]."""
    T, D = x.shape
    KW = weight.shape[-1]
    y = torch.empty_like(x)
    N = int(state_idx.shape[0])
    if T == 0 or N == 0:
        return y
    BLOCK_T, BLOCK_D = 16, 256
    grid = (N, triton.cdiv(D, BLOCK_D), triton.cdiv(max(int(max_seqlen), 1), BLOCK_T))
    _short_conv_varlen_kernel[grid](
        x, weight, conv_state, y, cu_seqlens, state_idx, has_initial_state,
        D, x.stride(0), y.stride(0), KW=KW, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, num_warps=4,
    )
    _short_conv_state_update_kernel[(N, triton.cdiv(D, BLOCK_D))](
        x, conv_state, cu_seqlens, state_idx, has_initial_state, D, x.stride(0),
        KW=KW, JP=triton.next_power_of_2(KW - 1), BLOCK_D=BLOCK_D, num_warps=4,
    )
    return y


@triton.jit
def _short_conv_decode_kernel(
    x_ptr,  # [B, D]
    w_ptr,  # [D, KW]
    state_ptr,  # [R, D, KW-1]
    y_ptr,
    sidx_ptr,
    D,
    stride_x,
    stride_y,
    KW: tl.constexpr,
    JP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_d = tl.program_id(1)
    srow = tl.load(sidx_ptr + b).to(tl.int64)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    md = offs < D
    j = tl.arange(0, JP)
    mj = j < (KW - 1)
    m2 = md[:, None] & mj[None, :]
    sp = state_ptr + srow * D * (KW - 1) + offs[:, None] * (KW - 1) + j[None, :]
    st = tl.load(sp, mask=m2, other=0.0).to(tl.float32)  # [BD, JP]
    xv = tl.load(x_ptr + b * stride_x + offs, mask=md, other=0.0).to(tl.float32)
    w_hist = tl.load(w_ptr + offs[:, None] * KW + j[None, :], mask=m2, other=0.0).to(tl.float32)
    w_last = tl.load(w_ptr + offs * KW + (KW - 1), mask=md, other=0.0).to(tl.float32)
    acc = tl.sum(st * w_hist, axis=1) + xv * w_last
    tl.store(y_ptr + b * stride_y + offs, (acc * tl.sigmoid(acc)).to(y_ptr.dtype.element_ty), mask=md)
    # shift state left by one and append x
    # new[j] = old[j+1] for j < KW-2; new[KW-2] = x
    nxt = tl.load(
        state_ptr + srow * D * (KW - 1) + offs[:, None] * (KW - 1) + tl.minimum(j + 1, KW - 2)[None, :],
        mask=m2,
        other=0.0,
    ).to(tl.float32)
    newv = tl.where((j == KW - 2)[None, :], xv[:, None], nxt)
    tl.debug_barrier()
    tl.store(sp, newv.to(state_ptr.dtype.element_ty), mask=m2)


def short_conv_decode(
    x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor, state_idx: torch.Tensor
) -> torch.Tensor:
    B, D = x.shape
    KW = weight.shape[-1]
    y = torch.empty_like(x)
    if B == 0:
        return y
    BLOCK_D = 256
    _short_conv_decode_kernel[(B, triton.cdiv(D, BLOCK_D))](
        x, weight, conv_state, y, state_idx, D, x.stride(0), y.stride(0),
        KW=KW, JP=triton.next_power_of_2(KW - 1), BLOCK_D=BLOCK_D, num_warps=4,
    )
    return y


# ---------------------------------------------------------------------------
# gated delta rule recurrence
# ---------------------------------------------------------------------------


@triton.jit
def _kda_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,  # [T, H*K] / [T, H*V] (row strides given)
    g_ptr,  # [T, H*K] raw forget gate
    b_ptr,  # [T, H] raw beta
    alog_ptr,  # [H] fp32
    dtb_ptr,  # [H*K] fp32
    o_ptr,  # [T, H, V]
    state_ptr,  # [R, H, V, K] fp32
    cu_ptr,
    sidx_ptr,
    init_ptr,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    stride_b,
    scale,
    lower_bound,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    n = i_nh // H
    h = i_nh % H
    bos = tl.load(cu_ptr + n).to(tl.int64)
    eos = tl.load(cu_ptr + n + 1).to(tl.int64)
    srow = tl.load(sidx_ptr + n).to(tl.int64)
    has_init = tl.load(init_ptr + n)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    sp = state_ptr + ((srow * H + h) * V + o_v[:, None]) * K + o_k[None, :]
    if has_init != 0:
        S = tl.load(sp).to(tl.float32)
    else:
        S = tl.zeros([BV, K], dtype=tl.float32)
    a_val = tl.exp(tl.load(alog_ptr + h).to(tl.float32))
    dtb = tl.load(dtb_ptr + h * K + o_k).to(tl.float32)
    for t in range(bos, eos):
        bq = tl.load(q_ptr + t * stride_q + h * K + o_k).to(tl.float32)
        bk = tl.load(k_ptr + t * stride_k + h * K + o_k).to(tl.float32)
        bv = tl.load(v_ptr + t * stride_v + h * V + o_v).to(tl.float32)
        bq = bq / tl.sqrt(tl.sum(bq * bq) + 1e-6) * scale
        bk = bk / tl.sqrt(tl.sum(bk * bk) + 1e-6)
        gr = tl.load(g_ptr + t * stride_g + h * K + o_k).to(tl.float32)
        g = lower_bound * tl.sigmoid(a_val * (gr + dtb))
        beta = tl.sigmoid(tl.load(b_ptr + t * stride_b + h).to(tl.float32))
        S = S * tl.exp(g)[None, :]
        bv = (bv - tl.sum(S * bk[None, :], axis=1)) * beta
        S = S + bv[:, None] * bk[None, :]
        o = tl.sum(S * bq[None, :], axis=1)
        tl.store(o_ptr + (t * H + h) * V + o_v, o.to(o_ptr.dtype.element_ty))
    tl.store(sp, S.to(state_ptr.dtype.element_ty))


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    b_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_idx: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    lower_bound: float,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """q/k/v/g_raw: [T, H*D] views (last-dim contiguous); b_raw [T, H]. Returns o [T, H, D]."""
    T = q.shape[0]
    H, D = int(num_heads), int(head_dim)
    o = torch.empty((T, H, D), dtype=out_dtype, device=q.device)
    N = int(state_idx.shape[0])
    if T == 0 or N == 0:
        return o
    BV = 32
    grid = (triton.cdiv(D, BV), N * H)
    _kda_recurrent_kernel[grid](
        q, k, v, g_raw, b_raw, A_log, dt_bias, o, state, cu_seqlens, state_idx, has_initial_state,
        q.stride(0), k.stride(0), v.stride(0), g_raw.stride(0), b_raw.stride(0),
        float(scale), float(lower_bound), H=H, K=D, V=D, BV=BV, num_warps=1, num_stages=2,
    )
    return o


# ---------------------------------------------------------------------------
# gated RMSNorm: rmsnorm(x) * w * sigmoid(g)  (fla FusedRMSNormGated, sigmoid)
# ---------------------------------------------------------------------------


@triton.jit
def _gated_rmsnorm_kernel(x_ptr, g_ptr, w_ptr, y_ptr, stride_x, stride_g, stride_y, eps, D: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + r * stride_x + offs).to(tl.float32)
    g = tl.load(g_ptr + r * stride_g + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x) / D + eps)
    y = x * rstd * w * tl.sigmoid(g)
    tl.store(y_ptr + r * stride_y + offs, y.to(y_ptr.dtype.element_ty))


def gated_rmsnorm(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x, g: [M, D] (D = head_dim) -> [M, D]."""
    M, D = x.shape
    y = torch.empty((M, D), dtype=g.dtype if g.dtype != torch.float32 else torch.bfloat16, device=x.device)
    if M == 0:
        return y
    _gated_rmsnorm_kernel[(M,)](x, g, weight, y, x.stride(0), g.stride(0), y.stride(0), float(eps), D=D, num_warps=1)
    return y


__all__ = [
    "short_conv_varlen",
    "short_conv_decode",
    "kda_recurrent",
    "gated_rmsnorm",
]
