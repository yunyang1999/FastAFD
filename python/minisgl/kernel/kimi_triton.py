"""Triton kernels for Kimi-K3 attention residuals and SiTU activation.

attention residual (``_apply_attn_res`` in modeling_kimi_linear.py):
    v      = [block_0, ..., block_{n-1}, prefix]          (fp32)
    k_j    = v_j * rsqrt(mean(v_j^2) + eps)
    s_j    = sum(k_j * norm_w * proj_w)
    out    = sum_j softmax(s)_j * v_j
SiTU-and-mul (``SituAndMul``): beta*tanh(g/beta)*sigmoid(g) * (lb*tanh(u/lb)).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_res_kernel(
    v_ptr,  # [T, NB, D] (slot NB-1 = prefix)
    w_ptr,  # [D] fp32 combined weight (norm_w * proj_w)
    out_ptr,  # [T, D]
    stride_t,
    stride_b,
    NB,
    D,
    eps,
    NBP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.arange(0, NBP)
    mj = j < NB
    base = v_ptr + t * stride_t
    sq = tl.zeros([NBP], dtype=tl.float32)
    dot = tl.zeros([NBP], dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        md = offs < D
        v = tl.load(base + j[:, None] * stride_b + offs[None, :], mask=mj[:, None] & md[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=md, other=0.0)
        sq += tl.sum(v * v, axis=1)
        dot += tl.sum(v * w[None, :], axis=1)
    s = dot * tl.rsqrt(sq / D + eps)
    s = tl.where(mj, s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=0))
    p = p / tl.sum(p, axis=0)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        md = offs < D
        v = tl.load(base + j[:, None] * stride_b + offs[None, :], mask=mj[:, None] & md[None, :], other=0.0).to(tl.float32)
        o = tl.sum(p[:, None] * v, axis=0)
        tl.store(out_ptr + t * D + offs, o.to(out_ptr.dtype.element_ty), mask=md)


def attn_res(v: torch.Tensor, nb: int, combined_w: torch.Tensor, eps: float) -> torch.Tensor:
    """v: [T, >=nb, D] buffer whose first ``nb`` slots are used."""
    T, _, D = v.shape
    out = torch.empty((T, D), dtype=v.dtype, device=v.device)
    if T == 0:
        return out
    _attn_res_kernel[(T,)](
        v, combined_w, out, v.stride(0), v.stride(1), nb, D, float(eps),
        NBP=triton.next_power_of_2(max(nb, 2)), BLOCK=512, num_warps=4,
    )
    return out


@triton.jit
def _tanh(x):
    e = tl.exp(-2.0 * tl.abs(x))
    r = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -r, r)


@triton.jit
def _situ_mul_kernel(x_ptr, y_ptr, stride_x, stride_y, I, beta, lbeta, HAS_LB: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < I
    g = tl.load(x_ptr + t * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + t * stride_x + I + offs, mask=m, other=0.0).to(tl.float32)
    a = beta * _tanh(g / beta) * tl.sigmoid(g)
    if HAS_LB:
        u = lbeta * _tanh(u / lbeta)
    tl.store(y_ptr + t * stride_y + offs, (a * u).to(y_ptr.dtype.element_ty), mask=m)


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float | None, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [T, 2I] (gate | up; rows may be strided, bf16 or fp32) -> [T, I]."""
    T, two_i = x.shape
    I = two_i // 2
    assert x.stride(1) == 1
    y = torch.empty((T, I), dtype=out_dtype or x.dtype, device=x.device)
    if T == 0:
        return y
    BLOCK = 1024
    _situ_mul_kernel[(T, triton.cdiv(I, BLOCK))](
        x, y, x.stride(0), y.stride(0), I, float(beta), float(linear_beta or 1.0),
        HAS_LB=linear_beta is not None, BLOCK=BLOCK, num_warps=4,
    )
    return y


__all__ = ["attn_res", "situ_and_mul"]
