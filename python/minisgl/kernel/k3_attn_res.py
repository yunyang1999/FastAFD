"""Kimi-K3 attention-residual aggregation: sglang's SM100 TMA kernel.

Vendored from sglang 0.5.18 (sglang/kernels/ops/kimi_k3/attn_res.py, Apache-2.0):
the warp-specialized TMA kernel ``sgl_jit/csrc/kimi_k3/attn_res/fused_tma.cuh``
(score -> online softmax -> weighted combine -> fused output RMSNorm) with the
benchmarked per-``nvb`` launch configuration. Only the local (no collective)
entry point is used.
"""

from __future__ import annotations

import functools

import torch

from minisgl.kernel.sgl_jit import cpp_args, load_jit

_DIM: int = 7168  # K3 hidden size, template parameter of the TMA kernel
_MAX_BANK_ROWS: int = 8  # K3 has <= 8 snapshots, upper bound of the nvb dispatch tables

# Benchmarked-best (chunk_rows, occupancy, consumer_regs) per nvb (GB200/GB300, H=7168),
# copied from sglang.
_TMA_BEST_CONFIG: dict[int, tuple[int, int, int]] = {
    1: (2, 2, 0),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


@functools.cache
def _module(chunk_rows: int, occupancy: int, consumer_regs: int):
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("attn_res_fused_tma requires SM100+ (tcgen05, cp.async.bulk)")
    args = cpp_args(_DIM, _MAX_BANK_ROWS, chunk_rows, occupancy, consumer_regs)
    return load_jit(
        "kimi_k3_attn_res_fused_tma",
        args,
        cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
        cuda_wrappers=[("run", f"AttnResFusedTmaKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def _tuning(nvb: int, num_tokens: int) -> tuple[int, int, int]:
    best = _TMA_BEST_CONFIG[nvb]
    if best[1] > 1 and num_tokens < 128:
        best = (4, 1, 200)
    return best


def covered(prefix: torch.Tensor, bank: torch.Tensor, nvb: int) -> bool:
    return (
        prefix.is_cuda
        and torch.cuda.get_device_capability(prefix.device)[0] >= 10
        and prefix.dtype == torch.bfloat16
        and bank.dtype == torch.bfloat16
        and prefix.shape[-1] == _DIM
        and 1 <= nvb <= _MAX_BANK_ROWS
        and nvb <= bank.shape[1]
        and prefix.is_contiguous()
        and bank.is_contiguous()
    )


def attn_res_fused_tma(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
) -> torch.Tensor:
    """RMSNorm_ow(sum_j softmax_j(<rmsnorm(v_j), cw>) v_j) over v = [bank[:, :nvb], prefix].

    prefix [T, H] bf16, bank [T, NB, H] bf16, cw / ow [H] bf16. With
    ``write_prefix`` the prefix row is also stored into ``bank[:, nvb]``.
    """
    out = torch.empty_like(prefix)
    if prefix.shape[0] == 0:
        return out
    _module(*_tuning(nvb, prefix.shape[0])).run(prefix, bank, cw, ow, out, int(nvb), float(eps), bool(write_prefix))
    return out


__all__ = ["attn_res_fused_tma", "covered"]
