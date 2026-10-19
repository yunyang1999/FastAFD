"""Shared MoE pieces for DeepSeek-V4 and Kimi-K3.

Routed expert weights are stored in the checkpoint's MXFP4 layout (also the
MegaMoE weight contract): E2M1 codes packed two per byte along K (low nibble =
even element) with one UE8M0 exponent byte per 32 elements of K:

    w13    uint8 [E_local, 2*I, D/2]   gate rows then up rows
    w13_sf uint8 [E_local, 2*I, D/32]
    w2     uint8 [E_local, D, I/2]
    w2_sf  uint8 [E_local, D, I/32]

Execution backends:
* ``megamoe``: colocated DeepGEMM MegaMoE (FP8 activations x FP4 weights), the
  same kernel family as the measured MegaMoE stage; experts are sharded across
  the TP group (EP = TP) and tokens are split across ranks (SP-MoE).
* ``torch``: reference grouped computation on dequantized BF16 weights (exact
  model activation incl. SiTU), used for numerics and small tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from minisgl.layers import BaseOP

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def moe_exec_backend() -> str:
    return os.environ.get("MINISGL_MOE_EXEC", "megamoe").lower()


# ---------------------------------------------------------------------------
# FP4 helpers
# ---------------------------------------------------------------------------


def ue8m0_to_float(sf: torch.Tensor) -> torch.Tensor:
    """uint8 UE8M0 exponents -> float32 powers of two (2**(e-127))."""
    return (sf.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp4(packed: torch.Tensor, sf: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """packed uint8 [..., N, K/2] + sf uint8 [..., N, K/32] -> [..., N, K]."""
    lut = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), dtype=torch.float32, device=packed.device)
    p = packed.to(torch.int32)
    lo = lut[p & 0xF]
    hi = lut[(p >> 4) & 0xF]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [..., N, K]
    scale = ue8m0_to_float(sf).repeat_interleave(32, dim=-1)
    return (vals * scale).to(dtype)


def _dequant_experts(packed: torch.Tensor, sf: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """dequant_mxfp4 over [E, N, K/2] in expert chunks (bounded fp32 temporaries)."""
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.bfloat16, device=packed.device)
    for e in range(0, packed.shape[0], chunk):
        out[e : e + chunk] = dequant_mxfp4(packed[e : e + chunk], sf[e : e + chunk])
    return out


def quant_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., N, K] float -> (packed uint8 [..., N, K/2], sf uint8 [..., N, K/32]).

    Power-of-two ceil scale (amax / 6) and round-to-nearest-even E2M1, the same
    contract as DeepGEMM's ``per_token_cast_to_fp4(use_ue8m0=True)``."""
    shp = w.shape
    xb = w.float().unflatten(-1, (-1, 32))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    bits = (amax / 6.0).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    e = e.clamp(1, 254)
    s = (e << 23).view(torch.float32)
    y = (xb / s).clamp(-6, 6)
    # same rounding as DeepGEMM's _quantize_to_fp4_e2m1 (bucketize, ties down)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    code = torch.bucketize(y.abs(), bounds).to(torch.int32)
    code = code | ((y < 0) & (code != 0)).to(torch.int32) << 3
    code = code.flatten(-2).view(*shp[:-1], shp[-1])
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, e.squeeze(-1).to(torch.uint8).view(*shp[:-1], shp[-1] // 32)


def random_mxfp4(shape_nk: tuple[int, ...], *, std: float, device, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random packed FP4 weights with RMS ~= std (dummy weights, no quant pass)."""
    *lead, n, k = shape_nk
    packed = torch.randint(0, 256, (*lead, n, k // 2), dtype=torch.uint8, device=device, generator=generator)
    rms_codes = (sum(v * v for v in FP4_VALUES) / len(FP4_VALUES)) ** 0.5
    import math

    e = int(round(127 + math.log2(std / rms_codes)))
    sf = torch.full((*lead, n, k // 32), e, dtype=torch.uint8, device=device)
    return packed, sf


def to_megamoe_weights(w13, w13_sf, w2, w2_sf):
    """Checkpoint MXFP4 layout -> MegaMoE transformed (l1, l2) tuples."""
    from minisgl.kernel import deepgemm as dg
    from minisgl.kernel.megamoe_m2n_mega import transform_weights_for_mega_moe

    E, n1, k1h = w13.shape
    _, n2, k2h = w2.shape
    l1_sf = dg.transform_sf_into_required_layout(ue8m0_to_float(w13_sf).contiguous(), n1, 2 * k1h, (1, 32), E)
    l2_sf = dg.transform_sf_into_required_layout(ue8m0_to_float(w2_sf).contiguous(), n2, 2 * k2h, (1, 32), E)
    return transform_weights_for_mega_moe(
        (w13.view(torch.int8).contiguous(), l1_sf), (w2.view(torch.int8).contiguous(), l2_sf)
    )


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


def swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    gate = gate.float()
    up = up.float()
    if limit is not None and limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up


def situ_and_mul(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    gate = gate.float()
    up = up.float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return a * up


@dataclass(frozen=True)
class ActivationSpec:
    kind: str  # "swiglu" | "situ"
    clamp: float | None = None
    beta: float = 1.0
    linear_beta: float | None = None

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.kind == "situ":
            return situ_and_mul(gate, up, self.beta, self.linear_beta)
        return swiglu_clamped(gate, up, self.clamp)


# ---------------------------------------------------------------------------
# routed expert holder
# ---------------------------------------------------------------------------


class FP4RoutedExperts(BaseOP):
    """Routed experts of one layer in MXFP4 checkpoint layout (local shard)."""

    def __init__(self, num_local: int, hidden: int, intermediate: int):
        self.w13 = torch.empty(num_local, 2 * intermediate, hidden // 2, dtype=torch.uint8)
        self.w13_sf = torch.empty(num_local, 2 * intermediate, hidden // 32, dtype=torch.uint8)
        self.w2 = torch.empty(num_local, hidden, intermediate // 2, dtype=torch.uint8)
        self.w2_sf = torch.empty(num_local, hidden, intermediate // 32, dtype=torch.uint8)
        self._hidden = hidden
        self._intermediate = intermediate
        self._bf16_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._mega: tuple | None = None

    def forward(self, *args, **kwargs):  # pragma: no cover - not called directly
        raise RuntimeError("FP4RoutedExperts is a weight holder")

    def bf16(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bf16_cache is None:
            self._bf16_cache = (_dequant_experts(self.w13, self.w13_sf), _dequant_experts(self.w2, self.w2_sf))
        return self._bf16_cache

    def megamoe(self, extra: tuple[torch.Tensor, ...] | None = None):
        """Transformed MegaMoE weights; ``extra`` = (w13, w13_sf, w2, w2_sf) of one
        replicated shared expert appended after the routed experts."""
        if self._mega is None:
            w13, w13_sf, w2, w2_sf = self.w13, self.w13_sf, self.w2, self.w2_sf
            if extra is not None:
                w13 = torch.cat([w13, extra[0]], 0)
                w13_sf = torch.cat([w13_sf, extra[1]], 0)
                w2 = torch.cat([w2, extra[2]], 0)
                w2_sf = torch.cat([w2_sf, extra[3]], 0)
            self._mega = to_megamoe_weights(w13, w13_sf, w2, w2_sf)
        return self._mega

    def drop_packed(self) -> None:
        dev = self.w13.device
        for name in ("w13", "w13_sf", "w2", "w2_sf"):
            setattr(self, name, torch.empty(0, dtype=torch.uint8, device=dev))


def torch_ref_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    act: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    expert_offset: int = 0,
) -> torch.Tensor:
    """y = sum_k w_k * expert_{id_k}(x) over locally-held experts (fp32 accumulate)."""
    T, D = x.shape
    E = w13.shape[0]
    inter = w13.shape[1] // 2
    out = torch.zeros(T, D, dtype=torch.float32, device=x.device)
    if T == 0:
        return out.to(x.dtype)
    local = topk_ids.long() - expert_offset
    valid = (local >= 0) & (local < E)
    if torch.cuda.is_current_stream_capturing():
        # static-shape variant for CUDA graphs: every local expert on every token
        dense = torch.zeros(T, E, dtype=torch.float32, device=x.device)
        dense.scatter_add_(1, local.clamp(0, E - 1), torch.where(valid, topk_weights.float(), 0.0))
        for e in range(E):
            h = x.float() @ w13[e].float().t()
            a = act(h[:, :inter], h[:, inter:])
            out += (a.to(x.dtype).float() @ w2[e].float().t()) * dense[:, e : e + 1]
        return out.to(x.dtype)
    for e in torch.unique(local[valid]).tolist():
        tok, slot = torch.where(local == e)
        h = x[tok].float() @ w13[e].float().t()
        a = act(h[:, :inter], h[:, inter:])
        y = a.to(x.dtype).float() @ w2[e].float().t()
        out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# colocated MegaMoE executor (EP over the TP group, SP token split)
# ---------------------------------------------------------------------------


class ColocatedMegaMoE:
    """One symmetric buffer shared by all MoE layers of a model on this rank."""

    def __init__(
        self,
        *,
        num_routed: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        num_shared: int,
        routed_scaling_factor: float,
        activation_clamp: float | None,
        max_tokens_per_rank: int,
    ):
        import torch.distributed as dist
        from minisgl.distributed import get_tp_info
        from minisgl.kernel.megamoe_mega import MegaMoESymmBuffer

        tp = get_tp_info()
        self.ep_size = tp.size
        self.rank = tp.rank
        assert num_routed % self.ep_size == 0
        self.num_routed = num_routed
        self.local_routed = num_routed // self.ep_size
        self.num_shared = int(num_shared)
        self.local_protocol = self.local_routed + self.num_shared
        self.hidden = hidden
        self.top_k = top_k
        self.rsf = float(routed_scaling_factor)
        self.clamp = activation_clamp
        group = dist.group.WORLD
        self.buf = MegaMoESymmBuffer(
            group,
            num_experts=self.local_protocol * self.ep_size,
            num_max_tokens_per_rank=max_tokens_per_rank,
            num_topk=top_k + self.num_shared,
            hidden=hidden,
            intermediate_hidden=intermediate,
        )
        self.max_tokens = self.buf.num_max_tokens_per_rank

    def protocol_routes(self, ids: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = ids.long()
        owner = torch.div(ids, self.local_routed, rounding_mode="floor")
        pid = owner * self.local_protocol + ids.remainder(self.local_routed)
        w = w.float()
        if self.num_shared:
            T = ids.shape[0]
            sid = torch.full((T, 1), self.rank * self.local_protocol + self.local_routed, dtype=torch.int64, device=ids.device)
            sw = torch.full((T, 1), 1.0 / self.rsf, dtype=torch.float32, device=ids.device)
            pid = torch.cat([pid, sid], 1)
            w = torch.cat([w, sw], 1)
        return pid.contiguous(), w.contiguous()

    def run_local(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, mega_weights) -> torch.Tensor:
        """x [n<=max_tokens, hidden]: dispatch + experts + combine (collective over EP)."""
        from minisgl.kernel import deepgemm as dg
        from minisgl.kernel.megamoe_mega import fp8_fp4_mega_moe

        n = int(x.shape[0])
        if n > self.max_tokens:
            raise RuntimeError(f"MegaMoE tokens {n} > buffer {self.max_tokens}")
        pid, pw = self.protocol_routes(ids, w)
        if n > 0:
            xf8, xsf = dg.per_token_cast_to_fp8(x.contiguous(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            self.buf.x[:n].copy_(xf8)
            self.buf.x_sf[:n].copy_(xsf)
            self.buf.topk_idx[:n].copy_(pid)
            self.buf.topk_weights[:n].copy_(pw)
        y = torch.empty((n, self.hidden), dtype=torch.bfloat16, device=x.device)
        fp8_fp4_mega_moe(y, mega_weights[0], mega_weights[1], self.buf, activation_clamp=self.clamp)
        if self.rsf != 1.0:
            y = y * self.rsf
        return y

    def __call__(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, mega_weights) -> torch.Tensor:
        """x is replicated across the TP group: split tokens (SP), run, all-gather."""
        if self.ep_size == 1:
            return self.run_local(x, ids, w, mega_weights)
        from minisgl.distributed import DistributedCommunicator

        T = int(x.shape[0])
        per = (T + self.ep_size - 1) // self.ep_size
        pad = per * self.ep_size - T
        if pad:
            x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
            ids = torch.cat([ids, ids.new_zeros(pad, ids.shape[1])])
            w = torch.cat([w, w.new_zeros(pad, w.shape[1])])
        sl = slice(self.rank * per, (self.rank + 1) * per)
        y_local = self.run_local(x[sl], ids[sl], w[sl], mega_weights)
        y = DistributedCommunicator().all_gather(y_local.contiguous())
        return y[:T]


__all__ = [
    "ActivationSpec",
    "ColocatedMegaMoE",
    "FP4RoutedExperts",
    "dequant_mxfp4",
    "moe_exec_backend",
    "quant_mxfp4",
    "random_mxfp4",
    "situ_and_mul",
    "swiglu_clamped",
    "to_megamoe_weights",
    "torch_ref_moe",
    "ue8m0_to_float",
]
