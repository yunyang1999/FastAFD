"""Deterministic dummy weights with sane magnitudes for DSV4 / K3.

``torch.randn_like`` on every tensor (the generic dummy path) produces NaNs in
deep hyper-connection / delta-rule stacks and invalid FP8/FP4 encodings, so these
models use name-based rules: linear weights ~ N(0, 1/sqrt(fan_in)), norms = 1,
FP8 weights block-quantised from such a bf16 tensor (128x128 scales), MXFP4
experts as random E2M1 codes with a matching UE8M0 scale, plus model-specific
parameters (hash tables, KDA decay, hyper-connection mixers).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from .moe_common import random_mxfp4


def _quant_fp8_block(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    out, inp = w.shape
    po, pi = (out + block - 1) // block * block, (inp + block - 1) // block * block
    wp = torch.zeros(po, pi, dtype=torch.float32, device=w.device)
    wp[:out, :inp] = w.float()
    v = wp.view(po // block, block, pi // block, block)
    amax = v.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    s = amax / 448.0
    q = (v / s).to(torch.float8_e4m3fn)
    return q.view(po, pi)[:out, :inp].contiguous(), s.view(po // block, pi // block).contiguous()


def _normal(shape, std, device, gen, dtype):
    return (torch.randn(shape, device=device, generator=gen, dtype=torch.float32) * std).to(dtype)


def init_dummy_state_dict(model, device: torch.device, seed: int = 1234) -> Dict[str, torch.Tensor]:
    meta = model.state_dict()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for name, t in meta.items():
        if name in out:
            continue
        shape, dtype = tuple(t.shape), t.dtype
        leaf = name.rsplit(".", 1)[-1]
        # ---- quantised linear (fp8 weight + fp32 block scale) ----
        if dtype == torch.float8_e4m3fn and leaf == "weight":
            w = _normal(shape, 1.0 / math.sqrt(shape[1]), device, gen, torch.float32)
            q, s = _quant_fp8_block(w)
            out[name] = q
            sname = name[: -len("weight")] + "weight_scale"
            if sname in meta:
                assert tuple(meta[sname].shape) == tuple(s.shape), (sname, meta[sname].shape, s.shape)
                out[sname] = s.to(meta[sname].dtype)
            continue
        if leaf == "weight_scale":
            continue  # produced with its weight
        # ---- MXFP4 routed experts ----
        if dtype == torch.uint8 and leaf in ("w13", "w2"):
            fan_in = shape[-1] * 2
            p, sf = random_mxfp4((*shape[:-1], fan_in), std=1.0 / math.sqrt(fan_in), device=device, generator=gen)
            out[name] = p
            out[name + "_sf"] = sf
            continue
        if leaf in ("w13_sf", "w2_sf"):
            continue
        # ---- model specific ----
        if leaf == "tid2eid":
            V, k = shape
            n_exp = int(model_num_experts(model))
            stride = max(1, n_exp // k)
            base = torch.randint(0, n_exp, (V, 1), device=device, generator=gen)
            out[name] = ((base + torch.arange(k, device=device)[None, :] * stride) % n_exp).to(dtype)
            continue
        if leaf == "A_log":
            u = torch.rand(shape, device=device, generator=gen) * 15 + 1
            out[name] = torch.log(u).to(dtype)
            continue
        if leaf == "dt_bias":
            out[name] = _normal(shape, 0.1, device, gen, dtype)
            continue
        if "conv" in name and len(shape) == 2:
            out[name] = _normal(shape, 0.5, device, gen, dtype)
            continue
        if leaf.startswith("hc_") and leaf.endswith("_fn"):
            out[name] = _normal(shape, 1.0 / math.sqrt(shape[1]), device, gen, dtype)
            continue
        if leaf.startswith("hc_") and leaf.endswith("_scale"):
            out[name] = torch.ones(shape, device=device, dtype=dtype)
            continue
        if leaf.startswith("hc_") and leaf.endswith("_base"):
            out[name] = _normal(shape, 0.1, device, gen, dtype)
            continue
        if leaf in ("attn_sink", "ape", "bias", "e_score_correction_bias"):
            out[name] = torch.zeros(shape, device=device, dtype=dtype)
            continue
        if name.endswith("norm.weight") or leaf == "norm_weight":
            out[name] = torch.ones(shape, device=device, dtype=dtype)
            continue
        if "embed" in name and len(shape) == 2:
            out[name] = _normal(shape, 1.0, device, gen, dtype)
            continue
        if dtype in (torch.int32, torch.int64):
            out[name] = torch.zeros(shape, device=device, dtype=dtype)
            continue
        if len(shape) >= 2:
            out[name] = _normal(shape, 1.0 / math.sqrt(shape[-1]), device, gen, dtype)
            continue
        out[name] = torch.zeros(shape, device=device, dtype=dtype)
    missing = [k for k in meta if k not in out]
    if missing:
        raise RuntimeError(f"dummy init did not produce: {missing[:8]}")
    return out


def model_num_experts(model) -> int:
    for attr in ("num_routed_experts", "_num_experts"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    inner = getattr(model, "model", None)
    args = getattr(inner, "_args", None)
    if args is not None and hasattr(args, "n_routed"):
        return int(args.n_routed)
    raise RuntimeError("cannot infer number of experts for tid2eid dummy init")


__all__ = ["init_dummy_state_dict"]
