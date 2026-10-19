"""AFD glue for DeepSeek-V4 and Kimi-K3.

The F side (expert workers) runs exactly the measured MegaMoE stage of
``moe/megamoe_model_profiles.py``; this module derives that stage contract from
the profile, cross-checks it against the HF config, and provides the
attention-stage class, the MoE layer ids and the expert-weight provider.

* DSV4 (``deepseek_v4_{flash,pro}_fp4``): routed experts + 1 shared expert
  (replicated per expert rank, synthetic protocol slot with weight 1/rsf),
  SwiGLU clamp 10, rsf applied on the attention side after combine.
* K3 (``kimi_k3_fp4``): 896 routed experts in the 3584-d latent space, no shared
  expert on the F side; router, latent down/up projection + norm, 2 shared experts
  and dense layer 0 stay on the attention side. SiLU stands in for SiTU in the
  MegaMoE epilogue (profile contract).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Iterator

import torch


@dataclass(frozen=True)
class HybridAfdSpec:
    profile_key: str
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    shared_experts_per_rank: int
    routed_scaling_factor: float
    activation: str
    activation_alpha: float
    activation_clamp: float | None
    activation_up_bias: float
    weight_precision: str
    gate_renormalize: bool
    moe_layer_ids: tuple[int, ...]

    def adapter_kwargs(self) -> dict:
        return dict(
            real_num_experts=self.num_experts,
            hidden_size=self.hidden,
            intermediate_size=self.intermediate,
            top_k=self.top_k,
            gate_renormalize=self.gate_renormalize,
            shared_experts_per_rank=self.shared_experts_per_rank,
            routed_scaling_factor=self.routed_scaling_factor,
            weight_precision=self.weight_precision,
            activation=self.activation,
            activation_alpha=self.activation_alpha,
            activation_clamp=self.activation_clamp,
            activation_up_bias=self.activation_up_bias,
        )


def _profile_key(mc) -> str | None:
    if getattr(mc, "is_deepseek_v4", False):
        d = mc.hf_text_config
        flash = int(d["hidden_size"]) == 4096
        return os.environ.get("MINISGL_AFD_MEGAMOE_PROFILE", "deepseek_v4_flash_fp4" if flash else "deepseek_v4_pro_fp4")
    if getattr(mc, "is_kimi_k3", False):
        return os.environ.get("MINISGL_AFD_MEGAMOE_PROFILE", "kimi_k3_fp4")
    return None


def hybrid_afd_spec(mc) -> HybridAfdSpec | None:
    key = _profile_key(mc)
    if key is None:
        return None
    from minisgl.moe.megamoe_model_profiles import get_megamoe_model_profile

    p = get_megamoe_model_profile(key)
    d = mc.hf_text_config
    L = int(mc.num_layers)
    if mc.is_deepseek_v4:
        expect = dict(hidden=int(d["hidden_size"]), inter=int(d["moe_intermediate_size"]),
                      experts=int(d["n_routed_experts"]), topk=int(d["num_experts_per_tok"]),
                      shared=int(d.get("n_shared_experts", 1)), rsf=float(d["routed_scaling_factor"]))
        moe_layers = tuple(range(L))
    else:
        expect = dict(hidden=int(d["routed_expert_hidden_size"]), inter=int(d["moe_intermediate_size"]),
                      experts=int(d["num_experts"]), topk=int(d["num_experts_per_token"]),
                      shared=0, rsf=float(d.get("routed_scaling_factor", 1.0)))
        moe_layers = tuple(range(int(d.get("first_k_dense_replace", 0)), L))
    got = dict(hidden=p.hidden_size, inter=p.intermediate_size, experts=p.num_routed_experts,
               topk=p.routed_top_k, shared=p.num_shared_experts, rsf=p.routed_scaling_factor)
    # reduced-expert test configs keep the profile's dims but not its expert count
    mism = {k: (got[k], v) for k, v in expect.items() if got[k] != v and k != "experts"}
    if mism:
        raise RuntimeError(f"MegaMoE profile {key} does not match the model config: {mism}")
    return HybridAfdSpec(
        profile_key=key,
        hidden=p.hidden_size,
        intermediate=p.intermediate_size,
        num_experts=expect["experts"],
        top_k=p.routed_top_k,
        shared_experts_per_rank=p.num_shared_experts,
        routed_scaling_factor=p.routed_scaling_factor,
        activation=p.activation,
        activation_alpha=p.activation_alpha,
        activation_clamp=p.activation_clamp,
        activation_up_bias=p.activation_up_bias,
        weight_precision=p.weight_precision,
        gate_renormalize=p.gate_renormalize,
        moe_layer_ids=moe_layers,
    )


def attention_stage_class(mc):
    if mc.is_deepseek_v4:
        from .deepseek_v4_afd import DeepseekV4AfdAttention

        return DeepseekV4AfdAttention
    from .kimi_k3_afd import KimiK3AfdAttention

    return KimiK3AfdAttention


def dummy_eg_layer_weights(spec: HybridAfdSpec, eg_rank: int, eg_size: int, layer_id: int, device) -> tuple:
    """Random MXFP4 expert weights for one MoE layer on one expert rank:
    (w13, w13_sf, w2, w2_sf) with the shared expert (if any) appended."""
    from .moe_common import random_mxfp4

    n_local = spec.num_experts // eg_size + spec.shared_experts_per_rank
    gen = torch.Generator(device=device)
    gen.manual_seed(1000003 * (layer_id + 1) + eg_rank)
    w13, s13 = random_mxfp4((n_local, 2 * spec.intermediate, spec.hidden), std=1 / math.sqrt(spec.hidden), device=device, generator=gen)
    w2, s2 = random_mxfp4((n_local, spec.hidden, spec.intermediate), std=1 / math.sqrt(spec.intermediate), device=device, generator=gen)
    return w13, s13, w2, s2


__all__ = ["HybridAfdSpec", "hybrid_afd_spec", "attention_stage_class", "dummy_eg_layer_weights"]
