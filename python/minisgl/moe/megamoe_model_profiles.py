"""Validated model contracts for MegaMoE latency measurements.

These profiles describe the MoE work executed by one decoder step.  They are
kept separate from serving-model registration because a shape-compatible
kernel benchmark is not evidence that the complete model can be served by
FastAFD.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

WeightPrecision = Literal["fp8", "fp4"]
ActivationKind = Literal["silu", "swigluoai"]


@dataclass(frozen=True)
class MegaMoEModelProfile:
    key: str
    model_path: str
    simulation_model_path: str
    architecture: str
    num_hidden_layers: int
    num_moe_layers: int
    hidden_size: int
    intermediate_size: int
    num_routed_experts: int
    routed_top_k: int
    num_shared_experts: int
    shared_intermediate_size: int
    gate_renormalize: bool
    routed_scaling_factor: float
    scoring_func: str
    has_router_correction_bias: bool
    activation: ActivationKind
    activation_alpha: float
    activation_clamp: float | None
    activation_up_bias: float
    weight_precision: WeightPrecision
    moe_quant_contract: str
    checkpoint_precision: str

    def validate(self, *, eg_size: int) -> None:
        expected_quant_contract = (
            "w4a8_mxfp4_mxfp8" if self.weight_precision == "fp4" else "fp8_e4m3_ue8m0"
        )
        if self.moe_quant_contract != expected_quant_contract:
            raise ValueError(
                f"{self.key}: weight_precision={self.weight_precision} requires "
                f"moe_quant_contract={expected_quant_contract}"
            )
        if self.hidden_size % 512:
            raise ValueError(f"{self.key}: hidden_size must be divisible by 512")
        if self.intermediate_size % 512:
            raise ValueError(f"{self.key}: intermediate_size must be divisible by 512")
        if self.num_shared_experts not in (0, 1):
            raise ValueError(f"{self.key}: split MegaMoE supports at most one shared expert")
        if self.num_shared_experts and self.shared_intermediate_size != self.intermediate_size:
            raise ValueError(
                f"{self.key}: shared and routed intermediate sizes must match for fused remote shared experts"
            )
        if eg_size <= 0 or self.num_routed_experts % eg_size:
            raise ValueError(
                f"{self.key}: num_routed_experts={self.num_routed_experts} "
                f"must be divisible by eg_size={eg_size}"
            )
        protocol_experts = self.num_routed_experts + eg_size * self.num_shared_experts
        if protocol_experts % eg_size:
            raise ValueError(
                f"{self.key}: protocol expert count {protocol_experts} "
                f"must be divisible by eg_size={eg_size}"
            )
        if self.activation == "silu":
            if self.activation_alpha != 1.0 or self.activation_up_bias != 0.0:
                raise ValueError(f"{self.key}: plain SiLU requires alpha=1 and up_bias=0")
        elif self.activation == "swigluoai":
            if self.activation_alpha <= 0 or self.activation_up_bias != 1.0:
                raise ValueError(f"{self.key}: SwiGLU-OAI requires positive alpha and up_bias=1")
            if self.activation_clamp is None:
                raise ValueError(f"{self.key}: SwiGLU-OAI requires an activation clamp")
        else:
            raise ValueError(f"{self.key}: unsupported activation {self.activation!r}")
        if self.routed_scaling_factor <= 0:
            raise ValueError(f"{self.key}: routed_scaling_factor must be positive")
        if not 0 < self.num_moe_layers <= self.num_hidden_layers:
            raise ValueError(f"{self.key}: invalid MoE layer count")

    @property
    def protocol_top_k(self) -> int:
        return self.routed_top_k + self.num_shared_experts

    def protocol_num_experts(self, eg_size: int) -> int:
        self.validate(eg_size=eg_size)
        return self.num_routed_experts + eg_size * self.num_shared_experts

    def local_experts(self, eg_size: int) -> int:
        return self.protocol_num_experts(eg_size) // eg_size

    def with_weight_precision(self, precision: WeightPrecision) -> "MegaMoEModelProfile":
        return replace(self, weight_precision=precision)


_PROFILES = {
    "qwen3_235b_fp8": MegaMoEModelProfile(
        key="qwen3_235b_fp8",
        model_path="Qwen/Qwen3-235B-A22B-FP8",
        simulation_model_path="Qwen/Qwen3-235B-A22B-FP8",
        architecture="Qwen3MoeForCausalLM",
        num_hidden_layers=94,
        num_moe_layers=94,
        hidden_size=4096,
        intermediate_size=1536,
        num_routed_experts=128,
        routed_top_k=8,
        num_shared_experts=0,
        shared_intermediate_size=0,
        gate_renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="softmax",
        has_router_correction_bias=False,
        activation="silu",
        activation_alpha=1.0,
        activation_clamp=None,
        activation_up_bias=0.0,
        weight_precision="fp8",
        moe_quant_contract="fp8_e4m3_ue8m0",
        checkpoint_precision=(
            "synthetic deterministic E4M3 benchmark weights with 128x128 source scales, "
            "requantized to per-32 UE8M0 for MegaMoE; no checkpoint weights are loaded"
        ),
    ),
    "minimax_m25_fp8": MegaMoEModelProfile(
        key="minimax_m25_fp8",
        model_path="MiniMaxAI/MiniMax-M2.5",
        simulation_model_path="MiniMaxAI/MiniMax-M2.5",
        architecture="MiniMaxM2ForCausalLM",
        num_hidden_layers=62,
        num_moe_layers=62,
        hidden_size=3072,
        intermediate_size=1536,
        num_routed_experts=256,
        routed_top_k=8,
        num_shared_experts=0,
        shared_intermediate_size=0,
        gate_renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        has_router_correction_bias=True,
        activation="silu",
        activation_alpha=1.0,
        activation_clamp=None,
        activation_up_bias=0.0,
        weight_precision="fp8",
        moe_quant_contract="fp8_e4m3_ue8m0",
        checkpoint_precision=(
            "synthetic deterministic E4M3 benchmark weights with 128x128 source scales, "
            "requantized to per-32 UE8M0 for MegaMoE; no checkpoint weights are loaded"
        ),
    ),
    "minimax_m3_fp8": MegaMoEModelProfile(
        key="minimax_m3_fp8",
        model_path="MiniMaxAI/MiniMax-M3",
        simulation_model_path="MiniMaxAI/MiniMax-M3",
        architecture="MiniMaxM3ForCausalLM",
        num_hidden_layers=60,
        num_moe_layers=57,
        hidden_size=6144,
        intermediate_size=3072,
        num_routed_experts=128,
        routed_top_k=4,
        num_shared_experts=1,
        shared_intermediate_size=3072,
        gate_renormalize=True,
        routed_scaling_factor=2.0,
        scoring_func="sigmoid",
        has_router_correction_bias=True,
        activation="swigluoai",
        activation_alpha=1.702,
        activation_clamp=7.0,
        activation_up_bias=1.0,
        weight_precision="fp8",
        moe_quant_contract="fp8_e4m3_ue8m0",
        checkpoint_precision=(
            "synthetic deterministic E4M3 benchmark weights with 128x128 source scales, "
            "requantized to per-32 UE8M0 for MegaMoE; no checkpoint weights are loaded"
        ),
    ),
    "deepseek_v4_flash_fp8": MegaMoEModelProfile(
        key="deepseek_v4_flash_fp8",
        model_path="sgl-project/DeepSeek-V4-Flash-FP8",
        simulation_model_path="sgl-project/DeepSeek-V4-Flash-FP8",
        architecture="DeepseekV4ForCausalLM",
        num_hidden_layers=43,
        num_moe_layers=43,
        hidden_size=4096,
        intermediate_size=2048,
        num_routed_experts=256,
        routed_top_k=6,
        num_shared_experts=1,
        shared_intermediate_size=2048,
        gate_renormalize=True,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        has_router_correction_bias=True,
        activation="silu",
        activation_alpha=1.0,
        activation_clamp=10.0,
        activation_up_bias=0.0,
        weight_precision="fp8",
        moe_quant_contract="fp8_e4m3_ue8m0",
        checkpoint_precision=(
            "synthetic deterministic E4M3 benchmark weights with 128x128 source scales, "
            "requantized to per-32 UE8M0 for MegaMoE; no checkpoint weights are loaded"
        ),
    ),
    "deepseek_v4_pro_fp8": MegaMoEModelProfile(
        key="deepseek_v4_pro_fp8",
        model_path="sgl-project/DeepSeek-V4-Pro-FP8",
        simulation_model_path="sgl-project/DeepSeek-V4-Pro-FP8",
        architecture="DeepseekV4ForCausalLM",
        num_hidden_layers=61,
        num_moe_layers=61,
        hidden_size=7168,
        intermediate_size=3072,
        num_routed_experts=384,
        routed_top_k=6,
        num_shared_experts=1,
        shared_intermediate_size=3072,
        gate_renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        has_router_correction_bias=True,
        activation="silu",
        activation_alpha=1.0,
        activation_clamp=10.0,
        activation_up_bias=0.0,
        weight_precision="fp8",
        moe_quant_contract="fp8_e4m3_ue8m0",
        checkpoint_precision=(
            "synthetic deterministic E4M3 benchmark weights with 128x128 source scales, "
            "requantized to per-32 UE8M0 for MegaMoE; no checkpoint weights are loaded"
        ),
    ),
}

_PROFILES["qwen3_235b_fp4"] = replace(
    _PROFILES["qwen3_235b_fp8"].with_weight_precision("fp4"),
    key="qwen3_235b_fp4",
    model_path="Qwen/Qwen3-235B-A22B-FP8",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)
_PROFILES["minimax_m25_fp4"] = replace(
    _PROFILES["minimax_m25_fp8"].with_weight_precision("fp4"),
    key="minimax_m25_fp4",
    model_path="MiniMaxAI/MiniMax-M2.5",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)
_PROFILES["minimax_m3_fp4"] = _PROFILES["minimax_m3_fp8"].with_weight_precision("fp4")
_PROFILES["minimax_m3_fp4"] = replace(
    _PROFILES["minimax_m3_fp4"],
    key="minimax_m3_fp4",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)

_PROFILES["deepseek_v4_flash_fp4"] = _PROFILES["deepseek_v4_flash_fp8"].with_weight_precision("fp4")
_PROFILES["deepseek_v4_flash_fp4"] = replace(
    _PROFILES["deepseek_v4_flash_fp4"],
    key="deepseek_v4_flash_fp4",
    model_path="deepseek-ai/DeepSeek-V4-Flash",
    simulation_model_path="deepseek-ai/DeepSeek-V4-Flash",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)
_PROFILES["deepseek_v4_pro_fp4"] = _PROFILES["deepseek_v4_pro_fp8"].with_weight_precision("fp4")
_PROFILES["deepseek_v4_pro_fp4"] = replace(
    _PROFILES["deepseek_v4_pro_fp4"],
    key="deepseek_v4_pro_fp4",
    model_path="deepseek-ai/DeepSeek-V4-Pro",
    simulation_model_path="deepseek-ai/DeepSeek-V4-Pro",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)

# Kimi-K3 LatentMoE: the 896 routed experts run entirely in the 3584 latent
# space, so the dispatched stage is (3584 -> 3072 -> 3584, top-16). The
# hidden<->latent projections and the two shared experts (full 7168 hidden,
# 2 x 3072) are dense GEMMs outside this stage, as in AIC's KimiK3 model.
# SiTU is not a MegaMoE epilogue; SiLU stands in with the same GEMM shapes.
_PROFILES["kimi_k3_fp4"] = MegaMoEModelProfile(
    key="kimi_k3_fp4",
    model_path="moonshotai/Kimi-K3",
    simulation_model_path="moonshotai/Kimi-K3",
    architecture="KimiK3ForConditionalGeneration",
    num_hidden_layers=93,
    num_moe_layers=92,
    hidden_size=3584,
    intermediate_size=3072,
    num_routed_experts=896,
    routed_top_k=16,
    num_shared_experts=0,
    shared_intermediate_size=0,
    gate_renormalize=True,
    routed_scaling_factor=1.0,
    scoring_func="sigmoid",
    has_router_correction_bias=True,
    activation="silu",
    activation_alpha=1.0,
    activation_clamp=None,
    activation_up_bias=0.0,
    weight_precision="fp4",
    moe_quant_contract="w4a8_mxfp4_mxfp8",
    checkpoint_precision=(
        "synthetic deterministic BF16 benchmark weights quantized to E2M1 with "
        "UE8M0 block-32 scales; no checkpoint weights are loaded"
    ),
)

MEGAMOE_MODEL_PROFILES = dict(_PROFILES)


def get_megamoe_model_profile(key: str) -> MegaMoEModelProfile:
    try:
        return MEGAMOE_MODEL_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MEGAMOE_MODEL_PROFILES))
        raise KeyError(f"unknown MegaMoE model profile {key!r}; choose one of: {choices}") from exc


__all__ = [
    "ActivationKind",
    "MEGAMOE_MODEL_PROFILES",
    "MegaMoEModelProfile",
    "WeightPrecision",
    "get_megamoe_model_profile",
]
