from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    parameters: Dict[str, Any] | None = None
    partial_rotary_factor: float | None = None


@dataclass(frozen=True)
class QuantConfig:
    method: str
    activation_scheme: str
    weight_block_size: Tuple[int, int]


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    use_routing_bias: bool
    scoring_func: str
    model_type: str
    architectures: list[str]
    quant: Optional[QuantConfig] = None
    use_qk_norm: bool = False
    rope_parameters: Dict[str, Any] | None = None
    rope_scaling: Dict[str, Any] | None = None
    rope_theta: float = 10000.0
    partial_rotary_factor: float | None = None
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def arch(self) -> str:
        return (self.architectures or [""])[0]

    @property
    def is_deepseek_v4(self) -> bool:
        return self.model_type == "deepseek_v4"

    @property
    def is_kimi_k3(self) -> bool:
        return self.model_type in ("kimi_k3", "kimi_linear")

    @property
    def hf_text_config(self) -> Dict[str, Any]:
        """Raw (text) HF config dict carrying model-specific fields."""
        return self.model_extra.get("hf_text_config", {})

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        model_type = getattr(config, "model_type", "") or ""
        archs = list(getattr(config, "architectures", None) or [])
        if model_type == "deepseek_v4" or "DeepseekV4ForCausalLM" in archs:
            return _deepseek_v4_from_hf(cls, config)
        if model_type in ("kimi_k3", "kimi_linear") or any(
            a in ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM") for a in archs
        ):
            return _kimi_k3_from_hf(cls, config)
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling", "rope_parameters"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        n_routed_experts = getattr(config, "n_routed_experts", None)
        if n_routed_experts is None:
            num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
            n_routed_experts = num_experts
        else:
            num_experts = n_routed_experts
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0) or config.intermediate_size
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        use_routing_bias = getattr(config, "use_routing_bias", False)
        scoring_func = getattr(config, "scoring_func", "softmax")
        if model_type == "minimax_m2" and scoring_func == "sigmoid" and not hasattr(config, "norm_topk_prob"):
            # vLLM hardcodes MiniMax M2 sigmoid-routing weights to be
            # renormalized; the HF config currently omits norm_topk_prob.
            norm_topk_prob = True
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])
        use_qk_norm = getattr(config, "use_qk_norm", False)
        model_extra: Dict[str, Any] = {}
        if model_type == "glm4_moe" or "Glm4MoeForCausalLM" in architectures:
            for attr, default in (
                ("n_shared_experts", 0),
                ("n_group", 0),
                ("topk_group", 0),
                ("first_k_dense_replace", 0),
                ("routed_scaling_factor", 1.0),
                ("attention_bias", False),
            ):
                value = getattr(config, attr, default)
                model_extra[attr] = default if value is None else value

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # MiniMax M2 also uses partial rotary via rotary_dim.
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", None) or getattr(config, "rope_theta", None)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", None)
            if rope_theta is None and rope_scaling is not None:
                rope_theta = rope_scaling["rope_theta"]
            if rope_theta is None:
                rope_theta = 10000.0
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None and isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor")
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is None:
            rotary_dim = (
                int(head_dim * float(partial_rotary_factor))
                if partial_rotary_factor is not None
                else head_dim
            )

        quant: Optional[QuantConfig] = None
        quant_config = getattr(config, "quantization_config", None)
        if quant_config is not None:
            quant_dict = (
                quant_config
                if isinstance(quant_config, dict)
                else getattr(quant_config, "to_dict", lambda: {})()
            )
            quant_method = quant_dict.get("quant_method", "") or ""
            if "fp8" in quant_method:
                block_size = quant_dict.get("weight_block_size", [128, 128])
                quant = QuantConfig(
                    method="fp8",
                    activation_scheme=quant_dict.get("activation_scheme", "dynamic"),
                    weight_block_size=(int(block_size[0]), int(block_size[1])),
                )
            elif (
                quant_method == "compressed-tensors"
                and quant_dict.get("format") == "float-quantized"
            ):
                # GLM FP8 checkpoints store per-output-channel FP8 weights plus
                # `.weight_scale`.  Runtime code repacks these channel-scale
                # tensors into DeepGEMM's 128x128 block-scale FP8 format once
                # after loading.
                quant = QuantConfig(
                    method="fp8_channel",
                    activation_scheme="dynamic",
                    weight_block_size=(1, 1),
                )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
                parameters=rope_parameters,
                partial_rotary_factor=partial_rotary_factor,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            use_routing_bias=use_routing_bias,
            scoring_func=scoring_func,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            use_qk_norm=use_qk_norm,
            rope_parameters=rope_parameters,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            model_extra=model_extra,
        )


# ---------------------------------------------------------------------------
# DeepSeek-V4 / Kimi-K3 (hybrid attention, latent / hyper-connection models)
# ---------------------------------------------------------------------------


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(vars(config))


def _rope_cfg(head_dim: int, rotary_dim: int, d: Dict[str, Any]) -> RotaryConfig:
    # YaRN for DeepSeek-V4 is applied model-locally (see models/deepseek_v4.py);
    # the generic rope factory is never used for these models.
    rope_scaling = d.get("rope_scaling")
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(d.get("max_position_embeddings", 4096)),
        base=float(d.get("rope_theta", 10000.0) or 10000.0),
        scaling=None,
        parameters=dict(rope_scaling) if rope_scaling else None,
        partial_rotary_factor=None,
    )


def _deepseek_v4_from_hf(cls, config: Any) -> "ModelConfig":
    d = _config_to_dict(config)
    num_layers = int(d["num_hidden_layers"])
    quant = None
    qc = d.get("quantization_config") or {}
    if "fp8" in str(qc.get("quant_method", "")):
        block = qc.get("weight_block_size", [128, 128])
        quant = QuantConfig(
            method="fp8",
            activation_scheme=qc.get("activation_scheme", "dynamic"),
            weight_block_size=(int(block[0]), int(block[1])),
        )
    head_dim = int(d["head_dim"])
    return cls(
        num_layers=num_layers,
        num_qo_heads=int(d["num_attention_heads"]),
        num_kv_heads=int(d.get("num_key_value_heads", 1)),
        head_dim=head_dim,
        hidden_size=int(d["hidden_size"]),
        vocab_size=int(d["vocab_size"]),
        intermediate_size=int(d["moe_intermediate_size"]),
        rms_norm_eps=float(d["rms_norm_eps"]),
        rotary_config=_rope_cfg(head_dim, int(d["qk_rope_head_dim"]), d),
        hidden_act=str(d.get("hidden_act", "silu")),
        tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        num_experts=int(d["n_routed_experts"]),
        num_experts_per_tok=int(d["num_experts_per_tok"]),
        moe_intermediate_size=int(d["moe_intermediate_size"]),
        norm_topk_prob=bool(d.get("norm_topk_prob", True)),
        use_routing_bias=True,
        scoring_func=str(d.get("scoring_func", "sqrtsoftplus")),
        model_type="deepseek_v4",
        architectures=list(d.get("architectures") or ["DeepseekV4ForCausalLM"]),
        quant=quant,
        rope_theta=float(d.get("rope_theta", 10000.0)),
        rope_scaling=d.get("rope_scaling"),
        model_extra={
            "hf_text_config": d,
            "n_shared_experts": int(d.get("n_shared_experts", 1)),
            "routed_scaling_factor": float(d.get("routed_scaling_factor", 1.0)),
            "first_k_dense_replace": 0,
        },
    )


def _kimi_k3_from_hf(cls, config: Any) -> "ModelConfig":
    top = _config_to_dict(config)
    text = top.get("text_config")
    if text is None:
        text = top
    d = _config_to_dict(text)
    archs = list(top.get("architectures") or ["KimiK3ForConditionalGeneration"])
    num_layers = int(d["num_hidden_layers"])
    qk_head = int(d["qk_nope_head_dim"]) + int(d["qk_rope_head_dim"])
    num_heads = int(d["num_attention_heads"])
    return cls(
        num_layers=num_layers,
        num_qo_heads=num_heads,
        num_kv_heads=int(d.get("num_key_value_heads") or num_heads),
        head_dim=qk_head,
        hidden_size=int(d["hidden_size"]),
        vocab_size=int(d["vocab_size"]),
        intermediate_size=int(d["intermediate_size"]),
        rms_norm_eps=float(d["rms_norm_eps"]),
        rotary_config=RotaryConfig(
            head_dim=qk_head,
            rotary_dim=int(d["qk_rope_head_dim"]),
            max_position=int(d.get("max_position_embeddings", 4096)),
            base=float(d.get("rope_theta", 10000.0) or 10000.0),
            scaling=None,
        ),
        hidden_act=str(d.get("hidden_act", "situ")),
        tie_word_embeddings=bool(top.get("tie_word_embeddings", d.get("tie_word_embeddings", False))),
        num_experts=int(d["num_experts"]),
        num_experts_per_tok=int(d["num_experts_per_token"]),
        moe_intermediate_size=int(d["moe_intermediate_size"]),
        norm_topk_prob=bool(d.get("moe_renormalize", True)),
        use_routing_bias=True,
        scoring_func=str(d.get("moe_router_activation_func", "sigmoid")),
        model_type="kimi_k3",
        architectures=archs,
        quant=None,
        rope_theta=float(d.get("rope_theta", 10000.0) or 10000.0),
        model_extra={
            "hf_text_config": d,
            "n_shared_experts": int(d.get("num_shared_experts") or 0),
            "routed_scaling_factor": float(d.get("routed_scaling_factor", 1.0)),
            "first_k_dense_replace": int(d.get("first_k_dense_replace", 0)),
        },
    )
