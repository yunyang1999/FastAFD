import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MiniMaxM2ForCausalLM": (".minimax_m2", "MiniMaxM2ForCausalLM"),
    "Glm4MoeForCausalLM": (".glm4_moe", "Glm4MoeForCausalLM"),
    "DeepseekV4ForCausalLM": (".deepseek_v4", "DeepseekV4ForCausalLM"),
    "KimiK3ForConditionalGeneration": (".kimi_k3", "KimiK3ForCausalLM"),
    "KimiLinearForCausalLM": (".kimi_k3", "KimiK3ForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}

def create_model(model_config: ModelConfig):
    arch = model_config.architectures[0]
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {arch} not supported")
    module_path, class_name = _MODEL_REGISTRY[arch]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["create_model"]
