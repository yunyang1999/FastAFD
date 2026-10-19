import functools
import os
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoConfig, PretrainedConfig

from minisgl.hf_support import DisabledTqdm, load_tokenizer, local_files_only, resolve_local_model_dir


_RAW_CONFIG_MODEL_TYPES = ("deepseek_v4", "kimi_k3", "kimi_linear")


def _load_raw_config_dict(resolved_model_path: str, local_files_only: bool) -> dict:
    import json

    if os.path.isdir(resolved_model_path):
        path = os.path.join(resolved_model_path, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=resolved_model_path,
            filename="config.json",
            local_files_only=local_files_only,
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def _load_hf_config_cached(resolved_model_path: str, local_files_only: bool) -> Any:
    try:
        return AutoConfig.from_pretrained(
            resolved_model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except Exception:
        # Architectures newer than the pinned transformers (deepseek_v4) or
        # shipped as remote code (kimi_k3) are parsed by minisgl itself from
        # the raw config.json; see ModelConfig.from_hf.
        raw = _load_raw_config_dict(resolved_model_path, local_files_only)
        if raw.get("model_type") not in _RAW_CONFIG_MODEL_TYPES:
            raise
        return PretrainedConfig(**raw)


def _load_hf_config(model_path: str) -> Any:
    local_files = local_files_only()
    resolved_model_path = resolve_local_model_dir(model_path, local_only=local_files)
    return _load_hf_config_cached(resolved_model_path, local_files)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
