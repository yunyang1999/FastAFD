from __future__ import annotations

import json
import logging
import os

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

HF_METADATA_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "generation_config.json",
    "chat_template.json",
)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def resolve_local_model_dir(model_path: str, *, local_only: bool | None = None) -> str:
    local_only = local_files_only() if local_only is None else bool(local_only)
    if os.path.isdir(model_path) or not local_only:
        return model_path
    try:
        return snapshot_download(
            model_path,
            local_files_only=True,
            allow_patterns=list(HF_METADATA_ALLOW_PATTERNS),
            tqdm_class=DisabledTqdm,
        )
    except Exception:
        logger.debug(
            "Failed to resolve cached local model dir for %s in offline mode; "
            "falling back to the original path.",
            model_path,
            exc_info=True,
        )
        return model_path


def _load_chat_template_from_path(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["chat_template"]


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    # Dummy-weight benchmarks of models whose tokenizer needs extra packages
    # (e.g. Kimi-K3 / tiktoken) may point at another tokenizer directory.
    model_path = os.environ.get("MINISGL_TOKENIZER_PATH") or model_path
    local_only = local_files_only()
    resolved_model_path = resolve_local_model_dir(model_path, local_only=local_only)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        try:
            local_path = os.path.join(resolved_model_path, "chat_template.json")
            if os.path.isfile(local_path):
                tokenizer.chat_template = _load_chat_template_from_path(local_path)
            elif not os.path.isdir(model_path):
                path = hf_hub_download(
                    repo_id=model_path,
                    filename="chat_template.json",
                    local_files_only=local_only,
                )
                tokenizer.chat_template = _load_chat_template_from_path(path)
        except Exception:
            pass
    return tokenizer


__all__ = [
    "DisabledTqdm",
    "HF_METADATA_ALLOW_PATTERNS",
    "load_tokenizer",
    "local_files_only",
    "resolve_local_model_dir",
]
