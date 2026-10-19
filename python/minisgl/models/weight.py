from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info, try_get_ep_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight, init_logger
from tqdm import tqdm

from .fp8_utils import dequant_fp8_block

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".w1", ".w3"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
_FP8_SCALE_SUFFIX = ".weight_scale_inv"
_FP8_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")
_SCALE_RENAME_SUFFIX = "_scale"

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
    ".w1": (".gate_up_proj", ("gate", "up")),
    ".w3": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
    ".w1": "gate",
    ".w3": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_GLM4_LAYER_PATTERN = re.compile(r"^model\.layers\.(?P<idx>\d+)\.")
_RENAME_SUBSTRINGS = {
    ".block_sparse_moe.": ".mlp.",
    ".w2": ".down_proj",
}

logger = init_logger(__name__)


def _config_field(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _is_glm4_moe_config(config: Any) -> bool:
    archs = _config_field(config, "architectures", default=[]) or []
    return "Glm4MoeForCausalLM" in archs or _config_field(
        config, "model_type", default=""
    ) == "glm4_moe"


def _glm4_layer_index_for_key(key: str) -> int | None:
    match = _GLM4_LAYER_PATTERN.match(key.removeprefix("language_model."))
    return int(match.group("idx")) if match is not None else None


def _glm4_is_expected_nextn_key(key: str, config: Any) -> bool:
    layer_idx = _glm4_layer_index_for_key(key)
    if layer_idx is None:
        return False
    num_layers = int(_config_field(config, "num_layers", "num_hidden_layers", default=0) or 0)
    num_nextn = int(_config_field(config, "num_nextn_predict_layers", default=0) or 0)
    return num_nextn > 0 and num_layers <= layer_idx < num_layers + num_nextn


def _scale_suffix_for_name(name: str) -> str | None:
    for suffix in _FP8_SCALE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def _shard_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    *,
    skip_tp_shard: bool = False,
    fp8_block_size: Optional[Tuple[int, int]] = None,
):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if skip_tp_shard:
        return value
    if value.dim() == 0:
        # A true scalar (e.g. a per-tensor FP8 weight_scale/input_scale, as
        # in nvidia_qwen3-235b-a22b-fp8) is one global value, not a
        # per-row/per-column quantity: every rank dequantizes its own weight
        # shard with the SAME scale, so replicate rather than shard.
        # `.chunk()` below raises "chunk expects at least a 1-dimensional
        # tensor" on a 0-d input; `_shard_fp8_scale_tensor` only handles the
        # block-wise scaling scheme (fp8_block_size is not None), so a
        # scalar reaches the generic chunk() path and crashes with it.
        return value
    if fp8_block_size is not None and (
        key.endswith(".weight_scale") or key.endswith(_SCALE_RENAME_SUFFIX)
    ):
        return _shard_fp8_scale_tensor(key, value, r, n, num_kv_heads, fp8_block_size)
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _slice_fp8_scale_dim(
    value: torch.Tensor,
    *,
    dim: int,
    r: int,
    n: int,
    block_size: int,
) -> torch.Tensor:
    padded_weight_dim = value.shape[dim] * int(block_size)
    weight_start = r * padded_weight_dim // n
    weight_end = (r + 1) * padded_weight_dim // n
    scale_start = weight_start // int(block_size)
    scale_end = div_ceil(weight_end, int(block_size))
    index = [slice(None)] * value.dim()
    index[dim] = slice(scale_start, scale_end)
    return value[tuple(index)].clone()


def _shard_fp8_scale_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    block_size: Tuple[int, int],
) -> torch.Tensor:
    block_out, block_in = block_size
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            blocks_per_head = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[
                head_idx * blocks_per_head : (head_idx + 1) * blocks_per_head
            ].clone()
        return _slice_fp8_scale_dim(value, dim=0, r=r, n=n, block_size=block_out)
    if any(key.count(sub) for sub in _SPLIT_DIM_1):
        return _slice_fp8_scale_dim(value, dim=1, r=r, n=n, block_size=block_in)
    return value


def _shard_fp8_channel_scale_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    *,
    skip_tp_shard: bool = False,
) -> torch.Tensor:
    if skip_tp_shard:
        return value
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            rows_per_head = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[
                head_idx * rows_per_head : (head_idx + 1) * rows_per_head
            ].clone()
        return value.chunk(n, dim=0)[r].clone()
    # Per-channel FP8 scales are per output row. Row-parallel weights shard
    # input columns, so their row scales are replicated.
    return value


def _find_paired_scale_key(
    *,
    normalized_weight_name: str,
    raw_weight_name: str,
    shard_keys: set[str],
) -> str | None:
    if not normalized_weight_name.endswith(".weight"):
        return None
    normalized_base = normalized_weight_name.removesuffix(".weight")
    raw_base = raw_weight_name.removesuffix(".weight")
    for suffix in _FP8_SCALE_SUFFIXES:
        raw_candidate = raw_base + suffix
        if raw_candidate in shard_keys:
            return raw_candidate
        normalized_candidate = normalized_base + suffix
        if normalized_candidate in shard_keys:
            return normalized_candidate
    return None


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    # k_scale/v_scale are per-layer KV-cache quantization scales (vLLM-style
    # naming: nvidia_qwen3-235b-a22b-fp8 stores them as
    # `...k_proj.k_scale` / `...v_proj.v_scale`), unrelated to the q/k/v
    # *weight* merge despite the `.k_proj`/`.v_proj` substring match below.
    # Q has no `q_scale` counterpart (queries aren't cached), so routing
    # these through the qkv merge group always leaves it one slot short and
    # `load_weight` asserts "Incomplete merge groups in checkpoint".
    if key.endswith(".k_scale") or key.endswith(".v_scale"):
        return None
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _rename_checkpoint_key(name: str) -> str:
    for old, new in _RENAME_SUBSTRINGS.items():
        if old in name:
            name = name.replace(old, new)
    return name


@dataclass(frozen=True)
class _ExpertPartition:
    start_idx: int
    end_idx: int
    capacity: int


def _get_local_expert_partition(num_experts: int) -> _ExpertPartition | None:
    """Return locally-owned global expert range when full EP is enabled."""
    ep_info = try_get_ep_info()
    if ep_info is None or ep_info.size == 1:
        return None
    local_num_experts = (int(num_experts) + int(ep_info.size) - 1) // int(ep_info.size)
    start = int(ep_info.rank) * local_num_experts
    end = min(int(num_experts), start + local_num_experts)
    return _ExpertPartition(start_idx=start, end_idx=end, capacity=local_num_experts)


def _is_expert_key(name: str) -> bool:
    return _EXPERT_PATTERN.match(name) is not None


def _is_remote_shared_expert_key(name: str, config: Any) -> bool:
    if not _is_glm4_moe_config(config):
        return False
    if int(_config_field(config, "n_shared_experts", default=0) or 0) <= 0:
        return False
    if os.environ.get("MINISGL_AFD_MOE_BACKEND") != "megamoe_m2n":
        return False
    return ".shared_experts." in name


def load_weight(
    model_path: str,
    device: torch.device,
    *,
    skip_expert_weights: bool = False,
    skip_non_expert_weights: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer.

    afd AG/EG role split: skip_expert_weights loads only the dense/attention/
    router/embed/lm_head weights (AG side); skip_non_expert_weights loads only the
    MoE expert weights (EG side)."""
    from .config import ModelConfig

    if skip_expert_weights and skip_non_expert_weights:
        raise ValueError("Cannot skip both expert and non-expert weights")

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()
    local_expert_partition = (
        _get_local_expert_partition(config.num_experts) if config.is_moe else None
    )
    local_num_experts = (
        local_expert_partition.capacity
        if local_expert_partition is not None
        else config.num_experts
    )
    real_local_num_experts = (
        local_expert_partition.end_idx - local_expert_partition.start_idx
        if local_expert_partition is not None
        else local_num_experts
    )
    fp8_block_size: Optional[Tuple[int, int]] = (
        config.quant.weight_block_size
        if config.quant is not None and config.quant.method == "fp8"
        else None
    )
    fp8_channel_scale = config.quant is not None and config.quant.method == "fp8_channel"
    logger.info(
        f"Loading weights from {len(files)} safetensors files"
        + (f" (FP8 block={fp8_block_size})" if fp8_block_size else "")
        + (" (FP8 channel)" if fp8_channel_scale else "")
    )

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            shard_keys = set(f.keys())
            for raw_name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if raw_name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                name = raw_name.removeprefix("language_model.")
                if _is_glm4_moe_config(config) and _glm4_is_expected_nextn_key(name, config):
                    continue

                scale_suffix = _scale_suffix_for_name(name)
                is_scale_tensor = scale_suffix is not None
                paired_scale_name = _find_paired_scale_key(
                    normalized_weight_name=name,
                    raw_weight_name=raw_name,
                    shard_keys=shard_keys,
                )
                has_paired_scale = (
                    paired_scale_name is not None
                    and paired_scale_name in shard_keys
                )

                if is_scale_tensor:
                    assert scale_suffix is not None
                    base_weight_name = name.removesuffix(scale_suffix)
                    if fp8_block_size is None and not fp8_channel_scale:
                        continue
                    name = (
                        base_weight_name + _SCALE_RENAME_SUFFIX
                        if _is_expert_key(base_weight_name)
                        else base_weight_name + ".weight_scale"
                    )
                name = _rename_checkpoint_key(name)

                pre_merge_expert_info = (
                    _get_expert_stack_info(name) if config.is_moe else None
                )
                is_remote_shared_expert = _is_remote_shared_expert_key(name, config)
                # afd AG/EG role split: drop the other role's weights.
                if skip_expert_weights and (
                    pre_merge_expert_info is not None or is_remote_shared_expert
                ):
                    continue
                if skip_non_expert_weights and (
                    pre_merge_expert_info is None and not is_remote_shared_expert
                ):
                    continue
                if local_expert_partition is not None and pre_merge_expert_info is not None:
                    _, expert_idx = pre_merge_expert_info
                    if not (
                        local_expert_partition.start_idx
                        <= expert_idx
                        < local_expert_partition.end_idx
                    ):
                        continue
                raw = f.get_tensor(raw_name)
                if fp8_channel_scale and is_scale_tensor:
                    tensor = _shard_fp8_channel_scale_tensor(
                        name,
                        raw,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=(
                            local_expert_partition is not None
                            and pre_merge_expert_info is not None
                        ),
                    )
                else:
                    tensor = _shard_tensor(
                        name,
                        raw,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=(
                            local_expert_partition is not None
                            and pre_merge_expert_info is not None
                        ),
                        fp8_block_size=fp8_block_size,
                    )
                del raw

                if (
                    fp8_block_size is not None
                    and has_paired_scale
                    and not _is_expert_key(name)
                    and tensor.dtype != torch.float8_e4m3fn
                ):
                    assert paired_scale_name is not None
                    scale_full = f.get_tensor(paired_scale_name)
                    scale_sharded = _shard_tensor(
                        name,
                        scale_full,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=False,
                        fp8_block_size=fp8_block_size,
                    )
                    del scale_full
                    tensor = dequant_fp8_block(tensor, scale_sharded, fp8_block_size)
                    del scale_sharded

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    if parts[0].dim() == 0:
                        # Per-tensor scalar scales (one per fused slot, e.g.
                        # q/k/v each with their own weight_scale) have no row
                        # dimension to concatenate along; stack them into a
                        # length-len(all_slots) vector instead so each
                        # component's scale is preserved. `torch.cat`
                        # requires >=1-d inputs and raises "zero-dimensional
                        # tensor ... cannot be concatenated" here otherwise.
                        out = (merged_key, torch.stack(parts, dim=0))
                    else:
                        out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    if local_expert_partition is not None:
                        if not (
                            local_expert_partition.start_idx
                            <= expert_idx
                            < local_expert_partition.end_idx
                        ):
                            continue
                        expert_idx = expert_idx - local_expert_partition.start_idx
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != real_local_num_experts:
                        continue
                    if local_num_experts == real_local_num_experts:
                        experts = [slots[idx] for idx in range(local_num_experts)]
                    else:
                        template = next(iter(slots.values()))
                        zero = torch.zeros_like(template)
                        experts = [slots.get(idx, zero) for idx in range(local_num_experts)]
                    del expert_buf[packed_key]
                    stacked = torch.stack(experts, dim=0)
                    if packed_key.endswith(_SCALE_RENAME_SUFFIX):
                        stacked = stacked.to(torch.float32)
                    yield packed_key, stacked
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    logger.info("Finished loading weights")
