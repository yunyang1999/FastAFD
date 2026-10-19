from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def hybrid_cache_sizing(
    model_config: ModelConfig,
    *,
    page_size: int,
    num_req_slots: int,
    scratch_rows: int,
) -> tuple[int, int] | None:
    """(bytes per page, fixed bytes) for models with non-MHA caches, else None."""
    if getattr(model_config, "is_deepseek_v4", False):
        from .dsv4_pool import DSV4CacheSpec

        spec = DSV4CacheSpec.from_model_config(model_config)
        per_page = int(spec.bytes_per_token() * page_size)
        fixed = spec.bytes_per_request() * num_req_slots + spec.fixed_bytes(scratch_rows)
        return per_page, fixed
    if getattr(model_config, "is_kimi_k3", False):
        from .kimi_pool import KimiCacheSpec

        spec = KimiCacheSpec.from_model_config(model_config)
        return int(spec.bytes_per_token() * page_size), spec.bytes_per_request() * num_req_slots
    return None


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    num_req_slots: int | None = None,
    scratch_rows: int | None = None,
) -> BaseKVCachePool:
    if getattr(model_config, "is_deepseek_v4", False):
        from .dsv4_pool import DSV4KVPool

        return DSV4KVPool(
            model_config,
            num_pages=num_pages,
            page_size=page_size,
            num_req_slots=int(num_req_slots or 1),
            scratch_rows=int(scratch_rows or 8192),
            dtype=dtype,
            device=device,
        )
    if getattr(model_config, "is_kimi_k3", False):
        from .kimi_pool import KimiKVPool

        return KimiKVPool(
            model_config,
            num_pages=num_pages,
            page_size=page_size,
            num_req_slots=int(num_req_slots or 1),
            device=device,
        )

    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "hybrid_cache_sizing",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
