"""Triton kernels vendored from sglang 0.5.18 (Apache-2.0); see each module header."""

from __future__ import annotations

import functools

import torch


@functools.cache
def is_arch_support_pdl() -> bool:
    """Programmatic dependent launch (Hopper+), as in sglang's JIT utils."""
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:  # pragma: no cover
        return False
