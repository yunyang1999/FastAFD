"""Opt-in tensor taps for module-level numerics tests (no-op unless enabled)."""

from __future__ import annotations

import torch

TAPS: list | None = None


def enable() -> list:
    global TAPS
    TAPS = []
    return TAPS


def disable() -> None:
    global TAPS
    TAPS = None


def tap(name: str, t: torch.Tensor) -> None:
    if TAPS is not None and not torch.cuda.is_current_stream_capturing():
        TAPS.append((name, t.detach().clone()))
