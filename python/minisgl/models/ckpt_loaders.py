"""Real-checkpoint loaders for DeepSeek-V4 and Kimi-K3 (see load_hybrid_checkpoint)."""

from __future__ import annotations

from typing import Dict

import torch


def load_hybrid_checkpoint(model, model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("real checkpoint loading for DSV4/K3 is implemented in a later commit")
