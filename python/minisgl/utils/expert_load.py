"""Accumulate the REAL expert routing distribution during a serving run.

Why this exists
---------------
The MegaMoE kernel benchmarks run ``ROUTING=balanced``, which makes expert load
exactly uniform by construction (``load_cv_percent == 0``) -- generated
arithmetically, with no router involved. That isolates kernel behaviour and is
deliberately not what production traffic looks like. Real routing is learned and
skewed, and the skew is what makes the F pool wait on its slowest rank.

This records what the real router actually decided, so the skew can be replayed
in the cheap kernel benchmark instead of being inferred from an expensive
end-to-end trace. The distribution is a property of (model weights, input
tokens) only -- the topology just decides which rank owns which expert range --
so one small run characterises every A:F split.

CUDA-graph safety
-----------------
Decode replays a captured graph, so the accumulator must not allocate, must not
resize, and must not synchronise. ``index_add_`` into a preallocated counter
with a preallocated ones-buffer satisfies all three, so it is captured into the
graph and runs on every replay. The only host read happens at dump time,
outside the graph.

Enable with ``MINISGL_DUMP_EXPERT_LOAD=1``; off by default and zero-cost when
off (the accumulate call returns on a module-level flag check).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

_TRUTHY = ("1", "true", "TRUE", "yes", "on")

_ENABLED: bool | None = None
_COUNTS: torch.Tensor | None = None      # [num_layers, num_experts], int32, on GPU
_ONES: torch.Tensor | None = None        # preallocated, grown only outside capture
_META: dict[str, Any] = {}
_STEPS = 0


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("MINISGL_DUMP_EXPERT_LOAD", "") in _TRUTHY
    return _ENABLED


def init(*, num_layers: int, num_experts: int, device, **meta: Any) -> None:
    """Allocate the counter. Safe to call repeatedly; only the first call binds."""
    global _COUNTS, _ONES, _META
    if not enabled() or _COUNTS is not None:
        return
    _COUNTS = torch.zeros((int(num_layers), int(num_experts)), dtype=torch.int32, device=device)
    _ONES = torch.ones(1 << 16, dtype=torch.int32, device=device)
    _META = {"num_layers": int(num_layers), "num_experts": int(num_experts), **meta}


def accumulate(layer: int, topk_ids: torch.Tensor) -> None:
    """Add this layer's routing decisions to the running counts.

    Graph-safe: no sync, no reallocation of anything a captured graph points at.
    Only called from the decode (graph) path, where token counts are fixed graph
    buckets, so every guard below is a backstop rather than a live path.
    """
    if _COUNTS is None or topk_ids is None:
        return
    if not 0 <= int(layer) < _COUNTS.shape[0]:
        return
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    if n > _ONES.numel():
        # Skip, never grow. Growing frees the old buffer, and any decode graph
        # captured earlier has that buffer's ADDRESS baked into it -- the next
        # replay would then write through a pointer the allocator has handed to
        # someone else. Unbounded n only ever arrives on the eager/prefill path,
        # which no longer records at all, so this is a backstop rather than a
        # path anyone takes.
        return
    # index_add_ does NOT bounds-check: an out-of-range index is an illegal
    # memory access, which surfaces later as a generic "unspecified launch
    # failure" in whatever kernel happens to sync next -- i.e. it would look
    # like someone else's bug. MoE routers legitimately emit sentinels (-1 for a
    # dropped token) and MegaMoE passes PROTOCOL ids whose range depends on the
    # shared-expert count, so clamp instead of trusting the range. clamp_ is
    # graph-safe: fixed shape, no allocation, no sync.
    idx = flat.to(torch.int64).clamp_(0, _COUNTS.shape[1] - 1)
    _COUNTS[int(layer)].index_add_(0, idx, _ONES[:n])


def reset() -> str | None:
    """Zero the counters. Call once when warmup ends and real serving begins.

    Graph capture forces this: the accumulate lives INSIDE the captured decode
    graph, which is the only reason it runs on every replay -- and the same
    property means it cannot be skipped for a particular step, because a graph
    replay takes no host-side branch. Warmup therefore routes its dummy capture
    batches through the counter whether or not that is wanted.

    At the run shapes here that is not a rounding error: the capture batches sum
    to roughly as many tokens as a short decode phase does, so leaving them in
    would blend a synthetic distribution into the real one at tens of percent.
    Zeroing between the two is the only place the two are separable.

    Returns what was discarded, for the log -- silently dropping data is how the
    same fix looks when it is wrong.
    """
    global _STEPS
    if _COUNTS is None:
        return None
    dropped = int(_COUNTS.sum().item())
    _COUNTS.zero_()
    _STEPS = 0
    return f"discarded {dropped} warmup assignments"


def mark_step() -> None:
    """Count one decode step, so the dump can report per-step averages."""
    global _STEPS
    if _COUNTS is not None:
        _STEPS += 1


def dump(output_dir: str | Path, tag: str) -> str | None:
    """Write the accumulated distribution. Host sync happens here, never in the loop."""
    if _COUNTS is None:
        return None
    counts = _COUNTS.to("cpu", dtype=torch.int64)
    per_layer = counts.tolist()
    total = counts.sum(dim=0)
    t = total.to(torch.float64)
    n = int(t.numel())
    mean = float(t.mean().item()) if n else 0.0
    mx = float(t.max().item()) if n else 0.0
    payload = {
        "schema": "fastafd.expert-load.v1",
        "note": "REAL routing from a serving run. counts[layer][expert] = "
                "token-to-expert assignments accumulated over the recorded steps.",
        **_META,
        "steps_recorded": _STEPS,
        "counts_per_layer": per_layer,
        "counts_total": total.tolist(),
        "summary": {
            "assignments": int(t.sum().item()),
            "active_experts": int((total > 0).sum().item()),
            "min": int(t.min().item()) if n else 0,
            "mean": mean,
            "max": int(mx),
            "max_over_mean": (mx / mean) if mean else None,
            "load_cv_percent": (float(t.std(unbiased=False).item()) / mean * 100.0)
            if mean else None,
        },
    }
    out = Path(output_dir) / f"expert_load_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    return str(out)
