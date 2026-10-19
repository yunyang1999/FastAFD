#!/usr/bin/env python3
"""Measure split FastAFD MoE backends on one NVLink domain.

The command is intentionally workload-driven. ``sequences-per-ag-rank`` is the
decode concurrency owned by one attention rank. With MTP next-N, each sequence
contributes ``next_n + 1`` physical verify tokens. Those tokens are divided
evenly over the requested microbatch lanes before entering the selected backend.

This is not a full-serving benchmark. It measures the fused A-side
quant/dispatch/wait/combine path together with the F-side expert kernels for all
MoE layers. Router, attention, dense-only layers, residuals, sampling, and
coordinator overhead are outside the boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from minisgl.kernel.deepep_moe import nccl_runtime_metadata
from minisgl.moe.megamoe_model_profiles import (
    MEGAMOE_MODEL_PROFILES,
    MegaMoEModelProfile,
    get_megamoe_model_profile,
)

INPUT_SEED_BASE = 0xAFD80000
SPLIT_SCHEMA = "fastafd.moe-m2n-model-benchmark.v2"
MEGAMOE_BACKEND = "megamoe"
DEEPEP_BACKEND = "deepep_deepgemm"


@dataclass(frozen=True)
class DeepEPFP4LayerWeights:
    l1: tuple[torch.Tensor, torch.Tensor]
    l2: tuple[torch.Tensor, torch.Tensor]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv_percent": 100.0 * stdev / mean if mean else 0.0,
    }


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def environment_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {value!r}")


def source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    diff = command_output(["git", "diff", "--binary"], cwd=root)
    local_dirty = bool(command_output(["git", "status", "--porcelain"], cwd=root))
    return {
        "root": str(root),
        "commit": os.environ.get("MEASUREMENT_SOURCE_COMMIT")
        or command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": os.environ.get("MEASUREMENT_SOURCE_BRANCH")
        or command_output(["git", "branch", "--show-current"], cwd=root),
        "dirty": environment_bool("MEASUREMENT_SOURCE_DIRTY", local_dirty),
        "tracked_diff_sha256": os.environ.get("MEASUREMENT_SOURCE_DIFF_SHA256")
        or hashlib.sha256(diff.encode()).hexdigest(),
        "source_tree_sha256": os.environ.get("MEASUREMENT_SOURCE_TREE_SHA256", ""),
    }


def gpu_provenance(local_rank: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(local_rank)
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count,
        "driver_power_clocks_memory": command_output(
            [
                "nvidia-smi",
                f"--id={local_rank}",
                "--query-gpu=driver_version,power.limit,clocks.max.sm,clocks.max.memory,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_node": os.environ.get("SLURMD_NODENAME", ""),
        "container_image": os.environ.get("MEASUREMENT_CONTAINER_IMAGE", ""),
    }


def _fp8_scales(experts: int, n: int, k: int) -> torch.Tensor:
    return torch.ones(
        (experts, math.ceil(n / 128), math.ceil(k / 128)),
        dtype=torch.float32,
        device="cuda",
    )


def make_fp8_experts(
    profile: MegaMoEModelProfile,
    local_experts: int,
    *,
    layer_id: int,
) -> Any:
    hidden = profile.hidden_size
    intermediate = profile.intermediate_size
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    gate_up = torch.full(
        (local_experts, 2 * intermediate, hidden),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    down = torch.full(
        (local_experts, hidden, intermediate),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return SimpleNamespace(
        gate_up_proj=gate_up,
        gate_up_proj_scale=_fp8_scales(local_experts, 2 * intermediate, hidden),
        down_proj=down,
        down_proj_scale=_fp8_scales(local_experts, hidden, intermediate),
        quant=SimpleNamespace(method="fp8", weight_block_size=(128, 128)),
        _fp8_scale_format="block",
    )


def _shared_fp8_linear(weight: torch.Tensor, scale: torch.Tensor) -> Any:
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        _fp8_scale_format="block",
        _fp8_block_size=(128, 128),
    )


def make_fp8_shared_expert(profile: MegaMoEModelProfile, *, layer_id: int) -> Any:
    hidden = profile.hidden_size
    intermediate = profile.intermediate_size
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    gate_up = torch.full(
        (2 * intermediate, hidden),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    down = torch.full(
        (hidden, intermediate),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return SimpleNamespace(
        gate_up_proj=_shared_fp8_linear(
            gate_up,
            _fp8_scales(1, 2 * intermediate, hidden)[0],
        ),
        down_proj=_shared_fp8_linear(
            down,
            _fp8_scales(1, hidden, intermediate)[0],
        ),
    )


def make_fp4_source_weights(
    profile: MegaMoEModelProfile,
    local_experts: int,
    *,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    l1 = torch.full(
        (local_experts, 2 * profile.intermediate_size, profile.hidden_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2 = torch.full(
        (local_experts, profile.hidden_size, profile.intermediate_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return l1, l2


def protocol_shape(
    profile: MegaMoEModelProfile,
    eg_size: int,
) -> tuple[int, int, int]:
    """Return local routed experts, global protocol experts, and protocol top-k."""
    profile.validate(eg_size=eg_size)
    local_routed = profile.num_routed_experts // eg_size
    local_protocol = local_routed + profile.num_shared_experts
    return local_routed, local_protocol * eg_size, profile.protocol_top_k


def make_deepep_protocol_routes(
    profile: MegaMoEModelProfile,
    tokens: int,
    *,
    rank: int,
    lane: int,
    eg_size: int,
    routing: str,
    hot_expert_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact split-M2N protocol routes consumed by DeepEP.

    Routed experts are remapped from the model's global expert numbering to
    contiguous per-F-rank protocol slots. A declared shared expert is appended
    on the same F rank selected by the fused MegaMoE M2N contract.
    """
    real_ids, weights, _real_counts = make_routes(
        profile,
        tokens,
        rank=rank,
        lane=lane,
        routing=routing,
        hot_expert_fraction=hot_expert_fraction,
    )
    local_routed, protocol_experts, protocol_topk = protocol_shape(profile, eg_size)
    owner = torch.div(real_ids, local_routed, rounding_mode="floor")
    local_id = real_ids.remainder(local_routed)
    protocol_ids = owner * (local_routed + profile.num_shared_experts) + local_id
    if profile.num_shared_experts:
        shared_owner = rank % eg_size
        shared_id = shared_owner * (local_routed + 1) + local_routed
        shared_ids = torch.full(
            (tokens, 1),
            shared_id,
            dtype=torch.int64,
            device="cuda",
        )
        shared_weights = torch.full(
            (tokens, 1),
            1.0 / profile.routed_scaling_factor,
            dtype=torch.float32,
            device="cuda",
        )
        protocol_ids = torch.cat((protocol_ids, shared_ids), dim=1)
        weights = torch.cat((weights, shared_weights), dim=1)
    if tuple(protocol_ids.shape) != (tokens, protocol_topk):
        raise RuntimeError(
            "DeepEP protocol route shape mismatch: "
            f"got={tuple(protocol_ids.shape)} expected={(tokens, protocol_topk)}"
        )
    counts = torch.bincount(protocol_ids.flatten(), minlength=protocol_experts)
    return protocol_ids.contiguous(), weights.contiguous(), counts


def make_deepep_fp4_weights(
    profile: MegaMoEModelProfile,
    *,
    local_protocol_experts: int,
    layer_id: int,
) -> DeepEPFP4LayerWeights:
    from minisgl.kernel.megamoe_m2n_mega import cast_weights_to_fp4

    l1, l2 = make_fp4_source_weights(
        profile,
        local_protocol_experts,
        layer_id=layer_id,
    )
    packed = DeepEPFP4LayerWeights(
        l1=cast_weights_to_fp4(l1),
        l2=cast_weights_to_fp4(l2),
    )
    del l1, l2
    return packed


def make_routes(
    profile: MegaMoEModelProfile,
    tokens: int,
    *,
    rank: int,
    lane: int,
    routing: str,
    hot_expert_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = profile.routed_top_k
    num_experts = profile.num_routed_experts
    active_experts = num_experts
    if routing == "hotset":
        active_experts = max(top_k, int(round(num_experts * hot_expert_fraction)))
    global_tokens = torch.arange(tokens, dtype=torch.int64, device="cuda")
    global_tokens += (rank * 4 + lane) * tokens
    offsets = torch.arange(top_k, dtype=torch.int64, device="cuda")
    if routing == "replay":
        # Reproduce a RECORDED real-router distribution rather than a synthetic
        # one. Deterministic per (rank, lane) so every process and every repeat
        # of a configuration draws the same assignments -- a benchmark that
        # resampled would add run-to-run variance on top of the effect being
        # measured. top_k experts per token are drawn WITHOUT replacement, which
        # is what a real top-k router does; drawing with replacement would let a
        # token land on one expert twice and understate the spread.
        if _REPLAY_WEIGHTS is None:
            raise SystemExit("routing=replay requires --replay-distribution")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(0x5EED0000 + rank * 4096 + lane)
        probs = _REPLAY_WEIGHTS.to(device=offsets.device, dtype=torch.float32).expand(
            tokens, -1
        )
        ids = torch.multinomial(probs, top_k, replacement=False, generator=gen)
    else:
        ids = (global_tokens[:, None] * top_k + offsets[None, :]) % active_experts
    raw_weights = torch.arange(top_k, 0, -1, dtype=torch.float32, device="cuda")
    weights = raw_weights.expand(tokens, -1).clone()
    if profile.gate_renormalize:
        weights /= weights.sum(dim=1, keepdim=True)
    counts = torch.bincount(ids.flatten(), minlength=num_experts)
    return ids.contiguous(), weights.contiguous(), counts


_REPLAY_WEIGHTS: torch.Tensor | None = None


def load_replay_distribution(path: str, num_experts: int) -> torch.Tensor:
    """Per-expert selection weights from a recorded serving run.

    Takes ``counts_total`` out of an expert_load JSON (written by
    ``minisgl.utils.expert_load`` during a real run) and normalises it to a
    probability over experts. The absolute token count in that run is
    irrelevant here -- only the SHAPE of the skew transfers, because this
    benchmark decides its own token count from the topology.
    """
    global _REPLAY_WEIGHTS
    with open(path) as fh:
        payload = json.load(fh)
    counts = payload.get("counts_total")
    if not counts:
        raise SystemExit(f"{path}: no counts_total")
    if len(counts) != num_experts:
        raise SystemExit(
            f"{path}: recorded {len(counts)} experts, profile has {num_experts}"
        )
    w = torch.tensor(counts, dtype=torch.float64)
    if float(w.sum()) <= 0:
        raise SystemExit(f"{path}: counts_total sums to zero")
    # Kept on CPU deliberately. This loader runs during argument handling,
    # BEFORE torch.cuda.set_device(local_rank), so ".to('cuda')" here would
    # resolve to cuda:0 for every rank -- and the sampler would then hand a
    # cuda:0 tensor to a rank living on cuda:3. The device is chosen at sample
    # time instead, where the correct one is current.
    _REPLAY_WEIGHTS = (w / w.sum()).cpu()
    return _REPLAY_WEIGHTS


def route_summary(counts: torch.Tensor) -> dict[str, float | int]:
    counts_f = counts.float()
    mean = float(counts_f.mean().item())
    std = float(counts_f.std(unbiased=False).item())
    sorted_counts = counts.sort().values
    p95_index = min(len(sorted_counts) - 1, math.ceil(0.95 * len(sorted_counts)) - 1)
    return {
        "assignments": int(counts.sum().item()),
        "active_experts": int((counts > 0).sum().item()),
        "min_assignments": int(counts.min().item()),
        "mean_assignments": mean,
        "p95_assignments": int(sorted_counts[p95_index].item()),
        "max_assignments": int(counts.max().item()),
        "load_cv_percent": 100.0 * std / mean if mean else 0.0,
    }


def register_megamoe_layers(
    adapter: Any,
    profile: MegaMoEModelProfile,
    *,
    layers: int,
) -> None:
    local_experts = profile.num_routed_experts // adapter.eg_size
    for layer_id in range(layers):
        if profile.weight_precision == "fp8":
            experts = make_fp8_experts(profile, local_experts, layer_id=layer_id)
            shared = (
                make_fp8_shared_expert(profile, layer_id=layer_id)
                if profile.num_shared_experts
                else None
            )
            adapter.register_eg_layer_weights(layer_id, experts, shared_experts=shared)
            del experts, shared
        else:
            l1, l2 = make_fp4_source_weights(profile, local_experts, layer_id=layer_id)
            shared_l1 = shared_l2 = None
            if profile.num_shared_experts:
                shared_l1, shared_l2 = make_fp4_source_weights(profile, 1, layer_id=layer_id)
            adapter.register_eg_layer_fp4_weights(
                layer_id,
                l1,
                l2,
                shared_l1_weight=shared_l1,
                shared_l2_weight=shared_l2,
            )
            del l1, l2, shared_l1, shared_l2
        if layer_id % 4 == 3:
            torch.cuda.empty_cache()


def run_megamoe_stage(
    *,
    adapter: Any,
    is_ag: bool,
    rank: int,
    layers: int,
    expected_tokens_per_lane: int,
    lane_streams: tuple[torch.cuda.Stream, ...],
    inputs: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> tuple[float, float]:
    del rank
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lane_done = [torch.cuda.Event() for _ in lane_streams]
    start.record(current)
    wall_start = time.perf_counter()
    if is_ag:
        for stream in lane_streams:
            stream.wait_event(start)
        for _layer in range(layers):
            for lane, stream in enumerate(lane_streams):
                hidden, ids, weights = inputs[lane]
                with torch.cuda.stream(stream):
                    output = adapter.ag_moe(
                        hidden,
                        ids,
                        weights,
                        lane=lane,
                        external_quant=False,
                    )
                    if adapter.routed_scaling_factor != 1.0:
                        output.mul_(adapter.routed_scaling_factor)
        for done, stream in zip(lane_done, lane_streams, strict=True):
            done.record(stream)
            current.wait_event(done)
    else:
        adapter.eg_moe_persistent(
            layers,
            num_mb=len(lane_streams),
            expected_tokens_per_rank=expected_tokens_per_lane,
        )
    end.record(current)
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def run_deepep_fp4_experts(
    *,
    profile: MegaMoEModelProfile,
    dispatch_output: Any,
    weights: DeepEPFP4LayerWeights,
    expert_alignment: int,
) -> torch.Tensor:
    """Run the production psum-layout FP8xFP4 DeepGEMM expert boundary."""
    from minisgl.kernel import deepgemm
    from minisgl.kernel.deepgemm_fused_quant import persistent_psum_silu_mul_quant

    hidden = dispatch_output.hidden_states
    hidden_scale = dispatch_output.hidden_states_scale
    psum = dispatch_output.recv_count
    if hidden.dtype != torch.float8_e4m3fn or hidden_scale is None:
        raise RuntimeError("split DeepEP FP4 requires FP8 activations with packed scales")
    if hidden.ndim != 2 or psum.ndim != 1:
        raise RuntimeError(
            f"invalid DeepEP psum layout: hidden={tuple(hidden.shape)} psum={tuple(psum.shape)}"
        )
    rows = int(hidden.shape[0])
    experts = int(psum.shape[0])
    if int(weights.l1[0].shape[0]) != experts or int(weights.l2[0].shape[0]) != experts:
        raise RuntimeError(
            "DeepEP weight/dispatch expert mismatch: "
            f"weights={int(weights.l1[0].shape[0])} dispatch={experts}"
        )
    expected_m = max(1, rows // max(1, experts))
    l1_output = torch.empty(
        (rows, 2 * profile.intermediate_size),
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    deepgemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (hidden.contiguous(), hidden_scale),
        weights.l1,
        l1_output,
        psum,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )
    l2_input, l2_input_scale = persistent_psum_silu_mul_quant(
        l1_output,
        psum,
        alignment=expert_alignment,
        topk_weights=dispatch_output.topk_weights,
        group_size=32,
        activation_clamp=profile.activation_clamp,
        activation_alpha=profile.activation_alpha,
        activation_up_bias=profile.activation_up_bias,
    )
    output = torch.empty(
        (rows, profile.hidden_size),
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    deepgemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (l2_input, l2_input_scale),
        weights.l2,
        output,
        psum,
        recipe=(1, 1, 32),
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )
    return output


def run_deepep_stage(
    *,
    profile: MegaMoEModelProfile,
    adapters: tuple[Any, ...],
    is_ag: bool,
    layers: int,
    physical_tokens_per_lane: int,
    lane_streams: tuple[torch.cuda.Stream, ...],
    inputs: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    weights: tuple[DeepEPFP4LayerWeights, ...],
    expert_alignment: int,
    output_cache: dict[int, torch.Tensor],
    ag_zero_cache: dict[tuple[int, int], torch.Tensor],
) -> tuple[float, float]:
    """Run the same per-layer/lane collective order as FastAFD serving."""
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lane_done = [torch.cuda.Event() for _ in lane_streams]
    start.record(current)
    wall_start = time.perf_counter()
    for stream in lane_streams:
        stream.wait_event(start)
    if is_ag:
        from minisgl.kernel.fp8_quant import per_token_cast_to_fp8

        dispatches: list[list[Any | None]] = [[None for _ in adapters] for _layer in range(layers)]
        retired: list[tuple[torch.cuda.Event, Any, torch.Tensor]] = []

        def launch_dispatch(layer: int, lane: int) -> None:
            hidden, ids, topk_weights = inputs[lane]
            adapter = adapters[lane]
            stream = lane_streams[lane]
            with torch.cuda.stream(stream):
                hidden_fp8, hidden_scale = per_token_cast_to_fp8(
                    hidden.contiguous(),
                    use_ue8m0=True,
                    gran_k=128,
                    use_packed_ue8m0=True,
                    backend="cuda",
                )
                dispatches[layer][lane] = adapter.dispatch(
                    hidden_fp8,
                    ids,
                    topk_weights,
                    hidden_states_scale=hidden_scale,
                    expert_alignment=expert_alignment,
                    num_max_dispatch_tokens_per_rank=physical_tokens_per_lane,
                )

        for lane in range(len(adapters)):
            launch_dispatch(0, lane)
        for layer in range(layers):
            for lane, stream in enumerate(lane_streams):
                dispatch = dispatches[layer][lane]
                if dispatch is None:
                    raise RuntimeError(f"missing DeepEP dispatch for layer={layer} lane={lane}")
                hidden = inputs[lane][0]
                adapter = adapters[lane]
                with torch.cuda.stream(stream):
                    shape = (int(dispatch.hidden_states.shape[0]), profile.hidden_size)
                    combine_input = ag_zero_cache.get(shape)
                    if combine_input is None:
                        combine_input = torch.zeros(
                            shape,
                            dtype=torch.bfloat16,
                            device=hidden.device,
                        )
                        ag_zero_cache[shape] = combine_input
                    output = adapter.combine(combine_input, dispatch)
                    if profile.routed_scaling_factor != 1.0:
                        output.mul_(profile.routed_scaling_factor)
                    output_cache[lane] = output
                    done = torch.cuda.Event()
                    done.record(stream)
                retired.append((done, dispatch, output))
                if len(retired) >= 4:
                    retired.pop(0)[0].synchronize()
                dispatches[layer][lane] = None
                if layer + 1 < layers:
                    launch_dispatch(layer + 1, lane)
        while retired:
            retired.pop(0)[0].synchronize()
        for done, stream in zip(lane_done, lane_streams, strict=True):
            done.record(stream)
            current.wait_event(done)
    else:
        empty_hidden = torch.empty(
            (0, profile.hidden_size),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        empty_scale = torch.empty(
            (0, profile.hidden_size // 128 // 4),
            dtype=torch.int32,
            device="cuda",
        )
        dispatch_done = [[torch.cuda.Event() for _ in adapters] for _layer in range(layers)]
        expert_done = [[torch.cuda.Event() for _ in adapters] for _layer in range(layers)]
        dispatches: list[list[Any | None]] = [[None for _ in adapters] for _layer in range(layers)]
        expert_outputs: list[list[torch.Tensor | None]] = [
            [None for _ in adapters] for _layer in range(layers)
        ]
        retired: list[tuple[torch.cuda.Event, Any, torch.Tensor]] = []

        def retire_one() -> None:
            event, _dispatch, _output = retired.pop(0)
            event.synchronize()

        def launch_dispatch(layer: int, lane: int) -> None:
            adapter = adapters[lane]
            stream = lane_streams[lane]
            with torch.cuda.stream(stream):
                dispatch = adapter.dispatch(
                    empty_hidden,
                    None,
                    None,
                    hidden_states_scale=empty_scale,
                    expert_alignment=expert_alignment,
                    num_max_dispatch_tokens_per_rank=physical_tokens_per_lane,
                )
                dispatch_done[layer][lane].record(stream)
            dispatches[layer][lane] = dispatch

        for lane in range(len(adapters)):
            launch_dispatch(0, lane)
        for layer in range(layers):
            layer_weights = weights[layer % len(weights)]
            for lane, stream in enumerate(lane_streams):
                dispatch = dispatches[layer][lane]
                if dispatch is None:
                    raise RuntimeError(f"missing DeepEP dispatch for layer={layer} lane={lane}")
                with torch.cuda.stream(current):
                    current.wait_event(dispatch_done[layer][lane])
                    output = run_deepep_fp4_experts(
                        profile=profile,
                        dispatch_output=dispatch,
                        weights=layer_weights,
                        expert_alignment=expert_alignment,
                    )
                    expert_done[layer][lane].record(current)
                expert_outputs[layer][lane] = output
                with torch.cuda.stream(stream):
                    stream.wait_event(expert_done[layer][lane])
                    output.record_stream(stream)
                    adapters[lane].combine(output, dispatch)
                    done = torch.cuda.Event()
                    done.record(stream)
                retired.append((done, dispatch, output))
                if len(retired) >= 4:
                    retire_one()
                dispatches[layer][lane] = None
                expert_outputs[layer][lane] = None
                if layer + 1 < layers:
                    launch_dispatch(layer + 1, lane)
        while retired:
            retire_one()
        for done, stream in zip(lane_done, lane_streams, strict=True):
            done.record(stream)
            current.wait_event(done)
    end.record(current)
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MEGAMOE_MODEL_PROFILES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ag-size", type=int, required=True)
    parser.add_argument("--eg-size", type=int, required=True)
    parser.add_argument("--sequences-per-ag-rank", type=int, required=True)
    parser.add_argument("--mtp-nextn", type=int, default=0)
    parser.add_argument("--microbatches", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--layers", type=int, default=0, help="0 means all model MoE layers")
    parser.add_argument(
        "--routing", choices=("balanced", "hotset", "replay"), default="balanced"
    )
    parser.add_argument(
        "--replay-distribution",
        default=None,
        help="expert_load JSON from a real serving run; required for --routing replay",
    )
    parser.add_argument("--hot-expert-fraction", type=float, default=0.25)
    parser.add_argument(
        "--backend",
        choices=("mega", "deepep"),
        default="mega",
        help="Measure split MegaMoE or split DeepEP+DeepGEMM",
    )
    parser.add_argument(
        "--weight-slots",
        type=int,
        default=0,
        help="DeepEP FP4 weight sets to allocate; 0 means one set per measured layer",
    )
    parser.add_argument("--expected-tokens-per-lane", type=int)
    parser.add_argument("--prefetch-mib", type=int, default=0)
    parser.add_argument("--ag-sms", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    profile = get_megamoe_model_profile(args.model)
    if args.routing == "replay":
        load_replay_distribution(args.replay_distribution, profile.num_routed_experts)
    profile.validate(eg_size=args.eg_size)
    layers = profile.num_moe_layers if args.layers == 0 else args.layers
    if not 1 <= layers <= profile.num_moe_layers:
        raise SystemExit(f"layers must be in [1, {profile.num_moe_layers}]")
    if args.sequences_per_ag_rank <= 0:
        raise SystemExit("sequences-per-ag-rank must be positive")
    if args.sequences_per_ag_rank % args.microbatches:
        raise SystemExit("sequences-per-ag-rank must divide evenly across microbatches")
    if args.mtp_nextn < 0:
        raise SystemExit("mtp-nextn must be non-negative")
    if args.routing == "replay" and not args.replay_distribution:
        raise SystemExit("--routing replay requires --replay-distribution")
    if args.replay_distribution and args.routing != "replay":
        raise SystemExit("--replay-distribution is only meaningful with --routing replay")
    if not 0 < args.hot_expert_fraction <= 1:
        raise SystemExit("hot-expert-fraction must be in (0, 1]")
    if args.warmups < 1 or args.iterations < 3:
        raise SystemExit("warmups must be >=1 and iterations must be >=3")
    if args.weight_slots < 0:
        raise SystemExit("weight-slots must be non-negative")
    if args.backend == "deepep" and profile.weight_precision != "fp4":
        raise SystemExit("split DeepEP+DeepGEMM measurement currently requires an FP4 profile")

    verify_width = args.mtp_nextn + 1
    sequences_per_lane = args.sequences_per_ag_rank // args.microbatches
    physical_tokens_per_lane = sequences_per_lane * verify_width
    expected_tokens = args.expected_tokens_per_lane or physical_tokens_per_lane
    if expected_tokens <= 0:
        raise SystemExit("expected-tokens-per-lane must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.ag_size + args.eg_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size}, expected ag_size+eg_size={args.ag_size + args.eg_size}"
        )
    os.environ["MINISGL_MEGAMOE_AG_SMS"] = str(args.ag_sms)
    if args.backend == "deepep" and args.microbatches > 1:
        os.environ["MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM"] = "1"
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    is_ag = rank < args.ag_size
    backend = MEGAMOE_BACKEND if args.backend == "mega" else DEEPEP_BACKEND
    megamoe_adapter: Any | None = None
    adapters: tuple[Any, ...]
    expert_alignment = 128
    if backend == MEGAMOE_BACKEND:
        from minisgl.moe.megamoe_m2n_afd import MegaMoEM2NAfdAdapter

        megamoe_adapter = MegaMoEM2NAfdAdapter(
            group=dist.group.WORLD,
            ag_size=args.ag_size,
            eg_size=args.eg_size,
            real_num_experts=profile.num_routed_experts,
            hidden_size=profile.hidden_size,
            intermediate_size=profile.intermediate_size,
            top_k=profile.routed_top_k,
            num_max_dispatch_tokens_per_rank=physical_tokens_per_lane,
            num_lanes=args.microbatches,
            gate_renormalize=profile.gate_renormalize,
            shared_experts_per_rank=profile.num_shared_experts,
            routed_scaling_factor=profile.routed_scaling_factor,
            weight_precision=profile.weight_precision,
            activation=profile.activation,
            activation_alpha=profile.activation_alpha,
            activation_clamp=profile.activation_clamp,
            activation_up_bias=profile.activation_up_bias,
            eg_expected_tokens_override=expected_tokens,
            eg_prefetch_bytes=args.prefetch_mib << 20,
        )
        adapters = (megamoe_adapter,)
    else:
        from minisgl.kernel import deepgemm
        from minisgl.moe.deepep_m2n_adapter import DeepEPM2NAdapter

        _local_routed, protocol_experts, protocol_topk = protocol_shape(
            profile,
            args.eg_size,
        )
        deepgemm.set_mk_alignment_for_contiguous_layout(expert_alignment)
        adapters = tuple(
            DeepEPM2NAdapter(
                group=dist.group.WORLD,
                ag_size=args.ag_size,
                eg_size=args.eg_size,
                real_num_experts=protocol_experts,
                hidden_size=profile.hidden_size,
                top_k=protocol_topk,
                num_max_dispatch_tokens_per_rank=physical_tokens_per_lane,
            )
            for _lane in range(args.microbatches)
        )

    torch.cuda.reset_peak_memory_stats()
    init_start = time.perf_counter()
    deepep_weights: tuple[DeepEPFP4LayerWeights, ...] = ()
    deepep_weight_slots = layers if args.weight_slots == 0 else min(layers, args.weight_slots)
    if not is_ag and backend == MEGAMOE_BACKEND:
        assert megamoe_adapter is not None
        register_megamoe_layers(megamoe_adapter, profile, layers=layers)
    elif not is_ag:
        deepep_weights = tuple(
            make_deepep_fp4_weights(
                profile,
                local_protocol_experts=profile.local_experts(args.eg_size),
                layer_id=layer_id,
            )
            for layer_id in range(deepep_weight_slots)
        )
    dist.barrier()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - init_start
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    lane_streams = tuple(torch.cuda.Stream() for _ in range(args.microbatches))
    route_experts = (
        profile.num_routed_experts
        if backend == MEGAMOE_BACKEND
        else profile.protocol_num_experts(args.eg_size)
    )
    local_route_counts = torch.zeros(
        route_experts,
        dtype=torch.int64,
        device="cuda",
    )
    inputs: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()
    if is_ag:
        materialized = []
        for lane in range(args.microbatches):
            if backend == MEGAMOE_BACKEND:
                ids, weights, counts = make_routes(
                    profile,
                    physical_tokens_per_lane,
                    rank=rank,
                    lane=lane,
                    routing=args.routing,
                    hot_expert_fraction=args.hot_expert_fraction,
                )
            else:
                ids, weights, counts = make_deepep_protocol_routes(
                    profile,
                    physical_tokens_per_lane,
                    rank=rank,
                    lane=lane,
                    eg_size=args.eg_size,
                    routing=args.routing,
                    hot_expert_fraction=args.hot_expert_fraction,
                )
            local_route_counts += counts
            input_seed = INPUT_SEED_BASE + rank * 4 + lane
            input_generator = torch.Generator(device="cuda")
            input_generator.manual_seed(input_seed)
            hidden = (
                torch.randn(
                    (physical_tokens_per_lane, profile.hidden_size),
                    dtype=torch.float32,
                    device="cuda",
                    generator=input_generator,
                )
                .mul_(0.375)
                .to(torch.bfloat16)
            )
            materialized.append((hidden, ids, weights))
        inputs = tuple(materialized)
    dist.all_reduce(local_route_counts, op=dist.ReduceOp.SUM)

    output_cache: dict[int, torch.Tensor] = {}
    ag_zero_cache: dict[tuple[int, int], torch.Tensor] = {}

    def run_once() -> tuple[float, float]:
        if backend == MEGAMOE_BACKEND:
            assert megamoe_adapter is not None
            return run_megamoe_stage(
                adapter=megamoe_adapter,
                is_ag=is_ag,
                rank=rank,
                layers=layers,
                expected_tokens_per_lane=expected_tokens,
                lane_streams=lane_streams,
                inputs=inputs,
            )
        return run_deepep_stage(
            profile=profile,
            adapters=adapters,
            is_ag=is_ag,
            layers=layers,
            physical_tokens_per_lane=physical_tokens_per_lane,
            lane_streams=lane_streams,
            inputs=inputs,
            weights=deepep_weights,
            expert_alignment=expert_alignment,
            output_cache=output_cache,
            ag_zero_cache=ag_zero_cache,
        )

    for _ in range(args.warmups):
        dist.barrier()
        run_once()
    dist.barrier()

    local_cuda: list[float] = []
    local_wall: list[float] = []
    gathered_iterations: list[list[dict[str, float]]] = []
    for _ in range(args.iterations):
        dist.barrier()
        cuda_ms, wall_ms = run_once()
        local_cuda.append(cuda_ms)
        local_wall.append(wall_ms)
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            {"cuda_ms": cuda_ms, "wall_ms": wall_ms},
            object_gather_list=gathered,
            dst=0,
        )
        if rank == 0:
            assert gathered is not None
            gathered_iterations.append(gathered)

    finite = True
    output_abs_mean = None
    if is_ag:
        outputs = (
            tuple(megamoe_adapter._y_cache.values())
            if backend == MEGAMOE_BACKEND
            else tuple(output_cache.values())
        )
        finite = all(bool(torch.isfinite(output).all().item()) for output in outputs)
        output_abs_mean = statistics.mean(
            float(output.float().abs().mean().item()) for output in outputs
        )
    finite_all = [None] * world_size if rank == 0 else None
    output_means = [None] * world_size if rank == 0 else None
    dist.gather_object(finite, object_gather_list=finite_all, dst=0)
    dist.gather_object(output_abs_mean, object_gather_list=output_means, dst=0)

    init_times = [None] * world_size if rank == 0 else None
    peak_memories = [None] * world_size if rank == 0 else None
    provenance = [None] * world_size if rank == 0 else None
    dist.gather_object(initialization_seconds, object_gather_list=init_times, dst=0)
    dist.gather_object(peak_memory_bytes, object_gather_list=peak_memories, dst=0)
    dist.gather_object(gpu_provenance(local_rank), object_gather_list=provenance, dst=0)

    if rank == 0:
        stage_cuda = [max(sample["cuda_ms"] for sample in rows) for rows in gathered_iterations]
        stage_wall = [max(sample["wall_ms"] for sample in rows) for rows in gathered_iterations]
        stage_summary = summarize(stage_cuda)
        if backend == MEGAMOE_BACKEND:
            measurement_boundary = (
                "split AFD MoE stage: A-side fused block-32 FP8 quant, dispatch, wait, "
                "top-k combine and routed-output scaling; F-side persistent fused "
                "FP8xFP4 expert L1, model activation, requant, L2 and return; includes "
                "the fused replicated shared expert when declared; excludes router, "
                "attention, dense-only layers, residual, sampling and coordinator"
            )
            assert megamoe_adapter is not None
            kernel_tuning = {
                "expected_tokens_per_lane": expected_tokens,
                "expected_tokens_source": megamoe_adapter.eg_expected_tokens_source,
                "prefetch_mib": args.prefetch_mib,
                "ag_sms": args.ag_sms,
            }
            backend_implementation = "FastAFD MegaMoE M2N persistent split kernel"
        else:
            torch_nccl_version = torch.cuda.nccl.version()
            measurement_boundary = (
                "split AFD MoE stage: A-side block-128 FP8 quant, DeepEP union dispatch, "
                "zero-source combine and routed-output scaling; F-side psum-layout "
                "FP8xFP4 DeepGEMM L1, model activation with top-k weighting, requant, "
                "L2 and DeepEP return; includes one protocol shared expert per F rank "
                "when declared; excludes router, attention, dense-only layers, residual, "
                "sampling and coordinator"
            )
            kernel_tuning = {
                "expert_alignment": expert_alignment,
                "input_quantization": "E4M3 with packed UE8M0 block-128 scales",
                "weight_quantization": "E2M1 with UE8M0 block-32 scales",
                "weight_slots": deepep_weight_slots,
                "torch_nccl_build_version": (
                    ".".join(map(str, torch_nccl_version))
                    if isinstance(torch_nccl_version, tuple)
                    else str(torch_nccl_version)
                ),
                "deepep_nccl": nccl_runtime_metadata(),
                "gin_disabled": os.environ.get("EP_DISABLE_GIN", "0") == "1",
                "per_buffer_comm_stream": os.environ.get(
                    "MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM",
                    "0",
                ),
            }
            backend_implementation = "FastAFD DeepEP M2N plus psum-layout DeepGEMM"
        stable = bool(stage_summary["cv_percent"] <= 3.0 and all(finite_all))
        full_resident_weights = bool(
            backend == MEGAMOE_BACKEND or deepep_weight_slots == layers
        )
        payload = {
            "schema": SPLIT_SCHEMA,
            "generated_at": utc_now(),
            "system_label": os.environ.get("MEASUREMENT_SYSTEM", "unspecified"),
            "moe_backend": backend,
            "backend_implementation": backend_implementation,
            "measurement_boundary": measurement_boundary,
            "model_profile": asdict(profile),
            "topology": {
                "world_size": world_size,
                "ag_size": args.ag_size,
                "eg_size": args.eg_size,
                "local_routed_experts_per_f_rank": profile.num_routed_experts // args.eg_size,
                "protocol_experts_per_f_rank": profile.local_experts(args.eg_size),
            },
            "workload": {
                "sequences_per_ag_rank": args.sequences_per_ag_rank,
                "mtp_nextn": args.mtp_nextn,
                "verify_width_q": verify_width,
                "microbatches": args.microbatches,
                "sequences_per_lane": sequences_per_lane,
                "physical_tokens_per_lane": physical_tokens_per_lane,
                "physical_tokens_per_step": (
                    args.ag_size * args.sequences_per_ag_rank * verify_width
                ),
                "physical_routed_assignments_per_f_rank_per_lane": (
                    args.ag_size * physical_tokens_per_lane * profile.routed_top_k / args.eg_size
                ),
                "accepted_tokens": None,
                "accepted_tokens_note": (
                    "acceptance is intentionally not applied to kernel latency; it converts "
                    "candidate output to useful tokens at the system-model layer"
                ),
                "layers": layers,
                "routing": args.routing,
                "hot_expert_fraction": args.hot_expert_fraction,
                "replay_distribution": args.replay_distribution,
                "routing_load": route_summary(local_route_counts.cpu()),
                "routing_identity": (
                    "model routed experts; MegaMoE injects shared route in the AG kernel"
                    if backend == MEGAMOE_BACKEND
                    else "per-F-rank protocol experts including the explicit shared route"
                ),
                "input_policy": (
                    "fixed deterministic representative hidden state per A-rank and lane; "
                    "the same input is reused independently for every measured MoE layer"
                ),
                "input_seed_base": INPUT_SEED_BASE,
                "input_seed_formula": "input_seed_base + ag_rank * 4 + lane",
            },
            "kernel_tuning": kernel_tuning,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "stage_cuda": stage_summary,
            "stage_wall": summarize(stage_wall),
            "stage_cuda_samples_ms": stage_cuda,
            "stage_wall_samples_ms": stage_wall,
            "rank_cuda_samples_ms": [
                [sample["cuda_ms"] for sample in rows] for rows in gathered_iterations
            ],
            "rank_wall_samples_ms": [
                [sample["wall_ms"] for sample in rows] for rows in gathered_iterations
            ],
            "all_outputs_finite_by_rank": finite_all,
            "ag_output_abs_mean_by_rank": output_means,
            "stable": stable,
            "stability_contract": "stage CUDA CV <= 3% and finite output on every rank",
            "eligible_for_profile": bool(stable and full_resident_weights),
            "qualification_contract": (
                "stable, finite, and one resident weight set per measured layer; "
                "a reduced DeepEP weight-slot run is diagnostic only"
            ),
            "initialization_seconds_by_rank": init_times,
            "peak_cuda_memory_bytes_by_rank": peak_memories,
            "source": source_provenance(),
            "provenance_by_rank": provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "stage_cuda": stage_summary}, indent=2))

    dist.barrier()
    for adapter in dict.fromkeys(adapters):
        adapter.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
