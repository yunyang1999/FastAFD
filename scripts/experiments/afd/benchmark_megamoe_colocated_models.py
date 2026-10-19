#!/usr/bin/env python3
"""Measure colocated MegaMoE for model-exact FP4 MoE profiles on B200.

Every rank owns both source tokens and an EP shard of the experts.  The timed
boundary is per-token FP8 quantization and symmetric-buffer staging, fused
MegaMoE dispatch/FP8xFP4 compute/combine, and routed-output scaling over all
MoE layers.  Router logits/top-k selection and non-MoE blocks are excluded.
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
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from minisgl.kernel import deepgemm
from minisgl.kernel.deepgemm_fused_quant import persistent_psum_silu_mul_quant
from minisgl.kernel.fp8_quant import (
    per_token_cast_to_fp8_cuda,
    per_token_cast_to_fp8_torch,
    psum_silu_mul_fp8_quant_cuda,
)
from minisgl.kernel.megamoe_m2n_mega import cast_weights_to_fp4, transform_weights_for_mega_moe
from minisgl.kernel.megamoe_mega import MegaMoESymmBuffer, fp8_fp4_mega_moe
from minisgl.moe.megamoe_model_profiles import (
    MEGAMOE_MODEL_PROFILES,
    MegaMoEModelProfile,
    get_megamoe_model_profile,
)
from official_deepep_reference import OfficialDeepEPReference

FP4_PROFILES = {
    key: profile
    for key, profile in MEGAMOE_MODEL_PROFILES.items()
    if profile.weight_precision == "fp4"
}
INPUT_SEED_BASE = 0xAFD40000
_RUNTIME_CLEANUPS: list[Callable[[], None]] = []


@dataclass(frozen=True)
class LayerWeights:
    """One quantized weight set shared by MegaMoE and the reference path."""

    baseline_l1: tuple[torch.Tensor, torch.Tensor]
    baseline_l2: tuple[torch.Tensor, torch.Tensor]
    mega_l1: tuple[torch.Tensor, torch.Tensor]
    mega_l2: tuple[torch.Tensor, torch.Tensor]


def cleanup_runtime_resources() -> None:
    while _RUNTIME_CLEANUPS:
        cleanup = _RUNTIME_CLEANUPS.pop()
        try:
            cleanup()
        except Exception as exc:
            print(f"runtime cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)


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


def measurement_plan(
    backends: tuple[str, ...], warmups: int, iterations: int, order: str
) -> list[tuple[str, bool]]:
    """Return `(backend, record)` steps for reproducible backend timing."""

    if order == "grouped":
        return [
            (backend, record)
            for backend in backends
            for record, repeats in ((False, warmups), (True, iterations))
            for _ in range(repeats)
        ]
    plan = []
    for record, repeats in ((False, warmups), (True, iterations)):
        for index in range(repeats):
            execution_order = backends if index % 2 == 0 else tuple(reversed(backends))
            plan.extend((backend, record) for backend in execution_order)
    return plan


def tensor_pair_nbytes(value: tuple[torch.Tensor, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in value)


def bf16_ordered_codes(values: torch.Tensor) -> torch.Tensor:
    """Map finite BF16 values to monotonic integer codes for ULP comparison."""
    if values.dtype != torch.bfloat16:
        raise ValueError(f"expected torch.bfloat16, got {values.dtype}")
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)


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


_REPLAY_WEIGHTS: torch.Tensor | None = None


def load_replay_distribution(path: str, num_experts: int) -> torch.Tensor:
    """Per-expert selection weights from a recorded serving run.

    Same contract as the split benchmark's loader: only the SHAPE of the
    recorded skew transfers, since this benchmark sets its own token count.
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
    # CPU here, device chosen at sample time. This loader runs during argument
    # handling, BEFORE torch.cuda.set_device(local_rank), so ".to('cuda')" would
    # resolve to cuda:0 for every rank and hand a cuda:0 tensor to a rank living
    # on cuda:3. Same defect, same fix, as the split benchmark.
    _REPLAY_WEIGHTS = (w / w.sum()).cpu()
    return _REPLAY_WEIGHTS


def protocol_shape(profile: MegaMoEModelProfile, ep_size: int) -> tuple[int, int, int]:
    profile.validate(eg_size=ep_size)
    local_routed = profile.num_routed_experts // ep_size
    local_protocol = local_routed + profile.num_shared_experts
    return local_routed, local_protocol * ep_size, profile.protocol_top_k


def make_protocol_routes(
    profile: MegaMoEModelProfile,
    tokens: int,
    *,
    rank: int,
    ep_size: int,
    routing: str,
    hot_expert_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_routed, protocol_experts, protocol_topk = protocol_shape(profile, ep_size)
    active_experts = profile.num_routed_experts
    if routing == "hotset":
        active_experts = max(
            profile.routed_top_k,
            int(round(profile.num_routed_experts * hot_expert_fraction)),
        )
    token_ids = torch.arange(tokens, dtype=torch.int64, device="cuda") + rank * tokens
    offsets = torch.arange(profile.routed_top_k, dtype=torch.int64, device="cuda")
    if routing == "replay":
        # Recorded real-router distribution. Deterministic per rank so repeats of
        # a configuration are identical, and without replacement so a token
        # cannot select the same expert twice -- see the split benchmark.
        if _REPLAY_WEIGHTS is None:
            raise SystemExit("routing=replay requires --replay-distribution")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(0x5EED0000 + rank * 4096)
        probs = _REPLAY_WEIGHTS.to(device=offsets.device, dtype=torch.float32).expand(
            tokens, -1
        )
        real_ids = torch.multinomial(
            probs, profile.routed_top_k, replacement=False, generator=gen
        )
    else:
        real_ids = (
            token_ids[:, None] * profile.routed_top_k + offsets[None, :]
        ) % active_experts
    owner = torch.div(real_ids, local_routed, rounding_mode="floor")
    local_id = real_ids.remainder(local_routed)
    protocol_ids = owner * (local_routed + profile.num_shared_experts) + local_id

    raw_weights = torch.arange(
        profile.routed_top_k,
        0,
        -1,
        dtype=torch.float32,
        device="cuda",
    )
    weights = raw_weights.expand(tokens, -1).clone()
    if profile.gate_renormalize:
        weights /= weights.sum(dim=1, keepdim=True)
    if profile.num_shared_experts:
        shared_id = rank * (local_routed + 1) + local_routed
        shared_ids = torch.full((tokens, 1), shared_id, dtype=torch.int64, device="cuda")
        shared_weights = torch.full(
            (tokens, 1),
            1.0 / profile.routed_scaling_factor,
            dtype=torch.float32,
            device="cuda",
        )
        protocol_ids = torch.cat((protocol_ids, shared_ids), dim=1)
        weights = torch.cat((weights, shared_weights), dim=1)
    counts = torch.bincount(protocol_ids.flatten(), minlength=protocol_experts)
    assert protocol_ids.shape == (tokens, protocol_topk)
    return protocol_ids.contiguous(), weights.contiguous(), counts


def route_summary(counts: torch.Tensor) -> dict[str, float | int]:
    counts_f = counts.float()
    mean = float(counts_f.mean().item())
    std = float(counts_f.std(unbiased=False).item())
    sorted_counts = counts.sort().values
    p95_index = min(len(sorted_counts) - 1, math.ceil(0.95 * len(sorted_counts)) - 1)
    return {
        "assignments": int(counts.sum().item()),
        "active_protocol_experts": int((counts > 0).sum().item()),
        "min_assignments": int(counts.min().item()),
        "mean_assignments": mean,
        "p95_assignments": int(sorted_counts[p95_index].item()),
        "max_assignments": int(counts.max().item()),
        "load_cv_percent": 100.0 * std / mean if mean else 0.0,
    }


def make_layer_weights(
    profile: MegaMoEModelProfile,
    *,
    local_protocol_experts: int,
    layer_id: int,
) -> LayerWeights:
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    l1 = torch.full(
        (local_protocol_experts, 2 * profile.intermediate_size, profile.hidden_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2 = torch.full(
        (local_protocol_experts, profile.hidden_size, profile.intermediate_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l1_fp4 = cast_weights_to_fp4(l1)
    l2_fp4 = cast_weights_to_fp4(l2)
    del l1, l2
    mega_l1, mega_l2 = transform_weights_for_mega_moe(l1_fp4, l2_fp4)
    return LayerWeights(
        baseline_l1=l1_fp4,
        baseline_l2=l2_fp4,
        mega_l1=mega_l1,
        mega_l2=mega_l2,
    )


def validate_gran32_packed_quant(hidden_size: int, rank: int) -> dict[str, Any]:
    """Require the production CUDA quantizer to match the Torch contract exactly."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xA1C000 + rank)
    values = torch.randn(
        (7, hidden_size),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.375)
    values[:, ::127].mul_(3.0)
    x = values.to(torch.bfloat16).contiguous()
    cuda_fp8, cuda_scales = per_token_cast_to_fp8_cuda(
        x,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    reference_fp8, reference_scales = per_token_cast_to_fp8_torch(
        x,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    fp8_mismatches = int(
        (cuda_fp8.view(torch.uint8) != reference_fp8.view(torch.uint8)).sum().item()
    )
    scale_mismatches = int((cuda_scales != reference_scales).sum().item())
    result = {
        "contract": "CUDA packed UE8M0 gran_k=32 equals Torch reference",
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "fp8_elements": cuda_fp8.numel(),
        "packed_scale_elements": cuda_scales.numel(),
        "fp8_byte_mismatches": fp8_mismatches,
        "packed_scale_mismatches": scale_mismatches,
        "passed": fp8_mismatches == 0 and scale_mismatches == 0,
    }
    del values, x, cuda_fp8, cuda_scales, reference_fp8, reference_scales
    return result


def validate_model_activation_quant(
    profile: MegaMoEModelProfile,
    rank: int,
) -> dict[str, Any]:
    """Check the baseline activation/quant path against its Torch contract."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xA2C000 + rank)
    rows = 7
    values = torch.randn(
        (rows, 2 * profile.intermediate_size),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.5)
    x = values.to(torch.bfloat16).contiguous()
    topk_weights = torch.linspace(0.2, 0.8, rows, dtype=torch.float32, device="cuda")
    psum = torch.tensor([rows], dtype=torch.int32, device="cuda")
    cuda_fp8, cuda_scales = psum_silu_mul_fp8_quant_cuda(
        x,
        psum,
        alignment=1,
        topk_weights=topk_weights,
        group_size=32,
        activation_clamp=profile.activation_clamp,
        activation_alpha=profile.activation_alpha,
        activation_up_bias=profile.activation_up_bias,
    )

    gate, up = x.float().chunk(2, dim=1)
    if profile.activation_clamp is not None:
        gate = gate.clamp(max=profile.activation_clamp)
        up = up.clamp(min=-profile.activation_clamp, max=profile.activation_clamp)
    activated = gate * torch.sigmoid(profile.activation_alpha * gate)
    activated *= up + profile.activation_up_bias
    activated *= topk_weights[:, None]
    reference_fp8, reference_scales = per_token_cast_to_fp8_torch(
        activated,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    fp8_mismatches = int(
        (cuda_fp8.view(torch.uint8) != reference_fp8.view(torch.uint8)).sum().item()
    )
    scale_mismatches = int((cuda_scales != reference_scales).sum().item())
    result = {
        "contract": (
            "CUDA psum model activation, top-k weighting and packed UE8M0 gran_k=32 "
            "equal the Torch reference"
        ),
        "input_shape": list(x.shape),
        "fp8_byte_mismatches": fp8_mismatches,
        "packed_scale_mismatches": scale_mismatches,
        "passed": fp8_mismatches == 0 and scale_mismatches == 0,
    }
    del (
        values,
        x,
        topk_weights,
        psum,
        cuda_fp8,
        cuda_scales,
        gate,
        up,
        activated,
        reference_fp8,
        reference_scales,
    )
    return result


def expand_block128_scales_for_block32(scales: torch.Tensor) -> torch.Tensor:
    """Repeat each packed block-128 UE8M0 scale for four block-32 groups."""
    rows = int(scales.shape[0])
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(rows, -1)
    expanded = scale_bytes.repeat_interleave(4, dim=1).contiguous()
    return expanded.view(torch.int32)


def mega_stage_once(
    *,
    profile: MegaMoEModelProfile,
    buffer: MegaMoESymmBuffer,
    weights: list[LayerWeights],
    layers: int,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    input_quant_granularity: int = 32,
) -> torch.Tensor:
    for layer_id in range(layers):
        layer = weights[layer_id % len(weights)]
        x_fp8, x_sf = deepgemm.per_token_cast_to_fp8(
            hidden,
            use_ue8m0=True,
            gran_k=input_quant_granularity,
            use_packed_ue8m0=True,
        )
        if input_quant_granularity == 128:
            x_sf = expand_block128_scales_for_block32(x_sf)
        elif input_quant_granularity != 32:
            raise ValueError("MegaMoE input quantization must use block-32 or block-128")
        buffer.x[: hidden.shape[0]].copy_(x_fp8)
        buffer.x_sf[: hidden.shape[0]].copy_(x_sf)
        buffer.topk_idx[: hidden.shape[0]].copy_(topk_ids)
        buffer.topk_weights[: hidden.shape[0]].copy_(topk_weights)
        fp8_fp4_mega_moe(
            output,
            layer.mega_l1,
            layer.mega_l2,
            buffer,
            activation_clamp=profile.activation_clamp,
            activation_alpha=profile.activation_alpha,
            activation_up_bias=profile.activation_up_bias,
            fast_math=True,
        )
        if profile.routed_scaling_factor != 1.0:
            output.mul_(profile.routed_scaling_factor)
    return output


def deepep_stage_once(
    *,
    profile: MegaMoEModelProfile,
    buffer: OfficialDeepEPReference,
    weights: list[LayerWeights],
    layers: int,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_alignment: int,
    weight_before_l2_quant: bool = False,
) -> torch.Tensor:
    output = hidden
    for layer_id in range(layers):
        layer = weights[layer_id % len(weights)]
        x_fp8, x_sf = deepgemm.per_token_cast_to_fp8(
            hidden,
            use_ue8m0=True,
            gran_k=128,
            use_packed_ue8m0=True,
        )
        dispatch = buffer.dispatch_and_scatter(
            x_fp8,
            x_sf,
            topk_ids,
            topk_weights,
        )
        recv_tokens = int(dispatch.hidden_states.shape[0])
        l1_output = torch.empty(
            (recv_tokens, 2 * profile.intermediate_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deepgemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (dispatch.hidden_states, dispatch.hidden_states_scale),
            layer.baseline_l1,
            l1_output,
            dispatch.m_indices,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
            disable_ue8m0_cast=False,
        )
        expert_weights = None
        gather_weights = None
        if weight_before_l2_quant:
            valid = (
                (dispatch.topk_ids >= 0)
                & (dispatch.topk_ids < buffer.num_experts)
                & (dispatch.output_index >= 0)
                & (dispatch.output_index < recv_tokens)
            )
            expert_weights = torch.ones(
                recv_tokens,
                dtype=torch.float32,
                device=dispatch.output_index.device,
            )
            expert_weights[dispatch.output_index[valid].long()] = dispatch.topk_weights[valid]
            gather_weights = torch.ones_like(dispatch.topk_weights)
        l2_input, l2_input_sf = persistent_psum_silu_mul_quant(
            l1_output,
            dispatch.psum_tokens_per_expert,
            alignment=expert_alignment,
            topk_weights=expert_weights,
            group_size=32,
            activation_clamp=profile.activation_clamp,
            activation_alpha=profile.activation_alpha,
            activation_up_bias=profile.activation_up_bias,
        )
        l2_output = torch.empty(
            (recv_tokens, profile.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deepgemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (l2_input, l2_input_sf),
            layer.baseline_l2,
            l2_output,
            dispatch.m_indices,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=False,
        )
        output = buffer.gather_and_combine(
            l2_output,
            dispatch,
            topk_weights=gather_weights,
        )
        if profile.routed_scaling_factor != 1.0:
            output.mul_(profile.routed_scaling_factor)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(FP4_PROFILES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--tokens-per-rank", type=int, required=True)
    parser.add_argument("--mtp-nextn", type=int, default=0)
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
        choices=("mega", "deepep", "both"),
        default="mega",
        help="Measure FastAFD MegaMoE, official DeepEP+DeepGEMM, or both",
    )
    parser.add_argument(
        "--measurement-order",
        choices=("grouped", "alternating"),
        default="grouped",
        help=(
            "Measure each backend after its own contiguous warmup by default; "
            "alternating is retained only for diagnosing cross-backend order effects"
        ),
    )
    parser.add_argument(
        "--weight-slots",
        type=int,
        default=2,
        help="Number of deterministic weight sets to allocate and cycle across layers",
    )
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    profile = get_megamoe_model_profile(args.model)
    profile.validate(eg_size=args.ep_size)
    if profile.weight_precision != "fp4":
        raise SystemExit("colocated MegaMoE currently supports FP4 weights only")
    layers = profile.num_moe_layers if args.layers == 0 else args.layers
    if not 1 <= layers <= profile.num_moe_layers:
        raise SystemExit(f"layers must be in [1, {profile.num_moe_layers}]")
    if args.tokens_per_rank <= 0:
        raise SystemExit("tokens-per-rank must be positive")
    if args.mtp_nextn < 0:
        raise SystemExit("mtp-nextn must be non-negative")
    if args.routing == "replay" and not args.replay_distribution:
        raise SystemExit("--routing replay requires --replay-distribution")
    if args.routing == "replay":
        # The loader was defined here but never called, so --replay-distribution
        # parsed cleanly and then every rank hit the sampler with _REPLAY_WEIGHTS
        # still None. The split benchmark calls it; this one did not.
        load_replay_distribution(args.replay_distribution, profile.num_routed_experts)
    if args.replay_distribution and args.routing != "replay":
        raise SystemExit("--replay-distribution is only meaningful with --routing replay")
    if not 0 < args.hot_expert_fraction <= 1:
        raise SystemExit("hot-expert-fraction must be in (0, 1]")
    if args.weight_slots < 1:
        raise SystemExit("weight-slots must be positive")
    if args.warmups < 1 or args.iterations < 3:
        raise SystemExit("warmups must be >=1 and iterations must be >=3")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.ep_size:
        raise SystemExit(f"WORLD_SIZE={world_size}, expected ep_size={args.ep_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    _RUNTIME_CLEANUPS.append(dist.destroy_process_group)

    quant_validation = validate_gran32_packed_quant(profile.hidden_size, rank)
    activation_validation = validate_model_activation_quant(profile, rank)
    quant_passed = torch.tensor(
        int(quant_validation["passed"] and activation_validation["passed"]),
        dtype=torch.int32,
        device="cuda",
    )
    dist.all_reduce(quant_passed, op=dist.ReduceOp.MIN)
    if not bool(quant_passed.item()):
        raise RuntimeError(
            "packed gran_k=32 FP8 quantization does not match the Torch reference; "
            f"rank {rank}: input={quant_validation}; activation={activation_validation}"
        )

    verify_width = args.mtp_nextn + 1
    physical_tokens = args.tokens_per_rank * verify_width
    local_routed, protocol_experts, protocol_topk = protocol_shape(profile, args.ep_size)
    local_protocol = protocol_experts // args.ep_size
    enabled_backends = ("mega", "deepep") if args.backend == "both" else (args.backend,)
    mega_buffer = (
        MegaMoESymmBuffer(
            dist.group.WORLD,
            num_experts=protocol_experts,
            num_max_tokens_per_rank=physical_tokens,
            num_topk=protocol_topk,
            hidden=profile.hidden_size,
            intermediate_hidden=profile.intermediate_size,
        )
        if "mega" in enabled_backends
        else None
    )
    if mega_buffer is not None:
        _RUNTIME_CLEANUPS.append(mega_buffer.destroy)
    expert_alignment = 128
    deepgemm.set_mk_alignment_for_contiguous_layout(expert_alignment)
    deepep_buffer = (
        OfficialDeepEPReference(
            group=dist.group.WORLD,
            hidden_size=profile.hidden_size,
            num_experts=protocol_experts,
            expert_alignment=expert_alignment,
        )
        if "deepep" in enabled_backends
        else None
    )
    if deepep_buffer is not None:
        _RUNTIME_CLEANUPS.append(deepep_buffer.destroy)
    topk_ids, topk_weights, route_counts = make_protocol_routes(
        profile,
        physical_tokens,
        rank=rank,
        ep_size=args.ep_size,
        routing=args.routing,
        hot_expert_fraction=args.hot_expert_fraction,
    )
    dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    input_seed = INPUT_SEED_BASE + rank
    input_generator = torch.Generator(device="cuda")
    input_generator.manual_seed(input_seed)
    hidden = (
        torch.randn(
            (physical_tokens, profile.hidden_size),
            dtype=torch.float32,
            device="cuda",
            generator=input_generator,
        )
        .mul_(0.375)
        .to(torch.bfloat16)
    )
    mega_output = torch.empty_like(hidden)

    torch.cuda.reset_peak_memory_stats()
    init_start = time.perf_counter()
    weight_slots = min(layers, args.weight_slots)
    weights: list[LayerWeights] = []
    for layer_id in range(weight_slots):
        weights.append(
            make_layer_weights(
                profile,
                local_protocol_experts=local_protocol,
                layer_id=layer_id,
            )
        )
        if layer_id % 4 == 3:
            torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - init_start
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    def run_backend(name: str, *, correctness_mode: bool = False) -> torch.Tensor:
        if name == "mega":
            assert mega_buffer is not None
            return mega_stage_once(
                profile=profile,
                buffer=mega_buffer,
                weights=weights,
                layers=layers,
                hidden=hidden,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=mega_output,
                input_quant_granularity=128 if correctness_mode else 32,
            )
        assert name == "deepep" and deepep_buffer is not None
        return deepep_stage_once(
            profile=profile,
            buffer=deepep_buffer,
            weights=weights,
            layers=layers,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
            weight_before_l2_quant=correctness_mode,
        )

    correctness: dict[str, Any] | None = None
    if args.backend == "both":
        dist.barrier()
        mega_reference = run_backend("mega", correctness_mode=True).clone()
        torch.cuda.synchronize()
        dist.barrier()
        deepep_reference = run_backend("deepep", correctness_mode=True)
        torch.cuda.synchronize()
        difference = mega_reference.float() - deepep_reference.float()
        finite = bool(
            torch.isfinite(mega_reference).all().item()
            and torch.isfinite(deepep_reference).all().item()
        )
        ulp_difference = (
            bf16_ordered_codes(mega_reference) - bf16_ordered_codes(deepep_reference)
        ).abs()
        max_bf16_ulp_error = int(ulp_difference.max().item())
        reference_norm = float(torch.linalg.vector_norm(deepep_reference.float()).item())
        relative_l2 = float(torch.linalg.vector_norm(difference).item()) / max(
            reference_norm, 1e-12
        )
        row_max_abs = difference.abs().amax(dim=1)
        differing_rows = torch.nonzero(row_max_abs > 0, as_tuple=False).flatten()
        row_diagnostics = []
        for row_idx_tensor in differing_rows[:16]:
            row_idx = int(row_idx_tensor.item())
            row_difference = difference[row_idx].abs()
            row_diagnostics.append(
                {
                    "row": row_idx,
                    "max_abs_error": float(row_difference.max().item()),
                    "mean_abs_error": float(row_difference.mean().item()),
                    "max_bf16_ulp_error": int(ulp_difference[row_idx].max().item()),
                    "mega_abs_mean": float(mega_reference[row_idx].float().abs().mean().item()),
                    "deepep_abs_mean": float(deepep_reference[row_idx].float().abs().mean().item()),
                    "protocol_expert_ids": topk_ids[row_idx].tolist(),
                    "protocol_topk_weights": topk_weights[row_idx].tolist(),
                }
            )
        correctness = {
            "contract": (
                "MegaMoE and official DeepEP+DeepGEMM use identical block-128 FP8 "
                f"inputs, routes, {profile.moe_quant_contract} quantization, model activation "
                "and pre-L2 top-k weighting "
                "for the numerical check; timed paths retain their production contracts"
            ),
            "exact": bool(torch.equal(mega_reference, deepep_reference)),
            "max_abs_error": float(difference.abs().max().item()),
            "mean_abs_error": float(difference.abs().mean().item()),
            "relative_l2_error": relative_l2,
            "rows_with_nonzero_error": int(differing_rows.numel()),
            "first_differing_rows": row_diagnostics,
            "all_outputs_finite": finite,
            "max_bf16_ulp_error": max_bf16_ulp_error,
            "threshold_bf16_ulp": 1,
            "legacy_diagnostic_threshold_relative_l2": 1e-3,
            "passed_contract": "all outputs finite and every BF16 value differs by at most 1 ULP",
            "passed": bool(finite and max_bf16_ulp_error <= 1),
        }
        del mega_reference, deepep_reference, difference, ulp_difference

    gathered_iterations: dict[str, list[list[dict[str, float]]]] = {
        name: [] for name in enabled_backends
    }
    output_by_backend: dict[str, torch.Tensor] = {}
    for name, record in measurement_plan(
        enabled_backends, args.warmups, args.iterations, args.measurement_order
    ):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        output_by_backend[name] = run_backend(name)
        end.record()
        end.synchronize()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            {
                "cuda_ms": float(start.elapsed_time(end)),
                "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
            },
            object_gather_list=gathered,
            dst=0,
        )
        if record and rank == 0:
            assert gathered is not None
            gathered_iterations[name].append(gathered)

    local_output_quality = {
        name: {
            "finite": bool(torch.isfinite(output).all().item()),
            "abs_mean": float(output.float().abs().mean().item()),
        }
        for name, output in output_by_backend.items()
    }
    output_quality_by_rank = [None] * world_size if rank == 0 else None
    correctness_by_rank = [None] * world_size if rank == 0 else None
    init_times = [None] * world_size if rank == 0 else None
    peak_memories = [None] * world_size if rank == 0 else None
    provenance = [None] * world_size if rank == 0 else None
    quant_validations = [None] * world_size if rank == 0 else None
    activation_validations = [None] * world_size if rank == 0 else None
    dist.gather_object(
        local_output_quality,
        object_gather_list=output_quality_by_rank,
        dst=0,
    )
    dist.gather_object(correctness, object_gather_list=correctness_by_rank, dst=0)
    dist.gather_object(initialization_seconds, object_gather_list=init_times, dst=0)
    dist.gather_object(peak_memory_bytes, object_gather_list=peak_memories, dst=0)
    dist.gather_object(gpu_provenance(local_rank), object_gather_list=provenance, dst=0)
    dist.gather_object(
        quant_validation,
        object_gather_list=quant_validations,
        dst=0,
    )
    dist.gather_object(
        activation_validation,
        object_gather_list=activation_validations,
        dst=0,
    )

    if rank == 0:
        assert output_quality_by_rank is not None
        backend_results: dict[str, Any] = {}
        for name in enabled_backends:
            stage_cuda = [
                max(sample["cuda_ms"] for sample in rows) for rows in gathered_iterations[name]
            ]
            stage_wall = [
                max(sample["wall_ms"] for sample in rows) for rows in gathered_iterations[name]
            ]
            stage_summary = summarize(stage_cuda)
            finite_by_rank = [item[name]["finite"] for item in output_quality_by_rank]
            backend_results[name] = {
                "stage_cuda": stage_summary,
                "stage_wall": summarize(stage_wall),
                "stage_cuda_samples_ms": stage_cuda,
                "stage_wall_samples_ms": stage_wall,
                "rank_cuda_samples_ms": [
                    [sample["cuda_ms"] for sample in rows] for rows in gathered_iterations[name]
                ],
                "rank_wall_samples_ms": [
                    [sample["wall_ms"] for sample in rows] for rows in gathered_iterations[name]
                ],
                "all_outputs_finite_by_rank": finite_by_rank,
                "output_abs_mean_by_rank": [
                    item[name]["abs_mean"] for item in output_quality_by_rank
                ],
                "stable": bool(stage_summary["cv_percent"] <= 3.0 and all(finite_by_rank)),
            }
        primary_backend = "mega" if "mega" in backend_results else "deepep"
        primary = backend_results[primary_backend]
        speedup = None
        conservative_speedup = None
        if args.backend == "both":
            speedup = (
                backend_results["deepep"]["stage_cuda"]["p50_ms"]
                / backend_results["mega"]["stage_cuda"]["p50_ms"]
            )
            conservative_speedup = (
                backend_results["deepep"]["stage_cuda"]["min_ms"]
                / backend_results["mega"]["stage_cuda"]["max_ms"]
            )
        correctness_passed = (
            None
            if args.backend != "both"
            else all(item is not None and item["passed"] for item in correctness_by_rank)
        )
        payload = {
            "schema": "fastafd.megamoe-colocated-model-benchmark.v2",
            "generated_at": utc_now(),
            "system_label": os.environ.get("MEASUREMENT_SYSTEM", "unspecified"),
            "measurement_boundary": (
                "colocated MoE stage with identical input, route, "
                f"the {profile.moe_quant_contract} quantization contract and "
                "model-specific activation contract: FastAFD MegaMoE fuses dispatch, L1, "
                "activation/requant, L2 and combine; the reference executes official DeepEP "
                "normal dispatch/combine, SGLang scatter/gather and two contiguous DeepGEMM "
                "GEMMs. Both include input FP8 quantization, "
                "routed-output scaling and the declared replicated shared-expert protocol route. "
                "Router, attention, dense-only layers, residual, sampling and scheduler are excluded."
            ),
            "selected_backends": list(enabled_backends),
            "primary_backend": primary_backend,
            "model_profile": asdict(profile),
            "topology": {
                "world_size": world_size,
                "ep_size": args.ep_size,
                "local_routed_experts": local_routed,
                "local_protocol_experts": local_protocol,
                "protocol_experts": protocol_experts,
            },
            "workload": {
                "logical_tokens_per_rank": args.tokens_per_rank,
                "mtp_nextn": args.mtp_nextn,
                "verify_width_q": verify_width,
                "physical_tokens_per_rank": physical_tokens,
                "physical_tokens_per_step": physical_tokens * args.ep_size,
                "physical_routed_assignments_per_expert_rank": (
                    physical_tokens * args.ep_size * profile.routed_top_k / args.ep_size
                ),
                "accepted_tokens": None,
                "accepted_tokens_note": (
                    "acceptance is not applied to kernel latency; the system model converts "
                    "verified candidates into useful output tokens"
                ),
                "layers": layers,
                "weight_slots": weight_slots,
                "baseline_weight_working_set_bytes": sum(
                    tensor_pair_nbytes(layer.baseline_l1) + tensor_pair_nbytes(layer.baseline_l2)
                    for layer in weights
                ),
                "megamoe_weight_working_set_bytes": sum(
                    tensor_pair_nbytes(layer.mega_l1) + tensor_pair_nbytes(layer.mega_l2)
                    for layer in weights
                ),
                "weight_reuse_note": (
                    "deterministic quantized weight slots are cycled across shape-identical MoE "
                    "layers; values do not change the kernel schedule or transferred byte count"
                ),
                "routing": args.routing,
                "hot_expert_fraction": args.hot_expert_fraction,
                "replay_distribution": args.replay_distribution,
                "routing_load": route_summary(route_counts.cpu()),
                "input_policy": (
                    "fixed deterministic representative hidden state reused independently "
                    "for every measured MoE layer; pure MoE outputs are not chained because "
                    "attention, normalization and residual paths are outside this boundary"
                ),
                "input_seed_base": INPUT_SEED_BASE,
                "input_seed_by_rank": [INPUT_SEED_BASE + item for item in range(world_size)],
            },
            "warmups": args.warmups,
            "iterations": args.iterations,
            "measurement_order": args.measurement_order,
            "measurement_order_note": (
                "grouped gives every backend an independent contiguous warmup and measurement "
                "window; alternating is diagnostic because the preceding comparison backend can "
                "change the next sample's clock and cache state"
            ),
            "backend_results": backend_results,
            "speedup_deepep_over_megamoe": speedup,
            "speedup_lower_bound_deepep_over_megamoe": conservative_speedup,
            "megamoe_outperforms_reference": bool(
                conservative_speedup is not None and conservative_speedup > 1.0
            ),
            "deepep_backend": (deepep_buffer.metadata() if deepep_buffer is not None else None),
            "correctness_by_rank": correctness_by_rank,
            "correctness_passed": correctness_passed,
            "eligible_for_profile": bool(
                args.backend == "both"
                and backend_results["mega"]["stable"]
                and backend_results["deepep"]["stable"]
                and all(backend_results["deepep"]["all_outputs_finite_by_rank"])
                and correctness_passed is True
            ),
            "stage_cuda": primary["stage_cuda"],
            "stage_wall": primary["stage_wall"],
            "stage_cuda_samples_ms": primary["stage_cuda_samples_ms"],
            "stage_wall_samples_ms": primary["stage_wall_samples_ms"],
            "rank_cuda_samples_ms": primary["rank_cuda_samples_ms"],
            "rank_wall_samples_ms": primary["rank_wall_samples_ms"],
            "all_outputs_finite_by_rank": primary["all_outputs_finite_by_rank"],
            "output_abs_mean_by_rank": primary["output_abs_mean_by_rank"],
            "stable": primary["stable"],
            "stability_contract": (
                "MegaMoE and DeepEP+DeepGEMM stage CUDA CV <= 3% and finite output on every rank"
            ),
            "qualification_contract": (
                "MegaMoE and DeepEP+DeepGEMM timings stable; matched output check passes; "
                "all outputs finite. Backend speedup is reported but does not filter valid timing evidence"
            ),
            "initialization_seconds_by_rank": init_times,
            "peak_cuda_memory_bytes_by_rank": peak_memories,
            "quantization_validation_by_rank": quant_validations,
            "activation_quant_validation_by_rank": activation_validations,
            "source": source_provenance(),
            "provenance_by_rank": provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "backend_results": {
                        name: result["stage_cuda"] for name, result in backend_results.items()
                    },
                    "speedup_deepep_over_megamoe": speedup,
                    "speedup_lower_bound_deepep_over_megamoe": conservative_speedup,
                    "correctness_passed": correctness_passed,
                },
                indent=2,
            )
        )

    dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_runtime_resources()
