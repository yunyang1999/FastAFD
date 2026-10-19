"""afd adapter for the split MegaMoE M2N mega kernels.

Selected with ``MINISGL_AFD_MOE_BACKEND=megamoe_m2n``.  Replaces the
DeepEP union adapter's per-layer dispatch/expert/combine sequence:

- AG ranks: :meth:`ag_moe` is ONE fused call (and one kernel) that quantizes
  the hidden states to FP8 (per-32 packed UE8M0), pushes route metadata to the
  EG owner ranks, waits for the expert write-back, and reduces top-k into a
  bf16 output.
- EG ranks: :meth:`eg_moe` is ONE persistent kernel that pulls FP8 tokens from
  the AG ranks, runs the fused FP8-activation expert pipeline (FP8 or FP4
  weights; model-specific SwiGLU and
  top-k weights fused in the L1 epilogue), and pushes BF16 expert outputs into
  the AG combine buffers.  Qwen FP8 checkpoint weights are requantized once at
  init to the per-32 UE8M0 mega format (:meth:`register_eg_layer_weights`).

Topology contract matches ``DeepEPM2NAdapter``: one union world, AG ranks
first; EG rank ``g`` owns experts ``[g*E_local, (g+1)*E_local)``; ``topk_ids``
are REAL model expert ids with ``-1`` for dummy routes.

Constraints (from the mega kernel): ``hidden % 512 == 0`` and
``moe_intermediate_size % 512 == 0`` (e.g. Qwen3-235B 4096/1536 OK,
Qwen3-30B's 768 intermediate is NOT supported yet).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from minisgl.kernel import megamoe_m2n_mega as _mm
from minisgl.layers.moe.moe_runner.deepgemm_grouped import (
    _requant_channel_weight_ue8m0_inplace,
)


class MegaMoEM2NAfdAdapter:
    backend = "megamoe_m2n"

    def __init__(
        self,
        *,
        group: Any,
        ag_size: int,
        eg_size: int,
        real_num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        num_max_dispatch_tokens_per_rank: int,
        num_lanes: int = 1,
        gate_renormalize: bool = True,
        shared_experts_per_rank: int = 0,
        routed_scaling_factor: float = 1.0,
        weight_precision: str = "fp8",
        activation: str = "silu",
        activation_alpha: float = 1.0,
        activation_clamp: float | None = None,
        activation_up_bias: float = 0.0,
        eg_expected_tokens_override: int | None = None,
        eg_prefetch_bytes: int | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("MegaMoEM2NAfdAdapter requires torch.distributed")
        if hidden_size % 512 != 0 or intermediate_size % 512 != 0:
            raise RuntimeError(
                "megamoe_m2n requires hidden and moe intermediate sizes divisible "
                f"by 512, got hidden={hidden_size} intermediate={intermediate_size}"
            )
        self.group = group
        self.rank = int(dist.get_rank(group=group))
        self.ag_size = int(ag_size)
        self.eg_size = int(eg_size)
        self.is_ag = self.rank < self.ag_size
        self.is_eg = not self.is_ag
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.routed_top_k = int(top_k)
        self.shared_experts_per_rank = int(shared_experts_per_rank)
        if self.shared_experts_per_rank not in (0, 1):
            raise RuntimeError(
                "megamoe_m2n currently supports at most one replicated shared "
                f"expert per EG rank, got {self.shared_experts_per_rank}"
            )
        self.top_k = self.routed_top_k + self.shared_experts_per_rank
        self.real_num_experts = int(real_num_experts)
        if self.real_num_experts % self.eg_size != 0:
            raise RuntimeError(
                f"routed expert count must divide eg_size: experts={self.real_num_experts} "
                f"eg_size={self.eg_size}"
            )
        self.num_experts = self.real_num_experts + self.eg_size * self.shared_experts_per_rank
        if self.num_experts % self.eg_size != 0:
            raise RuntimeError(
                f"protocol expert count must divide eg_size: experts={self.num_experts} "
                f"eg_size={self.eg_size}"
            )
        self.bucket = int(num_max_dispatch_tokens_per_rank)
        self.num_lanes = max(1, int(num_lanes))
        if self.num_lanes > 4:
            raise RuntimeError(f"megamoe_m2n supports at most four lanes, got {self.num_lanes}")
        self.routed_scaling_factor = float(routed_scaling_factor)
        if self.routed_scaling_factor <= 0:
            raise RuntimeError("routed_scaling_factor must be positive")
        self.weight_precision = str(weight_precision).lower()
        if self.weight_precision not in ("fp8", "fp4"):
            raise RuntimeError(
                f"megamoe_m2n weight_precision must be fp8 or fp4, got {weight_precision!r}"
            )
        self.use_fp8_weights = self.weight_precision == "fp8"
        self.activation = str(activation).lower()
        self.activation_alpha = float(activation_alpha)
        self.activation_clamp = None if activation_clamp is None else float(activation_clamp)
        self.activation_up_bias = float(activation_up_bias)
        if self.activation == "silu":
            if self.activation_alpha != 1.0 or self.activation_up_bias != 0.0:
                raise RuntimeError("plain SiLU requires activation_alpha=1 and up_bias=0")
        elif self.activation == "swigluoai":
            if (
                self.activation_alpha <= 0
                or self.activation_clamp is None
                or self.activation_up_bias != 1.0
            ):
                raise RuntimeError("SwiGLU-OAI requires positive alpha, a clamp, and up_bias=1")
        else:
            raise RuntimeError(
                f"megamoe_m2n activation must be silu or swigluoai, got {activation!r}"
            )

        self.buffers = [
            _mm.MegaMoEM2NSymmBuffer(
                group,
                ag_size=self.ag_size,
                eg_size=self.eg_size,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=self.bucket,
                num_topk=self.top_k,
                hidden=self.hidden_size,
                intermediate_hidden=self.intermediate_size,
            )
            for _ in range(self.num_lanes)
        ]
        self._y_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._eg_weights: dict[int, tuple] = {}
        self._eg_desc_blobs: dict[int, torch.Tensor] = {}
        self._eg_l1_ptrs: dict[int, int] = {}
        self._eg_global_to_compact: dict[int, int] = {}
        self._eg_l1_nbytes = 0
        self._desc_blob_all = None
        self._l1_ptrs_all = None
        self._destroyed = False

        # Grid sizing. AG leaves SMs free for attention-side compute. EG runs on
        # dedicated expert ranks, so it defaults to full-device kernels.
        self.ag_num_sms = int(os.environ.get("MINISGL_MEGAMOE_AG_SMS", "24"))
        self.eg_num_sms = 0
        is_minimax_m25_shape = (
            self.hidden_size == 3072
            and self.intermediate_size == 1536
            and self.real_num_experts == 256
            and self.routed_top_k == 8
            and self.shared_experts_per_rank == 0
        )
        # Optional override for the GEMM block/wave heuristic input.  GLM's
        # remote shared expert injects a replicated 9th route: routed experts
        # are sparse, but the shared expert sees one route per token from a
        # subset of AG ranks.  A slightly larger decode-scale heuristic keeps
        # the mixed routed/shared persistent kernel on the faster block_m=32
        # tiling.  Set MINISGL_MEGAMOE_EG_EXPECTED_TOKENS=0 to force the raw
        # per-call bucket-derived heuristic.  MiniMax M2.5 decode has far more
        # routed experts than Qwen; a smaller expected-token hint keeps the
        # persistent EG kernel on the fast decode tiling.
        eg_expected_env = os.environ.get("MINISGL_MEGAMOE_EG_EXPECTED_TOKENS")
        self.eg_expected_tokens_override = (
            int(eg_expected_tokens_override)
            if eg_expected_tokens_override is not None
            else int(eg_expected_env)
            if eg_expected_env is not None
            else (24 if is_minimax_m25_shape else (16 if self.shared_experts_per_rank else 0))
        )
        if self.eg_expected_tokens_override < 0:
            raise RuntimeError("eg_expected_tokens_override must be non-negative")
        self.eg_expected_tokens_source = (
            "argument"
            if eg_expected_tokens_override is not None
            else "env"
            if eg_expected_env is not None
            else (
                "minimax_m25_default"
                if is_minimax_m25_shape
                else ("remote_shared_default" if self.shared_experts_per_rank else "bucket")
            )
        )
        # The MiniMax M2.5 EG persistent kernel benefits from prefetching the
        # next layer's transformed FP8 weights; Qwen did not in the tuned runs.
        self.eg_prefetch_bytes = (
            int(eg_prefetch_bytes)
            if eg_prefetch_bytes is not None
            else (64 << 20) if is_minimax_m25_shape else 0
        )
        if self.eg_prefetch_bytes < 0:
            raise RuntimeError("eg_prefetch_bytes must be non-negative")
        self.gate_renormalize = bool(gate_renormalize)

    @property
    def role(self) -> str:
        return "ag" if self.is_ag else "eg"

    # -- AG side -----------------------------------------------------------------
    def ag_moe(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        lane: int = 0,
        external_quant: bool = False,
    ) -> torch.Tensor:
        """Fused MoE layer on an AG rank: quantize + dispatch + combine."""
        if not self.is_ag:
            raise RuntimeError("ag_moe called on an EG rank")
        buf = self.buffers[lane % self.num_lanes]
        n = int(hidden_states.shape[0])
        if n > self.bucket:
            raise RuntimeError(f"tokens={n} exceeds dispatch bucket={self.bucket}")

        # Per-32 FP8 quant, symm-buffer staging, dispatch, wait, and combine
        # reduce are fused into one AG kernel; the inputs are read in place.
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        if ids.dtype not in (torch.int32, torch.int64):
            ids = ids.long()
        w = topk_weights
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()
        input_topk = int(ids.shape[1])
        if int(w.shape[1]) != input_topk:
            raise RuntimeError(
                f"topk id/weight shape mismatch: ids={tuple(ids.shape)} weights={tuple(w.shape)}"
            )
        if input_topk not in (self.routed_top_k, self.top_k):
            raise RuntimeError(
                f"invalid topk width for megamoe_m2n: got {input_topk}, "
                f"expected routed={self.routed_top_k} or protocol={self.top_k}"
            )
        shared_weight = 0.0
        if self.shared_experts_per_rank:
            if input_topk == self.top_k:
                shared_weight = 0.0
            else:
                shared_weight = 1.0 / self.routed_scaling_factor

        key = (lane % self.num_lanes, n)
        y = self._y_cache.get(key)
        if y is None:
            y = torch.empty((n, self.hidden_size), dtype=torch.bfloat16,
                            device=hidden_states.device)
            self._y_cache[key] = y
        _mm.mega_moe_m2n_ag(y, x, ids, w, buf, rank=self.rank,
                            num_sms=self.ag_num_sms,
                            renormalize=self.gate_renormalize,
                            external_quant=bool(external_quant),
                            direct_combine=False,
                            input_num_topk=input_topk,
                            num_routed_experts=self.real_num_experts,
                            shared_topk_weight=shared_weight)
        return y

    # -- EG side -----------------------------------------------------------------
    def register_eg_layer_weights(
        self,
        layer_id: int,
        experts: Any,
        *,
        shared_experts: Any | None = None,
    ) -> None:
        """Requantize one layer's FP8 expert weights to the mega format.

        ``experts`` holds ``gate_up_proj [E_local, 2I, H]`` (fp8) +
        ``gate_up_proj_scale`` and ``down_proj``/``down_proj_scale``.  Qwen and
        MiniMax use 128x128 block scales; GLM compressed-tensors checkpoints use
        per-output-channel scales and are first repacked to the same block-scale
        FP8 representation.  One-time per layer; the originals can be freed
        afterwards by the caller.
        """
        if not self.is_eg:
            raise RuntimeError("register_eg_layer_weights called on an AG rank")
        if not self.use_fp8_weights:
            raise RuntimeError(
                "register_eg_layer_weights consumes FP8 checkpoint tensors; "
                "use register_eg_layer_fp4_weights for FP4 benchmark weights"
            )
        quant_method = getattr(getattr(experts, "quant", None), "method", None)
        scale_format = getattr(experts, "_fp8_scale_format", None)
        block_shape = tuple(getattr(getattr(experts, "quant", None), "weight_block_size", (128, 128)))
        if quant_method == "fp8_channel" or scale_format == "channel":
            experts.gate_up_proj_scale = _requant_channel_weight_ue8m0_inplace(
                experts.gate_up_proj, experts.gate_up_proj_scale)
            experts.down_proj_scale = _requant_channel_weight_ue8m0_inplace(
                experts.down_proj, experts.down_proj_scale)
            experts._fp8_scale_format = "block"
        elif block_shape != (128, 128):
            raise RuntimeError(
                "megamoe_m2n FP8 expert weights require 128x128 block scales "
                f"or GLM-style channel scales, got block_shape={block_shape}")
        l1_weight = experts.gate_up_proj
        l1_scale = experts.gate_up_proj_scale
        l2_weight = experts.down_proj
        l2_scale = experts.down_proj_scale
        if self.shared_experts_per_rank:
            if shared_experts is None:
                raise RuntimeError("remote shared expert is enabled but no shared weights were provided")
            shared_l1_weight = shared_experts.gate_up_proj.weight.unsqueeze(0).contiguous()
            shared_l1_scale = shared_experts.gate_up_proj.weight_scale.unsqueeze(0).contiguous()
            shared_l2_weight = shared_experts.down_proj.weight.unsqueeze(0).contiguous()
            shared_l2_scale = shared_experts.down_proj.weight_scale.unsqueeze(0).contiguous()
            shared_scale_format = getattr(shared_experts.gate_up_proj, "_fp8_scale_format", None)
            if shared_scale_format == "channel":
                shared_l1_scale = _requant_channel_weight_ue8m0_inplace(
                    shared_l1_weight, shared_l1_scale)
                shared_l2_scale = _requant_channel_weight_ue8m0_inplace(
                    shared_l2_weight, shared_l2_scale)
            elif tuple(getattr(shared_experts.gate_up_proj, "_fp8_block_size", (128, 128))) != (128, 128):
                raise RuntimeError(
                    "remote shared expert requires 128x128 block scales or "
                    "GLM-style channel scales"
                )
            l1_weight = torch.cat([l1_weight, shared_l1_weight], dim=0).contiguous()
            l1_scale = torch.cat([l1_scale, shared_l1_scale], dim=0).contiguous()
            l2_weight = torch.cat([l2_weight, shared_l2_weight], dim=0).contiguous()
            l2_scale = torch.cat([l2_scale, shared_l2_scale], dim=0).contiguous()
        l1 = _mm.requant_qwen_fp8_weights_per32(l1_weight, l1_scale, inplace=True)
        l2 = _mm.requant_qwen_fp8_weights_per32(l2_weight, l2_scale, inplace=True)
        l1_t, l2_t = _mm.transform_weights_for_mega_moe(l1, l2)
        self._register_transformed_eg_layer_weights(layer_id, l1_t, l2_t)

    def register_eg_layer_fp4_weights(
        self,
        layer_id: int,
        l1_weight: torch.Tensor,
        l2_weight: torch.Tensor,
        *,
        shared_l1_weight: torch.Tensor | None = None,
        shared_l2_weight: torch.Tensor | None = None,
    ) -> None:
        """Quantize BF16 expert weights into the MegaMoE FP4 benchmark format.

        This path is intended for deterministic synthetic or decoded-checkpoint
        benchmark weights.  It does not claim that arbitrary vendor checkpoint
        packing can be passed through unchanged.
        """
        if not self.is_eg:
            raise RuntimeError("register_eg_layer_fp4_weights called on an AG rank")
        if self.use_fp8_weights:
            raise RuntimeError("register_eg_layer_fp4_weights requires weight_precision='fp4'")
        expected_local = self.real_num_experts // self.eg_size
        expected_l1 = (expected_local, 2 * self.intermediate_size, self.hidden_size)
        expected_l2 = (expected_local, self.hidden_size, self.intermediate_size)
        if tuple(l1_weight.shape) != expected_l1 or tuple(l2_weight.shape) != expected_l2:
            raise RuntimeError(
                "routed FP4 source weight shape mismatch: "
                f"l1={tuple(l1_weight.shape)} expected={expected_l1}; "
                f"l2={tuple(l2_weight.shape)} expected={expected_l2}"
            )
        if l1_weight.dtype != torch.bfloat16 or l2_weight.dtype != torch.bfloat16:
            raise RuntimeError("FP4 source weights must be BF16 before quantization")
        if self.shared_experts_per_rank:
            expected_shared_l1 = (1, 2 * self.intermediate_size, self.hidden_size)
            expected_shared_l2 = (1, self.hidden_size, self.intermediate_size)
            if shared_l1_weight is None or shared_l2_weight is None:
                raise RuntimeError(
                    "remote shared expert is enabled but no shared weights were provided"
                )
            if (
                tuple(shared_l1_weight.shape) != expected_shared_l1
                or tuple(shared_l2_weight.shape) != expected_shared_l2
            ):
                raise RuntimeError(
                    "shared FP4 source weight shape mismatch: "
                    f"l1={tuple(shared_l1_weight.shape)} expected={expected_shared_l1}; "
                    f"l2={tuple(shared_l2_weight.shape)} expected={expected_shared_l2}"
                )
            l1_weight = torch.cat([l1_weight, shared_l1_weight], dim=0).contiguous()
            l2_weight = torch.cat([l2_weight, shared_l2_weight], dim=0).contiguous()
        l1 = _mm.cast_weights_to_fp4(l1_weight)
        l2 = _mm.cast_weights_to_fp4(l2_weight)
        l1_t, l2_t = _mm.transform_weights_for_mega_moe(l1, l2)
        self._register_transformed_eg_layer_weights(layer_id, l1_t, l2_t)

    def register_eg_layer_fp4_packed_weights(
        self,
        layer_id: int,
        w13: torch.Tensor,
        w13_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
    ) -> None:
        """Register already-quantised MXFP4 expert weights (checkpoint layout).

        ``w13`` uint8 [E_local(+shared), 2*I, H/2] (gate rows then up rows, E2M1
        codes, low nibble = even element), ``w13_sf`` uint8 [.., 2*I, H/32] UE8M0
        exponents; ``w2``/``w2_sf`` likewise with [.., H, I/2] / [.., H, I/32].
        The replicated shared expert, when enabled, is the last slot.
        """
        if not self.is_eg:
            raise RuntimeError("register_eg_layer_fp4_packed_weights called on an AG rank")
        if self.use_fp8_weights:
            raise RuntimeError("packed FP4 weights require weight_precision='fp4'")
        n_local = self.num_experts // self.eg_size
        expect = {
            "w13": (n_local, 2 * self.intermediate_size, self.hidden_size // 2),
            "w13_sf": (n_local, 2 * self.intermediate_size, self.hidden_size // 32),
            "w2": (n_local, self.hidden_size, self.intermediate_size // 2),
            "w2_sf": (n_local, self.hidden_size, self.intermediate_size // 32),
        }
        for name, t in (("w13", w13), ("w13_sf", w13_sf), ("w2", w2), ("w2_sf", w2_sf)):
            if tuple(t.shape) != expect[name] or t.dtype != torch.uint8:
                raise RuntimeError(f"{name}: got {tuple(t.shape)} {t.dtype}, expected {expect[name]} uint8")
        from minisgl.models.moe_common import to_megamoe_weights

        l1_t, l2_t = to_megamoe_weights(w13, w13_sf, w2, w2_sf)
        self._register_transformed_eg_layer_weights(layer_id, l1_t, l2_t)

    def _register_transformed_eg_layer_weights(
        self,
        layer_id: int,
        l1_t: tuple[torch.Tensor, torch.Tensor],
        l2_t: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        global_layer_id = int(layer_id)
        if global_layer_id in self._eg_global_to_compact:
            raise RuntimeError(f"MegaMoE weights already registered for layer {global_layer_id}")
        compact_layer_id = len(self._eg_global_to_compact)
        self._eg_global_to_compact[global_layer_id] = compact_layer_id
        self._eg_weights[compact_layer_id] = (l1_t, l2_t)
        # Per-layer weight tensormap blob for the (persistent) EG kernel
        blob = _mm.make_m2n_weight_descs(
            l1_t, l2_t,
            num_experts_per_rank=self.num_experts // self.eg_size,
            hidden=self.hidden_size,
            intermediate_hidden=self.intermediate_size,
        ).cuda()
        self._eg_desc_blobs[compact_layer_id] = blob
        self._eg_l1_ptrs[compact_layer_id] = int(l1_t[0].data_ptr())
        self._eg_l1_nbytes = int(l1_t[0].numel())
        self._desc_blob_all = None  # invalidate the stacked cache

    def _compact_layer_id(self, layer_id: int) -> int:
        try:
            return self._eg_global_to_compact[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"Layer {int(layer_id)} has no registered MegaMoE EG weights"
            ) from exc

    def _desc_tensors(self):
        if self._desc_blob_all is None:
            layers = sorted(self._eg_desc_blobs)
            assert layers == list(range(len(layers))), layers
            self._desc_blob_all = torch.stack(
                [self._eg_desc_blobs[i] for i in layers]).contiguous()
            self._l1_ptrs_all = torch.tensor(
                [self._eg_l1_ptrs[i] for i in layers],
                dtype=torch.int64, device="cuda")
        return self._desc_blob_all, self._l1_ptrs_all

    def _expected_tokens_for_eg(
        self,
        expected_tokens_per_rank: int | None,
        *,
        dual_lane_decode: bool,
    ) -> int:
        expected = int(expected_tokens_per_rank or self.bucket)
        decode_limit = self.bucket * (self.num_lanes if dual_lane_decode else 1)
        if 0 < self.eg_expected_tokens_override and expected <= decode_limit:
            expected = self.eg_expected_tokens_override
        return min(expected, self.bucket)

    def eg_moe(
        self,
        layer_id: int,
        *,
        lane: int = 0,
        expected_tokens_per_rank: int | None = None,
    ) -> None:
        """Fused MoE layer on an EG rank: recv + FP8 experts + send.

        ``expected_tokens_per_rank`` drives the GEMM block/wave heuristics and
        must be the CURRENT step's per-AG-rank token count (e.g. the decode
        graph bucket), NOT the max dispatch bucket — feeding the 8k prefill
        bucket here selects a block_m=192 prefill tiling with one expert per
        wave, several times slower for ~46-token decode steps.
        """
        if not self.is_eg:
            raise RuntimeError("eg_moe called on an AG rank")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=False,
        )
        blob, ptrs = self._desc_tensors()
        lid = self._compact_layer_id(layer_id)
        _mm.mega_moe_m2n_eg(
            blob[lid:lid + 1], ptrs[lid:lid + 1],
            self.buffers[lane % self.num_lanes],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            fast_math=True,
            use_fp8_weights=self.use_fp8_weights,
            activation_clamp=self.activation_clamp,
            activation_alpha=self.activation_alpha,
            activation_up_bias=self.activation_up_bias,
            num_sms=self.eg_num_sms,
            num_prefetch_bytes=self.eg_prefetch_bytes,
        )

    def eg_moe_dual(
        self,
        layer_id: int,
        *,
        expected_tokens_per_rank: int | None = None,
    ) -> None:
        """Dual-lane fused MoE layer on an EG rank: ONE full-width kernel
        serves both microbatch lanes back-to-back (the EG rank is dedicated,
        so the 76-SM lane-co-residency constraint disappears; the GEMMs run
        at full device width and lane 1 reuses lane 0's L2-resident
        weights)."""
        if not self.is_eg:
            raise RuntimeError("eg_moe_dual called on an AG rank")
        if self.num_lanes < 2:
            raise RuntimeError("eg_moe_dual requires two adapter lanes")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=True,
        )
        blob, ptrs = self._desc_tensors()
        lid = self._compact_layer_id(layer_id)
        _mm.mega_moe_m2n_eg(
            blob[lid:lid + 1], ptrs[lid:lid + 1],
            self.buffers[0],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            fast_math=True,
            use_fp8_weights=self.use_fp8_weights,
            activation_clamp=self.activation_clamp,
            activation_alpha=self.activation_alpha,
            activation_up_bias=self.activation_up_bias,
            num_sms=0,  # full device: nothing co-resides on a dedicated EG rank
            num_prefetch_bytes=self.eg_prefetch_bytes,
            sym_buffer1=self.buffers[1],
        )

    def eg_moe_persistent(
        self,
        num_layers: int,
        *,
        num_mb: int = 1,
        expected_tokens_per_rank: int | None = None,
    ) -> None:
        """ONE persistent kernel serves ALL layers and active microbatch lanes.

        The tuned mb=2 path still passes exactly lane0/lane1. For mb>2 the
        same persistent kernel receives lane2/lane3 buffers instead of graphing
        many per-layer EG launches.
        """
        if not self.is_eg:
            raise RuntimeError("eg_moe_persistent called on an AG rank")
        lanes = max(1, min(int(num_mb), self.num_lanes))
        if lanes > 4:
            raise RuntimeError(f"megamoe_m2n persistent supports <=4 lanes, got {lanes}")
        expected = self._expected_tokens_for_eg(
            expected_tokens_per_rank,
            dual_lane_decode=lanes >= 2,
        )
        blob, ptrs = self._desc_tensors()
        assert int(num_layers) == int(blob.shape[0]), (num_layers, blob.shape)
        _mm.mega_moe_m2n_eg(
            blob, ptrs,
            self.buffers[0],
            rank=self.rank,
            expected_num_tokens_per_rank=expected,
            num_experts_per_rank=self.num_experts // self.eg_size,
            l1_weights_nbytes=self._eg_l1_nbytes,
            num_layers=int(num_layers),
            fast_math=True,
            use_fp8_weights=self.use_fp8_weights,
            activation_clamp=self.activation_clamp,
            activation_alpha=self.activation_alpha,
            activation_up_bias=self.activation_up_bias,
            num_sms=0,
            num_prefetch_bytes=self.eg_prefetch_bytes,
            sym_buffer1=self.buffers[1] if lanes >= 2 else None,
            sym_buffer2=self.buffers[2] if lanes >= 3 else None,
            sym_buffer3=self.buffers[3] if lanes >= 4 else None,
        )

    def use_persistent(self, num_mb: int) -> bool:
        mb = int(num_mb)
        return self.use_dual_lane(mb) or (2 < mb <= 4 and self.num_lanes >= mb)

    def use_dual_lane(self, num_mb: int) -> bool:
        return (
            int(num_mb) == 2
            and self.num_lanes >= 2
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        for buf in self.buffers:
            buf.destroy()
        self._eg_weights.clear()
        self._eg_global_to_compact.clear()
        self._y_cache.clear()
        self._destroyed = True

    close = destroy


__all__ = ["MegaMoEM2NAfdAdapter"]
