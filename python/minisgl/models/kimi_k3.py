"""Kimi-K3 (KimiLinear text model) for minisgl.

Architecture (``modeling_kimi_linear.py``): 93 layers = KDA (gated delta-rule
linear attention with short convs) and MLA (NoPE, sigmoid output gate) layers,
attention residuals (``attn_res_block_size`` block summaries mixed by a softmax
over RMS-normalised candidates), layer 0 dense SiTU MLP, LatentMoE elsewhere:
router on the 7168 hidden state (sigmoid + correction bias, renormalised top-16),
routed experts in a 3584-d latent space (down-proj, experts, RMSNorm, up-proj),
plus 2 shared SiTU experts in the hidden space.

Module names follow the checkpoint (``language_model.`` prefix stripped).
The same tree serves the colocated engine and the AFD attention stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.utils import debug_taps, div_even, nvtx_annotate

from .base import BaseLLMModel
from .deepseek_v4 import LinearReplicatedQ
from .moe_common import ActivationSpec, ColocatedMegaMoE, FP4RoutedExperts, moe_exec_backend, torch_ref_moe

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class K3Args:
    dim: int
    n_layers: int
    vocab: int
    eps: float
    n_heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    kda_layers: tuple[int, ...]
    kda_heads: int
    kda_dim: int
    conv_k: int
    lower_bound: float
    dense_inter: int
    first_dense: int
    n_routed: int
    topk: int
    moe_inter: int
    latent: int
    n_shared: int
    rsf: float
    renorm: bool
    situ_beta: float
    situ_lbeta: float | None
    res_block: int

    @classmethod
    def from_config(cls, mc: "ModelConfig") -> "K3Args":
        d = mc.hf_text_config
        la = d["linear_attn_config"]
        L = int(mc.num_layers)
        kda1 = {int(x) for x in la["kda_layers"]}
        return cls(
            dim=int(d["hidden_size"]),
            n_layers=L,
            vocab=int(d["vocab_size"]),
            eps=float(d["rms_norm_eps"]),
            n_heads=int(d["num_attention_heads"]),
            q_lora=int(d["q_lora_rank"]),
            kv_lora=int(d["kv_lora_rank"]),
            qk_nope=int(d["qk_nope_head_dim"]),
            qk_rope=int(d["qk_rope_head_dim"]),
            v_head=int(d["v_head_dim"]),
            kda_layers=tuple(i for i in range(L) if (i + 1) in kda1),
            kda_heads=int(la["num_heads"]),
            kda_dim=int(la["head_dim"]),
            conv_k=int(la["short_conv_kernel_size"]),
            lower_bound=float(la.get("gate_lower_bound", -5.0)),
            dense_inter=int(d["intermediate_size"]),
            first_dense=int(d.get("first_k_dense_replace", 0)),
            n_routed=int(d["num_experts"]),
            topk=int(d["num_experts_per_token"]),
            moe_inter=int(d["moe_intermediate_size"]),
            latent=int(d.get("routed_expert_hidden_size") or d["hidden_size"]),
            n_shared=int(d.get("num_shared_experts") or 0),
            rsf=float(d.get("routed_scaling_factor", 1.0)),
            renorm=bool(d.get("moe_renormalize", True)),
            situ_beta=float(d.get("activation_situ_beta") or 1.0),
            situ_lbeta=(float(d["activation_situ_linear_beta"]) if d.get("activation_situ_linear_beta") else None),
            res_block=int(d.get("attn_res_block_size") or 0),
        )

    def is_kda(self, layer: int) -> bool:
        return layer in self.kda_layers

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_dense

    @property
    def max_blocks(self) -> int:
        return (self.n_layers + self.res_block - 1) // self.res_block if self.res_block else 0


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    if get_tp_info().size > 1:
        return DistributedCommunicator().all_reduce(x)
    return x


class SituMLP(BaseOP):
    """gate_up (column parallel, merged) -> SiTU*mul -> down (row parallel)."""

    def __init__(self, args: K3Args, inter: int):
        self.gate_up_proj = LinearColParallelMerged(args.dim, [inter, inter], has_bias=False)
        self.down_proj = LinearRowParallel(inter, args.dim, has_bias=False)
        self._beta = args.situ_beta
        self._lbeta = args.situ_lbeta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel.kimi_triton import situ_and_mul

        return self.down_proj.forward(situ_and_mul(self.gate_up_proj.forward(x), self._beta, self._lbeta))


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


class KimiDeltaAttention(BaseOP):
    def __init__(self, args: K3Args, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.h_local = div_even(args.kda_heads, tp.size)
        D = args.kda_dim
        proj = args.kda_heads * D
        self.qkv_proj = LinearColParallelMerged(args.dim, [proj, proj, proj], has_bias=False)
        self.conv_weight = torch.empty(3 * self.h_local * D, args.conv_k, dtype=torch.float32)
        self.f_a_proj = LinearReplicatedQ(args.dim, D)
        self.f_b_proj = LinearColParallelMerged(D, [proj], has_bias=False)
        self.b_proj = LinearColParallelMerged(args.dim, [args.kda_heads], has_bias=False)
        self.g_proj = LinearColParallelMerged(args.dim, [proj], has_bias=False)
        self.A_log = torch.empty(self.h_local, dtype=torch.float32)
        self.dt_bias = torch.empty(self.h_local * D, dtype=torch.float32)
        self.o_norm = RMSNorm(D, eps=args.eps)
        self.o_proj = LinearRowParallel(proj, args.dim, has_bias=False)
        self._D = D
        self._lb = args.lower_bound
        self._eps = args.eps

    @nvtx_annotate("KDA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel.kda_triton import gated_rmsnorm, kda_recurrent, short_conv_decode, short_conv_varlen

        ctx = get_global_ctx()
        md = ctx.batch.attn_metadata
        ssm, conv = ctx.attn_backend.pool.kda_state(self.layer_id)
        T = x.shape[0]
        H, D = self.h_local, self._D
        qkv = self.qkv_proj.forward(x)
        g_raw = self.f_b_proj.forward(self.f_a_proj.forward(x))
        b_raw = self.b_proj.forward(x)
        gate = self.g_proj.forward(x)
        if md.is_decode:
            y = short_conv_decode(qkv, self.conv_weight, conv, md.state_idx)
        else:
            y = short_conv_varlen(qkv, self.conv_weight, conv, md.cu_seqlens_q, md.state_idx, md.has_init, md.max_seqlen_q)
        q, k, v = y[:, : H * D], y[:, H * D : 2 * H * D], y[:, 2 * H * D :]
        o = kda_recurrent(
            q, k, v, g_raw, b_raw, self.A_log, self.dt_bias, ssm, md.cu_seqlens_q, md.state_idx, md.has_init,
            num_heads=H, head_dim=D, scale=D**-0.5, lower_bound=self._lb,
        )
        o = gated_rmsnorm(o.view(T * H, D), gate.view(T * H, D), self.o_norm.weight, self._eps)
        return self.o_proj.forward(o.view(T, H * D))


class KimiMLAAttention(BaseOP):
    def __init__(self, args: K3Args, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.h_local = div_even(args.n_heads, tp.size)
        self.q_a_proj = LinearReplicatedQ(args.dim, args.q_lora)
        self.q_a_layernorm = RMSNorm(args.q_lora, eps=args.eps)
        qk = args.qk_nope + args.qk_rope
        self.q_b_proj = LinearColParallelMerged(args.q_lora, [args.n_heads * qk], has_bias=False)
        self.kv_a_proj_with_mqa = LinearReplicatedQ(args.dim, args.kv_lora + args.qk_rope)
        self.kv_a_layernorm = RMSNorm(args.kv_lora, eps=args.eps)
        self.kv_b_proj = LinearColParallelMerged(args.kv_lora, [args.n_heads * (args.qk_nope + args.v_head)], has_bias=False)
        self.g_proj = LinearColParallelMerged(args.dim, [args.n_heads * args.v_head], has_bias=False)
        self.o_proj = LinearRowParallel(args.n_heads * args.v_head, args.dim, has_bias=False)
        self._a = args
        self._w_uk: torch.Tensor | None = None
        self._w_uv_t: torch.Tensor | None = None

    def _absorb(self) -> None:
        a = self._a
        w = self.kv_b_proj.weight.view(self.h_local, a.qk_nope + a.v_head, a.kv_lora)
        self._w_uk = w[:, : a.qk_nope, :].contiguous()  # [H, nope, lora]
        self._w_uv_t = w[:, a.qk_nope :, :].transpose(1, 2).contiguous()  # [H, lora, v]

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self._a
        ctx = get_global_ctx()
        backend = ctx.attn_backend
        T = x.shape[0]
        H = self.h_local
        if self._w_uk is None:
            self._absorb()
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x))).view(T, H, a.qk_nope + a.qk_rope)
        q_nope, q_rope = q[..., : a.qk_nope], q[..., a.qk_nope :]
        ckv_kr = self.kv_a_proj_with_mqa.forward(x)
        ckv = self.kv_a_layernorm.forward(ckv_kr[:, : a.kv_lora].contiguous())
        cache = backend.pool.mla_cache(self.layer_id)
        out_loc = ctx.batch.out_loc.long()
        cache[out_loc, : a.kv_lora] = ckv
        cache[out_loc, a.kv_lora :] = ckv_kr[:, a.kv_lora :]
        q_abs = torch.bmm(q_nope.transpose(0, 1), self._w_uk).transpose(0, 1)  # [T, H, lora]
        o_lat = backend.mla_attn(self.layer_id, q_abs.contiguous(), q_rope.contiguous())  # [T, H, lora]
        o = torch.bmm(o_lat.transpose(0, 1), self._w_uv_t).transpose(0, 1)  # [T, H, v]
        o = o.reshape(T, H * a.v_head) * torch.sigmoid(self.g_proj.forward(x))
        return self.o_proj.forward(o)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class K3TopK(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    routed_scaling_factor: float = 1.0
    skip_moe: bool = False
    local_output: torch.Tensor | None = None
    shared_output: torch.Tensor | None = None
    dispatch_fp8: bool = True
    external_quant: bool = False
    dispatch_input: torch.Tensor | None = None
    post_fn: object | None = None


class KimiMoEGate(BaseOP):
    def __init__(self, args: K3Args):
        self.weight = torch.empty(args.n_routed, args.dim, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(args.n_routed, dtype=torch.float32)
        self._topk = args.topk
        self._renorm = args.renorm
        self._rsf = args.rsf

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(F.linear(x.float(), self.weight))
        ids = torch.topk(scores + self.e_score_correction_bias, self._topk, dim=-1, sorted=False).indices
        w = scores.gather(1, ids)
        if self._topk > 1 and self._renorm:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return ids.to(torch.int32), w * self._rsf


class KimiLatentMoE(BaseOP):
    def __init__(self, args: K3Args, *, with_experts: bool = True):
        tp = get_tp_info()
        self.gate = KimiMoEGate(args)
        self.routed_expert_down_proj = LinearReplicatedQ(args.dim, args.latent)
        self.routed_expert_norm = RMSNorm(args.latent, eps=args.eps)
        self.routed_expert_up_proj = LinearReplicatedQ(args.latent, args.dim)
        self.shared_experts = SituMLP(args, args.moe_inter * args.n_shared) if args.n_shared else None
        self.n_local = args.n_routed // tp.size
        self.experts = FP4RoutedExperts(self.n_local, args.latent, args.moe_inter) if with_experts else None
        self._args = args
        self._offset = tp.rank * self.n_local
        self._act = ActivationSpec("situ", beta=args.situ_beta, linear_beta=args.situ_lbeta)

    def post_combine(self, y_latent: torch.Tensor, shared: torch.Tensor | None) -> torch.Tensor:
        y = self.routed_expert_up_proj.forward(self.routed_expert_norm.forward(y_latent))
        if shared is not None:
            y = y + shared
        return y

    def forward(self, x: torch.Tensor, executor) -> torch.Tensor:
        ids, w = self.gate.forward(x)
        latent = self.routed_expert_down_proj.forward(x)
        shared = self.shared_experts.forward(x) if self.shared_experts is not None else None
        if executor is not None:
            y = executor(latent, ids, w, self.experts.megamoe())
        else:
            w13, w2 = self.experts.bf16()
            y = torch_ref_moe(latent, ids, w, w13, w2, self._act, expert_offset=self._offset)
            y = _all_reduce(y)
        return self.post_combine(y, shared)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------


@dataclass
class K3Residual:
    """Attention-residual state carried between layers."""

    buf: torch.Tensor  # [T, max_blocks + 1, D]; slots [0, nb) are block summaries
    nb: int
    prefix: torch.Tensor | None  # running sum within the current block (None at block start)


class KimiDecoderLayer(BaseOP):
    def __init__(self, args: K3Args, layer_id: int, *, afd_attention_side: bool = False):
        self.layer_id = layer_id
        self.is_kda = args.is_kda(layer_id)
        self.self_attn = KimiDeltaAttention(args, layer_id) if self.is_kda else KimiMLAAttention(args, layer_id)
        if args.is_moe(layer_id):
            self.block_sparse_moe = KimiLatentMoE(args, with_experts=not afd_attention_side)
            self.mlp = None
        else:
            self.mlp = SituMLP(args, args.dense_inter)
            self.block_sparse_moe = None
        self.input_layernorm = RMSNorm(args.dim, eps=args.eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.eps)
        self.self_attention_res_norm = RMSNorm(args.dim, eps=args.eps)
        self.mlp_res_norm = RMSNorm(args.dim, eps=args.eps)
        self.self_attention_res_proj = torch.empty(1, args.dim)
        self.mlp_res_proj = torch.empty(1, args.dim)
        self._a = args
        self._w_attn: torch.Tensor | None = None
        self._w_mlp: torch.Tensor | None = None

    def _res_weights(self):
        if self._w_attn is None:
            self._w_attn = (self.self_attention_res_norm.weight.float() * self.self_attention_res_proj.view(-1).float()).contiguous()
            self._w_mlp = (self.mlp_res_norm.weight.float() * self.mlp_res_proj.view(-1).float()).contiguous()
        return self._w_attn, self._w_mlp

    def attention_part(self, res: K3Residual) -> tuple[torch.Tensor, K3Residual]:
        """From the layer input state up to the MLP input (post norm)."""
        from minisgl.kernel.kimi_triton import attn_res

        w_attn, w_mlp = self._res_weights()
        eps = self._a.eps
        prefix = res.prefix
        assert prefix is not None
        if res.nb > 0:
            res.buf[:, res.nb].copy_(prefix)
            hidden = attn_res(res.buf, res.nb + 1, w_attn, eps)
        else:
            hidden = prefix
        if self._a.res_block and self.layer_id % self._a.res_block == 0:
            res.buf[:, res.nb].copy_(prefix)
            res.nb += 1
            prefix = None
        h = self.input_layernorm.forward(hidden)
        debug_taps.tap(f"L{self.layer_id}.attn_in", h)
        h = self.self_attn.forward(h)
        debug_taps.tap(f"L{self.layer_id}.attn_out", h)
        prefix = h if prefix is None else prefix + h
        res.buf[:, res.nb].copy_(prefix)
        hidden = attn_res(res.buf, res.nb + 1, w_mlp, eps)
        res.prefix = prefix
        return self.post_attention_layernorm.forward(hidden), res

    @nvtx_annotate("K3Layer_{}", layer_id_field="layer_id")
    def forward(self, res: K3Residual, executor) -> K3Residual:
        h, res = self.attention_part(res)
        debug_taps.tap(f"L{self.layer_id}.moe_in", h)
        if self.block_sparse_moe is not None:
            m = self.block_sparse_moe.forward(h, executor)
        else:
            m = self.mlp.forward(h)
        debug_taps.tap(f"L{self.layer_id}.moe_out", m)
        res.prefix = res.prefix + m
        return res


class KimiK3Model(BaseOP):
    def __init__(self, config: "ModelConfig", *, afd_attention_side: bool = False):
        args = K3Args.from_config(config)
        self.embed_tokens = VocabParallelEmbedding(args.vocab, args.dim)
        self.layers = OPList([KimiDecoderLayer(args, i, afd_attention_side=afd_attention_side) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output_attn_res_norm = RMSNorm(args.dim, eps=args.eps)
        self.output_attn_res_proj = torch.empty(1, args.dim)
        self._args = args
        self._w_out: torch.Tensor | None = None

    def new_residual(self, h: torch.Tensor) -> K3Residual:
        a = self._args
        buf = torch.empty((h.shape[0], a.max_blocks + 1, a.dim), dtype=h.dtype, device=h.device)
        return K3Residual(buf=buf, nb=0, prefix=h)

    def final_hidden(self, res: K3Residual) -> torch.Tensor:
        from minisgl.kernel.kimi_triton import attn_res

        if self._w_out is None:
            self._w_out = (self.output_attn_res_norm.weight.float() * self.output_attn_res_proj.view(-1).float()).contiguous()
        res.buf[:, res.nb].copy_(res.prefix)
        h = attn_res(res.buf, res.nb + 1, self._w_out, self._args.eps)
        return self.norm.forward(h)

    def forward(self, input_ids: torch.Tensor, executor) -> torch.Tensor:
        res = self.new_residual(self.embed_tokens.forward(input_ids))
        for layer in self.layers.op_list:
            res = layer.forward(res, executor)
        return self.final_hidden(res)


class KimiK3ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = KimiK3Model(config)
        self.lm_head = ParallelLMHead(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self._config = config
        self._executor: ColocatedMegaMoE | None = None
        self.num_routed_experts = self.model._args.n_routed
        super().__init__()

    def post_load(self, *, max_tokens_per_rank: int) -> None:
        a = self.model._args
        for layer in self.model.layers.op_list:
            if not layer.is_kda:
                layer.self_attn._absorb()
        if moe_exec_backend() == "megamoe":
            self._executor = ColocatedMegaMoE(
                num_routed=a.n_routed,
                hidden=a.latent,
                intermediate=a.moe_inter,
                top_k=a.topk,
                num_shared=0,
                routed_scaling_factor=1.0,  # rsf already folded into the router weights
                activation_clamp=None,
                max_tokens_per_rank=max_tokens_per_rank,
            )
            for layer in self.model.layers.op_list:
                if layer.block_sparse_moe is not None:
                    layer.block_sparse_moe.experts.megamoe()
                    layer.block_sparse_moe.experts.drop_packed()
            torch.cuda.empty_cache()
        else:
            for layer in self.model.layers.op_list:
                if layer.block_sparse_moe is not None:
                    layer.block_sparse_moe.experts.bf16()
                    layer.block_sparse_moe.experts.drop_packed()
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        h = self.model.forward(ctx.batch.input_ids, self._executor)
        return self.lm_head.forward(h)


__all__ = ["KimiK3ForCausalLM", "KimiK3Model", "K3Args", "K3TopK", "K3Residual"]
