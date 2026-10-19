"""Plain-torch transcription of the Kimi-K3 text model (``modeling_kimi_linear.py``
of moonshotai/Kimi-K3), one sequence at a time, BF16 weights.

``fla`` kernels are replaced by their mathematical definitions: ShortConvolution
(causal depthwise conv + SiLU with a cached window), the KDA gated delta rule
(``fused_recurrent_kda`` with in-kernel l2norm / safe gate / sigmoid beta) and
FusedRMSNormGated(sigmoid). MLA is computed non-absorbed with eager softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x, w, eps):
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dtype)


def situ_and_mul(gate_up, beta, linear_beta):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (a * up).to(gate_up.dtype)


def apply_attn_res(prefix_sum, block_residual, proj_w, norm_w, eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    score_weight = norm_w.float() * proj_w.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype)


class RefKDA:
    def __init__(self, P, pf, cfg, dev):
        self.P, self.pf, self.cfg = P, pf, cfg
        la = cfg["linear_attn_config"]
        self.H, self.D, self.K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
        self.lb = la.get("gate_lower_bound")
        self.eps = cfg["rms_norm_eps"]
        proj = self.H * self.D
        self.conv_state = [torch.zeros(self.K - 1, proj, device=dev) for _ in range(3)]
        self.S = torch.zeros(self.H, self.D, self.D, device=dev)  # [H, V, K]

    def w(self, n):
        return self.P[f"{self.pf}.{n}"]

    def short_conv(self, x, i, wname):
        # x [L, C]; weight [C, K]
        w = self.w(wname).float()
        xx = torch.cat([self.conv_state[i], x.float()], 0)
        L = x.shape[0]
        out = torch.stack([(xx[t : t + self.K] * w.t()).sum(0) for t in range(L)])
        self.conv_state[i] = xx[-(self.K - 1) :].clone()
        return F.silu(out).to(x.dtype)

    def forward(self, x):  # x [L, dim]
        H, D = self.H, self.D
        q = self.short_conv(F.linear(x, self.w("q_proj.weight")), 0, "q_conv1d.weight")
        k = self.short_conv(F.linear(x, self.w("k_proj.weight")), 1, "k_conv1d.weight")
        v = self.short_conv(F.linear(x, self.w("v_proj.weight")), 2, "v_conv1d.weight")
        g = F.linear(F.linear(x, self.w("f_a_proj.weight")), self.w("f_b_proj.weight"))
        beta = F.linear(x, self.w("b_proj.weight")).float()
        L = x.shape[0]
        q, k, v, g = (t.view(L, H, D) for t in (q, k, v, g))
        A = torch.exp(self.w("A_log").float().view(H, 1))
        dtb = self.w("dt_bias").float().view(H, D)
        out = torch.zeros(L, H, D, device=x.device)
        S = self.S
        for t in range(L):
            qt = q[t].float() / torch.sqrt((q[t].float() ** 2).sum(-1, keepdim=True) + 1e-6) * D**-0.5
            kt = k[t].float() / torch.sqrt((k[t].float() ** 2).sum(-1, keepdim=True) + 1e-6)
            gt = self.lb * torch.sigmoid(A * (g[t].float() + dtb))
            bt = torch.sigmoid(beta[t])
            S = S * torch.exp(gt).unsqueeze(1)
            vt = (v[t].float() - torch.einsum("hvk,hk->hv", S, kt)) * bt.unsqueeze(-1)
            S = S + vt.unsqueeze(-1) * kt.unsqueeze(1)
            out[t] = torch.einsum("hvk,hk->hv", S, qt)
        self.S = S
        o = out.to(x.dtype)
        gate = F.linear(x, self.w("g_proj.weight")).view(L, H, D)
        of = o.float()
        of = of * torch.rsqrt(of.square().mean(-1, keepdim=True) + self.eps) * self.w("o_norm.weight").float()
        of = of * torch.sigmoid(gate.float())
        return F.linear(of.to(x.dtype).view(L, H * D), self.w("o_proj.weight"))


class RefMLA:
    def __init__(self, P, pf, cfg, dev):
        self.P, self.pf, self.cfg = P, pf, cfg
        self.H = cfg["num_attention_heads"]
        self.nope, self.rope, self.v = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        self.lora = cfg["kv_lora_rank"]
        self.eps = cfg["rms_norm_eps"]
        self.k_cache = None
        self.v_cache = None

    def w(self, n):
        return self.P[f"{self.pf}.{n}"]

    def forward(self, x):  # [L, dim]
        L = x.shape[0]
        H, nope, rope, vd = self.H, self.nope, self.rope, self.v
        q = F.linear(rmsnorm(F.linear(x, self.w("q_a_proj.weight")), self.w("q_a_layernorm.weight"), self.eps), self.w("q_b_proj.weight"))
        q = q.view(L, H, nope + rope)
        ckv = F.linear(x, self.w("kv_a_proj_with_mqa.weight"))
        k_pass, k_rot = ckv[:, : self.lora], ckv[:, self.lora :]
        kv = F.linear(rmsnorm(k_pass, self.w("kv_a_layernorm.weight"), self.eps), self.w("kv_b_proj.weight")).view(L, H, nope + vd)
        k_nope, value = kv[..., :nope], kv[..., nope:]
        key = torch.cat([k_nope, k_rot.view(L, 1, rope).expand(L, H, rope)], -1)
        if self.k_cache is None:
            self.k_cache, self.v_cache = key, value
        else:
            self.k_cache = torch.cat([self.k_cache, key], 0)
            self.v_cache = torch.cat([self.v_cache, value], 0)
        Tk = self.k_cache.shape[0]
        scores = torch.einsum("lhd,thd->hlt", q.float(), self.k_cache.float()) * (nope + rope) ** -0.5
        q_pos = torch.arange(Tk - L, Tk, device=x.device)
        mask = torch.arange(Tk, device=x.device)[None, :] > q_pos[:, None]
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = scores.softmax(-1).to(x.dtype)
        o = torch.einsum("hlt,thd->lhd", probs.float(), self.v_cache.float()).to(x.dtype).reshape(L, H * vd)
        o = o * torch.sigmoid(F.linear(x, self.w("g_proj.weight")))
        return F.linear(o, self.w("o_proj.weight"))


class RefKimiK3:
    """``P`` uses checkpoint names without the ``language_model.`` prefix;
    ``experts[l]`` = (w13 [E, 2I, latent], w2 [E, latent, I]) bf16 per MoE layer."""

    def __init__(self, P, cfg, experts, dev):
        self.P, self.cfg = P, cfg
        la = cfg["linear_attn_config"]
        kda1 = set(la["kda_layers"])
        self.attn = []
        for i in range(cfg["num_hidden_layers"]):
            pf = f"model.layers.{i}.self_attn"
            self.attn.append(RefKDA(P, pf, cfg, dev) if (i + 1) in kda1 else RefMLA(P, pf, cfg, dev))
        self.experts = experts
        self.eps = cfg["rms_norm_eps"]
        self.beta = cfg.get("activation_situ_beta") or 1.0
        self.lbeta = cfg.get("activation_situ_linear_beta")

    def mlp(self, x, pf):
        gu = torch.cat([F.linear(x, self.P[f"{pf}.gate_proj.weight"]), F.linear(x, self.P[f"{pf}.up_proj.weight"])], -1)
        return F.linear(situ_and_mul(gu, self.beta, self.lbeta), self.P[f"{pf}.down_proj.weight"])

    def moe(self, x, layer):
        cfg, P = self.cfg, self.P
        pf = f"model.layers.{layer}.block_sparse_moe"
        logits = F.linear(x.float(), P[f"{pf}.gate.weight"].float())
        scores = logits.sigmoid()
        sfc = scores + P[f"{pf}.gate.e_score_correction_bias"].unsqueeze(0)
        _, topk_idx = torch.topk(sfc, k=cfg["num_experts_per_token"], dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_idx)
        if cfg["num_experts_per_token"] > 1 and cfg.get("moe_renormalize", True):
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
        topk_w = topk_w * cfg.get("routed_scaling_factor", 1.0)
        h = F.linear(x, P[f"{pf}.routed_expert_down_proj.weight"])
        w13, w2 = self.experts[layer]
        inter = w13.shape[1] // 2
        out = torch.zeros(h.shape[0], topk_idx.shape[1], h.shape[1], device=x.device, dtype=h.dtype)
        for e in torch.unique(topk_idx).tolist():
            tok, slot = torch.where(topk_idx == e)
            gu = F.linear(h[tok], w13[e])
            a = situ_and_mul(gu, self.beta, self.lbeta)
            out[tok, slot] = F.linear(a, w2[e])
        y = (out.type(topk_w.dtype) * topk_w.unsqueeze(-1)).sum(1).type(out.dtype)
        y = rmsnorm(y, P[f"{pf}.routed_expert_norm.weight"], self.eps)
        y = F.linear(y, P[f"{pf}.routed_expert_up_proj.weight"])
        return y + self.mlp(x, f"{pf}.shared_experts")

    def forward(self, input_ids):
        cfg, P = self.cfg, self.P
        hs = F.embedding(input_ids, P["model.embed_tokens.weight"])
        L = hs.shape[0]
        block = hs.new_zeros(L, 0, hs.shape[1])
        bsz = cfg["attn_res_block_size"]
        for i in range(cfg["num_hidden_layers"]):
            pf = f"model.layers.{i}"
            prefix = hs
            if block.shape[1] > 0:
                hs = apply_attn_res(prefix, block, P[f"{pf}.self_attention_res_proj.weight"], P[f"{pf}.self_attention_res_norm.weight"], self.eps)
            if i % bsz == 0:
                block = torch.cat([block, prefix.unsqueeze(1)], 1)
                prefix = None
            h = rmsnorm(hs, P[f"{pf}.input_layernorm.weight"], self.eps)
            h = self.attn[i].forward(h)
            prefix = h if prefix is None else prefix + h
            hs = apply_attn_res(prefix, block, P[f"{pf}.mlp_res_proj.weight"], P[f"{pf}.mlp_res_norm.weight"], self.eps)
            hs = rmsnorm(hs, P[f"{pf}.post_attention_layernorm.weight"], self.eps)
            if i >= cfg.get("first_k_dense_replace", 0):
                hs = self.moe(hs, i)
            else:
                hs = self.mlp(hs, f"{pf}.mlp")
            prefix = prefix + hs
            hs = prefix
        hs = apply_attn_res(hs, block, P["model.output_attn_res_proj.weight"], P["model.output_attn_res_norm.weight"], self.eps)
        hs = rmsnorm(hs, P["model.norm.weight"], self.eps)
        return F.linear(hs[-1:].float(), P["lm_head.weight"].float())


__all__ = ["RefKimiK3", "apply_attn_res", "situ_and_mul"]
