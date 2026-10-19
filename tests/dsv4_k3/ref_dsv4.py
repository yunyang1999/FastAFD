"""Plain-torch transcription of the official DeepSeek-V4 reference
(``inference/model.py`` + ``inference/kernel.py`` of deepseek-ai/DeepSeek-V4-*),
single rank, one sequence at a time (bsz = 1), BF16 weights.

The TileLang kernels are replaced by torch equivalents with the same semantics:
``act_quant(inplace)`` (FP8 e4m3 quant-dequant, power-of-two scales),
``fp4_act_quant(inplace)``, ``sparse_attn`` (gather + softmax with sink),
``hc_split_sinkhorn`` and the Hadamard rotation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# {layer: LongTensor [T, k]}: use these routed-expert ids instead of the router's
# own top-k (isolates routing flips caused by near-ties under random weights)
FORCE_IDS: dict | None = None
# Emulate the FP8 activation quantisation the official inference applies in
# front of FP8-weight GEMMs (act_quant: block 128, UE8M0 scales) for the
# attention projections and the shared expert. Routed experts are excluded
# (the engine's torch MoE executor does not quantise activations).
ACT_QUANT: bool = False


def _aq(x):
    return act_quant_inplace(x.clone(), 128) if ACT_QUANT else x


# ---------------------------------------------------------------------------
# kernel.py equivalents
# ---------------------------------------------------------------------------


def _pow2_ceil(v):
    return torch.exp2(torch.ceil(torch.log2(v)))


def act_quant_inplace(x: torch.Tensor, block: int) -> torch.Tensor:
    shp = x.shape
    xb = x.float().unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / 448.0)
    q = (xb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    x.copy_(q.flatten(-2).view(shp))
    return x


_FP4_GRID = None


def fp4_act_quant_inplace(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    global _FP4_GRID
    shp = x.shape
    xb = x.float().unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
    s = _pow2_ceil(amax / 6.0)
    y = (xb / s).clamp(-6, 6)
    if _FP4_GRID is None or _FP4_GRID.device != x.device:
        _FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    a = y.abs()
    d = (a.unsqueeze(-1) - _FP4_GRID).abs()
    idx = d.argmin(-1)
    best = d.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    nxt = (idx + 1).clamp(max=7)
    tie = (d.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) == best) & (nxt != idx)
    idx = torch.where(tie & (idx % 2 == 1), nxt, idx)  # ties to even code
    r = _FP4_GRID[idx] * torch.sign(y)
    x.copy_((r * s).flatten(-2).view(shp))
    return x


def hadamard(n: int, device) -> torch.Tensor:
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * n**-0.5


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    return (x.float() @ hadamard(x.shape[-1], x.device)).to(x.dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    # q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k] (-1 invalid)
    b, m, h, d = q.shape
    out = torch.zeros_like(q)
    for bi in range(b):
        for t in range(m):
            idx = topk_idxs[bi, t]
            idx = idx[idx >= 0].long()
            k = kv[bi, idx].float()
            logits = (q[bi, t].float() @ k.t()) * scale  # [h, n]
            mx = torch.maximum(logits.max(-1).values, attn_sink.float())
            p = torch.exp(logits - mx[:, None])
            den = p.sum(-1) + torch.exp(attn_sink.float() - mx)
            out[bi, t] = ((p @ k) / den[:, None]).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# model.py equivalents
# ---------------------------------------------------------------------------


def rmsnorm(x, w, eps):
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(dtype)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn), 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis, inverse=False):
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    y.copy_(torch.view_as_real(xc * freqs_cis).flatten(-2))
    return y


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class RefCompressor:
    def __init__(self, P, prefix, args, ratio, head_dim, rotate, max_seq_len, dev):
        self.P, self.prefix = P, prefix
        self.ratio, self.head_dim, self.rotate = ratio, head_dim, rotate
        self.overlap = ratio == 4
        self.rd = args["rope_dim"]
        self.eps = args["eps"]
        coff = 1 + self.overlap
        self.kv_state = torch.zeros(1, coff * ratio, coff * head_dim, device=dev)
        self.score_state = torch.full((1, coff * ratio, coff * head_dim), float("-inf"), device=dev)
        self.kv_cache = None
        self.freqs_cis = None

    def w(self, n):
        return self.P[f"{self.prefix}.{n}"]

    def overlap_transform(self, t, value=0):
        b, s, _, _ = t.size()
        ratio, d = self.ratio, self.head_dim
        new = t.new_full((b, s, 2 * ratio, d), value)
        new[:, :, ratio:] = t[:, :, :, d:]
        new[:, 1:, :ratio] = t[:, :-1, :, :d]
        return new

    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.ratio, self.overlap, self.head_dim, self.rd
        dtype = x.dtype
        x = x.float()
        kv = F.linear(x, self.w("wkv.weight").float())
        score = F.linear(x, self.w("wgate.weight").float())
        ape = self.w("ape").float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset + remainder] = score[:, cutoff:] + ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:].clone()
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:].clone()
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        kv = rmsnorm(kv.to(dtype), self.w("norm.weight"), self.eps)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant_inplace(kv, 32)
        else:
            act_quant_inplace(kv[..., :-rd], 64)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class RefIndexer:
    def __init__(self, P, prefix, args, max_seq_len, dev):
        self.P, self.prefix = P, prefix
        self.n_heads = args["index_n_heads"]
        self.head_dim = args["index_head_dim"]
        self.rd = args["rope_dim"]
        self.index_topk = args["index_topk"]
        self.softmax_scale = self.head_dim**-0.5
        self.compressor = RefCompressor(P, prefix + ".compressor", args, 4, self.head_dim, True, max_seq_len, dev)
        self.kv_cache = torch.zeros(1, max_seq_len // 4, self.head_dim, device=dev, dtype=torch.bfloat16)
        self.freqs_cis = None

    def forward(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = 4
        rd = self.rd
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = F.linear(_aq(qr), self.P[f"{self.prefix}.wq_b.weight"])
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        fp4_act_quant_inplace(q, 32)
        self.compressor.forward(x, start_pos)
        weights = F.linear(x, self.P[f"{self.prefix}.weights_proj.weight"]) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class RefAttention:
    def __init__(self, P, layer_id, args, max_seq_len, dev):
        self.P = P
        self.prefix = f"layers.{layer_id}.attn"
        self.layer_id = layer_id
        self.args = args
        self.n_heads = args["n_heads"]
        self.head_dim = args["head_dim"]
        self.rd = args["rope_dim"]
        self.n_groups = args["o_groups"]
        self.o_lora = args["o_lora_rank"]
        self.win = args["window"]
        self.ratio = args["ratios"][layer_id]
        self.eps = args["eps"]
        self.scale = self.head_dim**-0.5
        if self.ratio:
            self.compressor = RefCompressor(P, self.prefix + ".compressor", args, self.ratio, self.head_dim, False, max_seq_len, dev)
            self.indexer = RefIndexer(P, self.prefix + ".indexer", args, max_seq_len, dev) if self.ratio == 4 else None
        kv_size = self.win + (max_seq_len // self.ratio if self.ratio else 0)
        self.kv_cache = torch.zeros(1, kv_size, self.head_dim, device=dev, dtype=torch.bfloat16)
        if self.ratio:
            orig, base = args["original_seq_len"], args["compress_rope_theta"]
        else:
            orig, base = 0, args["rope_theta"]
        self.freqs_cis = precompute_freqs_cis(self.rd, max_seq_len, orig, base, args["rope_factor"], args["beta_fast"], args["beta_slow"]).to(dev)

    def w(self, n):
        return self.P[f"{self.prefix}.{n}"]

    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win, ratio, rd = self.win, self.ratio, self.rd
        if self.ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        qr = q = rmsnorm(F.linear(_aq(x), self.w("wq_a.weight")), self.w("q_norm.weight"), self.eps)
        q = F.linear(_aq(q), self.w("wq_b.weight")).unflatten(-1, (self.n_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = F.linear(_aq(x), self.w("wkv.weight"))
        kv = rmsnorm(kv, self.w("kv_norm.weight"), self.eps)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_inplace(kv[..., :-rd], 64)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        if self.ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer.forward(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.ratio:
                if (kv_compress := self.compressor.forward(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.w("attn_sink"), topk_idxs, self.scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.ratio:
                self.compressor.forward(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.w("attn_sink"), topk_idxs, self.scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.w("wo_a").view(self.n_groups, self.o_lora, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return F.linear(_aq(o.flatten(2)), self.w("wo_b.weight"))


class RefMoE:
    def __init__(self, P, layer_id, args, experts_bf16, shared_bf16):
        self.P = P
        self.prefix = f"layers.{layer_id}.ffn"
        self.args = args
        self.hash = layer_id < args["n_hash_layers"]
        self.layer_id = layer_id
        self.experts = experts_bf16  # (w13 [E, 2I, D], w2 [E, D, I])
        self.shared = shared_bf16  # (w13 [2I, D], w2 [D, I])

    def expert(self, x, w13, w2, weights=None, aq=False):
        dtype = x.dtype
        inter = w13.shape[0] // 2
        h = F.linear(_aq(x) if aq else x, w13)
        gate = h[..., :inter].float()
        up = h[..., inter:].float()
        lim = self.args["swiglu_limit"]
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        x = x.to(dtype)
        return F.linear(_aq(x) if aq else x, w2)

    def forward(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, shape[-1])
        scores = F.linear(x.float(), self.P[f"{self.prefix}.gate.weight"].float())
        scores = F.softplus(scores).sqrt()
        original = scores
        if not self.hash:
            scores = scores + self.P[f"{self.prefix}.gate.bias"]
            indices = scores.topk(self.args["topk"], dim=-1)[1]
            if FORCE_IDS is not None and self.layer_id in FORCE_IDS:
                indices = FORCE_IDS[self.layer_id].to(x.device).long()
        else:
            indices = self.P[f"{self.prefix}.gate.tid2eid"][input_ids.flatten()].long()
        weights = original.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.args["rsf"]
        y = torch.zeros_like(x, dtype=torch.float32)
        w13, w2 = self.experts
        for i in torch.unique(indices).tolist():
            idx, top = torch.where(indices == i)
            y[idx] += self.expert(x[idx], w13[i], w2[i], weights[idx, top, None])
        y += self.expert(x, self.shared[0], self.shared[1], aq=True)
        return y.type_as(x).view(shape)


class RefDSV4:
    """One request at a time; call prefill(tokens) then decode(token) repeatedly."""

    def __init__(self, P: dict, args: dict, experts: list, shared: list, max_seq_len: int, dev):
        self.P, self.args = P, args
        self.layers = []
        for i in range(args["n_layers"]):
            self.layers.append((RefAttention(P, i, args, max_seq_len, dev), RefMoE(P, i, args, experts[i], shared[i])))
        self.pos = 0

    def hc_pre(self, x, fn, scale, base):
        a = self.args
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + a["eps"])
        mixes = F.linear(x, fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, a["hc_mult"], a["hc_iters"], a["hc_eps"])
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        a, P = self.args, self.P
        h = F.embedding(input_ids, P["embed.weight"])
        h = h.unsqueeze(2).repeat(1, 1, a["hc_mult"], 1)
        for i, (attn, moe) in enumerate(self.layers):
            pf = f"layers.{i}"
            residual = h
            x, post, comb = self.hc_pre(h, P[f"{pf}.hc_attn_fn"], P[f"{pf}.hc_attn_scale"], P[f"{pf}.hc_attn_base"])
            x = rmsnorm(x, P[f"{pf}.attn_norm.weight"], a["eps"])
            x = attn.forward(x, start_pos)
            h = self.hc_post(x, residual, post, comb)
            residual = h
            x, post, comb = self.hc_pre(h, P[f"{pf}.hc_ffn_fn"], P[f"{pf}.hc_ffn_scale"], P[f"{pf}.hc_ffn_base"])
            x = rmsnorm(x, P[f"{pf}.ffn_norm.weight"], a["eps"])
            x = moe.forward(x, input_ids)
            h = self.hc_post(x, residual, post, comb)
        # head
        shape, dtype = h.size(), h.dtype
        x = h.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + a["eps"])
        mixes = F.linear(x, P["hc_head_fn"]) * rsqrt
        pre = torch.sigmoid(mixes * P["hc_head_scale"] + P["hc_head_base"]) + a["hc_eps"]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2).to(dtype)
        y = rmsnorm(y, P["norm.weight"], a["eps"])
        return F.linear(y[:, -1].float(), P["head.weight"].float())


__all__ = ["RefDSV4", "act_quant_inplace", "fp4_act_quant_inplace", "hc_split_sinkhorn", "sparse_attn"]
