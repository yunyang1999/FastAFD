"""Unit tests for the DSV4 / KDA Triton kernels against plain-torch references.

Run on a GPU node:  python tests/dsv4_k3/test_kernels.py
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from minisgl.kernel import dsv4_triton as dk
from minisgl.kernel import kda_triton as kk

DEV = torch.device("cuda")
RESULTS: list[tuple[str, float, float, bool]] = []


def report(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    err = (got.float() - ref.float()).abs()
    mx, mean = err.max().item(), err.mean().item()
    ok = bool(mx <= tol)
    RESULTS.append((name, mx, mean, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={mx:.3e} mean_abs={mean:.3e} (tol {tol:g})")


# ---------------------------------------------------------------------------
# references (official DeepSeek-V4 inference/kernel.py semantics, plain torch)
# ---------------------------------------------------------------------------


def ref_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def ref_rope(x, cos, sin, pos, inverse=False):
    # x [..., R] interleaved pairs; cos/sin [P, R/2]
    xf = x.float().unflatten(-1, (-1, 2))
    c = cos[pos.long()]
    s = sin[pos.long()]
    while c.dim() < xf.dim() - 1:
        c = c.unsqueeze(-2)
        s = s.unsqueeze(-2)
    if inverse:
        s = -s
    x0, x1 = xf[..., 0], xf[..., 1]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)


def pow2_ceil(v):
    return torch.exp2(torch.ceil(torch.log2(v)))


def ref_fp8_sim(x, block):
    shp = x.shape
    xb = x.float().unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / 448.0)
    q = (xb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    return q.flatten(-2).view(shp)


def ref_fp4_sim(x, block=32):
    shp = x.shape
    xb = x.float().unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
    s = pow2_ceil(amax / 6.0)
    y = (xb / s).clamp(-6, 6)
    a = y.abs()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    # nearest, ties to even code
    d = (a.unsqueeze(-1) - grid).abs()
    idx = d.argmin(-1)
    # tie handling: if two codes are equally near choose the even one
    best = d.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    nxt = (idx + 1).clamp(max=7)
    tie = (d.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) == best) & (nxt != idx)
    idx = torch.where(tie & (idx % 2 == 1), nxt, idx)
    r = grid[idx] * torch.sign(y)
    return (r * s).flatten(-2).view(shp)


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * n**-0.5


def make_cos_sin(P, R, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, R, 2, dtype=torch.float32) / R))
    t = torch.arange(P, dtype=torch.float32)
    f = torch.outer(t, freqs)
    return f.cos().to(DEV), f.sin().to(DEV)


# ---------------------------------------------------------------------------
# DSV4 kernels
# ---------------------------------------------------------------------------


def test_hc():
    torch.manual_seed(0)
    T, HC, D = 37, 4, 4096
    x = (torch.randn(T, HC, D, device=DEV) * 2).bfloat16()
    fn = torch.randn((2 + HC) * HC, HC * D, device=DEV) * 0.02
    scale = torch.rand(3, device=DEV) + 0.5
    base = torch.randn((2 + HC) * HC, device=DEV) * 0.5
    eps = 1e-6
    mixes = dk.hc_prenorm_mixes(x, fn, eps)
    y, post, comb = dk.hc_pre_fused(x, mixes, scale, base, sinkhorn_iters=20, eps=eps)
    pre_r, post_r, comb_r = ref_sinkhorn(mixes, scale, base, HC, 20, eps)
    y_r = (pre_r.unsqueeze(-1) * x.float()).sum(1)
    report("hc_pre.y", y, y_r.bfloat16(), 0.02 * y_r.abs().max().item() / 8)
    report("hc_pre.post", post, post_r, 1e-5)
    report("hc_pre.comb", comb, comb_r, 1e-5)
    o = torch.randn(T, D, device=DEV).bfloat16()
    new = dk.hc_post_fused(o, x, post, comb)
    new_r = post_r.unsqueeze(-1) * o.float().unsqueeze(1) + (comb_r.unsqueeze(-1) * x.float().unsqueeze(2)).sum(1)
    report("hc_post", new, new_r.bfloat16(), 0.05)
    hfn = torch.randn(HC, HC * D, device=DEV) * 0.02
    hscale = torch.rand(1, device=DEV) + 0.5
    hbase = torch.randn(HC, device=DEV) * 0.5
    mh = dk.hc_prenorm_mixes(x, hfn, eps)
    yh = dk.hc_head_fused(x, mh, hscale, hbase, eps)
    pre_h = torch.sigmoid(mh * hscale + hbase) + eps
    yh_r = (pre_h.unsqueeze(-1) * x.float()).sum(1)
    report("hc_head", yh, yh_r.bfloat16(), 0.05)


def test_q_kv_rope():
    torch.manual_seed(1)
    T, H, D, R = 19, 8, 512, 64
    cos, sin = make_cos_sin(4096, R, 160000.0)
    pos = torch.randint(0, 4096, (T,), device=DEV, dtype=torch.int32)
    q = (torch.randn(T, H, D, device=DEV) * 3).bfloat16()
    q_ref = q.float()
    q_ref = (q_ref * torch.rsqrt(q_ref.square().mean(-1, keepdim=True) + 1e-6)).bfloat16().float()
    q_ref[..., -R:] = ref_rope(q_ref[..., -R:], cos, sin, pos)
    q2 = q.clone()
    dk.q_norm_rope_(q2, pos, cos, sin, eps=1e-6, rope_dim=R)
    report("q_norm_rope", q2, q_ref.bfloat16(), 0.03)

    # indexer q: rope + hadamard + fp4 sim (no norm)
    Hi, Di = 16, 128
    had = hadamard(Di, DEV)
    qi = torch.randn(T, Hi, Di, device=DEV).bfloat16()
    qi_ref = qi.float().clone()
    qi_ref[..., -R:] = ref_rope(qi_ref[..., -R:], cos, sin, pos)
    qi_ref = qi_ref.bfloat16().float() @ had
    qi_ref = ref_fp4_sim(qi_ref.bfloat16().float())
    qi2 = qi.clone()
    dk.q_norm_rope_(qi2, pos, cos, sin, eps=1e-6, rope_dim=R, do_norm=False, hadamard=had)
    mism = (qi2.float() != qi_ref.bfloat16().float()).float().mean().item()
    print(f"     indexer-q fp4 code mismatch fraction {mism:.4f}")
    report("indexer_q_rope_had_fp4", qi2, qi_ref.bfloat16(), 0.75)

    kv = torch.randn(T, D, device=DEV).bfloat16()
    w = torch.rand(D, device=DEV) + 0.5
    out = torch.zeros(64, 576, device=DEV, dtype=torch.bfloat16)
    slots = torch.randperm(64, device=DEV)[:T]
    dk.kv_norm_rope_store(kv, w, pos, cos, sin, out, slots, eps=1e-6, rope_dim=R, quant=1)
    x = kv.float()
    x = (w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))).bfloat16().float()
    x[:, -R:] = ref_rope(x[:, -R:], cos, sin, pos)
    x = x.bfloat16().float()
    x[:, :-R] = ref_fp8_sim(x[:, :-R], 64)
    report("kv_norm_rope_fp8sim", out[slots, :512], x.bfloat16(), 0.02)
    report("kv_tail_zero", out[slots, 512:], torch.zeros_like(out[slots, 512:]), 0.0)

    o = torch.randn(T, H, D, device=DEV).bfloat16()
    o_ref = o.float().clone()
    o_ref[..., -R:] = ref_rope(o_ref[..., -R:], cos, sin, pos, inverse=True)
    o2 = o.clone()
    dk.inverse_rope_(o2, pos, cos, sin, R)
    report("inverse_rope", o2, o_ref.bfloat16(), 0.02)


def ref_compress_decode(kv, score, ape, pos, st_kv, st_sc, ratio, overlap, d):
    # mirrors official Compressor.forward decode path (start_pos > 0) per request
    B = kv.shape[0]
    pooled = torch.zeros(B, d, device=kv.device)
    complete = torch.zeros(B, dtype=torch.bool, device=kv.device)
    for b in range(B):
        p = int(pos[b])
        sc = score[b] + ape[p % ratio]
        if overlap:
            st_kv[b, ratio + p % ratio] = kv[b]
            st_sc[b, ratio + p % ratio] = sc
            if (p + 1) % ratio == 0:
                kvs = torch.cat([st_kv[b, :ratio, :d], st_kv[b, ratio:, d:]], 0)
                scs = torch.cat([st_sc[b, :ratio, :d], st_sc[b, ratio:, d:]], 0)
                pooled[b] = (kvs * scs.softmax(0)).sum(0)
                st_kv[b, :ratio] = st_kv[b, ratio:].clone()
                st_sc[b, :ratio] = st_sc[b, ratio:].clone()
                complete[b] = True
        else:
            st_kv[b, p % ratio] = kv[b]
            st_sc[b, p % ratio] = sc
            if (p + 1) % ratio == 0:
                pooled[b] = (st_kv[b] * st_sc[b].softmax(0)).sum(0)
                complete[b] = True
    return pooled, complete


def test_compressor_decode():
    for ratio, overlap, d in ((4, True, 512), (128, False, 512), (4, True, 128)):
        torch.manual_seed(2)
        coff = 2 if overlap else 1
        B, R = 6, 8
        CD = coff * d
        rows = coff * ratio
        st_kv = torch.randn(R, rows, CD, device=DEV)
        st_sc = torch.randn(R, rows, CD, device=DEV)
        tidx = torch.randperm(R, device=DEV)[:B].int()
        ape = torch.randn(ratio, CD, device=DEV)
        ref_kv, ref_sc = st_kv[tidx.long()].clone(), st_sc[tidx.long()].clone()
        ok_all = True
        for step in range(2 * ratio + 1):
            pos = torch.tensor([step + 3 * b for b in range(B)], device=DEV, dtype=torch.int32)
            kv = torch.randn(B, CD, device=DEV)
            sc = torch.randn(B, CD, device=DEV)
            pooled = dk.compressor_decode_step(
                kv, sc, ape, pos, tidx, st_kv, st_sc, ratio=ratio, overlap=overlap, head_dim=d
            )
            pr, comp = ref_compress_decode(kv, sc, ape, pos, ref_kv, ref_sc, ratio, overlap, d)
            if comp.any():
                # pooled is stored as bf16 (the reference rounds it before its norm)
                e = ((pooled[comp].float() - pr[comp]).abs() / (pr[comp].abs() + 1e-3)).max().item()
                if e > 1e-2:
                    ok_all = False
                    print(f"   step {step} pooled err {e}")
        # with overlap only the first half of the previous-group rows is ever read
        got = st_kv[tidx.long()]
        if overlap:
            got, ref_cmp = torch.cat([got[:, :ratio, :d].flatten(1), got[:, ratio:].flatten(1)], 1), torch.cat(
                [ref_kv[:, :ratio, :d].flatten(1), ref_kv[:, ratio:].flatten(1)], 1)
        else:
            ref_cmp = ref_kv
        report(f"compressor_decode r{ratio} ov{int(overlap)} d{d} state", got, ref_cmp, 1e-5)
        RESULTS.append((f"compressor_decode r{ratio} pooled", 0.0, 0.0, ok_all))
        print(f"{'PASS' if ok_all else 'FAIL'} compressor_decode r{ratio} pooled")


# ---------------------------------------------------------------------------
# KDA kernels
# ---------------------------------------------------------------------------


def ref_short_conv(x_seq, w, state0):
    # x_seq [L, D], w [D, KW], state0 [D, KW-1] (or zeros)
    KW = w.shape[1]
    xx = torch.cat([state0.t().float(), x_seq.float()], 0)  # [(KW-1)+L, D]
    L = x_seq.shape[0]
    out = torch.zeros(L, x_seq.shape[1], device=x_seq.device)
    for t in range(L):
        win = xx[t : t + KW]  # [KW, D]
        out[t] = (win * w.t().float()).sum(0)
    new_state = xx[-(KW - 1) :].t()
    return F.silu(out), new_state


def test_short_conv():
    torch.manual_seed(3)
    D, KW, R = 384, 4, 6
    w = torch.randn(D, KW, device=DEV) * 0.5
    state = (torch.randn(R, D, KW - 1, device=DEV)).bfloat16()
    lens = [5, 1, 2, 17]
    init = [1, 0, 1, 1]
    sidx = [3, 0, 5, 1]
    x = torch.randn(sum(lens), D, device=DEV).bfloat16()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    ref_y = []
    ref_states = state.clone()
    off = 0
    for L, ini, si in zip(lens, init, sidx):
        st0 = state[si].float() if ini else torch.zeros(D, KW - 1, device=DEV)
        y, ns = ref_short_conv(x[off : off + L], w, st0)
        ref_y.append(y)
        ref_states[si] = ns.bfloat16()
        off += L
    st = state.clone()
    y = kk.short_conv_varlen(
        x, w, st, cu, torch.tensor(sidx, device=DEV, dtype=torch.int32),
        torch.tensor(init, device=DEV, dtype=torch.int32), max(lens),
    )
    report("short_conv_varlen.y", y, torch.cat(ref_y).bfloat16(), 0.03)
    report("short_conv_varlen.state", st, ref_states, 0.0)
    # decode
    st2 = st.clone()
    xd = torch.randn(4, D, device=DEV).bfloat16()
    yd = kk.short_conv_decode(xd, w, st2, torch.tensor(sidx, device=DEV, dtype=torch.int32))
    yr, sr = [], st.clone()
    for i, si in enumerate(sidx):
        y1, ns = ref_short_conv(xd[i : i + 1], w, st[si].float())
        yr.append(y1)
        sr[si] = ns.bfloat16()
    report("short_conv_decode.y", yd, torch.cat(yr).bfloat16(), 0.03)
    report("short_conv_decode.state", st2, sr, 0.0)


def ref_kda(q, k, v, g_raw, b_raw, A_log, dt_bias, S0, scale, lb):
    # q,k,v,g_raw: [L, H, D]; b_raw [L, H]; S0 [H, V, K]
    L, H, D = q.shape
    S = S0.clone().float()
    out = torch.zeros(L, H, D, device=q.device)
    for t in range(L):
        qt = q[t].float() / torch.sqrt((q[t].float() ** 2).sum(-1, keepdim=True) + 1e-6)
        kt = k[t].float() / torch.sqrt((k[t].float() ** 2).sum(-1, keepdim=True) + 1e-6)
        qt = qt * scale
        g = lb * torch.sigmoid(torch.exp(A_log).unsqueeze(-1) * (g_raw[t].float() + dt_bias.view(H, D)))
        beta = torch.sigmoid(b_raw[t].float())
        S = S * torch.exp(g).unsqueeze(1)
        vt = (v[t].float() - torch.einsum("hvk,hk->hv", S, kt)) * beta.unsqueeze(-1)
        S = S + vt.unsqueeze(-1) * kt.unsqueeze(1)
        out[t] = torch.einsum("hvk,hk->hv", S, qt)
    return out, S


def test_kda_recurrent():
    torch.manual_seed(4)
    H, D, R = 4, 128, 5
    lens = [7, 1, 3]
    init = [1, 0, 1]
    sidx = [2, 4, 0]
    T = sum(lens)
    q = torch.randn(T, H * D, device=DEV).bfloat16()
    k = torch.randn(T, H * D, device=DEV).bfloat16()
    v = torch.randn(T, H * D, device=DEV).bfloat16()
    g = torch.randn(T, H * D, device=DEV).bfloat16()
    b = torch.randn(T, H, device=DEV).bfloat16()
    A_log = torch.log(torch.rand(H, device=DEV) * 15 + 1)
    dtb = torch.randn(H * D, device=DEV) * 0.1
    state = torch.randn(R, H, D, D, device=DEV) * 0.1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    st = state.clone()
    o = kk.kda_recurrent(
        q, k, v, g, b, A_log, dtb, st, cu,
        torch.tensor(sidx, device=DEV, dtype=torch.int32),
        torch.tensor(init, device=DEV, dtype=torch.int32),
        num_heads=H, head_dim=D, scale=D**-0.5, lower_bound=-5.0,
    )
    off = 0
    ref_o, ref_st = [], state.clone()
    for L, ini, si in zip(lens, init, sidx):
        S0 = state[si] if ini else torch.zeros(H, D, D, device=DEV)
        sl = slice(off, off + L)
        oo, S = ref_kda(
            q[sl].view(L, H, D), k[sl].view(L, H, D), v[sl].view(L, H, D), g[sl].view(L, H, D),
            b[sl], A_log, dtb, S0, D**-0.5, -5.0,
        )
        ref_o.append(oo)
        ref_st[si] = S
        off += L
    report("kda_recurrent.o", o, torch.cat(ref_o).bfloat16(), 0.02)
    report("kda_recurrent.state", st, ref_st, 2e-4)

    x = torch.randn(T * H, D, device=DEV).bfloat16()
    gg = torch.randn(T * H, D, device=DEV).bfloat16()
    w = torch.rand(D, device=DEV) + 0.5
    y = kk.gated_rmsnorm(x, gg, w, 1e-5)
    xf = x.float()
    y_ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5) * w * torch.sigmoid(gg.float())
    report("gated_rmsnorm", y, y_ref.bfloat16(), 0.02)


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(0))
    test_hc()
    test_q_kv_rope()
    test_compressor_decode()
    test_short_conv()
    test_kda_recurrent()
    nfail = sum(1 for r in RESULTS if not r[3])
    print(f"SUMMARY: {len(RESULTS) - nfail}/{len(RESULTS)} passed")
    sys.exit(1 if nfail else 0)
