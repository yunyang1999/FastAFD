"""Vendored sglang decode kernels vs the minisgl reference implementations.

  python tests/dsv4_k3/test_sgl_kernels.py [--only kda,router,mhc]

* kda:    causal_conv1d_update + KDA packed decode (CUDA for B >= 8, Triton
          below) + sigmoid-gated RMSNorm vs short_conv_decode + kda_recurrent +
          gated_rmsnorm (outputs and updated conv / recurrent states)
* router: Triton fused router (DSV4 sqrt-softplus 256 -> 6) and the CUDA radix
          router (K3 sigmoid 896 -> 16) vs torch top-k
* mhc:    DeepGEMM TF32 prenorm GEMM (split-K) vs the fp32 reference
"""

from __future__ import annotations

import argparse
import sys

import torch

FAILS: list[str] = []


def report(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    got, ref = got.float(), ref.float()
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    rel = err / scale
    ok = rel <= tol and torch.isfinite(got).all().item()
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={err:.3e} rel={rel:.3e} (tol {tol})", flush=True)
    if not ok:
        FAILS.append(name)


def test_kda(B: int, H: int, dev) -> None:
    from minisgl.kernel.kda_triton import gated_rmsnorm, kda_recurrent, short_conv_decode
    from minisgl.kernel.sgl_triton.causal_conv1d import causal_conv1d_update
    from minisgl.kernel.sgl_triton.kda_decode import fused_recurrent_kda_packed_decode, layer_norm_gated_fwd

    D, KW, R = 128, 4, B + 3
    g = torch.Generator(device=dev).manual_seed(B * 1000 + H)
    rnd = lambda *s, dt=torch.bfloat16, sc=1.0: (torch.randn(*s, generator=g, device=dev) * sc).to(dt)
    qkv = rnd(B, 3 * H * D)
    g_raw = rnd(B, H * D)
    b_raw = rnd(B, H)
    gate = rnd(B, H * D)
    w = rnd(3 * H * D, KW, dt=torch.float32, sc=0.5)
    A_log = rnd(H, dt=torch.float32, sc=0.5)
    dt_bias = rnd(H * D, dt=torch.float32, sc=0.5)
    onw = rnd(D, sc=0.2) + 1
    conv0 = rnd(R, KW - 1, 3 * H * D)  # pool layout [R, KW-1, dim]
    ssm0 = rnd(R, H, D, D, dt=torch.float32, sc=0.05)
    idx = torch.randperm(R, generator=g, device=dev)[:B].to(torch.int32)
    lb, scale, eps = -5.0, D**-0.5, 1e-5
    # reference
    conv_r, ssm_r = conv0.clone(), ssm0.clone()
    y = short_conv_decode(qkv, w, conv_r.transpose(-1, -2), idx)
    q, k, v = y[:, : H * D], y[:, H * D : 2 * H * D], y[:, 2 * H * D :]
    cu = torch.arange(B + 1, dtype=torch.int32, device=dev)
    o = kda_recurrent(q, k, v, g_raw, b_raw, A_log, dt_bias, ssm_r, cu, idx, torch.ones_like(idx),
                      num_heads=H, head_dim=D, scale=scale, lower_bound=lb)
    o_r = gated_rmsnorm(o.view(B * H, D), gate.view(B * H, D), onw, eps).view(B, H * D)
    # sglang chain
    conv_s, ssm_s = conv0.clone(), ssm0.clone()
    y2 = causal_conv1d_update(qkv, conv_s.transpose(-1, -2), w, None, activation="silu", conv_state_indices=idx)
    out = torch.empty((B, 1, H, D), dtype=torch.bfloat16, device=dev)
    fused_recurrent_kda_packed_decode(mixed_qkv=y2, a=g_raw, b=b_raw, A_log=A_log, dt_bias=dt_bias, scale=scale,
                                      initial_state=ssm_s, out=out, ssm_state_indices=idx,
                                      use_qk_l2norm_in_kernel=True, lower_bound=lb)
    o_s, _, _, _ = layer_norm_gated_fwd(out.view(B * H, D), gate.view(B * H, D), onw, None,
                                        activation="sigmoid", eps=eps, is_rms_norm=True)
    tag = f"kda B={B} H={H}"
    report(f"{tag} conv_out", y2, y, 2e-2)
    report(f"{tag} conv_state", conv_s, conv_r, 0.0)
    report(f"{tag} ssm_state", ssm_s, ssm_r, 2e-3)
    report(f"{tag} out", o_s.view(B, H * D), o_r, 2e-2)


def test_router(dev) -> None:
    from minisgl.kernel.sgl_triton.moe_router import moe_fused_gate

    g = torch.Generator(device=dev).manual_seed(7)
    for (M, N, K, fn) in [(8, 256, 6, "sqrtsoftplus"), (128, 256, 6, "sqrtsoftplus"), (8, 896, 16, "sigmoid"), (100, 896, 16, "sigmoid")]:
        logits = torch.randn(M, N, generator=g, device=dev)
        bias = torch.randn(N, generator=g, device=dev) * 0.1
        w, ids = moe_fused_gate(logits, bias, K, scoring_func=fn, renormalize=True)
        act = torch.nn.functional.softplus(logits).sqrt() if fn == "sqrtsoftplus" else torch.sigmoid(logits)
        ref_ids = (act + bias).topk(K, dim=-1).indices
        ref_w = act.gather(1, ref_ids)
        ref_w = ref_w / ref_w.sum(-1, keepdim=True)
        s_got, o_got = ids.long().sort(-1)
        s_ref, o_ref = ref_ids.sort(-1)
        same = (s_got == s_ref).all().item()
        print(f"{'PASS' if same else 'FAIL'} router {fn} [{M},{N}] top{K}: id sets equal={same}", flush=True)
        if not same:
            FAILS.append(f"router {fn} {M}")
        report(f"router {fn} [{M},{N}] weights", w.gather(1, o_got), ref_w.gather(1, o_ref), 1e-5)


def test_mhc(dev) -> None:
    from minisgl.kernel.dsv4_triton import hc_prenorm_mixes, hc_prenorm_mixes_ref

    g = torch.Generator(device=dev).manual_seed(3)
    from minisgl.kernel.dsv4_triton import hc_pre_fused, hc_pre_split

    for T in (1, 8, 32, 128, 129, 300):
        x = (torch.randn(T, 4, 4096, generator=g, device=dev)).to(torch.bfloat16)
        fn = torch.randn(24, 4 * 4096, generator=g, device=dev) * 0.01
        report(f"mhc prenorm T={T}", hc_prenorm_mixes(x, fn, 1e-6), hc_prenorm_mixes_ref(x, fn, 1e-6), 5e-3)
        scale = torch.rand(3, generator=g, device=dev) + 0.5
        base = torch.randn(24, generator=g, device=dev) * 0.1
        nw = (torch.randn(4096, generator=g, device=dev) * 0.1 + 1).to(torch.bfloat16)
        y0, p0, c0 = hc_pre_fused(x, hc_prenorm_mixes_ref(x, fn, 1e-6), scale, base, sinkhorn_iters=20, eps=1e-6)
        y0 = y0.float()
        y0 = (y0 * torch.rsqrt(y0.square().mean(-1, keepdim=True) + 1e-6) * nw.float())
        y1, p1, c1 = hc_pre_split(x, fn, scale, base, rms_eps=1e-6, sinkhorn_iters=20, eps=1e-6, norm_weight=nw, norm_eps=1e-6)
        report(f"mhc pre fused+norm T={T} y", y1, y0, 2e-2)
        report(f"mhc pre fused+norm T={T} post", p1, p0, 1e-3)
        report(f"mhc pre fused+norm T={T} comb", c1, c0, 1e-3)


def _mla_ref(q, kv_rows, lens, scale):
    """q [B, H, 576]; kv_rows list of [L_b, 576] -> o [B, H, 512] (absorbed MLA, NoPE)."""
    outs = []
    for b, kv in enumerate(kv_rows):
        s = (q[b].float() @ kv.float().t()) * scale  # [H, L]
        p = torch.softmax(s, dim=-1)
        outs.append(p @ kv[:, :512].float())
    return torch.stack(outs)


def test_mla_perf(dev) -> None:
    """K3 MLA decode through the backend's padding path at serving shapes (96/48 heads)."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    g = torch.Generator(device=dev).manual_seed(23)
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    P = 64
    for H in (96, 48):
        Hp = 128 if H > 64 else 64
        for B, C in ((8, 4096), (32, 4096), (128, 4096), (32, 32768)):
            npg = (C + P - 1) // P
            npg += npg % 2
            cache = (torch.randn(B * npg + 2, P, 576, generator=g, device=dev) * 0.3).to(torch.bfloat16)
            bt = torch.arange(B * npg, device=dev, dtype=torch.int32).view(B, npg)
            sl = torch.full((B,), C, dtype=torch.int32, device=dev)
            q = (torch.randn(B, 1, Hp, 576, generator=g, device=dev) * 0.3).to(torch.bfloat16)
            q[:, :, H:] = 0

            def run():
                return trtllm_batch_decode_with_kv_cache_mla(
                    query=q, kv_cache=cache.view(-1, 1, P, 576), workspace_buffer=ws, qk_nope_head_dim=128,
                    kv_lora_rank=512, qk_rope_head_dim=64, block_tables=bt, seq_lens=sl, max_seq_len=C + 64,
                    bmm1_scale=192**-0.5, bmm2_scale=1.0)

            try:
                o = run()
                t = _time(run, iters=20)
                ok = bool(torch.isfinite(o).all())
                print(f"{'PASS' if ok else 'FAIL'} mla padded H={H}->{Hp} B={B} C={C}: {t:.1f} us", flush=True)
                if not ok:
                    FAILS.append(f"mla pad {H} {B} {C}")
            except Exception as e:
                print(f"FAIL mla padded H={H}->{Hp} B={B} C={C}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                FAILS.append(f"mla pad {H} {B} {C}")


def test_mla(dev) -> None:
    """flashinfer 0.6.6 trtllm-gen MLA decode (sglang's trtllm_mla kernel) per head count."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    g = torch.Generator(device=dev).manual_seed(11)
    P = 64
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    for H in (16, 32, 48, 64, 96, 128):
        B = 4
        lens = [200, 1, 777, 64]
        npg = [(L + P - 1) // P for L in lens]
        tot = sum(npg) + 3
        tot += tot % 2  # trtllm-gen: page count multiple of 128 / page_size
        cache = (torch.randn(tot, P, 576, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        perm = torch.randperm(tot, generator=g, device=dev).to(torch.int32)
        bt = torch.zeros(B, max(npg) + max(npg) % 2, dtype=torch.int32, device=dev)  # even column count
        o = 0
        rows = []
        for b, (L, n) in enumerate(zip(lens, npg)):
            pages = perm[o : o + n]
            o += n
            bt[b, :n] = pages
            rows.append(cache[pages.long()].reshape(-1, 576)[:L])
        q = (torch.randn(B, H, 576, generator=g, device=dev)).to(torch.bfloat16)
        scale = 192**-0.5
        ref = _mla_ref(q, rows, lens, scale)
        try:
            out = trtllm_batch_decode_with_kv_cache_mla(
                query=q.view(B, 1, H, 576), kv_cache=cache.view(tot, 1, P, 576), workspace_buffer=ws,
                qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64, block_tables=bt,
                seq_lens=torch.tensor(lens, dtype=torch.int32, device=dev), max_seq_len=max(lens),
                bmm1_scale=scale, bmm2_scale=1.0,
            )
            report(f"mla trtllm-gen H={H}", out.view(B, H, 512), ref, 2e-2)
        except Exception as e:  # unsupported head counts are reported, not fatal
            print(f"UNSUPPORTED mla trtllm-gen H={H}: {type(e).__name__}: {str(e)[:200]}", flush=True)


def test_attn_res(dev) -> None:
    """sglang fused TMA attention-residual kernel vs Triton reference + RMSNorm."""
    from minisgl.kernel import k3_attn_res
    from minisgl.kernel.kimi_triton import attn_res

    g = torch.Generator(device=dev).manual_seed(5)
    H, eps = 7168, 1e-5
    for T, nvb in ((8, 1), (32, 4), (128, 8), (3, 2)):
        bank = torch.randn(T, 9, H, generator=g, device=dev).to(torch.bfloat16)
        prefix = torch.randn(T, H, generator=g, device=dev).to(torch.bfloat16)
        w32 = torch.randn(H, generator=g, device=dev) * 0.05
        ow = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).to(torch.bfloat16)
        got = k3_attn_res.attn_res_fused_tma(prefix, bank.clone(), w32.to(torch.bfloat16), ow, nvb, eps)
        buf = bank.clone()
        buf[:, nvb] = prefix
        mix = attn_res(buf, nvb + 1, w32.to(torch.bfloat16).float(), eps).float()
        ref = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + eps) * ow.float()
        report(f"attn_res T={T} nvb={nvb}", got, ref, 2e-2)
        b2 = bank.clone()
        k3_attn_res.attn_res_fused_tma(prefix, b2, w32.to(torch.bfloat16), ow, nvb, eps, write_prefix=True)
        report(f"attn_res write_prefix T={T}", b2[:, nvb], prefix, 0.0)


def test_dsv4_elem(dev) -> None:
    """JIT q norm+RoPE / inverse RoPE vs the minisgl Triton kernels."""
    from minisgl.kernel.dsv4_sgl import freqs_real, fused_q_norm_rope, fused_rope_inplace
    from minisgl.kernel.dsv4_triton import inverse_rope_, q_norm_rope_

    g = torch.Generator(device=dev).manual_seed(9)
    P, R = 4096, 64
    ang = torch.rand(P, R // 2, generator=g, device=dev) * 6.28
    cos, sin = ang.cos(), ang.sin()
    fr = freqs_real(cos, sin)
    for T, H in ((8, 64), (128, 32)):
        q = torch.randn(T, H, 512, generator=g, device=dev).to(torch.bfloat16)
        pos = torch.randint(0, P, (T,), generator=g, device=dev).to(torch.int32)
        ref = q.clone()
        q_norm_rope_(ref, pos, cos, sin, eps=1e-6, rope_dim=R)
        got = torch.empty_like(q)
        fused_q_norm_rope(q, got, 1e-6, fr, pos)
        report(f"q_norm_rope T={T} H={H}", got, ref, 1e-2)
        o = torch.randn(T, H, 512, generator=g, device=dev).to(torch.bfloat16)
        ref = o.clone()
        inverse_rope_(ref, pos, cos, sin, R)
        fused_rope_inplace(o[..., -R:], fr, pos, inverse=True)
        report(f"inverse_rope T={T} H={H}", o, ref, 1e-2)


def test_indexer(dev) -> None:
    """sglang indexer decode (FP8 q/k, DeepGEMM paged logits, top-k v2) vs the
    BF16 reference scores: overlap of the selected top-512 sets."""
    from minisgl.kernel.dsv4_sgl import (
        fused_q_indexer_quant, freqs_real, gather_index_keys, paged_mqa_logits, paged_mqa_schedule,
        plan_topk_v2, store_index_keys, topk_transform_v2,
    )
    from minisgl.kernel.dsv4_triton import q_norm_rope_
    from minisgl.models.deepseek_v4 import hadamard_matrix

    g = torch.Generator(device=dev).manual_seed(13)
    H, D, P, R = 64, 128, 8192, 64
    ang = torch.rand(P, R // 2, generator=g, device=dev) * 6.28
    cos, sin = ang.cos(), ang.sin()
    fr = freqs_real(cos, sin)
    had = hadamard_matrix(D, str(dev))
    for T, ctx in ((4, 1000), (16, 5000), (8, 1)):
        n4 = (ctx + 1) // 4
        npg = max(1, (n4 + 63) // 64)
        tot_pages = T * npg + 2
        cache = torch.zeros(tot_pages, 64 * 132, dtype=torch.uint8, device=dev)
        perm = torch.randperm(tot_pages, generator=g, device=dev)[: T * npg].view(T, npg).to(torch.int32)
        normw = torch.ones(D, device=dev)
        rows_all = []
        for t in range(T):
            rows = (perm[t].long()[:, None] * 64 + torch.arange(64, device=dev)[None, :]).reshape(-1)[:n4]
            k = torch.randn(n4, D, generator=g, device=dev).to(torch.bfloat16)
            gpos = (torch.arange(n4, device=dev) * 4).to(torch.int32)
            store_index_keys(k, normw, gpos, cos, sin, had, cache, rows, eps=1e-6, rope_dim=R)
            rows_all.append(rows)
        q = torch.randn(T, H, D, generator=g, device=dev).to(torch.bfloat16)
        w = torch.randn(T, H, generator=g, device=dev).to(torch.bfloat16)
        pos = torch.full((T,), ctx, dtype=torch.int32, device=dev)
        c4_lens = torch.full((T,), n4, dtype=torch.int32, device=dev)
        scale = D**-0.5 * H**-0.5
        q_fp8, w_out = fused_q_indexer_quant(q, w, scale, fr, pos)
        sched = paged_mqa_schedule(c4_lens)
        logits = paged_mqa_logits(q_fp8, cache, w_out.view(T, H), c4_lens, perm, sched, npg * 64)
        plan = plan_topk_v2(c4_lens)
        out = torch.empty(T, 512, dtype=torch.int32, device=dev)
        topk_transform_v2(logits, c4_lens, perm, out, 64, plan)
        # reference: bf16 q (rope + hadamard), dequantised keys, relu-weighted sum
        qr = q.clone()
        q_norm_rope_(qr, pos, cos, sin, eps=0.0, rope_dim=R, do_norm=False, hadamard=had)
        ov = []
        for t in range(T):
            ks = gather_index_keys(cache, rows_all[t]).float()
            s = (qr[t].float() @ ks.t()).relu() * (w[t].float() * scale)[:, None]
            s = s.sum(0)
            if n4 > 0 and t == 0:
                d = (logits[t, :n4] - s).abs().max().item() / (s.abs().max().item() + 1e-6)
                print(f"INFO indexer logits vs bf16 q T={T} ctx={ctx}: rel={d:.3e} (FP8 q quantisation)", flush=True)
            k = min(512, n4)
            ref_sel = set(rows_all[t][s.topk(k).indices].tolist()) if k else set()
            got_sel = set(x for x in out[t].tolist() if x >= 0)
            ov.append(len(ref_sel & got_sel) / len(ref_sel) if ref_sel else float(not got_sel))
            if t == 0 and n4 > 0:
                # same FP8 q as the kernel (dequantised): logits must match DeepGEMM closely
                qs = (w_out[t, :, 0] / (w[t].float() * scale)).clamp_min(1e-12)
                qd = q_fp8[t].float() * qs[:, None]
                s8 = ((qd @ ks.t()).relu() * (w[t].float() * scale)[:, None]).sum(0)
                report(f"indexer logits (fp8 q) T={T} ctx={ctx}", logits[t, :n4], s8, 1e-2)
        ok = min(ov) >= 0.9 and all(len(set(x for x in out[t].tolist() if x >= 0)) == min(512, n4) for t in range(T))
        print(f"{'PASS' if ok else 'FAIL'} indexer topk T={T} ctx={ctx}: min overlap={min(ov):.3f}", flush=True)
        if not ok:
            FAILS.append(f"indexer topk {T} {ctx}")


def test_wo_a(dev) -> None:
    """FP8 wo_a grouped einsum (sglang FP8_WO_A_GEMM) vs BF16 einsum on the dequantised weight."""
    from minisgl.kernel.dsv4_sgl import wo_a_einsum, wo_a_weight_sf
    from minisgl.models.dummy_init import _quant_fp8_block

    g = torch.Generator(device=dev).manual_seed(17)
    G, R, D = 8, 1024, 4096
    w = torch.randn(G * R, D, generator=g, device=dev) / D**0.5
    wq, ws = _quant_fp8_block(w)
    wdq = (wq.float().view(G * R // 128, 128, D // 128, 128) * ws[:, None, :, None]).view(G * R, D)
    sf = wo_a_weight_sf(ws, G, R, D)
    for T in (1, 8, 32, 128, 129):
        o = torch.randn(T, G, D, generator=g, device=dev).to(torch.bfloat16)
        got = wo_a_einsum(o, wq.view(G, R, D), sf)
        ref = torch.einsum("tgd,grd->tgr", o.float(), wdq.view(G, R, D))
        report(f"wo_a fp8 einsum T={T}", got, ref, 3e-2)


def _time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    for _ in range(3):
        g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters / 10 * 1e3


def test_sparse_perf(dev) -> None:
    """trtllm-gen sparse MLA (DSV4): native head count vs zero-padded to 128 heads
    (the high-throughput 2-CTA kernel), correctness + timing."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    g = torch.Generator(device=dev).manual_seed(21)
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    rows = 1 << 20
    kv = (torch.randn(rows, 576, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    kv[:, 512:] = 0
    for H in (64, 32):
        for B in (8, 32, 128):
            for K in (128, 640):
                idx = torch.randint(0, rows, (B, 1, K), generator=g, device=dev, dtype=torch.int32)
                sl = torch.full((B,), K, dtype=torch.int32, device=dev)
                q = (torch.randn(B, 1, H, 576, generator=g, device=dev) * 0.3).to(torch.bfloat16)
                q[..., 512:] = 0
                sinks = torch.randn(H, generator=g, device=dev) * 0.1

                def run(qq, sk):
                    return trtllm_batch_decode_with_kv_cache_mla(
                        query=qq, kv_cache=kv.view(-1, 1, 64, 576), workspace_buffer=ws, qk_nope_head_dim=128,
                        kv_lora_rank=512, qk_rope_head_dim=64, block_tables=idx, seq_lens=sl, max_seq_len=rows,
                        sparse_mla_top_k=K, bmm1_scale=512**-0.5, bmm2_scale=1.0, sinks=sk, backend="trtllm-gen")

                q128 = torch.zeros(B, 1, 128, 576, dtype=torch.bfloat16, device=dev)
                q128[:, :, :H] = q
                s128 = torch.zeros(128, device=dev)
                s128[:H] = sinks
                o1 = run(q, sinks).view(B, H, 512)
                o2 = run(q128, s128).view(B, 128, 512)[:, :H]
                report(f"sparse pad128 H={H} B={B} K={K}", o2, o1, 2e-2)
                t1 = _time(lambda: run(q, sinks))
                t2 = _time(lambda: run(q128, s128))
                print(f"INFO sparse mla H={H} B={B} K={K}: native {t1:.1f} us, padded-128 {t2:.1f} us", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="kda,router,mhc,mla,attn_res,dsv4_elem,indexer,wo_a,mla_perf")
    args = ap.parse_args()
    dev = torch.device("cuda")
    only = set(args.only.split(","))
    for name, fn in (("attn_res", test_attn_res), ("dsv4_elem", test_dsv4_elem), ("indexer", test_indexer), ("wo_a", test_wo_a),
                     ("sparse_perf", test_sparse_perf), ("mla_perf", test_mla_perf)):
        if name in only:
            try:
                fn(dev)
            except Exception as e:  # keep going: report every kernel family
                import traceback

                traceback.print_exc()
                FAILS.append(f"{name}: {type(e).__name__}")
    if "mla" in only:
        test_mla(dev)
    if "mhc" in only:
        test_mhc(dev)
    if "router" in only:
        test_router(dev)
    if "kda" in only:
        for B in (1, 4, 8, 32, 128):
            for H in (96, 48, 12):
                test_kda(B, H, dev)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
